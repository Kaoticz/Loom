from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from services.events import EventBus, ToolActionEvent, ToolObservationEvent, ToolStatus

MAX_REPORTED_RESULT_LENGTH: int = 2000


class ToolReport:
    """
    The outcome a tool wants reported once it finishes.

    A tool fills this in as it runs; the surrounding context manager turns it
    into an observation event. Tools therefore never touch the event bus
    directly, and cannot forget to announce that they finished.

    What the agent receives and what the user is shown are usually the same
    text, but they need not be. Reading a file gives the agent the contents,
    which is the whole point of the tool, while the reader wants to know only
    that the file was read.
    """

    _result: str
    _observation: str | None
    _status: ToolStatus

    def __init__(self) -> None:
        """
        Create a report that defaults to a bare success.
        """
        self._result = ''
        self._observation = None
        self._status = ToolStatus.success

    @property
    def result(self) -> str:
        """
        The text describing what the tool produced.

        :returns: The recorded result.
        """
        return self._result

    @property
    def observation(self) -> str:
        """
        The text describing the outcome to the user.

        Falls back to the result, so a tool that says nothing about how it should
        be reported is reported by what it produced.

        :returns: The text to show the user.
        """
        return self._result if self._observation is None else self._observation

    @property
    def status(self) -> ToolStatus:
        """
        Whether the tool succeeded.

        :returns: The recorded status.
        """
        return self._status

    def succeed(self, result: str, observation: str | None = None) -> str:
        """
        Record that the tool completed successfully.

        :param result: What the tool produced, which is returned to the agent.
        :param observation: What to tell the user, where that should differ from
            what the agent receives. Defaults to the result itself.
        :returns: The recorded result.
        """
        self._result = result
        self._observation = observation
        self._status = ToolStatus.success
        return self._result

    def fail(self, result: str) -> str:
        """
        Record that the tool did not do what was asked.

        A failure is always reported as written: the reason is the whole of what
        the user needs, and hiding it behind a summary would leave them guessing.

        :param result: An explanation of what went wrong.
        :returns: The recorded result.
        """
        self._result = result
        self._observation = None
        self._status = ToolStatus.failure
        return self._result


def summarise_result(result: str, limit: int = MAX_REPORTED_RESULT_LENGTH) -> str:
    """
    Shorten a tool result so a large payload cannot flood the client.

    :param result: The full result text.
    :param limit: The maximum number of characters to send.
    :returns: The result, truncated with a notice if it was too long.
    """
    if len(result) <= limit:
        return result
    return f'{result[:limit]}\n... ({len(result) - limit} more characters)'


@asynccontextmanager
async def report_tool(
    events: EventBus,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> AsyncGenerator[ToolReport]:
    """
    Announce a tool's execution to the session's event bus.

    An action event is emitted on entry and an observation event on exit,
    including when the body raises, so a crashing tool still reports rather than
    leaving the client watching an action that never completes.

    Callers pass only the arguments worth displaying: file contents and search
    blocks are deliberately left out, since they can be very large.

    :param events: The session's event bus.
    :param tool_name: The name of the tool being run.
    :param arguments: The arguments worth showing to the user.
    :returns: An async generator yielding the report for the tool to fill in.
    """
    await events.emit(
        ToolActionEvent(
            tool_name=tool_name,
            arguments=arguments or {},
            status=ToolStatus.running,
        )
    )

    report: ToolReport = ToolReport()
    try:
        yield report
    except Exception as error:
        await events.emit(
            ToolObservationEvent(
                tool_name=tool_name,
                result=f'{type(error).__name__}: {error}',
                status=ToolStatus.failure,
            )
        )
        raise

    await events.emit(
        ToolObservationEvent(
            tool_name=tool_name,
            result=summarise_result(report.observation),
            status=report.status,
        )
    )


def describe_file_read(
    path: str, contents: str, start_line: int | None = None, end_line: int | None = None
) -> str:
    """
    Describe a successful file read in one line.

    The contents themselves are of no use to someone watching the session: they
    are what the agent asked for, not what the reader wants to see, and a long
    file pushes everything else off the screen. Naming the file and its size
    tells the reader what happened and nothing more.

    :param path: The file that was read.
    :param contents: What was read, used only to count lines.
    :param start_line: The first line read, when only part of the file was.
    :param end_line: The last line read, when only part of the file was.
    :returns: A single line describing the read.
    """
    if not contents:
        return f'Read {path} (empty).'

    lines: int = contents.count('\n') + (0 if contents.endswith('\n') else 1)
    if start_line is None and end_line is None:
        return f'Read {path} ({lines} line{"" if lines == 1 else "s"}).'

    first: str = 'start' if start_line is None else str(start_line)
    last: str = 'end' if end_line is None else str(end_line)
    return f'Read {path}, lines {first} to {last} ({lines} line{"" if lines == 1 else "s"}).'


def describe_symbol_lookup(symbol: str, found: int) -> str:
    """
    Describe a symbol lookup in one line.

    The definitions themselves are what the agent asked for; the reader wants to
    know whether the symbol was found and in how many places, since more than one
    declaration is usually worth noticing.

    :param symbol: The symbol that was looked up.
    :param found: How many declarations were found.
    :returns: A single line describing the lookup.
    """
    if found == 0:
        return f'No definition of {symbol} found.'
    if found == 1:
        return f'Found the definition of {symbol}.'
    return f'Found {found} definitions of {symbol}.'
