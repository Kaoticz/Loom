from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from services.event_bus import EventBus, ToolActionEvent, ToolObservationEvent, ToolStatus

MAX_REPORTED_RESULT_LENGTH: int = 2000


class ToolReport:
    """
    The outcome a tool wants reported once it finishes.

    A tool fills this in as it runs; the surrounding context manager turns it
    into an observation event. Tools therefore never touch the event bus
    directly, and cannot forget to announce that they finished.
    """

    _result: str
    _status: ToolStatus

    def __init__(self) -> None:
        """
        Create a report that defaults to a bare success.
        """
        self._result = ''
        self._status = ToolStatus.success

    @property
    def result(self) -> str:
        """
        The text describing what the tool produced.

        :returns: The recorded result.
        """
        return self._result

    @property
    def status(self) -> ToolStatus:
        """
        Whether the tool succeeded.

        :returns: The recorded status.
        """
        return self._status

    def succeed(self, result: str) -> str:
        """
        Record that the tool completed successfully.

        :param result: What the tool produced.
        :returns: The recorded result.
        """
        self._result = result
        self._status = ToolStatus.success
        return self._result

    def fail(self, result: str) -> str:
        """
        Record that the tool did not do what was asked.

        :param result: An explanation of what went wrong.
        :returns: The recorded result.
        """
        self._result = result
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
            result=summarise_result(report.result),
            status=report.status,
        )
    )
