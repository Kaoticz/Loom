from __future__ import annotations

import pytest

from services.events import (
    AgentEvent,
    EventBus,
    ToolActionEvent,
    ToolObservationEvent,
    ToolStatus,
)
from services.tool_reporting import (
    MAX_REPORTED_RESULT_LENGTH,
    ToolReport,
    describe_file_read,
    report_tool,
    summarise_result,
)


def collect(events: EventBus) -> list[AgentEvent]:
    """
    Subscribe a collector to the bus.

    :param events: The bus to observe.
    :returns: The list that received events are appended to.
    """
    seen: list[AgentEvent] = []

    async def handler(event: AgentEvent) -> None:
        seen.append(event)

    events.subscribe(handler)
    return seen


async def test_reports_action_then_observation(events: EventBus) -> None:
    """
    A tool announces itself before running and reports its result afterwards.
    """
    seen: list[AgentEvent] = collect(events)

    async with report_tool(events, 'edit_file', {'path': 'a.py'}) as report:
        report.succeed('Edited a.py successfully.')

    assert len(seen) == 2
    action, observation = seen
    assert isinstance(action, ToolActionEvent)
    assert action.tool_name == 'edit_file'
    assert action.arguments == {'path': 'a.py'}
    assert action.status is ToolStatus.running

    assert isinstance(observation, ToolObservationEvent)
    assert observation.result == 'Edited a.py successfully.'
    assert observation.status is ToolStatus.success


async def test_failure_is_reported_as_such(events: EventBus) -> None:
    """
    A tool that could not do its job reports a failed observation.
    """
    seen: list[AgentEvent] = collect(events)

    async with report_tool(events, 'edit_file', {'path': 'a.py'}) as report:
        report.fail('The search block was not found.')

    observation = seen[1]
    assert isinstance(observation, ToolObservationEvent)
    assert observation.status is ToolStatus.failure
    assert observation.success is False


async def test_raising_tool_still_reports(events: EventBus) -> None:
    """
    An exception still produces an observation, then propagates.

    Without this the client would show an action that never finishes.
    """
    seen: list[AgentEvent] = collect(events)

    with pytest.raises(RuntimeError):
        async with report_tool(events, 'run_test_command', {'command': 'pytest'}):
            raise RuntimeError('sandbox exploded')

    assert len(seen) == 2
    observation = seen[1]
    assert isinstance(observation, ToolObservationEvent)
    assert observation.status is ToolStatus.failure
    assert 'sandbox exploded' in observation.result


async def test_large_results_are_truncated(events: EventBus) -> None:
    """
    An enormous tool result is shortened before it reaches the client.
    """
    seen: list[AgentEvent] = collect(events)

    async with report_tool(events, 'read_file', {'path': 'big.py'}) as report:
        report.succeed('x' * (MAX_REPORTED_RESULT_LENGTH + 500))

    observation = seen[1]
    assert isinstance(observation, ToolObservationEvent)
    assert len(observation.result) < MAX_REPORTED_RESULT_LENGTH + 100
    assert 'more characters' in observation.result


def test_short_results_are_untouched() -> None:
    """
    A result within the limit is passed through unchanged.
    """
    assert summarise_result('all good') == 'all good'


def test_a_whole_file_read_is_described_by_name_and_size() -> None:
    """
    Requirement 13.1: the observation names the file instead of repeating it.
    """
    described: str = describe_file_read('src/app.py', 'one\ntwo\nthree\n')

    assert described == 'Read src/app.py (3 lines).'


def test_a_one_line_file_is_described_in_the_singular() -> None:
    """
    "1 lines" is the kind of detail that makes an interface feel unfinished.
    """
    assert describe_file_read('a.py', 'only\n') == 'Read a.py (1 line).'


def test_a_file_without_a_trailing_newline_is_counted_correctly() -> None:
    """
    The last line counts whether or not the file ends with a newline.
    """
    assert describe_file_read('a.py', 'one\ntwo') == 'Read a.py (2 lines).'


def test_an_empty_file_says_so() -> None:
    """
    Zero lines read as a mistake; saying the file is empty does not.
    """
    assert describe_file_read('a.py', '') == 'Read a.py (empty).'


def test_a_partial_read_names_the_range() -> None:
    """
    Reading part of a file is a different act from reading all of it.
    """
    described: str = describe_file_read('src/app.py', 'x\ny\n', 10, 11)

    assert described == 'Read src/app.py, lines 10 to 11 (2 lines).'


def test_an_open_ended_range_is_described_in_words() -> None:
    """
    A missing bound is the start or the end of the file, and should read as such.
    """
    assert 'lines 10 to end' in describe_file_read('a.py', 'x\n', 10, None)
    assert 'lines start to 20' in describe_file_read('a.py', 'x\n', None, 20)


def test_the_description_never_contains_the_contents() -> None:
    """
    The point of the requirement: a large file must not reach the client.
    """
    contents: str = '\n'.join(f'secret line {index}' for index in range(500))

    described: str = describe_file_read('big.py', contents)

    assert 'secret line' not in described
    assert len(described) < 60


def test_an_observation_defaults_to_the_result() -> None:
    """
    A tool that says nothing about reporting is reported by what it produced.
    """
    report: ToolReport = ToolReport()

    report.succeed('the whole answer')

    assert report.observation == 'the whole answer'


def test_an_observation_can_differ_from_the_result() -> None:
    """
    Requirement 13.2: the agent's return value and the user's view diverge.
    """
    report: ToolReport = ToolReport()

    returned: str = report.succeed('the entire file', observation='Read a.py (1 line).')

    assert returned == 'the entire file'
    assert report.observation == 'Read a.py (1 line).'


def test_a_failure_is_always_reported_as_written() -> None:
    """
    Requirement 13.3: the reason is the whole of what the user needs.
    """
    report: ToolReport = ToolReport()
    report.succeed('contents', observation='Read a.py (1 line).')

    report.fail('File a.py does not exist.')

    assert report.observation == 'File a.py does not exist.'
