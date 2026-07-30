from __future__ import annotations

import pytest

from services.event_bus import (
    AgentEvent,
    EventBus,
    ToolActionEvent,
    ToolObservationEvent,
    ToolStatus,
)
from services.tool_reporting import MAX_REPORTED_RESULT_LENGTH, report_tool, summarise_result


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
