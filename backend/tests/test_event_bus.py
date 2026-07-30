from __future__ import annotations

import pytest

from services.event_bus import (
    AgentEvent,
    CommitEvent,
    DiffEvent,
    EventBus,
    ThoughtEvent,
    ToolActionEvent,
    ToolObservationEvent,
    ToolStatus,
)
from pydantic import TypeAdapter, ValidationError


async def test_subscribe_returns_handler_and_deduplicates(events: EventBus) -> None:
    """
    Subscribing returns the handler and ignores repeat registrations.
    """
    received: list[AgentEvent] = []

    async def handler(event: AgentEvent) -> None:
        received.append(event)

    assert events.subscribe(handler) is handler
    events.subscribe(handler)
    assert len(events.handlers) == 1


async def test_emit_reports_delivery_count(events: EventBus) -> None:
    """
    Emitting reports how many handlers received the event.
    """
    received: list[AgentEvent] = []

    async def handler(event: AgentEvent) -> None:
        received.append(event)

    events.subscribe(handler)
    assert await events.emit(ThoughtEvent(content='hello')) == 1
    assert len(received) == 1


async def test_sessions_are_isolated() -> None:
    """
    An event raised on one bus never reaches a handler on another.
    """
    bus_a: EventBus = EventBus('a')
    bus_b: EventBus = EventBus('b')
    seen_b: list[AgentEvent] = []

    async def handler_b(event: AgentEvent) -> None:
        seen_b.append(event)

    bus_b.subscribe(handler_b)
    await bus_a.emit(ThoughtEvent(content='for a only'))
    assert seen_b == []


async def test_failing_handler_does_not_block_others(events: EventBus) -> None:
    """
    One raising handler cannot stop the remaining handlers from receiving events.
    """
    received: list[AgentEvent] = []

    async def exploding(event: AgentEvent) -> None:
        raise RuntimeError('handler exploded')

    async def working(event: AgentEvent) -> None:
        received.append(event)

    events.subscribe(exploding)
    events.subscribe(working)

    assert await events.emit(ThoughtEvent(content='x')) == 1
    assert len(received) == 1


async def test_unsubscribe_and_clear(events: EventBus) -> None:
    """
    Handlers can be removed individually or all at once.
    """
    async def handler(event: AgentEvent) -> None:
        return None

    events.subscribe(handler)
    assert events.unsubscribe(handler) is True
    assert events.unsubscribe(handler) is False

    events.subscribe(handler)
    assert events.clear() == 1
    assert events.handlers == ()


def test_observation_success_property() -> None:
    """
    An observation reports success only when its status is success.
    """
    assert ToolObservationEvent(tool_name='edit_file', result='ok').success is True
    assert ToolObservationEvent(
        tool_name='edit_file', result='no', status=ToolStatus.failure
    ).success is False


def test_events_are_frozen() -> None:
    """
    Events cannot be mutated after construction.
    """
    event: ToolActionEvent = ToolActionEvent(tool_name='edit_file', arguments={})
    with pytest.raises(ValidationError):
        event.tool_name = 'other'  # type: ignore[misc]


def test_discriminated_union_round_trip() -> None:
    """
    Events serialise to and parse back from their wire form by discriminator.
    """
    adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)

    parsed: AgentEvent = adapter.validate_python(
        {'type': 'diff', 'file_path': 'a.py', 'original': 'x', 'modified': 'y'}
    )
    assert isinstance(parsed, DiffEvent)
    assert parsed.file_path == 'a.py'

    observation: AgentEvent = adapter.validate_python(
        {'type': 'observation', 'tool_name': 'edit_file', 'result': 'no', 'status': 'failure'}
    )
    assert isinstance(observation, ToolObservationEvent)
    assert observation.success is False

    dumped: dict[str, object] = adapter.dump_python(
        CommitEvent(commit_hash='abc', message='m', files_changed=['a.py']), mode='json'
    )
    assert dumped['type'] == 'commit'


def test_unknown_discriminator_is_rejected() -> None:
    """
    An unrecognised event type fails validation rather than being accepted.
    """
    adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)
    with pytest.raises(ValidationError):
        adapter.validate_python({'type': 'not_a_real_event'})
