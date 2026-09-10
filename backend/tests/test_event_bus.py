from __future__ import annotations

import pytest

from services.events import (
    AgentEvent,
    CommitEvent,
    DiffEvent,
    EventBus,
    SystemEvent,
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


def recorder() -> tuple[list[AgentEvent], object]:
    """
    Build a handler that records the events it is given.

    :returns: The recorded events and the handler to subscribe.
    """
    seen: list[AgentEvent] = []

    async def handler(event: AgentEvent) -> None:
        seen.append(event)

    return seen, handler


def thought_streams(seen: list[AgentEvent]) -> list[str | None]:
    """
    Collect the stream identifier of every thought that was dispatched.

    :param seen: The events a handler recorded.
    :returns: One identifier per thought, in order.
    """
    return [event.stream_id for event in seen if isinstance(event, ThoughtEvent)]


async def test_consecutive_thoughts_share_a_stream(events: EventBus) -> None:
    """
    An unbroken run of reasoning carries one identifier throughout.

    Requirement 2.1: the identifier is stable for one continuous run, so the
    client can grow a single message instead of appending one per sentence.
    """
    seen, handler = recorder()
    events.subscribe(handler)  # type: ignore[arg-type]

    await events.emit(ThoughtEvent(content='First.'))
    await events.emit(ThoughtEvent(content='Second.'))
    await events.emit(ThoughtEvent(content='Third.'))

    streams: list[str | None] = thought_streams(seen)
    assert len(streams) == 3
    assert streams[0] is not None
    assert len(set(streams)) == 1


async def test_a_non_thought_event_starts_a_new_stream(events: EventBus) -> None:
    """
    Reasoning either side of a tool call is two runs, not one.

    Requirement 2.3: any non-thought event retires the current identifier.
    """
    seen, handler = recorder()
    events.subscribe(handler)  # type: ignore[arg-type]

    await events.emit(ThoughtEvent(content='Before.'))
    await events.emit(ToolActionEvent(tool_name='read_file', arguments={'path': 'a.py'}))
    await events.emit(ThoughtEvent(content='After.'))

    before, after = thought_streams(seen)
    assert before is not None
    assert after is not None
    assert before != after


@pytest.mark.parametrize(
    'interrupting',
    [
        ToolActionEvent(tool_name='run_command', arguments={}),
        ToolObservationEvent(tool_name='run_command', result='ok'),
        CommitEvent(commit_hash='abc1234', message='work', files_changed=['a.py']),
        DiffEvent(file_path='a.py', original='', modified='x'),
        SystemEvent(content='Indexing...'),
    ],
)
async def test_every_other_event_kind_breaks_the_stream(
    events: EventBus, interrupting: AgentEvent
) -> None:
    """
    Only thoughts continue a run; every other kind of event ends it.

    Requirement 2.5: actions, observations, commits, diffs and system messages
    are discrete events, and none of them coalesces or is coalesced through.
    """
    seen, handler = recorder()
    events.subscribe(handler)  # type: ignore[arg-type]

    await events.emit(ThoughtEvent(content='Before.'))
    await events.emit(interrupting)
    await events.emit(ThoughtEvent(content='After.'))

    before, after = thought_streams(seen)
    assert before != after


async def test_the_users_sequence_produces_two_runs(events: EventBus) -> None:
    """
    Prompt, thought, thought, thought, action, thought, observation gives two runs.

    Requirement 2.3 states this sequence explicitly: it must present as prompt,
    thought, action, thought, observation, so the three leading thoughts share one
    identifier and the trailing thought has its own.
    """
    seen, handler = recorder()
    events.subscribe(handler)  # type: ignore[arg-type]

    await events.emit(ThoughtEvent(content='One.'))
    await events.emit(ThoughtEvent(content='Two.'))
    await events.emit(ThoughtEvent(content='Three.'))
    await events.emit(ToolActionEvent(tool_name='edit_file', arguments={}))
    await events.emit(ThoughtEvent(content='Four.'))
    await events.emit(ToolObservationEvent(tool_name='edit_file', result='done'))

    first, second, third, fourth = thought_streams(seen)
    assert first == second == third
    assert fourth != first


async def test_the_stream_is_reported_while_reasoning(events: EventBus) -> None:
    """
    The active run is visible on the bus and retired by anything else.
    """
    assert events.thought_stream is None

    await events.emit(ThoughtEvent(content='Thinking.'))
    assert events.thought_stream is not None

    await events.emit(ToolActionEvent(tool_name='read_file', arguments={}))
    assert events.thought_stream is None


async def test_an_existing_stream_identifier_is_respected(events: EventBus) -> None:
    """
    A caller that labels a thought itself is not overridden by the bus.
    """
    seen, handler = recorder()
    events.subscribe(handler)  # type: ignore[arg-type]

    await events.emit(ThoughtEvent(content='Mine.', stream_id='supplied'))

    assert thought_streams(seen) == ['supplied']


async def test_streams_are_scoped_to_their_session() -> None:
    """
    Two sessions cannot collide on a stream identifier.
    """
    first: EventBus = EventBus('session-a')
    second: EventBus = EventBus('session-b')

    await first.emit(ThoughtEvent(content='a'))
    await second.emit(ThoughtEvent(content='b'))

    assert first.thought_stream != second.thought_stream


async def test_stamping_does_not_mutate_the_event_it_was_given(events: EventBus) -> None:
    """
    Events are frozen records, so labelling produces a copy.
    """
    original: ThoughtEvent = ThoughtEvent(content='Thinking.')

    await events.emit(original)

    assert original.stream_id is None
