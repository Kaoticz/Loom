from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from schemas.agent_messages import (
    MAX_DIFF_CHARACTERS,
    MAX_DIFF_LINES,
    ActionMessage,
    AgentMessage,
    CommitMessage,
    DiffMessage,
    DoneMessage,
    ErrorMessage,
    MessageType,
    ObservationMessage,
    PlanMessage,
    ReviewMessage,
    ThoughtMessage,
    to_message,
    truncate_diff_side,
)
from services.event_bus import (
    BaseAgentEvent,
    CommitEvent,
    DiffEvent,
    DoneEvent,
    ErrorEvent,
    PlanEvent,
    ReviewEvent,
    ThoughtEvent,
    ToolActionEvent,
    ToolObservationEvent,
    ToolStatus,
)

ADAPTER: TypeAdapter[AgentMessage] = TypeAdapter(AgentMessage)


def test_every_message_type_is_defined() -> None:
    """
    The protocol covers exactly the nine documented message types.
    """
    assert {member.value for member in MessageType} == {
        'thought', 'plan', 'action', 'observation',
        'commit', 'diff', 'review', 'done', 'error',
        # Not produced from an event: sent directly when a connection is refused.
        'session_busy',
    }


@pytest.mark.parametrize(
    'event,expected',
    [
        (ThoughtEvent(content='thinking'), ThoughtMessage),
        (PlanEvent(steps=['a', 'b']), PlanMessage),
        (ToolActionEvent(tool_name='edit_file', arguments={'path': 'a.py'}), ActionMessage),
        (ToolObservationEvent(tool_name='edit_file', result='ok'), ObservationMessage),
        (CommitEvent(commit_hash='abc', message='m', files_changed=[]), CommitMessage),
        (DiffEvent(file_path='a.py', original='x', modified='y'), DiffMessage),
        (ReviewEvent(approved=True, comments='fine'), ReviewMessage),
        (DoneEvent(summary='done'), DoneMessage),
        (ErrorEvent(content='boom'), ErrorMessage),
    ],
    ids=lambda value: type(value).__name__,
)
def test_each_event_maps_to_its_message(event: BaseAgentEvent, expected: type) -> None:
    """
    Every event has a corresponding wire message.
    """
    assert isinstance(to_message(event), expected)


def test_unmapped_event_is_rejected() -> None:
    """
    An event with no message type fails loudly rather than being dropped.
    """
    class UnknownEvent(BaseAgentEvent):
        """
        An event the protocol does not know about.
        """

    with pytest.raises(TypeError):
        to_message(UnknownEvent())


def test_fields_survive_conversion() -> None:
    """
    Message payloads carry the event's data unchanged.
    """
    action = to_message(
        ToolActionEvent(tool_name='edit_file', arguments={'path': 'a.py'})
    )
    assert isinstance(action, ActionMessage)
    assert action.tool_name == 'edit_file'
    assert action.arguments == {'path': 'a.py'}
    assert action.status is ToolStatus.running

    diff = to_message(DiffEvent(file_path='a.py', original='before', modified='after'))
    assert isinstance(diff, DiffMessage)
    assert (diff.file_path, diff.original, diff.modified) == ('a.py', 'before', 'after')

    commit = to_message(
        CommitEvent(commit_hash='abc123', message='Edit a.py', files_changed=['a.py'])
    )
    assert isinstance(commit, CommitMessage)
    assert commit.files_changed == ['a.py']


def test_failure_status_survives_conversion() -> None:
    """
    A failed observation stays failed on the wire.
    """
    message = to_message(
        ToolObservationEvent(tool_name='edit_file', result='no match', status=ToolStatus.failure)
    )
    assert isinstance(message, ObservationMessage)
    assert message.status is ToolStatus.failure


@pytest.mark.parametrize(
    'event',
    [
        ThoughtEvent(content='thinking'),
        PlanEvent(steps=['a']),
        ToolActionEvent(tool_name='t', arguments={}),
        ToolObservationEvent(tool_name='t', result='r'),
        CommitEvent(commit_hash='h', message='m', files_changed=[]),
        DiffEvent(file_path='p', original='o', modified='m'),
        ReviewEvent(approved=False, comments='c'),
        DoneEvent(summary='s'),
        ErrorEvent(content='e'),
    ],
    ids=lambda value: type(value).__name__,
)
def test_messages_round_trip_over_the_wire(event: BaseAgentEvent) -> None:
    """
    Serialised messages parse back into the same message type by discriminator.
    """
    payload: dict[str, object] = to_message(event).model_dump(mode='json')
    assert isinstance(payload['type'], str)
    assert type(ADAPTER.validate_python(payload)) is type(to_message(event))


def test_unknown_message_type_is_rejected() -> None:
    """
    A payload with an unrecognised type fails validation.
    """
    with pytest.raises(ValidationError):
        ADAPTER.validate_python({'type': 'nonsense', 'content': 'x'})


def test_messages_are_frozen() -> None:
    """
    Messages cannot be mutated after construction.
    """
    message: ThoughtMessage = ThoughtMessage(content='x')
    with pytest.raises(ValidationError):
        message.content = 'y'  # type: ignore[misc]


def test_ordinary_source_is_never_truncated() -> None:
    """
    Hand written source passes through untouched.

    Half of this project's own files exceed a few thousand characters, so the
    limit has to sit well clear of normal code.
    """
    source: str = 'def handler(request):\n    return process(request)\n' * 200
    body, truncated = truncate_diff_side(source)

    assert truncated is False
    assert body == source


def test_line_limit_applies() -> None:
    """
    A file with too many lines is cut and marked.
    """
    body, truncated = truncate_diff_side('line\n' * (MAX_DIFF_LINES + 500))

    assert truncated is True
    assert body.count('\n') <= MAX_DIFF_LINES + 1
    assert 'truncated' in body
    assert str(MAX_DIFF_LINES + 500) in body


def test_character_limit_applies_to_few_long_lines() -> None:
    """
    A minified bundle on one enormous line is cut by character count.

    The line limit alone would let a single-line file through at any size.
    """
    body, truncated = truncate_diff_side('x' * (MAX_DIFF_CHARACTERS + 10_000))

    assert truncated is True
    assert len(body) < MAX_DIFF_CHARACTERS + 200


def test_diff_message_reports_truncation() -> None:
    """
    An oversized diff is capped on the wire and flagged for the viewer.
    """
    message = to_message(
        DiffEvent(file_path='huge.lock', original='a\n' * 5000, modified='b\n' * 5000)
    )

    assert isinstance(message, DiffMessage)
    assert message.truncated is True
    assert len(message.original.splitlines()) <= MAX_DIFF_LINES + 2
    assert len(message.modified.splitlines()) <= MAX_DIFF_LINES + 2


def test_normal_diff_is_not_flagged() -> None:
    """
    A diff of ordinary size carries no truncation flag.
    """
    message = to_message(DiffEvent(file_path='a.py', original='x = 1\n', modified='x = 2\n'))

    assert isinstance(message, DiffMessage)
    assert message.truncated is False
    assert message.original == 'x = 1\n'


def test_truncation_is_one_sided_when_only_one_side_is_huge() -> None:
    """
    Deleting a huge file still flags truncation from the original side alone.
    """
    message = to_message(
        DiffEvent(file_path='gen.js', original='x\n' * 5000, modified='')
    )

    assert isinstance(message, DiffMessage)
    assert message.truncated is True
    assert message.modified == ''


def test_event_itself_keeps_the_full_content() -> None:
    """
    Only the wire representation is capped; the event remains complete.
    """
    event: DiffEvent = DiffEvent(file_path='huge.lock', original='a\n' * 5000, modified='')

    assert len(event.original.splitlines()) == 5000
