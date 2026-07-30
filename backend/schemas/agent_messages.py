from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

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


MAX_DIFF_LINES: int = 2000
MAX_DIFF_CHARACTERS: int = 250_000


def truncate_diff_side(
    text: str,
    max_lines: int = MAX_DIFF_LINES,
    max_characters: int = MAX_DIFF_CHARACTERS,
) -> tuple[str, bool]:
    """
    Shorten one side of a diff so an oversized file cannot flood the client.

    Whichever limit is reached first applies. Hand written source comfortably
    fits both; the limits exist for lockfiles, generated code and minified
    bundles, where sending the whole file helps nobody.

    :param text: The full contents of one side of the diff.
    :param max_lines: The greatest number of lines to send.
    :param max_characters: The greatest number of characters to send.
    :returns: The text to send, and whether anything was removed.
    """
    lines: list[str] = text.splitlines(keepends=True)
    kept: list[str] = lines[:max_lines]
    body: str = ''.join(kept)
    truncated: bool = len(kept) < len(lines)

    if len(body) > max_characters:
        body = body[:max_characters]
        truncated = True

    if not truncated:
        return text, False

    return (
        f'{body}\n... (truncated: showing the first {len(body.splitlines())} '
        f'of {len(lines)} lines)',
        True,
    )


class MessageType(str, Enum):
    """
    Discriminator for every message the backend can send over the WebSocket.
    """

    thought = 'thought'
    plan = 'plan'
    action = 'action'
    observation = 'observation'
    commit = 'commit'
    diff = 'diff'
    review = 'review'
    done = 'done'
    error = 'error'
    session_busy = 'session_busy'


class BaseAgentMessage(BaseModel):
    """
    Common base for every WebSocket message.

    The `type` discriminator is declared by each concrete subclass, so that
    narrowing it to a single member is not an incompatible override.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class ThoughtMessage(BaseAgentMessage):
    """
    A unit of the agent's reasoning, ready to display in the chat feed.
    """

    type: Annotated[Literal[MessageType.thought], Field()] = MessageType.thought
    content: Annotated[str, Field()]


class PlanMessage(BaseAgentMessage):
    """
    The plan the agent intends to follow, awaiting the user's approval.
    """

    type: Annotated[Literal[MessageType.plan], Field()] = MessageType.plan
    steps: Annotated[list[str], Field(default_factory=list)]
    requires_approval: Annotated[bool, Field()] = True


class ActionMessage(BaseAgentMessage):
    """
    A tool the agent is about to run.
    """

    type: Annotated[Literal[MessageType.action], Field()] = MessageType.action
    tool_name: Annotated[str, Field()]
    arguments: Annotated[dict[str, Any], Field(default_factory=dict)]
    status: Annotated[ToolStatus, Field()] = ToolStatus.running


class ObservationMessage(BaseAgentMessage):
    """
    What a tool returned once it finished.
    """

    type: Annotated[Literal[MessageType.observation], Field()] = MessageType.observation
    tool_name: Annotated[str, Field()]
    result: Annotated[str, Field()]
    status: Annotated[ToolStatus, Field()] = ToolStatus.success


class CommitMessage(BaseAgentMessage):
    """
    A micro-commit written to the task branch.
    """

    type: Annotated[Literal[MessageType.commit], Field()] = MessageType.commit
    commit_hash: Annotated[str, Field()]
    message: Annotated[str, Field()]
    files_changed: Annotated[list[str], Field(default_factory=list)]


class DiffMessage(BaseAgentMessage):
    """
    Both sides of a file modification, for the diff viewer.
    """

    type: Annotated[Literal[MessageType.diff], Field()] = MessageType.diff
    file_path: Annotated[str, Field()]
    original: Annotated[str, Field()]
    modified: Annotated[str, Field()]
    truncated: Annotated[bool, Field(description='Whether either side was shortened because the file is too large to display in full.')] = False


class ReviewMessage(BaseAgentMessage):
    """
    The reviewer agent's verdict on the cumulative diff.
    """

    type: Annotated[Literal[MessageType.review], Field()] = MessageType.review
    approved: Annotated[bool, Field()]
    comments: Annotated[str, Field()]


class DoneMessage(BaseAgentMessage):
    """
    The session finished successfully.
    """

    type: Annotated[Literal[MessageType.done], Field()] = MessageType.done
    summary: Annotated[str, Field()]
    branch_name: Annotated[str, Field()] = ''
    total_commits: Annotated[int, Field(ge=0)] = 0


class ErrorMessage(BaseAgentMessage):
    """
    The session hit an error it could not recover from.
    """

    type: Annotated[Literal[MessageType.error], Field()] = MessageType.error
    content: Annotated[str, Field()]


class SessionBusyMessage(BaseAgentMessage):
    """
    A session is already running, so this connection cannot start one.

    Distinct from an error: nothing has gone wrong, and the client should offer
    to retry rather than present the session as failed.
    """

    type: Annotated[Literal[MessageType.session_busy], Field()] = MessageType.session_busy
    content: Annotated[str, Field()]
    active_session_id: Annotated[str, Field(description='The session currently holding the slot.')] = ''


AgentMessage = Annotated[
    ThoughtMessage
    | PlanMessage
    | ActionMessage
    | ObservationMessage
    | CommitMessage
    | DiffMessage
    | ReviewMessage
    | DoneMessage
    | ErrorMessage
    | SessionBusyMessage,
    Field(discriminator='type'),
]


def to_message(event: BaseAgentEvent) -> AgentMessage:
    """
    Convert a domain event into the message sent over the WebSocket.

    Keeping the wire format distinct from the event lets the protocol change
    without disturbing the tools that raise events, and gives the client a
    single documented contract.

    :param event: The event emitted onto the session's event bus.
    :returns: The message to serialise and send.
    :raises TypeError: If the event has no corresponding message type.
    """
    if isinstance(event, ThoughtEvent):
        return ThoughtMessage(content=event.content)
    if isinstance(event, PlanEvent):
        return PlanMessage(steps=list(event.steps), requires_approval=event.requires_approval)
    if isinstance(event, ToolActionEvent):
        return ActionMessage(
            tool_name=event.tool_name,
            arguments=dict(event.arguments),
            status=event.status,
        )
    if isinstance(event, ToolObservationEvent):
        return ObservationMessage(
            tool_name=event.tool_name,
            result=event.result,
            status=event.status,
        )
    if isinstance(event, CommitEvent):
        return CommitMessage(
            commit_hash=event.commit_hash,
            message=event.message,
            files_changed=list(event.files_changed),
        )
    if isinstance(event, DiffEvent):
        # The cap lives here, at the wire boundary, so the event itself stays a
        # faithful record of the change.
        original, original_cut = truncate_diff_side(event.original)
        modified, modified_cut = truncate_diff_side(event.modified)
        return DiffMessage(
            file_path=event.file_path,
            original=original,
            modified=modified,
            truncated=original_cut or modified_cut,
        )
    if isinstance(event, ReviewEvent):
        return ReviewMessage(approved=event.approved, comments=event.comments)
    if isinstance(event, DoneEvent):
        return DoneMessage(
            summary=event.summary,
            branch_name=event.branch_name,
            total_commits=event.total_commits,
        )
    if isinstance(event, ErrorEvent):
        return ErrorMessage(content=event.content)

    raise TypeError(f'No WebSocket message is defined for {type(event).__name__}.')
