from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from services.plan_parser import PlanFormat, PlanStep

from services.event_bus import (
    BaseAgentEvent,
    CommitEvent,
    DiffEvent,
    DoneEvent,
    ErrorEvent,
    WarningEvent,
    ContextUsageEvent,
    PlanEvent,
    QuestionEvent,
    ReviewEvent,
    SandboxStage,
    SandboxUnavailableEvent,
    SessionStatus,
    SystemEvent,
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
    warning = 'warning'
    context_usage = 'context_usage'
    session_busy = 'session_busy'
    question = 'question'
    system = 'system'
    sandbox_unavailable = 'sandbox_unavailable'


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

    Units sharing a `stream_id` belong to one continuous run of reasoning and are
    joined into a single growing message by the client, so that streamed
    reasoning does not arrive as one bubble per sentence.
    """

    type: Annotated[Literal[MessageType.thought], Field()] = MessageType.thought
    content: Annotated[str, Field()]
    stream_id: Annotated[str | None, Field(description='The run of reasoning this unit continues, or None if it stands alone.')] = None


class PlanMessage(BaseAgentMessage):
    """
    The plan the agent intends to follow, awaiting the user's approval.
    """

    type: Annotated[Literal[MessageType.plan], Field()] = MessageType.plan
    steps: Annotated[list[PlanStep], Field(default_factory=list)]
    # Carried rather than inferred from whether descriptions are present: a
    # structured plan whose descriptions came back empty is still structured,
    # and inferring would show it as plain text.
    format: Annotated[PlanFormat, Field()] = PlanFormat.text
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


class QuestionMessage(BaseAgentMessage):
    """
    A question the agent has put to the user, with the answers it suggests.

    The suggestions are only suggestions: the interface adds an option to reply
    freely, which the agent neither supplies nor can suppress. A question with
    no suggestions at all is therefore still a usable one.
    """

    type: Annotated[Literal[MessageType.question], Field()] = MessageType.question
    question: Annotated[str, Field()]
    options: Annotated[list[str], Field()] = []


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


class ContextUsageMessage(BaseAgentMessage):
    """
    How full the model's context is, as counts the interface can render.
    """

    type: Annotated[Literal[MessageType.context_usage], Field()] = MessageType.context_usage
    used_tokens: Annotated[int, Field(ge=0, description='Tokens the next request will carry.')]
    window_tokens: Annotated[int, Field(gt=0, description='The context window being measured against.')]
    resolved: Annotated[bool, Field(description='False when the window is a fallback rather than the model\'s own, making the percentage a guess.')] = True


class WarningMessage(BaseAgentMessage):
    """
    Something failed but the session is carrying on.

    Separate from `ErrorMessage`, which ends the session. A client that renders
    both alike tells the user their run has died when it has not.
    """

    type: Annotated[Literal[MessageType.warning], Field()] = MessageType.warning
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


class SystemMessage(BaseAgentMessage):
    """
    A statement by Loom about the session, rather than the model's reasoning.

    When `key` is set, the client replaces the message already occupying that
    slot instead of appending a new one, which is how a long running activity
    reports progress without filling the feed. `processed` and `total` let the
    client render a counter directly rather than parsing it back out of `content`.
    """

    type: Annotated[Literal[MessageType.system], Field()] = MessageType.system
    content: Annotated[str, Field()]
    key: Annotated[str | None, Field(description='The feed slot to replace, or None to append a new message.')] = None
    status: Annotated[SessionStatus | None, Field(description='The phase the session has entered, when this message announces one.')] = None
    processed: Annotated[int | None, Field(ge=0, description='Units of work finished, when this message reports progress.')] = None
    total: Annotated[int | None, Field(ge=0, description='Units of work expected, when this message reports progress.')] = None


class SandboxUnavailableMessage(BaseAgentMessage):
    """
    Notice that the sandbox is unusable for the rest of the session.

    Shown as a banner rather than added to the chat feed, so it stays visible
    while the constraint holds instead of scrolling away.
    """

    type: Annotated[Literal[MessageType.sandbox_unavailable], Field()] = (
        MessageType.sandbox_unavailable
    )
    stage: Annotated[SandboxStage, Field(description='Which step of preparing the sandbox failed.')]
    reason: Annotated[str, Field(description='Why the sandbox could not be used.')]


AgentMessage = Annotated[
    ThoughtMessage
    | PlanMessage
    | ActionMessage
    | ObservationMessage
    | CommitMessage
    | DiffMessage
    | QuestionMessage
    | ReviewMessage
    | DoneMessage
    | ErrorMessage
    | WarningMessage
    | ContextUsageMessage
    | SessionBusyMessage
    | SystemMessage
    | SandboxUnavailableMessage,
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
        return ThoughtMessage(content=event.content, stream_id=event.stream_id)
    if isinstance(event, PlanEvent):
        return PlanMessage(
            steps=list(event.steps),
            # Dropped here once, which left every structured plan arriving as
            # `text` and rendering without its descriptions: the whole feature was
            # unreachable while every test stayed green.
            format=event.format,
            requires_approval=event.requires_approval,
        )
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
    if isinstance(event, QuestionEvent):
        return QuestionMessage(question=event.question, options=list(event.options))
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
    if isinstance(event, WarningEvent):
        return WarningMessage(content=event.content)
    if isinstance(event, ContextUsageEvent):
        return ContextUsageMessage(
            used_tokens=event.used_tokens,
            window_tokens=event.window_tokens,
            resolved=event.resolved,
        )
    if isinstance(event, SandboxUnavailableEvent):
        return SandboxUnavailableMessage(stage=event.stage, reason=event.reason)
    if isinstance(event, SystemEvent):
        return SystemMessage(
            content=event.content,
            key=event.key,
            status=event.status,
            processed=event.processed,
            total=event.total,
        )

    raise TypeError(f'No WebSocket message is defined for {type(event).__name__}.')
