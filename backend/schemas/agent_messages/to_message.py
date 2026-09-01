from services.events import (
    BaseAgentEvent,
    CommitEvent,
    CompactionEvent,
    DiffEvent,
    DoneEvent,
    ErrorEvent,
    WarningEvent,
    ContextUsageEvent,
    PlanEvent,
    QuestionEvent,
    ReviewEvent,
    SandboxUnavailableEvent,
    SystemEvent,
    ThoughtEvent,
    ToolActionEvent,
    ToolObservationEvent,
)

from .action_message import ActionMessage
from .agent_message import AgentMessage
from .commit_message import CommitMessage
from .compaction_message import CompactionMessage
from .context_usage_message import ContextUsageMessage
from .diff_message import DiffMessage, truncate_diff_side
from .done_message import DoneMessage
from .error_message import ErrorMessage
from .observation_message import ObservationMessage
from .plan_message import PlanMessage
from .question_message import QuestionMessage
from .review_message import ReviewMessage
from .sandbox_unavailable_message import SandboxUnavailableMessage
from .system_message import SystemMessage
from .thought_message import ThoughtMessage
from .warning_message import WarningMessage


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
            merged=event.merged,
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
    if isinstance(event, CompactionEvent):
        return CompactionMessage(
            tokens_before=event.tokens_before,
            tokens_after=event.tokens_after,
            run_path=event.run_path,
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
