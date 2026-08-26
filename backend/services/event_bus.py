from __future__ import annotations

from asyncio import Lock
from collections.abc import Awaitable, Callable, Sequence
from enum import Enum
from logging import Logger, getLogger
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from services.plan_parser import PlanFormat, PlanStep

logger: Logger = getLogger(__name__)


class AgentEventType(str, Enum):
    """
    Discriminator for every event that can travel through the event bus.
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
    system = 'system'
    sandbox_unavailable = 'sandbox_unavailable'
    question = 'question'


INDEXING_PROGRESS_KEY: str = 'indexing_progress'
"""
The reserved slot key shared by every indexing progress update.

Every progress update carries this key, so the client replaces one message in
place instead of appending thousands as a repository is walked.
"""


class SessionStatus(str, Enum):
    """
    The phase of work a session is announcing it has entered.

    The session already knows this: it is the state machine's own business. It is
    stated outright so the client does not have to infer it by matching the
    wording of announcements, which breaks silently the moment the wording
    changes.

    Only phases the session enters of its own accord appear here. Waiting for a
    plan or a review to be approved is driven by the plan and review messages
    themselves, not announced. `cancelled` is the exception that proves the rule:
    it is announced because nothing else tells the client the agent has stopped,
    and without it the interface would go on offering to stop work that already
    has.
    """

    indexing = 'indexing'
    classifying = 'classifying'
    planning = 'planning'
    executing = 'executing'
    answering = 'answering'
    reviewing = 'reviewing'
    re_working = 're_working'
    merging = 'merging'
    cancelled = 'cancelled'


class SandboxStage(str, Enum):
    """
    Which step of preparing the sandbox failed.

    Every cause used to surface as one indistinguishable sentence, which is what
    made an unusable sandbox hard to diagnose. Naming the step lets the interface
    and the reader tell a missing container runtime from a Dockerfile that will
    not build.
    """

    runtime_connectivity = 'runtime_connectivity'
    path_discovery = 'path_discovery'
    image_build = 'image_build'
    image_pull = 'image_pull'
    container_start = 'container_start'
    readiness_probe = 'readiness_probe'


class ToolStatus(str, Enum):
    """
    Lifecycle status of a tool invocation.
    """

    running = 'running'
    success = 'success'
    failure = 'failure'


class BaseAgentEvent(BaseModel):
    """
    Common base for all agent events emitted onto the event bus.

    The `type` discriminator is declared by each concrete subclass rather than
    here, so that narrowing it to a single enum member is not an incompatible
    override of a mutable base attribute.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class ThoughtEvent(BaseAgentEvent):
    """
    A unit of agent reasoning intended for display in the chat feed.

    `stream_id` names the continuous run of reasoning this unit belongs to. The
    buffer releases reasoning a sentence at a time so the feed updates promptly,
    but the reader wants one growing message rather than one bubble per sentence,
    so consecutive units sharing an identifier are joined by the client. The
    event bus assigns the identifier and retires it as soon as any other kind of
    event is emitted, which is what separates thinking before a tool call from
    thinking after it.
    """

    type: Annotated[Literal[AgentEventType.thought], Field()] = AgentEventType.thought
    content: Annotated[str, Field()]
    stream_id: Annotated[str | None, Field(description='The run of reasoning this unit continues, assigned by the event bus.')] = None


class PlanEvent(BaseAgentEvent):
    """
    A plan produced by the coder agent, awaiting user approval.
    """

    type: Annotated[Literal[AgentEventType.plan], Field()] = AgentEventType.plan
    steps: Annotated[list[PlanStep], Field(default_factory=list)]
    format: Annotated[PlanFormat, Field()] = PlanFormat.text
    requires_approval: Annotated[bool, Field()] = True


class ToolActionEvent(BaseAgentEvent):
    """
    Emitted by a tool immediately before it executes.
    """

    type: Annotated[Literal[AgentEventType.action], Field()] = AgentEventType.action
    tool_name: Annotated[str, Field(min_length=1)]
    arguments: Annotated[dict[str, Any], Field(default_factory=dict)]
    status: Annotated[ToolStatus, Field()] = ToolStatus.running


class ToolObservationEvent(BaseAgentEvent):
    """
    Emitted by a tool immediately after it executes.
    """

    type: Annotated[Literal[AgentEventType.observation], Field()] = AgentEventType.observation
    tool_name: Annotated[str, Field(min_length=1)]
    result: Annotated[str, Field()]
    status: Annotated[ToolStatus, Field()] = ToolStatus.success

    @property
    def success(self) -> bool:
        """
        Whether the observed tool call succeeded.

        :returns: True when the status is `ToolStatus.success`.
        """
        return self.status is ToolStatus.success


class CommitEvent(BaseAgentEvent):
    """
    Emitted after a micro-commit is written to the task branch.
    """

    type: Annotated[Literal[AgentEventType.commit], Field()] = AgentEventType.commit
    commit_hash: Annotated[str, Field()]
    message: Annotated[str, Field()]
    files_changed: Annotated[list[str], Field(default_factory=list)]


class DiffEvent(BaseAgentEvent):
    """
    Emitted after a file modification, carrying both sides of the comparison.
    """

    type: Annotated[Literal[AgentEventType.diff], Field()] = AgentEventType.diff
    file_path: Annotated[str, Field(min_length=1)]
    original: Annotated[str, Field()]
    modified: Annotated[str, Field()]


class QuestionEvent(BaseAgentEvent):
    """
    Emitted when the agent puts a question to the user and waits for an answer.
    """

    type: Annotated[Literal[AgentEventType.question], Field()] = AgentEventType.question
    question: Annotated[str, Field()]
    options: Annotated[list[str], Field()] = []


class ReviewEvent(BaseAgentEvent):
    """
    Emitted with the reviewer agent's verdict on the cumulative diff.
    """

    type: Annotated[Literal[AgentEventType.review], Field()] = AgentEventType.review
    approved: Annotated[bool, Field()]
    comments: Annotated[str, Field()]


class DoneEvent(BaseAgentEvent):
    """
    Emitted when the session reaches a successful conclusion.
    """

    type: Annotated[Literal[AgentEventType.done], Field()] = AgentEventType.done
    summary: Annotated[str, Field()]
    branch_name: Annotated[str, Field()] = ''
    total_commits: Annotated[int, Field(ge=0)] = 0


class ErrorEvent(BaseAgentEvent):
    """
    Emitted when an unrecoverable error occurs during the session.
    """

    type: Annotated[Literal[AgentEventType.error], Field()] = AgentEventType.error
    content: Annotated[str, Field()]


class ContextUsageEvent(BaseAgentEvent):
    """
    How much of the model's context window the next request will occupy.

    Carries the raw counts rather than a rendered string, so the interface can
    show a gauge and decide its own wording. `resolved` is false when the window
    is a fallback rather than the model's own, which makes the percentage a
    guess the interface should be able to mark as one.
    """

    type: Annotated[Literal[AgentEventType.context_usage], Field()] = (
        AgentEventType.context_usage
    )
    used_tokens: Annotated[int, Field(ge=0)]
    window_tokens: Annotated[int, Field(gt=0)]
    resolved: Annotated[bool, Field()] = True


class WarningEvent(BaseAgentEvent):
    """
    Emitted when something failed but the session is carrying on regardless.

    Distinct from `ErrorEvent`, which means the session is over: a client that
    renders both the same way tells the user their run has died when it has not.
    Distinct from `SystemEvent`, which announces what phase the session is in and
    would bury a failure among routine progress.
    """

    type: Annotated[Literal[AgentEventType.warning], Field()] = AgentEventType.warning
    content: Annotated[str, Field()]


class SystemEvent(BaseAgentEvent):
    """
    Loom talking about the session, as distinct from the model reasoning.

    A `key` marks the event as an update to a named slot in the feed rather than
    a new entry, so repeated updates about one long running activity replace
    each other instead of accumulating. `processed` and `total` are carried
    separately from `content` so the client can render a counter without having
    to parse prose.
    """

    type: Annotated[Literal[AgentEventType.system], Field()] = AgentEventType.system
    content: Annotated[str, Field()]
    key: Annotated[str | None, Field(description='The feed slot this event updates, or None to append a new entry.')] = None
    status: Annotated[SessionStatus | None, Field(description='The phase the session has entered, when this event announces one.')] = None
    processed: Annotated[int | None, Field(ge=0, description='Units of work finished, when this event reports progress.')] = None
    total: Annotated[int | None, Field(ge=0, description='Units of work expected, when this event reports progress.')] = None


class SandboxUnavailableEvent(BaseAgentEvent):
    """
    The sandbox cannot be used, so nothing the agent writes can be verified.

    Carried as its own event rather than as an announcement because the client
    shows this as a persistent banner: a line in the chat feed scrolls away, and
    the constraint applies for the rest of the session. Only the technical reason
    travels; the explanation of what it means for the user is the interface's to
    word.
    """

    type: Annotated[Literal[AgentEventType.sandbox_unavailable], Field()] = (
        AgentEventType.sandbox_unavailable
    )
    stage: Annotated[SandboxStage, Field(description='Which step of preparing the sandbox failed.')]
    reason: Annotated[str, Field(description='Why the sandbox could not be used.')]


AgentEvent = Annotated[
    ThoughtEvent
    | PlanEvent
    | ToolActionEvent
    | ToolObservationEvent
    | CommitEvent
    | DiffEvent
    | QuestionEvent
    | ReviewEvent
    | DoneEvent
    | ErrorEvent
    | WarningEvent
    | ContextUsageEvent
    | SystemEvent
    | SandboxUnavailableEvent,
    Field(discriminator='type'),
]

EventHandler = Callable[[AgentEvent], Awaitable[None]]


class EventBus:
    """
    Session-scoped asynchronous event emitter.

    Tools emit domain events onto the bus without knowing anything about the
    transport. The WebSocket controller subscribes a handler that serialises
    those events to the client. One instance exists per WebSocket session, so
    events raised in one session can never reach another.
    """

    _session_id: str
    _handlers: list[EventHandler]
    _lock: Lock
    _thought_stream: str | None
    _thought_streams_started: int

    def __init__(self, session_id: str) -> None:
        """
        Create an event bus bound to a single agent session.

        :param session_id: The identifier of the session this bus belongs to.
        """
        self._session_id = session_id
        self._handlers = []
        self._lock = Lock()
        self._thought_stream = None
        self._thought_streams_started = 0

    @property
    def thought_stream(self) -> str | None:
        """
        The run of reasoning currently in progress, if any.

        :returns: The active thought stream identifier, or None when the last
            event emitted was not a thought.
        """
        return self._thought_stream

    @property
    def session_id(self) -> str:
        """
        The identifier of the session this bus belongs to.

        :returns: The session identifier.
        """
        return self._session_id

    @property
    def handlers(self) -> Sequence[EventHandler]:
        """
        The currently registered handlers.

        :returns: An immutable view over the registered handlers.
        """
        return tuple(self._handlers)

    def subscribe(self, handler: EventHandler) -> EventHandler:
        """
        Register a handler to be invoked for every emitted event.

        Registering the same handler twice is a no-op.

        :param handler: The asynchronous callable to invoke on each event.
        :returns: The registered handler, so this method may be used as a decorator.
        """
        if handler in self._handlers:
            return handler
        self._handlers.append(handler)
        return handler

    def unsubscribe(self, handler: EventHandler) -> bool:
        """
        Remove a previously registered handler.

        :param handler: The handler to remove.
        :returns: True if the handler was registered and has been removed, False otherwise.
        """
        if handler not in self._handlers:
            return False
        self._handlers.remove(handler)
        return True

    def clear(self) -> int:
        """
        Remove every registered handler.

        :returns: The number of handlers that were removed.
        """
        removed: int = len(self._handlers)
        self._handlers.clear()
        return removed

    def _stamped(self, event: AgentEvent) -> AgentEvent:
        """
        Label a thought with the run of reasoning it continues.

        The bus is the one place every event passes through, so it is the only
        place the rule can be enforced for all emitters: any event that is not a
        thought retires the current identifier, and the next thought therefore
        starts a new message in the feed. An identifier already present on the
        event is respected rather than overwritten.

        :param event: The event about to be dispatched.
        :returns: The event to dispatch, labelled if it is an unlabelled thought.
        """
        if not isinstance(event, ThoughtEvent):
            self._thought_stream = None
            return event

        if event.stream_id is not None:
            return event

        if self._thought_stream is None:
            self._thought_streams_started += 1
            self._thought_stream = f'{self._session_id}:{self._thought_streams_started}'

        return event.model_copy(update={'stream_id': self._thought_stream})

    async def emit(self, event: AgentEvent) -> int:
        """
        Dispatch an event to every registered handler.

        Handlers are invoked sequentially in registration order. A handler that
        raises is logged and skipped so that one failing subscriber (for example
        a closed WebSocket) cannot prevent the others from observing the event.

        :param event: The event to dispatch.
        :returns: The number of handlers that received the event without raising.
        """
        stamped: AgentEvent = self._stamped(event)

        async with self._lock:
            snapshot: tuple[EventHandler, ...] = tuple(self._handlers)

        delivered: int = 0
        for handler in snapshot:
            try:
                await handler(stamped)
                delivered += 1
            except Exception as error:
                logger.warning(
                    'Event handler failed for %s event in session %s: %s',
                    stamped.type.value,
                    self._session_id,
                    error,
                )
        return delivered
