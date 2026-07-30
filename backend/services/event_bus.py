from __future__ import annotations

from asyncio import Lock
from collections.abc import Awaitable, Callable, Sequence
from enum import Enum
from logging import Logger, getLogger
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

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
    """

    type: Annotated[Literal[AgentEventType.thought], Field()] = AgentEventType.thought
    content: Annotated[str, Field()]


class PlanEvent(BaseAgentEvent):
    """
    A plan produced by the coder agent, awaiting user approval.
    """

    type: Annotated[Literal[AgentEventType.plan], Field()] = AgentEventType.plan
    steps: Annotated[list[str], Field(default_factory=list)]
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


AgentEvent = Annotated[
    ThoughtEvent
    | PlanEvent
    | ToolActionEvent
    | ToolObservationEvent
    | CommitEvent
    | DiffEvent
    | ReviewEvent
    | DoneEvent
    | ErrorEvent,
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

    def __init__(self, session_id: str) -> None:
        """
        Create an event bus bound to a single agent session.

        :param session_id: The identifier of the session this bus belongs to.
        """
        self._session_id = session_id
        self._handlers = []
        self._lock = Lock()

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

    async def emit(self, event: AgentEvent) -> int:
        """
        Dispatch an event to every registered handler.

        Handlers are invoked sequentially in registration order. A handler that
        raises is logged and skipped so that one failing subscriber (for example
        a closed WebSocket) cannot prevent the others from observing the event.

        :param event: The event to dispatch.
        :returns: The number of handlers that received the event without raising.
        """
        async with self._lock:
            snapshot: tuple[EventHandler, ...] = tuple(self._handlers)

        delivered: int = 0
        for handler in snapshot:
            try:
                await handler(event)
                delivered += 1
            except Exception as error:
                logger.warning(
                    'Event handler failed for %s event in session %s: %s',
                    event.type.value,
                    self._session_id,
                    error,
                )
        return delivered
