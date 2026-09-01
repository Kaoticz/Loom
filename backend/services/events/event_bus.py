from asyncio import Lock
from collections.abc import Awaitable, Callable, Sequence
from logging import Logger, getLogger
from typing import Annotated

from pydantic import Field

from .commit_event import CommitEvent
from .compaction_event import CompactionEvent
from .context_usage_event import ContextUsageEvent
from .diff_event import DiffEvent
from .done_event import DoneEvent
from .error_event import ErrorEvent
from .plan_event import PlanEvent
from .question_event import QuestionEvent
from .review_event import ReviewEvent
from .sandbox_unavailable_event import SandboxUnavailableEvent
from .system_event import SystemEvent
from .thought_event import ThoughtEvent
from .tool_action_event import ToolActionEvent
from .tool_observation_event import ToolObservationEvent
from .warning_event import WarningEvent

logger: Logger = getLogger(__name__)


INDEXING_PROGRESS_KEY: str = 'indexing_progress'
"""
The reserved slot key shared by every indexing progress update.

Every progress update carries this key, so the client replaces one message in
place instead of appending thousands as a repository is walked.
"""


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
    | CompactionEvent
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
