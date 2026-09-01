from typing import Annotated, Literal

from pydantic import Field

from .agent_event_type import AgentEventType
from .base_agent_event import BaseAgentEvent
from .session_status import SessionStatus


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
