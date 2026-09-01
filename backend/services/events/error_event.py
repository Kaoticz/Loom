from typing import Annotated, Literal

from pydantic import Field

from .agent_event_type import AgentEventType
from .base_agent_event import BaseAgentEvent


class ErrorEvent(BaseAgentEvent):
    """
    Emitted when an unrecoverable error occurs during the session.
    """

    type: Annotated[Literal[AgentEventType.error], Field()] = AgentEventType.error
    content: Annotated[str, Field()]
