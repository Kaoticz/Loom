from typing import Annotated, Literal

from pydantic import Field

from .agent_event_type import AgentEventType
from .base_agent_event import BaseAgentEvent


class ReviewEvent(BaseAgentEvent):
    """
    Emitted with the reviewer agent's verdict on the cumulative diff.
    """

    type: Annotated[Literal[AgentEventType.review], Field()] = AgentEventType.review
    approved: Annotated[bool, Field()]
    comments: Annotated[str, Field()]
