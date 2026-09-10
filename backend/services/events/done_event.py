from typing import Annotated, Literal

from pydantic import Field

from .agent_event_type import AgentEventType
from .base_agent_event import BaseAgentEvent


class DoneEvent(BaseAgentEvent):
    """
    Emitted when the session reaches a successful conclusion.
    """

    type: Annotated[Literal[AgentEventType.done], Field()] = AgentEventType.done
    summary: Annotated[str, Field()]
    branch_name: Annotated[str, Field()] = ''
    total_commits: Annotated[int, Field(ge=0)] = 0
    merged: Annotated[bool, Field()] = False
