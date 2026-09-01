from typing import Annotated, Any, Literal

from pydantic import Field

from .agent_event_type import AgentEventType
from .base_agent_event import BaseAgentEvent
from .tool_status import ToolStatus


class ToolActionEvent(BaseAgentEvent):
    """
    Emitted by a tool immediately before it executes.
    """

    type: Annotated[Literal[AgentEventType.action], Field()] = AgentEventType.action
    tool_name: Annotated[str, Field(min_length=1)]
    arguments: Annotated[dict[str, Any], Field(default_factory=dict)]
    status: Annotated[ToolStatus, Field()] = ToolStatus.running
