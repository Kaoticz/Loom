from typing import Annotated, Literal

from pydantic import Field

from .agent_event_type import AgentEventType
from .base_agent_event import BaseAgentEvent
from .tool_status import ToolStatus


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
