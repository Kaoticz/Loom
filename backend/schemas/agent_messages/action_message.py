from typing import Annotated, Any, Literal

from pydantic import Field

from services.events import ToolStatus

from .base_agent_message import BaseAgentMessage
from .message_type import MessageType


class ActionMessage(BaseAgentMessage):
    """
    A tool the agent is about to run.
    """

    type: Annotated[Literal[MessageType.action], Field()] = MessageType.action
    tool_name: Annotated[str, Field()]
    arguments: Annotated[dict[str, Any], Field(default_factory=dict)]
    status: Annotated[ToolStatus, Field()] = ToolStatus.running
