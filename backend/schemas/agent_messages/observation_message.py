from typing import Annotated, Literal

from pydantic import Field

from services.events import ToolStatus

from .base_agent_message import BaseAgentMessage
from .message_type import MessageType


class ObservationMessage(BaseAgentMessage):
    """
    What a tool returned once it finished.
    """

    type: Annotated[Literal[MessageType.observation], Field()] = MessageType.observation
    tool_name: Annotated[str, Field()]
    result: Annotated[str, Field()]
    status: Annotated[ToolStatus, Field()] = ToolStatus.success
