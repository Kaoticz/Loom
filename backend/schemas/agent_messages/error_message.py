from typing import Annotated, Literal

from pydantic import Field

from .base_agent_message import BaseAgentMessage
from .message_type import MessageType


class ErrorMessage(BaseAgentMessage):
    """
    The session hit an error it could not recover from.
    """

    type: Annotated[Literal[MessageType.error], Field()] = MessageType.error
    content: Annotated[str, Field()]
