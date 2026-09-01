from typing import Annotated, Literal

from pydantic import Field

from .base_agent_message import BaseAgentMessage
from .message_type import MessageType


class ReviewMessage(BaseAgentMessage):
    """
    The reviewer agent's verdict on the cumulative diff.
    """

    type: Annotated[Literal[MessageType.review], Field()] = MessageType.review
    approved: Annotated[bool, Field()]
    comments: Annotated[str, Field()]
