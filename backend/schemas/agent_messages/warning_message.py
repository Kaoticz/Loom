from typing import Annotated, Literal

from pydantic import Field

from .base_agent_message import BaseAgentMessage
from .message_type import MessageType


class WarningMessage(BaseAgentMessage):
    """
    Something failed but the session is carrying on.

    Separate from `ErrorMessage`, which ends the session. A client that renders
    both alike tells the user their run has died when it has not.
    """

    type: Annotated[Literal[MessageType.warning], Field()] = MessageType.warning
    content: Annotated[str, Field()]
