from typing import Annotated, Literal

from pydantic import Field

from .base_agent_message import BaseAgentMessage
from .message_type import MessageType


class DoneMessage(BaseAgentMessage):
    """
    The session finished successfully.
    """

    type: Annotated[Literal[MessageType.done], Field()] = MessageType.done
    summary: Annotated[str, Field()]
    branch_name: Annotated[str, Field()] = ''
    total_commits: Annotated[int, Field(ge=0)] = 0
    merged: Annotated[bool, Field()] = False
