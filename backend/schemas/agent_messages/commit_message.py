from typing import Annotated, Literal

from pydantic import Field

from .base_agent_message import BaseAgentMessage
from .message_type import MessageType


class CommitMessage(BaseAgentMessage):
    """
    A micro-commit written to the task branch.
    """

    type: Annotated[Literal[MessageType.commit], Field()] = MessageType.commit
    commit_hash: Annotated[str, Field()]
    message: Annotated[str, Field()]
    files_changed: Annotated[list[str], Field(default_factory=list)]
