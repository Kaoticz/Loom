from typing import Annotated, Literal

from pydantic import Field

from .base_agent_message import BaseAgentMessage
from .message_type import MessageType


class ContextUsageMessage(BaseAgentMessage):
    """
    How full the model's context is, as counts the interface can render.
    """

    type: Annotated[Literal[MessageType.context_usage], Field()] = MessageType.context_usage
    used_tokens: Annotated[int, Field(ge=0, description='Tokens the next request will carry.')]
    window_tokens: Annotated[int, Field(gt=0, description='The context window being measured against.')]
    resolved: Annotated[bool, Field(description='False when the window is a fallback rather than the model\'s own, making the percentage a guess.')] = True
