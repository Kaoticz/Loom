from typing import Annotated, Literal

from pydantic import Field

from services.events import SandboxStage

from .base_agent_message import BaseAgentMessage
from .message_type import MessageType


class SandboxUnavailableMessage(BaseAgentMessage):
    """
    Notice that the sandbox is unusable for the rest of the session.

    Shown as a banner rather than added to the chat feed, so it stays visible
    while the constraint holds instead of scrolling away.
    """

    type: Annotated[Literal[MessageType.sandbox_unavailable], Field()] = (
        MessageType.sandbox_unavailable
    )
    stage: Annotated[SandboxStage, Field(description='Which step of preparing the sandbox failed.')]
    reason: Annotated[str, Field(description='Why the sandbox could not be used.')]
