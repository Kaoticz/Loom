from typing import Annotated, Literal

from pydantic import Field

from .agent_event_type import AgentEventType
from .base_agent_event import BaseAgentEvent
from .sandbox_stage import SandboxStage


class SandboxUnavailableEvent(BaseAgentEvent):
    """
    The sandbox cannot be used, so nothing the agent writes can be verified.

    Carried as its own event rather than as an announcement because the client
    shows this as a persistent banner: a line in the chat feed scrolls away, and
    the constraint applies for the rest of the session. Only the technical reason
    travels; the explanation of what it means for the user is the interface's to
    word.
    """

    type: Annotated[Literal[AgentEventType.sandbox_unavailable], Field()] = (
        AgentEventType.sandbox_unavailable
    )
    stage: Annotated[SandboxStage, Field(description='Which step of preparing the sandbox failed.')]
    reason: Annotated[str, Field(description='Why the sandbox could not be used.')]
