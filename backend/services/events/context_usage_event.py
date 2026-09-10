from typing import Annotated, Literal

from pydantic import Field

from .agent_event_type import AgentEventType
from .base_agent_event import BaseAgentEvent


class ContextUsageEvent(BaseAgentEvent):
    """
    How much of the model's context window the next request will occupy.

    Carries the raw counts rather than a rendered string, so the interface can
    show a gauge and decide its own wording. `resolved` is false when the window
    is a fallback rather than the model's own, which makes the percentage a
    guess the interface should be able to mark as one.
    """

    type: Annotated[Literal[AgentEventType.context_usage], Field()] = (
        AgentEventType.context_usage
    )
    used_tokens: Annotated[int, Field(ge=0)]
    window_tokens: Annotated[int, Field(gt=0)]
    resolved: Annotated[bool, Field()] = True
