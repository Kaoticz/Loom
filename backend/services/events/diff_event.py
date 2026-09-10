from typing import Annotated, Literal

from pydantic import Field

from .agent_event_type import AgentEventType
from .base_agent_event import BaseAgentEvent


class DiffEvent(BaseAgentEvent):
    """
    Emitted after a file modification, carrying both sides of the comparison.
    """

    type: Annotated[Literal[AgentEventType.diff], Field()] = AgentEventType.diff
    file_path: Annotated[str, Field(min_length=1)]
    original: Annotated[str, Field()]
    modified: Annotated[str, Field()]
