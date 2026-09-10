from typing import Annotated, Literal

from pydantic import Field

from .agent_event_type import AgentEventType
from .base_agent_event import BaseAgentEvent
from .compaction_run_path import CompactionRunPath


class CompactionEvent(BaseAgentEvent):
    """
    A request history shortened so it can fit the model's context window.

    The counts are Loom estimates, not provider-reported usage, so the interface
    can explain the routine housekeeping without parsing prose.
    """

    type: Annotated[Literal[AgentEventType.compaction], Field()] = AgentEventType.compaction
    tokens_before: Annotated[int, Field(ge=0, description='Estimated tokens in the request history before compaction.')]
    tokens_after: Annotated[int, Field(ge=0, description='Estimated tokens in the request history after compaction.')]
    run_path: Annotated[CompactionRunPath, Field(description='The long-running agent path whose request was compacted.')]
