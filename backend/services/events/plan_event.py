from typing import Annotated, Literal

from pydantic import Field

from services.plan_format import PlanFormat
from services.plan_step import PlanStep

from .agent_event_type import AgentEventType
from .base_agent_event import BaseAgentEvent


class PlanEvent(BaseAgentEvent):
    """
    A plan produced by the coder agent, awaiting user approval.
    """

    type: Annotated[Literal[AgentEventType.plan], Field()] = AgentEventType.plan
    steps: Annotated[list[PlanStep], Field(default_factory=list)]
    format: Annotated[PlanFormat, Field()] = PlanFormat.text
    requires_approval: Annotated[bool, Field()] = True
