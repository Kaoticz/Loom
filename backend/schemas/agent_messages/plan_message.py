from typing import Annotated, Literal

from pydantic import Field

from services.plan_format import PlanFormat
from services.plan_step import PlanStep

from .base_agent_message import BaseAgentMessage
from .message_type import MessageType


class PlanMessage(BaseAgentMessage):
    """
    The plan the agent intends to follow, awaiting the user's approval.
    """

    type: Annotated[Literal[MessageType.plan], Field()] = MessageType.plan
    steps: Annotated[list[PlanStep], Field(default_factory=list)]
    # Carried rather than inferred from whether descriptions are present: a
    # structured plan whose descriptions came back empty is still structured,
    # and inferring would show it as plain text.
    format: Annotated[PlanFormat, Field()] = PlanFormat.text
    requires_approval: Annotated[bool, Field()] = True
