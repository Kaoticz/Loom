from typing import Annotated, Literal

from pydantic import Field

from .agent_event_type import AgentEventType
from .base_agent_event import BaseAgentEvent


class QuestionEvent(BaseAgentEvent):
    """
    Emitted when the agent puts a question to the user and waits for an answer.
    """

    type: Annotated[Literal[AgentEventType.question], Field()] = AgentEventType.question
    question: Annotated[str, Field()]
    options: Annotated[list[str], Field()] = []
