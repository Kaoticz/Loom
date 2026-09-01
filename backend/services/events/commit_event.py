from typing import Annotated, Literal

from pydantic import Field

from .agent_event_type import AgentEventType
from .base_agent_event import BaseAgentEvent


class CommitEvent(BaseAgentEvent):
    """
    Emitted after a micro-commit is written to the task branch.
    """

    type: Annotated[Literal[AgentEventType.commit], Field()] = AgentEventType.commit
    commit_hash: Annotated[str, Field()]
    message: Annotated[str, Field()]
    files_changed: Annotated[list[str], Field(default_factory=list)]
