from typing import Annotated, Literal

from pydantic import Field

from .agent_event_type import AgentEventType
from .base_agent_event import BaseAgentEvent


class WarningEvent(BaseAgentEvent):
    """
    Emitted when something failed but the session is carrying on regardless.

    Distinct from `ErrorEvent`, which means the session is over: a client that
    renders both the same way tells the user their run has died when it has not.
    Distinct from `SystemEvent`, which announces what phase the session is in and
    would bury a failure among routine progress.
    """

    type: Annotated[Literal[AgentEventType.warning], Field()] = AgentEventType.warning
    content: Annotated[str, Field()]
