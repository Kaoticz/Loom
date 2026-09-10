from typing import Annotated, Literal

from pydantic import Field

from .agent_event_type import AgentEventType
from .base_agent_event import BaseAgentEvent


class ThoughtEvent(BaseAgentEvent):
    """
    A unit of agent reasoning intended for display in the chat feed.

    `stream_id` names the continuous run of reasoning this unit belongs to. The
    buffer releases reasoning a sentence at a time so the feed updates promptly,
    but the reader wants one growing message rather than one bubble per sentence,
    so consecutive units sharing an identifier are joined by the client. The
    event bus assigns the identifier and retires it as soon as any other kind of
    event is emitted, which is what separates thinking before a tool call from
    thinking after it.
    """

    type: Annotated[Literal[AgentEventType.thought], Field()] = AgentEventType.thought
    content: Annotated[str, Field()]
    stream_id: Annotated[str | None, Field(description='The run of reasoning this unit continues, assigned by the event bus.')] = None
