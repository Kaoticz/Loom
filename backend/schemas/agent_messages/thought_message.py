from typing import Annotated, Literal

from pydantic import Field

from .base_agent_message import BaseAgentMessage
from .message_type import MessageType


class ThoughtMessage(BaseAgentMessage):
    """
    A unit of the agent's reasoning, ready to display in the chat feed.

    Units sharing a `stream_id` belong to one continuous run of reasoning and are
    joined into a single growing message by the client, so that streamed
    reasoning does not arrive as one bubble per sentence.
    """

    type: Annotated[Literal[MessageType.thought], Field()] = MessageType.thought
    content: Annotated[str, Field()]
    stream_id: Annotated[str | None, Field(description='The run of reasoning this unit continues, or None if it stands alone.')] = None
