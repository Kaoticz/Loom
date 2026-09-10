from typing import Annotated, Literal

from pydantic import Field

from services.events import SessionStatus

from .base_agent_message import BaseAgentMessage
from .message_type import MessageType


class SystemMessage(BaseAgentMessage):
    """
    A statement by Loom about the session, rather than the model's reasoning.

    When `key` is set, the client replaces the message already occupying that
    slot instead of appending a new one, which is how a long running activity
    reports progress without filling the feed. `processed` and `total` let the
    client render a counter directly rather than parsing it back out of `content`.
    """

    type: Annotated[Literal[MessageType.system], Field()] = MessageType.system
    content: Annotated[str, Field()]
    key: Annotated[str | None, Field(description='The feed slot to replace, or None to append a new message.')] = None
    status: Annotated[SessionStatus | None, Field(description='The phase the session has entered, when this message announces one.')] = None
    processed: Annotated[int | None, Field(ge=0, description='Units of work finished, when this message reports progress.')] = None
    total: Annotated[int | None, Field(ge=0, description='Units of work expected, when this message reports progress.')] = None
