from typing import Annotated, Literal

from pydantic import Field

from .base_agent_message import BaseAgentMessage
from .message_type import MessageType


class SessionBusyMessage(BaseAgentMessage):
    """
    A session is already running, so this connection cannot start one.

    Distinct from an error: nothing has gone wrong, and the client should offer
    to retry rather than present the session as failed.
    """

    type: Annotated[Literal[MessageType.session_busy], Field()] = MessageType.session_busy
    content: Annotated[str, Field()]
    active_session_id: Annotated[str, Field(description='The session currently holding the slot.')] = ''
