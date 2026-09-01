from typing import Annotated, Literal

from pydantic import Field

from .base_agent_message import BaseAgentMessage
from .message_type import MessageType


class QuestionMessage(BaseAgentMessage):
    """
    A question the agent has put to the user, with the answers it suggests.

    The suggestions are only suggestions: the interface adds an option to reply
    freely, which the agent neither supplies nor can suppress. A question with
    no suggestions at all is therefore still a usable one.
    """

    type: Annotated[Literal[MessageType.question], Field()] = MessageType.question
    question: Annotated[str, Field()]
    options: Annotated[list[str], Field()] = []
