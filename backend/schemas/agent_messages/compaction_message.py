from typing import Annotated, Literal

from pydantic import Field

from services.events import CompactionRunPath

from .base_agent_message import BaseAgentMessage
from .message_type import MessageType


class CompactionMessage(BaseAgentMessage):
    """
    A request history Loom shortened to fit the context window.

    The figures are estimated from the history rather than provider-reported
    counts, allowing the interface to word the notice without parsing prose.
    """

    type: Annotated[Literal[MessageType.compaction], Field()] = MessageType.compaction
    tokens_before: Annotated[int, Field(ge=0, description='Estimated tokens in the request history before compaction.')]
    tokens_after: Annotated[int, Field(ge=0, description='Estimated tokens in the request history after compaction.')]
    run_path: Annotated[CompactionRunPath, Field(description='The long-running agent path whose request was compacted.')]
