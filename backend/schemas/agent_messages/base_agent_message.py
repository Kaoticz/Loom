from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class BaseAgentMessage(BaseModel):
    """
    Common base for every WebSocket message.

    The `type` discriminator is declared by each concrete subclass, so that
    narrowing it to a single member is not an incompatible override.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
