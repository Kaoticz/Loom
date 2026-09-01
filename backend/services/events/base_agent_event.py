from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class BaseAgentEvent(BaseModel):
    """
    Common base for all agent events emitted onto the event bus.

    The `type` discriminator is declared by each concrete subclass rather than
    here, so that narrowing it to a single enum member is not an incompatible
    override of a mutable base attribute.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
