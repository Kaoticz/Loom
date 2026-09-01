from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from services.turn_role import TurnRole


class ConversationTurn(BaseModel):
    """
    A single exchange in the accumulated agent conversation.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    role: Annotated[TurnRole, Field()]
    content: Annotated[str, Field()]
    iteration: Annotated[int, Field(ge=0)] = 0
    tool_name: Annotated[str | None, Field()] = None
    metadata: Annotated[dict[str, Any], Field(default_factory=dict)]
    is_code: Annotated[bool, Field()] = False

    def render(self) -> str:
        """
        Render this turn as a labelled block for injection into a prompt.

        :returns: The rendered representation of the turn.
        """
        if self.role is TurnRole.tool_call:
            return f'[tool_call] {self.tool_name or "unknown"}: {self.content}'
        if self.role is TurnRole.tool_result:
            return f'[tool_result] {self.tool_name or "unknown"}: {self.content}'
        return f'[{self.role.value}] {self.content}'

    def estimated_tokens(self) -> int:
        """
        Approximate the token cost of rendering this turn.

        :returns: The estimated token count of the rendered turn.
        """
        divisor: int = 2 if self.is_code else 4
        return max(1, -(-len(self.render()) // divisor))
