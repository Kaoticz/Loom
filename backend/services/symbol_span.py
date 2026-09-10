from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field


class SymbolSpan(BaseModel):
    """
    The line range a symbol occupies, 1-based and inclusive at both ends.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    start_line: Annotated[int, Field(ge=1, description='First line of the definition.')]
    end_line: Annotated[int, Field(ge=1, description='Last line of the definition, inclusive.')]

    @property
    def length(self) -> int:
        """
        How many lines the symbol spans.

        :returns: The line count.
        """
        return self.end_line - self.start_line + 1

    def contains(self, other: SymbolSpan) -> bool:
        """
        Check whether this span fully encloses another, distinct span.

        :param other: The span to test.
        :returns: True if the other span lies inside this one and is not identical.
        """
        if self.start_line == other.start_line and self.end_line == other.end_line:
            return False
        return self.start_line <= other.start_line and other.end_line <= self.end_line
