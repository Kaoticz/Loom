from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """
    A contiguous run of lines to be embedded as one unit.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    start_line: Annotated[int, Field(ge=1, description='First line of the chunk.')]
    end_line: Annotated[int, Field(ge=1, description='Last line of the chunk, inclusive.')]

    @property
    def length(self) -> int:
        """
        How many lines the chunk spans.

        :returns: The line count.
        """
        return self.end_line - self.start_line + 1

    def text(self, lines: Sequence[str]) -> str:
        """
        Render the chunk from a file's lines.

        :param lines: The file's lines, without terminators.
        :returns: The chunk's source text.
        """
        return '\n'.join(lines[self.start_line - 1 : self.end_line])

    def halve(self) -> list[Chunk]:
        """
        Split this chunk into two roughly equal halves.

        Used when a model refuses a chunk as too long. Halving rather than
        recomputing a size is deliberate: the number of tokens a chunk costs
        depends on the model's tokenizer, which is not something this side can
        measure, so the only reliable approach is to keep dividing until the
        model accepts what it is given.

        :returns: Two chunks, or this one unchanged if it is a single line.
        """
        if self.length <= 1:
            return [self]

        middle: int = self.start_line + self.length // 2 - 1
        return [
            Chunk(start_line=self.start_line, end_line=middle),
            Chunk(start_line=middle + 1, end_line=self.end_line),
        ]
