from typing import Annotated

from pydantic import BaseModel, Field


class SymbolDefinition(BaseModel):
    """
    A symbol's declaration, with the source that defines it.

    The agent asks for a definition in order to read it, so the source is the
    substance of the answer and the location is what makes it actionable.
    """

    symbol_name: Annotated[str, Field(description='The declared name of the symbol.')]
    kind: Annotated[str, Field(description="What the symbol is, for example 'function' or 'class'.")]
    file_path: Annotated[str, Field(description='Repository-relative path of the declaring file.')]
    line_start: Annotated[int, Field(ge=1, description='1-based line where the definition begins.')]
    line_end: Annotated[int, Field(ge=1, description='1-based line where the definition ends.')]
    source: Annotated[str, Field(description='The source text of the definition.')]

    def render(self) -> str:
        """
        Render this definition as a readable block for the agent.

        :returns: The rendered definition.
        """
        header: str = (
            f'{self.kind} {self.symbol_name} '
            f'({self.file_path}:{self.line_start}-{self.line_end})'
        )
        if not self.source:
            return f'{header}\nThe source is no longer at these lines; the file may have changed.'
        return f'{header}\n{self.source}'
