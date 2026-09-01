from typing import Annotated

from pydantic import BaseModel, Field


class LexicalMatch(BaseModel):
    """
    A single line matching a lexical search, with its surrounding context.
    """

    file_path: Annotated[str, Field(description='Repository-relative path of the matching file.')]
    line_number: Annotated[int, Field(ge=1, description='1-based line number of the matching line.')]
    snippet: Annotated[str, Field(description='The matching line with a few lines of surrounding context.')]

    def render(self) -> str:
        """
        Render this match as a readable block for the agent.

        :returns: The rendered match.
        """
        return f'{self.file_path}:{self.line_number}\n{self.snippet}'
