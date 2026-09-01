from typing import Annotated

from pydantic import BaseModel, Field


class CodeSearchOutcome(BaseModel):
    """
    What a merged search produced, kept apart for its two different audiences.

    The agent needs the code, and the reader needs to know which files it came
    from. Carrying both means the display never has to pick a rendered blob back
    apart to find the paths, which would tie it to the exact rendering format.
    """

    rendered: Annotated[str, Field(description='The full text returned to the agent.')]
    file_paths: Annotated[list[str], Field(description='Matched files, unique, in order.')]
    notices: Annotated[list[str], Field(description='Warnings about the search itself.')]
