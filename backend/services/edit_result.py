from typing import Annotated

from pydantic import BaseModel, Field


class EditResult(BaseModel):
    """
    Outcome of a surgical file edit.
    """

    success: Annotated[bool, Field(description='Whether the search block was found and replaced.')]
    match_count: Annotated[int, Field(ge=0, description='How many times the search block occurred in the file. Only the first occurrence is replaced.')]
    message: Annotated[str, Field(description='Human readable description of the outcome, suitable for returning to the agent.')]
