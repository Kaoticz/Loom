from typing import Annotated

from pydantic import BaseModel, Field


class FileChange(BaseModel):
    """
    A single file's contribution to a cumulative diff.
    """

    path: Annotated[str, Field(description='Repository-relative path of the changed file.')]
    insertions: Annotated[int, Field(ge=0, description='Lines added, or zero for a binary file.')]
    deletions: Annotated[int, Field(ge=0, description='Lines removed, or zero for a binary file.')]

    @property
    def total_changed(self) -> int:
        """
        How many lines the change touches in total.

        :returns: The sum of insertions and deletions.
        """
        return self.insertions + self.deletions
