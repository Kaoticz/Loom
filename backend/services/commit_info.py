from typing import Annotated

from pydantic import BaseModel, Field


class CommitInfo(BaseModel):
    """
    Details of a commit that was just written.
    """

    commit_hash: Annotated[str, Field(description='The full SHA of the new commit.')]
    message: Annotated[str, Field(description='The commit message.')]
    files_changed: Annotated[list[str], Field(default_factory=list, description='Repository-relative paths touched by the commit.')]
