from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from services.commit_info import CommitInfo


class CommitResult(BaseModel):
    """
    What came of trying to commit.

    All three outcomes are distinct and the caller acts differently on each: a
    commit was made, there was nothing to commit, or it failed and the changes
    are still sitting in the working tree where review will not see them.
    """

    commit: Annotated[CommitInfo | None, Field(description='The commit, if one was made.')] = None
    error: Annotated[str | None, Field(description='Why the commit failed, if it did.')] = None

    @property
    def failed(self) -> bool:
        """
        Whether the commit was attempted and did not succeed.

        :returns: True if an error was recorded.
        """
        return self.error is not None
