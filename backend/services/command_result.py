from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


class CommandResult(BaseModel):
    """
    Outcome of a command executed inside the sandbox.
    """

    exit_code: Annotated[int, Field(description='Process exit status. Zero means the command succeeded.')]
    output: Annotated[str, Field(description='Combined stdout and stderr, with any secret values redacted.')]
    timed_out: Annotated[bool, Field(description='Whether the command was killed for exceeding its time limit.')] = False

    @property
    def success(self) -> bool:
        """
        Whether the command completed successfully.

        :returns: True if the command exited with status zero.
        """
        return self.exit_code == 0
