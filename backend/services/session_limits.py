from __future__ import annotations

import os
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MAX_PLAN_RETRIES: int = 10
DEFAULT_MAX_REVIEW_RETRIES: int = 10
DEFAULT_MAX_TOTAL_ITERATIONS: int = 20


class IterationLimits(BaseModel):
    """
    Ceilings that stop a session looping forever.

    The loop is otherwise unbounded by design, so each cap exists to catch a
    specific runaway: a plan the user keeps rejecting, a reviewer that never
    approves, and the overall plan-execute-review cycle.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    max_plan_retries: Annotated[int, Field(ge=1, description='Rejected plans tolerated before asking the user how to proceed.')] = DEFAULT_MAX_PLAN_RETRIES
    max_review_retries: Annotated[int, Field(ge=1, description='Rejected reviews tolerated before asking the user how to proceed.')] = DEFAULT_MAX_REVIEW_RETRIES
    max_total_iterations: Annotated[int, Field(ge=1, description='Hard ceiling on complete plan-execute-review cycles.')] = DEFAULT_MAX_TOTAL_ITERATIONS

    @classmethod
    def from_environment(cls) -> IterationLimits:
        """
        Read the limits from the environment, falling back to the defaults.

        A value that is not a positive integer is ignored rather than crashing
        the session, since a typo in configuration should not stop the agent
        from running.

        :returns: The configured limits.
        """
        return cls(
            max_plan_retries=_read_positive_int('LOOM_MAX_PLAN_RETRIES', DEFAULT_MAX_PLAN_RETRIES),
            max_review_retries=_read_positive_int('LOOM_MAX_REVIEW_RETRIES', DEFAULT_MAX_REVIEW_RETRIES),
            max_total_iterations=_read_positive_int('LOOM_MAX_TOTAL_ITERATIONS', DEFAULT_MAX_TOTAL_ITERATIONS),
        )


def _read_positive_int(name: str, default: int) -> int:
    """
    Read a positive integer from the environment.

    :param name: The environment variable to read.
    :param default: The value to use when unset or unusable.
    :returns: The configured value, or the default.
    """
    raw: str | None = os.getenv(name)
    if raw is None:
        return default
    try:
        value: int = int(raw)
    except ValueError:
        return default
    return value if value >= 1 else default
