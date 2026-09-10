from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from .planned_step import PlannedStep


class StructuredPlan(BaseModel):
    """
    An ordered list of steps, as the planner is asked to produce it.
    """

    steps: Annotated[
        list[PlannedStep],
        Field(description='The steps to carry out, in the order they should be done.'),
    ]
