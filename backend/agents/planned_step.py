from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


class PlannedStep(BaseModel):
    """
    One step of a plan, as the planner is asked to produce it.

    Kept separate from `plan_parser.PlanStep`, which is what the rest of the
    system passes around: this one is a description of what to ask the model
    for, and its field documentation is part of the request.
    """

    title: Annotated[
        str,
        Field(description='A short phrase naming the step, readable at a glance. Not numbered.'),
    ]
    description: Annotated[
        str,
        Field(
            description=(
                'What the step involves: which file or function, and what to check afterwards. '
                'Two or three sentences.'
            )
        ),
    ]
