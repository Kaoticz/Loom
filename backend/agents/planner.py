from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import Agent

PLANNER_SYSTEM_PROMPT: str = """You plan software changes. You do not make them.

Given a task and what has happened so far, produce an ordered list of concrete
steps that another agent will carry out.

Each step has two parts:

- A title: a short phrase naming the step, readable at a glance in a list.
  Aim for a handful of words. Do not number it; the order is already known.
- A description: what the step actually involves, and anything the agent
  carrying it out would otherwise have to guess — which file, which function,
  what to check afterwards. Two or three sentences is usually enough.

Write steps that can be acted on. "Improve error handling" is not a step;
"Catch the timeout in run_test_command and report it as a failed run" is.

Cover the whole task, and stop there. Do not invent work that was not asked for,
and do not include steps for reviewing or committing: that happens anyway.
"""


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


class StructuredPlan(BaseModel):
    """
    An ordered list of steps, as the planner is asked to produce it.
    """

    steps: Annotated[
        list[PlannedStep],
        Field(description='The steps to carry out, in the order they should be done.'),
    ]


planner_agent: Agent[None, StructuredPlan] = Agent(
    deps_type=type(None),
    output_type=StructuredPlan,
    system_prompt=PLANNER_SYSTEM_PROMPT,
)
"""
The planner.

Separate from the coder for two reasons. Its prompt is about planning rather
than about executing, and it carries none of the coder's tools — which is why
the planning prompt no longer has to tell the agent not to call any. That
instruction only existed because planning was borrowing an agent built to act.

Follows the shape already used by `reviewer_agent` and `classifier_agent`: a
narrow agent, its own prompt, a structured output type.
"""
