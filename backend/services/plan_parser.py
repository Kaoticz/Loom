from __future__ import annotations

import re
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field

PLAN_TERMINATOR: str = 'PLAN_COMPLETE'
NUMBERED_STEP: re.Pattern[str] = re.compile(r'^\s*(\d+)\s*[.)]\s+(?P<body>\S.*)$')
NO_STEPS_CORRECTION: str = (
    'Your plan contained no actionable steps. Please provide a numbered list, '
    'one concrete action per line, ending with a line containing only PLAN_COMPLETE.'
)


class ParsedPlan(BaseModel):
    """
    The steps recovered from an agent's plan output.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    steps: Annotated[list[str], Field(default_factory=list, description='The actionable steps, in order.')]
    used_fallback: Annotated[bool, Field(description='Whether the output had no numbered steps and was taken as a single step.')] = False

    @property
    def is_empty(self) -> bool:
        """
        Whether the agent produced nothing actionable.

        :returns: True if no steps were recovered.
        """
        return len(self.steps) == 0


def parse_plan(output: str) -> ParsedPlan:
    """
    Extract the numbered steps from an agent's plan.

    Small models rarely honour a response format exactly, so the parser is
    forgiving: preamble before the first step and anything after the terminator
    are discarded, lines that continue a step are folded into it, and output with
    no numbering at all is taken as a single step rather than being rejected.

    :param output: The raw text the agent produced.
    :returns: The recovered plan.
    """
    body: str = output.split(PLAN_TERMINATOR)[0]

    steps: list[str] = []
    current: list[str] = []

    for line in body.splitlines():
        match: re.Match[str] | None = NUMBERED_STEP.match(line)
        if match is not None:
            if current:
                steps.append(' '.join(current).strip())
            current = [match.group('body').strip()]
            continue

        stripped: str = line.strip()
        if current and stripped:
            # A wrapped continuation of the step above.
            current.append(stripped)
        elif current and not stripped:
            steps.append(' '.join(current).strip())
            current = []

    if current:
        steps.append(' '.join(current).strip())

    steps = [step for step in steps if step]
    if steps:
        return ParsedPlan(steps=steps, used_fallback=False)

    remainder: str = ' '.join(line.strip() for line in body.splitlines() if line.strip())
    if remainder:
        return ParsedPlan(steps=[remainder], used_fallback=True)

    return ParsedPlan(steps=[], used_fallback=False)


def render_plan(plan: ParsedPlan) -> str:
    """
    Render a plan back into numbered text, for inclusion in a later prompt.

    :param plan: The plan to render.
    :returns: The rendered plan, or an empty string if it has no steps.
    """
    return '\n'.join(f'{index}. {step}' for index, step in enumerate(plan.steps, start=1))
