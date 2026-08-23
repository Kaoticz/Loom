from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field

PLAN_TERMINATOR: str = 'PLAN_COMPLETE'
NUMBERED_STEP: re.Pattern[str] = re.compile(r'^\s*(\d+)\s*[.)]\s+(?P<body>\S.*)$')
NO_STEPS_CORRECTION: str = (
    'Your plan contained no actionable steps. Please provide a numbered list, '
    'one concrete action per line, ending with a line containing only PLAN_COMPLETE.'
)


class PlanFormat(str, Enum):
    """
    How a plan was produced, and therefore how it should be shown.

    Carried explicitly rather than inferred from whether descriptions happen to
    be present. A structured plan whose descriptions all came back empty is
    still a structured plan, and inferring the format would render it as plain
    text — a disagreement between what was produced and what is shown, which is
    the failure mode that keeps appearing in this project.
    """

    structured = 'structured'
    text = 'text'


class PlanStep(BaseModel):
    """
    One step of a plan.

    The description is optional because a model asked for one may return an
    empty string, and because a plan recovered from free text has titles only.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    title: Annotated[str, Field(description='A short name for the step.')]
    description: Annotated[
        str, Field(default='', description='What the step involves, if the model gave one.')
    ] = ''


class ParsedPlan(BaseModel):
    """
    The steps recovered from an agent's plan output.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    steps: Annotated[
        list[PlanStep], Field(default_factory=list, description='The actionable steps, in order.')
    ]
    used_fallback: Annotated[bool, Field(description='Whether the output had no numbered steps and was taken as a single step.')] = False
    format: Annotated[
        PlanFormat, Field(description='Whether the plan came back structured or as free text.')
    ] = PlanFormat.text

    @property
    def is_empty(self) -> bool:
        """
        Whether the agent produced nothing actionable.

        :returns: True if no steps were recovered.
        """
        return len(self.steps) == 0

    @property
    def titles(self) -> list[str]:
        """
        The step titles, in order.

        :returns: One title per step.
        """
        return [step.title for step in self.steps]


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
        # Free text carries no descriptions, so each recovered step is a title
        # on its own. The format says so, rather than leaving the reader to
        # infer it from the empty descriptions.
        return ParsedPlan(
            steps=[PlanStep(title=step) for step in steps],
            used_fallback=False,
            format=PlanFormat.text,
        )

    remainder: str = ' '.join(line.strip() for line in body.splitlines() if line.strip())
    if remainder:
        return ParsedPlan(
            steps=[PlanStep(title=remainder)], used_fallback=True, format=PlanFormat.text
        )

    return ParsedPlan(steps=[], used_fallback=False, format=PlanFormat.text)


def render_plan(plan: ParsedPlan) -> str:
    """
    Render a plan back into numbered text, for inclusion in a later prompt.

    Descriptions are included where they exist. They are the reason for asking
    for structure at all: a plan whose descriptions never reach the prompt that
    executes it has gained nothing but a nicer display.

    :param plan: The plan to render.
    :returns: The rendered plan, or an empty string if it has no steps.
    """
    lines: list[str] = []

    for index, step in enumerate(plan.steps, start=1):
        lines.append(f'{index}. {step.title}')
        if step.description:
            lines.append(f'   {step.description}')

    return '\n'.join(lines)
