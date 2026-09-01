from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from services.plan_format import PlanFormat
from services.plan_step import PlanStep


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
