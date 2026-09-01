from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field


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
