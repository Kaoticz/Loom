from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import Agent

ADJUDICATOR_SYSTEM_PROMPT: str = """You are deciding one narrow question about a task that was
given to an AI coding agent: did carrying it out require changing any file in the repository?

The agent has reported the task finished without modifying anything. You are given the original
task description and the agent's account of what it did. Judge the task, not the agent's effort.

Tasks that legitimately require no file changes include: summarising or explaining code,
answering a question about the repository, assessing or reviewing something, and investigating
whether a problem exists where the answer turns out to be no.

Tasks that require file changes include: adding or altering behaviour, fixing a defect, writing
or updating tests, renaming or moving code, and updating documentation to say something new.

Two cautions. A task phrased as a question can still call for a change if it asks for one
implicitly — "can you make the tests pass?" is a request to change files. And an agent that
merely failed to do the work will often describe its findings convincingly; ask what the task
asked for, not whether the summary reads well.

Return your decision as structured output:

- `changes_were_required`: true if the task could only have been completed by modifying at least
  one file. False if the task was answerable, or otherwise completable, without doing so.
- `reason`: one or two sentences saying why. If changes were required, name what should have been
  changed, because this is shown to the agent so it can continue working."""


class AdjudicationResult(BaseModel):
    """
    Structured verdict on whether a task needed any file to change.
    """

    changes_were_required: Annotated[bool, Field(description='True if the task could only have been completed by modifying at least one file.')]
    reason: Annotated[str, Field(description='One or two sentences explaining the verdict, naming what should have changed when changes were required.')]


adjudicator_agent: Agent[None, AdjudicationResult] = Agent(
    deps_type=type(None),
    output_type=AdjudicationResult,
    system_prompt=ADJUDICATOR_SYSTEM_PROMPT,
)


def build_adjudication_prompt(task: str, summary: str) -> str:
    """
    Describe the question for the adjudicator.

    :param task: The task as the user originally described it.
    :param summary: The agent's account of what it did.
    :returns: The prompt to send.
    """
    return (
        'Original task description:\n'
        f'{task.strip() or "(the task description was empty)"}\n\n'
        "The agent's summary of what it did:\n"
        f'{summary.strip() or "(the agent gave no summary)"}\n\n'
        'Did completing this task require changing at least one file?'
    )
