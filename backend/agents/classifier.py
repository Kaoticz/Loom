from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import Agent

"""
The subagent that decides whether a task calls for changing any file.

Asked once, after the repository is indexed and before anything is planned, so
that a task which only ever wanted an answer is never put through planning,
approval, review and a merge decision it has no use for.

It judges the task as the user wrote it. It is deliberately not shown any
account of what an agent did: by the time such an account exists the question
has already been settled, and asking again could only contradict it.
"""

CLASSIFIER_SYSTEM_PROMPT: str = """You are deciding one narrow question about a task that is about
to be given to an AI coding agent: to carry it out, will any file in the repository have to
change?

You are shown the task exactly as the user wrote it, and nothing else. No work has been done yet,
so there is nothing to weigh but the request itself.

Tasks that require no file changes include: summarising or explaining code, answering a question
about the repository, assessing or reviewing a design, and investigating whether a problem exists
where finding out is the whole of the request.

Tasks that require file changes include: adding or altering behaviour, fixing a defect, writing or
updating tests, renaming or moving code, and updating documentation to say something new.

The hard cases are requests that imply a change without stating one. "Can you make the tests
pass?" is a request to change files, not a question about whether they currently pass. "Is this
function used anywhere?" is a question. "Is this function used anywhere, and remove it if not" is
a request to change files. When a request asks you to find something out *and act on what you
find*, it requires changes.

When a request is genuinely ambiguous, answer that changes are required. Being wrong that way
costs the user a plan to approve; being wrong the other way ends the task with an answer where
work was wanted.

Return your decision as structured output:

- `changes_are_required`: true if carrying out this task will require modifying at least one file.
  False if it can be completed by reading, reasoning and answering.
- `reason`: one or two sentences saying why. This is shown to the user, so say what it was about
  the request that decided it."""


class ClassificationResult(BaseModel):
    """
    Structured verdict on whether a task will need any file to change.
    """

    changes_are_required: Annotated[bool, Field(description='True if carrying out this task requires modifying at least one file.')]
    reason: Annotated[str, Field(description='One or two sentences explaining the verdict, in terms of what the request asked for.')]


classifier_agent: Agent[None, ClassificationResult] = Agent(
    deps_type=type(None),
    output_type=ClassificationResult,
    system_prompt=CLASSIFIER_SYSTEM_PROMPT,
)


def build_classification_prompt(task: str) -> str:
    """
    Describe the question for the classifier.

    :param task: The task as the user originally described it.
    :returns: The prompt to send.
    """
    return (
        'Task description:\n'
        f'{task.strip() or "(the task description was empty)"}\n\n'
        'Will carrying out this task require changing at least one file?'
    )
