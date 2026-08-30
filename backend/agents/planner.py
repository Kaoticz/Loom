from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from agents.coder import PLANNING_TOOLS, AgentDeps

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

Each step is given to a coder with no memory of earlier steps. Make every step
self-contained: include the files, symbols, current behavior, decisions, and
acceptance checks it needs. Do not create a discovery-only step whose findings
the next step needs, such as "Find the toolbar". Investigate before producing
the plan, then either combine the discovery with the change it enables or put
the discovered location and relevant facts in that change step's description.
The coder may verify those facts, but must not need to rediscover an earlier
step's outcome to carry out its own work.

Cover the whole task, and stop there. Do not invent work that was not asked for,
and do not include steps for reviewing or committing: that happens anyway.

# Looking at the repository first

You have tools for reading the codebase: `list_files` to see how it is laid out,
`search_code` and `search_code_lexical` to find where something lives,
`get_symbol` to look up a definition, `read_file` to read one, and `web_search`
for anything outside the repository.

Use them before writing the plan. A plan written without looking names files
that do not exist and steps that are already done. You cannot change anything —
none of these tools writes — so there is no cost to looking beyond the time it
takes.

Stop looking once you know enough to write concrete steps. You are not reviewing
the codebase; you are finding out enough to describe the work.

# Asking the user

When an ambiguity would change what steps you produce — which of several things
they meant, which of two reasonable approaches they want, a decision the
repository does not settle — call `ask_user`. It puts the question to the user
and waits, and returns what they said.

Suggest up to six answers when you can see what the likely ones are. The
interface always offers the user a further option of typing their own reply, so
you neither need to add one nor can remove it, and a question with no suggested
answers at all is perfectly usable.

Ask only when the answer changes the plan. Do not ask permission for work you
were already asked to do, do not ask what you could find out by reading the
repository, and do not ask what you could settle by making a reasonable
assumption and saying so in the step that depends on it. Every question stops
the task until someone comes back to it.
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


planner_agent: Agent[AgentDeps, StructuredPlan] = Agent(
    deps_type=AgentDeps,
    output_type=StructuredPlan,
    system_prompt=PLANNER_SYSTEM_PROMPT,
    retries=5,
)
"""
The planner.

Separate from the coder for two reasons. Its prompt is about planning rather
than about executing, and it carries none of the coder's *writing* tools.

It does carry the read-only ones. A plan written without looking at the
repository is a guess about a codebase the planner has never seen, and naming
files that do not exist is the usual result. Being unable to write is structural
rather than instructed: `create_file`, `edit_file`, `run_test_command` and
`commit_changes` were never registered, so there is no switch to leave in the
wrong position — the same reasoning `answering_agent` is built on.

Follows the shape already used by `reviewer_agent` and `classifier_agent`: a
narrow agent, its own prompt, a structured output type.
"""


for _planning_tool in PLANNING_TOOLS:
    planner_agent.tool(_planning_tool)
