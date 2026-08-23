from __future__ import annotations

from agents.coder import CODER_SYSTEM_PROMPT, coder_agent
from agents.classifier import CLASSIFIER_SYSTEM_PROMPT, classifier_agent
from agents.reviewer import REVIEWER_SYSTEM_PROMPT, ReviewResult, reviewer_agent


def test_coder_prompt_covers_the_required_rules() -> None:
    """
    The coder prompt states each behaviour the execution loop depends on.
    """
    for fragment in ('PLAN_COMPLETE', 'complete_task', 'run_test_command', 'read_file'):
        assert fragment in CODER_SYSTEM_PROMPT


def test_coder_prompt_describes_multi_keyword_lexical_search() -> None:
    """
    The prompt must offer the capability the tool provides.

    There is history here: a previous requirement shipped entirely unreachable
    because the prompt forbade what the tool allowed, and every test passed
    because the tests called the tool directly. The tool's own description and
    this prompt are the only things the model ever sees, so both are asserted.
    """
    understanding: str = unwrapped(
        CODER_SYSTEM_PROMPT.split('# 2. Understand before you change')[1].split('# 3.')[0]
    )

    # That several keywords may go in one call, which is the whole capability.
    assert 'list of keywords' in understanding
    assert 'single call' in understanding

    # That the modes exist, and which one is the default.
    assert 'any of them' in understanding
    assert 'mode="all"' in understanding

    # That `all` disappoints often, so the agent is not left assuming it failed.
    assert 'different lines' in understanding


def test_coder_lexical_tool_description_matches_the_prompt() -> None:
    """
    The tool's own description carries the same offer as the prompt.

    The description is what reaches the model alongside the schema; a capability
    documented only in the prompt is easy to miss, and one documented only here
    can be contradicted by the prompt.
    """
    description: str = unwrapped(
        coder_agent._function_toolset.tools['search_code_lexical'].function.__doc__ or ''
    )

    assert 'as many keywords as you like in one call' in description
    assert '"any" returns lines containing at least one' in description
    assert '"all" returns only lines containing every keyword' in description
    # The default has to be stated, or the agent cannot know what it is getting.
    assert 'defaults to "any"' in description
    # And that an entry with spaces is a phrase rather than several keywords.
    assert 'single entry to search for a phrase' in description


def unwrapped(text: str) -> str:
    """
    Collapse runs of whitespace, so an assertion is about wording not wrapping.

    :param text: The prompt fragment to normalise.
    :returns: The same text with every run of whitespace reduced to one space.
    """
    return ' '.join(text.split())


def test_the_coder_prompt_permits_finishing_without_file_changes() -> None:
    """
    The instructions must not forbid what the tool now allows.

    Requirement 2.4 lets a summarising or explaining task finish without touching
    a file, but the agent only ever calls `complete_task` when the prompt says it
    may. An instruction that forbids it outright would leave the whole mechanism
    unreachable in a real session, and no test that invokes the tool directly
    would notice.
    """
    finishing: str = CODER_SYSTEM_PROMPT.split('# 5. Finishing')[1].split('# 6.')[0]

    # The unconditional prohibition this replaced.
    assert 'You must not call `complete_task` before you have made at least one file' in finishing
    assert 'Most tasks require changing files' in finishing, (
        'the modification rule must be stated as applying to those tasks, not to all of them'
    )
    assert 'require changing nothing' in finishing
    assert 'answer itself as the summary' in finishing


def test_the_coder_prompt_guards_the_changeless_path_against_misuse() -> None:
    """
    Two ways the changeless path could be abused are named explicitly.

    An agent told it may finish without changing anything has an easy escape from
    hard work, and an agent told it must change something has a reason to invent
    a pointless edit.
    """
    assert 'Do not invent a file change' in CODER_SYSTEM_PROMPT
    assert 'escape work you found difficult' in CODER_SYSTEM_PROMPT


def test_the_coder_prompt_says_the_answering_decision_is_already_made() -> None:
    """
    Requirement 2.1 moved the decision ahead of the work, and the prompt has to
    say so.

    An agent left to judge for itself whether this is an answering task would be
    making a decision that has already been made, and could contradict it. This
    is the same failure requirement 10 shipped with — a prompt that disagreed
    with the tool — which no test invoking the tool directly would catch.
    """
    finishing: str = CODER_SYSTEM_PROMPT.split('# 5. Finishing')[1].split('# 6.')[0]

    # Wrapping is not the contract, so it is normalised away: a reflowed
    # paragraph should not fail a test about what the prompt says.
    assert 'already been decided before you were given it' in unwrapped(finishing)
    assert 'you will be told plainly when it is' in unwrapped(finishing)


def test_the_coder_prompt_tells_an_answering_task_how_to_escalate() -> None:
    """
    Requirement 2.7: an answering task that turns out to need a change must not
    make one.

    The session escalates on a modification, so an agent that quietly edited a
    file would get the right outcome by the wrong route, having already changed
    the repository before anyone approved a plan.
    """
    finishing: str = CODER_SYSTEM_PROMPT.split('# 5. Finishing')[1].split('# 6.')[0]

    assert 'cannot be answered without changing a file, do not change one' in unwrapped(finishing)
    assert 'replanned as ordinary work' in unwrapped(finishing)


def test_the_coder_prompt_still_refuses_a_changeless_finish_elsewhere() -> None:
    """
    Requirement 2.10, as the agent is told it: on any other task, finishing
    without a modification is refused outright rather than judged.
    """
    finishing: str = CODER_SYSTEM_PROMPT.split('# 5. Finishing')[1].split('# 6.')[0]

    assert (
        'not told is an answering task, completing without a modification is refused'
        in unwrapped(finishing)
    )


def test_the_coder_prompt_describes_the_question_tool() -> None:
    """
    Requirement 4: a tool the prompt does not describe will not be called.

    The same failure requirement 10 shipped with, and phase 2 had to guard
    against again: the tool existed and the instructions never mentioned it, so
    nothing in a real session would ever have reached it.
    """
    asking: str = CODER_SYSTEM_PROMPT.split('# 4b. Asking the user')[1].split('# 5.')[0]

    assert '`ask_user`' in asking
    assert 'up to six' in unwrapped(asking)
    # The free-reply option is the interface's, and the prompt must not suggest
    # the agent should supply one of its own.
    assert 'neither need to add one nor can remove it' in unwrapped(asking)
    # When not to ask matters as much: every question stops the task.
    assert 'Do not ask permission for work you were already asked to do' in unwrapped(asking)


def test_the_classifier_prompt_states_its_question() -> None:
    """
    The classifier decides one thing, and its prompt should say which.
    """
    assert 'changes_are_required' in CLASSIFIER_SYSTEM_PROMPT
    assert 'reason' in CLASSIFIER_SYSTEM_PROMPT


def test_the_classifier_judges_the_request_rather_than_any_work() -> None:
    """
    Requirement 2.2: it is asked before the task starts, so there is no account
    of what an agent did, and the old wording about judging such an account
    would invite it to weigh something it is never shown.
    """
    assert 'No work has been done yet' in CLASSIFIER_SYSTEM_PROMPT
    assert "agent's account" not in CLASSIFIER_SYSTEM_PROMPT
    assert 'summary' not in CLASSIFIER_SYSTEM_PROMPT


def test_the_classifier_prompt_covers_implied_changes() -> None:
    """
    Requirement 2.2 names this as the risk the new question carries: a request
    that asks for a change without saying so.
    """
    assert 'make the tests' in CLASSIFIER_SYSTEM_PROMPT.lower()


def test_the_classifier_defers_model_selection() -> None:
    """
    Like the others, it must not bind a model at import time.
    """
    assert classifier_agent.model is None


def test_agents_defer_model_selection() -> None:
    """
    Neither agent binds a model at import time.

    A model string is resolved eagerly by PydanticAI, so a default would make
    importing the module fail whenever the provider is unconfigured.
    """
    assert coder_agent.model is None
    assert reviewer_agent.model is None


def test_reviewer_returns_structured_output() -> None:
    """
    The reviewer produces a structured verdict rather than free text.
    """
    assert reviewer_agent.output_type is ReviewResult
    assert set(ReviewResult.model_fields) == {'approved', 'comments'}


def test_reviewer_prompt_sets_its_calibration() -> None:
    """
    The reviewer prompt covers correctness, completeness and regressions.
    """
    lowered: str = REVIEWER_SYSTEM_PROMPT.lower()
    for fragment in ('correct', 'complete', 'regress'):
        assert fragment in lowered


def test_the_coder_prompt_permits_finishing_without_a_sandbox() -> None:
    """
    The instructions must not demand a verification that cannot be run.

    Requirement 11.1: without a sandbox the agent can neither run a command nor
    bring one up, so an unconditional demand to verify leaves it looping until
    the iteration ceiling. The gate waives the requirement; the prompt has to say
    so too, or the agent will never reach the gate.
    """
    finishing: str = CODER_SYSTEM_PROMPT.split('# 5. Finishing')[1].split('# 6.')[0]

    assert 'no sandbox' in finishing
    assert 'Do not keep retrying' in finishing
    assert 'unverified' in finishing
