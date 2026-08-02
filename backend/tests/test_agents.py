from __future__ import annotations

from agents.coder import CODER_SYSTEM_PROMPT, coder_agent
from agents.adjudicator import ADJUDICATOR_SYSTEM_PROMPT, adjudicator_agent
from agents.reviewer import REVIEWER_SYSTEM_PROMPT, ReviewResult, reviewer_agent


def test_coder_prompt_covers_the_required_rules() -> None:
    """
    The coder prompt states each behaviour the execution loop depends on.
    """
    for fragment in ('PLAN_COMPLETE', 'complete_task', 'run_test_command', 'read_file'):
        assert fragment in CODER_SYSTEM_PROMPT


def test_the_coder_prompt_permits_finishing_without_file_changes() -> None:
    """
    The instructions must not forbid what the tool now allows.

    Requirement 10 lets a summarising or explaining task finish without touching
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


def test_the_adjudicator_prompt_states_its_question() -> None:
    """
    The adjudicator decides one thing, and its prompt should say which.
    """
    assert 'changes_were_required' in ADJUDICATOR_SYSTEM_PROMPT
    assert 'reason' in ADJUDICATOR_SYSTEM_PROMPT


def test_the_adjudicator_defers_model_selection() -> None:
    """
    Like the others, it must not bind a model at import time.
    """
    assert adjudicator_agent.model is None


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
