from __future__ import annotations

from agents.coder import CODER_SYSTEM_PROMPT, coder_agent
from agents.reviewer import REVIEWER_SYSTEM_PROMPT, ReviewResult, reviewer_agent


def test_coder_prompt_covers_the_required_rules() -> None:
    """
    The coder prompt states each behaviour the execution loop depends on.
    """
    for fragment in ('PLAN_COMPLETE', 'complete_task', 'run_test_command', 'read_file'):
        assert fragment in CODER_SYSTEM_PROMPT


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
