from __future__ import annotations

import pytest

from services.session_limits import (
    DEFAULT_MAX_PLAN_RETRIES,
    DEFAULT_MAX_REVIEW_RETRIES,
    DEFAULT_MAX_TOTAL_ITERATIONS,
    IterationLimits,
)


def test_defaults_match_the_specification() -> None:
    """
    The out-of-the-box ceilings are ten, ten and twenty.
    """
    limits: IterationLimits = IterationLimits()
    assert limits.max_plan_retries == DEFAULT_MAX_PLAN_RETRIES == 10
    assert limits.max_review_retries == DEFAULT_MAX_REVIEW_RETRIES == 10
    assert limits.max_total_iterations == DEFAULT_MAX_TOTAL_ITERATIONS == 20


def test_environment_overrides_are_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Each ceiling can be configured from the environment.
    """
    monkeypatch.setenv('LOOM_MAX_PLAN_RETRIES', '3')
    monkeypatch.setenv('LOOM_MAX_REVIEW_RETRIES', '4')
    monkeypatch.setenv('LOOM_MAX_TOTAL_ITERATIONS', '5')

    limits: IterationLimits = IterationLimits.from_environment()
    assert (limits.max_plan_retries, limits.max_review_retries, limits.max_total_iterations) == (3, 4, 5)


@pytest.mark.parametrize('value', ['0', '-1', 'ten', ''])
def test_unusable_values_fall_back_to_defaults(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """
    A misconfigured ceiling is ignored rather than crashing the session.
    """
    monkeypatch.setenv('LOOM_MAX_PLAN_RETRIES', value)
    assert IterationLimits.from_environment().max_plan_retries == DEFAULT_MAX_PLAN_RETRIES


def test_limits_are_immutable() -> None:
    """
    Limits cannot be raised part way through a session.
    """
    from pydantic import ValidationError

    limits: IterationLimits = IterationLimits()
    with pytest.raises(ValidationError):
        limits.max_total_iterations = 999  # type: ignore[misc]
