from __future__ import annotations

import pytest

from services.session_fsm import (
    InvalidStateTransitionError,
    SessionState,
    SessionStateMachine,
    is_terminal,
    is_valid_transition,
)

HAPPY_PATH: list[SessionState] = [
    SessionState.planning,
    SessionState.waiting_plan_approval,
    SessionState.executing,
    SessionState.reviewing,
    SessionState.waiting_review_approval,
    SessionState.completed,
    SessionState.merging,
    SessionState.merged,
]


def test_starts_in_initializing() -> None:
    """
    A new machine begins in the initializing state.
    """
    assert SessionStateMachine().current is SessionState.initializing


def test_full_happy_path_reaches_merged() -> None:
    """
    The approved path runs from initializing through to merged.
    """
    machine: SessionStateMachine = SessionStateMachine()
    for target in HAPPY_PATH:
        machine.transition_to(target)

    assert machine.current is SessionState.merged
    assert machine.is_terminal is True
    assert len(machine.history) == len(HAPPY_PATH) + 1


def test_terminal_states_reject_further_transitions() -> None:
    """
    No transition is legal once the machine has terminated.
    """
    machine: SessionStateMachine = SessionStateMachine(SessionState.merged)
    with pytest.raises(InvalidStateTransitionError):
        machine.transition_to(SessionState.planning)


def test_plan_rejection_returns_to_planning() -> None:
    """
    A rejected plan sends the machine back to planning.
    """
    machine: SessionStateMachine = SessionStateMachine(SessionState.waiting_plan_approval)
    assert machine.transition_to(SessionState.planning) is SessionState.planning


def test_review_rejection_routes_through_re_working() -> None:
    """
    A rejected review re-plans by way of the re-working state.
    """
    machine: SessionStateMachine = SessionStateMachine(SessionState.waiting_review_approval)
    machine.transition_to(SessionState.re_working)
    assert machine.transition_to(SessionState.planning) is SessionState.planning


def test_user_can_reject_after_completion() -> None:
    """
    The user may send an approved task back for more work instead of merging.
    """
    machine: SessionStateMachine = SessionStateMachine(SessionState.completed)
    assert machine.transition_to(SessionState.re_working) is SessionState.re_working


def test_failure_is_reachable_from_working_states() -> None:
    """
    Any non-terminal state can fail.
    """
    for state in SessionState:
        if is_terminal(state):
            continue
        assert is_valid_transition(state, SessionState.failed), state


def test_fail_helper_moves_to_failed() -> None:
    """
    The fail helper moves the machine into the failed state.
    """
    machine: SessionStateMachine = SessionStateMachine(SessionState.executing)
    assert machine.fail() is SessionState.failed
    assert machine.is_terminal is True


def test_illegal_jump_is_rejected() -> None:
    """
    States cannot be skipped.
    """
    machine: SessionStateMachine = SessionStateMachine()
    assert machine.can_transition_to(SessionState.executing) is False
    with pytest.raises(InvalidStateTransitionError):
        machine.transition_to(SessionState.executing)


def test_terminal_set() -> None:
    """
    Only merged and failed end the session loop.
    """
    assert is_terminal(SessionState.merged) is True
    assert is_terminal(SessionState.failed) is True
    assert is_terminal(SessionState.completed) is False


def test_execution_may_complete_without_review() -> None:
    """
    A task adjudicated as needing no file changes has no diff to review.

    Requirement 10.5.
    """
    machine: SessionStateMachine = SessionStateMachine()
    machine.transition_to(SessionState.planning)
    machine.transition_to(SessionState.waiting_plan_approval)
    machine.transition_to(SessionState.executing)

    assert machine.transition_to(SessionState.completed) is SessionState.completed


def test_execution_still_reaches_review() -> None:
    """
    Adding the shortcut must not remove the ordinary route.
    """
    machine: SessionStateMachine = SessionStateMachine()
    machine.transition_to(SessionState.planning)
    machine.transition_to(SessionState.waiting_plan_approval)
    machine.transition_to(SessionState.executing)

    assert machine.transition_to(SessionState.reviewing) is SessionState.reviewing
