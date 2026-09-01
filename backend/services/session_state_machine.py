from __future__ import annotations

from collections.abc import Mapping, Sequence
from services.invalid_state_transition_error import InvalidStateTransitionError
from services.session_state import SessionState


TERMINAL_STATES: frozenset[SessionState] = frozenset(
    {
        SessionState.merged,
        SessionState.answered,
        SessionState.failed,
        SessionState.cancelled,
    }
)

TRANSITIONS: Mapping[SessionState, frozenset[SessionState]] = {
    SessionState.initializing: frozenset(
        {SessionState.classifying, SessionState.failed, SessionState.cancelled}
    ),
    SessionState.classifying: frozenset(
        {
            SessionState.planning,
            SessionState.answering,
            SessionState.failed,
            SessionState.cancelled,
        }
    ),
    SessionState.planning: frozenset(
        {SessionState.waiting_plan_approval, SessionState.failed, SessionState.cancelled}
    ),
    SessionState.waiting_plan_approval: frozenset(
        {
            SessionState.executing,
            SessionState.planning,
            SessionState.failed,
            SessionState.cancelled,
        }
    ),
    SessionState.executing: frozenset(
        {SessionState.reviewing, SessionState.failed, SessionState.cancelled}
    ),
    SessionState.answering: frozenset(
        {
            SessionState.answered,
            # The classification was wrong and the task does need changes after
            # all, so it rejoins the ordinary route at planning.
            SessionState.planning,
            SessionState.failed,
            SessionState.cancelled,
        }
    ),
    SessionState.answered: frozenset(),
    SessionState.reviewing: frozenset(
        {SessionState.waiting_review_approval, SessionState.failed, SessionState.cancelled}
    ),
    SessionState.waiting_review_approval: frozenset(
        {
            SessionState.completed,
            # The user can deliberately merge despite a rejected reviewer
            # verdict, skipping the ordinary final merge confirmation.
            SessionState.merging,
            SessionState.re_working,
            SessionState.failed,
            SessionState.cancelled,
        }
    ),
    SessionState.re_working: frozenset(
        {SessionState.planning, SessionState.failed, SessionState.cancelled}
    ),
    SessionState.completed: frozenset(
        {
            SessionState.merging,
            SessionState.re_working,
            SessionState.failed,
            SessionState.cancelled,
        }
    ),
    SessionState.merging: frozenset(
        {SessionState.merged, SessionState.failed, SessionState.cancelled}
    ),
    SessionState.merged: frozenset(),
    SessionState.failed: frozenset(),
    SessionState.cancelled: frozenset(),
}


def is_valid_transition(source: SessionState, target: SessionState) -> bool:
    """
    Check whether moving between two states is permitted.

    :param source: The current state.
    :param target: The candidate next state.
    :returns: True if the transition appears in the transition table.
    """
    return target in TRANSITIONS[source]


def is_terminal(state: SessionState) -> bool:
    """
    Check whether a state ends the session loop.

    :param state: The state to inspect.
    :returns: True if no further transitions are possible from this state.
    """
    return state in TERMINAL_STATES


class SessionStateMachine:
    """
    Tracks the current state of an agent session and enforces legal transitions.

    The controller drives this machine, dispatching to a handler per state. The
    recorded history is used for debugging and for persisting progress into
    `TaskMemory`.
    """

    _current: SessionState
    _history: list[SessionState]

    def __init__(self, initial: SessionState = SessionState.initializing) -> None:
        """
        Create a state machine.

        :param initial: The state the machine starts in.
        """
        self._current = initial
        self._history = [initial]

    @property
    def current(self) -> SessionState:
        """
        The state the machine currently occupies.

        :returns: The current state.
        """
        return self._current

    @property
    def history(self) -> Sequence[SessionState]:
        """
        Every state the machine has occupied, in order.

        :returns: An immutable view over the visited states.
        """
        return tuple(self._history)

    @property
    def is_terminal(self) -> bool:
        """
        Whether the machine has reached a state it cannot leave.

        :returns: True if the current state is terminal.
        """
        return is_terminal(self._current)

    def can_transition_to(self, target: SessionState) -> bool:
        """
        Check whether the machine may move to the given state right now.

        :param target: The candidate next state.
        :returns: True if the transition is legal from the current state.
        """
        return is_valid_transition(self._current, target)

    def transition_to(self, target: SessionState) -> SessionState:
        """
        Move the machine to the given state.

        :param target: The state to move to.
        :returns: The new current state.
        :raises InvalidStateTransitionError: If the transition is not permitted.
        """
        if not self.can_transition_to(target):
            raise InvalidStateTransitionError(self._current, target)
        self._current = target
        self._history.append(target)
        return self._current

    def cancel(self) -> SessionState:
        """
        Move the machine to the cancelled state from wherever it currently is.

        Distinct from failing: the user stopped the work rather than the work
        breaking, and history should be able to tell the two apart.

        :returns: The new current state.
        :raises InvalidStateTransitionError: If the machine has already terminated.
        """
        return self.transition_to(SessionState.cancelled)

    def fail(self) -> SessionState:
        """
        Move the machine to the failed state from wherever it currently is.

        :returns: The new current state.
        :raises InvalidStateTransitionError: If the machine has already terminated.
        """
        return self.transition_to(SessionState.failed)
