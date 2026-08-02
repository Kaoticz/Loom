from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum


class SessionState(str, Enum):
    """
    The states an agent session can occupy.

    `completed` means the reviewer approved the work and the session is waiting
    for the user's final merge decision; `merged` is the terminal state reached
    once that merge has succeeded. `cancelled` is distinct from `failed`: the
    user stopped the work, rather than the work breaking.
    """

    initializing = 'initializing'
    planning = 'planning'
    waiting_plan_approval = 'waiting_plan_approval'
    executing = 'executing'
    reviewing = 'reviewing'
    waiting_review_approval = 'waiting_review_approval'
    re_working = 're_working'
    completed = 'completed'
    merging = 'merging'
    merged = 'merged'
    failed = 'failed'
    cancelled = 'cancelled'


TERMINAL_STATES: frozenset[SessionState] = frozenset(
    {SessionState.merged, SessionState.failed, SessionState.cancelled}
)

TRANSITIONS: Mapping[SessionState, frozenset[SessionState]] = {
    SessionState.initializing: frozenset(
        {SessionState.planning, SessionState.failed, SessionState.cancelled}
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
        {
            SessionState.reviewing,
            # A task adjudicated as needing no file changes has no diff, so there
            # is nothing for review to judge.
            SessionState.completed,
            SessionState.failed,
            SessionState.cancelled,
        }
    ),
    SessionState.reviewing: frozenset(
        {SessionState.waiting_review_approval, SessionState.failed, SessionState.cancelled}
    ),
    SessionState.waiting_review_approval: frozenset(
        {
            SessionState.completed,
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


class InvalidStateTransitionError(RuntimeError):
    """
    Raised when a transition that is not part of the transition table is attempted.
    """

    source: SessionState
    target: SessionState

    def __init__(self, source: SessionState, target: SessionState) -> None:
        """
        Describe an illegal state transition.

        :param source: The state the machine was in.
        :param target: The state that was illegally requested.
        """
        self.source = source
        self.target = target
        super().__init__(f'Cannot transition from {source.value} to {target.value}.')


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
