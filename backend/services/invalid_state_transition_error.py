from __future__ import annotations

from services.session_state import SessionState


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
