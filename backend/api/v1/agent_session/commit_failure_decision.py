from enum import Enum


COMMIT_FAILURE_COMMITTED: str = 'I have committed the changes myself'
COMMIT_FAILURE_CONTINUE: str = 'Carry on without committing'
COMMIT_FAILURE_STOP: str = 'Stop the session'

COMMIT_FAILURE_OPTIONS: tuple[str, ...] = (
    COMMIT_FAILURE_COMMITTED,
    COMMIT_FAILURE_CONTINUE,
    COMMIT_FAILURE_STOP,
)
"""
What the user may do about commits that keep failing.

Committing by hand is offered first because it is usually the quickest fix and
Loom has no business guessing at someone's git configuration. The claim is
verified rather than trusted, so choosing it wrongly costs another question
rather than a lost step.
"""


class CommitFailureDecision(str, Enum):
    """
    The explicit decision made after repeated commit failures.
    """

    committed = 'committed'
    continue_ = 'continue'
    stop = 'stop'


def classify_commit_failure_reply(reply: str) -> CommitFailureDecision:
    """
    Classify a fixed-option or free-text commit-failure reply.

    The questioner permits typed replies, so preserve the existing broad claim
    interpretation in one boundary rather than scattering it through session flow.

    :param reply: The user's selected or typed reply.
    :returns: The action represented by the reply.
    """
    answer: str = reply.strip().lower()
    if answer == COMMIT_FAILURE_STOP.lower() or answer == 'stop':
        return CommitFailureDecision.stop
    if answer == COMMIT_FAILURE_COMMITTED.lower() or (
        'commit' in answer and 'without' not in answer and 'not ' not in answer
    ):
        return CommitFailureDecision.committed
    return CommitFailureDecision.continue_
