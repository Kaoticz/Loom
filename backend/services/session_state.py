from enum import Enum


class SessionState(str, Enum):
    """
    The states an agent session can occupy.

    `completed` means the reviewer approved the work and the session is waiting
    for the user's final merge decision; `merged` is the terminal state reached
    once that merge has succeeded. `cancelled` is distinct from `failed`: the
    user stopped the work, rather than the work breaking.

    `classifying` decides whether the task requires changing any file at all.
    A task that does not takes the `answering` path, which reads what it needs
    and finishes at `answered` — a terminal state, since there is no diff to
    review and no branch to merge. `answered` is deliberately not `completed`:
    in this codebase that status means the reviewer approved and a merge
    decision is pending, which would promise a merge that does not exist.
    """

    initializing = 'initializing'
    classifying = 'classifying'
    planning = 'planning'
    waiting_plan_approval = 'waiting_plan_approval'
    executing = 'executing'
    reviewing = 'reviewing'
    waiting_review_approval = 'waiting_review_approval'
    re_working = 're_working'
    answering = 'answering'
    answered = 'answered'
    completed = 'completed'
    merging = 'merging'
    merged = 'merged'
    failed = 'failed'
    cancelled = 'cancelled'
