from enum import Enum


class TaskStatus(str, Enum):
    """
    Lifecycle status of a task, as reported over the REST API.

    This is a coarse view of the session's finite state machine: the many
    working states collapse into `running`, since a caller polling the API cares
    whether a task is in progress, not which step it is on.

    `answered` is for a task that required no file change: it produced an answer
    rather than a diff, so it has neither been merged nor failed. It is kept
    apart from `completed`, which here means the reviewer approved the work and
    a merge decision is pending — a promise there is no branch to keep.
    """

    initialized = 'initialized'
    running = 'running'
    cancelled = 'cancelled'
    completed = 'completed'
    answered = 'answered'
    merged = 'merged'
    failed = 'failed'
