from enum import Enum


class TaskStatus(str, Enum):
    """
    Lifecycle status of a task, as reported over the REST API.

    This is a coarse view of the session's finite state machine: the many
    working states collapse into `running`, since a caller polling the API cares
    whether a task is in progress, not which step it is on.
    """

    initialized = 'initialized'
    running = 'running'
    completed = 'completed'
    merged = 'merged'
    failed = 'failed'


class TaskLogEntryType(str, Enum):
    """
    The kind of event a task log entry records.
    """

    tool_call = 'tool_call'
    commit = 'commit'
    review = 'review'
    plan = 'plan'
    rejection = 'rejection'
    status_change = 'status_change'
    summary = 'summary'
