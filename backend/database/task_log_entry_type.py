from enum import Enum


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
