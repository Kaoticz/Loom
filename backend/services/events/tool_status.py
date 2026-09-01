from enum import Enum


class ToolStatus(str, Enum):
    """
    Lifecycle status of a tool invocation.
    """

    running = 'running'
    success = 'success'
    failure = 'failure'
