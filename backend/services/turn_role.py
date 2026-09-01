from enum import Enum


class TurnRole(str, Enum):
    """
    The origin of a single turn in the accumulated conversation.
    """

    user = 'user'
    agent = 'agent'
    tool_call = 'tool_call'
    tool_result = 'tool_result'
    reviewer = 'reviewer'
    system = 'system'
