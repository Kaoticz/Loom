from enum import Enum


class MessageType(str, Enum):
    """
    Discriminator for every message the backend can send over the WebSocket.
    """

    thought = 'thought'
    plan = 'plan'
    action = 'action'
    observation = 'observation'
    commit = 'commit'
    diff = 'diff'
    review = 'review'
    done = 'done'
    error = 'error'
    warning = 'warning'
    context_usage = 'context_usage'
    session_busy = 'session_busy'
    question = 'question'
    system = 'system'
    sandbox_unavailable = 'sandbox_unavailable'
    compaction = 'compaction'
