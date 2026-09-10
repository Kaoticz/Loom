from enum import Enum


class AgentEventType(str, Enum):
    """
    Discriminator for every event that can travel through the event bus.
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
    system = 'system'
    sandbox_unavailable = 'sandbox_unavailable'
    question = 'question'
    compaction = 'compaction'
