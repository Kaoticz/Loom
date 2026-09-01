from enum import Enum


class SessionStatus(str, Enum):
    """
    The phase of work a session is announcing it has entered.

    The session already knows this: it is the state machine's own business. It is
    stated outright so the client does not have to infer it by matching the
    wording of announcements, which breaks silently the moment the wording
    changes.

    Only phases the session enters of its own accord appear here. Waiting for a
    plan or a review to be approved is driven by the plan and review messages
    themselves, not announced. `cancelled` is the exception that proves the rule:
    it is announced because nothing else tells the client the agent has stopped,
    and without it the interface would go on offering to stop work that already
    has.
    """

    indexing = 'indexing'
    classifying = 'classifying'
    planning = 'planning'
    executing = 'executing'
    answering = 'answering'
    reviewing = 'reviewing'
    re_working = 're_working'
    merging = 'merging'
    cancelled = 'cancelled'
