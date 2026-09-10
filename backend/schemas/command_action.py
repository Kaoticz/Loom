from __future__ import annotations

from enum import Enum


class CommandAction(str, Enum):
    """
    What the user has asked the session to do next.

    `stop` is unlike the rest: every other action answers a question the session
    asked, and arrives while it waits. `stop` arrives unprompted, while the agent
    is working and nothing is listening for an answer.

    `reply` answers a question the *agent* asked, not the session, and goes to
    the tool waiting on it rather than into the queue the session reads. Keeping
    it a distinct action is what stops a reply being counted as a rejection, and
    an approval being handed to a tool that asked something else entirely.
    """

    approve = 'approve'
    reject = 'reject'
    merge = 'merge'
    abort = 'abort'
    stop = 'stop'
    reply = 'reply'
    proceed = 'continue'
    auto_approve = 'auto_approve'
