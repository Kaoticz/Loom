from __future__ import annotations

import json
from enum import Enum
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field


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


class SessionCommand(BaseModel):
    """
    An instruction sent by the user over the WebSocket.

    The client currently sends only `{"approved": true}`, so `approved` is
    honoured as a shorthand. The richer `action` and `reason` fields are
    optional, which lets the protocol grow without breaking older clients.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra='ignore')

    approved: Annotated[bool | None, Field(description='Shorthand approval flag sent by the current client.')] = None
    action: Annotated[CommandAction | None, Field(description='An explicit instruction, taking precedence over the flag.')] = None
    reason: Annotated[str, Field(description='Why the user rejected, passed back to the agent as feedback.')] = ''
    reply: Annotated[str, Field(description="The user's answer to a question the agent asked.")] = ''

    @property
    def resolved_action(self) -> CommandAction:
        """
        The action to take, resolving the shorthand flag when needed.

        :returns: The explicit action if given, otherwise approve or reject.
        """
        if self.action is not None:
            return self.action
        return CommandAction.approve if self.approved else CommandAction.reject

    @property
    def is_approval(self) -> bool:
        """
        Whether the user approved whatever they were asked about.

        :returns: True for approve, merge, or auto-approve.
        """
        return self.resolved_action in {
            CommandAction.approve,
            CommandAction.merge,
            CommandAction.auto_approve,
        }


def parse_command(raw: str) -> SessionCommand:
    """
    Interpret a raw WebSocket payload as a user instruction.

    A payload that is not usable JSON is treated as a rejection carrying the
    text as its reason, so a malformed message stalls the session for review
    rather than crashing it or being read as approval.

    :param raw: The text received from the client.
    :returns: The parsed command.
    """
    payload: Any
    try:
        payload = json.loads(raw)
    except ValueError:
        return SessionCommand(approved=False, reason=raw.strip()[:500])

    if not isinstance(payload, dict):
        return SessionCommand(approved=False, reason=str(payload)[:500])

    try:
        return SessionCommand.model_validate(payload)
    except ValueError:
        return SessionCommand(approved=False, reason=str(payload)[:500])
