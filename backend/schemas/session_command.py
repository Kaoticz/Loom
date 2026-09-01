from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from .command_action import CommandAction


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
