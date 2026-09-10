from __future__ import annotations

import json
from typing import Any

from .session_command import SessionCommand


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
