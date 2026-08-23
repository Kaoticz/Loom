from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from database.enums import TaskLogEntryType, TaskStatus
from services.plan_parser import PlanFormat
from schemas.agent_commands import CommandAction
from schemas.agent_messages import MessageType
from services.event_bus import SandboxStage, SessionStatus, ToolStatus

"""
The vocabulary the backend and the frontend must agree on.

The frontend validates inbound data against its own Zod schemas, which means
every enum in this protocol is written down twice, in two languages. That is
tolerable only if drift is caught: adding a task status here and forgetting the
frontend would make the frontend reject every task carrying it, surfacing as an
unexplained validation error rather than as a missing case.

This module is the single source of truth. It is exported to a JSON artefact
that the frontend asserts against, so either side drifting fails a test on both.
"""

ARTEFACT_PATH: Path = (
    Path(__file__).resolve().parent.parent / 'frontend' / 'src' / 'lib' / 'protocol.generated.json'
)
"""Where the exported artefact lives, inside the frontend's source tree."""


def _values(enum: type[Enum]) -> list[str]:
    """
    List an enum's values in declaration order.

    :param enum: The enum to read.
    :returns: The value of every member, as strings.
    """
    return [str(member.value) for member in enum]


def protocol_vocabulary() -> dict[str, list[str]]:
    """
    Describe every enum the frontend mirrors.

    :returns: A mapping of vocabulary name to its permitted values.
    """
    return {
        'message_type': _values(MessageType),
        'session_status': _values(SessionStatus),
        'sandbox_stage': _values(SandboxStage),
        'tool_status': _values(ToolStatus),
        'task_status': _values(TaskStatus),
        'task_log_entry_type': _values(TaskLogEntryType),
        'command_action': _values(CommandAction),
        'plan_format': _values(PlanFormat),
    }


def render_artefact() -> str:
    """
    Render the artefact exactly as it should appear on disk.

    :returns: The JSON text, newline terminated so it is a well formed file.
    """
    header: dict[str, object] = {
        '_comment': (
            'Generated from the backend enums by scripts/export_protocol.py. '
            'Do not edit by hand; run the script and commit the result.'
        ),
        **protocol_vocabulary(),
    }
    return json.dumps(header, indent=4) + '\n'


def write_artefact() -> Path:
    """
    Write the artefact to disk.

    :returns: The path written.
    """
    ARTEFACT_PATH.write_text(render_artefact())
    return ARTEFACT_PATH
