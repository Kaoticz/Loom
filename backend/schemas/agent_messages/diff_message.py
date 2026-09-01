from typing import Annotated, Literal

from pydantic import Field

from .base_agent_message import BaseAgentMessage
from .message_type import MessageType


MAX_DIFF_LINES: int = 2000
MAX_DIFF_CHARACTERS: int = 250_000


def truncate_diff_side(
    text: str,
    max_lines: int = MAX_DIFF_LINES,
    max_characters: int = MAX_DIFF_CHARACTERS,
) -> tuple[str, bool]:
    """
    Shorten one side of a diff so an oversized file cannot flood the client.

    Whichever limit is reached first applies. Hand written source comfortably
    fits both; the limits exist for lockfiles, generated code and minified
    bundles, where sending the whole file helps nobody.

    :param text: The full contents of one side of the diff.
    :param max_lines: The greatest number of lines to send.
    :param max_characters: The greatest number of characters to send.
    :returns: The text to send, and whether anything was removed.
    """
    lines: list[str] = text.splitlines(keepends=True)
    kept: list[str] = lines[:max_lines]
    body: str = ''.join(kept)
    truncated: bool = len(kept) < len(lines)

    if len(body) > max_characters:
        body = body[:max_characters]
        truncated = True

    if not truncated:
        return text, False

    return (
        f'{body}\n... (truncated: showing the first {len(body.splitlines())} '
        f'of {len(lines)} lines)',
        True,
    )


class DiffMessage(BaseAgentMessage):
    """
    Both sides of a file modification, for the diff viewer.
    """

    type: Annotated[Literal[MessageType.diff], Field()] = MessageType.diff
    file_path: Annotated[str, Field()]
    original: Annotated[str, Field()]
    modified: Annotated[str, Field()]
    truncated: Annotated[bool, Field(description='Whether either side was shortened because the file is too large to display in full.')] = False
