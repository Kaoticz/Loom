from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

MAX_SUGGESTED_ANSWERS: int = 6
"""
How many answers the agent may suggest.

Beyond this a list stops being a choice and becomes something to read carefully,
at which point a free reply serves the reader better anyway. The interface adds
its own free-reply option on top, so the reader always sees one more than this.
"""


class ClientGone(Exception):
    """
    Raised when the client has disconnected and the session cannot continue.
    """


class StopRequested(Exception):
    """
    Raised when the user has asked the session to stop.
    """


EmitQuestion = Callable[[str, Sequence[str]], Awaitable[None]]
AwaitReply = Callable[[], Awaitable[str]]


def truncate_suggestions(options: Sequence[str]) -> tuple[list[str], bool]:
    """
    Limit suggested answers to what a reader can reasonably choose between.

    Extra options are dropped rather than the call being refused: failing a tool
    call over a cosmetic problem would cost the agent its turn for nothing,
    whereas truncating leaves the question usable and the free-reply option the
    interface adds is untouched either way. Blank suggestions are dropped too,
    since an empty button is not an answer.

    :param options: The answers the agent suggested.
    :returns: The answers to present, and whether any were dropped.
    """
    kept: list[str] = [option.strip() for option in options if option.strip()]
    if len(kept) <= MAX_SUGGESTED_ANSWERS:
        return kept, False
    return kept[:MAX_SUGGESTED_ANSWERS], True


class UserQuestioner:
    """
    The one thing a tool needs in order to put a question to the user.

    Deliberately not the inbox itself. A tool holding the inbox could read
    whatever arrived next, including an approval meant for the session, and the
    inbox exists precisely because two readers of one stream race each other.
    This exposes asking a question and awaiting its reply, and nothing else.
    """

    _emit_question: EmitQuestion
    _await_reply: AwaitReply

    def __init__(self, emit_question: EmitQuestion, await_reply: AwaitReply) -> None:
        """
        Create a questioner over the two operations it needs.

        :param emit_question: Puts the question and its suggested answers in
            front of the user.
        :param await_reply: Waits for the reply, and does nothing else.
        """
        self._emit_question = emit_question
        self._await_reply = await_reply

    async def ask(self, question: str, options: Sequence[str]) -> tuple[str, bool]:
        """
        Put a question to the user and wait for the answer.

        :param question: What to ask.
        :param options: The answers the agent suggests, before truncation.
        :returns: The user's reply, and whether suggestions had to be dropped.
        :raises StopRequested: If the user stopped the session while waiting.
        :raises ClientGone: If the client disconnected while waiting.
        """
        presented, truncated = truncate_suggestions(options)
        await self._emit_question(question, presented)
        return await self._await_reply(), truncated
