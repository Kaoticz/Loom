from __future__ import annotations

import re
from collections.abc import Sequence

DEFAULT_DEBOUNCE_SECONDS: float = 0.2
PARAGRAPH_BREAK: str = '\n\n'
SENTENCE_END: re.Pattern[str] = re.compile(r'(?<=[.!?])\s')


class ThoughtBuffer:
    """
    Accumulates streamed model output into complete thoughts.

    Forwarding every token as its own message floods the client and produces a
    chat feed of single words. Text is instead held until it reaches a natural
    boundary, or until it has been waiting long enough that the user would
    notice the silence.

    Time is supplied by the caller rather than read from a clock, so the
    debounce behaviour can be exercised deterministically.
    """

    _buffer: str
    _debounce_seconds: float
    _first_chunk_at: float | None

    def __init__(self, debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS) -> None:
        """
        Create an empty buffer.

        :param debounce_seconds: How long text may accumulate before it is sent regardless of boundaries.
        :raises ValueError: If the debounce interval is negative.
        """
        if debounce_seconds < 0:
            raise ValueError('The debounce interval cannot be negative.')
        self._buffer = ''
        self._debounce_seconds = debounce_seconds
        self._first_chunk_at = None

    @property
    def pending(self) -> str:
        """
        Text that has been accumulated but not yet released.

        :returns: The buffered text.
        """
        return self._buffer

    @property
    def is_empty(self) -> bool:
        """
        Whether anything is waiting to be sent.

        :returns: True if the buffer holds no text.
        """
        return not self._buffer.strip()

    def add(self, chunk: str, now: float) -> list[str]:
        """
        Add a streamed chunk and return any thoughts that are ready to send.

        A thought is released when the text reaches a paragraph break or the end
        of a sentence, or when the oldest buffered text has been waiting longer
        than the debounce interval.

        :param chunk: The newly streamed text.
        :param now: A monotonic timestamp, in seconds.
        :returns: The complete thoughts to send, in order. Often empty.
        """
        if not chunk:
            return self._release_if_stale(now)

        if self._first_chunk_at is None:
            self._first_chunk_at = now
        self._buffer += chunk

        ready: list[str] = self._split_on_boundaries()
        if ready:
            self._first_chunk_at = now if self._buffer else None
            return ready

        return self._release_if_stale(now)

    def flush(self) -> str | None:
        """
        Release whatever remains, for use when the stream ends.

        :returns: The remaining text, or None if nothing was buffered.
        """
        remaining: str = self._buffer.strip()
        self._buffer = ''
        self._first_chunk_at = None
        return remaining or None

    def _release_if_stale(self, now: float) -> list[str]:
        """
        Release the buffer if it has been waiting past the debounce interval.

        :param now: A monotonic timestamp, in seconds.
        :returns: A single-item list with the buffered text, or an empty list.
        """
        if self._first_chunk_at is None or self.is_empty:
            return []
        if now - self._first_chunk_at < self._debounce_seconds:
            return []

        text: str = self._buffer.strip()
        self._buffer = ''
        self._first_chunk_at = None
        return [text] if text else []

    def _split_on_boundaries(self) -> list[str]:
        """
        Take every complete paragraph and sentence out of the buffer.

        :returns: The complete segments found, leaving any partial tail buffered.
        """
        segments: list[str] = []

        while PARAGRAPH_BREAK in self._buffer:
            head, _, tail = self._buffer.partition(PARAGRAPH_BREAK)
            self._buffer = tail
            if head.strip():
                segments.append(head.strip())

        parts: Sequence[str] = SENTENCE_END.split(self._buffer)
        if len(parts) > 1:
            for part in parts[:-1]:
                if part.strip():
                    segments.append(part.strip())
            self._buffer = parts[-1]

        return segments
