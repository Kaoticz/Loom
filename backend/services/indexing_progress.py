from __future__ import annotations

from asyncio import Lock
from collections.abc import Awaitable, Callable
from time import monotonic

ProgressCallback = Callable[[int, int, str], Awaitable[None]]

MAX_UPDATES_PER_SECOND: float = 10.0
"""
The ceiling on throttled progress updates per second.

Progress now replaces a single message rather than appending one, so the old
quarter-second gap was needlessly coarse: it made the counter appear to jump
from the first file straight to the total. Ten updates a second reads as a
steadily advancing counter while still bounding traffic to a fixed rate, so a
repository of ten files and one of ten thousand cost the same per second.
"""

DEFAULT_MIN_INTERVAL_SECONDS: float = 1.0 / MAX_UPDATES_PER_SECOND


class ProgressReporter:
    """
    Reports indexing progress from concurrently processed files.

    Files are indexed in parallel, so progress cannot be derived from a loop
    index: whichever file finishes first would report its own position, making
    the count jump around and announce completion while other files are still
    running. A shared counter behind a lock reports how many files have actually
    finished.

    Messages are also throttled, because an update per file would send thousands
    of messages for a large repository. Throttling bounds the rate, not the
    destination: the client folds every update into one slot in the feed, and the
    first and last updates are always sent so the counter starts at zero and
    finishes at the total.
    """

    _callback: ProgressCallback | None
    _total: int
    _completed: int
    _lock: Lock
    _min_interval: float
    _last_emit: float | None

    def __init__(
        self,
        total: int,
        callback: ProgressCallback | None = None,
        min_interval: float = DEFAULT_MIN_INTERVAL_SECONDS,
    ) -> None:
        """
        Create a progress reporter.

        :param total: How many files will be processed.
        :param callback: Where to send progress updates, if anywhere.
        :param min_interval: The shortest gap between two throttled messages, in seconds.
        :raises ValueError: If the total is negative or the interval is negative.
        """
        if total < 0:
            raise ValueError('The total number of files cannot be negative.')
        if min_interval < 0:
            raise ValueError('The throttle interval cannot be negative.')

        self._callback = callback
        self._total = total
        self._completed = 0
        self._lock = Lock()
        self._min_interval = min_interval
        self._last_emit = None

    @property
    def total(self) -> int:
        """
        How many files are being processed.

        :returns: The total file count.
        """
        return self._total

    @property
    def completed(self) -> int:
        """
        How many files have finished, including those that were skipped.

        :returns: The completed file count.
        """
        return self._completed

    async def start(self, message: str = 'Starting indexing...') -> None:
        """
        Announce that indexing has begun.

        :param message: The message to send.
        """
        await self._emit(message, force=True)

    async def phase(self, message: str) -> None:
        """
        Report what is happening right now, such as extracting symbols.

        Throttled, so a per-file phase update does not flood the client.

        :param message: The message to send.
        """
        await self._emit(message, force=False)

    async def file_completed(self, path: str) -> int:
        """
        Record that a file has finished, whether it was indexed or skipped.

        Skipped files are counted too, otherwise the total could never be
        reached and progress would appear to stall near the end.

        :param path: The file that finished.
        :returns: How many files have now completed.
        """
        async with self._lock:
            self._completed += 1
            completed: int = self._completed

        # The final message is never throttled, so completion is always shown.
        await self._emit(f'Indexed {path}', force=completed >= self._total)
        return completed

    async def finish(self, message: str = 'Indexing complete.') -> None:
        """
        Announce that indexing has finished.

        :param message: The message to send.
        """
        await self._emit(message, force=True)

    async def _emit(self, message: str, force: bool) -> None:
        """
        Send a progress message, subject to throttling.

        :param message: The message to send.
        :param force: Whether to send regardless of how recently the last one went.
        """
        if self._callback is None:
            return

        now: float = monotonic()
        async with self._lock:
            if not force and self._last_emit is not None:
                if now - self._last_emit < self._min_interval:
                    return
            self._last_emit = now
            completed: int = self._completed

        await self._callback(completed, self._total, message)
