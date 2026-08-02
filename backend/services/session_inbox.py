from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from logging import Logger, getLogger

from schemas.agent_commands import CommandAction, SessionCommand, parse_command

logger: Logger = getLogger(__name__)

ReceiveText = Callable[[], Awaitable[str]]


class SessionInbox:
    """
    Owns reading the client's messages for the lifetime of a session.

    Every command except `stop` answers a question the session has just asked,
    and could be read where it is awaited. `stop` cannot: it arrives while the
    agent is working and nothing is waiting to hear from the user. Reading it
    would therefore mean a second coroutine reading the same socket, and two
    readers race for whichever message arrives next, so an approval could be
    swallowed by the stop watcher or the reverse.

    One reader avoids that. It takes every message, diverts a stop into a flag
    the working loops can check, and queues the rest for whoever is waiting.
    """

    _receive: ReceiveText
    _on_stop: Callable[[], Awaitable[None]] | None
    _queue: asyncio.Queue[SessionCommand]
    _stop: asyncio.Event
    _closed: asyncio.Event
    _reader: asyncio.Task[None] | None

    def __init__(
        self, receive: ReceiveText, on_stop: Callable[[], Awaitable[None]] | None = None
    ) -> None:
        """
        Create an inbox over a source of client messages.

        :param receive: Awaitable returning the next raw message from the client.
        :param on_stop: Run the moment a stop arrives, before anything notices the
            flag. This is what lets a command already running in the sandbox be
            killed outright instead of being waited out.
        """
        self._receive = receive
        self._on_stop = on_stop
        self._queue = asyncio.Queue()
        self._stop = asyncio.Event()
        self._closed = asyncio.Event()
        self._reader = None

    @property
    def stop_requested(self) -> bool:
        """
        Whether the user has asked the session to stop.

        :returns: True once a stop has been received.
        """
        return self._stop.is_set()

    def start(self) -> None:
        """
        Begin reading messages.

        Idempotent, so a session that starts twice does not end up with two
        readers competing for the socket.
        """
        if self._reader is None:
            self._reader = asyncio.create_task(self._read())

    async def next_command(self) -> SessionCommand:
        """
        Wait for the user's next instruction.

        A stop already received is reported as a stop command rather than
        leaving the caller waiting for an answer that will never come.

        :returns: The next command the user sent.
        :raises ConnectionError: If the connection closed before one arrived.
        """
        if self._stop.is_set() and self._queue.empty():
            return SessionCommand(action=CommandAction.stop)

        queued: asyncio.Task[SessionCommand] = asyncio.create_task(self._queue.get())
        closed: asyncio.Task[bool] = asyncio.create_task(self._closed.wait())
        stopped: asyncio.Task[bool] = asyncio.create_task(self._stop.wait())
        done, pending = await asyncio.wait(
            (queued, closed, stopped), return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()

        if queued in done:
            return queued.result()
        if self._stop.is_set():
            return SessionCommand(action=CommandAction.stop)
        raise ConnectionError('The client disconnected before answering.')

    async def close(self) -> None:
        """
        Stop reading and release the reader task.
        """
        self._closed.set()
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):
                pass
            self._reader = None

    async def _read(self) -> None:
        """
        Read messages until the connection closes, routing each one.
        """
        while True:
            try:
                text: str = await self._receive()
            except asyncio.CancelledError:
                raise
            except Exception:
                # The client has gone. Anything waiting on a command is released.
                self._closed.set()
                return

            # `parse_command` is total: unusable input becomes a rejection
            # rather than an error, so there is nothing to guard against here.
            command: SessionCommand = parse_command(text)

            if command.resolved_action is CommandAction.stop:
                self._stop.set()
                if self._on_stop is not None:
                    try:
                        await self._on_stop()
                    except Exception as error:
                        # Reading must continue even if the interruption failed;
                        # the flag alone still stops the session at its next check.
                        logger.warning('Could not interrupt work in progress: %s', error)
                continue

            await self._queue.put(command)
