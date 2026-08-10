from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from logging import Logger, getLogger

from schemas.agent_commands import CommandAction, SessionCommand, parse_command
from services.user_questions import ClientGone, StopRequested

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

    A reply to a question the agent asked is diverted too, to the tool awaiting
    it. It must not reach the queue: the session reads that queue for answers to
    its own questions, and a reply landing there would be resolved as a
    rejection. The diversion runs the other way as well — an approval never
    reaches a waiting tool, because only a `reply` is routed to one.
    """

    _receive: ReceiveText
    _on_stop: Callable[[], Awaitable[None]] | None
    _queue: asyncio.Queue[SessionCommand]
    _stop: asyncio.Event
    _closed: asyncio.Event
    _reader: asyncio.Task[None] | None
    _pending_reply: asyncio.Future[str] | None

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
        self._pending_reply = None

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

    async def await_reply(self) -> str:
        """
        Wait for the user's answer to a question the agent asked.

        Waits as long as an approval point does, rather than timing out: the
        agent asked because it cannot proceed without an answer, and inventing
        one after an interval would be worse than waiting.

        :returns: What the user replied.
        :raises StopRequested: If the user stopped the session while waiting.
        :raises ClientGone: If the client disconnected while waiting.
        :raises RuntimeError: If a question is already outstanding.
        """
        if self._pending_reply is not None and not self._pending_reply.done():
            raise RuntimeError('A question is already awaiting a reply.')

        # Checked before waiting as well as while waiting: a stop that arrived
        # a moment ago would otherwise leave the tool waiting for a user who has
        # already walked away.
        if self._stop.is_set():
            raise StopRequested()
        if self._closed.is_set():
            raise ClientGone()

        pending: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._pending_reply = pending

        stopped: asyncio.Task[bool] = asyncio.create_task(self._stop.wait())
        closed: asyncio.Task[bool] = asyncio.create_task(self._closed.wait())
        try:
            done, waiting = await asyncio.wait(
                (pending, stopped, closed), return_when=asyncio.FIRST_COMPLETED
            )
            for task in waiting:
                task.cancel()

            if pending in done:
                return pending.result()
            if self._stop.is_set():
                raise StopRequested()
            raise ClientGone()
        finally:
            self._pending_reply = None

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

            if command.resolved_action is CommandAction.reply:
                self._deliver_reply(command.reply)
                continue

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

    def _deliver_reply(self, reply: str) -> None:
        """
        Hand a reply to the tool waiting for it.

        A reply with nothing waiting is discarded rather than queued. Queueing
        it would leave it to be read as the answer to some later approval, which
        is the confusion this routing exists to prevent.

        :param reply: What the user answered.
        """
        if self._pending_reply is None or self._pending_reply.done():
            logger.warning('A reply arrived with no question outstanding; discarding it.')
            return
        self._pending_reply.set_result(reply)
