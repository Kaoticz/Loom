from __future__ import annotations

import asyncio
import json

import pytest

from schemas.agent_commands import CommandAction
from services.session_inbox import SessionInbox


def scripted(messages: list[str]) -> tuple[object, asyncio.Event]:
    """
    Build a receive function that yields the given messages then blocks.

    :param messages: The raw messages to deliver, in order.
    :returns: The receive callable and an event set once they are exhausted.
    """
    pending: list[str] = list(messages)
    drained: asyncio.Event = asyncio.Event()

    async def receive() -> str:
        if pending:
            return pending.pop(0)
        drained.set()
        # Block rather than close, so tests control when the connection ends.
        await asyncio.Event().wait()
        raise AssertionError('unreachable')

    return receive, drained


async def test_a_stop_is_noticed_without_anyone_waiting() -> None:
    """
    Stop arrives while the agent is working, with nothing awaiting a command.

    Requirement 8.3: it must be seen anyway, which is the whole reason a single
    reader owns the socket.
    """
    receive, drained = scripted([json.dumps({'action': 'stop'})])
    inbox: SessionInbox = SessionInbox(receive)  # type: ignore[arg-type]
    inbox.start()

    await asyncio.wait_for(drained.wait(), timeout=1)

    assert inbox.stop_requested is True
    await inbox.close()


async def test_ordinary_commands_are_queued_for_whoever_asks() -> None:
    """
    An approval sent while nothing is waiting is not lost.
    """
    receive, drained = scripted([json.dumps({'approved': True})])
    inbox: SessionInbox = SessionInbox(receive)  # type: ignore[arg-type]
    inbox.start()
    await asyncio.wait_for(drained.wait(), timeout=1)

    command = await asyncio.wait_for(inbox.next_command(), timeout=1)

    assert command.is_approval is True
    await inbox.close()


async def test_a_stop_does_not_swallow_an_earlier_approval() -> None:
    """
    Both are delivered, in order, by the one reader.

    Two coroutines reading the same socket would race for whichever message came
    next, and either could consume the other's.
    """
    receive, drained = scripted([
        json.dumps({'approved': True}),
        json.dumps({'action': 'stop'}),
    ])
    inbox: SessionInbox = SessionInbox(receive)  # type: ignore[arg-type]
    inbox.start()
    await asyncio.wait_for(drained.wait(), timeout=1)

    command = await asyncio.wait_for(inbox.next_command(), timeout=1)

    assert command.is_approval is True
    assert inbox.stop_requested is True
    await inbox.close()


async def test_waiting_for_a_command_is_released_by_a_stop() -> None:
    """
    A stop answers a question the session is waiting on.

    Otherwise the session would sit forever waiting for an approval the user has
    no intention of giving, having just asked to stop.
    """
    receive, _ = scripted([json.dumps({'action': 'stop'})])
    inbox: SessionInbox = SessionInbox(receive)  # type: ignore[arg-type]
    inbox.start()

    command = await asyncio.wait_for(inbox.next_command(), timeout=1)

    assert command.resolved_action is CommandAction.stop
    await inbox.close()


async def test_the_interruption_runs_the_moment_a_stop_arrives() -> None:
    """
    Work in progress is cut short rather than waited out.

    Requirement 8.4: a running sandbox command is killed outright.
    """
    interrupted: asyncio.Event = asyncio.Event()

    async def on_stop() -> None:
        interrupted.set()

    receive, _ = scripted([json.dumps({'action': 'stop'})])
    inbox: SessionInbox = SessionInbox(receive, on_stop=on_stop)  # type: ignore[arg-type]
    inbox.start()

    await asyncio.wait_for(interrupted.wait(), timeout=1)
    await inbox.close()


async def test_a_failing_interruption_still_stops_the_session() -> None:
    """
    If the sandbox cannot be killed, the stop still takes effect.
    """

    async def on_stop() -> None:
        raise RuntimeError('the runtime is gone')

    receive, drained = scripted([json.dumps({'action': 'stop'})])
    inbox: SessionInbox = SessionInbox(receive, on_stop=on_stop)  # type: ignore[arg-type]
    inbox.start()
    await asyncio.wait_for(drained.wait(), timeout=1)

    assert inbox.stop_requested is True
    await inbox.close()


async def test_an_unreadable_message_becomes_a_rejection_and_is_queued() -> None:
    """
    Nonsense from the client stalls for review rather than killing the reader.

    `parse_command` deliberately turns unusable input into a rejection, so the
    inbox queues it in order like any other command rather than discarding it.
    """
    receive, drained = scripted(['{ not json', json.dumps({'approved': True})])
    inbox: SessionInbox = SessionInbox(receive)  # type: ignore[arg-type]
    inbox.start()
    await asyncio.wait_for(drained.wait(), timeout=1)

    first = await asyncio.wait_for(inbox.next_command(), timeout=1)
    second = await asyncio.wait_for(inbox.next_command(), timeout=1)

    assert first.is_approval is False
    assert second.is_approval is True
    await inbox.close()


async def test_a_closed_connection_releases_a_waiting_caller() -> None:
    """
    A disconnect must not leave the session waiting forever.
    """

    async def receive() -> str:
        raise RuntimeError('socket closed')

    inbox: SessionInbox = SessionInbox(receive)
    inbox.start()

    with pytest.raises(ConnectionError):
        await asyncio.wait_for(inbox.next_command(), timeout=1)

    await inbox.close()


async def test_starting_twice_does_not_create_two_readers() -> None:
    """
    Two readers on one socket would race for messages.
    """
    receive, drained = scripted([json.dumps({'approved': True})])
    inbox: SessionInbox = SessionInbox(receive)  # type: ignore[arg-type]
    inbox.start()
    inbox.start()
    await asyncio.wait_for(drained.wait(), timeout=1)

    command = await asyncio.wait_for(inbox.next_command(), timeout=1)
    assert command.is_approval is True

    # Only one message existed, so a second reader would have to invent one.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(inbox.next_command(), timeout=0.1)

    await inbox.close()
