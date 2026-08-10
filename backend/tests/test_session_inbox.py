from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

import pytest

from schemas.agent_commands import CommandAction
from services.session_inbox import SessionInbox
from services.user_questions import ClientGone, StopRequested


def pipe(fail_after: int | None = None) -> tuple[SessionInbox, Callable[[str], Awaitable[None]]]:
    """
    Build an inbox over a channel the test can push messages into.

    Unlike `scripted`, nothing is queued up front, so a test can decide what to
    send after seeing what the inbox is waiting for.

    :param fail_after: Raise on this many receives, standing in for the client
        going away. None means the channel stays open.
    :returns: The inbox and a function that sends one message to it.
    """
    channel: asyncio.Queue[str] = asyncio.Queue()
    received: list[int] = [0]

    async def receive() -> str:
        if fail_after is not None and received[0] >= fail_after:
            raise ConnectionError('the client went away')
        received[0] += 1
        return await channel.get()

    async def send(message: str) -> None:
        await channel.put(message)
        # Give the reader a turn, so the message is routed before the test
        # asserts on what routing did.
        await asyncio.sleep(0)

    return SessionInbox(receive), send


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


async def test_a_reply_reaches_the_tool_waiting_for_it() -> None:
    """
    Requirement 4.1: the answer goes to whoever asked the question.
    """
    inbox, send = pipe()
    inbox.start()

    waiting: asyncio.Task[str] = asyncio.create_task(inbox.await_reply())
    await asyncio.sleep(0)
    await send(json.dumps({'action': 'reply', 'reply': 'The second one.'}))

    assert await asyncio.wait_for(waiting, timeout=1) == 'The second one.'
    await inbox.close()


async def test_a_reply_is_never_queued_as_a_session_command() -> None:
    """
    The session reads its queue for answers to its own questions.

    A reply landing there would be resolved as a rejection, since it is not an
    approval, and the agent would be told the user rejected something they were
    never asked about.
    """
    inbox, send = pipe()
    inbox.start()

    await send(json.dumps({'action': 'reply', 'reply': 'A stray answer.'}))
    await asyncio.sleep(0.05)
    await send(json.dumps({'approved': True}))

    command = await asyncio.wait_for(inbox.next_command(), timeout=1)
    assert command.is_approval is True
    await inbox.close()


async def test_an_approval_is_never_handed_to_a_waiting_tool() -> None:
    """
    The diversion runs both ways: only a reply answers a question.
    """
    inbox, send = pipe()
    inbox.start()

    waiting: asyncio.Task[str] = asyncio.create_task(inbox.await_reply())
    await asyncio.sleep(0)
    await send(json.dumps({'approved': True}))
    await asyncio.sleep(0.05)

    assert waiting.done() is False
    waiting.cancel()
    await inbox.close()


async def test_a_stop_releases_a_tool_awaiting_a_reply() -> None:
    """
    Requirement 4.10: stopping must not leave the agent waiting for an answer
    the user has plainly stopped intending to give.
    """
    inbox, send = pipe()
    inbox.start()

    waiting: asyncio.Task[str] = asyncio.create_task(inbox.await_reply())
    await asyncio.sleep(0)
    await send(json.dumps({'action': 'stop'}))

    with pytest.raises(StopRequested):
        await asyncio.wait_for(waiting, timeout=1)
    await inbox.close()


async def test_a_stop_already_received_is_not_waited_out() -> None:
    """
    A question asked after a stop has arrived must not block.
    """
    inbox, send = pipe()
    inbox.start()
    await send(json.dumps({'action': 'stop'}))
    await asyncio.sleep(0.05)

    with pytest.raises(StopRequested):
        await asyncio.wait_for(inbox.await_reply(), timeout=1)
    await inbox.close()


async def test_losing_the_client_releases_a_tool_awaiting_a_reply() -> None:
    """
    Requirement 4.11: a disconnection must not leave the session waiting for
    someone who has closed the tab.
    """
    inbox, _ = pipe(fail_after=0)
    inbox.start()

    with pytest.raises(ClientGone):
        await asyncio.wait_for(inbox.await_reply(), timeout=1)
    await inbox.close()


async def test_two_questions_cannot_be_outstanding_at_once() -> None:
    """
    A second waiter would strand the first, which would never be answered.
    """
    inbox, _ = pipe()
    inbox.start()

    first: asyncio.Task[str] = asyncio.create_task(inbox.await_reply())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError):
        await inbox.await_reply()

    first.cancel()
    await inbox.close()
