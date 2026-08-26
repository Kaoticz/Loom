from __future__ import annotations

import asyncio

from services.session_registry import SessionRegistry


async def test_first_session_gets_the_slot() -> None:
    """
    An idle registry grants the slot.
    """
    registry: SessionRegistry = SessionRegistry()

    assert await registry.acquire('first') is True
    assert registry.active_session_id == 'first'
    assert registry.is_busy is True


async def test_second_session_is_refused() -> None:
    """
    A second session cannot take a slot that is already held.

    Two sessions on one repository both branch and commit to it, so the second
    would review work it never did.
    """
    registry: SessionRegistry = SessionRegistry()
    await registry.acquire('first')

    assert await registry.acquire('second') is False
    assert registry.active_session_id == 'first'


async def test_releasing_frees_the_slot() -> None:
    """
    The slot becomes available once its holder releases it.
    """
    registry: SessionRegistry = SessionRegistry()
    await registry.acquire('first')

    assert await registry.release('first') is True
    assert registry.is_busy is False
    assert await registry.acquire('second') is True


async def test_a_refused_session_cannot_release_the_slot() -> None:
    """
    A refused connection must not free the running session's claim.

    The refused connection runs the same cleanup path on its way out, so
    releasing indiscriminately would evict the session that is actually working.
    """
    registry: SessionRegistry = SessionRegistry()
    await registry.acquire('first')

    assert await registry.release('second') is False
    assert registry.active_session_id == 'first'


async def test_reacquiring_is_idempotent() -> None:
    """
    The holder may claim its own slot again without being refused.
    """
    registry: SessionRegistry = SessionRegistry()
    await registry.acquire('first')

    assert await registry.acquire('first') is True


async def test_holding_the_slot_is_not_working_on_a_task() -> None:
    """
    A connected session with no prompt yet is busy but not working.

    The distinction is the whole point of the second flag: the slot is taken as
    the socket connects, which is far too early to mean an agent is doing
    anything.
    """
    registry: SessionRegistry = SessionRegistry()
    await registry.acquire('first')

    assert registry.is_busy is True
    assert registry.has_active_task is False


async def test_beginning_and_ending_a_task_tracks_the_work() -> None:
    """
    The working flag follows the task, not the connection.
    """
    registry: SessionRegistry = SessionRegistry()
    await registry.acquire('first')

    assert await registry.begin_task('first') is True
    assert registry.has_active_task is True
    assert registry.working_session_id == 'first'

    assert await registry.end_task('first') is True
    assert registry.has_active_task is False
    # The slot outlives the task: the socket is still open.
    assert registry.is_busy is True


async def test_a_session_without_the_slot_cannot_begin_a_task() -> None:
    """
    A refused connection has no work to report.

    It runs the same paths on its way out, so accepting its claim would block
    the repository on behalf of a session that is already closing.
    """
    registry: SessionRegistry = SessionRegistry()
    await registry.acquire('first')

    assert await registry.begin_task('second') is False
    assert registry.has_active_task is False


async def test_only_the_working_session_can_end_the_task() -> None:
    """
    One session must not be able to clear another's claim to be working.
    """
    registry: SessionRegistry = SessionRegistry()
    await registry.acquire('first')
    await registry.begin_task('first')

    assert await registry.end_task('second') is False
    assert registry.has_active_task is True


async def test_releasing_the_slot_also_ends_an_unfinished_task() -> None:
    """
    A session torn down mid-task must not leave the repository looking busy.

    Nothing reaches a terminal state when a connection drops, so without this
    the working flag would outlive the session that set it.
    """
    registry: SessionRegistry = SessionRegistry()
    await registry.acquire('first')
    await registry.begin_task('first')

    assert await registry.release('first') is True
    assert registry.has_active_task is False
    assert registry.is_busy is False


async def test_a_refused_session_cannot_end_the_running_task() -> None:
    """
    The release guard covers the working flag as well as the slot.
    """
    registry: SessionRegistry = SessionRegistry()
    await registry.acquire('first')
    await registry.begin_task('first')

    assert await registry.release('second') is False
    assert registry.has_active_task is True
    assert registry.working_session_id == 'first'


async def test_only_one_of_many_simultaneous_claims_wins() -> None:
    """
    Concurrent claims resolve to exactly one holder.
    """
    registry: SessionRegistry = SessionRegistry()

    results: list[bool] = list(
        await asyncio.gather(*(registry.acquire(f'session-{index}') for index in range(20)))
    )

    assert results.count(True) == 1
    assert registry.is_busy is True
