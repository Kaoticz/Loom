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
