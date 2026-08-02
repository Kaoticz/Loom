from __future__ import annotations

import asyncio

import pytest

from services.indexing_progress import (
    DEFAULT_MIN_INTERVAL_SECONDS,
    MAX_UPDATES_PER_SECOND,
    ProgressReporter,
)


def collector() -> tuple[list[tuple[int, int, str]], object]:
    """
    Build a progress callback that records what it receives.

    :returns: The recorded updates and the callback to pass to a reporter.
    """
    seen: list[tuple[int, int, str]] = []

    async def callback(current: int, total: int, message: str) -> None:
        seen.append((current, total, message))

    return seen, callback


async def test_counts_completions_not_positions() -> None:
    """
    Progress reflects how many files finished, whichever order they finish in.

    Deriving progress from a loop index means a file that finishes early reports
    its own position, so the count jumps around and can announce completion
    while other files are still running.
    """
    seen, callback = collector()
    reporter: ProgressReporter = ProgressReporter(3, callback, min_interval=0.0)  # type: ignore[arg-type]

    # Completing out of order, as concurrent workers do.
    await reporter.file_completed('late.py')
    await reporter.file_completed('early.py')
    await reporter.file_completed('middle.py')

    counts: list[int] = [current for current, _, _ in seen]
    assert counts == [1, 2, 3]
    assert reporter.completed == 3


async def test_concurrent_completions_are_not_lost() -> None:
    """
    Simultaneous completions each increment the counter exactly once.
    """
    seen, callback = collector()
    reporter: ProgressReporter = ProgressReporter(50, callback, min_interval=0.0)  # type: ignore[arg-type]

    await asyncio.gather(
        *(reporter.file_completed(f'file_{index}.py') for index in range(50))
    )

    assert reporter.completed == 50
    assert max(current for current, _, _ in seen) == 50


async def test_skipped_files_still_count() -> None:
    """
    Files that are skipped are counted, so the total is reachable.

    Binary and unreadable files are included in the total, so not counting them
    would leave progress permanently short of completion.
    """
    seen, callback = collector()
    reporter: ProgressReporter = ProgressReporter(2, callback, min_interval=0.0)  # type: ignore[arg-type]

    await reporter.file_completed('binary.png')
    await reporter.file_completed('code.py')

    assert seen[-1][0] == seen[-1][1] == 2


async def test_phase_messages_are_throttled() -> None:
    """
    Frequent phase updates are collapsed rather than flooding the client.
    """
    seen, callback = collector()
    reporter: ProgressReporter = ProgressReporter(100, callback, min_interval=60.0)  # type: ignore[arg-type]

    for index in range(100):
        await reporter.phase(f'Extracting symbols from file_{index}.py...')

    assert len(seen) == 1


async def test_start_and_finish_are_never_throttled() -> None:
    """
    The opening and closing messages always reach the client.
    """
    seen, callback = collector()
    reporter: ProgressReporter = ProgressReporter(1, callback, min_interval=60.0)  # type: ignore[arg-type]

    await reporter.start('Indexing 1 files...')
    await reporter.phase('this one is throttled away')
    await reporter.finish('Indexing complete.')

    messages: list[str] = [message for _, _, message in seen]
    assert messages == ['Indexing 1 files...', 'Indexing complete.']


async def test_final_completion_is_never_throttled() -> None:
    """
    The last file always reports, even when throttling would suppress it.

    Otherwise a run could finish with the client still showing an earlier count.
    """
    seen, callback = collector()
    reporter: ProgressReporter = ProgressReporter(3, callback, min_interval=60.0)  # type: ignore[arg-type]

    for name in ('a.py', 'b.py', 'c.py'):
        await reporter.file_completed(name)

    assert seen[-1][0] == 3
    assert seen[-1][1] == 3


async def test_no_callback_is_harmless() -> None:
    """
    A reporter without a callback still counts, and reports nowhere.
    """
    reporter: ProgressReporter = ProgressReporter(2)

    await reporter.start()
    await reporter.file_completed('a.py')
    await reporter.finish()

    assert reporter.completed == 1


async def test_empty_repository_completes() -> None:
    """
    A repository with no files still reports start and finish.
    """
    seen, callback = collector()
    reporter: ProgressReporter = ProgressReporter(0, callback)  # type: ignore[arg-type]

    await reporter.start()
    await reporter.finish()

    assert len(seen) == 2
    assert all(total == 0 for _, total, _ in seen)


@pytest.mark.parametrize('total,interval', [(-1, 0.25), (1, -0.5)])
def test_invalid_settings_are_rejected(total: int, interval: float) -> None:
    """
    A negative total or throttle interval is refused.
    """
    with pytest.raises(ValueError):
        ProgressReporter(total, None, min_interval=interval)


async def test_default_interval_allows_a_visibly_advancing_counter() -> None:
    """
    The default throttle admits several updates a second, not one every quarter second.

    Requirement 1.5: because progress now updates one message in place rather
    than appending, the old interval was needlessly coarse and made the count
    appear to jump straight from the first file to the total.
    """
    assert DEFAULT_MIN_INTERVAL_SECONDS == 1.0 / MAX_UPDATES_PER_SECOND
    assert MAX_UPDATES_PER_SECOND >= 5.0


async def test_update_volume_stays_bounded_for_many_files() -> None:
    """
    A large repository must not produce one socket message per file.

    Requirement 1.5 and the large-repository edge case: the counter advancing
    visibly must not mean an update per file.
    """
    seen, callback = collector()
    total: int = 5_000
    reporter: ProgressReporter = ProgressReporter(total, callback, min_interval=60.0)  # type: ignore[arg-type]

    for index in range(total):
        await reporter.file_completed(f'file_{index}.py')

    # One admitted immediately, one forced at the end. The rest are throttled.
    assert len(seen) < total
    assert len(seen) <= 3


async def test_counter_reaches_the_total() -> None:
    """
    The last update carries the total, however aggressive the throttle.

    Requirement 1.6: the final progress update is always sent, so the counter
    finishes at the total rather than stopping wherever the throttle last let a
    message through.
    """
    seen, callback = collector()
    reporter: ProgressReporter = ProgressReporter(40, callback, min_interval=60.0)  # type: ignore[arg-type]

    for index in range(40):
        await reporter.file_completed(f'file_{index}.py')

    assert seen[-1][0] == 40
    assert seen[-1][1] == 40
