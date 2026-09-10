from __future__ import annotations

from services.file_change import FileChange
from services.review_context import (
    build_review_prompt,
    collect_review_diffs,
    render_change_summary,
)


def test_summary_lists_every_file_with_counts() -> None:
    """
    The change summary names each file and how much it changed.
    """
    summary: str = render_change_summary([
        FileChange(path='a.py', insertions=10, deletions=2),
        FileChange(path='b.py', insertions=0, deletions=7),
    ])

    assert 'a.py (+10/-2)' in summary
    assert 'b.py (+0/-7)' in summary


def test_empty_change_set_is_stated_explicitly() -> None:
    """
    No changes reads as such rather than as an empty section.
    """
    assert render_change_summary([]) == '(no files changed)'


def test_prompt_includes_task_summary_and_diffs() -> None:
    """
    The reviewer receives the task, the file list and the diff bodies.
    """
    changes: list[FileChange] = [FileChange(path='a.py', insertions=1, deletions=0)]
    prompt: str = build_review_prompt('Add a timeout', changes, {'a.py': '+timeout=30\n'})

    assert 'Add a timeout' in prompt
    assert 'Files changed (1)' in prompt
    assert 'a.py (+1/-0)' in prompt
    assert '+timeout=30' in prompt


def test_oversized_diffs_are_omitted_but_named() -> None:
    """
    Diffs that do not fit the budget are named rather than silently dropped.

    A reviewer that cannot see a file must know it exists, or it will approve a
    change it never read.
    """
    changes: list[FileChange] = [
        FileChange(path='small.py', insertions=2, deletions=0),
        FileChange(path='huge.py', insertions=9000, deletions=0),
    ]
    diffs: dict[str, str] = {'small.py': '+two lines\n', 'huge.py': 'x' * 50_000}

    prompt: str = build_review_prompt('task', changes, diffs, budget_characters=2_000)

    assert '+two lines' in prompt
    assert 'huge.py' in prompt
    assert 'too large to include' in prompt
    assert 'x' * 5_000 not in prompt


def test_budget_is_respected() -> None:
    """
    The assembled prompt stays close to the requested budget.
    """
    changes: list[FileChange] = [
        FileChange(path=f'file_{index}.py', insertions=100, deletions=0) for index in range(20)
    ]
    diffs: dict[str, str] = {change.path: 'y' * 5_000 for change in changes}

    prompt: str = build_review_prompt('task', changes, diffs, budget_characters=12_000)
    assert len(prompt) < 14_000


def test_small_files_are_preferred_over_one_huge_file() -> None:
    """
    A single enormous file cannot crowd out every other diff.
    """
    changes: list[FileChange] = [
        FileChange(path='huge.py', insertions=10_000, deletions=0),
        FileChange(path='tiny_a.py', insertions=1, deletions=0),
        FileChange(path='tiny_b.py', insertions=1, deletions=0),
    ]
    diffs: dict[str, str] = {
        'huge.py': 'x' * 40_000,
        'tiny_a.py': '+alpha\n',
        'tiny_b.py': '+beta\n',
    }

    prompt: str = build_review_prompt('task', changes, diffs, budget_characters=3_000)

    assert '+alpha' in prompt
    assert '+beta' in prompt
    assert 'huge.py' in prompt


def test_ordering_by_size_is_what_admits_the_small_diffs() -> None:
    """
    Smallest first, proven by a case where only the ordering can decide it.

    The test above claims this property and cannot actually fail against it: it
    gives the huge file a diff far larger than the whole budget, so the size
    check excludes it whether or not anything was sorted, and the small files
    appear either way.

    Here every diff fits on its own and they cannot all fit together. Taken in
    the given order the medium file is admitted first and consumes enough of the
    budget that the last small file is turned away; taken smallest first, both
    small files are admitted and the medium one is not. Only the ordering
    separates those outcomes.
    """
    changes: list[FileChange] = [
        FileChange(path='medium.py', insertions=500, deletions=0),
        FileChange(path='tiny_a.py', insertions=1, deletions=0),
        FileChange(path='tiny_b.py', insertions=1, deletions=0),
    ]
    diffs: dict[str, str] = {
        'medium.py': 'm' * 2_500,
        'tiny_a.py': 'a' * 300,
        'tiny_b.py': 'b' * 300,
    }

    prompt: str = build_review_prompt('task', changes, diffs, budget_characters=3_000)

    assert 'a' * 300 in prompt, 'the smallest diffs should be admitted first'
    assert 'b' * 300 in prompt, 'and the second one should still fit'
    assert 'm' * 2_500 not in prompt, 'the medium diff should be the one left out'


async def test_fetching_is_ordered_so_one_big_file_stops_nothing() -> None:
    """
    A large change listed first must not shut out the small ones behind it.

    The loop stops at the first diff that will not fit and treats everything
    after it as omitted, so the order files arrive in decides what is read at
    all. The test below passes its files already in ascending order, which makes
    the sorting a no-op there and leaves the property unasserted — removing the
    sort changed nothing it could see.

    Here the largest is listed first, as `git` may well report it.
    """
    changes: list[FileChange] = [
        FileChange(path='huge.py', insertions=10_000, deletions=0),
        FileChange(path='tiny.py', insertions=1, deletions=0),
    ]
    bodies: dict[str, str] = {'huge.py': 'y' * 9_000, 'tiny.py': '+one\n'}

    async def fetch(path: str) -> str:
        return bodies[path]

    diffs, omitted = await collect_review_diffs(changes, fetch, budget_characters=3_000)

    assert 'tiny.py' in diffs, 'the small diff should be read despite the order given'
    assert 'huge.py' in omitted


async def test_only_affordable_diffs_are_fetched() -> None:
    """
    Diffs that could never fit the budget are not read at all.

    Fetching every file and then discarding most means a subprocess call per
    file on a large change, for content that is thrown away.
    """
    changes: list[FileChange] = [
        FileChange(path='tiny.py', insertions=1, deletions=0),
        FileChange(path='medium.py', insertions=50, deletions=0),
        FileChange(path='huge.py', insertions=10_000, deletions=0),
        FileChange(path='huger.py', insertions=20_000, deletions=0),
    ]
    bodies: dict[str, str] = {
        'tiny.py': '+one\n',
        'medium.py': 'x' * 300,
        'huge.py': 'y' * 5_000,
        'huger.py': 'z' * 9_000,
    }
    fetched: list[str] = []

    async def fetch(path: str) -> str:
        fetched.append(path)
        return bodies[path]

    diffs, omitted = await collect_review_diffs(changes, fetch, budget_characters=1_000)

    assert set(diffs) == {'tiny.py', 'medium.py'}
    assert sorted(omitted) == ['huge.py', 'huger.py']
    # The two oversized files stop the walk, so only one of them is ever read.
    assert 'huger.py' not in fetched
    assert len(fetched) == 3


async def test_all_diffs_fetched_when_they_fit() -> None:
    """
    A modest change set is read in full.
    """
    changes: list[FileChange] = [
        FileChange(path='a.py', insertions=2, deletions=0),
        FileChange(path='b.py', insertions=3, deletions=0),
    ]

    async def fetch(path: str) -> str:
        return f'+{path}\n'

    diffs, omitted = await collect_review_diffs(changes, fetch)

    assert set(diffs) == {'a.py', 'b.py'}
    assert omitted == []


def test_previous_feedback_is_included_on_re_review() -> None:
    """
    A re-review is told what it asked for last time.

    Without it the reviewer restates resolved points and the loop never
    converges.
    """
    changes: list[FileChange] = [FileChange(path='a.py', insertions=1, deletions=0)]
    prompt: str = build_review_prompt(
        'task',
        changes,
        {'a.py': '+fixed\n'},
        previous_feedback='The empty case is unhandled in a.py.',
    )

    assert 'The empty case is unhandled in a.py.' in prompt
    assert 'has been addressed' in prompt


def test_no_previous_feedback_section_on_first_review() -> None:
    """
    A first review carries no feedback section.
    """
    changes: list[FileChange] = [FileChange(path='a.py', insertions=1, deletions=0)]
    prompt: str = build_review_prompt('task', changes, {'a.py': '+new\n'})

    assert 'reviewed an earlier version' not in prompt


def test_omitted_paths_from_collection_are_reported() -> None:
    """
    Files skipped during collection are named in the prompt.
    """
    changes: list[FileChange] = [
        FileChange(path='a.py', insertions=1, deletions=0),
        FileChange(path='skipped.py', insertions=9_000, deletions=0),
    ]
    prompt: str = build_review_prompt(
        'task', changes, {'a.py': '+ok\n'}, omitted=['skipped.py']
    )

    assert 'skipped.py' in prompt
    assert 'too large to include' in prompt
