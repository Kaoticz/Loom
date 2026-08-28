from __future__ import annotations

from pathlib import Path

import pytest

from services.git_manager import GitManagerService
from tests.conftest import make_git_repo


@pytest.mark.parametrize('default_branch', ['main', 'master', 'trunk'])
async def test_detects_the_default_branch(tmp_path: Path, default_branch: str) -> None:
    """
    The repository's default branch is detected regardless of its name.
    """
    root: Path = tmp_path / f'repo-{default_branch}'
    root.mkdir()
    git: GitManagerService = make_git_repo(str(root), default_branch)
    assert git.default_branch == default_branch


async def test_default_branch_ignores_the_active_task_branch(tmp_path: Path) -> None:
    """
    Checking out a task branch does not make it look like the default branch.
    """
    root: Path = tmp_path / 'repo'
    root.mkdir()
    git: GitManagerService = make_git_repo(str(root), 'trunk')

    await git.create_branch('ai-task-abc')
    assert git.default_branch == 'trunk'


async def test_base_branch_is_recorded_at_branch_creation(git_repo: GitManagerService) -> None:
    """
    Creating a task branch records the branch it was cut from.
    """
    assert git_repo.base_branch is None
    await git_repo.create_branch('ai-task-abc')
    assert git_repo.base_branch == 'main'


async def test_merge_returns_to_the_base_branch(tmp_path: Path) -> None:
    """
    Merging puts the work on the branch the task started from.
    """
    root: Path = tmp_path / 'repo'
    root.mkdir()
    git: GitManagerService = make_git_repo(str(root), 'trunk')

    await git.create_branch('ai-task-abc')
    Path(root, 'feature.txt').write_text('new feature\n')
    await git.commit_changes('Add feature')

    assert await git.merge_branch('ai-task-abc', 'Add feature') == 'trunk'
    assert git.repo.active_branch.name == 'trunk'
    assert Path(root, 'feature.txt').exists()


async def test_explicit_merge_target_is_honoured(git_repo: GitManagerService) -> None:
    """
    An explicitly supplied target branch overrides detection.
    """
    root: str = git_repo.repo_path
    await git_repo.create_branch('side')
    Path(root, 'x.txt').write_text('x\n')
    await git_repo.commit_changes('side work')

    assert await git_repo.merge_branch('side', 'Add side work', target_branch='main') == 'main'


async def test_create_branch_checks_out_existing_branch(git_repo: GitManagerService) -> None:
    """
    Re-creating an existing branch checks it out instead of failing.
    """
    await git_repo.create_branch('ai-task-abc')
    await git_repo.create_branch('ai-task-abc')
    assert git_repo.repo.active_branch.name == 'ai-task-abc'


async def test_commit_returns_details(git_repo: GitManagerService) -> None:
    """
    A commit reports its hash, message and the files it touched.
    """
    Path(git_repo.repo_path, 'new.txt').write_text('content\n')
    info = await git_repo.commit_changes('Add new.txt')

    assert info is not None
    assert len(info.commit_hash) == 40
    assert info.message == 'Add new.txt'
    assert info.files_changed == ['new.txt']


async def test_commit_with_nothing_staged_returns_none(git_repo: GitManagerService) -> None:
    """
    Committing an unchanged tree reports that nothing happened.
    """
    assert await git_repo.commit_changes('No changes') is None


async def test_reads_a_file_from_another_branch(git_repo: GitManagerService) -> None:
    """
    The pre-edit version of a file is recoverable from the base branch.
    """
    root: str = git_repo.repo_path
    Path(root, 'tracked.py').write_text('original\n')
    await git_repo.commit_changes('Add tracked.py')

    await git_repo.create_branch('ai-task-abc')
    Path(root, 'tracked.py').write_text('modified\n')
    await git_repo.commit_changes('Edit tracked.py')

    assert await git_repo.get_file_at_branch('main', 'tracked.py') == 'original'
    assert Path(root, 'tracked.py').read_text() == 'modified\n'


async def test_missing_file_on_branch_returns_none(git_repo: GitManagerService) -> None:
    """
    A file that does not exist on the base branch reports None, not an error.

    This is the newly created file case, whose diff has an empty original side.
    """
    assert await git_repo.get_file_at_branch('main', 'never_existed.py') is None


async def test_unresolvable_branch_raises(git_repo: GitManagerService) -> None:
    """
    A branch that does not exist is an error, not an absent file.

    Reporting it as absent would make every diff show an empty original, so the
    whole session would look like it created every file from nothing.
    """
    with pytest.raises(ValueError, match='does not exist'):
        await git_repo.get_file_at_branch('no-such-branch', 'README.md')


async def test_cumulative_diff_covers_every_commit_on_the_branch(
    git_repo: GitManagerService
) -> None:
    """
    The diff spans the whole task, not only its most recent commit.

    Every edit is micro-committed, so diffing against the previous commit would
    show the reviewer one edit and present it as the entire change.
    """
    root: str = git_repo.repo_path
    await git_repo.create_branch('ai-task-abc')

    Path(root, 'first.py').write_text('value = 1\n')
    await git_repo.commit_changes('Add first.py')
    Path(root, 'second.py').write_text('value = 2\n')
    await git_repo.commit_changes('Add second.py')
    Path(root, 'third.py').write_text('value = 3\n')
    await git_repo.commit_changes('Add third.py')

    diff: str = await git_repo.get_cumulative_diff()

    for name in ('first.py', 'second.py', 'third.py'):
        assert name in diff, f'{name} missing from the cumulative diff'


async def test_changed_files_reports_counts_largest_first(
    git_repo: GitManagerService
) -> None:
    """
    The change summary lists each file with its insertions and deletions.
    """
    root: str = git_repo.repo_path
    await git_repo.create_branch('ai-task-abc')

    Path(root, 'small.py').write_text('a = 1\n')
    Path(root, 'large.py').write_text(''.join(f'line_{i} = {i}\n' for i in range(20)))
    await git_repo.commit_changes('Add two files')

    changes = await git_repo.get_changed_files()

    assert [change.path for change in changes] == ['large.py', 'small.py']
    assert changes[0].insertions == 20
    assert changes[1].insertions == 1
    assert all(change.deletions == 0 for change in changes)


async def test_per_file_diff_covers_only_that_file(git_repo: GitManagerService) -> None:
    """
    A single file's diff excludes everything else that changed.
    """
    root: str = git_repo.repo_path
    await git_repo.create_branch('ai-task-abc')

    Path(root, 'wanted.py').write_text('wanted = 1\n')
    Path(root, 'other.py').write_text('other = 2\n')
    await git_repo.commit_changes('Add both')

    diff: str = await git_repo.get_file_diff('wanted.py')

    assert 'wanted.py' in diff
    assert 'other.py' not in diff


async def test_commit_count_reflects_the_branch(git_repo: GitManagerService) -> None:
    """
    The commit count is what the task added, not the repository's history.
    """
    root: str = git_repo.repo_path
    await git_repo.create_branch('ai-task-abc')
    assert await git_repo.count_commits() == 0

    for index in range(3):
        Path(root, f'file_{index}.py').write_text(f'x = {index}\n')
        await git_repo.commit_changes(f'Add file_{index}.py')

    assert await git_repo.count_commits() == 3


async def test_nothing_changed_yields_an_empty_diff(git_repo: GitManagerService) -> None:
    """
    A branch with no commits produces no diff and no changed files.
    """
    await git_repo.create_branch('ai-task-abc')

    assert await git_repo.get_cumulative_diff() == ''
    assert await git_repo.get_changed_files() == []
    assert await git_repo.count_commits() == 0


async def test_merge_conflict_is_raised(git_repo: GitManagerService) -> None:
    """
    A conflicting merge raises rather than silently leaving a broken tree.
    """
    from git.exc import GitCommandError

    root: str = git_repo.repo_path
    Path(root, 'shared.py').write_text('original\n')
    await git_repo.commit_changes('Add shared.py')

    await git_repo.create_branch('ai-task-abc')
    Path(root, 'shared.py').write_text('from the branch\n')
    await git_repo.commit_changes('Branch edit')

    # Move main ahead with a conflicting change.
    git_repo.repo.heads['main'].checkout()
    Path(root, 'shared.py').write_text('from main\n')
    await git_repo.commit_changes('Main edit')

    with pytest.raises(GitCommandError):
        await git_repo.merge_branch('ai-task-abc', 'Apply branch edit')


async def test_squash_merge_creates_one_commit_and_deletes_the_task_branch(
    git_repo: GitManagerService,
) -> None:
    '''
    A squash merge applies every task change in one target-branch commit.
    '''
    root: str = git_repo.repo_path
    await git_repo.create_branch('ai-task-abc')
    for index in range(3):
        Path(root, f'file_{index}.py').write_text(f'value = {index}\n')
        await git_repo.commit_changes(f'Add file {index}')

    subjects: list[str] = await git_repo.get_commit_subjects()
    assert subjects == ['Add file 0', 'Add file 1', 'Add file 2']

    assert await git_repo.merge_branch('ai-task-abc', 'Add task files') == 'main'
    assert git_repo.repo.head.commit.message.strip() == 'Add task files'
    assert int(git_repo.repo.git.rev_list('--count', 'main').strip()) == 2
    assert await git_repo.delete_branch('ai-task-abc')
    assert 'ai-task-abc' not in git_repo.repo.heads
    assert not git_repo.repo.is_dirty(untracked_files=True)
    assert all(Path(root, f'file_{index}.py').exists() for index in range(3))
