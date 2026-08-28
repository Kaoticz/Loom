from git import Git, Head, Repo
from git.exc import GitCommandError
from pydantic import BaseModel, Field
from typing import Annotated
import asyncio
from logging import Logger, getLogger

logger: Logger = getLogger(__name__)


class FileChange(BaseModel):
    """
    A single file's contribution to a cumulative diff.
    """

    path: Annotated[str, Field(description='Repository-relative path of the changed file.')]
    insertions: Annotated[int, Field(ge=0, description='Lines added, or zero for a binary file.')]
    deletions: Annotated[int, Field(ge=0, description='Lines removed, or zero for a binary file.')]

    @property
    def total_changed(self) -> int:
        """
        How many lines the change touches in total.

        :returns: The sum of insertions and deletions.
        """
        return self.insertions + self.deletions


class CommitInfo(BaseModel):
    """
    Details of a commit that was just written.
    """

    commit_hash: Annotated[str, Field(description='The full SHA of the new commit.')]
    message: Annotated[str, Field(description='The commit message.')]
    files_changed: Annotated[list[str], Field(default_factory=list, description='Repository-relative paths touched by the commit.')]

class CommitResult(BaseModel):
    """
    What came of trying to commit.

    All three outcomes are distinct and the caller acts differently on each: a
    commit was made, there was nothing to commit, or it failed and the changes
    are still sitting in the working tree where review will not see them.
    """

    commit: Annotated[CommitInfo | None, Field(description='The commit, if one was made.')] = None
    error: Annotated[str | None, Field(description='Why the commit failed, if it did.')] = None

    @property
    def failed(self) -> bool:
        """
        Whether the commit was attempted and did not succeed.

        :returns: True if an error was recorded.
        """
        return self.error is not None


class GitManagerService:
    """
    Service for repository management using GitPython.
    """
    _repo: Repo | None = None
    _repo_path: str
    _base_branch: str | None = None

    def __init__(self, repo_path: str) -> None:
        """
        Create a git manager for a repository.

        :param repo_path: Path to the repository to manage.
        """
        self._repo_path = repo_path
        self._base_branch = None

    @property
    def repo_path(self) -> str:
        return self._repo_path

    @repo_path.setter
    def repo_path(self, value: str) -> None:
        if self._repo_path != value:
            self._repo_path = value
            self._repo = None
            self._base_branch = None

    @property
    def repo(self) -> Repo:
        """
        Lazy-loaded GitPython Repo instance.
        """
        if self._repo is None:
            # Ensure the directory is considered safe by Git
            try:
                Git().config('--global', '--add', 'safe.directory', self.repo_path)
            except Exception:
                pass # Already added or cannot add

            self._repo = Repo(self.repo_path)
            
            # Set agent identity
            with self._repo.config_writer() as writer:
                writer.set_value('user', 'name', 'Loom AI')
                writer.set_value('user', 'email', 'agent@loom.ai')
                writer.release()

        return self._repo

    @property
    def base_branch(self) -> str | None:
        """
        The branch the current task branch was created from, if one was created.

        :returns: The recorded base branch name, or None if no branch has been created yet.
        """
        return self._base_branch

    @property
    def default_branch(self) -> str:
        """
        Name of the repository's default branch.

        The remote's HEAD is consulted first, since that is authoritative for
        cloned repositories. Well known default names are checked next. The
        currently checked out branch is only used as a last resort, because
        during a task it is the agent's own branch rather than the default.

        :returns: The name of the default branch.
        :raises RuntimeError: If the repository has no branches at all.
        """
        for remote in self.repo.remotes:
            try:
                reference: str = self.repo.git.symbolic_ref(f'refs/remotes/{remote.name}/HEAD')
                return reference.rsplit('/', 1)[-1]
            except Exception:
                continue

        for candidate in ('main', 'master', 'trunk', 'develop'):
            if candidate in self.repo.heads:
                return candidate

        if not self.repo.head.is_detached:
            return str(self.repo.active_branch.name)

        if len(self.repo.heads) > 0:
            return str(self.repo.heads[0].name)

        raise RuntimeError('The repository has no branches to merge into.')

    async def create_branch(self, branch_name: str) -> None:
        """
        Create and checkout a new branch.

        :param branch_name: The name of the new branch.
        """
        def _create() -> None:
            if not self.repo.head.is_detached and self.repo.active_branch.name != branch_name:
                self._base_branch = str(self.repo.active_branch.name)
            if branch_name in self.repo.heads:
                self.repo.heads[branch_name].checkout()
            else:
                new_branch: Head = self.repo.create_head(branch_name)
                new_branch.checkout()
        await asyncio.to_thread(_create)

    async def commit_changes(self, message: str) -> CommitInfo | None:
        """
        Stage all changes and commit them.

        The details are returned rather than announced, so that this service
        stays unaware of the WebSocket: the caller decides what to report.

        :param message: The commit message.
        :returns: Details of the new commit, or None if there was nothing to commit.
        """
        def _commit() -> CommitInfo | None:
            self.repo.git.add(A=True)
            if not self.repo.index.diff('HEAD') and not self.repo.untracked_files:
                return None

            changed: list[str] = [str(item.a_path) for item in self.repo.index.diff('HEAD')]
            commit = self.repo.index.commit(message)
            return CommitInfo(
                commit_hash=str(commit.hexsha),
                message=message,
                files_changed=sorted(path for path in changed if path)
            )
        return await asyncio.to_thread(_commit)

    async def commit_changes_with_retry(self, message: str) -> CommitResult:
        """
        Commit, retrying once, and report the outcome rather than raising.

        A commit can fail for reasons that pass on their own — an index lock
        held by something else for a moment — and for reasons that never will,
        such as an unconfigured identity. One retry separates the two cheaply.

        The failure is returned rather than raised because the caller is part
        way through a plan: the changes are on disk either way, and killing the
        session loses the work along with the diagnosis.

        :param message: The commit message.
        :returns: The commit, the reason it failed, or neither if there was nothing to commit.
        """
        attempts: int = 2
        last: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                commit: CommitInfo | None = await self.commit_changes(message)
            except Exception as error:
                last = error
                logger.warning('Commit attempt %d of %d failed: %s', attempt, attempts, error)
                continue
            return CommitResult(commit=commit, error=None)

        return CommitResult(commit=None, error=str(last) if last is not None else 'unknown error')

    async def has_uncommitted_changes(self) -> bool:
        """
        Whether the working tree holds anything not yet committed.

        Used to tell "the commit failed" apart from "there was nothing to
        commit", and to notice work that would otherwise reach neither the
        reviewer nor the merge, both of which read only commits.

        :returns: True if anything is modified, staged or untracked.
        """
        def _dirty() -> bool:
            return bool(self.repo.is_dirty(untracked_files=True))

        return await asyncio.to_thread(_dirty)

    async def get_file_at_branch(self, branch: str, file_path: str) -> str | None:
        """
        Read a file's contents as they exist on another branch.

        This is how the pre-edit version of a file is recovered: the indexed
        copy in the database is refreshed by the indexer and no longer holds the
        original once a file has been touched.

        A file that is absent from the branch reports None, since that is the
        ordinary case for a newly created file. An unresolvable branch raises
        instead: treating it as "absent" would make every diff look like a new
        file, with nothing to indicate the base was wrong.

        :param branch: The branch to read from.
        :param file_path: Repository-relative path of the file.
        :returns: The file's contents on that branch, or None if it did not exist there.
        :raises ValueError: If the branch cannot be resolved in this repository.
        """
        def _show() -> str | None:
            try:
                self.repo.git.rev_parse('--verify', '--quiet', f'{branch}^{{commit}}')
            except GitCommandError as error:
                raise ValueError(
                    f'Cannot read {file_path!r} from {branch!r}: that branch does not exist '
                    f'in {self._repo_path}.'
                ) from error

            try:
                return str(self.repo.git.show(f'{branch}:{file_path}'))
            except GitCommandError:
                return None
        return await asyncio.to_thread(_show)

    async def get_diff(self) -> str:
        """
        Generate a git diff of the current changes.

        :returns: The diff string.
        """
        def _diff() -> str:
            return str(self.repo.git.diff('HEAD~1' if len(self.repo.heads) > 1 else None))
        return await asyncio.to_thread(_diff)

    async def get_cumulative_diff(self, base_branch: str | None = None) -> str:
        """
        Diff every change made on the task branch against where it started.

        A three-dot diff is used so the result describes the task's own work
        even if the base branch has moved on. Diffing against the previous
        commit would only ever show the most recent micro-commit.

        :param base_branch: The branch to compare against. Defaults to the recorded base.
        :returns: The cumulative diff, empty if nothing has changed.
        """
        base: str = base_branch or self._base_branch or self.default_branch

        def _diff() -> str:
            return str(self.repo.git.diff(f'{base}...HEAD'))
        return await asyncio.to_thread(_diff)

    async def get_changed_files(self, base_branch: str | None = None) -> list[FileChange]:
        """
        List the files the task branch has changed, with their line counts.

        :param base_branch: The branch to compare against. Defaults to the recorded base.
        :returns: The changed files, largest change first.
        """
        base: str = base_branch or self._base_branch or self.default_branch

        def _numstat() -> list[FileChange]:
            raw: str = str(self.repo.git.diff(f'{base}...HEAD', numstat=True))
            changes: list[FileChange] = []
            for line in raw.splitlines():
                parts: list[str] = line.split('\t')
                if len(parts) != 3:
                    continue
                added, removed, path = parts
                changes.append(
                    FileChange(
                        path=path.strip(),
                        # Git reports '-' for binary files.
                        insertions=int(added) if added.isdigit() else 0,
                        deletions=int(removed) if removed.isdigit() else 0,
                    )
                )
            return sorted(changes, key=lambda change: change.total_changed, reverse=True)
        return await asyncio.to_thread(_numstat)

    async def get_file_diff(self, file_path: str, base_branch: str | None = None) -> str:
        """
        Diff a single file's cumulative change on the task branch.

        :param file_path: Repository-relative path of the file.
        :param base_branch: The branch to compare against. Defaults to the recorded base.
        :returns: The diff for that file alone.
        """
        base: str = base_branch or self._base_branch or self.default_branch

        def _diff() -> str:
            return str(self.repo.git.diff(f'{base}...HEAD', '--', file_path))
        return await asyncio.to_thread(_diff)

    async def count_commits(self, base_branch: str | None = None) -> int:
        """
        Count the commits the task branch has added.

        :param base_branch: The branch to compare against. Defaults to the recorded base.
        :returns: The number of commits ahead of the base branch.
        """
        base: str = base_branch or self._base_branch or self.default_branch

        def _count() -> int:
            output: str = str(self.repo.git.rev_list('--count', f'{base}..HEAD'))
            return int(output.strip() or 0)
        return await asyncio.to_thread(_count)

    async def get_commit_subjects(self, base_branch: str | None = None) -> list[str]:
        '''
        List the task branch's commit subjects.

        :param base_branch: The branch to compare against. Defaults to the recorded base.
        :returns: Commit subjects in chronological order.
        '''
        base: str = base_branch or self._base_branch or self.default_branch

        def _subjects() -> list[str]:
            output: str = str(self.repo.git.log('--format=%s', '--reverse', f'{base}..HEAD'))
            return [subject for subject in output.splitlines() if subject.strip()]

        return await asyncio.to_thread(_subjects)

    async def merge_branch(
        self,
        branch_name: str,
        commit_message: str,
        target_branch: str | None = None,
    ) -> str:
        """
        Squash the given branch into one commit on its target branch.

        :param branch_name: The name of the branch to merge.
        :param commit_message: Subject for the resulting squash commit.
        :param target_branch: The branch to merge into. Defaults to the branch the task was cut from, then to the repository's default branch.
        :returns: The name of the branch that was merged into.
        :raises RuntimeError: If the repository has no branches to merge into.
        :raises GitCommandError: If the merge fails, for example because of a conflict.
        """
        def _merge() -> str:
            target: str = target_branch or self._base_branch or self.default_branch
            self.repo.heads[target].checkout()
            self.repo.git.merge('--squash', branch_name)
            self.repo.git.commit('-m', commit_message)
            return target
        return await asyncio.to_thread(_merge)

    async def delete_branch(self, branch_name: str) -> bool:
        '''
        Delete a task branch after its changes have been squash-merged.

        :param branch_name: The branch to delete.
        :returns: True when the branch was deleted, otherwise False.
        '''
        def _delete() -> bool:
            try:
                self.repo.git.branch('-D', branch_name)
            except GitCommandError as error:
                logger.warning('Could not delete task branch %s after merge: %s', branch_name, error)
                return False
            return True

        return await asyncio.to_thread(_delete)
