from git import Repo
import os
import asyncio

class GitManagerService:
    """
    Service for repository management using GitPython.
    """
    _repo: Repo | None = None
    _repo_path: str

    def __init__(self, repo_path: str):
        self._repo_path = repo_path

    @property
    def repo_path(self) -> str:
        return self._repo_path

    @repo_path.setter
    def repo_path(self, value: str):
        if self._repo_path != value:
            self._repo_path = value
            self._repo = None

    @property
    def repo(self) -> Repo:
        """
        Lazy-loaded GitPython Repo instance.
        """
        if self._repo is None:
            # Ensure the directory is considered safe by Git
            from git import Git
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

    async def create_branch(self, branch_name: str) -> None:
        """
        Create and checkout a new branch.

        :param branch_name: The name of the new branch.
        """
        def _create():
            if branch_name in self.repo.heads:
                self.repo.heads[branch_name].checkout()
            else:
                new_branch = self.repo.create_head(branch_name)
                new_branch.checkout()
        await asyncio.to_thread(_create)

    async def commit_changes(self, message: str) -> None:
        """
        Stage all changes and commit.

        :param message: The commit message.
        """
        def _commit():
            self.repo.git.add(A=True)
            self.repo.index.commit(message)
        await asyncio.to_thread(_commit)

    async def get_diff(self) -> str:
        """
        Generate a git diff of the current changes.

        :returns: The diff string.
        """
        def _diff():
            return self.repo.git.diff('HEAD~1' if len(self.repo.heads) > 1 else None)
        return await asyncio.to_thread(_diff)
        
    async def merge_branch(self, branch_name: str) -> None:
        """
        Merge the given branch into main.
        """
        def _merge():
            self.repo.heads.main.checkout()
            self.repo.git.merge(branch_name)
        await asyncio.to_thread(_merge)
