from __future__ import annotations

import logging
import os

from git import Repo
from git.exc import GitError, InvalidGitRepositoryError, NoSuchPathError

logger: logging.Logger = logging.getLogger(__name__)

BACKSTOP_DIRECTORIES: frozenset[str] = frozenset(
    {'.git', 'node_modules', '__pycache__', 'dist', 'build', '.venv', 'venv'}
)
"""
Directories skipped whether or not the repository ignores them.

A repository that does not ignore its own build output would otherwise have tens
of thousands of generated files embedded, which is slow, expensive, and buries
the code the reader actually asked about. This is a backstop, not the rule: the
rule is whatever the repository's own ignore files say.
"""


def list_indexable_files(repo_path: str) -> list[str]:
    """
    List the files in a repository that are worth indexing.

    Ignore rules are resolved by git itself rather than by reading `.gitignore`
    and matching patterns here. Git's rules are more involved than they first
    appear — nested ignore files each apply to their own subtree, later patterns
    override earlier ones, a leading `!` reinstates a path, `.git/info/exclude`
    and the user's global excludes both count, and a file that is already
    tracked stays tracked whatever the patterns say. Asking git means every one
    of those is answered correctly and stays correct.

    A directory that is not a git repository falls back to walking it, since
    there are no ignore rules to obey; the backstop directories are skipped
    either way.

    :param repo_path: Path to the repository root.
    :returns: Absolute paths of the files to index.
    """
    root: str = os.path.abspath(repo_path)

    try:
        return _files_git_keeps(root)
    except (InvalidGitRepositoryError, NoSuchPathError):
        # Both subclass GitError, so this clause must stay above the one below.
        # It exists only to say the ordinary thing quietly: a directory that is
        # not a repository is not a fault, while git failing on one that is
        # deserves a warning.
        logger.info('%s is not a git repository, so no ignore rules apply to it.', root)
    except GitError as error:
        # Indexing everything is a worse failure than indexing nothing, but
        # refusing to index at all would be worse still: the session cannot
        # start without it. The reason is logged so this is not silent.
        logger.warning('Could not ask git which files to index (%s); walking instead.', error)

    return _walk(root)


def _files_git_keeps(root: str) -> list[str]:
    """
    Ask git which files it tracks or would track.

    :param root: Absolute path to the repository root.
    :returns: Absolute paths of files git does not ignore.
    :raises InvalidGitRepositoryError: If the path is not a git repository.
    :raises NoSuchPathError: If the path does not exist.
    :raises GitError: If git could not answer.
    """
    repo: Repo = Repo(root)
    # `--cached` for tracked files and `--others` for untracked ones, together
    # covering everything present; `--exclude-standard` applies every ignore
    # source git knows about. NUL separation keeps paths with spaces or newlines
    # in them intact.
    listing: str = repo.git.ls_files('-z', '--cached', '--others', '--exclude-standard')

    kept: list[str] = []
    for relative in listing.split('\0'):
        if not relative:
            continue
        if _within_backstop(relative):
            continue
        absolute: str = os.path.join(root, relative)
        # `--cached` lists files that are tracked but no longer on disk, and a
        # path that is gone cannot be read.
        if os.path.isfile(absolute):
            kept.append(absolute)
    return kept


def _within_backstop(relative: str) -> bool:
    """
    Whether a repository-relative path lies inside a backstop directory.

    :param relative: The path, relative to the repository root.
    :returns: True if any segment of its directory is a backstop.
    """
    segments: list[str] = relative.replace(os.sep, '/').split('/')
    return any(segment in BACKSTOP_DIRECTORIES for segment in segments[:-1])


def _walk(root: str) -> list[str]:
    """
    Walk a directory that has no ignore rules to obey.

    :param root: Absolute path to walk.
    :returns: Absolute paths of the files found.
    """
    found: list[str] = []
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = [
            name for name in subdirectories if name not in BACKSTOP_DIRECTORIES
        ]
        found.extend(os.path.join(directory, name) for name in files)
    return found
