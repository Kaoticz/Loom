from __future__ import annotations

import os

PROJECT_PATH_MAX_LENGTH: int = 1024
"""
How long a stored project identifier may be.

Matches the width of `File.file_path`, since both hold filesystem paths and
there is no reason for one to accept what the other would truncate.
"""


def normalise_project_path(repo_path: str) -> str:
    """
    Reduce a repository path to the single spelling used to identify a project.

    A project is identified by where its repository sits on disk, because that
    is what the system already knows and it asks nothing new of the user. The
    consequence, which is worth being plain about: moving a repository makes it
    a different project, and its index has to be rebuilt.

    Normalisation matters more than it looks. `/repo`, `/repo/`, `./repo` and a
    symbolic link to `/repo` are one project to the person using it, but four
    distinct strings to a database. Left unnormalised, indexing under one
    spelling and searching under another would silently return nothing, which
    reads exactly like an empty index.

    :param repo_path: The repository path, in any spelling.
    :returns: The absolute, link-resolved path, without a trailing separator.
    """
    # `realpath` makes the path absolute, resolves symbolic links and drops any
    # trailing separator, which is every case that needs collapsing here.
    return os.path.realpath(os.path.expanduser(repo_path))
