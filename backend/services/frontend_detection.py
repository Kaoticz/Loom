from __future__ import annotations

import json
import logging
import os

from services.repository_files import list_indexable_files

logger: logging.Logger = logging.getLogger(__name__)

FRONTEND_PACKAGES: frozenset[str] = frozenset(
    {
        '@angular/core',
        '@sveltejs/kit',
        'astro',
        'next',
        'nuxt',
        'preact',
        'react',
        'react-dom',
        'remix',
        'solid-js',
        'svelte',
        'vite',
        'vue',
    }
)
"""
Package names that mean a repository builds a web frontend.

Deliberately framework and bundler names rather than "does a `package.json`
exist at all": an Express-only API is a Node project with no browser to drive,
and telling its agent about `agent-browser` would be noise on every task.
"""

INDEX_HTML_MAX_DEPTH: int = 1
"""
How far below the repository root an `index.html` still counts as a frontend.

A Vite project keeps `index.html` at its own root, which is the repository root
in a single-project repository and one level down in a monorepo laid out as
`frontend/`. Looking deeper would start counting documentation output and
fixture directories, which say nothing about whether there is an app to drive.
"""

DEPENDENCY_SECTIONS: tuple[str, ...] = (
    'dependencies',
    'devDependencies',
    'peerDependencies',
    'optionalDependencies',
)
"""
The `package.json` keys worth reading. A frontend framework can legitimately
appear in any of them, and a build tool is usually a dev dependency.
"""


def repository_has_frontend(repo_path: str) -> bool:
    """
    Decide whether a repository plausibly contains a web frontend.

    Answered from the repository itself rather than from the task text, because
    the wording of a task is a poor guide to whether there is a browser to
    drive, while a declared dependency on a frontend framework is a direct one.

    Files the repository ignores are not consulted, so a checked-out
    `node_modules` cannot make every repository look like a frontend.

    :param repo_path: Path to the repository root.
    :returns: True if a frontend framework or a root-level `index.html` is present.
    """
    root: str = os.path.abspath(repo_path)

    try:
        paths: list[str] = list_indexable_files(root)
    except OSError as error:
        # Guidance is an optimisation, not a correctness requirement: an
        # unreadable repository should run the task without it rather than fail.
        logger.warning('Could not inspect %s for a frontend: %s', root, error)
        return False

    for path in paths:
        name: str = os.path.basename(path)
        if name == 'package.json' and _declares_frontend_package(path):
            return True
        if name == 'index.html' and _depth_of(path, root) <= INDEX_HTML_MAX_DEPTH:
            return True

    return False


def _depth_of(path: str, root: str) -> int:
    """
    How many directories separate a file from the repository root.

    :param path: Absolute path to the file.
    :param root: Absolute path to the repository root.
    :returns: 0 for a file at the root, 1 for one directory below, and so on.
    """
    relative: str = os.path.relpath(path, root)
    return len(relative.split(os.sep)) - 1


def _declares_frontend_package(path: str) -> bool:
    """
    Whether a `package.json` depends on something that renders in a browser.

    :param path: Absolute path to the `package.json`.
    :returns: True if any dependency section names a known frontend package.
    """
    try:
        with open(path, encoding='utf-8') as handle:
            manifest: object = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        # A malformed manifest is the repository's problem, not this check's.
        logger.info('Skipping unreadable manifest %s: %s', path, error)
        return False

    if not isinstance(manifest, dict):
        return False

    for section in DEPENDENCY_SECTIONS:
        declared: object = manifest.get(section)
        if not isinstance(declared, dict):
            continue
        if any(str(name) in FRONTEND_PACKAGES for name in declared):
            return True

    return False
