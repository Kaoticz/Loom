from __future__ import annotations

import os
import subprocess
from typing import Any
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient, MockTransport, Request, Response
from tortoise import Tortoise

from services.code_search import CodeSearchService
from services.docker_sandbox import DockerSandboxManager
from services.embedding import EmbeddingService
from services.event_bus import EventBus
from services.file_editor import FileEditorService
from services.git_manager import GitManagerService
from services.session_context import SessionContext

SQLITE_MODELS: list[str] = [
    'database.models.embedding_model',
    'database.models.file',
    'database.models.symbol',
    'database.models.task_memory',
]


def postgres_config(dsn: str) -> dict[str, Any]:
    """
    Build a Tortoise configuration for a specific database.

    `database.tortoise_config.TORTOISE_ORM` captures DATABASE_URL when the module
    is first imported, so a suite that imported it under a different database
    would pin every later suite to that one.

    :param dsn: The connection string to use.
    :returns: The Tortoise configuration.
    """
    # Importing this installs the pgvector-aware client, which is required for
    # vector columns to bind.
    from database import vector_client as vector_client  # noqa: F401

    return {
        'connections': {'default': dsn},
        'apps': {
            'models': {
                'models': ['database.models', 'aerich.models'],
                'default_connection': 'default',
            }
        },
    }


def make_git_repo(path: str, default_branch: str = 'main') -> GitManagerService:
    """
    Initialise a git repository with a single commit.

    :param path: Directory to initialise the repository in.
    :param default_branch: Name to give the repository's first branch.
    :returns: A git manager bound to the new repository.
    """
    subprocess.run(['git', 'init', '-b', default_branch, path], capture_output=True, check=True)
    subprocess.run(['git', '-C', path, 'config', 'user.email', 't@t.t'], capture_output=True, check=True)
    subprocess.run(['git', '-C', path, 'config', 'user.name', 'Test'], capture_output=True, check=True)
    Path(path, 'README.md').write_text('seed\n')
    subprocess.run(['git', '-C', path, 'add', '-A'], capture_output=True, check=True)
    subprocess.run(['git', '-C', path, 'commit', '-m', 'init'], capture_output=True, check=True)
    return GitManagerService(repo_path=path)


def git_log(path: str) -> str:
    """
    Read a repository's commit subjects.

    :param path: Path to the repository.
    :returns: The one-line commit log.
    """
    return subprocess.run(
        ['git', '-C', path, 'log', '--oneline'], capture_output=True, text=True
    ).stdout


def stub_embedding_service(vector: list[float] | None = None) -> EmbeddingService:
    """
    Build an embedding service backed by a mock transport.

    :param vector: The embedding every call should return.
    :returns: An embedding service that never touches the network.
    """
    payload: list[float] = vector if vector is not None else [0.1, 0.2, 0.3]

    def handler(request: Request) -> Response:
        return Response(200, json={'embedding': payload})

    return EmbeddingService('http://ollama:11434', client=AsyncClient(transport=MockTransport(handler)))


@pytest_asyncio.fixture
async def sqlite_db() -> AsyncIterator[None]:
    """
    Provide an in-memory SQLite database with the non-vector models.

    The `code_embeddings` table is excluded because pgvector columns cannot be
    bound by SQLite; vector behaviour is covered by the Postgres suite.

    :returns: An iterator yielding once the schema is ready.
    """
    await Tortoise.init(db_url='sqlite://:memory:', modules={'models': SQLITE_MODELS})
    await Tortoise.generate_schemas()
    yield
    await Tortoise.close_connections()


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[str]:
    """
    Provide an empty directory to act as a repository root.

    :returns: An iterator yielding the directory path.
    """
    root: Path = tmp_path / 'repo'
    root.mkdir()
    yield str(root)


@pytest.fixture
def git_repo(tmp_path: Path) -> Iterator[GitManagerService]:
    """
    Provide an initialised git repository.

    :returns: An iterator yielding a git manager for the repository.
    """
    root: Path = tmp_path / 'gitrepo'
    root.mkdir()
    yield make_git_repo(str(root))


@pytest.fixture
def editor(repo: str) -> FileEditorService:
    """
    Provide a file editor rooted at a temporary repository.

    :param repo: The repository root.
    :returns: The file editor service.
    """
    return FileEditorService(root_path=repo)


@pytest.fixture
def session() -> SessionContext:
    """
    Provide a fresh session context.

    :returns: An empty session context.
    """
    return SessionContext()


@pytest.fixture
def events() -> EventBus:
    """
    Provide a session-scoped event bus.

    :returns: An event bus for a test session.
    """
    return EventBus('test-session')


@pytest.fixture
def search() -> CodeSearchService:
    """
    Provide a code search service.

    :returns: The code search service.
    """
    return CodeSearchService()


@pytest.fixture
def sandbox(repo: str) -> DockerSandboxManager:
    """
    Provide a sandbox manager pointed at a temporary repository.

    :param repo: The repository root.
    :returns: The sandbox manager, with no Docker client attached.
    """
    return DockerSandboxManager(repo_path=repo)


@pytest.fixture(scope='session')
def ast_grep_bin() -> str:
    """
    Locate the ast-grep executable to use in tests.

    :returns: The configured executable name or path.
    """
    return os.getenv('AST_GREP_BIN', 'ast-grep')


def ast_grep_available(binary: str) -> bool:
    """
    Check whether ast-grep can be executed.

    :param binary: The executable name or path.
    :returns: True if the binary runs successfully.
    """
    try:
        subprocess.run([binary, '--version'], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True
