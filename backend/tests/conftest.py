from __future__ import annotations

import os
import subprocess
from typing import Any
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from httpx import AsyncClient, MockTransport, Request, Response
from tortoise import Tortoise

from services.code_search import CodeSearchService
from services.projects import normalise_project_path
from services.docker_sandbox import DockerSandboxManager
from services.embedding import EmbeddingService
from services.event_bus import EventBus
from services.file_editor import FileEditorService
from services.git_manager import GitManagerService
from services.session_context import SessionContext

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / '.env', override=False)


TEST_REPO_PATH: str = '/tmp/loom-test-project'
"""
The repository path tests index and search under.

Files created directly by a test must carry `TEST_PROJECT`, or a search bound to
this path will not see them — which is exactly the silent mismatch that scoping
is meant to make impossible in production, so it is worth one shared constant
here rather than a literal per test.
"""

TEST_PROJECT: str = normalise_project_path(TEST_REPO_PATH)
"""The normalised form of :data:`TEST_REPO_PATH`, as stored on a row."""


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


DB_REQUIRED_MESSAGE: str = (
    'These tests need PostgreSQL with pgvector. Set TEST_DATABASE_URL to a database '
    'you do not mind being emptied, for example '
    'postgres://loom:loom_password@localhost:5432/loom_test'
)

def managed_tables() -> list[str]:
    """
    Every table the suite may write to, read from the registered models.

    Derived rather than listed. A hand-written list is a second place that has to
    agree with the schema, and it did not: `tasks` was missing from it, so task
    rows survived from one test into the next with nothing to indicate it. The
    comment that stood here claimed such an omission would fail visibly. It would
    not — it leaked in silence, which is why this reads the models instead.

    :returns: The table names, for truncation and for rebuilding the schema.
    """
    if not Tortoise.apps:
        raise RuntimeError('managed_tables needs Tortoise initialised; depend on `db` first')

    return [
        model._meta.db_table
        # Guarded: an app registered but not yet populated appears as None, and
        # reading it would fail here rather than where the mistake was made.
        for app in Tortoise.apps.values()
        if app
        for model in app.values()
    ]


@pytest.fixture
def require_database() -> None:
    """
    Refuse to run without a database, rather than skipping.

    For suites that manage their own databases and so do not take `db`. A skip
    reports success while never reaching the engine the product runs on, which
    is the thing this arrangement exists to prevent.
    """
    if not os.getenv('TEST_DATABASE_URL'):
        raise RuntimeError(DB_REQUIRED_MESSAGE)


@pytest_asyncio.fixture(scope='session', loop_scope='session')
async def _database_schema() -> AsyncIterator[None]:
    """
    Build the schema once for the whole session.

    Creating it per test meant DDL per test, which was affordable for the
    handful of suites that used PostgreSQL and is not now that every suite does.

    :returns: An iterator yielding once the schema exists.
    """
    dsn: str | None = os.getenv('TEST_DATABASE_URL')

    if not dsn:
        # Raised rather than skipped. A suite that skips its database tests
        # reports success while never touching the engine the product runs on,
        # which is what this phase exists to end.
        raise RuntimeError(DB_REQUIRED_MESSAGE)

    await Tortoise.init(config=postgres_config(dsn))

    # Rebuilt from scratch: generate_schemas only adds missing tables, so a
    # database left from an earlier schema would silently lack new columns.
    connection = Tortoise.get_connection('default')
    await connection.execute_script(
        f'DROP TABLE IF EXISTS {", ".join(managed_tables())} CASCADE;'
    )
    await Tortoise.generate_schemas(safe=True)

    yield

    await Tortoise.close_connections()


@pytest_asyncio.fixture
async def db(_database_schema: None) -> AsyncIterator[None]:
    """
    Provide an empty database for one test.

    Emptied by truncation rather than by rolling back a transaction. Rollback is
    faster, but Tortoise hands out connections from a pool, so code under test
    that opens its own connection would do its work outside the test's
    transaction and see an empty database — and code that commits would escape
    the rollback entirely. Truncation is correct regardless of what the code
    under test does with connections.

    `RESTART IDENTITY` matters: PostgreSQL sequences do not reset on their own,
    so without it a test asserting on a specific primary key would pass alone and
    fail in a suite.

    :returns: An iterator yielding once the database is empty.
    """
    # Connected per test rather than sharing the session's connection. Other
    # suites in this process initialise Tortoise against databases of their own
    # and close connections when they finish, so a shared handle is not reliably
    # open — nor reliably pointing at this database, which is worse. Opening a
    # pool costs a few milliseconds; creating the schema, which is the expensive
    # part, still happens once.
    await Tortoise.init(config=postgres_config(os.environ['TEST_DATABASE_URL']))

    connection = Tortoise.get_connection('default')
    await connection.execute_script(
        f'TRUNCATE {", ".join(managed_tables())} RESTART IDENTITY CASCADE;'
    )

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
    Provide a code search service bound to the shared test project.

    :returns: The code search service.
    """
    return CodeSearchService(repo_path=TEST_REPO_PATH)


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
