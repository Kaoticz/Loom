from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from unittest.mock import patch

import asyncpg
import pytest
import pytest_asyncio
from fastapi import WebSocket
from fastapi.testclient import TestClient

pytestmark = pytest.mark.usefixtures('require_database')


APP_DATABASE: str = 'loom_app_test'


def dsn_for(database: str) -> str:
    """
    Rewrite the test connection string to point at another database.

    :param database: The database name to target.
    :returns: The rewritten connection string.
    """
    parts = urlsplit(os.environ['TEST_DATABASE_URL'])
    return urlunsplit((parts.scheme, parts.netloc, f'/{database}', '', ''))


@pytest_asyncio.fixture(autouse=True)
async def app_database(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """
    Give the application its own empty database to migrate on startup.

    Other suites build the shared test database with `generate_schemas`, which
    creates tables the migrations then try to create again. The application owns
    its schema through migrations now, so it needs a database nothing else has
    written to.

    :returns: An iterator yielding once the database exists and Tortoise is clean.
    """
    from tortoise import Tortoise
    from tortoise.backends.base.executor import EXECUTOR_CACHE

    admin = await asyncpg.connect(dsn_for('postgres'))
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {APP_DATABASE}')
        await admin.execute(f'CREATE DATABASE {APP_DATABASE}')
    finally:
        await admin.close()

    await Tortoise.close_connections()
    await Tortoise._reset_apps()
    EXECUTOR_CACHE.clear()
    monkeypatch.setenv('DATABASE_URL', dsn_for(APP_DATABASE))

    yield

    await Tortoise.close_connections()
    await Tortoise._reset_apps()
    EXECUTOR_CACHE.clear()

    admin = await asyncpg.connect(dsn_for('postgres'))
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {APP_DATABASE}')
    finally:
        await admin.close()


def observe_connections(count: int) -> list[dict[str, Any]]:
    """
    Open several WebSocket connections and capture the services each one receives.

    The service objects themselves are retained, not their `id()` values: an id
    is only unique among objects that are alive at the same moment, so once the
    first connection's services are collected a later one can be handed the same
    address and appear identical.

    :param count: How many connections to open in turn.
    :returns: The injected services from each connection.
    """

    import main
    from api.v1.agent_socket_controller import AgentSocketController

    observed: list[dict[str, Any]] = []

    async def record(self: Any, websocket: WebSocket) -> None:
        await websocket.accept()
        observed.append({
            'controller': self,
            'docker': self.docker,
            'editor': self.editor,
            'git': self.git,
            'indexing': self.indexing,
            'embedding': self.embedding,
        })
        await websocket.close()

    with patch.object(AgentSocketController, 'handle_session', record):
        with TestClient(main.app) as client:
            for _ in range(count):
                with client.websocket_connect('/api/v1/agent-session'):
                    pass

    return observed


def test_each_connection_gets_its_own_services() -> None:
    """
    Two sessions never share a mutable service instance.

    Sessions reconfigure these services for their own repository, so a shared
    instance lets one session repoint another mid-run and land edits in the
    wrong working tree.
    """
    observed: list[dict[str, Any]] = observe_connections(2)

    assert len(observed) == 2
    first, second = observed
    for service in ('controller', 'docker', 'editor', 'git', 'indexing', 'embedding'):
        assert first[service] is not second[service], f'{service} was shared between sessions'


def test_http_routes_still_resolve_their_dependencies() -> None:
    """
    Moving the provider to session scope does not break the HTTP endpoints.
    """

    import main

    with TestClient(main.app) as client:
        assert client.post('/api/v1/tasks', json={'prompt': 'x'}).status_code == 200
        assert client.get('/api/v1/repos/list?path=/tmp').status_code == 200


def test_database_backed_endpoints_work_through_the_app() -> None:
    """
    A request that touches the database succeeds through the real app.

    Tortoise keeps its connections in a ContextVar, and the lifespan runs in a
    different task from request handlers, so without the global fallback every
    database-backed endpoint fails with "No TortoiseContext is currently
    active". Nothing caught this while the task endpoint returned a stub.
    """

    import main

    with TestClient(main.app) as client:
        created = client.post('/api/v1/tasks', json={'prompt': 'a real task'})
        assert created.status_code == 200, created.text
        task_id = created.json()['id']

        listed = client.get('/api/v1/tasks?limit=5')
        assert listed.status_code == 200
        assert listed.json()['total'] >= 1

        fetched = client.get(f'/api/v1/tasks/{task_id}')
        assert fetched.status_code == 200
        assert fetched.json()['task']['prompt'] == 'a real task'

        patched = client.patch(f'/api/v1/tasks/{task_id}', json={'status': 'completed'})
        assert patched.status_code == 200
        assert patched.json()['status'] == 'completed'

        assert client.get('/api/v1/tasks/not-a-uuid').status_code == 404


def test_browser_origins_are_allowed() -> None:
    """
    The frontend's origin may call the API from a browser.

    The frontend is served from a different port than the API, so every fetch it
    makes is cross-origin. Without CORS the model list, repository picker and
    task history all fail in the browser, while the agent session keeps working
    because WebSockets are exempt, which makes the fault easy to miss.
    """
    import main

    with TestClient(main.app) as client:
        response = client.get(
            '/api/v1/tasks?limit=1', headers={'Origin': 'http://localhost:5173'}
        )
        assert response.headers.get('access-control-allow-origin') == 'http://localhost:5173'

        preflight = client.options(
            '/api/v1/tasks',
            headers={
                'Origin': 'http://localhost:5173',
                'Access-Control-Request-Method': 'POST',
            },
        )
        assert preflight.status_code == 200
        assert 'POST' in (preflight.headers.get('access-control-allow-methods') or '')


def test_unknown_origins_are_not_allowed() -> None:
    """
    An origin that was not configured receives no permission to read responses.
    """
    import main

    with TestClient(main.app) as client:
        response = client.get(
            '/api/v1/tasks?limit=1', headers={'Origin': 'http://evil.example'}
        )
        assert response.headers.get('access-control-allow-origin') is None


def test_allowed_origins_are_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Serving the frontend from another address does not require a code change.
    """
    from main import allowed_origins

    monkeypatch.setenv('LOOM_CORS_ORIGINS', 'https://loom.example, https://other.example')
    assert allowed_origins() == ['https://loom.example', 'https://other.example']

    monkeypatch.setenv('LOOM_CORS_ORIGINS', '   ')
    assert 'http://localhost:5173' in allowed_origins()


def test_health_reports_ok_when_the_database_answers() -> None:
    """
    The health endpoint confirms the service can serve requests.
    """
    import main

    with TestClient(main.app) as client:
        response = client.get('/api/v1/health')
        assert response.status_code == 200
        assert response.json() == {'status': 'ok', 'database': True}


def test_health_fails_when_the_database_is_unreachable(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A service that cannot reach its database does not report itself healthy.

    Checking only that the port is open would keep an unusable container in
    rotation, and would let dependents start against a backend that cannot
    answer anything.
    """
    import main
    from tortoise import Tortoise

    with TestClient(main.app) as client:
        def broken(alias: str = 'default') -> object:
            raise RuntimeError('connection refused')

        monkeypatch.setattr(Tortoise, 'get_connection', broken)
        response = client.get('/api/v1/health')

        assert response.status_code == 503
        assert 'unreachable' in response.json()['detail']


@pytest.mark.usefixtures('app_database')
def test_index_status_and_deletion_go_through_the_app() -> None:
    """
    The endpoints answer about the project the caller names, and delete it.

    Exercised through the real app rather than by calling the functions, because
    the wiring is the part that has gone wrong before: the endpoints read the
    path from the request precisely so that a backend cannot answer about
    whichever repository it happened to be started with.
    """
    import main
    from database.models.file import File
    from services.projects import normalise_project_path

    mine: str = '/repos/mine'
    theirs: str = '/repos/theirs'

    with TestClient(main.app) as client:
        # Always present inside the context manager; asserted rather than
        # suppressed so the reason is on the page.
        assert client.portal is not None

        async def seed() -> None:
            for project in (mine, theirs):
                await File.create(
                    project=normalise_project_path(project),
                    file_path='src/main.py',
                    raw_content='x\n',
                    hash='h',
                )

        client.portal.call(seed)

        status = client.get('/api/v1/index/status', params={'path': mine})
        assert status.status_code == 200
        assert status.json()['indexed_files'] == 1
        assert status.json()['can_delete'] is True
        assert status.json()['blocked_reason'] == ''

        removed = client.request('DELETE', '/api/v1/index', params={'path': mine})
        assert removed.status_code == 200
        assert removed.json()['files_deleted'] == 1
        # The reader is told that nothing re-indexes until a session starts.
        assert 'session next starts' in removed.json()['message']

        # Scoped: the other project is untouched and still deletable.
        other = client.get('/api/v1/index/status', params={'path': theirs})
        assert other.json()['indexed_files'] == 1

        # And the emptied one now explains why it cannot be used.
        after = client.get('/api/v1/index/status', params={'path': mine})
        assert after.json()['can_delete'] is False
        assert 'Nothing is indexed' in after.json()['blocked_reason']

        # Repeating the delete is safe rather than confusing.
        again = client.request('DELETE', '/api/v1/index', params={'path': mine})
        assert again.status_code == 200
        assert again.json()['files_deleted'] == 0
        assert 'nothing was removed' in again.json()['message']


@pytest.mark.usefixtures('app_database')
def test_deletion_is_refused_while_a_session_is_running() -> None:
    """
    Refused rather than allowed, at the user's direction.

    Deleting the index underneath a working agent raises nothing it can report:
    every search simply returns empty, which reads as an unindexed repository.
    The status endpoint says the same thing, so a disabled control can explain
    itself with the same words rather than composing its own.
    """
    import main
    from services.session_registry import SessionRegistry

    with TestClient(main.app) as client:
        assert client.portal is not None
        registry: SessionRegistry = client.portal.call(
            main.app.state.dishka_container.get, SessionRegistry
        )
        client.portal.call(registry.acquire, 'session-1')

        try:
            status = client.get('/api/v1/index/status', params={'path': '/repos/mine'})
            assert status.json()['can_delete'] is False
            assert 'session is running' in status.json()['blocked_reason']

            refused = client.request('DELETE', '/api/v1/index', params={'path': '/repos/mine'})
            assert refused.status_code == 409
            assert 'session is running' in refused.json()['detail']
        finally:
            client.portal.call(registry.release, 'session-1')
