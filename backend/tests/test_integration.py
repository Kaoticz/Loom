from __future__ import annotations

import json
import os
import subprocess
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

pytestmark = pytest.mark.usefixtures('require_database')

INTEGRATION_DATABASE: str = 'loom_integration_test'


def dsn_for(database: str) -> str:
    """
    Rewrite the test connection string to point at another database.

    :param database: The database name to target.
    :returns: The rewritten connection string.
    """
    parts = urlsplit(os.environ['TEST_DATABASE_URL'])
    return urlunsplit((parts.scheme, parts.netloc, f'/{database}', '', ''))


@pytest_asyncio.fixture
async def integration_database(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """
    Give the application an empty database that it migrates itself on startup.

    :returns: An iterator yielding once the database exists.
    """
    from tortoise import Tortoise
    from tortoise.backends.base.executor import EXECUTOR_CACHE

    admin = await asyncpg.connect(dsn_for('postgres'))
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {INTEGRATION_DATABASE}')
        await admin.execute(f'CREATE DATABASE {INTEGRATION_DATABASE}')
    finally:
        await admin.close()

    await Tortoise.close_connections()
    await Tortoise._reset_apps()
    EXECUTOR_CACHE.clear()
    monkeypatch.setenv('DATABASE_URL', dsn_for(INTEGRATION_DATABASE))

    yield

    await Tortoise.close_connections()
    await Tortoise._reset_apps()
    EXECUTOR_CACHE.clear()

    admin = await asyncpg.connect(dsn_for('postgres'))
    try:
        # Pooled connections can outlive the app's shutdown, and Postgres refuses
        # to drop a database that anything is still attached to.
        await admin.execute(
            'SELECT pg_terminate_backend(pid) FROM pg_stat_activity '
            'WHERE datname = $1 AND pid <> pg_backend_pid()',
            INTEGRATION_DATABASE,
        )
        await admin.execute(f'DROP DATABASE IF EXISTS {INTEGRATION_DATABASE}')
    finally:
        await admin.close()


@pytest.fixture
def project(tmp_path: Path) -> Iterator[str]:
    """
    Provide a real git repository containing a little source to index.

    :returns: An iterator yielding the repository path.
    """
    root: Path = tmp_path / 'project'
    root.mkdir()
    (root / 'calculator.py').write_text(
        'def add(a: int, b: int) -> int:\n'
        '    return a + b\n'
        '\n'
        '\n'
        'class Calculator:\n'
        '    def multiply(self, a: int, b: int) -> int:\n'
        '        return a * b\n'
    )
    (root / 'README.md').write_text('# Project\n')

    subprocess.run(['git', 'init', '-b', 'main', str(root)], capture_output=True, check=True)
    subprocess.run(['git', '-C', str(root), 'config', 'user.email', 't@t.t'], capture_output=True, check=True)
    subprocess.run(['git', '-C', str(root), 'config', 'user.name', 'T'], capture_output=True, check=True)
    subprocess.run(['git', '-C', str(root), 'add', '-A'], capture_output=True, check=True)
    subprocess.run(['git', '-C', str(root), 'commit', '-m', 'init'], capture_output=True, check=True)

    yield str(root)


def scripted_coder() -> FunctionModel:
    """
    Build a coder model that plans, then edits and verifies a real file.

    Only the model is scripted. The tools it calls are the real ones, acting on
    a real repository and a real database.

    :returns: A model usable in place of a provider.
    """
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='1. Add a subtract function\nPLAN_COMPLETE')])

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        turn: int = sum(1 for message in messages if getattr(message, 'kind', '') == 'response')

        if turn == 0:
            yield 'I will add the function now. '
            yield {
                0: DeltaToolCall(
                    name='create_file',
                    json_args=json.dumps({
                        'path': 'subtract.py',
                        'content': 'def subtract(a: int, b: int) -> int:\n    return a - b\n',
                        'summary': 'add subtract helper',
                    }),
                )
            }
        elif turn == 1:
            yield 'Now verifying. '
            yield {0: DeltaToolCall(name='run_test_command', json_args=json.dumps({'command': 'true'}))}
        elif turn == 2:
            yield {
                0: DeltaToolCall(
                    name='complete_task',
                    json_args=json.dumps({'summary': 'Added subtract.py with a subtract function.'}),
                )
            }
        else:
            yield 'Finished.'

    return FunctionModel(respond, stream_function=stream)


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Replace only the pieces that need external services.

    The model, the embedding backend and the container runtime are substituted.
    Indexing, symbol extraction, chunking, Git, the database, the event bus, the
    message schema and the WebSocket itself are all exercised for real, because
    those are the layers where faults have actually hidden.

    :param monkeypatch: Fixture used to install the substitutes.
    """
    import api.v1.agent_session as agent_session_module
    from services.docker_sandbox import CommandResult, DockerSandboxManager
    from services.embedding import EmbeddingService

    model: FunctionModel = scripted_coder()
    # The resolver is a module level function now, so it is patched where the
    # session looks it up rather than on the class.
    monkeypatch.setattr(agent_session_module, 'resolve_model', lambda name: model)

    async def fake_embed(self: EmbeddingService, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def fake_dimensions(self: EmbeddingService) -> int:
        return 3

    monkeypatch.setattr(EmbeddingService, 'embed', fake_embed)
    monkeypatch.setattr(EmbeddingService, 'detect_dimensions', fake_dimensions)

    async def fake_spawn(self: DockerSandboxManager, session_id: str, **kwargs: Any) -> object:
        return object()

    async def fake_run(self: DockerSandboxManager, command: str) -> CommandResult:
        return CommandResult(exit_code=0, output='ok')

    async def fake_stop(self: DockerSandboxManager) -> None:
        return None

    monkeypatch.setattr(DockerSandboxManager, 'spawn_sandbox', fake_spawn)
    monkeypatch.setattr(DockerSandboxManager, 'run_test_command', fake_run)
    monkeypatch.setattr(DockerSandboxManager, 'stop_sandbox', fake_stop)

    from agents.reviewer import ReviewResult
    import agents.reviewer as reviewer_module

    async def fake_review(prompt: str, **kwargs: Any) -> Any:
        from unittest.mock import MagicMock

        result: Any = MagicMock()
        result.output = ReviewResult(approved=True, comments='The change looks correct.')
        return result

    monkeypatch.setattr(reviewer_module.reviewer_agent, 'run', fake_review)

    from agents.classifier import ClassificationResult
    import agents.classifier as classifier_module

    async def fake_classify(prompt: str, **kwargs: Any) -> Any:
        """
        Answer the classification the session now asks before planning.

        The scripted coder model cannot produce this structured output, and a
        session that cannot be classified refuses to start at all.
        """
        from unittest.mock import MagicMock

        result: Any = MagicMock()
        result.output = ClassificationResult(
            changes_are_required=True, reason='The task asks for a file to be created.'
        )
        return result

    monkeypatch.setattr(classifier_module.classifier_agent, 'run', fake_classify)


def drain(websocket: Any, until: set[str], limit: int = 200) -> list[dict[str, Any]]:
    """
    Read messages until one of the wanted types arrives.

    :param websocket: The connected test WebSocket.
    :param until: Message types that stop the read.
    :param limit: Safety bound on how many messages to read.
    :returns: Every message read, including the terminating one.
    :raises AssertionError: If the limit is reached before a wanted type appears.
    """
    collected: list[dict[str, Any]] = []
    for _ in range(limit):
        message: dict[str, Any] = websocket.receive_json()
        collected.append(message)
        if message.get('type') in until:
            return collected
    raise AssertionError(f'No message of type {until} arrived. Saw: {[m.get("type") for m in collected]}')


@pytest.mark.usefixtures('integration_database', 'wired')
def test_a_whole_session_runs_against_real_infrastructure(project: str) -> None:
    """
    Drive a session end to end over a real WebSocket against real services.

    Five separate faults have survived the unit suite because each test replaced
    precisely the component that was broken: ast-grep reading the working
    directory instead of the file, the sandbox being handed a session id as its
    image name, Tortoise's connection context being invisible to request
    handlers, missing CORS, and the sandbox bind mount using a path the host
    cannot resolve. This exercises those layers for real.
    """
    import main
    from database.models.code_embedding import CodeEmbedding
    from database.models.file import File

    with TestClient(main.app) as client:
        with client.websocket_connect('/api/v1/agent-session') as websocket:
            websocket.send_text(json.dumps({
                'prompt': 'Add a subtract function',
                'repo_path': project,
                'coder_model': 'scripted',
                'reviewer_model': 'scripted',
            }))

            # Indexing and planning.
            to_plan: list[dict[str, Any]] = drain(websocket, {'plan', 'error'})
            plan: dict[str, Any] = to_plan[-1]
            assert plan['type'] == 'plan', f'session failed: {plan}'
            # Steps are objects now; their titles are what this asserts on.
            assert [step['title'] for step in plan['steps']] == ['Add a subtract function']
            assert plan['requires_approval'] is True

            # Requirement 3 moved status announcements from `thought` to
            # `system`; the model's own reasoning is all that remains a thought.
            announced: str = ' '.join(
                m.get('content', '') for m in to_plan if m.get('type') == 'system'
            )
            assert 'Indexing' in announced
            assert not any(
                'Indexing' in m.get('content', '')
                for m in to_plan
                if m.get('type') == 'thought'
            )

            # Execution.
            websocket.send_text(json.dumps({'approved': True}))
            to_review: list[dict[str, Any]] = drain(websocket, {'review', 'error'})
            review: dict[str, Any] = to_review[-1]
            assert review['type'] == 'review', f'session failed: {review}'
            assert review['approved'] is True

            kinds: list[str] = [m.get('type', '') for m in to_review]
            assert 'action' in kinds, 'no tool action was reported'
            assert 'observation' in kinds, 'no tool observation was reported'
            assert 'commit' in kinds, 'no commit was reported'
            assert 'diff' in kinds, 'no diff was reported'

            actions: list[str] = [
                m['tool_name'] for m in to_review if m.get('type') == 'action'
            ]
            assert 'create_file' in actions
            assert 'complete_task' in actions

            diff: dict[str, Any] = next(m for m in to_review if m.get('type') == 'diff')
            assert diff['file_path'] == 'subtract.py'
            assert diff['original'] == '', 'a new file diffs against nothing'
            assert 'def subtract' in diff['modified']

            # Completion, then merge.
            websocket.send_text(json.dumps({'approved': True}))
            to_done: list[dict[str, Any]] = drain(websocket, {'done', 'error'})
            assert to_done[-1]['type'] == 'done', f'session failed: {to_done[-1]}'
            assert 'subtract' in to_done[-1]['summary']

            websocket.send_text(json.dumps({'approved': True}))
            merged: list[dict[str, Any]] = drain(websocket, {'done', 'error'})
            assert merged[-1]['type'] == 'done'
            assert 'Merged' in merged[-1]['summary']

    # The file really was written and committed on the real repository.
    assert (Path(project) / 'subtract.py').read_text().startswith('def subtract')
    log: str = subprocess.run(
        ['git', '-C', project, 'log', '--oneline'], capture_output=True, text=True
    ).stdout
    assert 'add subtract helper' in log

    branches: str = subprocess.run(
        ['git', '-C', project, 'branch'], capture_output=True, text=True
    ).stdout
    assert 'ai-task-' in branches


@pytest.mark.usefixtures('integration_database', 'wired')
def test_indexing_really_populated_the_database(project: str) -> None:
    """
    Indexing writes real files, symbols and embeddings, not stubs.
    """
    import asyncio

    import main
    from database.models.code_embedding import CodeEmbedding
    from database.models.file import File
    from database.models.symbol import Symbol
    from database.models.task import Task

    with TestClient(main.app) as client:
        with client.websocket_connect('/api/v1/agent-session') as websocket:
            websocket.send_text(json.dumps({
                'prompt': 'Add a subtract function',
                'repo_path': project,
            }))
            drain(websocket, {'plan', 'error'})

        async def inspect() -> tuple[int, int, int, int]:
            from tortoise import Tortoise

            from tests.conftest import postgres_config

            await Tortoise.close_connections()
            await Tortoise.init(config=postgres_config(dsn_for(INTEGRATION_DATABASE)))
            counts = (
                await File.all().count(),
                await Symbol.all().count(),
                await CodeEmbedding.all().count(),
                await Task.all().count(),
            )
            await Tortoise.close_connections()
            return counts

        files, symbols, embeddings, tasks = asyncio.run(inspect())

    assert files >= 1, 'the repository was not indexed'
    assert symbols >= 3, 'symbols were not extracted from real source'
    assert embeddings >= 1, 'no embeddings were stored'
    assert tasks == 1, 'the session did not register a task'
