from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient, MockTransport, Request, Response

import agents.coder as coder_module
from agents.coder import AgentDeps, coder_agent
from database.models.file import File
from services.code_search import CodeSearchService
from services.docker_sandbox import DockerSandboxManager
from services.event_bus import (
    AgentEvent,
    CommitEvent,
    DiffEvent,
    EventBus,
    ToolActionEvent,
    ToolObservationEvent,
    ToolStatus,
)
from services.file_editor import FileEditorService
from services.git_manager import GitManagerService
from services.session_context import SessionContext
from tests.conftest import git_log, stub_embedding_service

EXPECTED_TOOLS: set[str] = {
    'search_code',
    'search_code_lexical',
    'get_symbol',
    'read_file',
    'create_file',
    'edit_file',
    'run_test_command',
    'web_search',
    'complete_task',
}


async def invoke(tool_name: str, deps: AgentDeps, **kwargs: Any) -> str:
    """
    Call a registered tool's underlying function directly.

    :param tool_name: Name of the registered tool.
    :param deps: The dependencies to expose to the tool.
    :param kwargs: Arguments forwarded to the tool.
    :returns: The tool's response text.
    """
    context: Any = MagicMock()
    context.deps = deps
    result: Any = await coder_agent._function_toolset.tools[tool_name].function(context, **kwargs)
    return str(result)


@pytest.fixture
def deps(git_repo: GitManagerService, session: SessionContext, events: EventBus) -> AgentDeps:
    """
    Build agent dependencies backed by a real temporary git repository.

    :returns: The assembled dependencies.
    """
    root: str = git_repo.repo_path
    return AgentDeps(
        docker=DockerSandboxManager(repo_path=root),
        search=CodeSearchService(),
        editor=FileEditorService(root_path=root),
        git=git_repo,
        embedding=stub_embedding_service(),
        events=events,
        session=session,
    )


def test_all_tools_are_registered() -> None:
    """
    Every tool the system prompt refers to is registered on the agent.
    """
    assert set(coder_agent._function_toolset.tools) == EXPECTED_TOOLS


async def test_create_file_writes_commits_and_emits(deps: AgentDeps, events: EventBus) -> None:
    """
    Creating a file writes it, commits it, and reports both events.
    """
    seen: list[Any] = []

    async def collect(event: Any) -> None:
        seen.append(event)

    events.subscribe(collect)

    result: str = await invoke('create_file', deps, path='pkg/mod.py', content='def added():\n    return 1\n')

    root: str = deps.git.repo_path
    assert 'Created pkg/mod.py' in result
    assert Path(root, 'pkg/mod.py').read_text().startswith('def added():')
    assert 'pkg/mod.py' in deps.session.modified_files
    assert 'Create pkg/mod.py' in git_log(root)
    assert any(isinstance(event, ToolActionEvent) for event in seen)
    assert any(isinstance(event, ToolObservationEvent) for event in seen)


async def test_create_file_blocks_traversal(deps: AgentDeps) -> None:
    """
    Creating a file outside the repository is refused without raising.
    """
    assert 'Could not create' in await invoke('create_file', deps, path='../escape.py', content='x')


async def test_read_file_variants(deps: AgentDeps) -> None:
    """
    Reading returns full content, numbered ranges, and clear errors.
    """
    Path(deps.git.repo_path, 'a.py').write_text('one\ntwo\n')

    assert await invoke('read_file', deps, path='a.py') == 'one\ntwo\n'
    assert await invoke('read_file', deps, path='a.py', start_line=1, end_line=1) == '1: one'
    assert 'does not exist' in await invoke('read_file', deps, path='nope.py')
    assert 'Could not read' in await invoke('read_file', deps, path='a.py', start_line=99)
    assert 'Could not read' in await invoke('read_file', deps, path='../../etc/passwd')


async def test_edit_file_records_the_modification(deps: AgentDeps) -> None:
    """
    A successful edit is recorded so the completion gate can see it.
    """
    await invoke('create_file', deps, path='m.py', content='return 1\n')
    before: int = len(deps.session.modified_files)

    await invoke('edit_file', deps, path='m.py', search='return 1', replace='return 2')

    assert len(deps.session.modified_files) == before + 1
    assert 'return 2' in Path(deps.git.repo_path, 'm.py').read_text()


async def test_complete_task_gate(deps: AgentDeps) -> None:
    """
    Completion is refused until a change has been made and verified green.
    """
    assert 'rejected' in (await invoke('complete_task', deps, summary='done')).lower()

    await invoke('create_file', deps, path='m.py', content='x\n')
    assert 'not been verified' in await invoke('complete_task', deps, summary='done')

    deps.session.record_verification(False)
    assert 'verification command failed' in await invoke('complete_task', deps, summary='done')

    deps.session.record_verification(True)
    assert 'Task marked complete' in await invoke('complete_task', deps, summary='Added m.py.')
    assert deps.session.completion_requested is True


async def test_run_test_command_reports_the_verdict(deps: AgentDeps) -> None:
    """
    The tool states whether the command passed and records the outcome.
    """
    from tests.test_sandbox import attach_fake_container

    attach_fake_container(deps.docker, 0, b'2 passed')
    passing: str = await invoke('run_test_command', deps, command='pytest')
    assert 'Command passed' in passing
    assert deps.session.last_verification_passed is True

    attach_fake_container(deps.docker, 1, b'1 failed')
    failing: str = await invoke('run_test_command', deps, command='pytest')
    assert 'FAILED with exit code 1' in failing
    assert deps.session.last_verification_passed is False


@pytest.mark.usefixtures('sqlite_db')
async def test_lexical_tool_formats_results(deps: AgentDeps) -> None:
    """
    The lexical tool renders hits and reports cleanly when there are none.
    """
    await File.create(file_path='repo/users.py', raw_content='def find_user():\n    pass\n', hash='h')

    output: str = await invoke('search_code_lexical', deps, keyword='find_user')
    assert 'repo/users.py:' in output
    assert 'def find_user' in output

    assert 'No occurrences' in await invoke('search_code_lexical', deps, keyword='absent_xyz')


@pytest.mark.usefixtures('sqlite_db')
async def test_search_code_is_quiet_when_a_model_has_no_embeddings(deps: AgentDeps) -> None:
    """
    An unindexed model yields lexical results without a scary warning.

    No embeddings for the configured model is an ordinary state, not a fault.
    """
    await File.create(file_path='repo/users.py', raw_content='def find_user():\n    pass\n', hash='h')

    output: str = await invoke('search_code', deps, query='find_user')

    assert 'repo/users.py' in output
    assert '(lexical)' in output
    assert 'Semantic search is unavailable' not in output


@pytest.mark.usefixtures('sqlite_db')
async def test_search_code_degrades_visibly_when_semantic_fails(
    deps: AgentDeps, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A semantic backend failure still returns lexical hits, and says so.

    Silently returning lexical-only results would make a broken vector index
    look like a sparsely indexed repository.
    """
    await File.create(file_path='repo/users.py', raw_content='def find_user():\n    pass\n', hash='h')

    async def broken(*args: Any, **kwargs: Any) -> list[Any]:
        raise RuntimeError('vector backend is down')

    monkeypatch.setattr(deps.search, 'search_code_semantic', broken)
    output: str = await invoke('search_code', deps, query='find_user')

    assert 'repo/users.py' in output
    assert '(lexical)' in output
    assert 'Semantic search is unavailable' in output


def stub_searxng(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any] | None, status: int = 200
) -> None:
    """
    Point the web search tool at a mock SearXNG instance.

    :param monkeypatch: Fixture used to swap the HTTP client for the test's duration.
    :param payload: The JSON body to return, if any.
    :param status: The HTTP status code to return.
    """
    def handler(request: Request) -> Response:
        if payload is None:
            return Response(status, text='error')
        return Response(status, json=payload)

    def factory(*args: Any, **kwargs: Any) -> AsyncClient:
        return AsyncClient(transport=MockTransport(handler))

    monkeypatch.setattr(coder_module, 'AsyncClient', factory)


async def test_web_search_parses_and_limits_results(
    deps: AgentDeps, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Results are reduced to title, URL and a short snippet, capped at five.
    """
    payload: dict[str, Any] = {
        'number_of_results': 99,
        'results': [
            {'title': f'Result {index}', 'url': f'https://e.com/{index}',
             'content': 'C' * 400, 'engine': 'noise', 'score': 1.0}
            for index in range(9)
        ],
    }
    stub_searxng(monkeypatch, payload)
    output: str = await invoke('web_search', deps, query='anything')

    blocks: list[str] = output.split('\n\n')
    assert len(blocks) == 5
    assert 'Result 0' in output
    assert 'https://e.com/0' in output
    assert all(len(block.split('\n')[2]) <= 200 for block in blocks)
    assert 'number_of_results' not in output
    assert 'engine' not in output


@pytest.mark.parametrize(
    'payload,status,expected',
    [
        (None, 503, 'HTTP 503'),
        ({'results': []}, 200, 'No web results'),
    ],
    ids=['http_error', 'empty_results'],
)
async def test_web_search_failure_modes(
    deps: AgentDeps,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any] | None,
    status: int,
    expected: str,
) -> None:
    """
    Search failures are reported as readable text rather than raising.
    """
    stub_searxng(monkeypatch, payload, status)
    assert expected in await invoke('web_search', deps, query='q')


def collect(events: EventBus) -> list[AgentEvent]:
    """
    Subscribe a collector to the bus.

    :param events: The bus to observe.
    :returns: The list that received events are appended to.
    """
    seen: list[AgentEvent] = []

    async def handler(event: AgentEvent) -> None:
        seen.append(event)

    events.subscribe(handler)
    return seen


@pytest.mark.parametrize(
    'tool_name,kwargs',
    [
        ('read_file', {'path': 'README.md'}),
        ('get_symbol', {'symbol': 'Anything'}),
        ('complete_task', {'summary': 'nothing done'}),
    ],
    ids=['read_file', 'get_symbol', 'complete_task'],
)
@pytest.mark.usefixtures('sqlite_db')
async def test_every_tool_self_reports(
    deps: AgentDeps, events: EventBus, tool_name: str, kwargs: dict[str, Any]
) -> None:
    """
    Tools announce themselves through the bus rather than being parsed for.
    """
    seen: list[AgentEvent] = collect(events)
    await invoke(tool_name, deps, **kwargs)

    actions = [event for event in seen if isinstance(event, ToolActionEvent)]
    observations = [event for event in seen if isinstance(event, ToolObservationEvent)]
    assert [event.tool_name for event in actions] == [tool_name]
    assert [event.tool_name for event in observations] == [tool_name]


async def test_write_tools_never_report_file_contents(deps: AgentDeps, events: EventBus) -> None:
    """
    Bulky arguments are kept off the wire.

    Echoing whole files or search blocks back to the client would flood it for
    no benefit.
    """
    seen: list[AgentEvent] = collect(events)
    body: str = 'SECRET_BODY = 1\n' * 100

    await invoke('create_file', deps, path='big.py', content=body)
    await invoke('edit_file', deps, path='big.py', search='SECRET_BODY = 1', replace='X = 2')

    actions = [event for event in seen if isinstance(event, ToolActionEvent)]
    assert [event.arguments for event in actions] == [{'path': 'big.py'}, {'path': 'big.py'}]
    assert all('SECRET_BODY' not in str(event.arguments) for event in actions)


async def test_create_file_emits_commit_and_diff(deps: AgentDeps, events: EventBus) -> None:
    """
    Creating a file announces the commit and both sides of the diff.
    """
    seen: list[AgentEvent] = collect(events)
    await deps.git.create_branch('ai-task-diff')

    await invoke('create_file', deps, path='new.py', content='print(1)\n')

    commits = [event for event in seen if isinstance(event, CommitEvent)]
    diffs = [event for event in seen if isinstance(event, DiffEvent)]

    assert len(commits) == 1
    assert commits[0].commit_hash
    assert commits[0].message == 'Create new.py'

    assert len(diffs) == 1
    assert diffs[0].file_path == 'new.py'
    assert diffs[0].original == ''
    assert diffs[0].modified == 'print(1)\n'


async def test_edit_file_diff_uses_the_base_branch_version(
    deps: AgentDeps, events: EventBus
) -> None:
    """
    The original side of a diff comes from the branch the task started on.

    Reading it back from the working tree or the indexed copy would return the
    already-edited content, making every diff appear empty.
    """
    root: str = deps.git.repo_path
    Path(root, 'tracked.py').write_text('value = 1\n')
    await deps.git.commit_changes('Add tracked.py')
    await deps.git.create_branch('ai-task-diff')

    seen: list[AgentEvent] = collect(events)
    await invoke('edit_file', deps, path='tracked.py', search='value = 1', replace='value = 2')

    diffs = [event for event in seen if isinstance(event, DiffEvent)]
    assert len(diffs) == 1
    assert diffs[0].original == 'value = 1'
    assert diffs[0].modified == 'value = 2\n'


async def test_failed_edit_emits_no_commit_or_diff(deps: AgentDeps, events: EventBus) -> None:
    """
    A rejected edit reports its failure and nothing else.
    """
    Path(deps.git.repo_path, 'tracked.py').write_text('value = 1\n')
    await deps.git.commit_changes('Add tracked.py')
    await deps.git.create_branch('ai-task-diff')

    seen: list[AgentEvent] = collect(events)
    await invoke('edit_file', deps, path='tracked.py', search='not present', replace='x')

    assert [event for event in seen if isinstance(event, CommitEvent)] == []
    assert [event for event in seen if isinstance(event, DiffEvent)] == []
    observations = [event for event in seen if isinstance(event, ToolObservationEvent)]
    assert observations[0].status is ToolStatus.failure


async def test_failing_command_reports_a_failed_observation(
    deps: AgentDeps, events: EventBus
) -> None:
    """
    A non-zero exit is reported as a failed observation, not a success.
    """
    from tests.test_sandbox import attach_fake_container

    seen: list[AgentEvent] = collect(events)
    attach_fake_container(deps.docker, 1, b'1 failed')
    await invoke('run_test_command', deps, command='pytest')

    observations = [event for event in seen if isinstance(event, ToolObservationEvent)]
    assert observations[0].status is ToolStatus.failure


async def test_diff_original_is_captured_before_the_write(
    deps: AgentDeps, events: EventBus
) -> None:
    """
    The original side is read before the file changes, not after committing.

    When no separate task branch exists the commit lands on the base branch, so
    reading afterwards would return the edited content and every diff would be
    empty.
    """
    root: str = deps.git.repo_path
    Path(root, 'tracked.py').write_text('value = 1\n')
    await deps.git.commit_changes('Add tracked.py')

    # Deliberately no create_branch: base_branch is unset and falls back to the
    # branch currently checked out.
    assert deps.git.base_branch is None

    seen: list[AgentEvent] = collect(events)
    await invoke('edit_file', deps, path='tracked.py', search='value = 1', replace='value = 2')

    diffs = [event for event in seen if isinstance(event, DiffEvent)]
    assert len(diffs) == 1
    assert diffs[0].original == 'value = 1'
    assert diffs[0].modified == 'value = 2\n'


async def test_a_broken_base_branch_does_not_block_edits(
    deps: AgentDeps, events: EventBus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A diff that cannot be rendered must not prevent the edit from happening.

    No diff is emitted rather than a wrong one, so the viewer never shows a file
    as newly created when it was only modified.
    """
    root: str = deps.git.repo_path
    Path(root, 'tracked.py').write_text('value = 1\n')
    await deps.git.commit_changes('Add tracked.py')

    async def broken(branch: str, file_path: str) -> str | None:
        raise ValueError('that branch does not exist')

    monkeypatch.setattr(deps.git, 'get_file_at_branch', broken)
    seen: list[AgentEvent] = collect(events)

    result: str = await invoke(
        'edit_file', deps, path='tracked.py', search='value = 1', replace='value = 2'
    )

    assert 'Edited tracked.py' in result
    assert Path(root, 'tracked.py').read_text() == 'value = 2\n'
    assert [event for event in seen if isinstance(event, DiffEvent)] == []
    assert len([event for event in seen if isinstance(event, CommitEvent)]) == 1


async def test_new_file_still_gets_a_diff_with_an_empty_original(
    deps: AgentDeps, events: EventBus
) -> None:
    """
    A file absent from the base branch diffs against an empty original.

    This must stay distinguishable from the case where the base version could
    not be determined, which emits no diff at all.
    """
    await deps.git.create_branch('ai-task-new')
    seen: list[AgentEvent] = collect(events)

    await invoke('create_file', deps, path='brand_new.py', content='x = 1\n')

    diffs = [event for event in seen if isinstance(event, DiffEvent)]
    assert len(diffs) == 1
    assert diffs[0].original == ''
    assert diffs[0].modified == 'x = 1\n'


async def test_missing_sandbox_tells_the_agent_not_to_retry(deps: AgentDeps) -> None:
    """
    An unavailable sandbox is explained, not reported as a raw error.

    A stack trace fragment tells the agent nothing actionable, so it retries a
    command that can never succeed instead of adapting and saying so.
    """
    from services.docker_sandbox import CommandResult

    async def unavailable(command: str) -> CommandResult:
        raise RuntimeError('Error while fetching server API version')

    deps.docker.run_test_command = unavailable  # type: ignore[method-assign]

    result: str = await invoke('run_test_command', deps, command='pytest')

    assert 'sandbox is not available' in result
    assert 'Do not retry' in result
    assert 'could not be verified' in result
