from __future__ import annotations

import os

from collections.abc import Sequence

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient, MockTransport, Request, Response

import agents.coder_tools.search as coder_search_module
from agents.coder import AgentDeps, coder_agent
from database.models.file import File
from database.models.symbol import Symbol
from services.code_search import CodeSearchService
from services.lexical_match_mode import LexicalMatchMode
from services.docker_sandbox import DockerSandboxManager
from services.events import (
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
from services.indexing import IndexingService
from services.session_context import SessionContext
from services.user_questioner import UserQuestioner
from tests.conftest import TEST_PROJECT, TEST_REPO_PATH, git_log, stub_embedding_service

EXPECTED_TOOLS: set[str] = {
    'ask_user',
    'complete_step',
    'record_test_command',
    'forget_test_command',
    'skip_verification',
    'accept_step_without_changes',
    'search_code',
    'search_code_lexical',
    'get_symbol',
    'read_file',
    'list_files',
    'create_file',
    'edit_file',
    'delete_file',
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
        search=CodeSearchService(repo_path=TEST_REPO_PATH),
        editor=FileEditorService(root_path=root),
        git=git_repo,
        indexing=MagicMock(spec=IndexingService),
        embedding=stub_embedding_service(),
        events=events,
        session=session,
    )


def test_all_tools_are_registered() -> None:
    """
    Every tool the system prompt refers to is registered on the agent.
    """
    assert set(coder_agent._function_toolset.tools) == EXPECTED_TOOLS


async def test_create_file_writes_and_emits(deps: AgentDeps, events: EventBus) -> None:
    """
    Creating a file writes it and reports it, without committing.

    The commit happens when the step finishes, not per write: a commit per edit
    turned one step into a string of commits nobody chose the boundaries of.
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
    assert 'Create pkg/mod.py' not in git_log(root), 'the write should not have committed'
    indexing: MagicMock = cast(MagicMock, deps.indexing)
    indexing.index_file.assert_awaited_once_with('pkg/mod.py')
    # Kept for the step's eventual commit message.
    assert 'Create pkg/mod.py' in deps.session.pending_change_summaries
    assert any(isinstance(event, ToolActionEvent) for event in seen)
    assert any(isinstance(event, ToolObservationEvent) for event in seen)


async def test_create_file_blocks_traversal(deps: AgentDeps) -> None:
    """
    Creating a file outside the repository is refused without raising.
    """
    assert 'Could not create' in await invoke('create_file', deps, path='../escape.py', content='x')


async def test_create_file_refuses_an_empty_path(deps: AgentDeps) -> None:
    """
    An empty path is refused with an explanation, never written to the root.
    """
    result: str = await invoke('create_file', deps, path='   ', content='x')

    assert 'no path was given' in result
    assert deps.session.modified_files == ()


async def test_edit_file_refuses_an_empty_path_or_search(deps: AgentDeps) -> None:
    """
    An empty path or search block is refused rather than raising or misbehaving.
    """
    no_path: str = await invoke('edit_file', deps, path='  ', search='a', replace='b')
    no_search: str = await invoke('edit_file', deps, path='m.py', search=' ', replace='b')

    assert 'no path was given' in no_path
    assert 'search block is required' in no_search


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


async def test_indexing_failure_is_reported_after_a_write(deps: AgentDeps) -> None:
    """
    A disk write remains accounted for, but never looks fully successful when its index fails.
    """
    indexing: MagicMock = cast(MagicMock, deps.indexing)
    indexing.index_file.side_effect = RuntimeError('embedding backend unavailable')

    result: str = await invoke('create_file', deps, path='indexed.py', content='value = 1\n')

    assert 'could not refresh its index' in result
    assert Path(deps.git.repo_path, 'indexed.py').exists()
    assert 'indexed.py' in deps.session.modified_files


@pytest.mark.usefixtures('db')
async def test_create_and_edit_are_immediately_visible_to_indexed_searches(
    deps: AgentDeps,
) -> None:
    """
    Create and edit refresh lexical and symbol searches before their tools return.
    """
    root: str = deps.git.repo_path
    deps.search.repo_path = root
    deps.indexing = IndexingService(repo_path=root, embedding=deps.embedding)

    await invoke(
        'create_file',
        deps,
        path='fresh.py',
        content='def stale_marker():\n    return 1\n',
    )
    assert 'stale_marker' in await invoke('search_code_lexical', deps, keywords=['stale_marker'])
    assert 'stale_marker' in await invoke('get_symbol', deps, symbol='stale_marker')

    await invoke(
        'edit_file',
        deps,
        path='fresh.py',
        search='stale_marker',
        replace='fresh_marker',
    )
    assert 'No occurrences' in await invoke('search_code_lexical', deps, keywords=['stale_marker'])
    assert 'fresh_marker' in await invoke('search_code_lexical', deps, keywords=['fresh_marker'])
    assert 'No definition' in await invoke('get_symbol', deps, symbol='stale_marker')
    assert 'fresh_marker' in await invoke('get_symbol', deps, symbol='fresh_marker')


@pytest.mark.usefixtures('db')
async def test_delete_file_removes_disk_index_and_records_a_deletion_diff(
    deps: AgentDeps, events: EventBus
) -> None:
    """
    A deletion removes its indexed rows, counts as a modification, and emits an empty new side.
    """
    root: str = deps.git.repo_path
    Path(root, 'obsolete.py').write_text('def obsolete_marker():\n    return 1\n')
    await deps.git.commit_changes('Add obsolete.py')
    deps.search.repo_path = root
    deps.indexing = IndexingService(repo_path=root, embedding=deps.embedding)
    await deps.indexing.index_file('obsolete.py')
    seen: list[AgentEvent] = collect(events)

    result: str = await invoke('delete_file', deps, path='obsolete.py', summary='remove obsolete marker')

    assert 'Deleted obsolete.py' in result
    assert not Path(root, 'obsolete.py').exists()
    assert 'obsolete.py' in deps.session.modified_files
    assert 'Delete obsolete.py: remove obsolete marker' in deps.session.pending_change_summaries
    assert 'No occurrences' in await invoke(
        'search_code_lexical', deps, keywords=['obsolete_marker']
    )
    assert 'No definition' in await invoke('get_symbol', deps, symbol='obsolete_marker')
    diffs: list[DiffEvent] = [event for event in seen if isinstance(event, DiffEvent)]
    assert len(diffs) == 1
    assert diffs[0].original == 'def obsolete_marker():\n    return 1'
    assert diffs[0].modified == ''


async def test_delete_file_refuses_empty_traversal_and_missing_paths(deps: AgentDeps) -> None:
    """
    Unsafe or nonexistent deletion requests are rejected without changing the session.
    """
    empty: str = await invoke('delete_file', deps, path='   ')
    traversal: str = await invoke('delete_file', deps, path='../escape.py')
    missing: str = await invoke('delete_file', deps, path='missing.py')

    assert 'no path was given' in empty
    assert 'outside of root' in traversal
    assert 'does not exist' in missing
    assert deps.session.modified_files == ()


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


@pytest.mark.parametrize('command', ['', '   '], ids=['empty', 'blank'])
async def test_run_test_command_refuses_an_empty_command(deps: AgentDeps, command: str) -> None:
    """
    An empty command is refused without reaching the sandbox.
    """
    from services.command_result import CommandResult

    ran: list[str] = []

    async def record(value: str) -> CommandResult:
        ran.append(value)
        return CommandResult(exit_code=0, output='')

    setattr(deps.docker, 'run_test_command', record)

    result: str = await invoke('run_test_command', deps, command=command)

    assert ran == [], 'the empty command should not have reached the sandbox'
    assert 'No command was given' in result


async def test_playwright_is_refused_before_reaching_the_sandbox(deps: AgentDeps) -> None:
    """
    The command never runs, and the refusal points at `agent-browser`.

    Refused before execution rather than after: installing a browser stack is
    slow and leaves the sandbox changed, so running it and complaining
    afterwards would achieve neither.
    """
    from services.command_result import CommandResult

    ran: list[str] = []

    async def record(command: str) -> CommandResult:
        ran.append(command)
        return CommandResult(exit_code=0, output='')

    deps.docker.run_test_command = record  # type: ignore[method-assign]

    result: str = await invoke('run_test_command', deps, command='npx playwright install')

    assert ran == [], 'the command should not have reached the sandbox'
    assert 'agent-browser' in result
    assert 'was not run' in result


async def test_a_refused_command_is_not_recorded_as_a_verification(deps: AgentDeps) -> None:
    """
    Nothing ran, so nothing was verified.

    Recording it either way would corrupt the completion gate: a pass would let
    the agent finish on the strength of a command that never executed.
    """
    before: bool | None = deps.session.last_verification_passed

    await invoke('run_test_command', deps, command='pip install selenium')

    assert deps.session.last_verification_passed is before


async def test_agent_browser_itself_reaches_the_sandbox(deps: AgentDeps) -> None:
    """
    The tool the agent is being pointed at must not be caught by the guard.
    """
    from tests.test_sandbox import attach_fake_container

    attach_fake_container(deps.docker, 0, b'installed')

    result: str = await invoke(
        'run_test_command', deps, command='npx --yes agent-browser install'
    )

    assert 'Command passed' in result


@pytest.mark.usefixtures('db')
async def test_lexical_tool_formats_results(deps: AgentDeps) -> None:
    """
    The lexical tool renders hits and reports cleanly when there are none.
    """
    await File.create(project=TEST_PROJECT, file_path='repo/users.py', raw_content='def find_user():\n    pass\n', hash='h')

    output: str = await invoke('search_code_lexical', deps, keywords=['find_user'])
    assert 'repo/users.py:' in output
    assert 'def find_user' in output

    assert 'No occurrences' in await invoke(
        'search_code_lexical', deps, keywords=['absent_xyz']
    )


@pytest.mark.usefixtures('db')
async def test_lexical_tool_searches_several_keywords_at_once(deps: AgentDeps) -> None:
    """
    Three unrelated names in one call return matches for each.

    This is the reported failure that prompted the capability: the agent wanted
    three symbols at once, and a single joined string found nothing because no
    line contained all three.
    """
    await File.create(project=TEST_PROJECT, file_path='repo/a.py', raw_content='def alpha():\n    pass\n', hash='a')
    await File.create(project=TEST_PROJECT, file_path='repo/b.py', raw_content='def beta():\n    pass\n', hash='b')
    await File.create(project=TEST_PROJECT, file_path='repo/c.py', raw_content='def gamma():\n    pass\n', hash='c')

    output: str = await invoke(
        'search_code_lexical', deps, keywords=['alpha', 'beta', 'gamma']
    )

    assert 'def alpha' in output
    assert 'def beta' in output
    assert 'def gamma' in output


@pytest.mark.usefixtures('db')
async def test_lexical_tool_all_mode_requires_every_keyword_on_one_line(
    deps: AgentDeps,
) -> None:
    """
    In *all* mode a line needs every keyword, and the failure suggests *any*.
    """
    await File.create(
        project=TEST_PROJECT,
        file_path='repo/mixed.py',
        raw_content='alpha_and_beta_together = 1\nalpha_alone = 2\n',
        hash='m',
    )

    both: str = await invoke(
        'search_code_lexical', deps, keywords=['alpha', 'beta'], mode=LexicalMatchMode.all_of
    )
    # Asserted by line number: the non-matching line still appears in the
    # matching line's context, so searching the text for it proves nothing.
    assert 'repo/mixed.py:1' in both
    assert 'repo/mixed.py:2' not in both

    missing: str = await invoke(
        'search_code_lexical', deps, keywords=['alpha', 'zzz_absent'],
        mode=LexicalMatchMode.all_of,
    )
    assert 'No single line contains all of' in missing
    # The other mode is worth suggesting unprompted; *all* fails often.
    assert 'mode="any"' in missing


@pytest.mark.usefixtures('db')
async def test_lexical_tool_names_the_keywords_it_tried(deps: AgentDeps) -> None:
    """
    A fruitless search says what was searched for, not merely that it failed.
    """
    output: str = await invoke(
        'search_code_lexical', deps, keywords=['absent_one', 'absent_two']
    )

    assert 'absent_one' in output
    assert 'absent_two' in output


@pytest.mark.usefixtures('db')
@pytest.mark.parametrize(
    'keywords',
    [[], ['   '], ['', '  ']],
    ids=['no_keywords', 'one_blank', 'all_blank'],
)
async def test_lexical_tool_refuses_an_empty_keyword_list(
    deps: AgentDeps, keywords: list[str]
) -> None:
    """
    Nothing usable is refused with an explanation, not searched for.

    Searching for an empty string would match every line in the repository,
    which is the opposite of what the caller wanted.
    """
    await File.create(project=TEST_PROJECT, file_path='repo/users.py', raw_content='def find_user():\n', hash='h')

    output: str = await invoke('search_code_lexical', deps, keywords=keywords)

    assert 'No keywords were given' in output
    assert 'find_user' not in output


@pytest.mark.usefixtures('db')
async def test_lexical_tool_reports_keywords_beyond_the_cap(deps: AgentDeps) -> None:
    """
    Surplus keywords are named back, not dropped in silence.

    An agent told nothing would read the absence of a term as proof it is not in
    the repository, when in fact it was never looked for.
    """
    await File.create(project=TEST_PROJECT, file_path='repo/users.py', raw_content='def find_user():\n', hash='h')

    surplus: list[str] = [f'extra_{index}' for index in range(3)]
    output: str = await invoke(
        'search_code_lexical',
        deps,
        keywords=['find_user', *[f'absent_{index}' for index in range(9)], *surplus],
    )

    for keyword in surplus:
        assert keyword in output

    # The results that were found still come back alongside the notice.
    assert 'def find_user' in output


@pytest.mark.usefixtures('db')
async def test_lexical_tool_is_quiet_when_nothing_was_dropped(deps: AgentDeps) -> None:
    """
    A call within the cap carries no notice about surplus keywords.
    """
    await File.create(project=TEST_PROJECT, file_path='repo/users.py', raw_content='def find_user():\n', hash='h')

    output: str = await invoke('search_code_lexical', deps, keywords=['find_user'])

    assert 'were not' not in output
    assert 'batches' not in output


@pytest.mark.usefixtures('db')
async def test_search_code_is_quiet_when_a_model_has_no_embeddings(deps: AgentDeps) -> None:
    """
    An unindexed model yields lexical results without a scary warning.

    No embeddings for the configured model is an ordinary state, not a fault.
    """
    await File.create(project=TEST_PROJECT, file_path='repo/users.py', raw_content='def find_user():\n    pass\n', hash='h')

    output: str = await invoke('search_code', deps, query='find_user')

    assert 'repo/users.py' in output
    assert '(lexical)' in output
    assert 'Semantic search is unavailable' not in output


@pytest.mark.usefixtures('db')
async def test_search_code_degrades_visibly_when_semantic_fails(
    deps: AgentDeps, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A semantic backend failure still returns lexical hits, and says so.

    Silently returning lexical-only results would make a broken vector index
    look like a sparsely indexed repository.
    """
    await File.create(project=TEST_PROJECT, file_path='repo/users.py', raw_content='def find_user():\n    pass\n', hash='h')

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

    monkeypatch.setattr(coder_search_module, 'AsyncClient', factory)


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


@pytest.mark.parametrize('query', ['', '   '], ids=['empty', 'blank'])
async def test_web_search_refuses_an_empty_query(
    deps: AgentDeps, monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    """
    An empty query is refused without ever reaching SearXNG.
    """
    stub_searxng(monkeypatch, None, 200)
    result: str = await invoke('web_search', deps, query=query)

    assert 'No query was given' in result


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
@pytest.mark.usefixtures('db')
async def test_every_tool_self_reports(
    deps: AgentDeps,
    events: EventBus,
    tool_name: str,
    kwargs: dict[str, Any],
) -> None:
    """
    Tools announce themselves through the bus rather than being parsed for.
    """
    # `complete_task` with nothing modified is refused, but a refusal is still
    # reported: what is under test is that the tool announces itself either way.
    deps.session.record_classification(True)
    seen: list[AgentEvent] = collect(events)
    await invoke(tool_name, deps, **kwargs)

    actions = [event for event in seen if isinstance(event, ToolActionEvent)]
    observations = [event for event in seen if isinstance(event, ToolObservationEvent)]
    assert [event.tool_name for event in actions] == [tool_name]
    assert [event.tool_name for event in observations] == [tool_name]


async def observation_for(
    tool_name: str, deps: AgentDeps, events: EventBus, **kwargs: Any
) -> tuple[str, str]:
    """
    Invoke a tool and capture both of its outputs.

    :param tool_name: The tool to call.
    :param deps: The dependencies to expose to it.
    :param events: The bus the observation is emitted on.
    :param kwargs: Arguments forwarded to the tool.
    :returns: What the agent received, and what the reader was shown.
    """
    seen: list[AgentEvent] = collect(events)
    returned: str = await invoke(tool_name, deps, **kwargs)
    observations = [event for event in seen if isinstance(event, ToolObservationEvent)]

    assert len(observations) == 1
    return returned, observations[0].result


@pytest.mark.usefixtures('db')
async def test_search_code_shows_paths_to_the_reader_and_code_to_the_agent(
    deps: AgentDeps, events: EventBus
) -> None:
    """
    The two outputs must diverge: paths for the feed, snippets for the agent.

    Both halves are asserted deliberately. A test that only checked the
    observation had no code in it would pass just as well if the change had
    gutted the tool's result too, which is precisely what must not happen.
    """
    await File.create(
        project=TEST_PROJECT,
        file_path='repo/users.py',
        raw_content='def find_user():\n    SECRET_MARKER = 1\n    return SECRET_MARKER\n',
        hash='h',
    )

    returned, observation = await observation_for(
        'search_code', deps, events, query='find_user'
    )

    # The agent still gets the code.
    assert 'SECRET_MARKER' in returned
    assert 'def find_user' in returned

    # The reader gets the file, and none of its contents.
    assert 'repo/users.py' in observation
    assert 'SECRET_MARKER' not in observation
    assert 'def find_user' not in observation


@pytest.mark.usefixtures('db')
@pytest.mark.parametrize('query', ['', '   '], ids=['empty', 'blank'])
async def test_search_code_refuses_an_empty_query(
    deps: AgentDeps, events: EventBus, query: str
) -> None:
    """
    An empty semantic query is refused with an explanation, never embedded.

    Embedding an empty string raises `EmbeddingError`; letting that escape would
    abort the whole session instead of telling the agent what to fix.
    """
    returned, observation = await observation_for(
        'search_code', deps, events, query=query
    )

    assert 'No query was given' in returned
    assert 'No files matched' in observation


@pytest.mark.usefixtures('db')
async def test_lexical_search_shows_paths_to_the_reader_and_code_to_the_agent(
    deps: AgentDeps, events: EventBus
) -> None:
    """
    The lexical tool renders context lines, so it floods the feed just as badly.
    """
    await File.create(
        project=TEST_PROJECT,
        file_path='repo/users.py',
        raw_content='def find_user():\n    SECRET_MARKER = 1\n    return SECRET_MARKER\n',
        hash='h',
    )

    returned, observation = await observation_for(
        'search_code_lexical', deps, events, keywords=['find_user']
    )

    assert 'SECRET_MARKER' in returned
    assert 'repo/users.py' in observation
    assert 'SECRET_MARKER' not in observation


@pytest.mark.usefixtures('db')
async def test_a_file_matching_many_times_is_shown_once(
    deps: AgentDeps, events: EventBus
) -> None:
    """
    Nine matching lines in one file is one file to the reader.
    """
    await File.create(
        project=TEST_PROJECT,
        file_path='repo/repeats.py',
        raw_content='\n'.join(f'target_{index} = target' for index in range(9)),
        hash='h',
    )

    _, observation = await observation_for(
        'search_code_lexical', deps, events, keywords=['target']
    )

    assert observation.count('repo/repeats.py') == 1
    assert 'Matched 1 file' in observation


@pytest.mark.usefixtures('db')
async def test_a_search_matching_nothing_says_so(deps: AgentDeps, events: EventBus) -> None:
    """
    An empty result reads as a sentence, not as an empty list.
    """
    _, observation = await observation_for(
        'search_code', deps, events, query='absent_xyz'
    )

    assert 'No files matched' in observation
    assert observation.strip() != ''


@pytest.mark.usefixtures('db')
async def test_a_half_failed_search_says_so_in_the_feed(
    deps: AgentDeps, events: EventBus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A search whose semantic half died must not read as a complete one.

    Showing only 'Matched 1 file' would let a broken vector index look like a
    thin repository, which is the mistake this notice exists to prevent.
    """
    await File.create(project=TEST_PROJECT, file_path='repo/users.py', raw_content='def find_user():\n', hash='h')

    async def explode(*_: Any, **__: Any) -> list[Any]:
        raise RuntimeError('vector backend down')

    monkeypatch.setattr(deps.search, 'search_code_semantic', explode)

    _, observation = await observation_for(
        'search_code', deps, events, query='find_user'
    )

    assert 'Semantic search is unavailable' in observation
    # The caveat has to arrive before the count, or it reads as a footnote.
    assert observation.index('Semantic search is unavailable') < observation.index('Matched')



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


async def test_create_file_emits_a_diff_but_no_commit(
    deps: AgentDeps, events: EventBus
) -> None:
    """
    Creating a file announces the diff and nothing about a commit.

    The diff is per edit because the live viewer shows each edit as it lands,
    which has nothing to do with commit cadence.
    """
    seen: list[AgentEvent] = collect(events)
    await deps.git.create_branch('ai-task-diff')

    await invoke('create_file', deps, path='new.py', content='print(1)\n')

    commits = [event for event in seen if isinstance(event, CommitEvent)]
    diffs = [event for event in seen if isinstance(event, DiffEvent)]

    assert commits == []

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
    assert [event for event in seen if isinstance(event, CommitEvent)] == []


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
    from services.command_result import CommandResult

    async def unavailable(command: str) -> CommandResult:
        raise RuntimeError('Error while fetching server API version')

    deps.docker.run_test_command = unavailable  # type: ignore[method-assign]

    result: str = await invoke('run_test_command', deps, command='pytest')

    assert 'sandbox is not available' in result
    assert 'Do not retry' in result
    assert 'could not be verified' in result


async def test_a_task_classified_as_needing_no_changes_completes(deps: AgentDeps) -> None:
    """
    A task that only ever called for an answer is allowed to finish.

    Requirement 2.4: the classification was made before the work began, so the
    gate reads it rather than asking anything.
    """
    deps.session.record_classification(False)

    result: str = await invoke('complete_task', deps, summary='It is a finite state machine.')

    assert 'no file changes' in result
    assert deps.session.completion_requested is True
    assert deps.session.completed_without_changes is True


async def test_a_task_classified_as_needing_changes_is_refused_when_none_were_made(
    deps: AgentDeps,
) -> None:
    """
    Requirement 2.10: producing nothing is still a refusal, and says why.
    """
    deps.session.record_classification(True)

    result: str = await invoke('complete_task', deps, summary='I had a look and it seems fine.')

    assert 'rejected' in result.lower()
    assert 'judged to require changing at least one file' in result
    assert deps.session.completion_requested is False


async def test_an_unclassified_session_is_refused_rather_than_excused(
    deps: AgentDeps,
) -> None:
    """
    A missing classification must not become a free pass.

    Nothing should reach the gate unclassified, but if anything ever does, the
    safe reading is the strict one: treating the absence as "no changes needed"
    would let any task finish having done nothing at all.
    """
    assert deps.session.changes_are_required is None

    result: str = await invoke('complete_task', deps, summary='Nothing to do here.')

    assert 'rejected' in result.lower()
    assert deps.session.completion_requested is False


async def test_the_completion_gate_asks_nothing_of_a_subagent(
    deps: AgentDeps, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Requirement 2.9: the question is settled before the task starts.

    Asserted by making any use of the subagent an error. A gate that consulted
    it would be re-litigating a decision already made, and could contradict it.
    """
    import agents.classifier as classifier_module

    async def refuse(prompt: str, **kwargs: Any) -> Any:
        raise AssertionError('the classifier was consulted at completion time')

    monkeypatch.setattr(classifier_module.classifier_agent, 'run', refuse)
    deps.session.record_classification(False)

    assert 'no file changes' in await invoke('complete_task', deps, summary='An answer.')


async def test_the_ordinary_completion_gate_is_unchanged(deps: AgentDeps) -> None:
    """
    Requirement 2.3: a task that changes files finishes exactly as it did.
    """
    deps.session.record_classification(True)

    await invoke('create_file', deps, path='m.py', content='x\n')
    assert 'not been verified' in await invoke('complete_task', deps, summary='done')

    deps.session.record_verification(False)
    assert 'verification command failed' in await invoke('complete_task', deps, summary='done')

    deps.session.record_verification(True)
    assert 'Task marked complete' in await invoke('complete_task', deps, summary='Added m.py.')
    assert deps.session.completion_requested is True


async def test_an_empty_answer_is_refused_even_when_no_changes_were_needed(
    deps: AgentDeps, events: EventBus
) -> None:
    """
    A task that produced no answer has not been completed, whatever the verdict.

    The refusal is reported back to the agent instead of being allowed to raise
    and fail the session, the same way every tool failure is surfaced.
    """
    deps.session.record_classification(False)
    seen: list[AgentEvent] = collect(events)

    result: str = await invoke('complete_task', deps, summary='   ')

    observations = [event for event in seen if isinstance(event, ToolObservationEvent)]
    assert 'rejected' in result
    assert 'summary is required' in result
    assert len(observations) == 1
    assert observations[0].status is ToolStatus.failure


async def test_a_file_read_is_reported_as_one_line(
    deps: AgentDeps, events: EventBus, git_repo: GitManagerService
) -> None:
    """
    Requirement 13.1: the observation names the file rather than repeating it.
    """
    Path(git_repo.repo_path, 'big.py').write_text(
        '\n'.join(f'secret line {index}' for index in range(300)) + '\n'
    )
    seen: list[AgentEvent] = collect(events)

    await invoke('read_file', deps, path='big.py')

    observations = [event for event in seen if isinstance(event, ToolObservationEvent)]
    assert len(observations) == 1
    assert observations[0].result == 'Read big.py (300 lines).'
    assert 'secret line' not in observations[0].result


async def test_the_agent_still_receives_the_file(
    deps: AgentDeps, git_repo: GitManagerService
) -> None:
    """
    Requirement 13.2: shortening the observation must not shorten the tool.

    Handing the agent a one-line confirmation instead of the file would make the
    tool useless while looking, from the outside, exactly the same.
    """
    Path(git_repo.repo_path, 'a.py').write_text('first\nsecond\nthird\n')

    returned: str = await invoke('read_file', deps, path='a.py')

    assert 'first' in returned
    assert 'second' in returned
    assert 'third' in returned


async def test_a_partial_read_reports_its_range(
    deps: AgentDeps, events: EventBus, git_repo: GitManagerService
) -> None:
    """
    Reading part of a file is a different act from reading all of it.
    """
    Path(git_repo.repo_path, 'a.py').write_text('one\ntwo\nthree\nfour\n')
    seen: list[AgentEvent] = collect(events)

    await invoke('read_file', deps, path='a.py', start_line=2, end_line=3)

    observations = [event for event in seen if isinstance(event, ToolObservationEvent)]
    assert 'lines 2 to 3' in observations[0].result


async def test_a_failed_read_still_says_why(deps: AgentDeps, events: EventBus) -> None:
    """
    Requirement 13.3: a failure reports the reason, not a bare confirmation.
    """
    seen: list[AgentEvent] = collect(events)

    await invoke('read_file', deps, path='nowhere.py')

    observations = [event for event in seen if isinstance(event, ToolObservationEvent)]
    assert observations[0].status is ToolStatus.failure
    assert 'does not exist' in observations[0].result


@pytest.mark.usefixtures('db')
async def test_get_symbol_returns_the_definition_itself(deps: AgentDeps) -> None:
    """
    The agent asks for a definition and must receive one.

    This previously returned `str(results)` on a list of database rows, which
    renders as their default representation, so the agent learned the primary
    keys and nothing else. The tool advertised something it did not do, and
    failed silently: the agent would simply find it useless.
    """
    file_obj = await File.create(
        project=TEST_PROJECT,
        file_path='repo/users.py',
        raw_content='import os\n\n\ndef find_user(name):\n    return name\n',
        hash='h',
    )
    await Symbol.create(
        file=file_obj, symbol_name='find_user', kind='function', line_start=4, line_end=5
    )

    returned: str = await invoke('get_symbol', deps, symbol='find_user')

    assert 'def find_user(name):' in returned
    assert 'return name' in returned
    assert 'repo/users.py:4-5' in returned
    # The shape of the old defect: an object repr rather than source.
    assert '<Symbol' not in returned
    assert 'object at 0x' not in returned


@pytest.mark.usefixtures('db')
async def test_get_symbol_reports_every_declaration(deps: AgentDeps) -> None:
    """
    A name declared twice is worth seeing twice, not resolved arbitrarily.
    """
    for path in ('repo/a.py', 'repo/b.py'):
        file_obj = await File.create(
            project=TEST_PROJECT,
            file_path=path, raw_content='def handler():\n    pass\n', hash='h'
        )
        await Symbol.create(
            file=file_obj, symbol_name='handler', kind='function', line_start=1, line_end=2
        )

    returned: str = await invoke('get_symbol', deps, symbol='handler')

    assert 'repo/a.py:1-2' in returned
    assert 'repo/b.py:1-2' in returned


@pytest.mark.usefixtures('db')
async def test_get_symbol_says_plainly_when_nothing_is_indexed(deps: AgentDeps) -> None:
    """
    An empty result must read as an answer, not as a malfunction.
    """
    returned: str = await invoke('get_symbol', deps, symbol='nowhere')

    assert 'No definition' in returned
    assert 'nowhere' in returned


@pytest.mark.parametrize('symbol', ['', '   '], ids=['empty', 'blank'])
async def test_get_symbol_refuses_an_empty_symbol(deps: AgentDeps, symbol: str) -> None:
    """
    An empty symbol is refused with an explanation rather than a lookup.
    """
    result: str = await invoke('get_symbol', deps, symbol=symbol)

    assert 'No symbol was given' in result


@pytest.mark.usefixtures('db')
async def test_get_symbol_reports_a_count_rather_than_the_source(
    deps: AgentDeps, events: EventBus
) -> None:
    """
    The definitions are for the agent; the reader wants to know it was found.
    """
    file_obj = await File.create(
        project=TEST_PROJECT,
        file_path='repo/users.py',
        raw_content='def find_user(name):\n    return name\n',
        hash='h',
    )
    await Symbol.create(
        file=file_obj, symbol_name='find_user', kind='function', line_start=1, line_end=2
    )
    seen: list[AgentEvent] = collect(events)

    await invoke('get_symbol', deps, symbol='find_user')

    observations = [event for event in seen if isinstance(event, ToolObservationEvent)]
    assert observations[0].result == 'Found the definition of find_user.'
    assert 'def find_user' not in observations[0].result


def questioner_answering(reply: str) -> tuple[UserQuestioner, list[tuple[str, list[str]]]]:
    """
    Build a questioner that answers immediately, recording what it was asked.

    :param reply: The answer the user gives.
    :returns: The questioner and the list its questions are recorded in.
    """
    asked: list[tuple[str, list[str]]] = []

    async def emit(question: str, options: Sequence[str]) -> None:
        asked.append((question, list(options)))

    async def answer() -> str:
        return reply

    return UserQuestioner(emit_question=emit, await_reply=answer), asked


async def test_ask_user_returns_the_reply_to_the_agent(deps: AgentDeps) -> None:
    """
    Requirement 4.1: the tool's result is what the user said.
    """
    deps.questioner, asked = questioner_answering('Use the second one.')

    result: str = await invoke(
        'ask_user', deps, question='Which retry policy?', options=['The first', 'The second']
    )

    assert 'Use the second one.' in result
    assert asked == [('Which retry policy?', ['The first', 'The second'])]


async def test_ask_user_accepts_a_question_with_no_suggestions(deps: AgentDeps) -> None:
    """
    Requirement 4.7: the free-reply option alone is a usable interface, so a
    question offering nothing else is still worth asking.
    """
    deps.questioner, asked = questioner_answering('Whatever you think.')

    result: str = await invoke('ask_user', deps, question='How should this behave?')

    assert 'Whatever you think.' in result
    assert asked == [('How should this behave?', [])]


async def test_ask_user_truncates_rather_than_refusing(deps: AgentDeps) -> None:
    """
    Requirement 4.6a: extra options cost the agent nothing but a correction.

    Refusing the call would lose the agent its turn over something cosmetic,
    and the question would have been perfectly usable without the extras.
    """
    deps.questioner, asked = questioner_answering('The third.')

    result: str = await invoke(
        'ask_user',
        deps,
        question='Which one?',
        options=[f'Option {index}' for index in range(1, 10)],
    )

    assert asked[0][1] == [f'Option {index}' for index in range(1, 7)]
    assert 'The third.' in result
    # The agent has to learn, or it will send nine again next time.
    assert 'were dropped' in result


async def test_ask_user_says_nothing_was_dropped_when_nothing_was(deps: AgentDeps) -> None:
    """
    The correction must not appear on a question that did not need it.
    """
    deps.questioner, _ = questioner_answering('The first.')

    result: str = await invoke('ask_user', deps, question='Which?', options=['a', 'b'])

    assert 'were dropped' not in result


async def test_ask_user_refuses_an_empty_question(deps: AgentDeps) -> None:
    """
    An empty question puts nothing in front of the user to answer.
    """
    deps.questioner, asked = questioner_answering('...')

    result: str = await invoke('ask_user', deps, question='   ')

    assert 'rejected' in result.lower()
    assert asked == []


async def test_ask_user_says_so_when_there_is_no_one_to_ask(deps: AgentDeps) -> None:
    """
    The tools are also driven with no session behind them.

    Hanging would be the worst outcome; so would pretending an answer.
    """
    assert deps.questioner is None

    result: str = await invoke('ask_user', deps, question='Which one?')

    assert 'no user to ask' in result


async def test_ask_user_reports_itself_like_every_other_tool(
    deps: AgentDeps, events: EventBus
) -> None:
    """
    The question is a tool call, and the feed should show it as one.
    """
    deps.questioner, _ = questioner_answering('Yes.')
    seen: list[AgentEvent] = collect(events)

    await invoke('ask_user', deps, question='Shall I?')

    actions = [event for event in seen if isinstance(event, ToolActionEvent)]
    assert [event.tool_name for event in actions] == ['ask_user']


async def test_list_files_names_its_argument_like_every_other_file_tool(
    deps: AgentDeps,
) -> None:
    """
    The tools share a vocabulary, or the model spends its retries learning it.

    `list_files` once took `subdirectory` while `read_file`, `create_file` and
    `edit_file` all took `path`. A model calling it the way the rest of the
    toolset taught it produced a schema validation error, and pydantic-ai counts
    those as retries — five wrong guesses ended the session outright.
    """
    from agents.coder import coder_agent

    schema: dict[str, Any] = (
        coder_agent._function_toolset.tools['list_files'].tool_def.parameters_json_schema
    )

    assert 'path' in schema['properties']
    assert 'subdirectory' not in schema['properties']


async def test_list_files_lists_the_repository(deps: AgentDeps) -> None:
    """
    The plain case: everything not ignored.
    """
    root: str = deps.git.repo_path
    Path(root, 'alpha.py').write_text('a = 1\n', encoding='utf-8')
    Path(root, 'pkg').mkdir(exist_ok=True)
    Path(root, 'pkg', 'beta.py').write_text('b = 2\n', encoding='utf-8')

    result: str = await invoke('list_files', deps)

    assert 'alpha.py' in result
    assert f'pkg{os.sep}beta.py' in result


async def test_list_files_narrows_to_a_directory(deps: AgentDeps) -> None:
    """
    Passing a directory lists only what is under it.
    """
    root: str = deps.git.repo_path
    Path(root, 'alpha.py').write_text('a = 1\n', encoding='utf-8')
    Path(root, 'pkg').mkdir(exist_ok=True)
    Path(root, 'pkg', 'beta.py').write_text('b = 2\n', encoding='utf-8')

    result: str = await invoke('list_files', deps, path='pkg')

    assert 'beta.py' in result
    assert 'alpha.py' not in result


async def test_a_directory_that_does_not_exist_is_not_a_failure(deps: AgentDeps) -> None:
    """
    A miss is a legitimate answer to "what is in here", not an error.

    Reported as a failure it tells the model nothing, so it guesses again — and
    repeated guessing is what exhausts the retry budget and ends the session.
    The reply names the directories that do exist so one turn is enough.
    """
    root: str = deps.git.repo_path
    Path(root, 'pkg').mkdir(exist_ok=True)
    Path(root, 'pkg', 'beta.py').write_text('b = 2\n', encoding='utf-8')

    result: str = await invoke('list_files', deps, path='source')

    assert 'Nothing in the repository matches' in result
    assert 'pkg' in result, 'the reply should name a directory that does exist'


async def test_a_miss_is_answered_at_the_depth_that_was_asked_for(
    deps: AgentDeps,
) -> None:
    """
    A wrong guess three levels down is answered three levels down.

    Replying with the repository root would make the model start over rather
    than correct the part it got wrong.
    """
    root: str = deps.git.repo_path
    Path(root, 'pkg', 'inner').mkdir(parents=True, exist_ok=True)
    Path(root, 'pkg', 'inner', 'beta.py').write_text('b = 2\n', encoding='utf-8')

    result: str = await invoke('list_files', deps, path='pkg/missing')

    assert f'pkg{os.sep}inner' in result


async def test_a_miss_below_a_wrong_parent_falls_back_to_the_top(
    deps: AgentDeps,
) -> None:
    """
    Answering at a depth whose parent is also wrong would repeat the mistake.
    """
    root: str = deps.git.repo_path
    Path(root, 'pkg').mkdir(exist_ok=True)
    Path(root, 'pkg', 'beta.py').write_text('b = 2\n', encoding='utf-8')

    result: str = await invoke('list_files', deps, path='nowhere/deeper')

    assert 'pkg' in result


async def test_edit_file_survives_an_absolute_path_and_a_stray_argument(
    deps: AgentDeps,
) -> None:
    """
    The exact call shape that killed a real session.

    The model sent an absolute path and an argument nobody defined. The schema
    forbade unknown properties, so pydantic-ai rejected the call before it ran,
    counted it as a retry, and five such attempts ended the session — over an
    edit that was otherwise perfectly well formed.
    """
    root: str = deps.git.repo_path
    Path(root, 'a.py').write_text('value = 1\n', encoding='utf-8')

    result: str = await invoke(
        'edit_file',
        deps,
        path=os.path.join(root, 'a.py'),
        search='value = 1',
        replace='value = 2',
        summary='bump',
        suffix_added=True,
    )

    assert 'Could not edit' not in result
    assert Path(root, 'a.py').read_text() == 'value = 2\n'
    # Named rather than dropped in silence, so the model stops sending it.
    assert 'suffix_added' in result
    # Recorded relative, or the commit message and git lookups go wrong.
    assert 'a.py' in deps.session.modified_files
    assert root not in ' '.join(deps.session.modified_files)


async def test_every_tool_parameter_is_described_to_the_model() -> None:
    """
    A parameter with only a name leaves the model guessing what it means.

    `edit_file` took `search` and `replace` with nothing said about either, so a
    model had to infer from the names alone that they are exact text copied from
    the file. Guessing wrong costs a retry, and five wrong guesses end the run.
    """
    from agents.coder import answering_agent, coder_agent
    from agents.planner import planner_agent

    undocumented: list[str] = []
    for agent in (coder_agent, answering_agent, planner_agent):
        for name, tool in agent._function_toolset.tools.items():
            schema: dict[str, Any] = tool.tool_def.parameters_json_schema
            for parameter, spec in schema.get('properties', {}).items():
                if not spec.get('description'):
                    undocumented.append(f'{name}.{parameter}')

    assert undocumented == [], f'parameters with no description: {sorted(set(undocumented))}'


async def test_the_file_tools_agree_on_what_a_path_is() -> None:
    """
    A shared vocabulary, or the model spends its retries discovering it.

    `list_files` once took `subdirectory` while everything else took `path`,
    which ended a session on its own.
    """
    from agents.coder import coder_agent

    for name in ('read_file', 'create_file', 'edit_file', 'delete_file', 'list_files'):
        schema: dict[str, Any] = (
            coder_agent._function_toolset.tools[name].tool_def.parameters_json_schema
        )
        assert 'path' in schema['properties'], f'{name} does not take a path'


async def test_a_missing_file_says_how_to_find_the_right_one(deps: AgentDeps) -> None:
    """
    Refusing without saying what to do next invites another guess.
    """
    result: str = await invoke('read_file', deps, path='nowhere.py')

    assert 'does not exist' in result
    assert 'list_files' in result


@pytest.mark.parametrize('spelling', ['.', './', '', 'REPO_ROOT', 'REPO_ROOT/'])
async def test_list_files_understands_every_way_of_naming_the_root(
    deps: AgentDeps, spelling: str
) -> None:
    """
    A model asking for "." means the repository, and was told nothing matched.

    Refusing a spelling that obviously means the root does more than waste a
    turn: the agent never sees the layout, so it never finds the README that
    says how the project is tested.
    """
    root: str = deps.git.repo_path
    Path(root, 'README.md').write_text('# Run tests with: pytest\n', encoding='utf-8')

    result: str = await invoke('list_files', deps, path=spelling.replace('REPO_ROOT', root))

    assert 'README.md' in result
    assert 'Nothing in the repository matches' not in result


async def test_list_files_ignores_a_trailing_slash_or_dot_prefix(deps: AgentDeps) -> None:
    """
    "./pkg", "pkg/" and "pkg" all name the same directory.
    """
    root: str = deps.git.repo_path
    Path(root, 'pkg').mkdir(exist_ok=True)
    Path(root, 'pkg', 'beta.py').write_text('b = 2\n', encoding='utf-8')

    for spelling in ('pkg', './pkg', 'pkg/'):
        result: str = await invoke('list_files', deps, path=spelling)
        assert 'beta.py' in result, f'{spelling!r} did not list the directory'


async def test_a_missing_tool_does_not_count_as_a_failed_verification(
    deps: AgentDeps,
) -> None:
    """
    Nothing was verified, so nothing should be charged to the allowance.

    The allowance exists for a repository whose tests were already failing, and
    once it is spent the completion gate releases. Charging a missing tool to it
    turns "pytest is not installed" into permission to stop verifying, which is
    exactly the wandering this guards against.
    """
    from services.command_result import CommandResult

    async def missing(command: str) -> CommandResult:
        return CommandResult(exit_code=127, output='bash: pytest: command not found')

    deps.docker.run_test_command = missing  # type: ignore[method-assign]
    before: int = deps.session.failed_verifications

    result: str = await invoke('run_test_command', deps, command='pytest -q')

    assert deps.session.failed_verifications == before, 'the allowance was charged'
    assert 'not installed' in result
    assert 'Install it inside the sandbox' in result


async def test_a_missing_python_module_is_recognised_too(deps: AgentDeps) -> None:
    """
    `python -m pytest` without pytest exits 1, not 127.

    The exit code is indistinguishable from a real failure, and this is the most
    likely way the problem shows up in a Python repository.
    """
    from services.command_result import CommandResult

    async def missing(command: str) -> CommandResult:
        return CommandResult(exit_code=1, output='/usr/bin/python: No module named pytest')

    deps.docker.run_test_command = missing  # type: ignore[method-assign]
    before: int = deps.session.failed_verifications

    result: str = await invoke('run_test_command', deps, command='python -m pytest')

    assert deps.session.failed_verifications == before
    assert 'not installed' in result


async def test_a_real_test_failure_still_counts(deps: AgentDeps) -> None:
    """
    Narrowness matters: a genuine defect must not be read as a missing tool.

    Telling the agent to install something when the code is broken would send it
    after the wrong problem entirely.
    """
    from services.command_result import CommandResult

    async def failing(command: str) -> CommandResult:
        return CommandResult(exit_code=1, output='FAILED tests/test_a.py::test_b - assert 1 == 2')

    deps.docker.run_test_command = failing  # type: ignore[method-assign]
    before: int = deps.session.failed_verifications

    result: str = await invoke('run_test_command', deps, command='pytest -q')

    assert deps.session.failed_verifications == before + 1
    assert 'not installed' not in result


async def test_a_command_that_was_never_run_cannot_be_recorded(deps: AgentDeps) -> None:
    """
    The exact trace from a real session, and why it mattered.

    The agent ran `node -v && npm -v`, then recorded `npm install && vite build
    && npm run typecheck && npm run test` — four things it had never tried
    together, one of which reinstalls the world on every use. Every later step
    would have inherited it.
    """
    from services.command_result import CommandResult

    async def probe(command: str) -> CommandResult:
        return CommandResult(exit_code=0, output='v25.9.0\n11.12.1')

    deps.docker.run_test_command = probe  # type: ignore[method-assign]
    await invoke('run_test_command', deps, command='node -v && npm -v')

    result: str = await invoke(
        'record_test_command',
        deps,
        command='npm install && vite build && npm run typecheck && npm run test',
    )

    assert 'rejected' in result
    assert 'has not been run and passed' in result
    assert deps.session.test_command == ''


async def test_a_command_that_passed_can_be_recorded(deps: AgentDeps) -> None:
    """
    The intended path: run it, watch it pass, then record it.
    """
    from services.command_result import CommandResult

    async def passing(command: str) -> CommandResult:
        return CommandResult(exit_code=0, output='4 passed')

    deps.docker.run_test_command = passing  # type: ignore[method-assign]
    await invoke('run_test_command', deps, command='npm run test')

    result: str = await invoke('record_test_command', deps, command='npm run test')

    assert 'rejected' not in result
    assert deps.session.test_command == 'npm run test'


async def test_a_command_that_failed_cannot_be_recorded(deps: AgentDeps) -> None:
    """
    A command that ran and failed has not been shown to verify anything.
    """
    from services.command_result import CommandResult

    async def failing(command: str) -> CommandResult:
        return CommandResult(exit_code=1, output='1 failed')

    deps.docker.run_test_command = failing  # type: ignore[method-assign]
    await invoke('run_test_command', deps, command='npm run test')

    result: str = await invoke('record_test_command', deps, command='npm run test')

    assert 'rejected' in result
    assert deps.session.test_command == ''


async def test_recording_must_match_what_was_run(deps: AgentDeps) -> None:
    """
    Near enough is not enough: the extra part is the part that was never tried.
    """
    from services.command_result import CommandResult

    async def passing(command: str) -> CommandResult:
        return CommandResult(exit_code=0, output='ok')

    deps.docker.run_test_command = passing  # type: ignore[method-assign]
    await invoke('run_test_command', deps, command='npm run test')

    result: str = await invoke(
        'record_test_command', deps, command='npm install && npm run test'
    )

    assert 'rejected' in result
