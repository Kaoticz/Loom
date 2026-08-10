from __future__ import annotations

from collections.abc import Sequence

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient, MockTransport, Request, Response

import agents.coder as coder_module
from agents.coder import AgentDeps, coder_agent
from database.models.file import File
from database.models.symbol import Symbol
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
from services.user_questions import UserQuestioner
from tests.conftest import git_log, stub_embedding_service

EXPECTED_TOOLS: set[str] = {
    'ask_user',
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
    deps: AgentDeps,
) -> None:
    """
    A task that produced no answer has not been completed, whatever the verdict.
    """
    deps.session.record_classification(False)

    with pytest.raises(ValueError):
        await invoke('complete_task', deps, summary='   ')


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


@pytest.mark.usefixtures('sqlite_db')
async def test_get_symbol_returns_the_definition_itself(deps: AgentDeps) -> None:
    """
    The agent asks for a definition and must receive one.

    This previously returned `str(results)` on a list of database rows, which
    renders as their default representation, so the agent learned the primary
    keys and nothing else. The tool advertised something it did not do, and
    failed silently: the agent would simply find it useless.
    """
    file_obj = await File.create(
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


@pytest.mark.usefixtures('sqlite_db')
async def test_get_symbol_reports_every_declaration(deps: AgentDeps) -> None:
    """
    A name declared twice is worth seeing twice, not resolved arbitrarily.
    """
    for path in ('repo/a.py', 'repo/b.py'):
        file_obj = await File.create(
            file_path=path, raw_content='def handler():\n    pass\n', hash='h'
        )
        await Symbol.create(
            file=file_obj, symbol_name='handler', kind='function', line_start=1, line_end=2
        )

    returned: str = await invoke('get_symbol', deps, symbol='handler')

    assert 'repo/a.py:1-2' in returned
    assert 'repo/b.py:1-2' in returned


@pytest.mark.usefixtures('sqlite_db')
async def test_get_symbol_says_plainly_when_nothing_is_indexed(deps: AgentDeps) -> None:
    """
    An empty result must read as an answer, not as a malfunction.
    """
    returned: str = await invoke('get_symbol', deps, symbol='nowhere')

    assert 'No definition' in returned
    assert 'nowhere' in returned


@pytest.mark.usefixtures('sqlite_db')
async def test_get_symbol_reports_a_count_rather_than_the_source(
    deps: AgentDeps, events: EventBus
) -> None:
    """
    The definitions are for the agent; the reader wants to know it was found.
    """
    file_obj = await File.create(
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
