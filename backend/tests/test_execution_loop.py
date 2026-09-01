from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from collections.abc import AsyncIterator, Callable

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

import api.v1.agent_session.agent_run_helper as agent_run_helper_module
import api.v1.agent_session.agent_session as agent_session_module
import api.v1.agent_session.context_overflow_recovery as context_overflow_recovery_module
from api.v1.agent_session import AgentSession
from services.events import (
    AgentEvent,
    CompactionEvent,
    CompactionRunPath,
    DoneEvent,
    SessionStatus,
    SystemEvent,
    ThoughtEvent,
    ToolActionEvent,
)
from services.git_manager import GitManagerService
from services.docker_sandbox import CommandResult
from services.session_fsm import SessionState
from tests.test_agent_session import FakeGit, build_session, opening
from services.plan_parser import ParsedPlan, PlanStep
from pydantic_ai_harness.compaction import TieredCompaction
from services.compaction import (
    DEFAULT_CONTEXT_WINDOW,
    TARGET_FRACTION,
    build_compaction,
)


def scripted_model(responses: list[ModelResponse]) -> FunctionModel:
    """
    Build a model that replays a fixed sequence of responses.

    Text is streamed a word at a time so the buffering behaviour is exercised,
    and tool calls are emitted as delta chunks, matching how a real provider
    behaves under a streamed request.

    :param responses: The responses to return, one per model call.
    :returns: A model usable in place of a real provider.
    """
    remaining: list[ModelResponse] = list(responses)

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        response: ModelResponse = (
            remaining.pop(0) if remaining
            else ModelResponse(parts=[TextPart(content='Nothing further.')])
        )
        for part in response.parts:
            if isinstance(part, TextPart):
                for index, word in enumerate(part.content.split(' ')):
                    yield word if index == 0 else f' {word}'
            elif isinstance(part, ToolCallPart):
                yield {
                    0: DeltaToolCall(
                        name=part.tool_name,
                        json_args=json.dumps(part.args),
                    )
                }

    return FunctionModel(stream_function=stream)


def install_model(
    session: AgentSession,
    monkeypatch: pytest.MonkeyPatch,
    responses: list[ModelResponse],
) -> None:
    """
    Force the session to use a scripted model for the coder agent.

    :param session: The session to patch.
    :param monkeypatch: Fixture used to swap the resolver.
    :param responses: The responses the model should return in order.
    """
    model: FunctionModel = scripted_model(responses)
    # The resolver is a module level function now, so it is patched where the
    # session looks it up rather than on the class.
    monkeypatch.setattr(agent_run_helper_module, 'resolve_model', lambda name: model)


def prepare(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[ModelResponse],
    git_repo: GitManagerService,
) -> tuple[AgentSession, list[AgentEvent]]:
    """
    Build a session whose tools act on a real repository.

    :param monkeypatch: Fixture used to swap the model resolver.
    :param responses: The model responses to replay.
    :param git_repo: A real initialised repository for the tools to work in.
    :returns: The session and the list collecting its events.
    """
    session, _, events, _ = build_session([opening()], git=FakeGit())
    install_model(session, monkeypatch, responses)

    session._task = 'Add a marker file'
    session._deps.git = git_repo
    session._deps.editor.root_path = git_repo.repo_path
    # Built rather than copied with a raw list: `model_copy(update=...)` skips
    # validation, so a string here would sit in the field untyped and only fail
    # much later, wherever the step was first treated as a step.
    session._plan = ParsedPlan(steps=[PlanStep(title='Create the marker file')])
    return session, events


async def run_execution(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[ModelResponse],
    git_repo: GitManagerService,
) -> tuple[SessionState, list[AgentEvent]]:
    """
    Drive only the executing state with a scripted model.

    :param monkeypatch: Fixture used to swap the model resolver.
    :param responses: The model responses to replay.
    :param git_repo: A real initialised repository for the tools to work in.
    :returns: The resulting state and the events emitted.
    """
    session, events = prepare(monkeypatch, responses, git_repo)
    state: SessionState = await session._handle_executing()
    return state, events


async def test_tool_calls_after_prose_still_execute(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A tool call accompanied by prose is still executed.

    Interleaving narration with a tool call is exactly what small models do.
    """
    responses: list[ModelResponse] = [
        ModelResponse(
            parts=[
                TextPart(content='I will start by creating the file.'),
                ToolCallPart(
                    tool_name='create_file',
                    args={'path': 'marker.txt', 'content': 'created\n', 'summary': 'add marker'},
                ),
            ]
        ),
        ModelResponse(parts=[TextPart(content='Done.')]),
    ]

    state, events = await run_execution(monkeypatch, responses, git_repo)

    assert state is SessionState.reviewing
    actions = [event for event in events if isinstance(event, ToolActionEvent)]
    assert [action.tool_name for action in actions] == ['create_file']
    assert Path(git_repo.repo_path, 'marker.txt').read_text() == 'created\n'


async def test_prose_is_streamed_as_thoughts(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    The model's narration reaches the client as readable thoughts.
    """
    responses: list[ModelResponse] = [
        ModelResponse(parts=[TextPart(content='First I will read the file. Then I will edit it.')]),
    ]

    _, events = await run_execution(monkeypatch, responses, git_repo)

    thoughts: list[str] = [
        event.content for event in events if isinstance(event, ThoughtEvent)
    ]
    joined: str = ' '.join(thoughts)
    assert 'First I will read the file.' in joined
    assert 'Then I will edit it.' in joined


async def test_multiple_tools_run_in_sequence(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A multi-step ReAct sequence runs every tool, not just the first.

    This is the case that distinguishes the two streaming APIs: `run_stream`
    stops the graph after the first round of tool calls, so the second tool
    would never run and the agent could never reach complete_task.
    """
    responses: list[ModelResponse] = [
        ModelResponse(
            parts=[
                TextPart(content='Creating the file.'),
                ToolCallPart(
                    tool_name='create_file',
                    args={'path': 'a.py', 'content': 'x = 1\n', 'summary': 'add a'},
                ),
            ]
        ),
        ModelResponse(
            parts=[
                TextPart(content='Now reading it back.'),
                ToolCallPart(tool_name='read_file', args={'path': 'a.py'}),
            ]
        ),
        ModelResponse(parts=[TextPart(content='All done.')]),
    ]

    _, events = await run_execution(monkeypatch, responses, git_repo)

    actions = [
        event.tool_name for event in events if isinstance(event, ToolActionEvent)
    ]
    assert actions == ['create_file', 'read_file']


async def test_completion_signal_is_recorded(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Calling complete_task through the real loop records the completion.
    """
    responses: list[ModelResponse] = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='create_file',
                    args={'path': 'a.py', 'content': 'x = 1\n', 'summary': 'add a'},
                )
            ]
        ),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='complete_task', args={'summary': 'Added a.py as requested.'}
                )
            ]
        ),
        ModelResponse(parts=[TextPart(content='Finished.')]),
    ]

    session, _ = prepare(monkeypatch, responses, git_repo)

    await session._handle_executing()

    assert session._session.has_modified_files is True
    assert 'a.py' in session._session.modified_files


@pytest.mark.parametrize(
    ('tool_name', 'summary'),
    [
        ('complete_step', 'Created the marker file.'),
        ('accept_step_without_changes', 'No change was needed.'),
    ],
)
async def test_a_successful_step_terminal_tool_ends_the_run(
    monkeypatch: pytest.MonkeyPatch,
    git_repo: GitManagerService,
    tool_name: str,
    summary: str,
) -> None:
    """
    A successful step terminal tool prevents any later scripted tool call.

    :param monkeypatch: Fixture used to swap the model resolver and sandbox command.
    :param git_repo: A real initialised repository for the tools to work in.
    :param tool_name: The terminal tool the scripted model calls.
    :param summary: The completion summary passed to the terminal tool.
    """
    responses: list[ModelResponse]
    expected_actions: list[str]
    if tool_name == 'complete_step':
        responses = [
            ModelResponse(parts=[ToolCallPart(
                tool_name='create_file',
                args={'path': 'marker.txt', 'content': 'created\n', 'summary': 'add marker'},
            )]),
            ModelResponse(parts=[ToolCallPart(
                tool_name='run_test_command', args={'command': 'test marker'},
            )]),
            ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args={'summary': summary})]),
            ModelResponse(parts=[ToolCallPart(tool_name='read_file', args={'path': 'marker.txt'})]),
        ]
        expected_actions = ['create_file', 'run_test_command', 'complete_step']
    else:
        responses = [
            ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args={'summary': summary})]),
            ModelResponse(parts=[ToolCallPart(tool_name='read_file', args={'path': 'marker.txt'})]),
        ]
        expected_actions = ['accept_step_without_changes']

    async def pass_test(command: str) -> CommandResult:
        """
        Return a passing command result for the scripted model.

        :param command: The command the model asked the sandbox to run.
        :returns: A successful command result.
        """
        return CommandResult(exit_code=0, output='passed')

    session, events = prepare(monkeypatch, responses, git_repo)
    monkeypatch.setattr(session._deps.docker, 'run_test_command', pass_test)

    assert await session._handle_executing() is SessionState.reviewing

    actions: list[str] = [
        event.tool_name for event in events if isinstance(event, ToolActionEvent)
    ]
    assert actions == expected_actions
    assert session._session.terminal_success_count == 1


async def test_a_refused_completion_keeps_running_until_a_later_success(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A refused completion does not end the run, but its later success does.

    :param monkeypatch: Fixture used to swap the model resolver and sandbox command.
    :param git_repo: A real initialised repository for the tools to work in.
    """
    from tests.test_agent_tools import invoke

    responses: list[ModelResponse] = [
        ModelResponse(parts=[ToolCallPart(
            tool_name='complete_task', args={'summary': 'Added the marker file.'},
        )]),
        ModelResponse(parts=[ToolCallPart(
            tool_name='run_test_command', args={'command': 'test marker'},
        )]),
        ModelResponse(parts=[ToolCallPart(
            tool_name='complete_task', args={'summary': 'Added the marker file.'},
        )]),
        ModelResponse(parts=[ToolCallPart(
            tool_name='complete_task', args={'summary': 'This call must not run.'},
        )]),
    ]

    async def pass_test(command: str) -> CommandResult:
        """
        Return a passing command result for the scripted model.

        :param command: The command the model asked the sandbox to run.
        :returns: A successful command result.
        """
        return CommandResult(exit_code=0, output='passed')

    session, events = prepare(monkeypatch, responses, git_repo)
    await invoke('create_file', session._deps, path='marker.txt', content='created\n')
    events.clear()
    monkeypatch.setattr(session._deps.docker, 'run_test_command', pass_test)

    assert await session._handle_executing() is SessionState.reviewing

    actions: list[str] = [
        event.tool_name for event in events if isinstance(event, ToolActionEvent)
    ]
    assert actions == ['complete_task', 'run_test_command', 'complete_task']


async def test_an_answering_agent_stops_after_its_first_completion(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    The read-only answering path stops at its first accepted completion.

    :param monkeypatch: Fixture used to swap the model resolver.
    :param git_repo: A real initialised repository for the tools to work in.
    """
    responses: list[ModelResponse] = [
        ModelResponse(parts=[ToolCallPart(
            tool_name='complete_task', args={'summary': 'The retry policy tries three times.'},
        )]),
        ModelResponse(parts=[ToolCallPart(
            tool_name='complete_task', args={'summary': 'This call must not run.'},
        )]),
    ]
    session, events = prepare(monkeypatch, responses, git_repo)
    session._session.record_classification(False)

    assert await session._handle_answering() is SessionState.answered

    actions: list[str] = [
        event.tool_name for event in events if isinstance(event, ToolActionEvent)
    ]
    assert actions == ['complete_task']
    done: list[DoneEvent] = [event for event in events if isinstance(event, DoneEvent)]
    assert done[0].summary == 'The retry policy tries three times.'


async def test_agent_that_never_completes_is_still_reviewed(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    An agent that stops without marking the step complete still reaches review.

    Its work should be judged rather than discarded, and the user told why.
    Abandoning the remaining steps over a missing tool call would throw away
    more than it protects.
    """
    responses: list[ModelResponse] = [
        ModelResponse(parts=[TextPart(content='I think that is enough.')]),
    ]

    state, events = await run_execution(monkeypatch, responses, git_repo)

    assert state is SessionState.reviewing
    announced: str = ' '.join(
        event.content for event in events if isinstance(event, SystemEvent)
    )
    assert 'without being marked complete' in announced


async def test_executing_phase_is_announced_with_its_status(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Entering execution is announced as a system message stating the phase.

    Requirement 3.1: the announcement is `SYSTEM`, and it names the phase rather
    than leaving the client to recognise the wording.
    """
    from services.events import SessionStatus

    responses: list[ModelResponse] = [
        ModelResponse(parts=[TextPart(content='All done.')]),
    ]

    _, events = await run_execution(monkeypatch, responses, git_repo)

    executing = [
        event
        for event in events
        if isinstance(event, SystemEvent) and event.status is SessionStatus.executing
    ]
    assert executing, 'entering execution was not announced with its status'
    assert 'Executing plan' in executing[0].content


async def test_a_stop_during_streaming_unwinds_immediately(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A stop is honoured per streamed event, not after the step completes.

    Isolates the check inside the stream handler. Requirement 8.4 calls for this
    to be effectively immediate; without it a stop would wait out the whole
    execution step, which is the longest thing the session does.
    """
    from api.v1.agent_session import StopRequested

    class AlreadyStopped:
        """An inbox reporting a stop that arrived before streaming began."""

        stop_requested: bool = True

        def start(self) -> None:
            """Begin reading, which this stub need not do."""

        async def next_command(self) -> Any:
            """Never asked, because the stream unwinds first."""
            raise AssertionError('the stream should have unwound before asking')

        async def close(self) -> None:
            """Release nothing."""

    responses: list[ModelResponse] = [
        ModelResponse(parts=[TextPart(content='Thinking about it.')]),
    ]
    session, _ = prepare(monkeypatch, responses, git_repo)
    session._inbox = AlreadyStopped()  # type: ignore[assignment]

    with pytest.raises(StopRequested):
        await session._handle_executing()


async def test_an_answering_task_ends_with_its_answer(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Requirement 2.5: the answer is presented and the session ends there.

    No review, because there is no diff to judge, and no merge decision,
    because there is no branch to merge.
    """
    responses: list[ModelResponse] = [
        ModelResponse(parts=[TextPart(content='Nothing needed changing.')]),
    ]
    session, events = prepare(monkeypatch, responses, git_repo)
    session._session.record_classification(False)
    session._session.accept_completion_without_changes('The retry policy tries three times.')

    state: SessionState = await session._handle_answering()

    assert state is SessionState.answered

    done: list[DoneEvent] = [event for event in events if isinstance(event, DoneEvent)]
    assert len(done) == 1
    assert done[0].summary == 'The retry policy tries three times.'
    # An empty branch is what tells the client there is nothing to merge.
    assert done[0].branch_name == ''
    assert done[0].total_commits == 0


async def test_an_answering_task_that_gave_no_answer_escalates(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Requirement 2.7: the classification did not hold, so the task is planned.

    Ending without an answer is the only way this can happen now. The agent has
    no tool that could change a file, so an answering task that finds it needs
    one can do nothing but say so and stop — which is exactly what the prompt
    tells it to do.
    """
    responses: list[ModelResponse] = [
        ModelResponse(parts=[TextPart(content='This cannot be answered without editing the retry policy.')]),
    ]
    session, _ = prepare(monkeypatch, responses, git_repo)
    session._session.record_classification(False)

    state: SessionState = await session._handle_answering()

    assert state is SessionState.planning
    # The verdict is corrected, or the completion gate would go on excusing a
    # finish with nothing modified once the task is replanned.
    assert session._session.changes_are_required is True


async def test_the_escalation_is_announced(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Requirement 2.7: the user is told, in the way other phase changes are told.
    """
    responses: list[ModelResponse] = [
        ModelResponse(parts=[TextPart(content='This needs a change I cannot make.')]),
    ]
    session, events = prepare(monkeypatch, responses, git_repo)
    session._session.record_classification(False)

    await session._handle_answering()

    announcements: list[SystemEvent] = [
        event for event in events if isinstance(event, SystemEvent)
    ]
    assert any('need file changes after all' in event.content for event in announcements)
    assert any(event.status is SessionStatus.planning for event in announcements)


async def test_a_task_that_changed_files_still_goes_to_review(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Requirement 2.3: the ordinary path is untouched.
    """
    responses: list[ModelResponse] = [
        ModelResponse(parts=[TextPart(content='Done.')]),
    ]
    session, _ = prepare(monkeypatch, responses, git_repo)
    session._session.record_modification('a.py')
    session._session.record_verification(True)
    session._session.request_completion('Changed a.py.')

    state: SessionState = await session._handle_executing()

    assert state is SessionState.reviewing


def recording_model(seen: list[list[str]]) -> FunctionModel:
    """
    Build a model that records which tools it was offered.

    The tool definitions a provider receives are the only place the guarantee is
    observable: an agent cannot be shown to lack a tool by asking it nicely not
    to use one.

    :param seen: A list each call appends its offered tool names to.
    :returns: A model that answers immediately, having noted its tools.
    """

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        seen.append(sorted(tool.name for tool in info.function_tools))
        yield 'Nothing to report.'

    return FunctionModel(stream_function=stream)


async def test_an_answering_task_is_offered_no_tool_that_writes(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A task needing no file changes cannot make any.

    The prompt also tells the agent not to write, but an instruction is only
    obeyed until it is not, and by then the repository has already changed. The
    tools are therefore absent rather than merely discouraged.
    """
    offered: list[list[str]] = []
    session, _ = prepare(monkeypatch, [], git_repo)
    monkeypatch.setattr(agent_run_helper_module, 'resolve_model', lambda name: recording_model(offered))
    session._session.record_classification(False)

    await session._handle_answering()

    assert offered, 'the model was never called'
    for tools in offered:
        assert 'create_file' not in tools
        assert 'edit_file' not in tools
        assert 'delete_file' not in tools
        assert 'run_test_command' not in tools
        # Reading and answering must still be possible, or the path is useless.
        assert 'read_file' in tools
        assert 'complete_task' in tools


async def test_an_ordinary_task_is_still_offered_the_writing_tools(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Requirement 2.3: withholding the tools must not leak into the normal path.
    """
    offered: list[list[str]] = []
    session, _ = prepare(monkeypatch, [], git_repo)
    monkeypatch.setattr(agent_run_helper_module, 'resolve_model', lambda name: recording_model(offered))
    session._session.record_classification(True)

    await session._handle_executing()

    assert offered, 'the model was never called'
    assert 'create_file' in offered[0]
    assert 'edit_file' in offered[0]
    assert 'delete_file' in offered[0]
    assert 'run_test_command' in offered[0]


def prompt_recording_model(seen: list[str]) -> FunctionModel:
    """
    Build a model that records the user prompt it was sent.

    The prompt the provider receives is where conditional guidance is either
    present or absent, so that is where it has to be observed.

    :param seen: A list each call appends the prompt text to.
    :returns: A model that answers immediately, having noted its prompt.
    """

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        for message in messages:
            for part in message.parts:
                content: object = getattr(part, 'content', None)
                if isinstance(content, str):
                    seen.append(content)
        yield 'Nothing to report.'

    return FunctionModel(stream_function=stream)


async def test_a_frontend_repository_is_told_to_use_agent_browser(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService, tmp_path: Path
) -> None:
    """
    A repository declaring a frontend framework gets the browser guidance.
    """
    from services.browser_tooling import AGENT_BROWSER_INSTALL_COMMAND

    repo: Path = tmp_path / 'frontend-repo'
    repo.mkdir()
    (repo / 'package.json').write_text(
        json.dumps({'name': 'app', 'dependencies': {'react': '^18.0.0'}}), encoding='utf-8'
    )

    seen: list[str] = []
    session, _ = prepare(monkeypatch, [], git_repo)
    monkeypatch.setattr(
        agent_run_helper_module, 'resolve_model', lambda name: prompt_recording_model(seen)
    )
    session._services.git.repo_path = str(repo)

    await session._handle_executing()

    prompt: str = '\n'.join(seen)
    assert 'agent-browser' in prompt
    assert AGENT_BROWSER_INSTALL_COMMAND in prompt


async def test_a_backend_only_repository_is_told_nothing_about_browsers(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService, tmp_path: Path
) -> None:
    """
    A repository with no browser to drive is not given browser instructions.

    Irrelevant guidance on every unrelated task is not free: it occupies the
    context the actual work needs.
    """
    repo: Path = tmp_path / 'backend-repo'
    repo.mkdir()
    (repo / 'pyproject.toml').write_text('[project]\nname = "api"\n', encoding='utf-8')

    seen: list[str] = []
    session, _ = prepare(monkeypatch, [], git_repo)
    monkeypatch.setattr(
        agent_run_helper_module, 'resolve_model', lambda name: prompt_recording_model(seen)
    )
    session._services.git.repo_path = str(repo)

    await session._handle_executing()

    assert seen, 'the model was never called'
    assert 'agent-browser' not in '\n'.join(seen)


async def test_a_node_backend_is_not_mistaken_for_a_frontend(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService, tmp_path: Path
) -> None:
    """
    A `package.json` alone must not trigger the guidance.

    An Express API is a Node project with no browser to drive; firing on any
    manifest at all would put browser instructions on every Node task.
    """
    repo: Path = tmp_path / 'node-api'
    repo.mkdir()
    (repo / 'package.json').write_text(
        json.dumps({'name': 'api', 'dependencies': {'express': '^4.0.0'}}), encoding='utf-8'
    )

    seen: list[str] = []
    session, _ = prepare(monkeypatch, [], git_repo)
    monkeypatch.setattr(
        agent_run_helper_module, 'resolve_model', lambda name: prompt_recording_model(seen)
    )
    session._services.git.repo_path = str(repo)

    await session._handle_executing()

    assert seen, 'the model was never called'
    assert 'agent-browser' not in '\n'.join(seen)


def budget_recording(seen: list[int | None]) -> Callable[..., TieredCompaction[Any]]:
    """
    Wrap `build_compaction` so the window it was asked for is observable.

    Recorded at the point the budget is built rather than by intercepting the
    agent, because that is the question being asked: which window reached the
    run.

    :param seen: A list each call appends its window argument to.
    :returns: A stand-in for `build_compaction`.
    """

    def record(context_window: int | None, **kwargs: object) -> TieredCompaction[Any]:
        seen.append(context_window)
        return build_compaction(context_window)

    return record


async def test_a_compaction_notice_logs_and_emits_its_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    One compaction becomes one structured event and one informative log line.
    """
    session, _, events, _ = build_session([])
    caplog.set_level(logging.INFO, logger=agent_session_module.__name__)

    await session._report_compaction(CompactionRunPath.coder_streaming, 21_300, 12_840)

    notices: list[CompactionEvent] = [
        event for event in events if isinstance(event, CompactionEvent)
    ]
    assert notices == [
        CompactionEvent(
            tokens_before=21_300,
            tokens_after=12_840,
            run_path=CompactionRunPath.coder_streaming,
        )
    ]
    assert caplog.messages == [
        'Compacted coder_streaming request history from an estimated 21300 to 12840 tokens.'
    ]


async def test_the_coder_run_carries_the_configured_budget(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    The user's coder window reaches the run as a compaction capability.

    Attached per run rather than fixed on the agent: the budget is a user
    setting, and a capability attached at import time could not see it.
    """
    windows: list[int | None] = []
    session, _ = prepare(monkeypatch, [], git_repo)
    monkeypatch.setattr(agent_run_helper_module, 'resolve_model', lambda name: quiet_model())
    monkeypatch.setattr(agent_session_module, 'build_compaction', budget_recording(windows))
    session._coder_context_window = 40_000
    session._session.record_classification(True)

    await session._handle_executing()

    assert windows == [40_000]
    # The ladder is budgeted as a fraction of a stated window, so the window is
    # what has to arrive intact.
    assert build_compaction(windows[0]).context_window == 40_000


async def test_an_unset_budget_still_compacts(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Leaving the setting alone must not mean leaving the history unbounded.

    A local model with no configured window is exactly the case compaction was
    added for, so the default has to apply rather than the feature switching off.
    """
    windows: list[int | None] = []
    session, _ = prepare(monkeypatch, [], git_repo)
    monkeypatch.setattr(agent_run_helper_module, 'resolve_model', lambda name: quiet_model())
    monkeypatch.setattr(agent_session_module, 'build_compaction', budget_recording(windows))
    session._coder_context_window = 0
    session._session.record_classification(True)

    await session._handle_executing()

    assert windows == [0]
    assert build_compaction(windows[0]).context_window == DEFAULT_CONTEXT_WINDOW


def quiet_model() -> FunctionModel:
    """
    Build a model that answers at once, so a run ends without doing anything.

    :returns: A model that says nothing of substance.
    """

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        yield 'Done.'

    return FunctionModel(stream_function=stream)


async def tool_names_offered(agent: Any) -> set[str]:
    """
    List the tools an agent will offer a model.

    :param agent: The agent to inspect.
    :returns: The tool names.
    """
    from pydantic_ai.models.function import AgentInfo

    seen: set[str] = set()

    async def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.update(tool.name for tool in info.function_tools)
        return ModelResponse(parts=[TextPart('Done.')])

    await agent.run(
        'anything',
        deps=MagicMock(),
        model=FunctionModel(capture),
        output_type=str,
    )
    return seen


async def test_the_planner_can_look_at_the_repository() -> None:
    """
    A plan written without reading the code is a guess.

    The planner had no tools at all, so its structured path named files it had
    never seen. These are what let it look.
    """
    from agents.planner import planner_agent

    offered: set[str] = await tool_names_offered(planner_agent)

    assert {'search_code', 'search_code_lexical', 'get_symbol', 'read_file'} <= offered
    assert 'list_files' in offered
    assert 'web_search' in offered


async def test_the_planner_cannot_change_anything() -> None:
    """
    Planning happens before the user has approved anything.

    Structural rather than instructed: the writing tools were never registered,
    so there is no switch to leave in the wrong position.
    """
    from agents.planner import planner_agent

    offered: set[str] = await tool_names_offered(planner_agent)

    assert offered.isdisjoint({
        'create_file',
        'edit_file',
        'delete_file',
        'run_test_command',
        'commit_changes',
        'complete_task',
    })


async def test_a_structured_plan_is_used_as_it_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The planner's primary path: structure arrives and is used without fallback.

    Previously uncovered. Every test touching planning drove the text fallback,
    so a regression that broke structured planning outright would not have
    failed anything.
    """
    import agents.planner as planner_module
    from agents.planner import PlannedStep, StructuredPlan
    from services.plan_parser import PlanFormat
    from tests.test_agent_session import build_session

    calls: list[dict[str, Any]] = []

    async def structured(prompt: str, **kwargs: Any) -> Any:
        calls.append(kwargs)

        class Result:
            output: StructuredPlan = StructuredPlan(steps=[
                PlannedStep(title='Read the config', description='Find the timeout.'),
                PlannedStep(title='Raise the timeout', description='Set it to 30s.'),
            ])

        return Result()

    monkeypatch.setattr(planner_module.planner_agent, 'run', structured)

    session, _, _, _ = build_session([json.dumps({'prompt': 'do something'})])
    plan = await session._planning_coordinator.plan_structured('irrelevant')

    assert plan.titles == ['Read the config', 'Raise the timeout']
    # Structured, not text: the fallback must not have run.
    assert plan.format is PlanFormat.structured
    assert len(calls) == 1, 'the fallback ran despite structure arriving'
    # The planner needs deps now, or its tools have nothing to act through.
    assert calls[0].get('deps') is not None


async def test_a_structured_plan_drops_steps_without_a_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A titleless step is not actionable and would render as an empty row.
    """
    import agents.planner as planner_module
    from agents.planner import PlannedStep, StructuredPlan
    from tests.test_agent_session import build_session

    async def structured(prompt: str, **kwargs: Any) -> Any:
        class Result:
            output: StructuredPlan = StructuredPlan(steps=[
                PlannedStep(title='  ', description='Nameless.'),
                PlannedStep(title='Real step', description='Actionable.'),
            ])

        return Result()

    monkeypatch.setattr(planner_module.planner_agent, 'run', structured)

    session, _, _, _ = build_session([json.dumps({'prompt': 'do something'})])
    plan = await session._planning_coordinator.plan_structured('irrelevant')

    assert plan.titles == ['Real step']


async def test_the_planner_can_ask_the_user() -> None:
    """
    An ambiguity that changes the plan is better settled before it is written.

    Discovering it halfway through executing costs the steps already taken.
    """
    from agents.planner import planner_agent

    offered: set[str] = await tool_names_offered(planner_agent)

    assert 'ask_user' in offered


async def test_the_planner_has_no_task_to_complete() -> None:
    """
    `complete_task` belongs to execution, not to planning.

    The planner's job ends when it returns a plan; marking a task complete from
    inside planning would claim work that has not started.
    """
    from agents.planner import planner_agent

    offered: set[str] = await tool_names_offered(planner_agent)

    assert 'complete_task' not in offered


async def test_a_planner_question_reaches_the_user_and_the_answer_comes_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The whole point: the model asks, the user answers, planning continues.

    Driven through a real `planner_agent.run` rather than by calling the tool
    directly, so the registration, the deps and the answer path are all
    exercised together.
    """
    from pydantic_ai.models.function import AgentInfo
    from tests.test_agent_session import build_session

    session, _, _, _ = build_session([json.dumps({'prompt': 'do something'})])

    answered: list[str] = []

    class Questioner:
        async def ask(self, question: str, options: list[str]) -> tuple[str, bool]:
            answered.append(question)
            return 'Use exponential backoff.', False

    session._deps.questioner = Questioner()  # type: ignore[assignment]

    turns: list[int] = []

    async def converse(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        turns.append(1)
        if len(turns) == 1:
            return ModelResponse(parts=[ToolCallPart(
                tool_name='ask_user',
                args={'question': 'Which retry policy?', 'options': ['Fixed', 'Backoff']},
                tool_call_id='ask-1',
            )])
        # The answer has to be visible to the model on the next turn, or asking
        # achieved nothing.
        replies: str = ' '.join(
            str(getattr(part, 'content', ''))
            for message in messages
            for part in message.parts
        )
        assert 'exponential backoff' in replies
        return ModelResponse(parts=[TextPart('1. Add exponential backoff to the client.')])

    monkeypatch.setattr(
        agent_run_helper_module, 'resolve_model', lambda name: FunctionModel(converse)
    )

    plan = await session._planning_coordinator.plan_as_text('irrelevant')

    assert answered == ['Which retry policy?']
    assert 'exponential backoff' in plan.lower()


async def test_a_planner_question_with_nobody_to_ask_does_not_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A session with no user must fail the question, not wait forever for one.
    """
    from tests.test_agent_session import build_session
    from tests.test_agent_tools import invoke

    session, _, _, _ = build_session([json.dumps({'prompt': 'do something'})])
    session._deps.questioner = None

    result: str = await invoke('ask_user', session._deps, question='Which one?')

    assert 'no user to ask' in result


def counting_model(seen: list[str], responses: list[ModelResponse]) -> FunctionModel:
    """
    Build a model that records each prompt it is sent, replaying scripted turns.

    One entry per distinct prompt, so the number of entries is the number of
    runs rather than the number of model requests.

    :param seen: A list each run appends its user prompt to.
    :param responses: The response to give on each successive turn.
    :returns: The scripted model.
    """
    turns: list[ModelResponse] = list(responses)

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        latest: str = ''
        for message in messages:
            for part in message.parts:
                content: object = getattr(part, 'content', None)
                if isinstance(content, str) and 'Approved plan' in content:
                    latest = content
        # Only when the prompt changes. A step that calls a tool makes a second
        # request within the same run, so appending every request would count
        # tool rounds as steps.
        if latest and (not seen or seen[-1] != latest):
            seen.append(latest)

        # Execution streams, so the turns have to be replayed as deltas.
        response: ModelResponse = (
            turns.pop(0) if turns else ModelResponse(parts=[TextPart('Done.')])
        )
        for part in response.parts:
            if isinstance(part, TextPart):
                yield part.content
            elif isinstance(part, ToolCallPart):
                yield {
                    0: DeltaToolCall(name=part.tool_name, json_args=json.dumps(part.args))
                }

    return FunctionModel(stream_function=stream)


def three_step_plan() -> ParsedPlan:
    """
    Build a plan with three distinguishable steps.

    :returns: The plan.
    """
    return ParsedPlan(steps=[
        PlanStep(title='Find the timeout', description='Locate where it is set.'),
        PlanStep(title='Raise the timeout', description='Change it to 30 seconds.'),
        PlanStep(title='Document the change', description='Note it in the changelog.'),
    ])


async def test_a_three_step_plan_runs_three_times(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    One run per step, not one run for the plan.

    Handed everything at once, a small model stops after whichever part it found
    most legible, because nothing obliges it to account for the rest.
    """
    prompts: list[str] = []
    session, _ = prepare(monkeypatch, [], git_repo)
    session._plan = three_step_plan()
    monkeypatch.setattr(
        agent_run_helper_module, 'resolve_model', lambda name: counting_model(prompts, [])
    )

    assert await session._handle_executing() is SessionState.reviewing
    assert len(prompts) == 3, f'expected three runs, got {len(prompts)}'


async def test_each_step_is_told_which_step_it_is(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A step has to know which one it is, or it will redo or skip work.
    """
    prompts: list[str] = []
    session, _ = prepare(monkeypatch, [], git_repo)
    session._plan = three_step_plan()
    monkeypatch.setattr(
        agent_run_helper_module, 'resolve_model', lambda name: counting_model(prompts, [])
    )

    await session._handle_executing()

    assert 'step 1 of 3' in prompts[0].lower()
    assert 'Find the timeout' in prompts[0]
    assert 'step 3 of 3' in prompts[2].lower()
    assert 'Document the change' in prompts[2]


async def test_a_later_step_is_told_what_earlier_steps_did(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Summaries, not history: what a step needs is what the others achieved.

    Carrying the raw tool calls forward would recreate the context pressure
    compaction exists to relieve.
    """
    prompts: list[str] = []
    session, _ = prepare(monkeypatch, [], git_repo)
    session._plan = three_step_plan()
    monkeypatch.setattr(
        agent_run_helper_module, 'resolve_model', lambda name: counting_model(prompts, [
            ModelResponse(parts=[ToolCallPart(
                tool_name='accept_step_without_changes',
                args={'summary': 'The timeout lives in config.py line 12.'},
                tool_call_id='step-1',
            )]),
        ])
    )

    await session._handle_executing()

    assert 'config.py line 12' in prompts[1], 'the second step was not told the first step findings'
    # The first step cannot be told about itself.
    assert 'config.py line 12' not in prompts[0]


async def test_a_step_cannot_finish_without_verifying_its_change(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    The same gate `complete_task` applies, one step earlier.

    Catching an unverified change at the end of the task means the agent moved
    on several steps before hearing about it.
    """
    from tests.test_agent_tools import invoke

    session, _ = prepare(monkeypatch, [], git_repo)
    session._session.begin_step('Raise the timeout')
    session._session.record_modification('config.py')

    result: str = await invoke('complete_step', session._deps, summary='Raised it.')

    assert 'rejected' in result
    assert 'has not been verified' in result


async def test_a_step_that_changed_nothing_cannot_complete_normally(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A step claiming completion without changing anything is pointed at the
    investigative escape hatch rather than silently accepted.
    """
    from tests.test_agent_tools import invoke

    session, _ = prepare(monkeypatch, [], git_repo)
    session._session.begin_step('Raise the timeout')

    result: str = await invoke('complete_step', session._deps, summary='Done.')

    assert 'rejected' in result
    assert 'accept_step_without_changes' in result


async def test_an_investigative_step_may_finish_without_changes(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    The planner can read the repository now, so it writes steps that only read.

    Without this the model would loop to the iteration ceiling on a step it was
    never asked to make a change for.
    """
    from tests.test_agent_tools import invoke

    session, _ = prepare(monkeypatch, [], git_repo)
    session._session.begin_step('Find the timeout')

    result: str = await invoke(
        'accept_step_without_changes', session._deps, summary='It is in config.py.'
    )

    assert 'rejected' not in result
    assert list(session._session.step_summaries) == ['It is in config.py.']


async def test_the_escape_hatch_refuses_a_step_that_did_change_something(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Otherwise it becomes a way past the verification gate.
    """
    from tests.test_agent_tools import invoke

    session, _ = prepare(monkeypatch, [], git_repo)
    session._session.begin_step('Raise the timeout')
    session._session.record_modification('config.py')

    result: str = await invoke(
        'accept_step_without_changes', session._deps, summary='Nothing to do.'
    )

    assert 'rejected' in result
    assert 'did change something' in result


async def test_step_state_does_not_disturb_the_task_wide_record(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    The whole-task counters feed `complete_task`, the DoneEvent and the
    answering path, none of which know a plan has steps. Resetting them per step
    would make the task look as though only its last step had happened.
    """
    session, _ = prepare(monkeypatch, [], git_repo)
    context = session._session

    context.begin_step('First')
    context.record_modification('a.py')
    context.record_verification(True)
    context.request_step_completion('Changed a.py')

    context.begin_step('Second')
    assert context.step_has_modified_files is False, 'step state should have reset'
    assert context.has_modified_files is True, 'task-wide state must not reset'
    assert list(context.modified_files) == ['a.py']


async def test_the_final_summary_covers_every_step(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    The reviewer and the DoneEvent must see the whole plan's work.

    Assembled from the steps rather than by a further model call, which would
    cost a request and could disagree with the steps it summarised.
    """
    session, _ = prepare(monkeypatch, [], git_repo)
    session._plan = three_step_plan()
    context = session._session
    context.begin_step('Find the timeout')
    context.record_step_completion('Found it in config.py.')
    context.begin_step('Raise the timeout')
    context.record_step_completion('Set it to 30 seconds.')

    assert await session._finish_execution() is SessionState.reviewing

    summary: str | None = context.completion_summary
    assert summary is not None
    assert 'Found it in config.py.' in summary
    assert 'Set it to 30 seconds.' in summary


async def test_a_stray_task_completion_does_not_shrink_the_summary(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A model that calls `complete_task` mid-plan must not speak for the whole plan.

    It is told not to, but being told is not a guarantee. Letting that one call
    stand would leave every later step's work unmentioned in the summary the
    reviewer and the DoneEvent read.
    """
    session, _ = prepare(monkeypatch, [], git_repo)
    session._plan = three_step_plan()
    context = session._session

    context.begin_step('Find the timeout')
    context.record_step_completion('Found it in config.py.')
    # As though the model ignored its instructions here.
    context.record_modification('config.py')
    context.record_verification(True)
    context.request_completion('I have finished everything.')
    context.begin_step('Raise the timeout')
    context.record_step_completion('Set it to 30 seconds.')

    await session._finish_execution()

    summary: str | None = context.completion_summary
    assert summary is not None
    assert 'Found it in config.py.' in summary
    assert 'Set it to 30 seconds.' in summary


async def test_a_step_commits_once_when_it_finishes(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    One commit per step, not one per file written.

    A commit per write turned one step into a string of commits nobody chose the
    boundaries of, and a plan into a history that could not be read back a step
    at a time.
    """
    from tests.test_agent_tools import invoke

    session, _ = prepare(monkeypatch, [], git_repo)
    await git_repo.create_branch('ai-task-commits')
    session._deps.session.begin_step('Add two files')

    await invoke('create_file', session._deps, path='one.py', content='a = 1\n')
    await invoke('create_file', session._deps, path='two.py', content='b = 2\n')
    session._deps.session.record_verification(True)

    before: int = len(await git_repo.get_changed_files())
    result: str = await invoke('complete_step', session._deps, summary='Added both files.')

    assert 'rejected' not in result
    assert 'committed as' in result
    # Both files land in the one commit the step made.
    after: list[str] = [change.path for change in await git_repo.get_changed_files()]
    assert 'one.py' in after and 'two.py' in after
    assert before == 0, 'the writes should not have committed on their own'


async def test_a_failing_commit_warns_and_does_not_kill_the_session(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    The changes are on disk either way, so failing the session loses the work
    along with the diagnosis.
    """
    from services.events import WarningEvent
    from tests.test_agent_tools import invoke

    session, events = prepare(monkeypatch, [], git_repo)
    await git_repo.create_branch('ai-task-failing')
    session._deps.session.begin_step('Add a file')
    await invoke('create_file', session._deps, path='one.py', content='a = 1\n')
    session._deps.session.record_verification(True)

    attempts: list[int] = []

    async def refuse(message: str) -> object:
        attempts.append(1)
        raise RuntimeError('committer identity unknown')

    monkeypatch.setattr(git_repo, 'commit_changes', refuse)

    result: str = await invoke('complete_step', session._deps, summary='Added it.')

    # Retried once before giving up.
    assert len(attempts) == 2
    assert 'warning' in result.lower()
    warnings = [event for event in events if isinstance(event, WarningEvent)]
    assert warnings, 'the failure was not announced as a warning'
    assert 'identity unknown' in warnings[0].content
    # The step still counts as done; the work is not discarded.
    assert list(session._deps.session.step_summaries) == ['Added it.']


async def test_an_uncommitted_step_is_named_in_the_final_summary(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    The reviewer and the merge read only commits, so uncommitted work would
    otherwise be described as though it had landed.
    """
    session, _ = prepare(monkeypatch, [], git_repo)
    context = session._session
    context.begin_step('Raise the timeout')
    context.record_modification('config.py')
    context.record_verification(True)
    context.record_commit_failure('Raise the timeout')
    context.record_step_completion('Raised it to 30s.')

    await session._finish_execution()

    summary: str | None = context.completion_summary
    assert summary is not None
    assert 'Raise the timeout' in summary
    assert 'could not be committed' in summary


async def test_repeated_commit_failures_stop_and_ask(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A run of failures is a configuration problem, not bad luck.

    Every further step past it adds work that neither the review nor the merge
    can see, so the user is asked while it is still recoverable.
    """
    from api.v1.agent_session import COMMIT_FAILURE_COMMITTED

    session, _ = prepare(monkeypatch, [], git_repo)
    context = session._session
    for _ in range(3):
        context.record_commit_failure('Raise the timeout')

    asked: list[str] = []
    committed: list[bool] = [False]

    class Questioner:
        async def ask(self, question: str, options: list[str]) -> tuple[str, bool]:
            asked.append(question)
            # First answer claims a commit that was not made; the second is true.
            if not committed[0]:
                committed[0] = True
                return COMMIT_FAILURE_COMMITTED, False
            return COMMIT_FAILURE_COMMITTED, False

    session._deps.questioner = Questioner()  # type: ignore[assignment]

    dirty: list[bool] = [True, False]

    async def has_changes() -> bool:
        return dirty.pop(0)

    monkeypatch.setattr(session._services.git, 'has_uncommitted_changes', has_changes)

    await session._commit_sweeper.resolve_commit_failures()

    # Asked twice: the first claim was checked and found untrue.
    assert len(asked) == 2
    assert context.consecutive_commit_failures == 0


async def test_carrying_on_after_commit_failures_stops_asking(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A user who has chosen to carry on should not be asked again every step.

    The stranded steps stay recorded, so the final summary still reports them.
    """
    from api.v1.agent_session import COMMIT_FAILURE_CONTINUE

    session, _ = prepare(monkeypatch, [], git_repo)
    context = session._session
    for _ in range(3):
        context.record_commit_failure('Raise the timeout')

    class Questioner:
        async def ask(self, question: str, options: list[str]) -> tuple[str, bool]:
            return COMMIT_FAILURE_CONTINUE, False

    session._deps.questioner = Questioner()  # type: ignore[assignment]

    await session._commit_sweeper.resolve_commit_failures()

    assert context.consecutive_commit_failures == 0
    assert list(context.uncommitted_steps) == ['Raise the timeout']


async def test_commit_failures_with_nobody_to_ask_do_not_hang(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Stopping a session nobody is watching would discard the work for no benefit.
    """
    from services.events import WarningEvent

    session, events = prepare(monkeypatch, [], git_repo)
    context = session._session
    for _ in range(3):
        context.record_commit_failure('Raise the timeout')
    session._deps.questioner = None

    await session._commit_sweeper.resolve_commit_failures()

    warnings = [event for event in events if isinstance(event, WarningEvent)]
    assert warnings
    assert 'no one to ask' in warnings[-1].content


async def test_an_answering_task_commits_nothing(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A task that was never going to change files must not touch the repository.

    `complete_task` commits now, and the answering path shares that tool, so the
    early return for a no-changes completion is what keeps a question from
    sweeping the user's own uncommitted work into a commit they never asked for.
    """
    from tests.test_agent_tools import invoke

    session, _ = prepare(monkeypatch, [], git_repo)
    session._deps.session.record_classification(False)

    attempts: list[str] = []

    async def record(message: str) -> object:
        attempts.append(message)
        raise AssertionError('an answering task must not commit')

    monkeypatch.setattr(git_repo, 'commit_changes', record)

    result: str = await invoke(
        'complete_task', session._deps, summary='The retry logic lives in client.py.'
    )

    assert 'rejected' not in result
    assert attempts == []


async def test_leftover_changes_are_swept_into_a_commit(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Work no step committed still has to reach the reviewer.

    A step that ended without reporting completion does not commit, so without a
    sweep its changes would sit in the working tree, invisible to both the review
    and the merge — the exact loss this phase exists to prevent.
    """
    from pathlib import Path

    from services.events import WarningEvent

    session, events = prepare(monkeypatch, [], git_repo)
    await git_repo.create_branch('ai-task-sweep')
    session._branch_created = True
    session._services.git = git_repo
    Path(git_repo.repo_path, 'stray.py').write_text('value = 1\n', encoding='utf-8')

    await session._commit_sweeper.sweep_uncommitted()

    committed: list[str] = [change.path for change in await git_repo.get_changed_files()]
    assert 'stray.py' in committed
    warnings = [event for event in events if isinstance(event, WarningEvent)]
    assert warnings, 'the sweep was not announced'
    assert 'not committed by the step' in warnings[-1].content


async def test_a_clean_tree_is_not_swept(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    An empty commit and a warning about nothing would be noise.
    """
    from services.events import WarningEvent

    session, events = prepare(monkeypatch, [], git_repo)
    await git_repo.create_branch('ai-task-clean')
    session._branch_created = True
    session._services.git = git_repo

    await session._commit_sweeper.sweep_uncommitted()

    assert [event for event in events if isinstance(event, WarningEvent)] == []


async def test_an_established_command_is_carried_to_later_steps(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A command worked out once should not be worked out again every step.

    Establishing it is most of the cost when a project needs its environment
    building first, and asking the user twice for the same answer is worse.
    """
    prompts: list[str] = []
    session, _ = prepare(monkeypatch, [], git_repo)
    session._plan = three_step_plan()
    # It has to have passed before it can be recorded.
    session._session.note_command_passed('pytest -q tests/')
    session._session.record_test_command('pytest -q tests/')
    monkeypatch.setattr(
        agent_run_helper_module, 'resolve_model', lambda name: counting_model(prompts, [])
    )

    await session._handle_executing()

    assert all('pytest -q tests/' in prompt for prompt in prompts)


async def test_a_waived_plan_never_mentions_testing_again(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A user who said to skip tests should not be asked at every remaining step.

    The same applies to a repository with no tests: the answer cannot change
    between steps, so putting the question again is pure interruption.
    """
    prompts: list[str] = []
    session, _ = prepare(monkeypatch, [], git_repo)
    session._plan = three_step_plan()
    session._session.waive_verification('The user asked to skip tests for this task.')
    monkeypatch.setattr(
        agent_run_helper_module, 'resolve_model', lambda name: counting_model(prompts, [])
    )

    await session._handle_executing()

    for prompt in prompts:
        assert 'not required for this plan' in prompt
        assert 'do not ask about them again' in prompt.lower()


async def test_a_forgotten_command_is_not_carried_forward(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A command that turned out to be wrong must not be inherited.

    Silently reusing it in the next step keeps producing failures that look like
    the code's fault, which is the failure mode this whole path exists to avoid.
    """
    from tests.test_agent_tools import invoke

    session, _ = prepare(monkeypatch, [], git_repo)
    # It has to have passed before it can be recorded.
    session._session.note_command_passed('pytest -q packages/api')
    session._session.record_test_command('pytest -q packages/api')

    result: str = await invoke(
        'forget_test_command', session._deps, reason='This step is in the web package.'
    )

    assert 'rejected' not in result
    assert session._session.test_command == ''
    # The next step's prompt must not name the command that was just discarded.
    following: str = session._build_step_prompt(
        PlanStep(title='Next', description='Do the next thing.'), 2, 3
    )
    assert 'pytest -q packages/api' not in following
    assert 'established earlier in this plan' not in following


async def test_a_repository_with_no_tests_can_still_finish_a_step(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Otherwise the step gate makes such a repository unworkable.

    The gate demands a verified change; a project with nothing to run it with
    could never satisfy it, and the agent would loop until the ceiling.
    """
    from tests.test_agent_tools import invoke

    session, _ = prepare(monkeypatch, [], git_repo)
    context = session._session
    context.begin_step('Add the helper')
    context.record_modification('helper.py')

    blocked: str = await invoke('complete_step', session._deps, summary='Added it.')
    assert 'has not been verified' in blocked

    await invoke('skip_verification', session._deps, reason='The repository has no tests.')
    allowed: str = await invoke('complete_step', session._deps, summary='Added it.')

    assert 'rejected' not in allowed


async def test_an_unverified_plan_says_so_in_its_summary(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Skipping verification must not read as an oversight to whoever reviews it.
    """
    session, _ = prepare(monkeypatch, [], git_repo)
    context = session._session
    context.begin_step('Add the helper')
    context.record_modification('helper.py')
    context.waive_verification('The repository has no tests.')
    context.record_step_completion('Added the helper.')

    await session._finish_execution()

    summary: str | None = context.completion_summary
    assert summary is not None
    assert 'not verified' in summary
    assert 'no tests' in summary


async def test_skipping_requires_a_reason(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A waiver with no reason produces a summary that cannot explain itself.
    """
    from tests.test_agent_tools import invoke

    session, _ = prepare(monkeypatch, [], git_repo)

    result: str = await invoke('skip_verification', session._deps, reason='   ')

    assert 'rejected' in result
    assert session._session.verification_required is True


async def test_a_waived_plan_does_not_also_demand_verification(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    The prompt must not tell the agent to verify after verification was waived.

    Two instructions that contradict each other leave the model to pick one, and
    the one it picks is the one this phase was trying to avoid: running a command
    against a repository that has nothing to run.
    """
    session, _ = prepare(monkeypatch, [], git_repo)
    session._session.waive_verification('The repository has no tests.')

    stepwise: str = session._build_step_prompt(
        PlanStep(title='Add the helper', description='Write it.'), 1, 2
    )
    whole: str = session._build_step_prompt(None, 1, 1)

    for prompt in (stepwise, whole):
        assert 'not required for this plan' in prompt
        assert 'Verify every change with run_test_command' not in prompt


async def test_an_unwaived_plan_still_demands_verification(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Removing the instruction when it is waived must not remove it otherwise.
    """
    session, _ = prepare(monkeypatch, [], git_repo)

    stepwise: str = session._build_step_prompt(
        PlanStep(title='Add the helper', description='Write it.'), 1, 2
    )
    whole: str = session._build_step_prompt(None, 1, 1)

    for prompt in (stepwise, whole):
        assert 'Verify every change with run_test_command' in prompt


async def test_an_established_command_reaches_the_whole_task_prompt(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    An unparsed plan runs as one whole task, and still needs the command.
    """
    session, _ = prepare(monkeypatch, [], git_repo)
    # It has to have passed before it can be recorded.
    session._session.note_command_passed('npm test')
    session._session.record_test_command('npm test')

    assert 'npm test' in session._build_step_prompt(None, 1, 1)


async def test_a_window_too_large_is_narrowed_and_the_run_retried(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    The configured window is a claim Loom cannot check, so it reacts instead.

    A model may be clamped to whatever it was trained for regardless of what the
    server permits, and there is no way to ask it. Failing the session over a
    number the user could only have guessed at loses the step for nothing.
    """
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    from services.events import WarningEvent

    session, events = prepare(monkeypatch, [], git_repo)
    session._coder_context_window = 128_048
    session._session.record_classification(True)

    import agents.coder as coder_module

    attempts: list[int] = []
    original = coder_module.coder_agent.run

    async def overflow_once(*args: Any, **kwargs: Any) -> Any:
        attempts.append(session._coder_context_window)
        if len(attempts) == 1:
            raise UnexpectedModelBehavior(
                'Model token limit (8192) exceeded before any response was generated.'
            )
        return await original(*args, **kwargs)

    monkeypatch.setattr(agent_run_helper_module, 'resolve_model', lambda name: quiet_model())
    monkeypatch.setattr(coder_module.coder_agent, 'run', overflow_once)

    await session._handle_executing()

    assert attempts == [128_048, 64_024], 'the retry did not use a narrower window'
    warnings = [event for event in events if isinstance(event, WarningEvent)]
    assert warnings, 'the user was not told the window was too large'
    assert 'smaller than' in warnings[0].content


async def test_tool_retry_exhaustion_retries_once_after_compaction(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    One retry can recover from a malformed tool call without abandoning the step.
    """
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    session, _ = prepare(monkeypatch, [], git_repo)
    session._session.record_classification(True)

    import agents.coder as coder_module

    attempts: list[int] = []

    original = coder_module.coder_agent.run

    async def fails_once(*args: Any, **kwargs: Any) -> Any:
        """
        Raise the exhausted-tool error once, then allow normal execution.

        :param args: Positional run arguments.
        :param kwargs: Keyword run arguments.
        :returns: The normal agent result after recovery.
        :raises UnexpectedModelBehavior: For the simulated first exhausted tool call.
        """
        attempts.append(1)
        if len(attempts) == 1:
            raise UnexpectedModelBehavior("Tool 'edit_file' exceeded max retries count of 5.")
        return await original(*args, **kwargs)

    compactions: list[int] = []

    async def compact(*_: object) -> None:
        """
        Record the one forced compaction before retrying.

        :returns: No retained history for this mocked failed run.
        """
        compactions.append(1)
        return None

    monkeypatch.setattr(agent_run_helper_module, 'resolve_model', lambda name: quiet_model())
    monkeypatch.setattr(coder_module.coder_agent, 'run', fails_once)
    monkeypatch.setattr(session, '_compact_after_tool_retry_exhaustion', compact)

    await session._handle_executing()

    assert len(attempts) == 2
    assert compactions == [1]


async def test_persistent_tool_retry_exhaustion_marks_the_step_incomplete(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A second exhausted-tool failure is recorded and execution continues safely.
    """
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    from services.events import WarningEvent

    session, events = prepare(monkeypatch, [], git_repo)
    session._session.record_classification(True)

    import agents.coder as coder_module

    attempts: list[int] = []
    compactions: list[int] = []

    async def always_fails(*_: object, **__: object) -> None:
        """
        Simulate a tool whose retries remain exhausted after recovery.

        :raises UnexpectedModelBehavior: Always, for the tool retry exhaustion.
        """
        attempts.append(1)
        raise UnexpectedModelBehavior("Tool 'edit_file' exceeded max retries count of 5.")

    async def compact(*_: object) -> None:
        """
        Record the sole recovery compaction.

        :returns: No retained history for this mocked failed run.
        """
        compactions.append(1)
        return None

    monkeypatch.setattr(agent_run_helper_module, 'resolve_model', lambda name: quiet_model())
    monkeypatch.setattr(coder_module.coder_agent, 'run', always_fails)
    monkeypatch.setattr(session, '_compact_after_tool_retry_exhaustion', compact)

    state: SessionState = await session._handle_executing()

    warnings: list[WarningEvent] = [event for event in events if isinstance(event, WarningEvent)]
    assert state is SessionState.reviewing
    assert attempts == [1, 1]
    assert compactions == [1]
    assert len(session._session.step_summaries) == 1
    assert 'incomplete because a tool exhausted its retry budget' in session._session.step_summaries[0]
    assert 'compacted the request and retried once' in warnings[-1].content


async def test_tool_retry_exhaustion_is_tolerated_for_an_unparsed_plan(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Whole-plan execution has the same recovery path when no step object exists.
    """
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    from services.events import WarningEvent

    session, events = prepare(monkeypatch, [], git_repo)
    session._plan = ParsedPlan(steps=[])

    import agents.coder as coder_module

    async def always_fails(*_: object, **__: object) -> None:
        """
        Simulate persistent tool retry exhaustion.

        :raises UnexpectedModelBehavior: Always, for the tool retry exhaustion.
        """
        raise UnexpectedModelBehavior("Tool 'search_code' exceeded max retries count of 5.")

    async def compact(*_: object) -> None:
        """
        Provide the single recovery compaction expected by the executor.

        :returns: No retained history for this mocked failed run.
        """
        return None

    monkeypatch.setattr(coder_module.coder_agent, 'run', always_fails)
    monkeypatch.setattr(session, '_compact_after_tool_retry_exhaustion', compact)

    state: SessionState = await session._handle_executing()

    warnings: list[WarningEvent] = [event for event in events if isinstance(event, WarningEvent)]
    assert state is SessionState.reviewing
    assert len(warnings) == 1
    assert warnings[0].content.startswith('Step 1 could not complete')


async def test_a_provider_context_rejection_is_narrowed_and_retried(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A hard 400 context rejection carries the same recovery signal as an empty generation.
    """
    from pydantic_ai.exceptions import ModelHTTPError

    session, _ = prepare(monkeypatch, [], git_repo)
    session._coder_context_window = 128_048
    session._session.record_classification(True)

    import agents.coder as coder_module

    attempts: list[int] = []
    original = coder_module.coder_agent.run

    async def reject_once(*args: Any, **kwargs: Any) -> Any:
        attempts.append(session._coder_context_window)
        if len(attempts) == 1:
            raise ModelHTTPError(
                400,
                'ollama:model',
                {'error': 'requested context length exceeds the model limit'},
            )
        return await original(*args, **kwargs)

    monkeypatch.setattr(agent_run_helper_module, 'resolve_model', lambda name: quiet_model())
    monkeypatch.setattr(coder_module.coder_agent, 'run', reject_once)

    await session._handle_executing()

    assert attempts == [128_048, 64_024]


async def test_an_unworkable_window_is_explained_before_the_first_step(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    The check existed and was never called, so nobody was ever told.

    The provider's own error names neither the window nor what filled it, so a
    run that cannot possibly work reads as a bug in Loom rather than as a
    setting to change.
    """
    from services.events import WarningEvent

    session, events = prepare(monkeypatch, [], git_repo)
    session._plan = three_step_plan()
    session._coder_context_window = 4_096
    monkeypatch.setattr(agent_run_helper_module, 'resolve_model', lambda name: quiet_model())

    await session._handle_executing()

    warnings = [event for event in events if isinstance(event, WarningEvent)]
    assert warnings, 'an unworkable window was not reported at all'
    assert 'instructions and tool definitions' in warnings[0].content
    assert 'practical minimum' in warnings[0].content or 'minimum' in warnings[0].content


async def test_a_workable_window_is_not_complained_about(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A warning on every run would train the user to ignore it.
    """
    from services.events import WarningEvent

    session, events = prepare(monkeypatch, [], git_repo)
    session._plan = three_step_plan()
    session._coder_context_window = 64_000

    async def ignore_provider_usage(provider_input_tokens: int) -> None:
        """
        Keep this fixed-overhead test independent of provider usage simulation.

        :param provider_input_tokens: The reported usage, deliberately ignored.
        """
        _ = provider_input_tokens

    monkeypatch.setattr(session, '_reconcile_provider_usage', ignore_provider_usage)
    monkeypatch.setattr(agent_session_module, 'resolve_model', lambda name: quiet_model())

    await session._handle_executing()

    complaints = [
        event
        for event in events
        if isinstance(event, WarningEvent) and 'tool definitions' in event.content
    ]
    assert complaints == []


async def test_narrowing_stops_once_it_cannot_help(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Halving five times and failing identically each time is not progress.

    Every request carries the instructions and tool schemas whatever the window
    is. Below that floor the model has nowhere to put the work, so narrowing
    again would fail the same way while looking like it were doing something.
    """
    from services.events import WarningEvent

    session, events = prepare(monkeypatch, [], git_repo)
    session._coder_context_window = 8_003

    narrowed: bool = await session._overflow_recovery.narrow_after_overflow(8_003)

    assert narrowed is False, 'narrowing should have stopped'
    warnings = [event for event in events if isinstance(event, WarningEvent)]
    assert warnings
    assert 'instructions and tool definitions' in warnings[-1].content


async def test_narrowing_still_happens_while_it_can_help(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    The floor must not stop a narrowing that would have worked.
    """
    session, _ = prepare(monkeypatch, [], git_repo)
    session._coder_context_window = 128_048

    assert await session._overflow_recovery.narrow_after_overflow(128_048) is True
    assert session._coder_context_window == 64_024


async def test_context_usage_is_reported_with_both_readings(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    The count and the share answer different questions.

    The count says how much room the work is taking; the share says how close it
    is to running out. A reading is published per request so the number is
    measured rather than inferred from a failure after the fact.
    """
    from pydantic_ai_harness.compaction import ContextUsage

    from services.events import ContextUsageEvent

    session, events = prepare(monkeypatch, [], git_repo)

    await session._report_context_usage(
        ContextUsage(used_tokens=79_400, window_tokens=128_048, resolved=True)
    )

    readings = [event for event in events if isinstance(event, ContextUsageEvent)]
    assert len(readings) == 1
    assert readings[0].used_tokens == 79_400
    assert readings[0].window_tokens == 128_048
    assert readings[0].resolved is True


async def test_an_unresolved_window_is_passed_through(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A percentage taken against a fallback window is a guess, and must say so.
    """
    from pydantic_ai_harness.compaction import ContextUsage

    from services.events import ContextUsageEvent

    session, events = prepare(monkeypatch, [], git_repo)

    await session._report_context_usage(
        ContextUsage(used_tokens=500, window_tokens=200_000, resolved=False)
    )

    readings = [event for event in events if isinstance(event, ContextUsageEvent)]
    assert readings[0].resolved is False


async def test_provider_input_truncation_narrows_once_with_concrete_counts(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A server processing half the estimated prompt corrects the next request.
    """
    from pydantic_ai_harness.compaction import ContextUsage

    from services.events import WarningEvent

    session, events = prepare(monkeypatch, [], git_repo)
    await session._report_context_usage(
        ContextUsage(used_tokens=900, window_tokens=32_768, resolved=True), fixed_overhead=100
    )

    await session._reconcile_provider_usage(400)
    await session._reconcile_provider_usage(400)

    warnings: list[WarningEvent] = [event for event in events if isinstance(event, WarningEvent)]
    assert len(warnings) == 1
    assert '400 prompt tokens' in warnings[0].content
    assert '1,000' in warnings[0].content
    assert 'reduced its context window from 32,768 to 16,384' in warnings[0].content
    assert session._coder_context_window == 16_384
    assert session._planner_context_window == 16_384


async def test_new_provider_truncation_observations_each_narrow_the_window(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Each request can independently prove the serving window remains too large.
    """
    from pydantic_ai_harness.compaction import ContextUsage

    session, _ = prepare(monkeypatch, [], git_repo)
    session._coder_context_window = 128_048
    session._planner_context_window = 128_048

    await session._report_context_usage(
        ContextUsage(used_tokens=900, window_tokens=128_048, resolved=True), fixed_overhead=100
    )
    await session._reconcile_provider_usage(400)
    await session._report_context_usage(
        ContextUsage(used_tokens=900, window_tokens=64_024, resolved=True), fixed_overhead=100
    )
    await session._reconcile_provider_usage(400)

    assert session._coder_context_window == 32_012
    assert session._planner_context_window == 32_012


async def test_forced_summary_precedes_narrowing_and_retries_with_its_history(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    A recovered summary is retried before a same-window reduction is considered.
    """
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    from services.compaction import RequestHistoryCapture

    session, _ = prepare(monkeypatch, [], git_repo)
    history: RequestHistoryCapture = RequestHistoryCapture()
    compacted: list[ModelMessage] = [ModelResponse(parts=[TextPart(content='Summary.')])]
    calls: list[str] = []

    async def summarize(
        captured: RequestHistoryCapture,
        budget: int,
        run_path: CompactionRunPath,
    ) -> list[ModelMessage] | None:
        """
        Record one successful forced-summary recovery.

        :param captured: The failed request history, unused by this stand-in.
        :param budget: The unchanged recovery budget, unused by this stand-in.
        :param run_path: The recovered agent path, unused by this stand-in.
        :returns: The history for the retry.
        """
        _ = captured
        _ = budget
        _ = run_path
        calls.append('summary')
        return compacted

    async def run(message_history: list[ModelMessage] | None) -> str:
        """
        Fail once, then verify that the summary was supplied to the retry.

        :param message_history: The optional history supplied to this attempt.
        :returns: The recovered output.
        :raises UnexpectedModelBehavior: On the simulated first context overflow.
        """
        if message_history is None:
            calls.append('first-run')
            raise UnexpectedModelBehavior(
                'Model token limit (8192) exceeded before any response was generated.'
            )
        calls.append('summary-retry')
        assert message_history == compacted
        return 'recovered'

    async def narrow(budget: int) -> bool:
        """
        Record an unexpected narrowing attempt.

        :param budget: The rejected budget, unused by this stand-in.
        :returns: Always False because the call is a test failure.
        """
        _ = budget
        calls.append('narrow')
        return False

    monkeypatch.setattr(session._overflow_recovery, 'summarize_after_overflow', summarize)
    monkeypatch.setattr(session._overflow_recovery, 'narrow_after_overflow', narrow)

    result: str = await session._overflow_recovery.run_with_overflow_recovery(
        run,
        lambda: 32_768,
        history,
        CompactionRunPath.coder_streaming,
    )

    assert result == 'recovered'
    assert calls == ['first-run', 'summary', 'summary-retry']


async def test_forced_summary_emits_a_distinct_recovery_warning(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Users can distinguish successful summary recovery from window narrowing.
    """
    from services.compaction import RequestHistoryCapture
    from services.events import WarningEvent

    session, events = prepare(monkeypatch, [], git_repo)
    history: RequestHistoryCapture = RequestHistoryCapture()
    messages: list[ModelMessage] = [
        ModelResponse(parts=[TextPart(content='Long context that needs summarizing.' * 100)])
    ]
    history._messages = messages
    compacted: list[ModelMessage] = [ModelResponse(parts=[TextPart(content='Summary.')])]

    async def summarize(*_: object, **__: object) -> list[ModelMessage]:
        """
        Replace the failed history with a shorter summary.

        :returns: The shortened history.
        """
        return compacted

    monkeypatch.setattr(context_overflow_recovery_module, 'force_summarize_history', summarize)

    result: list[ModelMessage] | None = await session._overflow_recovery.summarize_after_overflow(
        history, 32_768, CompactionRunPath.coder_streaming
    )

    warnings: list[WarningEvent] = [event for event in events if isinstance(event, WarningEvent)]
    assert result == compacted
    assert len(warnings) == 1
    assert 'summarized the preceding request' in warnings[0].content


async def test_provider_underestimate_reserves_budget_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Provider usage above the heuristic reduces later compaction budgets.
    """
    from pydantic_ai_harness.compaction import ContextUsage

    from services.events import WarningEvent

    session, events = prepare(monkeypatch, [], git_repo)
    await session._report_context_usage(
        ContextUsage(used_tokens=900, window_tokens=32_768, resolved=True), fixed_overhead=100
    )

    await session._reconcile_provider_usage(1_600)

    capability: object = session._compaction_for(
        10_000, CompactionRunPath.coder_streaming
    )
    assert getattr(capability, '_inner').context_window == 9_400
    await session._reconcile_provider_usage(2_000)
    capability = session._compaction_for(10_000, CompactionRunPath.coder_streaming)
    assert getattr(capability, '_inner').context_window == 9_000

    warnings: list[WarningEvent] = [event for event in events if isinstance(event, WarningEvent)]
    assert len(warnings) == 1
    assert '1,600 prompt tokens' in warnings[0].content
    assert '600 additional context tokens' in warnings[0].content


async def test_agreeing_or_absent_provider_usage_emits_no_warning(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Normal estimator variation and unavailable usage leave the feed unchanged.
    """
    from pydantic_ai_harness.compaction import ContextUsage

    from services.events import WarningEvent

    session, events = prepare(monkeypatch, [], git_repo)
    await session._report_context_usage(
        ContextUsage(used_tokens=900, window_tokens=32_768, resolved=True), fixed_overhead=100
    )

    await session._reconcile_provider_usage(1_200)
    await session._reconcile_provider_usage(0)

    assert not [event for event in events if isinstance(event, WarningEvent)]


async def test_provider_usage_capability_reconciles_each_successful_response(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    The session's response hook drives reconciliation from provider usage.
    """
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.usage import RequestUsage
    from pydantic_ai_harness.compaction import ContextUsage

    from services.events import WarningEvent

    session, events = prepare(monkeypatch, [], git_repo)
    await session._report_context_usage(
        ContextUsage(used_tokens=900, window_tokens=32_768, resolved=True), fixed_overhead=100
    )
    response: ModelResponse = ModelResponse(
        parts=[TextPart(content='Done.')], usage=RequestUsage(input_tokens=400)
    )

    returned: ModelResponse = await session._provider_usage_monitor().after_model_request(
        MagicMock(), request_context=MagicMock(), response=response
    )

    assert returned is response
    warnings: list[WarningEvent] = [event for event in events if isinstance(event, WarningEvent)]
    assert len(warnings) == 1
