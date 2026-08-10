from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from collections.abc import AsyncIterator

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

import api.v1.agent_session as agent_session_module
from api.v1.agent_session import AgentSession
from services.event_bus import (
    AgentEvent,
    DoneEvent,
    SessionStatus,
    SystemEvent,
    ThoughtEvent,
    ToolActionEvent,
)
from services.git_manager import GitManagerService
from services.session_fsm import SessionState
from tests.test_agent_session import FakeGit, build_session, opening


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
    monkeypatch.setattr(agent_session_module, 'resolve_model', lambda name: model)


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
    session._plan = session._plan.model_copy(update={'steps': ['Create the marker file']})
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


async def test_agent_that_never_completes_is_still_reviewed(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    An agent that stops without calling complete_task still reaches review.

    Its work should be judged rather than discarded, and the user told why.
    """
    responses: list[ModelResponse] = [
        ModelResponse(parts=[TextPart(content='I think that is enough.')]),
    ]

    state, events = await run_execution(monkeypatch, responses, git_repo)

    assert state is SessionState.reviewing
    announced: str = ' '.join(
        event.content for event in events if isinstance(event, SystemEvent)
    )
    assert 'without calling complete_task' in announced


async def test_executing_phase_is_announced_with_its_status(
    monkeypatch: pytest.MonkeyPatch, git_repo: GitManagerService
) -> None:
    """
    Entering execution is announced as a system message stating the phase.

    Requirement 3.1: the announcement is `SYSTEM`, and it names the phase rather
    than leaving the client to recognise the wording.
    """
    from services.event_bus import SessionStatus

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
    monkeypatch.setattr(agent_session_module, 'resolve_model', lambda name: recording_model(offered))
    session._session.record_classification(False)

    await session._handle_answering()

    assert offered, 'the model was never called'
    for tools in offered:
        assert 'create_file' not in tools
        assert 'edit_file' not in tools
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
    monkeypatch.setattr(agent_session_module, 'resolve_model', lambda name: recording_model(offered))
    session._session.record_classification(True)

    await session._handle_executing()

    assert offered, 'the model was never called'
    assert 'create_file' in offered[0]
    assert 'edit_file' in offered[0]
    assert 'run_test_command' in offered[0]
