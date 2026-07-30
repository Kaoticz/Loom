from __future__ import annotations

import json
import os
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from api.v1.agent_session import AgentSession, SessionServices
from agents.coder import AgentDeps
from schemas.agent_messages import MessageType
from services.code_search import CodeSearchService
from services.conversation_history import ConversationHistory
from services.docker_sandbox import DockerSandboxManager
from services.event_bus import AgentEvent, ErrorEvent, EventBus, ThoughtEvent
from services.file_editor import FileEditorService
from services.git_manager import FileChange, GitManagerService
from services.session_context import SessionContext
from services.session_fsm import SessionState, SessionStateMachine
from services.session_limits import IterationLimits
from tests.conftest import stub_embedding_service


class FakeWebSocket:
    """
    A WebSocket stand-in that replays scripted client messages.
    """

    _inbound: list[str]
    sent: list[dict[str, Any]]

    def __init__(self, inbound: list[str]) -> None:
        """
        Create a fake socket.

        :param inbound: The messages the client will send, in order.
        """
        self._inbound = list(inbound)
        self.sent = []

    async def receive_text(self) -> str:
        """
        Return the next scripted client message.

        :returns: The message text.
        :raises AssertionError: If the session asked for more input than was scripted.
        """
        if not self._inbound:
            raise AssertionError('The session requested more input than the test provided.')
        return self._inbound.pop(0)

    async def send_json(self, payload: dict[str, Any]) -> None:
        """
        Record a message sent to the client.

        :param payload: The serialised message.
        """
        self.sent.append(payload)


class FakeGit:
    """
    A git manager stand-in recording the calls a session makes.
    """

    repo_path: str
    created_branches: list[str]
    merged: list[str]
    changes: list[FileChange]
    merge_error: Exception | None

    def __init__(self, changes: list[FileChange] | None = None) -> None:
        """
        Create a fake git manager.

        :param changes: The changed files the session should observe.
        """
        self.repo_path = '/tmp/repo'
        self.created_branches = []
        self.merged = []
        self.changes = changes if changes is not None else [
            FileChange(path='a.py', insertions=3, deletions=1)
        ]
        self.merge_error = None

    async def create_branch(self, branch_name: str) -> None:
        """
        Record a branch creation.

        :param branch_name: The branch requested.
        """
        self.created_branches.append(branch_name)

    async def get_changed_files(self, base_branch: str | None = None) -> list[FileChange]:
        """
        Return the configured change list.

        :param base_branch: Ignored.
        :returns: The changed files.
        """
        return self.changes

    async def get_file_diff(self, file_path: str, base_branch: str | None = None) -> str:
        """
        Return a token diff for a file.

        :param file_path: The file to diff.
        :param base_branch: Ignored.
        :returns: A short diff body.
        """
        return f'--- a/{file_path}\n+++ b/{file_path}\n+changed\n'

    async def count_commits(self, base_branch: str | None = None) -> int:
        """
        Return a fixed commit count.

        :param base_branch: Ignored.
        :returns: The number of commits.
        """
        return 3

    async def merge_branch(self, branch_name: str, target_branch: str | None = None) -> str:
        """
        Record a merge, or raise if the test configured a failure.

        :param branch_name: The branch to merge.
        :param target_branch: Ignored.
        :returns: The branch merged into.
        :raises Exception: If a merge error was configured.
        """
        if self.merge_error is not None:
            raise self.merge_error
        self.merged.append(branch_name)
        return 'main'


class FakeIndexing:
    """
    An indexer stand-in that reports no work.
    """

    repo_path: str = '/tmp/repo'

    async def index_repository(self, progress_callback: Any = None) -> None:
        """
        Pretend to index the repository.

        :param progress_callback: Ignored.
        """
        return None


class FakeServices:
    """
    The bundle of services an `AgentSession` reaches through.
    """

    git: FakeGit
    indexing: FakeIndexing
    docker: Any
    editor: Any
    embedding: Any

    def __init__(self, git: FakeGit) -> None:
        """
        Assemble the fake services.

        :param git: The git stand-in to use.
        """
        self.git = git
        self.indexing = FakeIndexing()
        self.docker = MagicMock()
        self.editor = MagicMock()
        self.embedding = MagicMock()

        async def spawn(session_id: str) -> None:
            return None

        self.docker.spawn_sandbox = spawn


def build_session(
    inbound: list[str],
    git: FakeGit | None = None,
    limits: IterationLimits | None = None,
) -> tuple[AgentSession, FakeWebSocket, list[AgentEvent], FakeServices]:
    """
    Assemble a session driver wired to fakes.

    :param inbound: The client messages to replay.
    :param git: An optional git stand-in.
    :param limits: Optional iteration ceilings.
    :returns: The session, its socket, the events it emitted, and its services.
    """
    socket: FakeWebSocket = FakeWebSocket(inbound)
    services: FakeServices = FakeServices(git or FakeGit())
    events: EventBus = EventBus('test')
    captured: list[AgentEvent] = []

    async def collect(event: AgentEvent) -> None:
        captured.append(event)

    events.subscribe(collect)

    context: SessionContext = SessionContext()
    deps: AgentDeps = AgentDeps(
        docker=DockerSandboxManager(repo_path='/tmp/repo'),
        search=CodeSearchService(),
        editor=FileEditorService(root_path='/tmp/repo'),
        git=GitManagerService(repo_path='/tmp/repo'),
        embedding=stub_embedding_service(),
        events=events,
        session=context,
    )

    session: AgentSession = AgentSession(
        websocket=socket,  # type: ignore[arg-type]
        # The fakes implement only the members a session actually reaches for.
        services=cast(SessionServices, services),
        session_id='testsess',
        events=events,
        deps=deps,
        session=context,
        history=ConversationHistory(),
        machine=SessionStateMachine(),
        limits=limits or IterationLimits(),
    )
    return session, socket, captured, services


def stub_agents(
    monkeypatch: pytest.MonkeyPatch,
    plan_outputs: list[str],
    review_results: list[tuple[bool, str]],
) -> dict[str, list[str]]:
    """
    Replace the coder and reviewer agents with scripted stand-ins.

    :param monkeypatch: Fixture used to swap the agents out.
    :param plan_outputs: The successive outputs the coder should return.
    :param review_results: The successive verdicts the reviewer should return.
    :returns: A record of the prompts each agent received.
    """
    import agents.coder as coder_module
    import agents.reviewer as reviewer_module
    import api.v1.agent_session as module
    from agents.reviewer import ReviewResult

    seen: dict[str, list[str]] = {'coder': [], 'reviewer': []}
    plans: list[str] = list(plan_outputs)
    reviews: list[tuple[bool, str]] = list(review_results)

    async def fake_run(prompt: str, **kwargs: Any) -> Any:
        seen['coder'].append(prompt)
        result: Any = MagicMock()
        result.output = plans.pop(0) if plans else '1. Fallback step'
        return result

    async def fake_review(prompt: str, **kwargs: Any) -> Any:
        seen['reviewer'].append(prompt)
        approved, comments = reviews.pop(0) if reviews else (True, 'Looks fine.')
        result: Any = MagicMock()
        result.output = ReviewResult(approved=approved, comments=comments)
        return result

    monkeypatch.setattr(coder_module.coder_agent, 'run', fake_run)
    monkeypatch.setattr(reviewer_module.reviewer_agent, 'run', fake_review)
    monkeypatch.setattr(
        module.AgentSession, '_handle_executing', _fake_execute, raising=True
    )
    return seen


async def _fake_execute(self: AgentSession) -> SessionState:
    """
    Stand in for the streaming execution phase.

    :param self: The session being driven.
    :returns: The reviewing state.
    """
    self._session.record_modification('a.py')
    self._session.record_verification(True)
    self._session.request_completion('Did the work.')
    return SessionState.reviewing


def opening(prompt: str = 'Add a timeout') -> str:
    """
    Build the opening client message.

    :param prompt: The task description.
    :returns: The serialised message.
    """
    return json.dumps({'prompt': prompt, 'repo_path': ''})


def types_of(events: list[AgentEvent]) -> list[str]:
    """
    List the message types a run produced.

    :param events: The captured events.
    :returns: The wire type of each event, in order.
    """
    from schemas.agent_messages import to_message

    return [str(to_message(event).type.value) for event in events]


async def test_happy_path_reaches_merged(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Approve the plan, approve the review, merge: the session ends merged.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, events, services = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    assert await session.run() is SessionState.merged
    assert services.git.created_branches == ['ai-task-testsess']
    assert services.git.merged == ['ai-task-testsess']

    kinds: list[str] = types_of(events)
    assert MessageType.plan.value in kinds
    assert MessageType.review.value in kinds
    assert kinds.count(MessageType.done.value) == 2


async def test_rejected_plan_is_replanned_with_the_reason(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A rejection reason reaches the agent's next planning prompt.
    """
    seen = stub_agents(
        monkeypatch, ['1. First attempt', '1. Second attempt'], [(True, 'Fine.')]
    )
    session, _, _, _ = build_session([
        opening(),
        json.dumps({'approved': False, 'reason': 'You forgot the database migration.'}),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    assert await session.run() is SessionState.merged
    assert len(seen['coder']) == 2
    assert 'You forgot the database migration.' in seen['coder'][1]


async def test_rejected_review_loops_back_through_planning(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Reviewer criticism is fed into the next plan.
    """
    seen = stub_agents(
        monkeypatch,
        ['1. First attempt', '1. Second attempt'],
        [(False, 'The empty case is unhandled.'), (True, 'Better.')],
    )
    session, _, events, _ = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    assert await session.run() is SessionState.merged
    assert len(seen['reviewer']) == 2
    assert 'The empty case is unhandled.' in seen['coder'][1]
    assert types_of(events).count(MessageType.review.value) == 2


async def test_user_can_reject_the_finished_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Rejecting at the merge prompt sends the session back for more work.
    """
    seen = stub_agents(
        monkeypatch, ['1. First', '1. Second'], [(True, 'Fine.'), (True, 'Fine again.')]
    )
    session, _, _, services = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': False, 'reason': 'Please also update the README.'}),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    assert await session.run() is SessionState.merged
    assert 'Please also update the README.' in seen['coder'][1]
    assert services.git.merged == ['ai-task-testsess']


async def test_plan_retry_cap_pauses_for_the_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Repeated plan rejections pause and ask the user what to do.
    """
    stub_agents(monkeypatch, ['1. Attempt'] * 6, [(True, 'Fine.')])
    session, _, events, _ = build_session(
        [
            opening(),
            json.dumps({'approved': False, 'reason': 'no'}),
            json.dumps({'approved': False, 'reason': 'no'}),
            json.dumps({'action': 'abort'}),
        ],
        limits=IterationLimits(max_plan_retries=2),
    )

    from services.event_bus import ThoughtEvent

    assert await session.run() is SessionState.failed
    thoughts: str = ' '.join(
        event.content for event in events if isinstance(event, ThoughtEvent)
    )
    assert 'rejected 2 times' in thoughts


async def test_auto_approve_stops_asking(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Choosing auto-approve lets the session run without further plan prompts.
    """
    stub_agents(monkeypatch, ['1. Attempt'] * 4, [(True, 'Fine.')])
    session, socket, _, _ = build_session(
        [
            opening(),
            json.dumps({'approved': False, 'reason': 'no'}),
            json.dumps({'action': 'auto_approve'}),
            json.dumps({'approved': True}),
        ],
        limits=IterationLimits(max_plan_retries=1),
    )

    assert await session.run() is SessionState.merged


async def test_total_iteration_ceiling_fails_the_session(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The hard ceiling ends the session rather than looping forever.
    """
    stub_agents(
        monkeypatch,
        ['1. Attempt'] * 10,
        [(False, 'Still wrong.')] * 10,
    )
    session, _, events, _ = build_session(
        [opening()] + [json.dumps({'approved': True})] * 10,
        limits=IterationLimits(max_total_iterations=2, max_review_retries=9),
    )

    assert await session.run() is SessionState.failed
    assert MessageType.error.value in types_of(events)


async def test_no_changes_is_reported_as_a_rejection(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A session that changed nothing is not sent to the reviewer as approved.
    """
    stub_agents(monkeypatch, ['1. Attempt'], [(True, 'unused')])
    session, _, events, _ = build_session(
        [
            opening(),
            json.dumps({'approved': True}),
            json.dumps({'action': 'abort'}),
        ],
        git=FakeGit(changes=[]),
        limits=IterationLimits(max_review_retries=1),
    )

    from services.event_bus import ReviewEvent

    assert await session.run() is SessionState.failed
    reviews = [event for event in events if isinstance(event, ReviewEvent)]
    assert any('nothing to review' in review.comments for review in reviews)


async def test_merge_conflict_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A failed merge reports the problem and leaves the work on the branch.
    """
    stub_agents(monkeypatch, ['1. Attempt'], [(True, 'Fine.')])
    git: FakeGit = FakeGit()
    git.merge_error = RuntimeError('CONFLICT (content): merge conflict in a.py')

    session, _, events, _ = build_session(
        [opening(), json.dumps({'approved': True}), json.dumps({'approved': True})],
        git=git,
    )

    from services.event_bus import ErrorEvent

    assert await session.run() is SessionState.failed
    errors = [event for event in events if isinstance(event, ErrorEvent)]
    assert len(errors) == 1
    assert 'conflict' in errors[0].content
    assert 'still on the branch' in errors[0].content
    assert git.merged == []


async def test_review_prompt_is_built_from_the_cumulative_diff(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The reviewer receives the task, the changed file list and the diffs.
    """
    seen = stub_agents(monkeypatch, ['1. Attempt'], [(True, 'Fine.')])
    session, _, _, _ = build_session([
        opening('Add a timeout to the sandbox'),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    await session.run()

    prompt: str = seen['reviewer'][0]
    assert 'Add a timeout to the sandbox' in prompt
    assert 'Files changed (1)' in prompt
    assert 'a.py (+3/-1)' in prompt
    assert '+changed' in prompt


async def test_re_review_receives_the_previous_verdict(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The second review is told what the first one asked for.
    """
    seen = stub_agents(
        monkeypatch,
        ['1. First attempt', '1. Second attempt'],
        [(False, 'The empty case is unhandled.'), (True, 'Now correct.')],
    )
    session, _, _, _ = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    assert await session.run() is SessionState.merged
    assert len(seen['reviewer']) == 2
    assert 'The empty case is unhandled.' not in seen['reviewer'][0]
    assert 'The empty case is unhandled.' in seen['reviewer'][1]
    assert 'has been addressed' in seen['reviewer'][1]


async def test_review_only_fetches_diffs_it_will_use(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A large change set does not read every diff before budgeting.
    """
    changes: list[FileChange] = [
        FileChange(path=f'file_{index}.py', insertions=5_000, deletions=0)
        for index in range(30)
    ]
    git: FakeGit = FakeGit(changes=changes)
    fetched: list[str] = []

    async def fetch(file_path: str, base_branch: str | None = None) -> str:
        fetched.append(file_path)
        return 'x' * 20_000

    git.get_file_diff = fetch  # type: ignore[method-assign]

    stub_agents(monkeypatch, ['1. Attempt'], [(True, 'Fine.')])
    session, _, _, _ = build_session(
        [opening(), json.dumps({'approved': True}), json.dumps({'approved': True})],
        git=git,
    )

    assert await session.run() is SessionState.merged
    assert len(fetched) < len(changes)


async def test_repo_path_does_not_leak_into_the_environment(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A session's repository path never reaches the process environment.

    Services read REPO_PATH only when they are constructed, which happens as the
    socket connects. Exporting it here cannot affect the running session, but it
    would become the default for the next session that omits a repo_path,
    silently pointing it at the previous session's repository.
    """
    monkeypatch.delenv('REPO_PATH', raising=False)

    session, _, _, services = build_session([
        json.dumps({'prompt': 'do something', 'repo_path': '/repos/project-a'})
    ])

    await session._receive_task()

    assert os.getenv('REPO_PATH') is None
    assert services.git.repo_path == '/repos/project-a'


async def test_repo_path_is_applied_to_every_service(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Each service is repointed directly rather than through shared configuration.
    """
    monkeypatch.delenv('REPO_PATH', raising=False)

    session, _, _, services = build_session([
        json.dumps({'prompt': 'do something', 'repo_path': '/repos/project-b'})
    ])

    await session._receive_task()

    assert services.docker.repo_path == '/repos/project-b'
    assert services.editor.root_path == '/repos/project-b'
    assert services.git.repo_path == '/repos/project-b'
    assert services.indexing.repo_path == '/repos/project-b'


async def test_failure_leaves_the_machine_in_the_failed_state(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An exception escaping a handler still moves the machine to failed.

    Leaving the state reading `executing` records a lie, which matters as soon
    as anything resumes from it.
    """
    stub_agents(monkeypatch, ['1. Attempt'], [(True, 'Fine.')])
    session, _, _, services = build_session([opening(), json.dumps({'approved': True})])

    async def explode(base_branch: str | None = None) -> list[FileChange]:
        raise RuntimeError('the database went away')

    services.git.get_changed_files = explode  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        await session.run()

    assert session._machine.current is SessionState.failed
    assert session._machine.is_terminal is True


async def test_empty_repository_stops_for_confirmation(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An empty repository is reported and the session waits rather than planning.

    Planning against nothing yields a plan the agent cannot carry out, with no
    indication of why.
    """
    stub_agents(monkeypatch, ['1. Attempt'], [(True, 'Fine.')])
    session, _, events, _ = build_session([opening(), json.dumps({'approved': False})])
    monkeypatch.setattr(type(session), '_repository_is_empty', _always_empty)

    assert await session.run() is SessionState.failed

    said: str = ' '.join(
        event.content for event in events if isinstance(event, ThoughtEvent)
    )
    assert 'repository looks' in said
    errors = [event for event in events if isinstance(event, ErrorEvent)]
    assert errors and 'nothing in the repository' in errors[0].content


async def test_empty_repository_can_be_continued(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The user may choose to carry on with an empty repository.
    """
    stub_agents(monkeypatch, ['1. Create the first file'], [(True, 'Fine.')])
    session, _, _, services = build_session([
        opening(),
        json.dumps({'approved': True}),   # continue despite the empty repository
        json.dumps({'approved': True}),   # approve the plan
        json.dumps({'approved': True}),   # approve the review
        json.dumps({'approved': True}),   # merge
    ])
    monkeypatch.setattr(type(session), '_repository_is_empty', _always_empty)

    assert await session.run() is SessionState.merged
    assert services.git.created_branches == ['ai-task-testsess']


async def _always_empty(self: AgentSession) -> bool:
    """
    Stand in for the emptiness check.

    :param self: The session being driven.
    :returns: Always True.
    """
    return True


async def test_a_lost_client_stops_the_session_between_states(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A session whose client has gone stops rather than continuing to work.

    Otherwise the agent keeps calling the model, editing files and committing for
    a browser that is no longer there.
    """
    from asyncio import Event

    stub_agents(monkeypatch, ['1. Attempt'], [(True, 'Fine.')])
    session, _, _, services = build_session([opening()])

    lost: Event = Event()
    session._connection_lost = lost
    lost.set()

    assert await session.run() is SessionState.failed
    assert services.git.created_branches == [], 'no work should have started'


async def test_a_present_client_is_not_treated_as_lost(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A live connection runs to completion as normal.
    """
    stub_agents(monkeypatch, ['1. Attempt'], [(True, 'Fine.')])
    session, _, _, _ = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    assert await session.run() is SessionState.merged


async def test_losing_the_client_mid_execution_unwinds(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Execution stops promptly rather than running to the end of the phase.

    Execution is the expensive state, so waiting for it to finish would keep
    calling the model for nobody.
    """
    from asyncio import Event

    from api.v1.agent_session import ClientGone

    session, _, _, _ = build_session([opening()])
    lost: Event = Event()
    session._connection_lost = lost

    async def executing_that_notices() -> SessionState:
        lost.set()
        raise ClientGone()

    monkeypatch.setattr(session, '_handle_executing', executing_that_notices)
    session._machine.transition_to(SessionState.planning)
    session._machine.transition_to(SessionState.waiting_plan_approval)
    session._machine.transition_to(SessionState.executing)

    assert await session.run() is SessionState.failed
