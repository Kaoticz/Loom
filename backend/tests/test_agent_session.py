from __future__ import annotations

import json
from collections.abc import Mapping
import os
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from schemas.agent_commands import CommandAction, SessionCommand
from api.v1.agent_session import AgentSession, SessionServices, StopRequested
from agents.coder import AgentDeps
from database.enums import TaskStatus
from schemas.agent_messages import MessageType
from services.code_search import CodeSearchService
from services.projects import normalise_project_path
from services.conversation_history import ConversationHistory
from services.docker_sandbox import DockerSandboxManager
from services.event_bus import (
    INDEXING_PROGRESS_KEY,
    AgentEvent,
    DoneEvent,
    ErrorEvent,
    EventBus,
    SandboxUnavailableEvent,
    SessionStatus,
    SystemEvent,
    ThoughtEvent,
    WarningEvent,
)
from services.indexing_progress import ProgressReporter
from services.file_editor import FileEditorService
from services.git_manager import CommitResult, FileChange, GitManagerService
from services.session_context import SessionContext
from services.session_fsm import SessionState, SessionStateMachine
from services.session_limits import IterationLimits
from services.session_registry import SessionRegistry
from tests.conftest import TEST_REPO_PATH, stub_embedding_service


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

    `dirty` answers `has_uncommitted_changes`, which the session consults when a
    user claims to have committed by hand. False by default: a fake repository
    with no writes to it has nothing outstanding.
    """

    repo_path: str
    created_branches: list[str]
    merged: list[str]
    changes: list[FileChange]
    merge_error: Exception | None
    dirty: bool

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
        self.dirty = False

    async def create_branch(self, branch_name: str) -> None:
        """
        Record a branch creation.

        :param branch_name: The branch requested.
        """
        self.created_branches.append(branch_name)

    async def has_uncommitted_changes(self) -> bool:
        """
        Whether the fake working tree has anything outstanding.

        :returns: Whatever `dirty` has been set to.
        """
        return self.dirty

    async def commit_changes_with_retry(self, message: str) -> CommitResult:
        """
        Report a clean commit without touching a real repository.

        :param message: The commit message, ignored.
        :returns: A result with nothing to commit and no error.
        """
        return CommitResult(commit=None, error=None)

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
    An indexer stand-in that reports whatever work it was given.

    `files` is empty by default, so a session that does not care about indexing
    sees the same do-nothing behaviour as before. Setting it makes the fake drive
    a real `ProgressReporter`, which is what lets a test check that progress
    reaches the client the way the production indexer would send it.
    """

    repo_path: str = '/tmp/repo'
    files: list[str]
    chunk_lines: int | None

    @property
    def project(self) -> str:
        """
        Mirror the real service, which derives this rather than caching it.

        :returns: The normalised path identifying the current project.
        """
        return normalise_project_path(self.repo_path)

    def __init__(self) -> None:
        """
        Create an indexer that reports no work.
        """
        self.files = []
        self.chunk_lines = None

    async def index_repository(self, progress_callback: Any = None) -> None:
        """
        Pretend to index the repository, reporting progress for any configured files.

        :param progress_callback: Where to report progress, if anywhere.
        """
        if progress_callback is None or not self.files:
            return None

        reporter: ProgressReporter = ProgressReporter(
            len(self.files), progress_callback, min_interval=0.0
        )
        await reporter.start(f'Indexing {len(self.files)} files...')
        for path in self.files:
            await reporter.file_completed(path)
        await reporter.finish(f'Indexing complete: {reporter.completed} files processed.')
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
    search: Any
    registry: SessionRegistry

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
        self.search = MagicMock()
        # A real one rather than a mock: the session reports the start and end
        # of its task through it, and the tests below assert on what it says.
        self.registry = SessionRegistry()

        async def spawn(
            session_id: str,
            dockerfile: str = '',
            environment: Mapping[str, str] | None = None,
        ) -> None:
            """
            Pretend to start a sandbox.

            The signature mirrors `DockerSandboxManager.spawn_sandbox` exactly. It
            previously took only the session id, so the real call raised a
            TypeError that the session swallowed as a warning, and every test ran
            against a sandbox that had silently failed to start.
            """
            return None

        async def reachable() -> str | None:
            """
            Report the sandbox as usable.

            Tests that care about it being unusable override this.
            """
            return None

        self.docker.spawn_sandbox = spawn
        self.docker.check_reachable = reachable


async def no_context_window_measurement(model_name: str) -> int | None:
    """
    Avoid external Ollama requests in session tests that do not cover probing.

    :param model_name: The model that would have been measured.
    :returns: None, matching an unavailable metadata probe.
    """
    return None


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
        search=CodeSearchService(repo_path=TEST_REPO_PATH),
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
        context_window_probe=no_context_window_measurement,
    )
    return session, socket, captured, services


def stub_agents(
    monkeypatch: pytest.MonkeyPatch,
    plan_outputs: list[str],
    review_results: list[tuple[bool, str]],
    classifications: list[tuple[bool, str]] | None = None,
) -> dict[str, list[str]]:
    """
    Replace the coder, reviewer and classifying agents with scripted stand-ins.

    Also replaces `_handle_executing` wholesale with `_fake_execute`, so a test
    using this never runs the real executing state. That is what makes the
    surrounding session tests cheap, but it means the execution prompt, the
    tools the coder is offered, and anything else that state builds are not
    observable from here: `seen['coder']` records the planning prompts only.
    Assert on any of that from `test_execution_loop.py`, which drives
    `_handle_executing()` directly against a real repository.

    :param monkeypatch: Fixture used to swap the agents out.
    :param plan_outputs: The successive outputs the coder should return.
    :param review_results: The successive verdicts the reviewer should return.
    :param classifications: The successive classifier verdicts, defaulting to
        "changes are required" so that a test not concerned with the read-only
        path follows the ordinary route.
    :returns: A record of the prompts each agent received.
    """
    import agents.classifier as classifier_module
    import agents.coder as coder_module
    import agents.reviewer as reviewer_module
    import agents.planner as planner_module
    import api.v1.agent_session as module
    from agents.classifier import ClassificationResult
    from agents.reviewer import ReviewResult

    seen: dict[str, list[str]] = {'coder': [], 'reviewer': [], 'classifier': []}
    plans: list[str] = list(plan_outputs)
    reviews: list[tuple[bool, str]] = list(review_results)
    verdicts: list[tuple[bool, str]] = list(
        classifications if classifications is not None else [(True, 'The task asks for a change.')]
    )

    async def fake_classify(prompt: str, **kwargs: Any) -> Any:
        seen['classifier'].append(prompt)
        required, reason = verdicts.pop(0) if verdicts else (True, 'The task asks for a change.')
        result: Any = MagicMock()
        result.output = ClassificationResult(changes_are_required=required, reason=reason)
        return result

    async def fake_run(prompt: str, **kwargs: Any) -> Any:
        seen['coder'].append(prompt)
        result: Any = MagicMock()
        result.output = plans.pop(0) if plans else '1. Fallback step'
        return result

    async def fake_plan_text(prompt: str, **kwargs: Any) -> Any:
        # Structured output is refused, standing in for the small models this
        # fallback exists for; the text request is answered. Before the fallback
        # moved onto the planner, the structured call failed by reaching for a
        # model that was not there, which is what drove these tests down the
        # text path. Refusing explicitly keeps that behaviour without the
        # network attempt.
        if kwargs.get('output_type') is not str:
            raise RuntimeError('this model does not do structured output')
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

    monkeypatch.setattr(classifier_module.classifier_agent, 'run', fake_classify)
    monkeypatch.setattr(coder_module.coder_agent, 'run', fake_run)
    monkeypatch.setattr(planner_module.planner_agent, 'run', fake_plan_text)
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

    # Aborting is the user stopping, not the work breaking, so it is recorded
    # the same way the stop button is.
    assert await session.run() is SessionState.cancelled
    announced: str = ' '.join(
        event.content for event in events if isinstance(event, SystemEvent)
    )
    assert 'rejected 2 times' in announced
    # Requirement 3.3: this is Loom talking about the session, so it must not
    # arrive labelled as the model's reasoning.
    assert not any(
        'rejected 2 times' in event.content
        for event in events
        if isinstance(event, ThoughtEvent)
    )


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

    # The user aborted at the cap, which is a cancellation rather than a failure.
    assert await session.run() is SessionState.cancelled
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
    # The search service too. Left behind, the agent would read the project
    # named by REPO_PATH while the indexer wrote to this one, and every search
    # would come back empty rather than failing.
    assert services.search.repo_path == '/repos/project-b'


async def test_the_indexer_and_the_search_agree_on_the_project(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Whatever the session is pointed at, both sides name the same project.

    They are repointed independently, so nothing but this stops them drifting
    apart — and a drift produces empty searches rather than an error.
    """
    from services.code_search import CodeSearchService
    from services.indexing import IndexingService
    from unittest.mock import MagicMock as Mock

    monkeypatch.delenv('REPO_PATH', raising=False)

    indexing: IndexingService = IndexingService(repo_path='/repos/first', embedding=Mock())
    search: CodeSearchService = CodeSearchService(repo_path='/repos/first')

    assert indexing.project == search.project

    # As a start message naming another repository would do.
    indexing.repo_path = '/repos/second/'
    search.repo_path = '/repos/second'

    assert indexing.project == search.project, 'a cached project would still name the first'


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


async def test_a_finished_session_reports_that_its_task_ended(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The registry learns the task is over as soon as the machine terminates.

    The socket may well stay open afterwards, so anything waiting for the
    connection to close would keep the repository blocked for as long as the
    user left the tab open.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, _, services = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])
    await services.registry.acquire('testsess')

    assert await session.run() is SessionState.merged
    assert services.registry.has_active_task is False
    # The slot is released by the socket controller, not by the session.
    assert services.registry.is_busy is True


async def test_a_crashed_session_does_not_leave_a_task_running(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An exception escaping the loop still clears the working flag.

    Otherwise a single crash would block index deletion for the lifetime of the
    process, with nothing left running to explain why.
    """
    stub_agents(monkeypatch, ['1. Attempt'], [(True, 'Fine.')])
    session, _, _, services = build_session([opening(), json.dumps({'approved': True})])
    await services.registry.acquire('testsess')

    async def explode(base_branch: str | None = None) -> list[FileChange]:
        raise RuntimeError('the database went away')

    services.git.get_changed_files = explode  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        await session.run()

    assert services.registry.has_active_task is False


async def test_a_prompt_marks_the_task_as_under_way(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The working flag is set once a prompt arrives, not when the socket connects.

    Checked part way through rather than at the end, since by then the task has
    finished and the flag has legitimately been cleared again.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, _, services = build_session([opening(), json.dumps({'approved': True})])
    await services.registry.acquire('testsess')

    assert services.registry.has_active_task is False, 'connecting is not working'

    seen: list[bool] = []

    async def watch(base_branch: str | None = None) -> list[FileChange]:
        seen.append(services.registry.has_active_task)
        raise RuntimeError('stop here')

    services.git.get_changed_files = watch  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        await session.run()

    assert seen == [True]


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
        event.content for event in events if isinstance(event, SystemEvent)
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
    session._machine.transition_to(SessionState.classifying)
    session._machine.transition_to(SessionState.planning)
    session._machine.transition_to(SessionState.waiting_plan_approval)
    session._machine.transition_to(SessionState.executing)

    assert await session.run() is SessionState.failed


async def test_indexing_progress_reaches_the_client_as_one_keyed_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The session reports indexing as keyed system events, not as thoughts.

    Requirement 1: this covers the wiring between the indexer and the event bus,
    which the reporter's own tests and the message mapping tests both stop short
    of. Without it, a regression to one prose thought per file would go unnoticed.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, events, services = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])
    services.indexing.files = ['a.py', 'b.py', 'c.py']

    await session.run()

    progress: list[SystemEvent] = [
        event
        for event in events
        if isinstance(event, SystemEvent) and event.key == INDEXING_PROGRESS_KEY
    ]

    assert progress, 'indexing reported no progress at all'
    assert all(event.total == 3 for event in progress)
    assert progress[-1].processed == 3
    # The counter must be carried as numbers, not narrated into a thought.
    assert not any('Indexing [' in event.content for event in events if isinstance(event, ThoughtEvent))


async def test_indexing_progress_shares_one_slot_across_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Every progress update names the same slot, so the client replaces one message.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, events, services = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])
    services.indexing.files = ['a.py', 'b.py']

    await session.run()

    keys: set[str | None] = {
        event.key
        for event in events
        if isinstance(event, SystemEvent) and event.key is not None
    }
    assert keys == {INDEXING_PROGRESS_KEY}


async def test_status_announcements_are_system_not_thought(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Loom talking about the session is labelled `SYSTEM`, not `THOUGHT`.

    Requirement 3.1 and 3.3: the distinction is the point. The model's reasoning
    is the only thing that should reach the client as a thought.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, events, _ = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    await session.run()

    announced: str = ' '.join(
        event.content for event in events if isinstance(event, SystemEvent)
    )
    # 'Executing plan...' is absent here because this harness stubs out the
    # executing state; test_execution_loop covers it against the real loop.
    for phrase in (
        'Indexing repository',
        'Generating plan',
        'Starting review',
    ):
        assert phrase in announced, f'{phrase!r} was not announced as a system message'

    thoughts: str = ' '.join(
        event.content for event in events if isinstance(event, ThoughtEvent)
    )
    for phrase in ('Indexing repository', 'Generating plan', 'Starting review'):
        assert phrase not in thoughts, f'{phrase!r} still reaches the client as a thought'


async def test_announcements_state_the_phase_they_enter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The session states its phase rather than leaving it to be read out of prose.

    The client previously matched announcement wording to derive the status,
    which broke silently whenever the wording changed. The backend already holds
    this information, so it sends it.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, events, _ = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    await session.run()

    announced: dict[SessionStatus, str] = {
        event.status: event.content
        for event in events
        if isinstance(event, SystemEvent) and event.status is not None
    }

    # The executing phase is stubbed out in this harness; test_execution_loop
    # covers its announcement against the real loop.
    assert SessionStatus.indexing in announced
    assert SessionStatus.planning in announced
    assert SessionStatus.reviewing in announced
    assert SessionStatus.merging in announced


async def test_incidental_announcements_claim_no_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An announcement that does not mark a phase leaves the status alone.

    A warning about the sandbox, or a note about available secrets, says nothing
    about where the session has got to, so it must not move the client's status.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, events, _ = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    await session.run()

    for event in events:
        if isinstance(event, SystemEvent) and 'secrets are available' in event.content:
            assert event.status is None


async def test_unreachable_sandbox_is_reported_before_indexing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The banner does not wait for indexing to finish.

    Requirement 6.2: the reachability check runs when the connection is
    established, so the user learns straight away that nothing will be verified.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, events, services = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    async def unreachable() -> str | None:
        return 'Cannot connect to the Docker daemon'

    services.docker.check_reachable = unreachable
    services.indexing.files = ['a.py']

    await session.run()

    notices = [event for event in events if isinstance(event, SandboxUnavailableEvent)]
    assert len(notices) == 1
    assert 'Docker daemon' in notices[0].reason

    # It must arrive before any indexing progress.
    first_notice = next(
        index for index, event in enumerate(events) if isinstance(event, SandboxUnavailableEvent)
    )
    first_progress = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, SystemEvent) and event.key == INDEXING_PROGRESS_KEY
    )
    assert first_notice < first_progress


async def test_sandbox_failure_is_not_announced_in_the_feed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The warning is a banner, not a chat message.

    Requirement 6.1 and 6.7: the announcement that used to carry this was removed.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, events, services = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    async def unreachable() -> str | None:
        return 'no docker here'

    services.docker.check_reachable = unreachable

    await session.run()

    said = ' '.join(
        event.content
        for event in events
        if isinstance(event, (SystemEvent, ThoughtEvent))
    )
    assert 'sandbox is unavailable' not in said


async def test_the_sandbox_notice_is_sent_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An early failure and a later one describe the same predicament.

    Requirement 6.5: a second notification must not stack another banner.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, events, services = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    async def unreachable() -> str | None:
        return 'unreachable'

    async def failing_spawn(session_id: str) -> None:
        raise RuntimeError('and it will not start either')

    services.docker.check_reachable = unreachable
    services.docker.spawn_sandbox = failing_spawn

    await session.run()

    notices = [event for event in events if isinstance(event, SandboxUnavailableEvent)]
    assert len(notices) == 1


async def test_a_usable_sandbox_reports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A working sandbox produces no banner at all.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, events, _ = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    await session.run()

    assert not [event for event in events if isinstance(event, SandboxUnavailableEvent)]


async def test_a_stop_ends_the_session_as_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Stopping is recorded as cancelled, not failed.

    Requirement 8.6: history has to show the difference between work the user
    stopped and work that broke.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, events, _ = build_session([opening(), json.dumps({'action': 'stop'})])

    assert await session.run() is SessionState.cancelled

    said = ' '.join(event.content for event in events if isinstance(event, SystemEvent))
    assert 'Stopped at your request' in said


async def test_a_stop_kills_work_already_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The sandbox is torn down the moment a stop arrives.

    Requirement 8.4: a running command is killed outright rather than waited for.
    A command blocked inside the container runtime has no cancellation of its
    own, so removing the container is what ends it.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, _, services = build_session([opening(), json.dumps({'action': 'stop'})])

    stopped: list[bool] = []

    async def stop_sandbox() -> None:
        stopped.append(True)

    services.docker.stop_sandbox = stop_sandbox

    await session.run()

    assert stopped, 'the sandbox was never interrupted'


async def test_a_stop_leaves_the_branch_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Nothing already written is undone.

    Requirement 8.5: files, commits and the task branch all survive a stop.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, _, services = build_session([opening(), json.dumps({'action': 'stop'})])

    await session.run()

    assert not getattr(services.git, 'deleted_branches', []), 'the branch was removed'
    assert not getattr(services.git, 'reverted', []), 'work was reverted'


class StubInbox:
    """
    An inbox whose behaviour a test controls exactly.

    Lets each of the three places a stop is honoured be exercised on its own.
    The real inbox reads a socket, and with fast fakes the session reaches a
    waiting state so quickly that any one of the three would cover for the
    others, which tells us nothing about whether each is needed.
    """

    stop_requested: bool
    asked: int
    _commands: list[Any]

    def __init__(self, stop_requested: bool = False, commands: list[Any] | None = None) -> None:
        """
        Create a stub inbox.

        :param stop_requested: Whether a stop has already been received.
        :param commands: Commands to hand out, in order.
        """
        self.stop_requested = stop_requested
        self._commands = list(commands or [])
        self.asked = 0

    def start(self) -> None:
        """
        Begin reading, which a stub need not do.
        """

    async def next_command(self) -> Any:
        """
        Hand out the next scripted command.

        :returns: The next command.
        """
        self.asked += 1
        if self._commands:
            return self._commands.pop(0)
        return SessionCommand(action=CommandAction.stop)

    async def close(self) -> None:
        """
        Release the stub, which holds nothing.
        """


async def test_a_stop_while_waiting_unwinds_rather_than_rejecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A stop received while the session waits for an answer stops it.

    Isolates the check in `_receive_command`. Returned as an ordinary command a
    stop resolves to a rejection, which would send the agent round again instead
    of stopping it.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, _, _ = build_session([opening()])
    session._inbox = StubInbox(commands=[SessionCommand(action=CommandAction.stop)])  # type: ignore[assignment]

    with pytest.raises(StopRequested):
        await session._receive_command()


async def test_a_stop_between_states_is_noticed_without_waiting_for_a_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A stop during a state that neither streams nor waits still takes effect.

    Isolates the run loop's check. Indexing and merging ask the user nothing and
    stream nothing, so without this the stop would go unnoticed until the session
    happened to reach a state that does one or the other.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, _, _ = build_session([opening()])
    inbox: StubInbox = StubInbox(stop_requested=True)
    session._inbox = inbox  # type: ignore[assignment]

    assert await session.run() is SessionState.cancelled
    assert inbox.asked == 0, 'the stop should have been seen without awaiting a command'


async def test_cancellation_tells_the_client_the_agent_has_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The stop announcement states the phase, so the interface can leave it.

    Requirement 8.7. Without the status the client has no way to know the agent
    has stopped: the socket simply closes, and its last known phase is whatever
    it was working on, so the prompt box goes on offering to stop work that
    already has.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, events, _ = build_session([opening(), json.dumps({'action': 'stop'})])

    await session.run()

    cancelled = [
        event
        for event in events
        if isinstance(event, SystemEvent) and event.status is SessionStatus.cancelled
    ]
    assert cancelled, 'the client was never told the session had stopped'
    assert 'Stopped at your request' in cancelled[0].content


async def test_the_banner_and_the_completion_gate_agree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Sandbox availability is recorded once and read by both.

    Requirement 11.4. Held separately, the interface could show the banner while
    the gate went on demanding a verification that could never be run, and the
    agent would loop while the user was looking at a notice saying why it could
    not possibly succeed.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, events, services = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    async def unreachable() -> str | None:
        return 'no docker here'

    services.docker.check_reachable = unreachable
    context: SessionContext = session._session

    assert context.sandbox_available is True
    await session.run()

    assert context.sandbox_available is False
    assert [event for event in events if isinstance(event, SandboxUnavailableEvent)]


async def test_a_working_sandbox_leaves_the_gate_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Requirement 11.3: nothing is waived while the sandbox works.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, _, _ = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    await session.run()

    assert session._session.sandbox_available is True


async def test_the_chunk_size_reaches_the_indexer(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A size chosen in the interface has to arrive where chunks are built.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, _, services = build_session([
        json.dumps({'prompt': 'do it', 'chunk_lines': 80}),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    await session.run()

    assert services.indexing.chunk_lines == 80


async def test_an_absent_chunk_size_means_automatic(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A client that says nothing gets the behaviour that always works.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, _, services = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    await session.run()

    assert services.indexing.chunk_lines is None


@pytest.mark.parametrize(
    'sent,expected',
    [(None, None), (0, None), (-1, None), (True, None), ('80', None), (5, 10), (2000, 1000)],
    ids=['null', 'zero', 'negative', 'boolean', 'string', 'below minimum', 'above maximum'],
)
async def test_an_unusable_chunk_size_does_not_stop_the_session(
    monkeypatch: pytest.MonkeyPatch, sent: object, expected: int | None
) -> None:
    """
    The value arrives from a client, so it is bounded rather than trusted.

    A client sending nonsense gets automatic sizing, which always works, rather
    than a session that refuses to start.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, _, services = build_session([
        json.dumps({'prompt': 'do it', 'chunk_lines': sent}),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    await session.run()

    assert services.indexing.chunk_lines == expected


async def _answer_instead_of_executing(self: AgentSession) -> SessionState:
    """
    Stand in for the answering phase, which streams a real model run.

    :param self: The session being driven.
    :returns: The state the answering phase reaches.
    """
    self._session.accept_completion_without_changes('It retries three times, then gives up.')
    return await self._present_answer()


async def test_a_task_needing_no_changes_is_never_planned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Requirements 2.4 and 2.5, end to end.

    The session runs to a close on the client's opening message alone: no plan
    is offered, so no approval is asked for, and no merge decision is either.
    Any of those would have blocked waiting for a reply that never comes.
    """
    import api.v1.agent_session as module

    stub_agents(
        monkeypatch,
        [],
        [],
        classifications=[(False, 'The task asks how something works.')],
    )
    monkeypatch.setattr(
        module.AgentSession, '_handle_answering', _answer_instead_of_executing, raising=True
    )
    session, _, events, _ = build_session([opening('How does the retry policy work?')])

    assert await session.run() is SessionState.answered

    assert 'plan' not in types_of(events)
    done: list[DoneEvent] = [event for event in events if isinstance(event, DoneEvent)]
    assert len(done) == 1
    assert done[0].summary == 'It retries three times, then gives up.'
    # No branch means no merge is on offer, which is what ends the session here.
    assert done[0].branch_name == ''


async def test_a_task_needing_no_changes_is_recorded_as_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Requirement 2.8: not `completed`, which here promises a pending merge.
    """
    import api.v1.agent_session as module

    stub_agents(monkeypatch, [], [], classifications=[(False, 'It is a question.')])
    monkeypatch.setattr(
        module.AgentSession, '_handle_answering', _answer_instead_of_executing, raising=True
    )
    session, _, _, _ = build_session([opening('Explain the retry policy.')])
    recorded: list[TaskStatus] = []

    async def capture(status: TaskStatus, summary: str | None = None) -> None:
        recorded.append(status)

    monkeypatch.setattr(session, '_record_status', capture)

    await session.run()

    assert TaskStatus.answered in recorded
    assert TaskStatus.completed not in recorded
    assert TaskStatus.merged not in recorded


async def test_a_task_needing_changes_still_plans_and_merges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Requirement 2.3: the ordinary route is untouched by the new one.
    """
    stub_agents(
        monkeypatch,
        ['1. Do the thing'],
        [(True, 'Looks correct.')],
        classifications=[(True, 'The task asks for a timeout to be added.')],
    )
    session, _, events, _ = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    assert await session.run() is SessionState.merged
    assert 'plan' in types_of(events)


async def test_the_classifier_is_asked_about_the_task_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Requirement 2.2: it judges the request, and is shown nothing else.
    """
    seen = stub_agents(
        monkeypatch,
        ['1. Do the thing'],
        [(True, 'Fine.')],
        classifications=[(True, 'The task asks for a change.')],
    )
    session, _, _, _ = build_session([
        opening('Add a timeout to the HTTP client'),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    await session.run()

    assert len(seen['classifier']) == 1
    assert 'Add a timeout to the HTTP client' in seen['classifier'][0]


async def test_the_classifier_is_asked_before_any_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Requirement 2.1: the question is settled before a plan is produced.
    """
    order: list[str] = []
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Fine.')])
    session, _, _, _ = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    original_classify = session._handle_classifying
    original_plan = session._handle_planning

    async def note_classify() -> SessionState:
        order.append('classify')
        return await original_classify()

    async def note_plan() -> SessionState:
        order.append('plan')
        return await original_plan()

    monkeypatch.setattr(session, '_handle_classifying', note_classify)
    monkeypatch.setattr(session, '_handle_planning', note_plan)

    await session.run()

    assert order[:2] == ['classify', 'plan']


async def test_an_unreachable_classifier_fails_the_session_with_the_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A classifier that cannot be reached must not be guessed around.

    Guessing "no changes" would end real work with an answer; guessing
    "changes" would put a question through planning and a merge decision.
    """
    import agents.classifier as classifier_module

    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Fine.')])

    async def unreachable(prompt: str, **kwargs: Any) -> Any:
        raise ConnectionError('the model host refused the connection')

    monkeypatch.setattr(classifier_module.classifier_agent, 'run', unreachable)
    session, _, events, _ = build_session([opening()])

    assert await session.run() is SessionState.failed

    errors: list[ErrorEvent] = [event for event in events if isinstance(event, ErrorEvent)]
    assert any('could not be classified' in event.content for event in errors)
    assert any('refused the connection' in event.content for event in errors)
    assert 'plan' not in types_of(events)


async def test_a_task_needing_no_changes_creates_no_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A task that changes nothing must leave nothing behind.

    Creating the branch while initialising left an empty one in the repository
    for every question anyone asked, and a branch name on a history record for
    a task that never touched the tree.
    """
    import api.v1.agent_session as module

    stub_agents(monkeypatch, [], [], classifications=[(False, 'It is a question.')])
    monkeypatch.setattr(
        module.AgentSession, '_handle_answering', _answer_instead_of_executing, raising=True
    )
    session, _, _, services = build_session([opening('Explain the retry policy.')])

    assert await session.run() is SessionState.answered

    assert services.git.created_branches == []


async def test_a_task_needing_changes_creates_its_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Deferring branch creation must not lose it for work that does change files.
    """
    stub_agents(
        monkeypatch,
        ['1. Do the thing'],
        [(True, 'Fine.')],
        classifications=[(True, 'The task asks for a change.')],
    )
    session, _, _, services = build_session([
        opening(),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    assert await session.run() is SessionState.merged

    assert services.git.created_branches == ['ai-task-testsess']


async def test_an_escalated_task_gets_a_branch_before_it_plans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Requirement 2.7: a task that escalates joins the ordinary route in full,
    branch included, or the work it goes on to do has nowhere to land.
    """
    import api.v1.agent_session as module

    stub_agents(
        monkeypatch,
        ['1. Do the thing'],
        [(True, 'Fine.')],
        classifications=[(False, 'It looks like a question.')],
    )

    async def answer_that_escalates(self: AgentSession) -> SessionState:
        return await self._escalate_to_planning()

    monkeypatch.setattr(
        module.AgentSession, '_handle_answering', answer_that_escalates, raising=True
    )
    session, _, _, services = build_session([
        opening('Is the retry policy right, and fix it if not'),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    assert await session.run() is SessionState.merged

    assert services.git.created_branches == ['ai-task-testsess']


async def test_the_branch_is_created_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Creating the same branch twice fails, so the second attempt is a no-op.

    Exercised directly rather than through a session run: today exactly one of
    the two routes into planning creates it, so a run cannot reach the guard.
    It is kept because `create_branch` is unforgiving of a repeat, and a third
    caller added later would find out the hard way.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Fine.')])
    session, _, _, services = build_session([opening()])

    await session._ensure_branch()
    await session._ensure_branch()

    assert services.git.created_branches == ['ai-task-testsess']


@pytest.mark.usefixtures('db')
async def test_an_empty_repository_is_judged_per_project(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Another project being indexed says nothing about this one.

    Before the index was scoped, one file anywhere made every repository look
    non-empty. The session would then plan against nothing and give no
    indication why, which is the situation this check exists to prevent.
    """
    from database.models.file import File

    monkeypatch.delenv('REPO_PATH', raising=False)

    session, _, _, services = build_session([json.dumps({'prompt': 'do something'})])
    services.indexing.repo_path = '/repos/mine'

    # Somebody else's project is fully indexed.
    await File.create(
        project=normalise_project_path('/repos/theirs'),
        file_path='src/main.py',
        raw_content='theirs\n',
        hash='h1',
    )

    assert await session._repository_is_empty() is True

    await File.create(
        project=normalise_project_path('/repos/mine'),
        file_path='src/main.py',
        raw_content='mine\n',
        hash='h2',
    )

    assert await session._repository_is_empty() is False


async def test_planning_falls_back_to_text_when_structure_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A planner that cannot produce structure must not cost us the plan.

    This is the requirement most easily lost, because everything looks correct
    for as long as the model cooperates — and the small models this runs against
    frequently will not. Asserted by making the structured attempt fail outright
    and requiring a usable plan to arrive anyway.
    """
    import agents.planner as planner_module
    from services.plan_parser import PlanFormat

    async def refuse(*_: object, **__: object) -> object:
        raise RuntimeError('this model does not do structured output')

    monkeypatch.setattr(planner_module.planner_agent, 'run', refuse)

    session, _, _, _ = build_session([json.dumps({'prompt': 'do something'})])

    async def planner_text(_self: object, _prompt: str) -> str:
        return '1. Read the file.\n2. Edit it.\nPLAN_COMPLETE'

    monkeypatch.setattr(type(session), '_plan_as_text', planner_text)

    plan = await session._plan_structured('irrelevant')

    assert plan.titles == ['Read the file.', 'Edit it.']
    # Marked as text, so the reader is shown text rather than empty accordions.
    assert plan.format is PlanFormat.text


async def test_a_structured_planning_overflow_retries_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A recoverable overflow must retry structured output instead of falling back to text.
    """
    from agents.planner import PlannedStep, StructuredPlan
    from pydantic_ai.exceptions import UnexpectedModelBehavior
    import agents.planner as planner_module
    from services.plan_parser import PlanFormat

    session, _, _, _ = build_session([json.dumps({'prompt': 'do something'})])
    session._planner_context_window = 128_048
    attempts: list[int] = []

    class Result:
        """Provide the successful structured planner output."""

        output: StructuredPlan = StructuredPlan(
            steps=[PlannedStep(title='Read the file.', description='Inspect it.')]
        )

    async def overflow_once(*_: object, **__: object) -> Result:
        attempts.append(session._planner_context_window)
        if len(attempts) == 1:
            raise UnexpectedModelBehavior(
                'Model token limit (8192) exceeded before any response was generated.'
            )
        return Result()

    monkeypatch.setattr(planner_module.planner_agent, 'run', overflow_once)

    plan = await session._plan_structured('irrelevant')

    assert attempts == [128_048, 64_024]
    assert plan.titles == ['Read the file.']
    assert plan.format is PlanFormat.structured


async def test_a_text_planning_overflow_is_narrowed_and_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Text planning uses the same recoverable-overflow loop as the other long runs.
    """
    from pydantic_ai.exceptions import UnexpectedModelBehavior
    import agents.planner as planner_module

    session, _, _, _ = build_session([json.dumps({'prompt': 'do something'})])
    session._planner_context_window = 128_048
    attempts: list[int] = []

    class Result:
        """Provide the successful text planner output."""

        output: str = '1. Read the file.'

    async def overflow_once(*_: object, **__: object) -> Result:
        attempts.append(session._planner_context_window)
        if len(attempts) == 1:
            raise UnexpectedModelBehavior(
                'Model token limit (8192) exceeded before any response was generated.'
            )
        return Result()

    monkeypatch.setattr(planner_module.planner_agent, 'run', overflow_once)

    assert await session._plan_as_text('irrelevant') == '1. Read the file.'
    assert attempts == [128_048, 64_024]


async def test_stop_requested_during_text_planning_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A stop request must propagate rather than becoming another model attempt.
    """
    import agents.planner as planner_module

    session, _, _, _ = build_session([json.dumps({'prompt': 'do something'})])
    attempts: list[int] = []

    async def stop(*_: object, **__: object) -> object:
        attempts.append(1)
        raise StopRequested()

    monkeypatch.setattr(planner_module.planner_agent, 'run', stop)

    with pytest.raises(StopRequested):
        await session._plan_as_text('irrelevant')

    assert attempts == [1]


async def test_structure_that_says_nothing_falls_back_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Structure that arrives empty is no better than structure that fails.

    A planner returning zero steps would otherwise produce an empty plan and end
    the round, when the text path might still recover something.
    """
    import agents.planner as planner_module
    from agents.planner import StructuredPlan
    from services.plan_parser import PlanFormat

    class EmptyResult:
        output: StructuredPlan = StructuredPlan(steps=[])

    async def empty(*_: object, **__: object) -> EmptyResult:
        return EmptyResult()

    monkeypatch.setattr(planner_module.planner_agent, 'run', empty)

    session, _, _, _ = build_session([json.dumps({'prompt': 'do something'})])

    async def planner_text(_self: object, _prompt: str) -> str:
        return '1. Recovered step.'

    monkeypatch.setattr(type(session), '_plan_as_text', planner_text)

    plan = await session._plan_structured('irrelevant')

    assert plan.titles == ['Recovered step.']
    assert plan.format is PlanFormat.text


async def test_the_context_windows_reach_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Windows chosen in the interface have to arrive where budgets are built.

    The two are read independently, so a client setting only one does not drag
    the other along with it.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, _, _ = build_session([
        json.dumps({
            'prompt': 'do it',
            'coder_context_window': 40_000,
            'planner_context_window': 8_192,
        }),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    await session.run()

    assert session._coder_context_window == 40_000
    assert session._planner_context_window == 8_192


async def test_a_measured_window_budgets_every_long_run_path() -> None:
    """
    The serving model's limit reaches compaction, monitoring, and `num_ctx`.
    """
    session, _, _, _ = build_session([json.dumps({'prompt': 'do it'})])
    session._coder_context_window = 128_048
    session._planner_context_window = 128_048
    calls: list[str] = []

    async def measured(model_name: str) -> int | None:
        """
        Report the model limit the probe discovered.

        :param model_name: The model that would have been probed.
        :returns: The serving context window.
        """
        calls.append(model_name)
        return 8_192

    session._context_window_probe = measured
    assert await session._measure_context_window() == 8_192
    assert await session._measure_context_window() == 8_192
    assert calls == [session._coder_model]

    from services.event_bus import CompactionRunPath

    capability: object = session._compaction_for(
        session._coder_context_window, CompactionRunPath.coder_streaming
    )
    assert getattr(capability, '_inner').context_window == 8_192
    assert session._usage_monitor_for(session._planner_context_window).context_window == 8_192
    assert session._model_settings_for(session._coder_context_window).get('extra_body') == {
        'options': {'num_ctx': 8_192}
    }


async def test_auto_context_windows_use_a_larger_measurement() -> None:
    """
    An unset setting does not cap a measured model at Loom's generic default.
    """
    session, _, _, _ = build_session([json.dumps({'prompt': 'do it'})])

    async def measured(model_name: str) -> int | None:
        """
        Report a serving window larger than the generic fallback.

        :param model_name: The model that would have been probed.
        :returns: The model's measured context window.
        """
        _ = model_name
        return 131_072

    session._context_window_probe = measured

    assert await session._measure_context_window() == 131_072
    assert session._coder_context_window == 131_072
    assert session._planner_context_window == 131_072


async def test_compaction_capabilities_put_proactive_pruning_before_the_ladder() -> None:
    """
    Every long-running path evaluates the eager cap before reactive compaction.
    """
    from pydantic_ai.capabilities import AbstractCapability
    from services.compaction import ObservedCompaction
    from services.event_bus import CompactionRunPath

    session, _, _, _ = build_session([json.dumps({'prompt': 'do it'})])
    capabilities: list[AbstractCapability[Any]] = session._compaction_capabilities_for(
        32_768, CompactionRunPath.coder_streaming
    )

    assert len(capabilities) == 2
    assert all(isinstance(capability, ObservedCompaction) for capability in capabilities)
    assert type(getattr(capabilities[0], '_inner')).__name__ == 'ClearToolResults'
    assert type(getattr(capabilities[1], '_inner')).__name__ == 'TieredCompaction'


async def test_all_agents_keep_five_tool_retries() -> None:
    """
    Graceful exhaustion recovery supplements, rather than reduces, agent retries.
    """
    from agents.adjudicator import adjudicator_agent
    from agents.classifier import classifier_agent
    from agents.coder import answering_agent, coder_agent
    from agents.planner import planner_agent
    from agents.reviewer import reviewer_agent

    agents: tuple[object, ...] = (
        coder_agent,
        answering_agent,
        classifier_agent,
        reviewer_agent,
        planner_agent,
        adjudicator_agent,
    )

    assert all(getattr(agent, '_max_tool_retries') == 5 for agent in agents)


async def test_a_too_small_measured_window_warns_once_before_any_agent_run() -> None:
    """
    A known-unworkable serving limit receives the existing human-facing warning.
    """
    session, _, events, _ = build_session([json.dumps({'prompt': 'do it'})])

    async def measured(model_name: str) -> int | None:
        """
        Report a context window too small for the coder's fixed overhead.

        :param model_name: The model that would have been probed.
        :returns: The serving context window.
        """
        return 2_048

    session._context_window_probe = measured
    await session._receive_task()
    _ = await session._measure_context_window()
    await session._warn_if_context_is_too_small()
    await session._warn_if_context_is_too_small()

    warnings: list[WarningEvent] = [event for event in events if isinstance(event, WarningEvent)]
    assert len(warnings) == 1
    assert 'about 2,048 tokens' in warnings[0].content


@pytest.mark.parametrize(
    'sent',
    [None, 0, -1, 'lots', True, 1.5],
)
async def test_an_unusable_context_window_means_the_default(
    monkeypatch: pytest.MonkeyPatch, sent: object
) -> None:
    """
    A client sending nonsense gets the default rather than a refused session.

    Zero is the stored form of "unset", which `build_compaction` reads as the
    default, so nothing is left unbounded by a bad value.
    """
    stub_agents(monkeypatch, ['1. Do the thing'], [(True, 'Looks correct.')])
    session, _, _, _ = build_session([
        json.dumps({'prompt': 'do it', 'coder_context_window': sent}),
        json.dumps({'approved': True}),
        json.dumps({'approved': True}),
    ])

    await session.run()

    assert session._coder_context_window == 0


async def test_the_text_fallback_cannot_write_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The fallback runs the planner, which carries no writing tools.

    It used to run the coder, whose tools include `edit_file`, `create_file`
    and `commit_changes`. That let a plan-shaped request start writing and
    committing before the user had approved anything. The planner does get
    deps, because its read-only tools need something to act through — being
    unable to write comes from those tools never having been registered.
    """
    from agents.planner import planner_agent
    import agents.planner as planner_module

    async def refuse(*_: object, **__: object) -> object:
        raise RuntimeError('this model does not do structured output')

    seen: list[object] = []

    async def record(*args: object, **kwargs: object) -> object:
        if kwargs.get('output_type') is not str:
            raise RuntimeError('structured output requested')
        seen.append(kwargs.get('deps'))

        class Result:
            output: str = '1. Recovered step.'

        return Result()

    calls: list[str] = []

    async def dispatch(*args: object, **kwargs: object) -> object:
        calls.append('call')
        if len(calls) == 1:
            return await refuse(*args, **kwargs)
        return await record(*args, **kwargs)

    monkeypatch.setattr(planner_module.planner_agent, 'run', dispatch)

    session, _, _, _ = build_session([json.dumps({'prompt': 'do something'})])
    plan = await session._plan_structured('irrelevant')

    assert plan.titles == ['Recovered step.']
    # Deps are passed, so the read-only tools can reach the repository.
    assert seen and seen[0] is not None
    registered: set[str] = {tool.name for tool in planner_agent._function_toolset.tools.values()}
    assert registered.isdisjoint({'create_file', 'edit_file', 'run_test_command'})


async def test_the_text_fallback_uses_the_planner_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The budget follows the agent whose history is compacted, which is now the planner.
    """
    import agents.planner as planner_module
    import api.v1.agent_session as session_module
    from services.compaction import build_compaction

    windows: list[int | None] = []
    # Captured before patching: reaching for it through the module afterwards
    # would find this very function and recurse.
    original = build_compaction

    def record(context_window: int | None, **kwargs: object) -> object:
        windows.append(context_window)
        return original(context_window)

    monkeypatch.setattr(session_module, 'build_compaction', record)

    session, _, _, _ = build_session([json.dumps({'prompt': 'do something'})])
    session._planner_context_window = 8_192

    async def answer(*_: object, **__: object) -> object:
        class Result:
            output: str = '1. A step.'

        return Result()

    monkeypatch.setattr(planner_module.planner_agent, 'run', answer)

    await session._plan_as_text('irrelevant')

    assert windows == [8_192]
    assert original(8_192).target_tokens == build_compaction(8_192).target_tokens
