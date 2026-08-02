from time import monotonic
from asyncio import Event
from collections.abc import AsyncIterable
from typing import Any, Awaitable, Callable, Protocol

from agents.coder import AgentDeps, coder_agent
from agents.reviewer import ReviewResult, reviewer_agent
from fastapi import WebSocket
from pydantic_ai import RunContext
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import (
    AgentStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from schemas.agent_commands import CommandAction, SessionCommand, parse_command
from services.conversation_history import ConversationHistory, TurnRole
from services.embedding import DEFAULT_EMBEDDING_MODEL, EmbeddingService
from services.event_bus import (
    INDEXING_PROGRESS_KEY,
    DoneEvent,
    ErrorEvent,
    EventBus,
    PlanEvent,
    ReviewEvent,
    SandboxStage,
    SandboxUnavailableEvent,
    SessionStatus,
    SystemEvent,
    ThoughtEvent,
)
from services.docker_sandbox import DockerSandboxManager, SandboxError
from services.file_editor import FileEditorService
from services.git_manager import FileChange, GitManagerService
from services.indexing import IndexingService
from services.plan_parser import NO_STEPS_CORRECTION, ParsedPlan, parse_plan, render_plan
from services.review_context import build_review_prompt, collect_review_diffs
from services.model_resolution import resolve_model
from services.session_context import SessionContext
from services.session_inbox import SessionInbox
from services.session_fsm import SessionState, SessionStateMachine
from services.session_limits import IterationLimits
from services.task_recorder import TaskRecorder
from database.enums import TaskStatus
from database.models.file import File
from database.models.task import Task
from uuid import UUID, uuid4
from services.thought_buffer import ThoughtBuffer
import json
import logging
import os

logger = logging.getLogger(__name__)


class ClientGone(Exception):
    """
    Raised to unwind out of a long running state once the client has left.
    """


class StopRequested(Exception):
    """
    Raised to unwind out of a long running state once the user has pressed stop.
    """


def _read_secrets(raw: Any) -> dict[str, str]:
    """
    Interpret the secrets supplied in the opening message.

    The client sends a list of key and value pairs. Anything unusable is dropped
    rather than raising, since a malformed entry should not stop the session.

    :param raw: The value of the message's `secrets` field.
    :returns: The secrets, keyed by name.
    """
    secrets: dict[str, str] = {}
    if not isinstance(raw, list):
        return secrets

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        key: str = str(entry.get('key', '')).strip()
        if key:
            secrets[key] = str(entry.get('value', ''))
    return secrets


class SessionServices(Protocol):
    """
    The services a session needs from its controller.

    Declared structurally so the session can be typed against the controller
    without importing it, which would be circular. Typing this as `Any` would
    silence the checkers on every service access.
    """

    docker: DockerSandboxManager
    editor: FileEditorService
    git: GitManagerService
    indexing: IndexingService
    embedding: EmbeddingService

CONTEXT_TOKEN_BUDGET: int = 4096
DEFAULT_MODEL: str = 'ollama:llama3'


class AgentSession:
    """
    Drives one agent session through the finite state machine.

    Each state has a handler that does its work and returns the state to move to
    next, so the loop itself stays a plain dispatch and every transition is
    validated in one place.
    """

    _websocket: WebSocket
    _services: SessionServices
    _session_id: str
    _events: EventBus
    _deps: AgentDeps
    _session: SessionContext
    _history: ConversationHistory
    _machine: SessionStateMachine
    _limits: IterationLimits

    _task: str
    _branch_name: str
    _coder_model: str
    _reviewer_model: str
    _plan: ParsedPlan
    _plan_rejections: int
    _review_rejections: int
    _total_iterations: int
    _auto_approve: bool
    _last_review: ReviewResult | None
    _dockerfile: str
    _inbox: SessionInbox
    _secrets: dict[str, str]
    _task_id: str
    _recorder: TaskRecorder | None
    _connection_lost: Event

    def __init__(
        self,
        websocket: WebSocket,
        services: SessionServices,
        session_id: str,
        events: EventBus,
        deps: AgentDeps,
        session: SessionContext,
        history: ConversationHistory,
        machine: SessionStateMachine,
        limits: IterationLimits,
        connection_lost: Event | None = None,
    ) -> None:
        """
        Create a session driver.

        :param websocket: The client connection.
        :param services: The controller holding the shared services.
        :param session_id: Identifier for this session.
        :param events: The session's event bus.
        :param deps: Dependencies handed to the agent's tools.
        :param session: Mutable state shared with the tools.
        :param history: The accumulated conversation.
        :param machine: The state machine to drive.
        :param limits: The iteration ceilings to enforce.
        :param connection_lost: Set when the client can no longer be reached.
        """
        self._websocket = websocket
        self._inbox = SessionInbox(websocket.receive_text, on_stop=self._interrupt_work)
        self._services = services
        self._session_id = session_id
        self._events = events
        self._deps = deps
        self._session = session
        self._history = history
        self._machine = machine
        self._limits = limits
        self._connection_lost = connection_lost if connection_lost is not None else Event()

        self._task = ''
        self._branch_name = f'ai-task-{session_id}'
        self._coder_model = DEFAULT_MODEL
        self._reviewer_model = DEFAULT_MODEL
        self._plan = ParsedPlan(steps=[])
        self._plan_rejections = 0
        self._review_rejections = 0
        self._total_iterations = 0
        self._auto_approve = False
        self._last_review = None
        self._dockerfile = ''
        self._secrets = {}
        self._task_id = ''
        self._recorder = None

    async def _interrupt_work(self) -> None:
        """
        Cut short anything already running, so stopping does not mean waiting.

        A command running in the sandbox has no cancellation of its own: the call
        is blocked inside the container runtime. Removing the container ends it
        outright, which is what requirement 8.4 asks for. The repository is a
        bind mount, so nothing the agent has written is lost with it.
        """
        await self._services.docker.stop_sandbox()

    async def _cancel(self) -> SessionState:
        """
        Stop because the user asked to.

        Reached only between states or from the execution stream, so no tool call
        is part way through and no commit is interrupted. Everything already
        written stays: files on disk, commits, and the task branch.

        :returns: The cancelled state.
        """
        logger.info('Cancelling session %s at the user request.', self._session_id)
        if not self._machine.is_terminal:
            self._machine.cancel()
        await self._record_status(TaskStatus.cancelled)
        await self._announce(
            'Stopped at your request. Everything done so far has been kept.',
            SessionStatus.cancelled,
        )
        return SessionState.cancelled

    async def _abandon(self) -> SessionState:
        """
        Stop because the client is gone.

        Called only between states, or from the execution stream, so no tool call
        is part way through and no commit is interrupted. Work already committed
        stays on the task branch.

        :returns: The failed state.
        """
        logger.info('Abandoning session %s: the client is no longer reachable.', self._session_id)
        if not self._machine.is_terminal:
            self._machine.fail()
        await self._record_status(TaskStatus.failed)
        return self._machine.current

    async def mark_failed(self) -> None:
        """
        Record that the session ended in failure.

        Called when an exception escapes the state machine. Without it a task
        whose session crashed would stay at `initialized` forever, so the history
        would show it as never having started.
        """
        await self._record_status(TaskStatus.failed)

    @property
    def branch_name(self) -> str:
        """
        The branch this session works on.

        :returns: The task branch name.
        """
        return self._branch_name

    async def run(self) -> SessionState:
        """
        Drive the session until it reaches a terminal state.

        :returns: The terminal state the session ended in.
        """
        await self._receive_task()
        self._inbox.start()

        while not self._machine.is_terminal:
            if self._connection_lost.is_set():
                return await self._abandon()
            if self._inbox.stop_requested:
                return await self._cancel()

            try:
                next_state: SessionState = await self._advance(self._machine.current)
            except ClientGone:
                return await self._abandon()
            except StopRequested:
                return await self._cancel()
            except Exception:
                # The machine has to record the failure itself. Letting the
                # exception escape while the state still reads `executing` leaves
                # a lie behind, which matters as soon as anything resumes from it.
                if not self._machine.is_terminal:
                    self._machine.fail()
                await self._record_status(TaskStatus.failed)
                raise
            self._machine.transition_to(next_state)

        if self._machine.current is SessionState.failed:
            await self._record_status(TaskStatus.failed)

        return self._machine.current

    async def _advance(self, state: SessionState) -> SessionState:
        """
        Run the handler for the current state.

        :param state: The state to handle.
        :returns: The state to move to next.
        :raises RuntimeError: If the state has no handler.
        """
        handlers: dict[SessionState, Callable[[], Awaitable[SessionState]]] = {
            SessionState.initializing: self._handle_initializing,
            SessionState.planning: self._handle_planning,
            SessionState.waiting_plan_approval: self._handle_waiting_plan_approval,
            SessionState.executing: self._handle_executing,
            SessionState.reviewing: self._handle_reviewing,
            SessionState.waiting_review_approval: self._handle_waiting_review_approval,
            SessionState.re_working: self._handle_re_working,
            SessionState.completed: self._handle_completed,
            SessionState.merging: self._handle_merging,
        }

        handler: Callable[[], Awaitable[SessionState]] | None = handlers.get(state)
        if handler is None:
            raise RuntimeError(f'No handler is defined for state {state.value}.')
        return await handler()

    async def _receive_task(self) -> None:
        """
        Read the opening message and configure the session from it.

        :raises ValueError: If the client sent no prompt.
        """
        message: dict[str, Any] = json.loads(await self._websocket.receive_text())

        self._task = str(message.get('prompt', '')).strip()
        if not self._task:
            await self._events.emit(ErrorEvent(content='No prompt provided.'))
            raise ValueError('The client sent no prompt.')

        self._coder_model = str(message.get('coder_model', DEFAULT_MODEL))
        self._reviewer_model = str(message.get('reviewer_model', DEFAULT_MODEL))
        # Carried on the context because the adjudicating subagent runs inside a
        # tool call, where the session is out of reach.
        self._session.configure(self._task, self._reviewer_model)
        self._services.embedding.embedding_model = str(
            message.get('embedding_model', DEFAULT_EMBEDDING_MODEL)
        )

        self._task_id = str(message.get('task_id', '') or '')
        self._dockerfile = str(message.get('dockerfile', '') or '')
        self._secrets = _read_secrets(message.get('secrets'))

        repo_path: str | None = message.get('repo_path')
        if repo_path:
            # Deliberately not exported to the environment. Services read
            # REPO_PATH only when they are constructed, which happens as the
            # socket connects, so setting it here cannot affect this session but
            # would silently repoint the next session that omits a repo_path.
            self._services.docker.repo_path = repo_path
            self._services.editor.root_path = repo_path
            self._services.git.repo_path = repo_path
            self._services.indexing.repo_path = repo_path

        self._history.add(TurnRole.user, self._task)

    async def _say(self, content: str) -> None:
        """
        Relay a unit of the model's reasoning to the client.

        Reserved for the model's own words. Anything Loom says about the session
        goes through `_announce` instead, so the two remain distinguishable in
        the feed.

        :param content: The reasoning to send.
        """
        await self._events.emit(ThoughtEvent(content=content))

    async def _announce(self, content: str, status: SessionStatus | None = None) -> None:
        """
        Tell the client something about the session itself.

        This is Loom talking, not the model thinking, and it is labelled
        accordingly. Where the announcement marks entry into a phase of work, the
        phase is stated outright rather than left to be inferred from the wording.

        :param content: The text to send.
        :param status: The phase being entered, if this announcement marks one.
        """
        await self._events.emit(SystemEvent(content=content, status=status))

    async def _report_sandbox_unusable(self, reason: str | None, stage: SandboxStage) -> None:
        """
        Tell the client the sandbox cannot be used, once per session.

        Reported at most once: the early reachability check and a later failure
        to start describe the same predicament, and the client should not have to
        work out that two notices mean one problem. Availability is recorded on
        the session context, which is also what the completion gate reads, so the
        banner and the gate cannot disagree.

        :param reason: Why the sandbox is unusable, or None if it is fine.
        :param stage: Which step of preparing the sandbox failed.
        """
        if reason is None or not self._session.sandbox_available:
            return
        self._session.mark_sandbox_unavailable()
        await self._events.emit(SandboxUnavailableEvent(stage=stage, reason=reason))

    async def _handle_initializing(self) -> SessionState:
        """
        Register the task, index the repository, start the sandbox and branch.

        :returns: The next state.
        """
        await self._start_recording()
        await self._report_sandbox_unusable(
            await self._services.docker.check_reachable(), SandboxStage.runtime_connectivity
        )
        await self._announce('Indexing repository...', SessionStatus.indexing)

        async def progress(current: int, total: int, message: str) -> None:
            """
            Report indexing progress to the client.

            Every update carries the same slot key, so the client replaces one
            message rather than accumulating one per file. The counts travel as
            numbers so the client need not parse them back out of the text.

            :param current: Files processed so far.
            :param total: Total files to process.
            :param message: The file just processed.
            """
            await self._events.emit(
                SystemEvent(
                    content=message,
                    key=INDEXING_PROGRESS_KEY,
                    processed=current,
                    total=total,
                )
            )

        await self._services.indexing.index_repository(progress_callback=progress)

        await self._announce(
            'Building sandbox image...' if self._dockerfile.strip() else 'Starting sandbox...',
            SessionStatus.indexing,
        )
        try:
            await self._services.docker.spawn_sandbox(
                self._session_id, dockerfile=self._dockerfile, environment=self._secrets
            )
        except Exception as error:
            # Editing still works without a sandbox, but nothing can be verified,
            # and the completion gate requires a passing verification. Reported as
            # a banner rather than a chat line, which would scroll away from a
            # constraint that lasts the whole session.
            stage: SandboxStage = (
                error.stage
                if isinstance(error, SandboxError)
                else SandboxStage.container_start
            )
            await self._report_sandbox_unusable(str(error), stage)

        if self._secrets:
            await self._announce(
                'The following secrets are available inside the sandbox as environment '
                f'variables: {", ".join(sorted(self._secrets))}. Their values are hidden.'
            )

        if await self._repository_is_empty():
            return await self._confirm_empty_repository()

        await self._services.git.create_branch(self._branch_name)
        if self._recorder is not None:
            self._recorder.task.branch_name = self._branch_name
            await self._recorder.record_status(TaskStatus.running)
        return SessionState.planning

    async def _repository_is_empty(self) -> bool:
        """
        Check whether indexing found anything to work with.

        :returns: True if no file was indexed.
        """
        try:
            return await File.all().count() == 0
        except Exception as error:
            logger.warning('Could not check whether the repository is empty: %s', error)
            return False

    async def _confirm_empty_repository(self) -> SessionState:
        """
        Tell the user the repository is empty and ask how to proceed.

        Planning against nothing produces a plan the agent cannot carry out, and
        no indication of why, so the session stops here for an answer.

        :returns: The next state.
        """
        await self._announce(
            f'No files were indexed in {self._services.git.repo_path}. The repository looks '
            'empty, or everything in it was excluded. Approve to continue anyway, or reject '
            'to stop and point Loom at a different repository.'
        )

        command: SessionCommand = await self._receive_command()
        if not command.is_approval:
            await self._events.emit(
                ErrorEvent(content='Stopped: there is nothing in the repository to work on.')
            )
            return SessionState.failed

        await self._services.git.create_branch(self._branch_name)
        if self._recorder is not None:
            self._recorder.task.branch_name = self._branch_name
            await self._recorder.record_status(TaskStatus.running)
        return SessionState.planning

    async def _start_recording(self) -> None:
        """
        Attach this session to a task record and begin logging.

        A task supplied in the opening message is adopted; otherwise one is
        created, so a client that connects directly still produces history.
        Failing to record must not stop the work, so problems are logged.
        """
        try:
            task: Task | None = None
            if self._task_id:
                try:
                    task = await Task.get_or_none(id=UUID(self._task_id))
                except ValueError:
                    task = None

            if task is None:
                task = await Task.create(
                    id=uuid4(),
                    prompt=self._task,
                    repo_path=self._services.git.repo_path,
                    coder_model=self._coder_model,
                    reviewer_model=self._reviewer_model,
                    branch_name=self._branch_name,
                    status=TaskStatus.initialized,
                )

            self._task_id = str(task.id)
            recorder: TaskRecorder = TaskRecorder(task)
            self._recorder = recorder
            self._events.subscribe(recorder.handle_event)
        except Exception as error:
            logger.warning('Task history is unavailable for this session: %s', error)
            self._recorder = None

    async def _handle_planning(self) -> SessionState:
        """
        Ask the coder agent for a plan and parse it.

        :returns: The next state.
        """
        self._total_iterations += 1
        if self._total_iterations > self._limits.max_total_iterations:
            await self._events.emit(
                ErrorEvent(
                    content=(
                        f'Reached the ceiling of {self._limits.max_total_iterations} '
                        'iterations without completing the task. Stopping.'
                    )
                )
            )
            return SessionState.failed

        await self._announce('Generating plan...', SessionStatus.planning)
        self._history.advance_iteration()

        prompt: str = self._build_planning_prompt()
        plan: ParsedPlan = parse_plan(await self._run_coder(prompt))

        if plan.is_empty:
            # One corrective attempt before giving up on this round.
            plan = parse_plan(await self._run_coder(f'{prompt}\n\n{NO_STEPS_CORRECTION}'))

        if plan.is_empty:
            await self._events.emit(
                ErrorEvent(content='The agent produced no actionable plan steps.')
            )
            return SessionState.failed

        self._plan = plan
        self._history.add(TurnRole.agent, render_plan(plan))
        return SessionState.waiting_plan_approval

    async def _run_coder(self, prompt: str) -> str:
        """
        Run the coder agent to completion and return its text output.

        :param prompt: The prompt to send.
        :returns: The agent's output.
        """
        result: AgentRunResult[str] = await coder_agent.run(
            prompt, deps=self._deps, model=resolve_model(self._coder_model)
        )
        return result.output

    def _build_planning_prompt(self) -> str:
        """
        Build the planning prompt, including everything learned so far.

        Feeding the accumulated history back in is what lets a rejected plan or a
        reviewer's criticism actually change the next attempt.

        :returns: The prompt to send to the coder agent.
        """
        context: str = self._history.build_context(CONTEXT_TOKEN_BUDGET)
        sections: list[str] = [f'Task: {self._task}']
        if context:
            sections.append(f'What has happened so far:\n{context}')
        sections.append('Produce a step-by-step plan for this task. Do not call any tools yet.')
        return '\n\n'.join(sections)

    async def _handle_waiting_plan_approval(self) -> SessionState:
        """
        Show the plan to the user and act on their answer.

        :returns: The next state.
        """
        await self._events.emit(
            PlanEvent(steps=list(self._plan.steps), requires_approval=not self._auto_approve)
        )

        if self._auto_approve:
            return SessionState.executing

        command: SessionCommand = await self._receive_command()
        if command.is_approval:
            return SessionState.executing

        self._plan_rejections += 1
        self._history.add(
            TurnRole.user,
            command.reason or 'The user rejected this plan without giving a reason.',
            metadata={'rejection': True}
        )
        await self._record_rejection('plan', command.reason)

        if self._plan_rejections >= self._limits.max_plan_retries:
            return await self._ask_how_to_proceed(
                f'The plan has been rejected {self._plan_rejections} times.',
                SessionState.planning
            )

        return SessionState.planning

    async def _handle_executing(self) -> SessionState:
        """
        Run the coder agent against the approved plan.

        The graph is run with `run()` and an event stream handler rather than
        `run_stream()`. `run_stream()` stops the graph once it has an output
        matching the output type, which for a text-output agent means it
        executes only the first round of tool calls and abandons the rest. A
        ReAct sequence such as read, edit, test, complete would be cut off after
        its first step, so the agent could never finish the work.

        :returns: The next state.
        """
        await self._announce('Executing plan...', SessionStatus.executing)

        prompt: str = (
            f'Task: {self._task}\n\n'
            f'Approved plan:\n{render_plan(self._plan)}\n\n'
            'Execute this plan now. Verify every change with run_test_command, and call '
            'complete_task once the work is finished.'
        )

        if self._secrets:
            prompt += (
                '\n\nThese environment variables are set inside the sandbox: '
                f'{", ".join(sorted(self._secrets))}. Reference them by name; their '
                'values are not available to you and must never be written into a file.'
            )

        buffer: ThoughtBuffer = ThoughtBuffer()

        async def stream_thoughts(
            context: RunContext[AgentDeps],
            stream: AsyncIterable[AgentStreamEvent],
        ) -> None:
            """
            Turn the model's text deltas into thoughts as the run proceeds.

            Tool activity is not reported here: the tools already announce
            themselves through the event bus.

            :param context: The run context, unused.
            :param stream: The events produced while the agent runs.
            """
            async for event in stream:
                if self._inbox.stop_requested:
                    # Checked per streamed event, which is what makes stopping
                    # feel immediate rather than waiting out the current step.
                    raise StopRequested()

                if self._connection_lost.is_set():
                    # Checked here as well as between states: execution is the
                    # expensive phase, and waiting for it to finish would keep
                    # calling the model for a client that has gone.
                    raise ClientGone()

                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    for thought in buffer.add(event.part.content, monotonic()):
                        await self._say(thought)
                elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                    for thought in buffer.add(event.delta.content_delta, monotonic()):
                        await self._say(thought)

        result: AgentRunResult[str] = await coder_agent.run(
            prompt,
            deps=self._deps,
            model=resolve_model(self._coder_model),
            event_stream_handler=stream_thoughts,
        )

        remaining: str | None = buffer.flush()
        if remaining is not None:
            await self._say(remaining)

        output: str = result.output
        self._history.add(TurnRole.agent, output)

        if self._session.completed_without_changes:
            # Nothing was changed, and a subagent has confirmed the task never
            # called for a change. There is no diff, so review would have nothing
            # to judge and would reject the task for producing nothing.
            await self._announce('The task needed no file changes. Presenting the answer.')
            return SessionState.completed

        if not self._session.completion_requested:
            # Stopping without signalling completion usually means the agent lost
            # track of the loop rather than finished, so the reviewer decides.
            await self._announce(
                'The agent stopped without calling complete_task. '
                'Reviewing whatever has changed so far.'
            )

        return SessionState.reviewing

    async def _handle_reviewing(self) -> SessionState:
        """
        Review the cumulative diff produced on the task branch.

        :returns: The next state.
        """
        await self._announce('Starting review...', SessionStatus.reviewing)

        changes: list[FileChange] = await self._services.git.get_changed_files()
        if not changes:
            self._last_review = ReviewResult(
                approved=False,
                comments='No files were changed, so there is nothing to review.'
            )
            return SessionState.waiting_review_approval

        # Only the diffs that will actually fit the budget are read.
        diffs: dict[str, str]
        omitted: list[str]
        diffs, omitted = await collect_review_diffs(
            changes, self._services.git.get_file_diff
        )

        # Carry the previous verdict in, so a re-review checks whether its own
        # points were addressed instead of restating them.
        previous: str | None = (
            self._last_review.comments
            if self._last_review is not None and not self._last_review.approved
            else None
        )

        review: AgentRunResult[ReviewResult] = await reviewer_agent.run(
            build_review_prompt(
                self._task, changes, diffs, omitted=omitted, previous_feedback=previous
            ),
            model=resolve_model(self._reviewer_model)
        )
        self._last_review = review.output
        return SessionState.waiting_review_approval

    async def _handle_waiting_review_approval(self) -> SessionState:
        """
        Send the reviewer's verdict and act on it.

        :returns: The next state.
        """
        review: ReviewResult = self._last_review or ReviewResult(
            approved=False, comments='The review produced no verdict.'
        )

        await self._events.emit(ReviewEvent(approved=review.approved, comments=review.comments))
        self._history.add(
            TurnRole.reviewer, review.comments, metadata={'approved': review.approved}
        )

        if review.approved:
            return SessionState.completed

        self._review_rejections += 1
        if self._review_rejections >= self._limits.max_review_retries:
            return await self._ask_how_to_proceed(
                f'The reviewer has rejected the work {self._review_rejections} times.',
                SessionState.re_working
            )

        return SessionState.re_working

    async def _handle_re_working(self) -> SessionState:
        """
        Fold the latest feedback back into planning.

        :returns: The next state.
        """
        await self._announce('Reworking based on the feedback received.', SessionStatus.re_working)
        return SessionState.planning

    async def _handle_completed(self) -> SessionState:
        """
        Report the finished work and wait for the user's merge decision.

        :returns: The next state.
        """
        commits: int = await self._services.git.count_commits()
        summary: str = (
            self._session.completion_summary
            or 'The agent finished without providing a summary.'
        )

        await self._record_status(TaskStatus.completed, summary)
        await self._events.emit(
            DoneEvent(summary=summary, branch_name=self._branch_name, total_commits=commits)
        )

        command: SessionCommand = await self._receive_command()
        if command.is_approval:
            return SessionState.merging

        self._history.add(
            TurnRole.user,
            command.reason or 'The user rejected the finished work without giving a reason.',
            metadata={'rejection': True}
        )
        await self._record_rejection('result', command.reason)
        return SessionState.re_working

    async def _handle_merging(self) -> SessionState:
        """
        Merge the task branch back into the branch it came from.

        :returns: The next state.
        """
        await self._announce(f'Merging {self._branch_name}...', SessionStatus.merging)

        try:
            target: str = await self._services.git.merge_branch(self._branch_name)
        except Exception as error:
            await self._events.emit(
                ErrorEvent(
                    content=(
                        f'Merging {self._branch_name} failed, most likely because of a '
                        f'conflict: {error}. The work is still on the branch.'
                    )
                )
            )
            return SessionState.failed

        await self._record_status(TaskStatus.merged)
        await self._events.emit(
            DoneEvent(
                summary=f'Merged {self._branch_name} into {target}.',
                branch_name=self._branch_name,
                total_commits=await self._services.git.count_commits(target)
            )
        )
        return SessionState.merged

    async def _record_rejection(self, stage: str, reason: str) -> None:
        """
        Note that the user sent work back, if history is being recorded.

        :param stage: What was rejected.
        :param reason: The feedback the user gave.
        """
        if self._recorder is None:
            return
        try:
            await self._recorder.record_rejection(stage, reason)
        except Exception as error:
            logger.warning('Could not record a rejection: %s', error)

    async def _record_status(self, status: TaskStatus, summary: str = '') -> None:
        """
        Move the task record to a new status, if history is being recorded.

        :param status: The status the task has reached.
        :param summary: A summary to store alongside it.
        """
        if self._recorder is None:
            return
        try:
            await self._recorder.record_status(status, summary)
        except Exception as error:
            logger.warning('Could not record a status change: %s', error)

    async def _receive_command(self) -> SessionCommand:
        """
        Wait for the user's next instruction.

        A stop unwinds rather than being returned. Every caller is waiting for an
        answer to a question, and a stop answers none of them: returned as an
        ordinary command it would resolve to a rejection and send the agent round
        again, which is the opposite of what was asked.

        :returns: The parsed command.
        :raises StopRequested: If the user asked to stop.
        """
        command: SessionCommand = await self._inbox.next_command()
        if command.resolved_action is CommandAction.stop:
            raise StopRequested()
        return command

    async def _ask_how_to_proceed(
        self,
        situation: str,
        retry_state: SessionState,
    ) -> SessionState:
        """
        Pause at an iteration cap and let the user decide what happens next.

        :param situation: A description of the cap that was reached.
        :param retry_state: The state to return to if the user wants to keep going.
        :returns: The state to move to next.
        """
        await self._announce(
            f'{situation} Reply with "continue" to keep going, "auto_approve" to accept the '
            'agent\'s work from here on, or "abort" to stop.'
        )

        command: SessionCommand = await self._receive_command()
        action: CommandAction = command.resolved_action

        if action is CommandAction.abort:
            # Recorded the same way as the stop button: both are the user
            # deciding to stop, and history should not tell them apart.
            raise StopRequested()

        if action is CommandAction.auto_approve:
            self._auto_approve = True

        self._plan_rejections = 0
        self._review_rejections = 0
        return retry_state



