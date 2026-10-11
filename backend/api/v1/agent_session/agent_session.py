from asyncio import Event
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from agents.classifier import (
    ClassificationResult,
    classifier_agent,
    build_classification_prompt,
)
from agents.coder import CODER_SYSTEM_PROMPT, AgentDeps, answering_agent, coder_agent
from agents.planner import PLANNER_SYSTEM_PROMPT, planner_agent
from agents.structured_plan import StructuredPlan
from agents.commit_message import commit_message_agent
from agents.reviewer import ReviewResult, reviewer_agent
from fastapi import WebSocket
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai_harness.compaction import ContextUsage, ReportContextUsage, estimate_token_count
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelMessage
from schemas.agent_commands import parse_command
from schemas.command_action import CommandAction
from schemas.session_command import SessionCommand
from services.conversation_history import ConversationHistory
from services.feedback_trail import FeedbackSource, FeedbackTrail, render_for_coder
from services.embedding import DEFAULT_EMBEDDING_MODEL, EmbeddingService
from services.events import (
    CommitEvent,
    CompactionEvent,
    CompactionRunPath,
    WarningEvent,
    INDEXING_PROGRESS_KEY,
    DoneEvent,
    ErrorEvent,
    QuestionEvent,
    EventBus,
    PlanEvent,
    ReviewEvent,
    SandboxStage,
    SandboxUnavailableEvent,
    SessionStatus,
    ContextUsageEvent,
    SystemEvent,
    ThoughtEvent,
)
from services.docker_sandbox import DockerSandboxManager
from services.sandbox_error import SandboxError
from services.file_editor import FileEditorService
from services.commit_result import CommitResult
from services.file_change import FileChange
from services.git_manager import GitManagerService
from services.code_search import CodeSearchService
from services.compaction import (
    build_compaction,
    force_summarize_history,
    looks_like_context_overflow,
    build_usage_monitor,
    resolve_context_window,
)
from services.request_history_capture import RequestHistoryCapture
from services.context_budget import (
    ReportProviderUsage,
    context_window_complaint,
    estimate_fixed_overhead,
    provider_usage_correction,
    provider_usage_divergence,
)
from services.provider_usage_divergence import ProviderUsageDivergence
from services.turn_role import TurnRole
from services.browser_tooling import BROWSER_GUIDANCE
from services.frontend_detection import repository_has_frontend
from services.indexing import IndexingService, clamp_chunk_lines
from services.parsed_plan import ParsedPlan
from services.plan_format import PlanFormat
from services.plan_parser import NO_STEPS_CORRECTION, parse_plan, render_plan
from services.plan_step import PlanStep
from services.review_context import build_review_prompt, collect_review_diffs
from services.model_resolution import (
    effective_context_window,
    measure_ollama_context_window,
    model_settings,
    resolve_model,
)
from services.ollama_native_model import OllamaNativeModelSettings
from services.session_context import SessionContext
from services.session_inbox import SessionInbox
from services.client_gone import ClientGone
from services.stop_requested import StopRequested
from services.user_questioner import UserQuestioner
from services.session_state import SessionState
from services.session_state_machine import SessionStateMachine
from services.session_limits import IterationLimits
from services.session_registry import SessionRegistry
from services.task_recorder import TaskRecorder
from database.task_status import TaskStatus
from database.models.file import File
from database.models.task import Task
from uuid import UUID, uuid4
from .agent_run_helper import AgentRunHelper
from .commit_failure_decision import (
    COMMIT_FAILURE_COMMITTED,
    COMMIT_FAILURE_OPTIONS,
    COMMIT_FAILURE_STOP,
)
from .commit_sweeper import (
    MAX_CONSECUTIVE_COMMIT_FAILURES,
    CommitSweeper,
)
from .context_overflow_recovery import ContextOverflowRecovery
from .planning_coordinator import PlanningCoordinator
from .session_services import SessionServices
import json
import logging
import os

logger = logging.getLogger(__name__)

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
    _feedback_trail: FeedbackTrail
    _machine: SessionStateMachine
    _limits: IterationLimits

    _task: str
    _branch_name: str
    _branch_created: bool
    _coder_model: str
    _reviewer_model: str
    _coder_context_window: int
    _planner_context_window: int
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
    _has_frontend: bool | None
    _last_context_usage: ContextUsage | None
    _last_estimated_input_tokens: int | None
    _provider_usage_budget_correction: int
    _provider_usage_observation: int
    _provider_truncation_recovery_observation: int | None
    _provider_underestimate_warning_emitted: bool
    _measured_context_window: int | None
    _context_window_probe_completed: bool
    _context_window_warning_emitted: bool
    _context_window_probe: Callable[[str], Awaitable[int | None]]
    _commit_message_generator: Callable[[str], Awaitable[str]]
    _overflow_recovery: ContextOverflowRecovery
    _commit_sweeper: CommitSweeper
    _agent_run_helper: AgentRunHelper
    _planning_coordinator: PlanningCoordinator

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
        context_window_probe: Callable[[str], Awaitable[int | None]] | None = None,
        commit_message_generator: Callable[[str], Awaitable[str]] | None = None,
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
        :param context_window_probe: Optional source for serving-model context limits.
        :param commit_message_generator: Optional generator for a squash commit subject.
        """
        self._websocket = websocket
        self._inbox = SessionInbox(websocket.receive_text, on_stop=self._interrupt_work)
        self._services = services
        self._session_id = session_id
        self._events = events
        self._deps = deps
        # The tools reach the user only through this: asking a question and
        # awaiting one reply, rather than the whole inbox, which a tool could
        # use to consume an approval meant for the session.
        self._deps.questioner = UserQuestioner(
            emit_question=self._ask_question, await_reply=self._inbox.await_reply
        )
        self._session = session
        self._history = history
        self._feedback_trail = FeedbackTrail()
        self._machine = machine
        self._limits = limits
        self._connection_lost = connection_lost if connection_lost is not None else Event()
        self._context_window_probe = (
            context_window_probe
            if context_window_probe is not None
            else measure_ollama_context_window
        )
        self._commit_message_generator = (
            commit_message_generator
            if commit_message_generator is not None
            else self._generate_commit_message
        )

        self._task = ''
        self._branch_name = f'ai-task-{session_id}'
        self._branch_created = False
        self._coder_model = DEFAULT_MODEL
        self._reviewer_model = DEFAULT_MODEL
        # Zero means "unset", which `build_compaction` reads as the default.
        self._coder_context_window = 0
        self._planner_context_window = 0
        self._plan = ParsedPlan(steps=[])
        self._plan_rejections = 0
        self._review_rejections = 0
        self._total_iterations = 0
        self._auto_approve = False
        self._last_review = None
        self._dockerfile = ''
        self._secrets = {}
        # Unresolved rather than False: the repository is not known until a task
        # arrives with its path, which is after this point.
        self._has_frontend = None
        self._last_context_usage = None
        self._last_estimated_input_tokens = None
        self._provider_usage_budget_correction = 0
        self._provider_usage_observation = 0
        self._provider_truncation_recovery_observation = None
        self._provider_underestimate_warning_emitted = False
        self._measured_context_window = None
        self._context_window_probe_completed = False
        self._context_window_warning_emitted = False
        self._task_id = ''
        self._recorder = None
        self._overflow_recovery = ContextOverflowRecovery(self)
        self._commit_sweeper = CommitSweeper(self)
        self._agent_run_helper = AgentRunHelper(self)
        self._planning_coordinator = PlanningCoordinator(self)

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
        _ = await self._measure_context_window()
        await self._warn_if_context_is_too_small()
        # Only now is a task genuinely under way. The slot was taken when the
        # socket connected, which is too early to stand for "an agent is
        # working": anything reading that would block on an idle tab.
        _ = await self._services.registry.begin_task(self._session_id)
        self._inbox.start()

        try:
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
        finally:
            # In a `finally` so that an escaping exception cannot leave the
            # session looking as though it were still working. The connection may
            # well outlive this — a finished review is still worth reading — and
            # the index is safe to touch the moment nothing is working on it.
            _ = await self._services.registry.end_task(self._session_id)

    async def _advance(self, state: SessionState) -> SessionState:
        """
        Run the handler for the current state.

        :param state: The state to handle.
        :returns: The state to move to next.
        :raises RuntimeError: If the state has no handler.
        """
        handlers: dict[SessionState, Callable[[], Awaitable[SessionState]]] = {
            SessionState.initializing: self._handle_initializing,
            SessionState.classifying: self._handle_classifying,
            SessionState.planning: self._planning_coordinator.handle_planning,
            SessionState.waiting_plan_approval: self._handle_waiting_plan_approval,
            SessionState.executing: self._handle_executing,
            SessionState.reviewing: self._handle_reviewing,
            SessionState.waiting_review_approval: self._handle_waiting_review_approval,
            SessionState.re_working: self._handle_re_working,
            SessionState.answering: self._handle_answering,
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
        self._coder_context_window = _read_context_window(message.get('coder_context_window'))
        self._planner_context_window = _read_context_window(
            message.get('planner_context_window')
        )
        self._services.embedding.embedding_model = str(
            message.get('embedding_model', DEFAULT_EMBEDDING_MODEL)
        )
        # Absent or null means automatic sizing, which is the default and the
        # only mode that recovers from a chunk the model will not take.
        self._services.indexing.chunk_lines = clamp_chunk_lines(
            _read_chunk_lines(message.get('chunk_lines'))
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
            # Repointed with the rest. Left behind, the agent would search the
            # project named by REPO_PATH while the indexer wrote to this one,
            # and every search would come back empty rather than failing.
            self._services.search.repo_path = repo_path

        self._history.add(TurnRole.user, self._task)

    async def _measure_context_window(self) -> int | None:
        """
        Measure the configured Ollama model once and apply its serving limit.

        :returns: The measured context window, or None when it could not be read.
        """
        if self._context_window_probe_completed:
            return self._measured_context_window

        measured: int | None = await self._context_window_probe(self._coder_model)
        self._context_window_probe_completed = True
        self._measured_context_window = measured
        if measured is None:
            return None

        self._coder_context_window = effective_context_window(
            self._coder_context_window, measured
        )
        self._planner_context_window = effective_context_window(
            self._planner_context_window, measured
        )
        return measured

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

        if self._recorder is not None:
            await self._recorder.record_status(TaskStatus.running)
        return SessionState.classifying

    async def _repository_is_empty(self) -> bool:
        """
        Check whether indexing found anything to work with.

        :returns: True if no file was indexed.
        """
        try:
            # Scoped to this project: another project's index says nothing about
            # whether this repository had anything to work with.
            return await File.filter(project=self._services.indexing.project).count() == 0
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

        if self._recorder is not None:
            await self._recorder.record_status(TaskStatus.running)
        return SessionState.classifying

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
                    coder_context_window=self._coder_context_window,
                    planner_context_window=self._planner_context_window,
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

    async def _report_context_usage(self, usage: ContextUsage, fixed_overhead: int = 0) -> None:
        """
        Publish how full the context is, measured rather than guessed.

        Sent as counts rather than a sentence, so the interface can show a gauge
        and word it however it likes. It replaces the previous reading rather
        than joining the feed: one line per request would bury the conversation
        in a number that is only interesting at a glance.

        `resolved` is passed through untouched, because a percentage taken
        against a fallback window is a guess and the interface should be able to
        mark it as one.

        :param usage: The reading for the request about to be sent.
        :param fixed_overhead: Tokens sent outside the compactable message history.
        """
        self._last_context_usage = usage
        self._last_estimated_input_tokens = max(usage.used_tokens + fixed_overhead, 0)
        self._provider_usage_observation += 1
        await self._events.emit(
            ContextUsageEvent(
                used_tokens=max(usage.used_tokens, 0),
                window_tokens=max(usage.window_tokens, 1),
                resolved=usage.resolved,
            )
        )

    async def _reconcile_provider_usage(self, provider_input_tokens: int) -> None:
        """
        Compare the preceding request estimate with provider-reported prompt usage.

        The provider is the authority on what it processed. A severe shortfall is
        useful evidence of server-side truncation, while a severe excess reserves
        room from subsequent compaction budgets without changing the request that
        is already in flight.

        :param provider_input_tokens: Prompt tokens reported by the successful response.
        """
        estimated_input_tokens: int | None = self._last_estimated_input_tokens
        if estimated_input_tokens is None:
            return

        divergence: ProviderUsageDivergence = provider_usage_divergence(
            estimated_input_tokens, provider_input_tokens
        )
        if divergence is ProviderUsageDivergence.input_truncated:
            if self._provider_truncation_recovery_observation == self._provider_usage_observation:
                return
            self._provider_truncation_recovery_observation = self._provider_usage_observation
            _ = await self._overflow_recovery.narrow_after_input_truncation(
                resolve_context_window(self._coder_context_window),
                estimated_input_tokens,
                provider_input_tokens,
            )
            return

        if divergence is not ProviderUsageDivergence.estimate_too_low:
            return

        correction: int = provider_usage_correction(estimated_input_tokens, provider_input_tokens)
        self._provider_usage_budget_correction = max(
            self._provider_usage_budget_correction, correction
        )
        if self._provider_underestimate_warning_emitted:
            return
        self._provider_underestimate_warning_emitted = True
        await self._events.emit(
            WarningEvent(
                content=(
                    f'The provider processed {provider_input_tokens:,} prompt tokens, but Loom estimated '
                    f'{estimated_input_tokens:,}. Loom will reserve {correction:,} additional context tokens '
                    'for later requests so its compaction budget reflects the provider\'s accounting.'
                )
            )
        )

    async def _run_with_overflow_recovery(
        self,
        run: Callable[[list[ModelMessage] | None], Awaitable[Any]],
        context_window: Callable[[], int],
        history: RequestHistoryCapture,
        run_path: CompactionRunPath,
        on_retry: Callable[[], object] | None = None,
    ) -> Any:
        """
        Delegate legacy private callers to context-overflow recovery.

        :param run: Starts a model-run attempt with optional recovered history.
        :param context_window: Reads the budget that owns each attempt.
        :param history: Captures the final compacted request before a failed call.
        :param run_path: Identifies recovery compaction in the event feed.
        :param on_retry: Clears state belonging to an abandoned attempt.
        :returns: The successful model-run result.
        :raises Exception: If the failure is unrelated or cannot be recovered.
        """
        return await self._overflow_recovery.run_with_overflow_recovery(
            run, context_window, history, run_path, on_retry
        )

    async def _report_compaction(
        self,
        run_path: CompactionRunPath,
        tokens_before: int,
        tokens_after: int,
    ) -> None:
        """
        Log and publish one request-history compaction.

        :param run_path: The long-running path whose request was compacted.
        :param tokens_before: Estimated history tokens before compaction.
        :param tokens_after: Estimated history tokens after compaction.
        """
        try:
            logger.info(
                'Compacted %s request history from an estimated %s to %s tokens.',
                run_path.value,
                tokens_before,
                tokens_after,
            )
            await self._events.emit(
                CompactionEvent(
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                    run_path=run_path,
                )
            )
        except Exception:
            logger.exception('Could not publish a compaction event.')

    def _compaction_for(
        self, context_window: int, run_path: CompactionRunPath
    ) -> AbstractCapability[Any]:
        """
        Build a compaction ladder against this session's effective window.

        :param context_window: The effective window for the current run path.
        :param run_path: The long-running path the capability is attached to.
        :returns: The configured compaction capability.
        """
        corrected_window: int = context_window
        if self._provider_usage_budget_correction:
            corrected_window = max(
                resolve_context_window(context_window) - self._provider_usage_budget_correction, 1
            )
        async def report(tokens_before: int, tokens_after: int) -> None:
            """
            Attach this capability's path to a compaction notice.

            :param tokens_before: Estimated history tokens before compaction.
            :param tokens_after: Estimated history tokens after compaction.
            """
            await self._report_compaction(run_path, tokens_before, tokens_after)

        if self._measured_context_window is None and not self._provider_usage_budget_correction:
            return build_compaction(corrected_window, on_compaction=report)
        return build_compaction(
            corrected_window, allow_subminimum=True, on_compaction=report
        )

    def _compaction_capabilities_for(
        self, context_window: int, run_path: CompactionRunPath
    ) -> list[AbstractCapability[Any]]:
        """
        Build the ordered compaction capabilities for one long-running run.

        :param context_window: The effective window for the current run path.
        :param run_path: The long-running path the capabilities are attached to.
        :returns: The reactive compaction ladder.
        """
        return [self._compaction_for(context_window, run_path)]

    def _usage_monitor_for(
        self,
        context_window: int,
        system_prompt: str = CODER_SYSTEM_PROMPT,
        agent: Agent[Any, Any] | None = None,
    ) -> ReportContextUsage:
        """
        Build a usage monitor against this session's effective window.

        :param context_window: The effective window for the current run path.
        :param system_prompt: The prompt sent outside the compactable history.
        :param agent: The agent whose tool schemas are sent with the prompt.
        :returns: The configured usage-monitoring capability.
        """
        overhead_agent: Agent[Any, Any] = agent if agent is not None else coder_agent
        fixed_overhead: int = estimate_fixed_overhead(system_prompt, overhead_agent)

        async def report(usage: ContextUsage) -> None:
            """
            Record the history estimate with this run path's fixed overhead.

            :param usage: The harness's pre-request history estimate.
            """
            await self._report_context_usage(usage, fixed_overhead)

        if self._measured_context_window is None:
            return build_usage_monitor(context_window, report)
        return build_usage_monitor(
            context_window, report, allow_subminimum=True
        )

    def _provider_usage_monitor(self) -> ReportProviderUsage:
        """
        Build the capability that receives each successful provider response.

        :returns: The response-usage reporting capability.
        """
        return ReportProviderUsage(self._reconcile_provider_usage)

    def _model_settings_for(self, context_window: int) -> OllamaNativeModelSettings:
        """
        Build provider settings against this session's effective window.

        :param context_window: The effective window for the current run path.
        :returns: The provider settings for the run.
        """
        return model_settings(
            context_window, allow_subminimum=self._measured_context_window is not None
        )

    async def _handle_classifying(self) -> SessionState:
        """
        Decide whether the task requires changing any file, before planning it.

        Asked of the task alone. A task that needs no changes is never planned,
        never put to plan approval and never offered a merge, because none of
        those decide anything about an answer.

        :returns: The next state.
        """
        await self._announce(
            'Working out whether this task requires changing any files...',
            SessionStatus.classifying,
        )

        try:
            verdict: AgentRunResult[ClassificationResult] = await classifier_agent.run(
                build_classification_prompt(self._task),
                model=resolve_model(self._reviewer_model),
                model_settings=self._model_settings_for(self._coder_context_window),
                usage_limits=UsageLimits(request_limit=None),
            )
        except Exception as error:
            # Guessing either way is worse than stopping: guessing "no changes"
            # would end real work with an answer, and guessing "changes" would
            # put a question through planning and a merge decision.
            logger.exception('The task could not be classified.')
            await self._events.emit(
                ErrorEvent(content=f'The task could not be classified, so it was not started: {error}')
            )
            return SessionState.failed

        decision: ClassificationResult = verdict.output
        self._session.record_classification(decision.changes_are_required)

        if decision.changes_are_required:
            await self._announce(f'This task requires changing files. {decision.reason}')
            await self._ensure_branch()
            return SessionState.planning

        await self._announce(
            f'This task needs no file changes. {decision.reason} '
            'Reading the repository to answer it.'
        )
        return SessionState.answering

    async def _ask_question(self, question: str, options: Sequence[str]) -> None:
        """
        Put the agent's question to the user.

        The question joins the feed rather than appearing as a transient prompt:
        it and the answer are part of what happened, and a reader scrolling back
        should find the exchange that shaped the work.

        :param question: What the agent asked.
        :param options: The answers it suggested, already truncated.
        """
        await self._events.emit(QuestionEvent(question=question, options=list(options)))

    async def _ensure_branch(self) -> None:
        """
        Create the task branch, once, for work that is going to change files.

        Deliberately not done while initialising. A task that needs no changes
        never reaches here, so it leaves no empty branch behind and no branch
        name on its history record — it did not touch the repository at all,
        and nothing about it should suggest otherwise.
        """
        if self._branch_created:
            return

        await self._services.git.create_branch(self._branch_name)
        self._branch_created = True
        if self._recorder is not None:
            self._recorder.task.branch_name = self._branch_name

    async def _handle_answering(self) -> SessionState:
        """
        Let the agent read what it needs and answer, without planning.

        :returns: The next state.
        """
        await self._announce('Answering the task...', SessionStatus.answering)

        prompt: str = (
            f'Task: {self._task}\n\n'
            'This task was judged to need no file changes, so you have read-only access: the '
            'tools that write files, edit them or run commands are not available to you here. '
            'Read whatever you need to answer it, then call complete_task with your answer as '
            'the summary. If it turns out the task cannot be carried out without changing a '
            'file, say so in your summary and stop; it will be replanned as ordinary work.'
        )

        await self._agent_run_helper.run_coder_streaming(prompt, answering_agent)

        # Nothing is checked about the working tree here, because the answering
        # agent has no tool that could have changed it: `create_file`,
        # `edit_file` and `run_test_command` are not registered on it. The only
        # way this path fails to produce an answer is by not producing one —
        # which is what an agent does when it finds the task cannot be carried
        # out without a change it is unable to make.
        if not self._session.completion_requested:
            return await self._escalate_to_planning()

        return await self._present_answer()

    async def _escalate_to_planning(self) -> SessionState:
        """
        Take a task that turned out to need changes onto the ordinary route.

        Reached when the answering agent modified something, or stopped without
        giving an answer. Either way the classification did not hold, and the
        work is planned as it would have been in the first place.

        The verdict is corrected on the way past, so that the completion gate
        stops excusing a finish with nothing modified.

        :returns: The next state.
        """
        self._session.record_classification(True)
        await self._ensure_branch()
        await self._announce(
            'This task turned out to need file changes after all. Planning the work.',
            SessionStatus.planning,
        )
        return SessionState.planning

    async def _present_answer(self) -> SessionState:
        """
        Report the answer and end the session.

        No branch and no commit count are reported, because nothing was changed:
        there is no merge to offer, and offering one would promise a diff that
        does not exist.

        :returns: The next state.
        """
        answer: str = (
            self._session.completion_summary
            or 'The agent finished without providing an answer.'
        )

        await self._record_status(TaskStatus.answered, answer)
        await self._events.emit(DoneEvent(summary=answer))
        return SessionState.answered

    async def _handle_planning(self) -> SessionState:
        """Delegate planning to the session planning coordinator.

        :returns: The next state.
        """
        return await self._planning_coordinator.handle_planning()

    async def _plan_structured(self, prompt: str) -> ParsedPlan:
        """Delegate structured planning to the session planning coordinator.

        :param prompt: The planning prompt.
        :returns: The parsed structured plan.
        """
        return await self._planning_coordinator.plan_structured(prompt)

    async def _plan_as_text(self, prompt: str) -> str:
        """Delegate text planning to the session planning coordinator.

        :param prompt: The planning prompt.
        :returns: The planner output.
        """
        return await self._planning_coordinator.plan_as_text(prompt)

    def _build_planning_prompt(self) -> str:
        """Delegate planning-prompt construction to the planning coordinator.

        :returns: The planning prompt.
        """
        return self._planning_coordinator.build_planning_prompt()

    async def _handle_waiting_plan_approval(self) -> SessionState:
        """
        Show the plan to the user and act on their answer.

        :returns: The next state.
        """
        await self._events.emit(
            PlanEvent(
                steps=list(self._plan.steps),
                format=self._plan.format,
                requires_approval=not self._auto_approve,
            )
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
        self._feedback_trail.add(
            FeedbackSource.user_plan_rejection,
            self._history.iteration,
            command.reason or 'The user rejected this plan without giving a reason.',
        )
        await self._record_rejection('plan', command.reason)

        if self._plan_rejections >= self._limits.max_plan_retries:
            return await self._ask_how_to_proceed(
                f'The plan has been rejected {self._plan_rejections} times.',
                SessionState.planning
            )

        return SessionState.planning

    def _repository_has_frontend(self) -> bool:
        """
        Whether this session's repository plausibly contains a web frontend.

        Cached for the life of the session. The answer is settled by files on
        disk, and execution is re-entered on every replanning loop, so asking
        git to list the repository again each time would pay the same cost for
        the same answer.

        :returns: True if the repository looks like it has a frontend.
        """
        if self._has_frontend is None:
            self._has_frontend = repository_has_frontend(self._services.git.repo_path)
        return self._has_frontend

    async def _handle_executing(self) -> SessionState:
        """
        Work through the approved plan one step at a time.

        One run per step rather than one run for the whole plan. Handed
        everything at once, a small model reads the plan as a description of the
        finished state and stops after whichever part it found most legible,
        because nothing obliges it to account for the rest. A step at a time
        makes finishing each one an explicit act with its own gate.

        A step is given the earlier steps' summaries rather than their history.
        Carrying the raw tool calls forward would recreate the context pressure
        compaction exists to relieve, and by the tenth step the first step's file
        reads are the least useful thing in the window.

        :returns: The next state.
        """
        await self._warn_if_context_is_too_small()
        await self._announce('Executing plan...', SessionStatus.executing)

        steps: Sequence[PlanStep] = self._plan.steps
        if not steps:
            # Nothing to iterate over, but the task still has to be executed:
            # an unparsed plan is not an empty one.
            await self._execute_step(None, 1, 1)
            return await self._finish_execution()

        for position, step in enumerate(steps, start=1):
            if self._inbox.stop_requested:
                raise StopRequested()

            await self._announce(
                f'Step {position} of {len(steps)}: {step.title}', SessionStatus.executing
            )
            await self._execute_step(step, position, len(steps))
            await self._commit_sweeper.resolve_commit_failures()

        return await self._finish_execution()

    async def _execute_step(
        self, step: PlanStep | None, position: int, total: int
    ) -> None:
        """
        Run the coder against one step of the plan.

        The graph is run with `run()` and an event stream handler rather than
        `run_stream()`. `run_stream()` stops the graph once it has an output
        matching the output type, which for a text-output agent means it
        executes only the first round of tool calls and abandons the rest. A
        ReAct sequence such as read, edit, test, complete would be cut off after
        its first step, so the agent could never finish the work.

        :param step: The step to work on, or None to execute the plan as a whole.
        :param position: Which step this is, counting from one.
        :param total: How many steps the plan has.
        """
        before: int = len(self._session.step_summaries)
        if step is not None:
            self._session.begin_step(step.title)

        prompt: str = self._build_step_prompt(step, position, total)
        history: RequestHistoryCapture = RequestHistoryCapture()
        try:
            await self._agent_run_helper.run_coder_streaming(
                prompt, coder_agent, request_history=history
            )
        except UnexpectedModelBehavior as error:
            if not _looks_like_tool_retry_exhaustion(error):
                raise

            compacted: list[ModelMessage] | None = await self._compact_after_tool_retry_exhaustion(
                history
            )
            try:
                await self._agent_run_helper.run_coder_streaming(
                    prompt,
                    coder_agent,
                    request_history=history,
                    message_history=compacted,
                )
            except UnexpectedModelBehavior as retry_error:
                if not _looks_like_tool_retry_exhaustion(retry_error):
                    raise
                await self._record_tool_retry_exhaustion(step, position)
                return

        if step is not None and len(self._session.step_summaries) == before:
            # Not a failure. The agent may well have done the work and simply
            # not said so, and abandoning the remaining steps over a missing
            # tool call would throw away more than it protects.
            await self._announce(
                f'Step {position} finished without being marked complete. '
                'Continuing to the next step.'
            )
            self._session.record_step_completion(
                f'{step.title} — the agent did not report what it did for this step.'
            )

    async def _compact_after_tool_retry_exhaustion(
        self, history: RequestHistoryCapture
    ) -> list[ModelMessage] | None:
        """
        Force one summary pass before retrying an exhausted tool call.

        :param history: The final compacted request from the failed coder run.
        :returns: The compacted history for one retry, if request history was captured.
        """
        messages: list[ModelMessage] = history.messages
        if not messages:
            return None

        before: int = estimate_token_count(messages)
        try:
            compacted: list[ModelMessage] = await force_summarize_history(
                messages,
                self._coder_context_window,
                resolve_model(self._coder_model),
                allow_subminimum=self._measured_context_window is not None,
            )
        except Exception:
            logger.exception('Could not compact request history after tool retries were exhausted.')
            return messages

        after: int = estimate_token_count(compacted)
        if after < before:
            await self._report_compaction(CompactionRunPath.coder_streaming, before, after)
        return compacted

    async def _record_tool_retry_exhaustion(
        self,
        step: PlanStep | None,
        position: int,
    ) -> None:
        """
        Mark a persistently failing step incomplete and continue execution.

        :param step: The failed step, or None when executing an unparsed plan.
        :param position: The one-based position of the failed step.
        """
        description: str = (
            f'Step {position}' if step is None else f'Step {position} ({step.title})'
        )
        await self._events.emit(
            WarningEvent(
                content=(
                    f'{description} could not complete because a tool exhausted its retry budget even '
                    'after Loom compacted the request and retried once. Continuing to the next step.'
                )
            )
        )
        if step is None:
            return

        self._session.record_step_completion(
            f'{step.title} — incomplete because a tool exhausted its retry budget after one '
            'compaction retry.'
        )

    async def _warn_if_context_is_too_small(self) -> None:
        """
        Say so, once, if the window cannot hold what every request must carry.

        The provider's own error for this names neither the window nor what
        filled it, so it reads as a bug in Loom rather than as a setting to
        change. Checked before the first step rather than after a failure,
        because by then the run is over.
        """
        if self._context_window_warning_emitted:
            return

        window: int = (
            self._coder_context_window
            if self._measured_context_window is not None
            else resolve_context_window(self._coder_context_window)
        )
        complaint: str = context_window_complaint(
            window,
            estimate_fixed_overhead(CODER_SYSTEM_PROMPT, coder_agent),
        )
        if complaint:
            await self._events.emit(WarningEvent(content=complaint))
            self._context_window_warning_emitted = True

    async def _sweep_uncommitted(self) -> None:
        """Delegate the final working-tree sweep to the commit sweeper.

        :returns: None.
        """
        await self._commit_sweeper.sweep_uncommitted()

    async def _resolve_commit_failures(self) -> None:
        """Delegate repeated commit-failure resolution to the commit sweeper.

        :returns: None.
        :raises StopRequested: If the user chooses to stop.
        """
        await self._commit_sweeper.resolve_commit_failures()

    def _build_step_prompt(self, step: PlanStep | None, position: int, total: int) -> str:
        """
        Build the prompt for one step's run.

        :param step: The step being worked on, or None for the whole plan.
        :param position: Which step this is, counting from one.
        :param total: How many steps the plan has.
        :returns: The prompt to send.
        """
        prompt: str = (
            f'Task: {self._task}\n\n'
            f'Approved plan:\n{render_plan(self._plan)}\n\n'
        )

        established: str = self._session.test_command
        waiver: str = self._session.verification_waiver
        if waiver:
            # Applies to both branches. A decision the user already made must not
            # be put to them again, and a prompt that still demands verification
            # after it was waived contradicts the tool that waived it.
            prompt += (
                f'Verification is not required for this plan: {waiver}\n'
                'Do not run tests and do not ask about them again.\n\n'
            )
        elif established:
            prompt += (
                'Verify your work with this command, established earlier in this plan:\n'
                f'    {established}\n'
                'If it turns out to be the wrong command for the work in hand, rather than a '
                'real test failure, call forget_test_command and work out the right one.\n\n'
            )

        if step is None:
            prompt += (
                'Execute this plan now. '
                + ('' if waiver else 'Verify every change with run_test_command. ')
                + 'Call complete_task once the work is finished.'
            )
        else:
            done: Sequence[str] = self._session.step_summaries
            if done:
                # The summaries, not the history: what a step needs from the ones
                # before it is what they achieved, not how they went about it.
                completed: str = '\n'.join(
                    f'- {summary}' for summary in done
                )
                prompt += f'Steps already done:\n{completed}\n\n'

            prompt += (
                f'You are working on step {position} of {total}, and only on this step:\n'
                f'{step.title}\n{step.description}\n\n'
                'Do the work this step calls for and nothing belonging to a later step. '
                + ('' if waiver else 'Verify every change with run_test_command. ')
                + 'Call complete_step when the step '
                'is done, or accept_step_without_changes if the step was investigative and '
                'required no change. Do not call complete_task: the plan is not finished until '
                'every step is.'
            )

        feedback: str = render_for_coder(self._feedback_trail)
        if feedback:
            prompt += f'\n\n{feedback}'

        if self._repository_has_frontend():
            prompt += f'\n\n{BROWSER_GUIDANCE}'

        if self._secrets:
            prompt += (
                '\n\nThese environment variables are set inside the sandbox: '
                f'{", ".join(sorted(self._secrets))}. Reference them by name; their '
                'values are not available to you and must never be written into a file.'
            )

        return prompt

    async def _finish_execution(self) -> SessionState:
        """
        Settle the task's completion summary once every step has run.

        Assembled from the steps' own summaries rather than by asking a model to
        write one. A rollup call would cost a request and could disagree with the
        steps it was summarising, and the reviewer reads the diff regardless.

        :returns: The next state.
        """
        await self._commit_sweeper.sweep_uncommitted()

        summaries: Sequence[str] = self._session.step_summaries
        if summaries:
            # Recorded even if completion was already requested. A model that
            # ignored its instructions and called `complete_task` part way
            # through a plan would otherwise have that one step's summary stand
            # for the whole task, with every later step's work unmentioned.
            self._session.record_rollup_completion(
                '\n'.join(f'{index}. {text}' for index, text in enumerate(summaries, start=1))
            )
        elif not self._session.completion_requested:
                # Stopping without signalling completion usually means the agent
                # lost track of the loop rather than finished, so the reviewer
                # decides.
                await self._announce(
                    'The agent stopped without reporting any completed step. '
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
            model=resolve_model(self._reviewer_model),
            model_settings=self._model_settings_for(self._coder_context_window),
            usage_limits=UsageLimits(request_limit=None),
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
        if not review.approved:
            self._feedback_trail.add(
                FeedbackSource.reviewer_rejection, self._history.iteration, review.comments
            )

        if review.approved:
            return SessionState.completed

        command: SessionCommand = await self._receive_command()
        if command.resolved_action is CommandAction.merge:
            summary: str = self._session.completion_summary or 'The agent finished without providing a summary.'
            await self._record_status(TaskStatus.completed, summary)
            return SessionState.merging

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
        self._feedback_trail.add(
            FeedbackSource.user_result_rejection,
            self._history.iteration,
            command.reason or 'The user rejected the finished work without giving a reason.',
        )
        await self._record_rejection('result', command.reason)
        return SessionState.re_working

    async def _handle_merging(self) -> SessionState:
        """
        Merge the task branch back into the branch it came from.

        :returns: The next state.
        """
        await self._announce(f'Merging {self._branch_name}...', SessionStatus.merging)

        commit_message: str = await self._squash_commit_message()
        try:
            target: str = await self._services.git.merge_branch(self._branch_name, commit_message)
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

        _ = await self._services.git.delete_branch(self._branch_name)
        await self._record_status(TaskStatus.merged)
        await self._events.emit(
            DoneEvent(
                summary=f'Merged {self._branch_name} into {target}.',
                branch_name=self._branch_name,
                total_commits=1,
                merged=True,
            )
        )
        return SessionState.merged

    async def _squash_commit_message(self) -> str:
        '''
        Generate a squash commit subject, falling back to the task prompt.

        :returns: A non-empty commit subject.
        '''
        try:
            subjects: list[str] = await self._services.git.get_commit_subjects()
            prompt: str = self._commit_message_prompt(subjects)
            message: str = (await self._commit_message_generator(prompt)).strip()
        except Exception as error:
            logger.warning('Could not generate squash commit message: %s', error)
            message = ''

        if message:
            subject: str = message.splitlines()[0].strip().strip('"`')
            if len(subject) > 2 and subject[0].isdigit() and subject[1] in '.)':
                subject = subject[2:].lstrip()
            subject = subject.rstrip('.')
            if subject:
                return subject

        fallback: str = self._task.splitlines()[0].strip()[:72]
        await self._events.emit(
            WarningEvent(
                content=(
                    'The commit message could not be generated; the squashed commit will use '
                    'the task prompt instead.'
                )
            )
        )
        return fallback

    def _commit_message_prompt(self, subjects: list[str]) -> str:
        '''
        Build the compact context for the commit-message agent.

        :param subjects: Commit subjects written on the task branch.
        :returns: The prompt supplied to the commit-message agent.
        '''
        feedback: list[str] = [
            turn.content
            for turn in self._history.turns
            if turn.role is TurnRole.user and turn.content != self._task
        ]
        return (
            f'Original task:\n{self._task}\n\n'
            f'User feedback:\n{chr(10).join(feedback) or "(none)"}\n\n'
            f'Task branch commit subjects:\n{chr(10).join(subjects) or "(none)"}'
        )

    async def _generate_commit_message(self, prompt: str) -> str:
        '''
        Ask the dedicated agent to write a squash commit subject.

        :param prompt: The task and feedback context for the agent.
        :returns: The agent's proposed commit subject.
        '''
        result: AgentRunResult[str] = await commit_message_agent.run(
            prompt,
            model=resolve_model(self._coder_model),
            model_settings=self._model_settings_for(self._coder_context_window),
            usage_limits=UsageLimits(request_limit=None),
        )
        return result.output

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





def _looks_like_tool_retry_exhaustion(error: UnexpectedModelBehavior) -> bool:
    """
    Recognize a tool call that used all retries available to its agent.

    :param error: The model-behavior error that interrupted a coder run.
    :returns: True only for Pydantic AI's exhausted-tool-retry failure.
    """
    return 'exceeded max retries' in str(error).lower()


def _read_chunk_lines(value: Any) -> int | None:
    """
    Interpret the chunk size the client sent.

    Anything unusable is treated as a request for automatic sizing, because a
    client sending nonsense should get the behaviour that always works rather
    than a session that refuses to start.

    :param value: The value from the opening message.
    :returns: The requested size, or None for automatic sizing.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _read_context_window(value: Any) -> int:
    """
    Interpret a context-window setting the client sent.

    Anything unusable becomes zero, which means "unset" and leaves the default
    to apply. A client sending nonsense should get the behaviour that always
    works rather than a session that refuses to start, matching how
    `_read_chunk_lines` treats the same problem.

    :param value: The value from the opening message.
    :returns: The requested window in tokens, or zero for the default.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value if value > 0 else 0
