from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable
from time import monotonic
from typing import TYPE_CHECKING, Any, TypeVar, cast

from agents.coder import CODER_SYSTEM_PROMPT, AgentDeps
from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolResultEvent,
    ModelMessage,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from services.compaction import RequestHistoryCapture
from services.conversation_history import TurnRole
from services.events import CompactionRunPath
from services.model_resolution import resolve_model
from services.session_state import SessionState
from services.thought_buffer import ThoughtBuffer
from services.client_gone import ClientGone
from services.stop_requested import StopRequested

from .terminal_tool_completed import TerminalToolCompleted

if TYPE_CHECKING:
    from .agent_session import AgentSession


_RunResult = TypeVar('_RunResult')

TERMINAL_TOOL_NAMES: frozenset[str] = frozenset({
    'complete_task',
    'complete_step',
    'accept_step_without_changes',
})


class AgentRunHelper:
    """
    Assemble and execute the capability-equipped long-running agent paths.

    :param session: The session that supplies budgets, dependencies, and event handling.
    """

    _session: AgentSession

    def __init__(self, session: AgentSession) -> None:
        """
        Bind the helper to one session.

        :param session: The owning session.
        """
        self._session = session

    async def run_agent(
        self,
        prompt: str,
        agent: Agent[Any, Any],
        context_window: Callable[[], int],
        run_path: CompactionRunPath,
        system_prompt: str,
        request_history: RequestHistoryCapture | None = None,
        message_history: list[ModelMessage] | None = None,
        output_type: object | None = None,
        event_stream_handler: Callable[
            [RunContext[AgentDeps], AsyncIterable[AgentStreamEvent]], Awaitable[None]
        ] | None = None,
        on_retry: Callable[[], object] | None = None,
    ) -> AgentRunResult[_RunResult]:
        """
        Run an agent with the session's standard capabilities and overflow recovery.

        :param prompt: The instruction for the agent.
        :param agent: The agent to execute.
        :param context_window: Reads the live budget for this run path.
        :param run_path: The path attached to compaction events.
        :param system_prompt: The prompt used to calculate fixed overhead.
        :param request_history: Optional capture reused by a caller-managed retry.
        :param message_history: Optional prior history for the first attempt.
        :param output_type: Optional per-run output type override.
        :param event_stream_handler: Optional handler for streamed agent events.
        :param on_retry: Optional state cleanup after an abandoned attempt.
        :returns: The completed agent result.
        :raises Exception: If the model call cannot be recovered.
        """
        history: RequestHistoryCapture = (
            request_history if request_history is not None else RequestHistoryCapture()
        )

        async def run_attempt(
            overflow_history: list[ModelMessage] | None,
        ) -> AgentRunResult[Any]:
            """
            Run one long-running model attempt.

            :param overflow_history: History recovered from a preceding overflow.
            :returns: The completed agent result.
            """
            budget: int = context_window()
            kwargs: dict[str, object] = {
                'message_history': (
                    overflow_history if overflow_history is not None else message_history
                ),
                'deps': self._session._deps,
                'model': resolve_model(self._session._coder_model),
                'model_settings': self._session._model_settings_for(budget),
                'usage_limits': UsageLimits(request_limit=None),
                'capabilities': [
                    *self._session._compaction_capabilities_for(budget, run_path),
                    history,
                    self._session._usage_monitor_for(budget, system_prompt, agent),
                    self._session._provider_usage_monitor(),
                ],
            }
            if output_type is not None:
                kwargs['output_type'] = output_type
            if event_stream_handler is not None:
                kwargs['event_stream_handler'] = event_stream_handler
            runner: Any = agent
            result: AgentRunResult[Any] = await runner.run(prompt, **kwargs)
            return result

        result: AgentRunResult[Any] = await self._session._overflow_recovery.run_with_overflow_recovery(
            run_attempt,
            context_window,
            history,
            run_path,
            on_retry,
        )
        return cast(AgentRunResult[_RunResult], result)

    async def run_coder_streaming(
        self,
        prompt: str,
        agent: Agent[AgentDeps, str],
        request_history: RequestHistoryCapture | None = None,
        message_history: list[ModelMessage] | None = None,
    ) -> str:
        """
        Run a coder agent, streaming its reasoning to the client as it goes.

        :param prompt: The instruction to give the agent.
        :param agent: The agent to run.
        :param request_history: Captures requests for caller-managed recovery.
        :param message_history: Prior history to resume after caller-managed recovery.
        :returns: The agent's final text output.
        :raises StopRequested: If the user asked for the session to stop.
        :raises ClientGone: If the client disconnected while the agent was running.
        """
        buffer: ThoughtBuffer = ThoughtBuffer()
        terminal_successes_before_run: int = self._session._session.terminal_success_count

        async def stream_thoughts(
            context: RunContext[AgentDeps],
            stream: AsyncIterable[AgentStreamEvent],
        ) -> None:
            """
            Turn model text deltas into thoughts while monitoring session interrupts.

            :param context: The run context, unused.
            :param stream: The events produced while the agent runs.
            :raises StopRequested: If the user asked for the session to stop.
            :raises ClientGone: If the client disconnected while the agent was running.
            :raises TerminalToolCompleted: If a terminal tool succeeds during this run.
            """
            _ = context
            async for event in stream:
                if self._session._inbox.stop_requested:
                    raise StopRequested()
                if self._session._connection_lost.is_set():
                    raise ClientGone()
                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    for thought in buffer.add(event.part.content, monotonic()):
                        await self._session._say(thought)
                elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                    for thought in buffer.add(event.delta.content_delta, monotonic()):
                        await self._session._say(thought)
                elif (
                    isinstance(event, FunctionToolResultEvent)
                    and event.part.tool_name in TERMINAL_TOOL_NAMES
                    and self._session._session.terminal_success_count > terminal_successes_before_run
                ):
                    raise TerminalToolCompleted(event.part.tool_name)

        try:
            result: AgentRunResult[str] = await self.run_agent(
                prompt,
                agent,
                lambda: self._session._coder_context_window,
                CompactionRunPath.coder_streaming,
                CODER_SYSTEM_PROMPT,
                request_history,
                message_history,
                event_stream_handler=stream_thoughts,
                on_retry=buffer.flush,
            )
        except TerminalToolCompleted as completion:
            buffered_thought: str | None = buffer.flush()
            if buffered_thought is not None:
                await self._session._say(buffered_thought)
            completion_output: str = (
                self._session._session.completion_summary or ''
                if str(completion) == 'complete_task'
                else self._session._session.step_summaries[-1]
            )
            self._session._history.add(TurnRole.agent, completion_output)
            return completion_output

        remaining: str | None = buffer.flush()
        if remaining is not None:
            await self._session._say(remaining)
        output: str = result.output
        self._session._history.add(TurnRole.agent, output)
        return output
