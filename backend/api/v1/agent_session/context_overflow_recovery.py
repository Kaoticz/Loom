from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

from agents.coder import CODER_SYSTEM_PROMPT, coder_agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness.compaction import estimate_token_count
from services.compaction import (
    RequestHistoryCapture,
    force_summarize_history,
    looks_like_context_overflow,
    narrowed_window,
    resolve_context_window,
)
from services.context_budget import context_window_complaint, estimate_fixed_overhead
from services.event_bus import CompactionRunPath, WarningEvent
from services.model_resolution import resolve_model

if TYPE_CHECKING:
    from .agent_session import AgentSession


logger = logging.getLogger(__name__)

_RunResult = TypeVar('_RunResult')


class ContextOverflowRecovery:
    """
    Recover long-running model calls from context-window overflows.

    :param session: The session whose budgets, events, and model are recovered.
    """

    _session: AgentSession

    def __init__(self, session: AgentSession) -> None:
        """
        Bind recovery to one session.

        :param session: The owning session.
        """
        self._session = session

    async def narrow_context_window(
        self,
        budget: int,
        warning_content: Callable[[int, int], str],
    ) -> bool:
        """
        Halve both working windows unless their fixed overhead makes that useless.

        :param budget: The window demonstrated to be too optimistic.
        :param warning_content: Builds the cause-specific warning after narrowing.
        :returns: True if the window was narrowed and another run is worthwhile.
        """
        narrower: int = narrowed_window(budget)
        configured: int = resolve_context_window(budget)
        if narrower >= configured:
            return False

        overhead: int = estimate_fixed_overhead(CODER_SYSTEM_PROMPT, coder_agent)
        complaint: str = context_window_complaint(narrower, overhead)
        if complaint:
            await self._session._events.emit(WarningEvent(content=complaint))
            return False

        self._session._coder_context_window = narrower
        self._session._planner_context_window = min(
            narrower, resolve_context_window(self._session._planner_context_window)
        )
        await self._session._events.emit(WarningEvent(content=warning_content(configured, narrower)))
        return True

    async def narrow_after_overflow(self, budget: int) -> bool:
        """
        React to a run that had no room to answer in.

        :param budget: The window the failed run was budgeted against.
        :returns: True if the window was narrowed and the run is worth retrying.
        """
        return await self.narrow_context_window(
            budget,
            lambda configured, narrower: (
                f'The model had no room to reply, so its context window is smaller than the '
                f'{configured:,} tokens configured. Retrying with {narrower:,}. Set the context '
                'window to what the model actually supports to avoid this.'
            ),
        )

    async def narrow_after_input_truncation(
        self,
        budget: int,
        estimated_input_tokens: int,
        provider_input_tokens: int,
    ) -> bool:
        """
        Reduce the working window after a provider reports truncated input.

        :param budget: The active coder window before the correction.
        :param estimated_input_tokens: Loom's estimate for the truncated request.
        :param provider_input_tokens: Tokens the provider reports processing.
        :returns: True if a smaller working window was applied.
        """
        return await self.narrow_context_window(
            budget,
            lambda configured, narrower: (
                f'The model/server processed {provider_input_tokens:,} prompt tokens, but Loom '
                f'estimated {estimated_input_tokens:,}. It appears to be truncating input to fit '
                f'its context limit, so Loom reduced its context window from {configured:,} to '
                f'{narrower:,}. Set the context window to what the model actually supports to '
                'avoid this.'
            ),
        )

    async def summarize_after_overflow(
        self,
        history: RequestHistoryCapture,
        budget: int,
        run_path: CompactionRunPath,
    ) -> list[ModelMessage] | None:
        """
        Force one summary of a failed request before reducing its window.

        :param history: The last compacted request history from the failed run.
        :param budget: The unchanged window against which that request ran.
        :param run_path: The agent path whose request is being recovered.
        :returns: A smaller history worth retrying, or None when narrowing is needed.
        """
        messages: list[ModelMessage] = history.messages
        if not messages:
            return None

        before: int = estimate_token_count(messages)
        try:
            compacted: list[ModelMessage] = await force_summarize_history(
                messages,
                budget,
                resolve_model(self._session._coder_model),
                allow_subminimum=self._session._measured_context_window is not None,
            )
        except Exception:
            logger.exception('Could not summarize the request history after a context overflow.')
            await self._session._events.emit(
                WarningEvent(
                    content=(
                        'The model had no room to reply. Loom tried summarizing the preceding request, '
                        'but the summary could not be created before reducing the context window.'
                    )
                )
            )
            return None

        after: int = estimate_token_count(compacted)
        if after >= before:
            await self._session._events.emit(
                WarningEvent(
                    content=(
                        'The model had no room to reply. Loom tried summarizing the preceding request, '
                        'but it did not free enough context to retry before reducing the context window.'
                    )
                )
            )
            return None

        await self._session._report_compaction(run_path, before, after)
        await self._session._events.emit(
            WarningEvent(
                content=(
                    'The model had no room to reply. Loom summarized the preceding request and will '
                    'retry before reducing the context window.'
                )
            )
        )
        return compacted

    async def run_with_overflow_recovery(
        self,
        run: Callable[[list[ModelMessage] | None], Awaitable[_RunResult]],
        context_window: Callable[[], int],
        history: RequestHistoryCapture,
        run_path: CompactionRunPath,
        on_retry: Callable[[], object] | None = None,
    ) -> _RunResult:
        """
        Retry a long-running model call after a recoverable context overflow.

        :param run: Starts one model-run attempt with optional recovered history.
        :param context_window: Reads the budget that owns each attempt.
        :param history: Captures the final compacted request before a failed call.
        :param run_path: Identifies recovery compaction in the event feed.
        :param on_retry: Clears state belonging to an abandoned attempt.
        :returns: The successful model-run result.
        :raises Exception: If the failure is unrelated or cannot be recovered.
        """
        message_history: list[ModelMessage] | None = None
        summary_attempted_window: int | None = None
        while True:
            try:
                return await run(message_history)
            except Exception as error:
                if not looks_like_context_overflow(error):
                    raise
                budget: int = context_window()
                if summary_attempted_window != budget:
                    summary_attempted_window = budget
                    message_history = await self.summarize_after_overflow(history, budget, run_path)
                    if message_history is not None:
                        if on_retry is not None:
                            _ = on_retry()
                        continue
                if not await self.narrow_after_overflow(budget):
                    raise
                if on_retry is not None:
                    _ = on_retry()
