from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from logging import Logger, getLogger
from typing import TYPE_CHECKING, Any, overload, override

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ContextUsage,
    DeduplicateFileReads,
    ReportContextUsage,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
    compact_now,
    estimate_token_count,
)
from pydantic_ai.exceptions import ModelHTTPError

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import RunContext


logger: Logger = getLogger(__name__)

DEFAULT_CONTEXT_WINDOW: int = 32_768
"""
The context window assumed when the user configures none.

Deliberately far below the library's own 200,000-token default, which describes
a large cloud model. Loom runs against locally-hosted models, where a window
that size is the exception. Assuming too much is the damaging direction: the
budget is never reached, nothing is ever compacted, and the run fails on a
context overflow the compaction was added to prevent. Assuming too little only
compacts sooner than strictly necessary.
"""

MINIMUM_CONTEXT_WINDOW: int = 2_048
"""
The smallest window worth honouring.

Below this there is no room for a system prompt, a task and a reply, so a
smaller number is treated as a mistake rather than obeyed into uselessness.
"""

MAX_REPLY_TOKENS: int = 8_192
"""
The longest single reply the model may generate.

A cap, not a share of the window. Headroom scales with the window; the length of
one answer does not — a bigger window does not make a coder's reply longer, it
makes room for more history. Deriving this from the window instead gave a
128,000-token window a 38,000-token reply budget, which is both meaningless as
an answer and larger than the space actually left to generate into, so the model
returned nothing and the run failed with "token limit exceeded before any
response was generated".

Generous enough for the longest legitimate reply, which is a `create_file` call
carrying a whole file, and still small beside any window worth configuring.
"""

TARGET_FRACTION: float = 0.7
"""
How much of the window the conversation may occupy before compaction runs.

The remaining 30% is not spare: the model's next response and the tool results
it triggers have to fit alongside the history, and both arrive after the point
where the budget was last measured.
"""

COMPACTION_REPORT_MINIMUM_RECLAIM_TOKENS: int = 1
"""
The smallest estimated reduction that counts as a compaction notice.

A ladder may run because its trigger fired yet leave the messages intact. Only a
history that became smaller is useful to announce, and one token is enough to
show that it did.
"""

IRRECOVERABLE_TOOLS: frozenset[str] = frozenset({'ask_user'})
"""
Tools whose results must never be blanked, however old.

Everything else compaction discards can be obtained again: a file can be re-read
and a search re-run, so dropping the result costs time at worst. A user's answer
cannot. Blanking it leaves the agent having stopped the task to ask a question,
received an answer, and then forgotten it — so it either asks again, stopping the
task a second time, or guesses at something it was explicitly told.
"""

KEEP_RECENT_MESSAGES: int = 20
"""
How many recent messages the summarizing and sliding tiers keep verbatim.

The tail is where the agent is working. Summarising or discarding it would take
away the step it is part way through, which is the one part of the history it
cannot do without.
"""

CLAMP_KEEP_HEAD_CHARS: int = 2_000
CLAMP_KEEP_TAIL_CHARS: int = 2_000
"""
How much of an oversized message survives clamping, from each end.

Both ends rather than just the head: a test run puts its summary at the end and
a stack trace's most specific frame is its last, so keeping only the beginning
would discard the part worth reading.
"""

OVERSIZED_PART_DIVISOR: int = 10
"""
The share of the window a single message may occupy before it is clamped.

One tool result should not be able to crowd out the conversation it belongs to.
`ToolReport.succeed()` applies no size limit of its own, so a single `read_file`
on a large file would otherwise arrive whole.
"""


def read_file_key(call: ToolCallPart) -> str | None:
    """
    Identify the file a `read_file` call returned, for deduplication.

    Keyed on the line range as well as the path. `read_file` takes optional
    `start_line` and `end_line`, so two calls naming one file may have returned
    different parts of it. Keying on the path alone would treat a read of lines
    1-50 as superseded by a read of lines 200-250 and blank a range the agent is
    still working from, which is a silent wrong answer rather than a slow one.

    Every other tool returns None, since only `read_file` returns file contents
    that a later call can make stale.

    :param call: The tool call to identify.
    :returns: A stable key for the range read, or None if this is not a file read.
    """
    if call.tool_name != 'read_file':
        return None

    try:
        arguments: dict[str, Any] = call.args_as_dict()
    except Exception:
        # An unparsable call is not a file read anyone can supersede safely.
        return None

    path: object = arguments.get('path')
    if not isinstance(path, str):
        return None

    return f'{path}:{arguments.get("start_line")}-{arguments.get("end_line")}'


def _strip_redundant_system_prompts(messages: list[ModelMessage]) -> list[ModelMessage]:
    """
    Remove system prompts from non-leading request messages.

    Compaction creates a leading summary request that already contains the
    original system prompts. A preserved user request can retain another copy,
    which strict chat templates reject after the leading system message. Leave
    system-only requests unchanged because removing their sole part would create
    an empty message that disappears during provider mapping.

    :param messages: The compacted history to normalize.
    :returns: A copied history with redundant non-leading system prompts removed.
    """
    normalized: list[ModelMessage] = []
    for index, message in enumerate(messages):
        if index == 0 or not isinstance(message, ModelRequest):
            normalized.append(message)
            continue

        parts: list[ModelRequestPart] = [
            part for part in message.parts if not isinstance(part, SystemPromptPart)
        ]
        if not parts:
            normalized.append(message)
            continue
        normalized.append(replace(message, parts=parts))
    return normalized


def compaction_target(window: int) -> int:
    """
    How much of the window the conversation may occupy.

    Whichever is tighter: the fixed share, or the window less the room one reply
    needs. The share governs a large window, the reply allowance a small one,
    where a percentage leaves too little to answer in.

    :param window: The resolved context window in tokens.
    :returns: The budget in tokens, always leaving room for a reply.
    """
    return min(int(window * TARGET_FRACTION), max(window - MAX_REPLY_TOKENS, window // 4))


def response_allowance(context_window: int | None) -> int:
    """
    How many tokens to let the model generate for one reply.

    Passed to the provider as `max_tokens`, so that running out of room is a
    bounded reply rather than a failed run. Never more than the space compaction
    actually leaves, so the two cannot disagree, and never more than one reply
    could sensibly need however large the window is.

    :param context_window: The configured window in tokens, or None for the default.
    :returns: The reply budget in tokens.
    """
    window: int = resolve_context_window(context_window)
    return min(MAX_REPLY_TOKENS, window - compaction_target(window))


def looks_like_no_room_to_answer(error: Exception) -> bool:
    """
    Recognise a run that ended because the prompt filled the model's window.

    The model produced nothing and stopped for length, which means the context
    was already full before generation began. Loom cannot ask a model how large
    its window really is — a locally-hosted one may be clamped to whatever it was
    trained for, whatever the server permits — so this failure is the only
    reliable evidence that the configured window is optimistic.

    :param error: The exception that ended the run.
    :returns: True if the prompt left no room for a reply.
    """
    message: str = str(error).lower()
    return 'token limit' in message and 'before any response' in message


def looks_like_context_overflow(error: Exception) -> bool:
    """
    Recognize an empty generation or provider rejection caused by a full context.

    Pydantic AI wraps an empty length-finished generation in
    ``UnexpectedModelBehavior``. Providers that reject an oversized request
    before generation instead surface a 4xx ``ModelHTTPError``. Both failures
    prove the working window is too optimistic, while other model failures must
    retain their existing handling.

    :param error: The exception that ended the model run.
    :returns: True when narrowing the context window may let a retry succeed.
    """
    if looks_like_no_room_to_answer(error):
        return True
    if not isinstance(error, ModelHTTPError) or not 400 <= error.status_code < 500:
        return False

    message: str = str(error).lower()
    context_markers: tuple[str, ...] = (
        'context length',
        'context window',
        'maximum context',
        'token limit',
    )
    return any(marker in message for marker in context_markers)


def narrowed_window(context_window: int | None) -> int:
    """
    Halve the working window after a run found it too large.

    Halving rather than stepping down: the gap between a configured window and a
    real one is usually a factor, not a margin — 128k configured against a 32k
    model — so a small decrement would take many failed runs to close.

    :param context_window: The window that proved too large.
    :returns: A smaller window, never below the minimum worth honouring.
    """
    return max(resolve_context_window(context_window) // 2, MINIMUM_CONTEXT_WINDOW)


def resolve_context_window(configured: int | None) -> int:
    """
    Settle on the context window to budget against.

    :param configured: The user's setting, or None if they left it unset.
    :returns: The window in tokens, never below `MINIMUM_CONTEXT_WINDOW`.
    """
    if configured is None or configured <= 0:
        return DEFAULT_CONTEXT_WINDOW
    return max(configured, MINIMUM_CONTEXT_WINDOW)


class ObservedCompaction(AbstractCapability[Any]):
    """
    Observe a compaction capability without changing its decisions.

    :param inner: The compaction capability that may rewrite request messages.
    :param on_compaction: Receives the estimated token counts before and after a rewrite.
    """

    def __init__(
        self,
        inner: AbstractCapability[Any],
        on_compaction: Callable[[int, int], Awaitable[None]],
    ) -> None:
        """
        Create a compaction observer.

        :param inner: The capability that performs compaction.
        :param on_compaction: Receives counts when the capability shortened a request.
        """
        super().__init__()
        self._inner: AbstractCapability[Any] = inner
        self._on_compaction: Callable[[int, int], Awaitable[None]] = on_compaction

    @override
    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """
        Delegate compaction and report a genuine reduction when one occurred.

        Observer failures are deliberately contained: a missing notice must not
        prevent the model request it was meant to describe.

        :param ctx: The active model run context.
        :param request_context: The model request that may be compacted.
        :returns: The request context returned by the wrapped capability.
        """
        try:
            before: int = estimate_token_count(request_context.messages)
        except Exception:
            logger.exception('Could not measure request tokens before compaction.')
            return await self._inner.before_model_request(ctx, request_context)

        compacted: ModelRequestContext = await self._inner.before_model_request(ctx, request_context)
        try:
            after: int = estimate_token_count(compacted.messages)
            if before - after < COMPACTION_REPORT_MINIMUM_RECLAIM_TOKENS:
                return compacted
            await self._on_compaction(before, after)
        except Exception:
            logger.exception('Could not report a compaction event.')
        return compacted


class RequestHistoryCapture(AbstractCapability[Any]):
    """
    Retain the most recent compacted request history for recovery.

    The agent discards its graph state when a model request raises. Capturing the
    history after ordinary capabilities run lets the session retry from a forced
    summary instead of throwing that useful context away before narrowing.
    """

    def __init__(self) -> None:
        """
        Create an empty request-history capture.
        """
        super().__init__()
        self._messages: list[ModelMessage] = []

    @property
    def messages(self) -> list[ModelMessage]:
        """
        Return a copy of the last captured request history.

        :returns: The history prepared for the most recent model request.
        """
        return list(self._messages)

    @override
    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """
        Save the already-compacted history without changing the request.

        :param ctx: The active model run context, unused by this observer.
        :param request_context: The request whose prepared history is retained.
        :returns: The original request context, unchanged.
        """
        self._messages = list(request_context.messages)
        return request_context


@dataclass
class SummarizeAndStripToolMessages(AbstractCapability[Any]):
    """
    Summarize history and then remove replaceable tool messages in one tier.

    The enclosing `TieredCompaction` calls this tier only after the history is
    above its target. The tool-part strip lives inside `compact`, rather than as
    a later tier, so it always follows a summary even when that summary alone
    brings the history back below the target.

    :param summarizer: The strategy that summarizes the older portion of history.
    :param exclude_tools: Tool names whose call/result parts remain intact.
    """

    summarizer: SummarizingCompaction[Any]
    exclude_tools: frozenset[str]

    def with_focus(self, focus: str) -> SummarizeAndStripToolMessages:
        """
        Prioritize focus in the summary without changing stripping behavior.

        :param focus: The topic the summary must preserve.
        :returns: An equivalent combined tier whose summary receives the focus.
        """
        return replace(self, summarizer=self.summarizer.with_focus(focus))

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[Any],
    ) -> list[ModelMessage]:
        """
        Summarize history and unconditionally remove nonessential tool parts.

        :param messages: The history that exceeded the enclosing ladder's target.
        :param ctx: The running model context used by the summarizer.
        :returns: The summarized history with replaceable tool parts removed.
        """
        summarized: list[ModelMessage] = await self.summarizer.compact(messages, ctx)
        stripped: list[ModelMessage] = []
        for message in summarized:
            if isinstance(message, ModelRequest):
                parts: list[ModelRequestPart] = [
                    part for part in message.parts if not self._is_replaceable_tool_part(part)
                ]
                if parts:
                    stripped.append(replace(message, parts=parts))
                continue

            response_parts: list[ModelResponsePart] = [
                part for part in message.parts if not self._is_replaceable_tool_part(part)
            ]
            if response_parts:
                stripped.append(replace(message, parts=response_parts))
        return _strip_redundant_system_prompts(stripped)

    def _is_replaceable_tool_part(self, part: ModelRequestPart | ModelResponsePart) -> bool:
        """
        Identify a tool part that can be removed from the working history.

        :param part: One request or response message part.
        :returns: True when the part belongs to a nonessential tool.
        """
        return (
            isinstance(part, (ToolCallPart, ToolReturnPart))
            and part.tool_name not in self.exclude_tools
        )


def build_forced_summarization(
    context_window: int | None,
    *,
    allow_subminimum: bool = False,
) -> SummarizingCompaction[Any]:
    """
    Build the summary strategy used immediately after a context overflow.

    `compact_now` drives this strategy directly, bypassing its normal budget
    trigger. The normal fraction remains configured so the strategy's shape
    matches ordinary summary compaction while its pass is deliberately forced.

    :param context_window: The effective window of the failed request.
    :param allow_subminimum: Preserve a positive model measurement below Loom's floor.
    :returns: A summarization strategy for one forced recovery attempt.
    """
    window: int = _window_for_consumer(context_window, allow_subminimum)
    return SummarizingCompaction(
        max_fraction=TARGET_FRACTION,
        context_window=window,
        keep_messages=KEEP_RECENT_MESSAGES,
        model=None,
    )


async def force_summarize_history(
    messages: list[ModelMessage],
    context_window: int | None,
    model: Any,
    *,
    allow_subminimum: bool = False,
) -> list[ModelMessage]:
    """
    Summarize a failed request history outside an active agent run.

    :param messages: The compacted history sent to the failed model request.
    :param context_window: The effective window of that failed request.
    :param model: The model that writes the recovery summary.
    :param allow_subminimum: Preserve a positive model measurement below Loom's floor.
    :returns: The summarized history, or the original history when no summary is possible.
    """
    strategy: SummarizingCompaction[Any] = build_forced_summarization(
        context_window, allow_subminimum=allow_subminimum
    )
    summarized: list[ModelMessage] = await compact_now(strategy, messages, model=model)
    return _strip_redundant_system_prompts(summarized)


@overload
def build_compaction(
    context_window: int | None,
    *,
    allow_subminimum: bool = False,
    on_compaction: None = None,
) -> TieredCompaction[Any]: ...


@overload
def build_compaction(
    context_window: int | None,
    *,
    allow_subminimum: bool = False,
    on_compaction: Callable[[int, int], Awaitable[None]],
) -> ObservedCompaction: ...


def build_compaction(
    context_window: int | None,
    *,
    allow_subminimum: bool = False,
    on_compaction: Callable[[int, int], Awaitable[None]] | None = None,
) -> TieredCompaction[Any] | ObservedCompaction:
    """
    Build the compaction ladder for one agent run.

    Ordered cheapest to most expensive, stopping as soon as the history fits:

    1. `DeduplicateFileReads` — a superseded read is redundant rather than old,
       so dropping it costs nothing at all.
    2. `ClampOversizedMessages` — one runaway result can cause the overrun by
       itself, and trimming it alone may be enough.
    3. `SummarizeAndStripToolMessages` — compresses older history with a real
       model call, then discards all replaceable tool calls and results in the
       same tier.
    4. `SlidingWindowCompaction` — a hard floor. If summarizing and stripping still cannot
       bring the history under target, the oldest turns go, with the first user
       message kept so the task itself is never lost.

    The budget is expressed as a fraction of the window rather than an absolute
    token count, and the window is stated explicitly. Stating it matters: left
    to resolve itself, the fraction is taken of whatever a model registry
    reports, and a locally-hosted model is not in that registry.

    :param context_window: The configured or measured window in tokens, or None for the default.
    :param allow_subminimum: Preserve a positive model measurement below Loom's usual floor.
    :param on_compaction: Receives estimates whenever the ladder shortens a request.
    :returns: The capability to pass to a single `Agent.run()`.
    """
    window: int = _window_for_consumer(context_window, allow_subminimum)

    ladder: TieredCompaction[Any] = TieredCompaction(
        tiers=[
            DeduplicateFileReads(file_key=read_file_key),
            ClampOversizedMessages(
                max_part_tokens=max(MINIMUM_CONTEXT_WINDOW // 2, window // OVERSIZED_PART_DIVISOR),
                keep_head_chars=CLAMP_KEEP_HEAD_CHARS,
                keep_tail_chars=CLAMP_KEEP_TAIL_CHARS,
            ),
            SummarizeAndStripToolMessages(
                summarizer=SummarizingCompaction(
                    max_fraction=TARGET_FRACTION,
                    context_window=window,
                    keep_messages=KEEP_RECENT_MESSAGES,
                    # Inherits the running agent's model, so a summary is written by
                    # whatever the user configured rather than by a second choice.
                    model=None,
                ),
                exclude_tools=IRRECOVERABLE_TOOLS,
            ),
            SlidingWindowCompaction(
                max_fraction=TARGET_FRACTION,
                context_window=window,
                keep_messages=KEEP_RECENT_MESSAGES,
                preserve_first_user_message=True,
            ),
        ],
        target_fraction=TARGET_FRACTION,
        context_window=window,
    )
    if on_compaction is None:
        return ladder
    return ObservedCompaction(ladder, on_compaction)


def build_usage_monitor(
    context_window: int | None,
    on_usage: Callable[[ContextUsage], Awaitable[None]],
    *,
    allow_subminimum: bool = False,
) -> ReportContextUsage:
    """
    Build the capability that reports how full the context actually is.

    Registered after the compaction ladder, so what it reports is the corrected
    history for the request about to be sent rather than what triggered the
    compaction.

    This is measurement rather than estimation. It reuses the same estimator and
    the same window the compaction uses, and its readings carry `resolved`,
    which is False when the window is a fallback rather than the model's own —
    so a reading that is a guess can say so instead of being mistaken for fact.

    :param context_window: The configured or measured window in tokens, or None for the default.
    :param on_usage: Called with each reading.
    :param allow_subminimum: Preserve a positive model measurement below Loom's usual floor.
    :returns: The monitoring capability.
    """
    return ReportContextUsage(
        on_usage=on_usage,
        context_window=_window_for_consumer(context_window, allow_subminimum),
    )


def _window_for_consumer(context_window: int | None, allow_subminimum: bool) -> int:
    """
    Resolve a requested window while preserving a known small model limit.

    :param context_window: The requested window, or None for the default.
    :param allow_subminimum: Whether a positive window may remain below Loom's floor.
    :returns: The window suitable for compaction or monitoring.
    """
    if allow_subminimum and context_window is not None and context_window > 0:
        return context_window
    return resolve_context_window(context_window)
