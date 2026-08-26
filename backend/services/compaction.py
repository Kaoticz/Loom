from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic_ai.messages import ToolCallPart
from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    ContextUsage,
    DeduplicateFileReads,
    ReportContextUsage,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
)

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

KEEP_RECENT_TOOL_PAIRS: int = 5
"""
How many recent tool call/result pairs `ClearToolResults` leaves intact.

A coder working a step reads a file, edits it, runs the tests, reads the
failure and edits again. Five pairs keeps that cycle whole, so the results the
agent is actively reasoning about are never the ones blanked.
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


def build_compaction(context_window: int | None) -> TieredCompaction[Any]:
    """
    Build the compaction ladder for one agent run.

    Ordered cheapest to most expensive, stopping as soon as the history fits:

    1. `DeduplicateFileReads` — a superseded read is redundant rather than old,
       so dropping it costs nothing at all.
    2. `ClampOversizedMessages` — one runaway result can cause the overrun by
       itself, and trimming it alone may be enough.
    3. `ClearToolResults` — tool output dominates an agent's history, and the
       agent can re-run a tool if it needs the data again. This is the tier that
       does the real work in a long ReAct loop.
    4. `SummarizingCompaction` — compresses rather than discards, at the cost of
       a real model call, so it sits behind the free tiers.
    5. `SlidingWindowCompaction` — a hard floor. If summarizing still cannot
       bring the history under target, the oldest turns go, with the first user
       message kept so the task itself is never lost.

    The budget is expressed as a fraction of the window rather than an absolute
    token count, and the window is stated explicitly. Stating it matters: left
    to resolve itself, the fraction is taken of whatever a model registry
    reports, and a locally-hosted model is not in that registry.

    :param context_window: The configured window in tokens, or None for the default.
    :returns: The capability to pass to a single `Agent.run()`.
    """
    window: int = resolve_context_window(context_window)

    return TieredCompaction(
        tiers=[
            DeduplicateFileReads(file_key=read_file_key),
            ClampOversizedMessages(
                max_part_tokens=max(MINIMUM_CONTEXT_WINDOW // 2, window // OVERSIZED_PART_DIVISOR),
                keep_head_chars=CLAMP_KEEP_HEAD_CHARS,
                keep_tail_chars=CLAMP_KEEP_TAIL_CHARS,
            ),
            ClearToolResults(
                max_fraction=TARGET_FRACTION,
                context_window=window,
                keep_pairs=KEEP_RECENT_TOOL_PAIRS,
                exclude_tools=IRRECOVERABLE_TOOLS,
            ),
            SummarizingCompaction(
                max_fraction=TARGET_FRACTION,
                context_window=window,
                keep_messages=KEEP_RECENT_MESSAGES,
                # Inherits the running agent's model, so a summary is written by
                # whatever the user configured rather than by a second choice.
                model=None,
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


def build_usage_monitor(
    context_window: int | None,
    on_usage: Callable[[ContextUsage], Awaitable[None]],
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

    :param context_window: The configured window in tokens, or None for the default.
    :param on_usage: Called with each reading.
    :returns: The monitoring capability.
    """
    return ReportContextUsage(
        on_usage=on_usage,
        context_window=resolve_context_window(context_window),
    )
