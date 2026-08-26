from __future__ import annotations

from typing import Any

from pydantic_ai.messages import ToolCallPart
from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    DeduplicateFileReads,
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
    Build the compaction pipeline for one agent run.

    `target_tokens` is derived here rather than left to `target_fraction`,
    because the fraction is resolved against whatever window the `genai-prices`
    registry reports for the model. A locally-hosted Ollama model is not in that
    registry, and its identifier may well name someone else's deployment, so the
    resolved window would be a guess. The library's `context_window` override
    does not help either: it is consulted only alongside `target_fraction`.
    Passing an absolute `target_tokens` takes the registry out of the path
    entirely.

    Tiers run cheap-to-expensive and stop as soon as the history fits.
    `SummarizingCompaction` is deliberately absent: it costs a real model call,
    which is the one thing compaction is supposed to save.

    :param context_window: The configured window in tokens, or None for the default.
    :returns: The capability to pass to a single `Agent.run()`.
    """
    window: int = resolve_context_window(context_window)
    target: int = compaction_target(window)

    return TieredCompaction(
        tiers=[
            # Cheapest first: a superseded file read is redundant rather than
            # merely old, so dropping it costs the agent nothing at all.
            DeduplicateFileReads(file_key=read_file_key),
            # Then trim anything individually enormous, which may be enough on
            # its own when one large result caused the overrun.
            ClampOversizedMessages(
                max_part_tokens=max(MINIMUM_CONTEXT_WINDOW // 2, window // OVERSIZED_PART_DIVISOR),
                keep_head_chars=CLAMP_KEEP_HEAD_CHARS,
                keep_tail_chars=CLAMP_KEEP_TAIL_CHARS,
            ),
            # Last, and most lossy: blank old tool results outright. The
            # `max_tokens` trigger is never consulted, because `TieredCompaction`
            # drives its tiers directly and decides when to stop; the constructor
            # requires one regardless, so it is set to the same budget rather
            # than to a number that would contradict it if it ever were read.
            ClearToolResults(
                max_tokens=target,
                keep_pairs=KEEP_RECENT_TOOL_PAIRS,
                exclude_tools=IRRECOVERABLE_TOOLS,
            ),
        ],
        target_tokens=target,
    )
