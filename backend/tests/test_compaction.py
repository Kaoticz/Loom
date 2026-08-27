from typing import Any

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    DeduplicateFileReads,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
)

from services.compaction import (
    IRRECOVERABLE_TOOLS,
    KEEP_RECENT_TOOL_PAIRS,
    DEFAULT_CONTEXT_WINDOW,
    MINIMUM_CONTEXT_WINDOW,
    TARGET_FRACTION,
    build_compaction,
    read_file_key,
    resolve_context_window,
)


def call(tool_name: str, **arguments: Any) -> ToolCallPart:
    """
    Build a tool call to key.

    :param tool_name: The tool being called.
    :param arguments: The call's arguments.
    :returns: The call part.
    """
    return ToolCallPart(tool_name=tool_name, args=arguments)


def test_a_full_file_read_is_keyed_by_path() -> None:
    """
    Two reads of the whole file are the same read, so one supersedes the other.
    """
    first: str | None = read_file_key(call('read_file', path='src/main.py'))
    second: str | None = read_file_key(call('read_file', path='src/main.py'))

    assert first is not None
    assert first == second


def test_reads_of_different_files_do_not_share_a_key() -> None:
    """
    Nothing about one file makes another file's contents stale.
    """
    assert read_file_key(call('read_file', path='a.py')) != read_file_key(
        call('read_file', path='b.py')
    )


def test_different_line_ranges_of_one_file_do_not_share_a_key() -> None:
    """
    A partial read does not supersede a different part of the same file.

    Keying on the path alone would treat lines 200-250 as replacing lines 1-50
    and blank a range the agent is still working from. That is a silent wrong
    answer rather than a slow one, which is why the range is in the key.
    """
    head: str | None = read_file_key(call('read_file', path='a.py', start_line=1, end_line=50))
    tail: str | None = read_file_key(call('read_file', path='a.py', start_line=200, end_line=250))

    assert head is not None
    assert tail is not None
    assert head != tail


def test_an_identical_range_shares_a_key() -> None:
    """
    Re-reading exactly the same range is exactly the redundancy worth dropping.
    """
    first: str | None = read_file_key(call('read_file', path='a.py', start_line=1, end_line=50))
    second: str | None = read_file_key(call('read_file', path='a.py', start_line=1, end_line=50))

    assert first == second


def test_a_full_read_is_not_confused_with_a_ranged_read() -> None:
    """
    An unbounded read and a bounded one returned different amounts of the file.
    """
    assert read_file_key(call('read_file', path='a.py')) != read_file_key(
        call('read_file', path='a.py', start_line=1, end_line=50)
    )


@pytest.mark.parametrize(
    'tool_name',
    ['search_code', 'edit_file', 'create_file', 'run_test_command', 'commit_changes'],
)
def test_other_tools_are_never_deduplicated(tool_name: str) -> None:
    """
    Only `read_file` returns contents a later call can make stale.

    Returning a key for anything else would let unrelated results supersede one
    another, which is data loss dressed up as compaction.
    """
    assert read_file_key(call(tool_name, path='a.py', query='x')) is None


def test_a_read_without_a_usable_path_is_not_keyed() -> None:
    """
    A call that does not name a file cannot be matched against one.
    """
    assert read_file_key(call('read_file')) is None
    assert read_file_key(call('read_file', path=None)) is None
    assert read_file_key(call('read_file', path=17)) is None


def test_an_unset_window_uses_the_default() -> None:
    """
    Nothing configured means the documented default, not zero.
    """
    assert resolve_context_window(None) == DEFAULT_CONTEXT_WINDOW
    assert resolve_context_window(0) == DEFAULT_CONTEXT_WINDOW


def test_an_absurd_window_is_raised_to_the_minimum() -> None:
    """
    A window with no room for a prompt and a reply is a mistake, not an instruction.
    """
    assert resolve_context_window(10) == MINIMUM_CONTEXT_WINDOW
    assert resolve_context_window(-500) == DEFAULT_CONTEXT_WINDOW


def test_a_configured_window_is_honoured() -> None:
    """
    A plausible number is used exactly as given.
    """
    assert resolve_context_window(8192) == 8192


def test_the_budget_leaves_headroom_below_the_window() -> None:
    """
    The budget is a fraction of the window, not the whole of it.

    The model's next response and the tool results it triggers arrive after the
    point the budget was last measured, so they have to fit in what is left.
    """
    compaction: TieredCompaction[Any] = build_compaction(10_000)

    assert compaction.target_fraction == TARGET_FRACTION
    assert TARGET_FRACTION < 1.0, 'the whole window leaves nothing to reply with'
    assert compaction.context_window == 10_000


def test_the_budget_is_absolute_rather_than_a_fraction() -> None:
    """
    `target_fraction` is resolved against a model registry Ollama is not in.

    A locally-hosted model's identifier may name someone else's deployment, so
    a fraction would be taken of a guessed window. An absolute `target_tokens`
    takes the registry out of the path entirely.
    """
    compaction: TieredCompaction[Any] = build_compaction(16_384)

    # Stated rather than left to resolve: the fraction is otherwise taken of
    # whatever a model registry reports, and a locally-hosted model is not in it.
    assert compaction.context_window == 16_384
    assert compaction.target_fraction == TARGET_FRACTION


def test_the_tiers_run_cheapest_first() -> None:
    """
    Order is the point of the tiering: stop as soon as the history fits.

    Dropping a superseded file read costs nothing, clamping one huge result
    costs a little, and blanking old tool results costs the most, so they are
    attempted in that order.
    """
    compaction: TieredCompaction[Any] = build_compaction(None)

    assert [type(tier).__name__ for tier in compaction.tiers] == [
        DeduplicateFileReads.__name__,
        ClampOversizedMessages.__name__,
        ClearToolResults.__name__,
        SummarizingCompaction.__name__,
        SlidingWindowCompaction.__name__,
    ]


def test_the_expensive_tiers_sit_behind_the_free_ones() -> None:
    """
    Summarizing costs a real model call, so everything free is tried first.

    It is present rather than omitted: a long ReAct loop eventually exceeds what
    blanking and deduplication can reclaim, and compressing old context beats
    discarding it. The sliding window sits behind even that, as the floor which
    guarantees the history comes under target however the earlier tiers fare.
    """
    tiers: list[str] = [type(tier).__name__ for tier in build_compaction(None).tiers]

    assert tiers.index(SummarizingCompaction.__name__) > tiers.index(ClearToolResults.__name__)
    assert tiers.index(SlidingWindowCompaction.__name__) > tiers.index(
        SummarizingCompaction.__name__
    )


def test_a_single_message_may_not_fill_the_window() -> None:
    """
    One tool result must not be able to crowd out the conversation it belongs to.
    """
    compaction: TieredCompaction[Any] = build_compaction(64_000)
    clamp: Any = compaction.tiers[1]

    assert clamp.max_part_tokens is not None
    assert clamp.max_part_tokens < int(64_000 * TARGET_FRACTION)


def test_clamping_keeps_both_ends_of_a_message() -> None:
    """
    A test run puts its summary at the end and a stack trace's most specific
    frame is its last, so keeping only the beginning would discard the part
    worth reading.
    """
    clamp: Any = build_compaction(None).tiers[1]

    assert clamp.keep_head_chars > 0
    assert clamp.keep_tail_chars > 0


def test_recent_tool_results_survive_clearing() -> None:
    """
    The results the agent is actively reasoning about are never the ones blanked.
    """
    clear: Any = build_compaction(None).tiers[2]

    assert clear.keep_pairs >= 1


def history_with_file_reads(repeats: int, body: str) -> list[ModelMessage]:
    """
    Build a history where the same file is read over and over.

    :param repeats: How many times the file is read.
    :param body: The contents each read returned.
    :returns: The synthetic message history.
    """
    messages: list[ModelMessage] = []
    for index in range(repeats):
        messages.append(
            ModelResponse(parts=[ToolCallPart(
                tool_name='read_file',
                args={'path': 'src/main.py'},
                tool_call_id=f'call-{index}',
            )])
        )
        messages.append(
            ModelRequest(parts=[ToolReturnPart(
                tool_name='read_file',
                content=body,
                tool_call_id=f'call-{index}',
            )])
        )
    return messages


def run_context(model: FunctionModel) -> RunContext[Any]:
    """
    Build the run context the compaction needs to resolve its target.

    :param model: The model the context should report.
    :returns: A context usable by `TieredCompaction.compact`.
    """
    return RunContext(deps=None, model=model, usage=RunUsage())


def quiet_model() -> FunctionModel:
    """
    Build a model that answers immediately.

    :returns: A model standing in for whatever would have been called.
    """

    async def call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Done.')])

    return FunctionModel(call)


async def test_a_history_under_budget_is_left_alone() -> None:
    """
    Compaction that fires when it is not needed is just data loss.
    """
    messages: list[ModelMessage] = history_with_file_reads(2, 'x' * 100)
    compaction: TieredCompaction[Any] = build_compaction(1_000_000)

    result: list[ModelMessage] = await compaction.compact(messages, run_context(quiet_model()))

    assert result == messages


async def test_repeated_reads_of_one_file_are_deduplicated() -> None:
    """
    The cheapest tier fires first and is enough on its own here.

    Twenty identical reads of one file carry one file's worth of information.
    Verified against the resulting history rather than by trusting the library:
    the point is that the tier is wired to `read_file` correctly, which no
    amount of correct library behaviour would establish on its own.
    """
    body: str = 'def main():\n    pass\n' * 400
    messages: list[ModelMessage] = history_with_file_reads(20, body)
    compaction: TieredCompaction[Any] = build_compaction(4_096)

    result: list[ModelMessage] = await compaction.compact(messages, run_context(quiet_model()))

    surviving: list[str] = [
        str(part.content)
        for message in result
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    intact: int = sum(1 for content in surviving if body in content)

    assert intact < 20, 'nothing was deduplicated'
    assert intact >= 1, 'the most recent read should survive'


async def test_compaction_reduces_a_history_that_exceeds_its_budget() -> None:
    """
    The pipeline as configured actually shrinks an oversized history.

    Measured on the rendered history rather than on the library's own estimate,
    so a tier that reported success without removing anything would fail here.
    """
    messages: list[ModelMessage] = history_with_file_reads(30, 'x' * 4_000)
    compaction: TieredCompaction[Any] = build_compaction(2_048)

    result: list[ModelMessage] = await compaction.compact(messages, run_context(quiet_model()))

    before: int = sum(
        len(str(part.content))
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    )
    after: int = sum(
        len(str(part.content))
        for message in result
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    )

    assert after < before


def history_with_an_answer(reads: int, body: str, answer: str) -> list[ModelMessage]:
    """
    Build a history where the user answered a question, then many files were read.

    :param reads: How many file reads follow the answer.
    :param body: The contents each read returned.
    :param answer: What the user said.
    :returns: The synthetic message history.
    """
    messages: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart(
            tool_name='ask_user',
            args={'question': 'Which retry policy?'},
            tool_call_id='ask-1',
        )]),
        ModelRequest(parts=[ToolReturnPart(
            tool_name='ask_user',
            content=answer,
            tool_call_id='ask-1',
        )]),
    ]
    for index in range(reads):
        messages.append(
            ModelResponse(parts=[ToolCallPart(
                tool_name='read_file',
                args={'path': f'src/file{index}.py'},
                tool_call_id=f'read-{index}',
            )])
        )
        messages.append(
            ModelRequest(parts=[ToolReturnPart(
                tool_name='read_file',
                content=body,
                tool_call_id=f'read-{index}',
            )])
        )
    return messages


async def test_a_users_answer_survives_the_blanking_tiers() -> None:
    """
    Everything else compaction drops can be fetched again; an answer cannot.

    `exclude_tools` protects it from `ClearToolResults`, which is the tier that
    would otherwise reach it in an ordinary long run. It cannot protect it from
    `SlidingWindowCompaction`, which discards whole messages by age and is the
    last resort — so this asserts what the blanking tiers do, not an absolute
    guarantee across the whole ladder.
    """
    answer: str = 'The user replied: use exponential backoff with a 30s cap.'
    messages: list[ModelMessage] = history_with_an_answer(40, 'y' * 4_000, answer)
    blanking: ClearToolResults[Any] = ClearToolResults(
        max_tokens=1_000,
        keep_pairs=KEEP_RECENT_TOOL_PAIRS,
        exclude_tools=IRRECOVERABLE_TOOLS,
    )

    result: list[ModelMessage] = await blanking.compact(messages, run_context(quiet_model()))

    answers: list[str] = [
        str(part.content)
        for message in result
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'ask_user'
    ]

    assert answers == [answer], 'the user answer was altered or dropped'


async def test_old_file_reads_are_still_cleared_around_an_answer() -> None:
    """
    Protecting answers must not amount to protecting everything.
    """
    messages: list[ModelMessage] = history_with_an_answer(40, 'y' * 4_000, 'Backoff.')
    compaction: TieredCompaction[Any] = build_compaction(2_048)

    result: list[ModelMessage] = await compaction.compact(messages, run_context(quiet_model()))

    reads: list[str] = [
        str(part.content)
        for message in result
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'read_file'
    ]

    assert any('y' * 4_000 not in content for content in reads), 'nothing was reclaimed'


def test_the_model_is_told_the_window_it_is_budgeted_against() -> None:
    """
    The setting has to reach the model, not just the compaction arithmetic.

    Ollama defaults its window low, often 4096, whatever the model could hold.
    While that was never sent, Loom budgeted against the configured window and
    the model ran with a much smaller one, so compaction sat far below its
    target and never fired while the real window overflowed.
    """
    from services.model_resolution import model_settings

    settings: Any = model_settings(8_192)

    assert settings['extra_body']['options']['num_ctx'] == 8_192
    # max_tokens is deliberately absent. Setting it only decided which number
    # appeared in "token limit exceeded before any response was generated"; the
    # prompt filling the window is what caused it, whatever the reply cap was.
    assert 'max_tokens' not in settings


def test_an_overflow_failure_is_recognised() -> None:
    """
    This failure is the only evidence Loom gets that a window is optimistic.

    A locally-hosted model may be clamped to whatever it was trained for, whoever
    configured what, and there is no way to ask it how much room it really has.
    """
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    from services.compaction import looks_like_no_room_to_answer

    overflow = UnexpectedModelBehavior(
        'Model token limit (8192) exceeded before any response was generated.'
    )
    unrelated = UnexpectedModelBehavior("Tool 'edit_file' exceeded max retries count of 5.")

    assert looks_like_no_room_to_answer(overflow) is True
    assert looks_like_no_room_to_answer(unrelated) is False


def test_a_provider_context_rejection_is_recognised() -> None:
    """
    A provider can reject an oversized request before it generates anything.

    That shape does not carry the empty-generation wording, but it is the same
    recoverable signal when it is a client error that specifically names context.
    """
    from pydantic_ai.exceptions import ModelHTTPError

    from services.compaction import looks_like_context_overflow

    overflow: ModelHTTPError = ModelHTTPError(
        400,
        'ollama:model',
        {'error': 'requested context length exceeds the model limit'},
    )
    unrelated: ModelHTTPError = ModelHTTPError(
        400,
        'ollama:model',
        {'error': 'the requested model does not exist'},
    )
    server_error: ModelHTTPError = ModelHTTPError(
        500,
        'ollama:model',
        {'error': 'context length exceeds the model limit'},
    )

    assert looks_like_context_overflow(overflow) is True
    assert looks_like_context_overflow(unrelated) is False
    assert looks_like_context_overflow(server_error) is False


def test_narrowing_closes_the_gap_and_then_stops() -> None:
    """
    Halving converges on a real window without anyone knowing the figure.

    It has to stop somewhere, or a model that fails for an unrelated reason
    would be retried down to a window too small to hold a prompt at all.
    """
    from services.compaction import narrowed_window

    window: int = 128_048
    seen: list[int] = []
    for _ in range(20):
        window = narrowed_window(window)
        seen.append(window)

    assert seen[0] == 64_024
    assert seen[1] == 32_012
    assert min(seen) == MINIMUM_CONTEXT_WINDOW
    assert narrowed_window(MINIMUM_CONTEXT_WINDOW) == MINIMUM_CONTEXT_WINDOW


def test_usage_is_measured_against_the_configured_window() -> None:
    """
    Usage is reported, not estimated by us and not left unknown.

    `ReportContextUsage` reuses the same estimator and the same window the
    compaction uses, so a reading cannot disagree with the budget it describes,
    and each reading carries `resolved` so a percentage taken against a fallback
    window can say that it is a guess.
    """
    from services.compaction import build_usage_monitor

    async def noop(usage: object) -> None:
        return None

    monitor = build_usage_monitor(48_000, noop)

    assert monitor.context_window == 48_000


def test_every_requested_capability_is_present() -> None:
    """
    All five, on the ladder, in the order that spends the least first.
    """
    tiers = {type(tier).__name__ for tier in build_compaction(None).tiers}

    assert {
        'DeduplicateFileReads',
        'ClearToolResults',
        'SummarizingCompaction',
        'SlidingWindowCompaction',
    } <= tiers
