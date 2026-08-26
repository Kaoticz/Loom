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
    SummarizingCompaction,
    TieredCompaction,
)

from services.compaction import (
    DEFAULT_CONTEXT_WINDOW,
    MINIMUM_CONTEXT_WINDOW,
    TARGET_FRACTION,
    build_compaction,
    read_file_key,
    MAX_REPLY_TOKENS,
    response_allowance,
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

    assert compaction.target_tokens is not None
    assert compaction.target_tokens <= int(10_000 * TARGET_FRACTION)
    # The gap has to be big enough to answer in, or the model runs out of room
    # to generate and returns nothing at all.
    assert 10_000 - compaction.target_tokens >= response_allowance(10_000)


def test_the_budget_is_absolute_rather_than_a_fraction() -> None:
    """
    `target_fraction` is resolved against a model registry Ollama is not in.

    A locally-hosted model's identifier may name someone else's deployment, so
    a fraction would be taken of a guessed window. An absolute `target_tokens`
    takes the registry out of the path entirely.
    """
    compaction: TieredCompaction[Any] = build_compaction(16_384)

    assert compaction.target_tokens is not None
    assert compaction.target_fraction is None


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
    ]


def test_no_tier_calls_a_model() -> None:
    """
    `SummarizingCompaction` is deliberately absent.

    It costs a real model call, which is the expense compaction exists to
    avoid, and it would run against the same local model that is already
    struggling for room.
    """
    compaction: TieredCompaction[Any] = build_compaction(None)

    assert not any(isinstance(tier, SummarizingCompaction) for tier in compaction.tiers)


def test_a_single_message_may_not_fill_the_window() -> None:
    """
    One tool result must not be able to crowd out the conversation it belongs to.
    """
    compaction: TieredCompaction[Any] = build_compaction(64_000)
    clamp: Any = compaction.tiers[1]

    assert clamp.max_part_tokens is not None
    assert clamp.max_part_tokens < compaction.target_tokens


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


async def test_a_users_answer_is_never_cleared() -> None:
    """
    Everything else compaction drops can be fetched again; an answer cannot.

    Blanking it leaves the agent having stopped the task to ask, been told, and
    then forgotten — so it asks again, stopping the task twice, or guesses at
    something it was explicitly told.
    """
    answer: str = 'The user replied: use exponential backoff with a 30s cap.'
    messages: list[ModelMessage] = history_with_an_answer(40, 'y' * 4_000, answer)
    compaction: TieredCompaction[Any] = build_compaction(2_048)

    result: list[ModelMessage] = await compaction.compact(messages, run_context(quiet_model()))

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


@pytest.mark.parametrize('window', [2_048, 4_096, 8_192, 32_768, 128_048, 1_000_000])
def test_a_reply_budget_never_grows_with_the_window(window: int) -> None:
    """
    How long one answer may be does not depend on how much history fits.

    Derived from the window, a 128,048-token setting produced a 38,415-token
    reply budget — meaningless as an answer, and larger than the room actually
    left to generate into, so the model returned nothing at all and the run
    failed reporting that very number.
    """
    assert response_allowance(window) <= MAX_REPLY_TOKENS


@pytest.mark.parametrize('window', [2_048, 4_096, 8_192, 32_768, 128_048, 1_000_000])
def test_every_window_leaves_room_to_answer_in(window: int) -> None:
    """
    A small window is where this bites: 30% of very little is not enough.

    Filling the window to the brim does not fail loudly. The model returns an
    empty response with a length finish reason, which surfaces as "token limit
    exceeded before any response was generated".
    """
    compaction: TieredCompaction[Any] = build_compaction(window)
    allowance: int = response_allowance(window)

    assert compaction.target_tokens is not None
    assert compaction.target_tokens > 0
    assert compaction.target_tokens + allowance <= window


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
    assert settings['max_tokens'] == response_allowance(8_192)
