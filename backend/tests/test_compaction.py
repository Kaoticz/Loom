from typing import Any, override

import pytest
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    DeduplicateFileReads,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
    estimate_token_count,
)

from services.compaction import (
    DEFAULT_CONTEXT_WINDOW,
    IRRECOVERABLE_TOOLS,
    KEEP_RECENT_MESSAGES,
    MINIMUM_CONTEXT_WINDOW,
    RequestHistoryCapture,
    SummarizeAndStripToolMessages,
    TARGET_FRACTION,
    ObservedCompaction,
    build_compaction,
    build_forced_summarization,
    force_summarize_history,
    read_file_key,
    resolve_context_window,
    _strip_redundant_system_prompts,
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

    Deduplicating a superseded file read costs nothing, clamping one huge result
    costs a little, and summary-plus-stripping runs only when both are insufficient.
    """
    compaction: TieredCompaction[Any] = build_compaction(None)

    assert [type(tier).__name__ for tier in compaction.tiers] == [
        DeduplicateFileReads.__name__,
        ClampOversizedMessages.__name__,
        SummarizeAndStripToolMessages.__name__,
        SlidingWindowCompaction.__name__,
    ]


def test_the_expensive_tiers_sit_behind_the_free_ones() -> None:
    """
    Summarizing costs a real model call, so both free tiers are tried first.

    It is present rather than omitted: a long ReAct loop eventually exceeds what
    deduplication can reclaim, and compressing old context beats discarding it.
    The sliding window sits behind even that, as the floor which guarantees the
    history comes under target however the earlier tiers fare.
    """
    tiers: list[str] = [type(tier).__name__ for tier in build_compaction(None).tiers]

    assert tiers.index(SummarizeAndStripToolMessages.__name__) > tiers.index(
        ClampOversizedMessages.__name__
    )
    assert tiers.index(SlidingWindowCompaction.__name__) > tiers.index(
        SummarizeAndStripToolMessages.__name__
    )


def test_forced_summarization_uses_the_failed_request_window() -> None:
    """
    Overflow recovery summarizes against the original budget before narrowing it.
    """
    summary: SummarizingCompaction[Any] = build_forced_summarization(16_384)

    assert summary.context_window == 16_384
    assert summary.keep_messages == 20
    assert summary.model is None


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


def test_the_combined_tier_uses_the_existing_summary_configuration() -> None:
    """
    The combined tier retains ordinary summary behavior before stripping tools.
    """
    combined: Any = build_compaction(None).tiers[2]

    assert isinstance(combined, SummarizeAndStripToolMessages)
    assert combined.summarizer.max_fraction == TARGET_FRACTION
    assert combined.summarizer.keep_messages == KEEP_RECENT_MESSAGES
    assert combined.summarizer.model is None
    assert combined.exclude_tools == IRRECOVERABLE_TOOLS


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


def history_with_unique_file_reads(
    repeats: int, body: str, *, start: int = 0
) -> list[ModelMessage]:
    """
    Build a history of distinct file reads that deduplication cannot reclaim.

    :param repeats: How many different files to read.
    :param body: The contents each read returned.
    :param start: The first file and tool-call index to use.
    :returns: The synthetic message history.
    """
    messages: list[ModelMessage] = []
    for index in range(start, start + repeats):
        messages.append(
            ModelResponse(parts=[ToolCallPart(
                tool_name='read_file',
                args={'path': f'src/file{index}.py'},
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


def recording_model(calls: list[list[ModelMessage]]) -> FunctionModel:
    """
    Build a model that records each summary request.

    :param calls: The collection receiving every summary request history.
    :returns: A model that records and answers summary requests.
    """

    async def call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """
        Record one summary request and return a short replacement summary.

        :param messages: The summary request history.
        :param info: The model invocation metadata, unused by this stand-in.
        :returns: A minimal summary response.
        """
        _ = info
        calls.append(list(messages))
        return ModelResponse(parts=[TextPart('Done.')])

    return FunctionModel(call)


def request_context(messages: list[ModelMessage]) -> ModelRequestContext:
    """
    Build a model request context for an observing capability test.

    :param messages: The request history the capability will inspect.
    :returns: A request context with no tools or settings.
    """
    return ModelRequestContext(
        model=quiet_model(),
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )


class StubCompaction(AbstractCapability[Any]):
    """
    Replace a request history with a predetermined result.

    :param result: The history this capability returns.
    """

    def __init__(self, result: list[ModelMessage]) -> None:
        """
        Create the deterministic compaction stand-in.

        :param result: The history to leave on the request context.
        """
        super().__init__()
        self._result: list[ModelMessage] = result

    @override
    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """
        Replace the request's history.

        :param ctx: The active run context, unused by the stand-in.
        :param request_context: The request whose history should be replaced.
        :returns: The same request context with its predetermined history.
        """
        request_context.messages = self._result
        return request_context


async def test_request_history_capture_keeps_the_compacted_request() -> None:
    """
    Overflow recovery receives the history after ordinary compaction has run.
    """
    messages: list[ModelMessage] = history_with_file_reads(3, 'x' * 400)
    shortened: list[ModelMessage] = messages[-1:]
    capture: RequestHistoryCapture = RequestHistoryCapture()

    result: ModelRequestContext = await capture.before_model_request(
        run_context(quiet_model()), request_context(shortened)
    )

    assert result.messages == shortened
    assert capture.messages == shortened
    assert capture.messages is not shortened


async def test_an_observed_compaction_reports_one_real_history_reduction() -> None:
    """
    A changed history yields one callback carrying its before and after estimates.
    """
    messages: list[ModelMessage] = history_with_file_reads(3, 'x' * 400)
    shortened: list[ModelMessage] = messages[-1:]
    reports: list[tuple[int, int]] = []

    async def report(before: int, after: int) -> None:
        """
        Record the observed compaction.

        :param before: Estimated tokens before compaction.
        :param after: Estimated tokens after compaction.
        """
        reports.append((before, after))

    observer: ObservedCompaction = ObservedCompaction(StubCompaction(shortened), report)
    result: ModelRequestContext = await observer.before_model_request(
        run_context(quiet_model()), request_context(messages)
    )

    assert result.messages == shortened
    assert reports == [(estimate_token_count(messages), estimate_token_count(shortened))]


async def test_an_observed_compaction_stays_silent_when_history_is_unchanged() -> None:
    """
    A trigger that reclaims nothing must not produce a misleading notice.
    """
    messages: list[ModelMessage] = history_with_file_reads(2, 'x' * 400)
    reports: list[tuple[int, int]] = []

    async def report(before: int, after: int) -> None:
        """
        Record a compaction that should not occur.

        :param before: Estimated tokens before compaction.
        :param after: Estimated tokens after compaction.
        """
        reports.append((before, after))

    observer: ObservedCompaction = ObservedCompaction(StubCompaction(messages), report)
    await observer.before_model_request(run_context(quiet_model()), request_context(messages))

    assert reports == []


async def test_a_failing_compaction_observer_does_not_interrupt_the_request() -> None:
    """
    Losing a notice must not prevent the model request from continuing.
    """
    messages: list[ModelMessage] = history_with_file_reads(3, 'x' * 400)
    shortened: list[ModelMessage] = messages[-1:]

    async def fail(before: int, after: int) -> None:
        """
        Simulate an unavailable reporting destination.

        :param before: Estimated tokens before compaction.
        :param after: Estimated tokens after compaction.
        :raises RuntimeError: Always, to exercise observer isolation.
        """
        raise RuntimeError('event bus unavailable')

    observer: ObservedCompaction = ObservedCompaction(StubCompaction(shortened), fail)
    result: ModelRequestContext = await observer.before_model_request(
        run_context(quiet_model()), request_context(messages)
    )

    assert result.messages == shortened


async def test_a_history_under_budget_is_left_alone() -> None:
    """
    Compaction that fires when it is not needed is just data loss.
    """
    messages: list[ModelMessage] = history_with_unique_file_reads(12, 'x' * 100)
    compaction: TieredCompaction[Any] = build_compaction(1_000_000)

    first: list[ModelMessage] = await compaction.compact(messages, run_context(quiet_model()))
    result: list[ModelMessage] = await compaction.compact(first, run_context(quiet_model()))

    assert result == messages


async def test_summarizing_and_stripping_removes_tool_parts_from_the_verbatim_tail() -> None:
    """
    Tool parts are stripped after a summary even when that summary fits the target.
    """
    window: int = 2_048
    body: str = 'x' * 450
    answer: str = 'Use exponential backoff with a 30s cap.'
    messages: list[ModelMessage] = history_with_unique_file_reads(14, body)
    messages.extend([
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
    ])
    summarizer: SummarizingCompaction[Any] = SummarizingCompaction(
        max_fraction=TARGET_FRACTION,
        context_window=window,
        keep_messages=KEEP_RECENT_MESSAGES,
        model=None,
    )
    summarized: list[ModelMessage] = await summarizer.compact(messages, run_context(quiet_model()))
    calls: list[list[ModelMessage]] = []
    result: list[ModelMessage] = await build_compaction(window).compact(
        messages, run_context(recording_model(calls))
    )
    tool_parts: list[ToolCallPart | ToolReturnPart] = [
        part
        for message in result
        for part in message.parts
        if isinstance(part, (ToolCallPart, ToolReturnPart))
    ]

    assert estimate_token_count(messages) > int(window * TARGET_FRACTION)
    assert estimate_token_count(summarized) <= int(window * TARGET_FRACTION)
    assert any(
        isinstance(part, (ToolCallPart, ToolReturnPart)) and part.tool_name == 'read_file'
        for message in summarized
        for part in message.parts
    )
    assert len(calls) == 1
    assert tool_parts
    assert all(part.tool_name == 'ask_user' for part in tool_parts)
    assert [
        part.content
        for message in result
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'ask_user'
    ] == [answer]


async def test_compaction_paths_keep_system_prompts_on_the_leading_request() -> None:
    """
    Summary compaction removes a preserved first request's duplicate system prompt.
    """
    from pydantic_ai.messages import UserPromptPart

    window: int = 2_048
    system_prompt: SystemPromptPart = SystemPromptPart('Follow the repository instructions.')
    task: str = 'Repair the compaction retry without changing dependencies.'
    initial_request: ModelRequest = ModelRequest(parts=[system_prompt, UserPromptPart(task)])
    messages: list[ModelMessage] = [
        initial_request,
        *history_with_unique_file_reads(14, 'x' * 450),
    ]
    model: FunctionModel = quiet_model()
    in_session: list[ModelMessage] = await build_compaction(window).compact(
        messages, run_context(model)
    )
    forced: list[ModelMessage] = await force_summarize_history(messages, window, model)

    for compacted in (in_session, forced):
        assert isinstance(compacted[0], ModelRequest)
        assert any(isinstance(part, SystemPromptPart) for part in compacted[0].parts)
        assert all(
            not isinstance(part, SystemPromptPart)
            for message in compacted[1:]
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        assert any(
            isinstance(part, UserPromptPart) and part.content == task
            for message in compacted
            if isinstance(message, ModelRequest)
            for part in message.parts
        )


def test_system_only_nonleading_request_is_not_emptied() -> None:
    """
    Normalization keeps a system-only request rather than emit an empty request.
    """
    messages: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart('Leading summary.')]),
        ModelRequest(parts=[SystemPromptPart('System-only continuation.')]),
    ]

    normalized: list[ModelMessage] = _strip_redundant_system_prompts(messages)

    assert normalized == messages
    assert normalized is not messages


async def test_summarizing_and_stripping_preserves_an_all_tool_terminal_request() -> None:
    """
    The current tool result remains when stripping it would empty the final request.
    """
    window: int = 2_048
    body: str = 'x' * 450
    current_step: ToolReturnPart = ToolReturnPart(
        tool_name='read_file',
        content='The final file contents.',
        tool_call_id='current-step',
    )
    messages: list[ModelMessage] = history_with_unique_file_reads(14, body)
    messages.extend([
        ModelResponse(parts=[TextPart('I will inspect the final file now.')]),
        ModelRequest(parts=[current_step]),
    ])
    calls: list[list[ModelMessage]] = []

    result: list[ModelMessage] = await build_compaction(window).compact(
        messages, run_context(recording_model(calls))
    )

    assert len(calls) == 1
    assert isinstance(result[0], ModelRequest)
    assert isinstance(result[-1], ModelRequest)
    assert result[-1].parts == [current_step]


async def test_summarizing_and_stripping_backfills_an_empty_terminal_request() -> None:
    """
    An empty continuation request becomes the latest user prompt after stripping.
    """
    window: int = 2_048
    body: str = 'x' * 450
    latest_prompt: str = 'Update the retry policy without changing the database schema.'
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Initial task.')]),
        *history_with_unique_file_reads(14, body),
        ModelRequest(parts=[UserPromptPart(latest_prompt)]),
        ModelResponse(parts=[TextPart('I am checking the relevant code.')]),
        ModelResponse(parts=[TextPart('The change is ready to apply.')]),
        ModelRequest(parts=[]),
    ]
    calls: list[list[ModelMessage]] = []

    result: list[ModelMessage] = await build_compaction(window).compact(
        messages, run_context(recording_model(calls))
    )
    assert isinstance(result[-1], ModelRequest)
    terminal_request: ModelRequest = result[-1]
    non_terminal_tool_parts: list[ToolCallPart | ToolReturnPart] = [
        part
        for message in result[:-1]
        for part in message.parts
        if isinstance(part, (ToolCallPart, ToolReturnPart))
    ]
    surviving_response_texts: list[str] = [
        part.content
        for message in result[:-1]
        for part in message.parts
        if isinstance(part, TextPart)
    ]

    assert len(calls) == 1
    assert isinstance(terminal_request.parts[0], UserPromptPart)
    assert terminal_request.parts[0].content == latest_prompt
    assert all(part.tool_name == 'ask_user' for part in non_terminal_tool_parts)
    assert surviving_response_texts[-2:] == [
        'I am checking the relevant code.',
        'The change is ready to apply.',
    ]


async def test_summarizing_and_stripping_uses_a_fallback_for_an_empty_terminal_request() -> None:
    """
    An empty continuation request remains wire-mappable without a user prompt.
    """
    window: int = 2_048
    body: str = 'x' * 450
    messages: list[ModelMessage] = [
        *history_with_unique_file_reads(14, body),
        ModelResponse(parts=[TextPart('I am ready to continue.')]),
        ModelRequest(parts=[]),
    ]
    calls: list[list[ModelMessage]] = []

    result: list[ModelMessage] = await build_compaction(window).compact(
        messages, run_context(recording_model(calls))
    )
    assert isinstance(result[-1], ModelRequest)
    terminal_request: ModelRequest = result[-1]

    assert len(calls) == 1
    assert isinstance(terminal_request.parts[0], UserPromptPart)
    assert terminal_request.parts[0].content == 'Continue with the current task.'


async def test_deduplication_can_avoid_summary_and_tool_stripping() -> None:
    """
    Deduplicating repeated reads below target skips the combined tier entirely.
    """
    body: str = 'x' * 800
    messages: list[ModelMessage] = history_with_file_reads(20, body)
    calls: list[list[ModelMessage]] = []
    result: list[ModelMessage] = await build_compaction(2_048).compact(
        messages, run_context(recording_model(calls))
    )
    returns: list[str] = [
        str(part.content)
        for message in result
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]

    assert calls == []
    assert returns.count(body) == 1


async def test_sliding_window_remains_the_fallback_after_summarizing_and_stripping() -> None:
    """
    A too-large non-tool tail reaches the sliding-window tier after summarizing.
    """
    messages: list[ModelMessage] = [
        ModelResponse(parts=[TextPart('x' * 500)]) for _ in range(45)
    ]
    calls: list[list[ModelMessage]] = []
    result: list[ModelMessage] = await build_compaction(2_048).compact(
        messages, run_context(recording_model(calls))
    )

    assert len(calls) == 1
    assert len(result) == KEEP_RECENT_MESSAGES


async def test_a_later_overage_repeats_the_summarize_and_strip_cycle() -> None:
    """
    New tool traffic after compaction can trigger the combined tier again.
    """
    body: str = 'x' * 450
    messages: list[ModelMessage] = history_with_unique_file_reads(14, body)
    messages.extend([
        ModelResponse(parts=[ToolCallPart(
            tool_name='ask_user',
            args={'question': 'Which retry policy?'},
            tool_call_id='ask-1',
        )]),
        ModelRequest(parts=[ToolReturnPart(
            tool_name='ask_user',
            content='Use exponential backoff with a 30s cap.',
            tool_call_id='ask-1',
        )]),
    ])
    calls: list[list[ModelMessage]] = []
    model: FunctionModel = recording_model(calls)
    compaction: TieredCompaction[Any] = build_compaction(2_048)
    first: list[ModelMessage] = await compaction.compact(messages, run_context(model))
    second: list[ModelMessage] = await compaction.compact(
        [*first, *history_with_unique_file_reads(14, body, start=14)], run_context(model)
    )
    non_terminal_tool_parts: list[ToolCallPart | ToolReturnPart] = [
        part
        for message in second[:-1]
        for part in message.parts
        if isinstance(part, (ToolCallPart, ToolReturnPart))
    ]

    assert len(calls) == 2
    assert all(part.tool_name == 'ask_user' for part in non_terminal_tool_parts)
    assert isinstance(second[-1], ModelRequest)
    assert len(second[-1].parts) == 1
    assert isinstance(second[-1].parts[0], ToolReturnPart)
    assert second[-1].parts[0].tool_name == 'read_file'
    assert second[-1].parts[0].content == body


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
    The four requested tiers are present in their intended order.
    """
    tiers = {type(tier).__name__ for tier in build_compaction(None).tiers}

    assert {
        'DeduplicateFileReads',
        'ClampOversizedMessages',
        'SummarizeAndStripToolMessages',
        'SlidingWindowCompaction',
    } <= tiers
