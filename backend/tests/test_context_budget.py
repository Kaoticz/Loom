from agents.coder import CODER_SYSTEM_PROMPT, coder_agent
from services.compaction import DEFAULT_CONTEXT_WINDOW
from services.context_budget import (
    WORKING_HEADROOM_TOKENS,
    context_window_complaint,
    estimate_fixed_overhead,
)


def test_the_fixed_overhead_is_measured_from_what_is_actually_sent() -> None:
    """
    The system prompt and the tool schemas go out on every request.

    Compaction budgets the message history and cannot touch either, so they are
    a floor: no amount of trimming brings a request below them.
    """
    overhead: int = estimate_fixed_overhead(CODER_SYSTEM_PROMPT, coder_agent)

    assert overhead > 0
    # Sanity: the tool schemas are a real share of it, not a rounding error.
    assert overhead > len(CODER_SYSTEM_PROMPT) // 4


def test_a_window_that_cannot_hold_the_overhead_is_complained_about() -> None:
    """
    The provider's own error names neither the window nor what filled it.
    """
    complaint: str = context_window_complaint(4_096, 5_467)

    assert complaint
    assert '4,096' in complaint
    assert '5,467' in complaint
    assert 'num_ctx' in complaint


def test_a_window_with_no_room_to_work_is_complained_about() -> None:
    """
    Fitting the overhead exactly is not enough to do anything with.
    """
    assert context_window_complaint(5_500, 5_467)


def test_a_workable_window_is_not_complained_about() -> None:
    """
    A warning on every run would train the user to ignore it.
    """
    assert context_window_complaint(5_467 + WORKING_HEADROOM_TOKENS, 5_467) == ''
    assert context_window_complaint(32_768, 5_467) == ''


def test_the_default_window_can_carry_the_real_overhead() -> None:
    """
    Loom's own default must not be one that cannot work.

    This fails if the system prompt or the tool schemas grow past what the
    default window can hold, which is the drift that produced the original
    report: instructions accumulated over several phases until a small model
    could not fit a request at all.
    """
    overhead: int = estimate_fixed_overhead(CODER_SYSTEM_PROMPT, coder_agent)

    assert context_window_complaint(DEFAULT_CONTEXT_WINDOW, overhead) == '', (
        f'the fixed overhead has grown to {overhead} tokens, which no longer leaves '
        f'{WORKING_HEADROOM_TOKENS} tokens of room in the default {DEFAULT_CONTEXT_WINDOW}'
    )
