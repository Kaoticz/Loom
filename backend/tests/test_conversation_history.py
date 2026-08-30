from __future__ import annotations

import pytest

from services.conversation_history import (
    ConversationHistory,
    FILE_WRITING_TOOLS,
    TurnRole,
    estimate_tokens,
    truncate_to_tokens,
)


def test_token_estimation_heuristics() -> None:
    """
    Prose is counted at four characters per token and code at two.
    """
    assert estimate_tokens('a' * 40) == 10
    assert estimate_tokens('a' * 40, is_code=True) == 20
    assert estimate_tokens('') == 0


def test_truncation_respects_budget() -> None:
    """
    Truncated text fits the budget and is marked as truncated.
    """
    truncated: str = truncate_to_tokens('a' * 400, 20)
    assert estimate_tokens(truncated) <= 20
    assert truncated.endswith('(truncated)')


def test_empty_history_yields_empty_context() -> None:
    """
    A history with no turns renders nothing.
    """
    assert ConversationHistory().build_context(1000) == ''


def test_single_turn_is_rendered_verbatim() -> None:
    """
    A short history is returned in full.
    """
    history: ConversationHistory = ConversationHistory()
    history.add(TurnRole.user, 'Add a timeout to run_test_command.')
    assert 'Add a timeout' in history.build_context(1000)


def test_rejects_zero_recent_turns() -> None:
    """
    At least one recent turn must be preserved.
    """
    with pytest.raises(ValueError):
        ConversationHistory(recent_turns_kept=0)


def build_busy_history() -> ConversationHistory:
    """
    Build a history large enough to force compression.

    :returns: A history spanning four iterations of work.
    """
    history: ConversationHistory = ConversationHistory()
    history.add(TurnRole.user, 'Fix the bug.')
    for iteration in range(1, 5):
        history.advance_iteration()
        history.add(TurnRole.agent, 'Reasoning block. ' * 60)
        history.add(
            TurnRole.tool_call,
            'search/replace',
            tool_name='edit_file',
            metadata={'path': f'backend/services/module_{iteration}.py'},
            is_code=True,
        )
        history.add(
            TurnRole.tool_result,
            'ok',
            tool_name='run_test_command',
            metadata={'success': iteration != 2},
        )
        history.add(
            TurnRole.reviewer,
            'Needs a guard clause on the empty case.',
            metadata={'approved': False},
        )
    return history


def test_compression_respects_budget_and_summarises() -> None:
    """
    An oversized history is compressed into per-iteration summaries within budget.
    """
    history: ConversationHistory = build_busy_history()
    context: str = history.build_context(400)

    assert estimate_tokens(context) <= 400
    assert 'Summary of earlier work' in context
    assert 'In iteration 1, you' in context
    assert 'module_1.py' in context
    assert 'test command(s)' in context
    assert '1 failed' in context
    assert 'reviewer rejected' in context


def test_most_recent_turn_survives_compression() -> None:
    """
    The freshest turn is preserved verbatim even under compression.
    """
    history: ConversationHistory = build_busy_history()
    assert 'Needs a guard clause' in history.build_context(400)


def test_generous_budget_skips_compression() -> None:
    """
    A history that fits the budget is not summarised.
    """
    history: ConversationHistory = build_busy_history()
    assert 'Summary of earlier work' not in history.build_context(100_000)


@pytest.mark.parametrize('budget', [0, 5, 20, 50, 200])
def test_any_budget_is_respected(budget: int) -> None:
    """
    The rendered context never exceeds the requested budget.
    """
    history: ConversationHistory = build_busy_history()
    assert estimate_tokens(history.build_context(budget)) <= budget


def test_iteration_counter_advances() -> None:
    """
    Advancing the iteration increments the recorded index.
    """
    history: ConversationHistory = ConversationHistory()
    assert history.iteration == 0
    assert history.advance_iteration() == 1
    turn = history.add(TurnRole.agent, 'thinking')
    assert turn.iteration == 1


def test_delete_file_is_summarised_as_a_file_modification() -> None:
    """
    A deletion stays in compacted context as a file change, not an unrelated tool call.
    """
    assert 'delete_file' in FILE_WRITING_TOOLS
