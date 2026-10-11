from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class FeedbackSource(str, Enum):
    """
    Where a piece of carried-forward feedback came from.

    The planner sees every source, while the coder only sees the two that
    describe executed work.
    """

    reviewer_rejection = 'reviewer_rejection'
    user_plan_rejection = 'user_plan_rejection'
    user_result_rejection = 'user_result_rejection'


class FeedbackEntry(BaseModel):
    """
    One piece of feedback carried forward across re-working rounds.

    The text is always the full, untruncated reviewer comment or rejection
    reason, and the iteration records which round it came from.
    """

    source: Annotated[FeedbackSource, Field()]
    iteration: Annotated[int, Field(ge=1)]
    text: Annotated[str, Field()]


_SOURCE_LABELS: dict[FeedbackSource, str] = {
    FeedbackSource.reviewer_rejection: 'reviewer rejection',
    FeedbackSource.user_plan_rejection: 'user plan rejection',
    FeedbackSource.user_result_rejection: 'user result rejection',
}

_CODER_VISIBLE_SOURCES: frozenset[FeedbackSource] = frozenset({
    FeedbackSource.reviewer_rejection,
    FeedbackSource.user_result_rejection,
})


class FeedbackTrail:
    """
    Accumulates reviewer and user feedback across re-working rounds.

    Every entry persists in full for the life of the trail: nothing is
    summarized, truncated, or capped, so a later planning or execution round
    can read exactly what was said in an earlier one.
    """

    _entries: list[FeedbackEntry]

    def __init__(self) -> None:
        """
        Create an empty feedback trail.
        """
        self._entries = []

    @property
    def entries(self) -> Sequence[FeedbackEntry]:
        """
        Every entry recorded so far, in order.

        :returns: An immutable view over the recorded entries.
        """
        return tuple(self._entries)

    def add(self, source: FeedbackSource, iteration: int, text: str) -> FeedbackEntry:
        """
        Record a piece of feedback from one round.

        :param source: Where the feedback came from.
        :param iteration: The round it came from, counting from one.
        :param text: The full reviewer comment or rejection reason.
        :returns: The recorded entry.
        :raises ValidationError: If the iteration is below one.
        """
        entry: FeedbackEntry = FeedbackEntry(source=source, iteration=iteration, text=text)
        self._entries.append(entry)
        return entry


def render_for_planner(trail: FeedbackTrail) -> str:
    """
    Render every entry in full for the planner's prompt.

    :param trail: The accumulated feedback.
    :returns: The labeled feedback, or an empty string when there is none.
    """
    return _render(trail.entries)


def render_for_coder(trail: FeedbackTrail) -> str:
    """
    Render only executed-work feedback in full for the coder's step prompt.

    Plan-rejection feedback describes a plan the coder never executed, so it is
    withheld here rather than added as noise; the planner still sees it.

    :param trail: The accumulated feedback.
    :returns: The labeled feedback, or an empty string when there is nothing to show.
    """
    shown: list[FeedbackEntry] = [
        entry for entry in trail.entries if entry.source in _CODER_VISIBLE_SOURCES
    ]
    return _render(shown)


def _render(entries: Sequence[FeedbackEntry]) -> str:
    """
    Render entries in full, each labeled by source and round.

    :param entries: The entries to render, already in order.
    :returns: The labeled feedback, or an empty string when there is nothing to render.
    """
    if not entries:
        return ''
    blocks: list[str] = [
        f'- Round {entry.iteration} ({_SOURCE_LABELS[entry.source]}):\n{entry.text}'
        for entry in entries
    ]
    return 'Feedback from earlier rounds:\n' + '\n'.join(blocks)
