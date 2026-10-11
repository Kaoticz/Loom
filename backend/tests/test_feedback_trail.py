from __future__ import annotations

import pytest
from pydantic import ValidationError

from services.feedback_trail import FeedbackSource, FeedbackTrail, render_for_coder, render_for_planner


def test_entries_come_back_in_order_with_everything_preserved() -> None:
    """
    The trail is a faithful record: what was added is what is read back.
    """
    trail: FeedbackTrail = FeedbackTrail()
    trail.add(FeedbackSource.reviewer_rejection, 1, 'The empty case is unhandled.')
    trail.add(FeedbackSource.user_plan_rejection, 2, 'You forgot the migration.')
    trail.add(FeedbackSource.user_result_rejection, 3, 'Please also update the README.')

    entries = list(trail.entries)

    assert [(entry.source, entry.iteration, entry.text) for entry in entries] == [
        (FeedbackSource.reviewer_rejection, 1, 'The empty case is unhandled.'),
        (FeedbackSource.user_plan_rejection, 2, 'You forgot the migration.'),
        (FeedbackSource.user_result_rejection, 3, 'Please also update the README.'),
    ]


def test_entries_are_an_immutable_view() -> None:
    """
    Callers read the trail but cannot rewrite it from the outside.
    """
    trail: FeedbackTrail = FeedbackTrail()
    trail.add(FeedbackSource.reviewer_rejection, 1, 'The empty case is unhandled.')

    assert isinstance(trail.entries, tuple)


def test_an_iteration_below_one_is_rejected() -> None:
    """
    Iterations count from one, so a zero or negative round is a caller bug.
    """
    trail: FeedbackTrail = FeedbackTrail()

    with pytest.raises(ValidationError):
        trail.add(FeedbackSource.reviewer_rejection, 0, 'Too early for feedback.')


def test_planner_sees_every_source_in_full() -> None:
    """
    The planner acts on plan-level objections too, so nothing is withheld.
    """
    trail: FeedbackTrail = FeedbackTrail()
    trail.add(FeedbackSource.reviewer_rejection, 1, 'The empty case is unhandled in a.py.')
    trail.add(FeedbackSource.user_plan_rejection, 2, 'You forgot the database migration.')
    trail.add(FeedbackSource.user_result_rejection, 3, 'Please also update the README.')

    rendered: str = render_for_planner(trail)

    assert 'The empty case is unhandled in a.py.' in rendered
    assert 'You forgot the database migration.' in rendered
    assert 'Please also update the README.' in rendered


def test_coder_sees_reviewer_and_result_feedback_but_not_plan_feedback() -> None:
    """
    Plan-rejection feedback describes a plan the coder never executed.

    Showing it would add noise rather than something actionable, so its absence
    is asserted explicitly rather than inferred from the other two being present.
    """
    trail: FeedbackTrail = FeedbackTrail()
    trail.add(FeedbackSource.reviewer_rejection, 1, 'The empty case is unhandled in a.py.')
    trail.add(FeedbackSource.user_plan_rejection, 2, 'You forgot the database migration.')
    trail.add(FeedbackSource.user_result_rejection, 3, 'Please also update the README.')

    rendered: str = render_for_coder(trail)

    assert 'The empty case is unhandled in a.py.' in rendered
    assert 'Please also update the README.' in rendered
    assert 'You forgot the database migration.' not in rendered


def test_coder_sees_nothing_for_a_plan_rejection_only_trail() -> None:
    """
    With only plan-level feedback there is nothing actionable for the coder.

    An empty string lets the caller skip the section entirely rather than send
    a header with nothing under it.
    """
    trail: FeedbackTrail = FeedbackTrail()
    trail.add(FeedbackSource.user_plan_rejection, 1, 'You forgot the database migration.')

    assert render_for_coder(trail) == ''


def test_empty_trail_renders_to_empty_for_both_audiences() -> None:
    """
    The first round has no feedback yet, so neither prompt grows a section.
    """
    trail: FeedbackTrail = FeedbackTrail()

    assert render_for_planner(trail) == ''
    assert render_for_coder(trail) == ''


def test_long_feedback_passes_through_unmodified() -> None:
    """
    The trail never summarizes or truncates: several paragraphs arrive intact.
    """
    long_text: str = '\n\n'.join(
        f'Paragraph {index}: ' + 'the timeout handling needs care. ' * 40 for index in range(4)
    )
    planner_trail: FeedbackTrail = FeedbackTrail()
    planner_trail.add(FeedbackSource.reviewer_rejection, 2, long_text)
    coder_trail: FeedbackTrail = FeedbackTrail()
    coder_trail.add(FeedbackSource.user_result_rejection, 3, long_text)

    assert long_text in render_for_planner(planner_trail)
    assert long_text in render_for_coder(coder_trail)


def test_entries_are_labeled_by_source_and_round() -> None:
    """
    Feedback from several rounds is only useful when its origin is visible.
    """
    trail: FeedbackTrail = FeedbackTrail()
    trail.add(FeedbackSource.reviewer_rejection, 2, 'The empty case is unhandled.')

    rendered: str = render_for_planner(trail)

    assert '2' in rendered
    assert 'reviewer rejection' in rendered
