from __future__ import annotations

import pytest

from services.parsed_plan import ParsedPlan
from services.plan_format import PlanFormat
from services.plan_parser import parse_plan, render_plan
from services.plan_step import PlanStep


def test_extracts_numbered_steps() -> None:
    """
    A well formed plan yields its steps in order.
    """
    plan: ParsedPlan = parse_plan(
        '1. Read the file.\n2. Apply the edit.\n3. Run the tests.\nPLAN_COMPLETE'
    )

    assert plan.titles == ['Read the file.', 'Apply the edit.', 'Run the tests.']
    assert plan.used_fallback is False


@pytest.mark.parametrize('separator', ['.', ')'])
def test_accepts_both_numbering_styles(separator: str) -> None:
    """
    Steps numbered with a period or a parenthesis are both recognised.
    """
    plan: ParsedPlan = parse_plan(f'1{separator} First step\n2{separator} Second step')
    assert plan.titles == ['First step', 'Second step']


def test_discards_preamble_and_postamble() -> None:
    """
    Chatter around the list is ignored.
    """
    plan: ParsedPlan = parse_plan(
        "Sure! Here's my plan:\n\n"
        '1. Read the file.\n'
        '2. Edit it.\n'
        'PLAN_COMPLETE\n\n'
        'Let me know if you want changes!'
    )

    assert plan.titles == ['Read the file.', 'Edit it.']
    assert 'Sure' not in ' '.join(plan.titles)
    assert 'Let me know' not in ' '.join(plan.titles)


def test_folds_wrapped_continuation_lines() -> None:
    """
    A step spanning several lines is kept as one step.
    """
    plan: ParsedPlan = parse_plan(
        '1. Add a timeout parameter to\n'
        '   run_test_command in docker_sandbox.py\n'
        '2. Run the tests.'
    )

    assert plan.titles == [
        'Add a timeout parameter to run_test_command in docker_sandbox.py',
        'Run the tests.',
    ]


def test_unnumbered_output_falls_back_to_one_step() -> None:
    """
    Output with no numbering is treated as a single step.

    Small models frequently ignore the requested format, and rejecting their
    output outright would stall the session.
    """
    plan: ParsedPlan = parse_plan('Just edit the config file and rerun the suite.')

    assert plan.titles == ['Just edit the config file and rerun the suite.']
    assert plan.used_fallback is True


@pytest.mark.parametrize('text', ['', '   \n\n  ', 'PLAN_COMPLETE'])
def test_empty_output_yields_no_steps(text: str) -> None:
    """
    Empty output produces an empty plan, which the caller must handle.
    """
    plan: ParsedPlan = parse_plan(text)
    assert plan.is_empty is True
    assert plan.titles == []


def test_renders_back_to_numbered_text() -> None:
    """
    A plan renders back into numbered text for use in a later prompt.
    """
    plan: ParsedPlan = ParsedPlan(steps=[PlanStep(title='First'), PlanStep(title='Second')])
    assert render_plan(plan) == '1. First\n2. Second'


def test_round_trip_is_stable() -> None:
    """
    Rendering and re-parsing a plan preserves its steps.
    """
    original: ParsedPlan = parse_plan('1. Alpha\n2. Beta\n3. Gamma')
    assert parse_plan(render_plan(original)).titles == original.titles


def test_descriptions_reach_the_rendered_plan() -> None:
    """
    A description that never reaches the prompt has bought nothing.

    The point of asking for structure is that the agent executing the plan gets
    more than a title. Rendering is where that is realised, and it is a separate
    step from producing the structure, so it is asserted separately.
    """
    plan: ParsedPlan = ParsedPlan(
        steps=[
            PlanStep(title='Add a timeout', description='In run_test_command, defaulting to 30s.'),
            PlanStep(title='Run the suite'),
        ],
        format=PlanFormat.structured,
    )

    rendered: str = render_plan(plan)

    assert '1. Add a timeout' in rendered
    assert 'In run_test_command, defaulting to 30s.' in rendered
    # A step without one contributes no empty line to the prompt.
    assert '2. Run the suite' in rendered
    assert rendered.count('\n') == 2


def test_text_plans_are_marked_as_text() -> None:
    """
    The format is carried, not inferred from the absence of descriptions.
    """
    plan: ParsedPlan = parse_plan('1. Alpha\n2. Beta')

    assert plan.format is PlanFormat.text
    assert all(step.description == '' for step in plan.steps)
