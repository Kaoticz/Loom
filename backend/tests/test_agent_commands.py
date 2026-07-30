from __future__ import annotations

import json

import pytest

from schemas.agent_commands import CommandAction, SessionCommand, parse_command


def test_shorthand_approval_is_understood() -> None:
    """
    The client's minimal `{"approved": true}` form is honoured.
    """
    command: SessionCommand = parse_command(json.dumps({'approved': True}))
    assert command.is_approval is True
    assert command.resolved_action is CommandAction.approve


def test_shorthand_rejection_carries_a_reason() -> None:
    """
    A rejection can carry the feedback the agent needs to replan.
    """
    command: SessionCommand = parse_command(
        json.dumps({'approved': False, 'reason': 'Missing the migration.'})
    )
    assert command.is_approval is False
    assert command.reason == 'Missing the migration.'


@pytest.mark.parametrize(
    'action,approves',
    [
        ('approve', True),
        ('merge', True),
        ('auto_approve', True),
        ('reject', False),
        ('abort', False),
        ('continue', False),
    ],
)
def test_explicit_actions_resolve(action: str, approves: bool) -> None:
    """
    An explicit action is recognised and classified correctly.
    """
    command: SessionCommand = parse_command(json.dumps({'action': action}))
    assert command.resolved_action.value == action
    assert command.is_approval is approves


def test_explicit_action_wins_over_the_flag() -> None:
    """
    An explicit action takes precedence over the shorthand flag.
    """
    command: SessionCommand = parse_command(json.dumps({'approved': True, 'action': 'abort'}))
    assert command.resolved_action is CommandAction.abort


@pytest.mark.parametrize('raw', ['not json at all', '[]', '"just a string"', ''])
def test_unusable_payloads_are_treated_as_rejections(raw: str) -> None:
    """
    A malformed message stalls the session for review rather than approving it.

    Defaulting to approval would let a broken client merge unreviewed work.
    """
    command: SessionCommand = parse_command(raw)
    assert command.is_approval is False


def test_unknown_fields_are_ignored() -> None:
    """
    Extra fields from a newer client do not break parsing.
    """
    command: SessionCommand = parse_command(
        json.dumps({'approved': True, 'something_new': 'value'})
    )
    assert command.is_approval is True


def test_missing_flag_is_a_rejection() -> None:
    """
    An empty object is not read as approval.
    """
    assert parse_command('{}').is_approval is False
