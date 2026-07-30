from __future__ import annotations

import pytest

from services.session_context import DEFAULT_MAX_FAILED_VERIFICATIONS, SessionContext


def test_starts_empty(session: SessionContext) -> None:
    """
    A new context has no work recorded.
    """
    assert session.has_modified_files is False
    assert session.completion_requested is False
    assert session.modified_files == ()


def test_completion_requires_a_modification(session: SessionContext) -> None:
    """
    Completing before any file changed is refused.
    """
    with pytest.raises(ValueError, match='before any file has been modified'):
        session.request_completion('done')


def test_completion_requires_a_verification(session: SessionContext) -> None:
    """
    Completing before the change is verified is refused.
    """
    session.record_modification('a.py')
    with pytest.raises(ValueError, match='has not been verified'):
        session.request_completion('done')


def test_completion_requires_the_verification_to_pass(session: SessionContext) -> None:
    """
    A failing verification does not unlock completion.
    """
    session.record_modification('a.py')
    session.record_verification(False)
    with pytest.raises(ValueError, match='verification command failed'):
        session.request_completion('done')


def test_completion_succeeds_after_passing_verification(session: SessionContext) -> None:
    """
    A modification followed by a passing verification unlocks completion.
    """
    session.record_modification('a.py')
    session.record_verification(True)

    recorded: str = session.request_completion('Fixed the bug in a.py.')
    assert session.completion_requested is True
    assert 'Fixed the bug' in recorded
    assert 'WARNING' not in recorded


def test_new_modification_relocks_the_gate(session: SessionContext) -> None:
    """
    Editing again after verifying requires a fresh verification.
    """
    session.record_modification('a.py')
    session.record_verification(True)
    session.record_modification('b.py')

    assert session.verified_since_modification is False
    with pytest.raises(ValueError, match='has not been verified'):
        session.request_completion('done')


def test_allowance_releases_the_gate_after_repeated_failures(session: SessionContext) -> None:
    """
    Persistent verification failures eventually allow completion.

    A repository whose tests were already failing would otherwise trap the agent
    in a loop it cannot exit.
    """
    session.record_modification('a.py')
    for _ in range(DEFAULT_MAX_FAILED_VERIFICATIONS):
        session.record_verification(False)

    assert session.verification_allowance_exhausted is True
    recorded: str = session.request_completion('Changed a.py.')
    assert session.completion_requested is True
    assert 'WARNING' in recorded
    assert 'not verified' in recorded


def test_allowance_is_configurable() -> None:
    """
    The number of tolerated failures can be tuned per session.
    """
    context: SessionContext = SessionContext(max_failed_verifications=1)
    context.record_modification('a.py')
    context.record_verification(False)
    assert context.verification_allowance_exhausted is True
    assert 'WARNING' in context.request_completion('done')


def test_negative_allowance_rejected() -> None:
    """
    A negative failure allowance is invalid.
    """
    with pytest.raises(ValueError):
        SessionContext(max_failed_verifications=-1)


def test_empty_summary_rejected(session: SessionContext) -> None:
    """
    A completion summary must contain something.
    """
    session.record_modification('a.py')
    session.record_verification(True)
    with pytest.raises(ValueError):
        session.request_completion('   ')


def test_failed_verifications_are_counted(session: SessionContext) -> None:
    """
    Only failing verifications increment the failure counter.
    """
    session.record_modification('a.py')
    session.record_verification(False)
    session.record_verification(True)
    session.record_verification(False)
    assert session.failed_verifications == 2
