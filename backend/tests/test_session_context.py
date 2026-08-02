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


def test_completion_without_changes_bypasses_the_usual_gate() -> None:
    """
    A task that needed no change has nothing to verify.

    The usual gate demands a modification and a passing verification, neither of
    which a summarising task will ever have.
    """
    context: SessionContext = SessionContext()

    recorded: str = context.accept_completion_without_changes('It retries three times.')

    assert recorded == 'It retries three times.'
    assert context.completion_requested is True
    assert context.completion_summary == 'It retries three times.'
    assert context.completed_without_changes is True


def test_an_empty_answer_is_not_a_completion() -> None:
    """
    Completing with nothing to show is not completing.
    """
    context: SessionContext = SessionContext()

    with pytest.raises(ValueError):
        context.accept_completion_without_changes('   ')

    assert context.completion_requested is False


def test_an_ordinary_completion_is_not_marked_as_changeless() -> None:
    """
    The flag must distinguish the two routes, not merely record completion.
    """
    context: SessionContext = SessionContext()
    context.record_modification('a.py')
    context.record_verification(True)

    context.request_completion('Changed a.py.')

    assert context.completion_requested is True
    assert context.completed_without_changes is False


def test_completion_is_reachable_without_a_sandbox() -> None:
    """
    A change made with no way to verify it can still be completed.

    Requirement 11.1: the agent can neither run a command nor bring the sandbox
    up, so demanding a verification makes completion unreachable and the agent
    loops until the iteration ceiling.
    """
    context: SessionContext = SessionContext()
    context.mark_sandbox_unavailable()
    context.record_modification('a.py')

    recorded: str = context.request_completion('Added the endpoint.')

    assert 'Added the endpoint.' in recorded
    assert context.completion_requested is True


def test_an_unverified_completion_says_why_it_is_unverified() -> None:
    """
    Requirement 11.2: the reviewer and the user both need to know.
    """
    context: SessionContext = SessionContext()
    context.mark_sandbox_unavailable()
    context.record_modification('a.py')

    recorded: str = context.request_completion('Added the endpoint.')

    assert 'no sandbox was available' in recorded
    assert 'not verified' in recorded


def test_the_sandbox_warning_is_distinct_from_the_failing_test_warning() -> None:
    """
    Unverifiable is not the same as verified-and-failing.

    One says nothing could be checked; the other says something was checked and
    did not pass. Collapsing them would misdescribe the diff to the reviewer.
    """
    without: SessionContext = SessionContext()
    without.mark_sandbox_unavailable()
    without.record_modification('a.py')

    failing: SessionContext = SessionContext(max_failed_verifications=1)
    failing.record_modification('a.py')
    failing.record_verification(False)

    assert 'no sandbox' in without.request_completion('Done.')

    failing_summary: str = failing.request_completion('Done.')
    assert 'no sandbox' not in failing_summary
    assert 'failing verification command' in failing_summary


def test_the_gates_still_apply_when_a_sandbox_exists() -> None:
    """
    Requirement 11.3: the waiver is for the sandboxless case alone.
    """
    context: SessionContext = SessionContext()
    context.record_modification('a.py')

    with pytest.raises(ValueError, match='not been verified'):
        context.request_completion('Done.')

    context.record_verification(False)
    with pytest.raises(ValueError, match='verification command failed'):
        context.request_completion('Done.')


def test_a_missing_sandbox_still_requires_a_modification() -> None:
    """
    The waiver covers verification, not the work itself.

    A task that needed changes and got none is still incomplete, sandbox or no
    sandbox; that question belongs to the adjudicator, not to this gate.
    """
    context: SessionContext = SessionContext()
    context.mark_sandbox_unavailable()

    with pytest.raises(ValueError, match='before any file has been modified'):
        context.request_completion('Done.')


def test_sandbox_availability_does_not_come_back() -> None:
    """
    A sandbox that failed to start will not appear mid-session.

    A gate that could re-arm itself would strand the agent at a requirement it
    had already been excused from.
    """
    context: SessionContext = SessionContext()
    assert context.sandbox_available is True

    context.mark_sandbox_unavailable()
    context.mark_sandbox_unavailable()

    assert context.sandbox_available is False


def test_the_task_is_not_reassignable_once_the_session_is_running() -> None:
    """
    What the session was asked to do is fixed for its lifetime.

    The adjudicator judges the agent's work against this text, so anything able
    to rewrite it part way through could change the standard the work is held to.
    """
    context: SessionContext = SessionContext()
    context.configure('Explain the retry policy.', 'reviewer-model')

    assert context.task == 'Explain the retry policy.'
    assert context.reviewer_model == 'reviewer-model'

    with pytest.raises(AttributeError):
        context.task = 'Something else entirely.'  # type: ignore[misc]


def test_a_fresh_context_knows_no_task() -> None:
    """
    Neither field is known before the opening message has been read.
    """
    context: SessionContext = SessionContext()

    assert context.task == ''
    assert context.reviewer_model == ''
