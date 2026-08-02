from __future__ import annotations

from collections.abc import Sequence

DEFAULT_MAX_FAILED_VERIFICATIONS: int = 3


class SessionContext:
    """
    Mutable state shared between the agent's tools for the duration of a session.

    Tools are otherwise stateless, but a few decisions depend on what has already
    happened: `complete_task` must refuse to finish before any work was done, and
    the controller needs to know when the agent has signalled completion. This
    object is the single place that knowledge lives, and is passed to tools
    through `AgentDeps`.
    """

    _modified_files: list[str]
    _verified_since_modification: bool
    _last_verification_passed: bool
    _failed_verifications: int
    _max_failed_verifications: int
    _completion_summary: str | None
    _completed_without_changes: bool
    _sandbox_available: bool
    _task: str
    _reviewer_model: str

    def __init__(self, max_failed_verifications: int = DEFAULT_MAX_FAILED_VERIFICATIONS) -> None:
        """
        Create an empty session context.

        :param max_failed_verifications: How many failing verification commands to tolerate before allowing the agent to finish anyway.
        :raises ValueError: If the allowance is negative.
        """
        if max_failed_verifications < 0:
            raise ValueError('The failed verification allowance cannot be negative.')
        self._modified_files = []
        self._verified_since_modification = False
        self._last_verification_passed = False
        self._failed_verifications = 0
        self._max_failed_verifications = max_failed_verifications
        self._completion_summary = None
        self._completed_without_changes = False
        self._sandbox_available = True
        self._task = ''
        self._reviewer_model = ''

    @property
    def modified_files(self) -> Sequence[str]:
        """
        Every file the agent has created or edited this session, in order.

        :returns: An immutable view over the modified file paths.
        """
        return tuple(self._modified_files)

    @property
    def has_modified_files(self) -> bool:
        """
        Whether the agent has modified at least one file.

        :returns: True if any file has been created or edited.
        """
        return len(self._modified_files) > 0

    @property
    def verified_since_modification(self) -> bool:
        """
        Whether a verification command has run since the last file modification.

        :returns: True if the most recent modification has been followed by a test run.
        """
        return self._verified_since_modification

    @property
    def last_verification_passed(self) -> bool:
        """
        Whether the most recent verification command exited successfully.

        :returns: True if a verification command has run and passed.
        """
        return self._verified_since_modification and self._last_verification_passed

    @property
    def failed_verifications(self) -> int:
        """
        How many verification commands have failed this session.

        :returns: The running count of failing verification commands.
        """
        return self._failed_verifications

    @property
    def verification_allowance_exhausted(self) -> bool:
        """
        Whether enough verifications have failed to release the completion gate.

        A repository can have tests that were already failing before the agent
        started. Without an escape hatch the agent could never finish such a
        task, so after a fixed number of failures it is allowed to complete with
        the failure recorded for the reviewer to judge.

        :returns: True if the agent may complete despite a failing verification.
        """
        return self._failed_verifications >= self._max_failed_verifications

    @property
    def completion_requested(self) -> bool:
        """
        Whether the agent has signalled that its work is finished.

        :returns: True if `complete_task` has been accepted.
        """
        return self._completion_summary is not None

    @property
    def completion_summary(self) -> str | None:
        """
        The summary the agent supplied when it signalled completion.

        :returns: The summary text, or None if completion has not been signalled.
        """
        return self._completion_summary

    @property
    def task(self) -> str:
        """
        The task as the user originally described it.

        :returns: The task description, or an empty string before it is known.
        """
        return self._task

    @property
    def reviewer_model(self) -> str:
        """
        The model configured for judging the work.

        :returns: The reviewer model identifier, or an empty string before it is known.
        """
        return self._reviewer_model

    def configure(self, task: str, reviewer_model: str) -> None:
        """
        Record what the session was asked to do, and which model judges it.

        Both are needed by the adjudicating subagent, which runs from inside a
        tool call where the session itself is out of reach. They are set here
        rather than left as open attributes so that the task cannot be quietly
        rewritten part way through by anything holding the context.

        :param task: The task as the user described it.
        :param reviewer_model: The model identifier to use for judging the work.
        """
        self._task = task
        self._reviewer_model = reviewer_model

    @property
    def sandbox_available(self) -> bool:
        """
        Whether commands can be run in a sandbox during this session.

        The single place this is recorded. The banner the user sees and the
        completion gate the agent meets both read it here, so the interface
        cannot say the sandbox is gone while the gate goes on demanding a
        verification that can never be run.

        :returns: True while the sandbox is usable.
        """
        return self._sandbox_available

    def mark_sandbox_unavailable(self) -> None:
        """
        Record that no sandbox is available for the rest of the session.

        One way only: a sandbox that failed to start is not going to be brought
        up mid-session, and a gate that could re-arm itself would strand the
        agent at a requirement it had already been excused.
        """
        self._sandbox_available = False

    @property
    def completed_without_changes(self) -> bool:
        """
        Whether completion was accepted for a task that needed no file changed.

        The session skips review in that case: there is no diff to review, and
        the reviewing state would otherwise reject the task for having produced
        nothing.

        :returns: True if completion was adjudicated as needing no changes.
        """
        return self._completed_without_changes

    def accept_completion_without_changes(self, summary: str) -> str:
        """
        Accept completion for a task adjudicated as requiring no file changes.

        The usual gate demands a modification and a passing verification. Neither
        applies to a task that was only ever going to produce an answer, so this
        bypasses both rather than weakening them for everyone.

        :param summary: The agent's description of what it found.
        :returns: The recorded summary.
        :raises ValueError: If the summary is empty.
        """
        cleaned: str = summary.strip()
        if not cleaned:
            raise ValueError('A completion summary is required.')
        self._completion_summary = cleaned
        self._completed_without_changes = True
        return cleaned

    def record_modification(self, path: str) -> int:
        """
        Record that the agent created or edited a file.

        :param path: The repository-relative path that was modified.
        :returns: The total number of modifications recorded so far.
        """
        self._modified_files.append(path)
        self._verified_since_modification = False
        self._last_verification_passed = False
        return len(self._modified_files)

    def record_verification(self, passed: bool) -> bool:
        """
        Record that the agent ran a verification command.

        :param passed: Whether the command exited successfully.
        :returns: Whether the recorded verification passed.
        """
        self._verified_since_modification = True
        self._last_verification_passed = passed
        if not passed:
            self._failed_verifications += 1
        return self._last_verification_passed

    def request_completion(self, summary: str) -> str:
        """
        Record the agent's completion signal.

        :param summary: The agent's description of what it accomplished.
        :returns: The recorded summary.
        :raises ValueError: If no file has been modified yet, or the summary is empty.
        """
        if not self.has_modified_files:
            raise ValueError(
                'Cannot complete the task before any file has been modified. '
                'Make the changes the task requires, verify them with run_test_command, '
                'and only then call complete_task.'
            )
        # Without a sandbox the agent cannot run a command and cannot bring one
        # up, so demanding a verification makes completion unreachable and the
        # agent loops until the iteration ceiling. The requirement is waived
        # rather than weakened: it still applies wherever a sandbox exists.
        if self._sandbox_available and not self._verified_since_modification:
            raise ValueError(
                f'The last change to {self._modified_files[-1]} has not been verified yet. '
                'Run a relevant test, build or lint command with run_test_command, '
                'and only call complete_task once it passes.'
            )
        if (
            self._sandbox_available
            and not self._last_verification_passed
            and not self.verification_allowance_exhausted
        ):
            remaining: int = self._max_failed_verifications - self._failed_verifications
            raise ValueError(
                'The most recent verification command failed. Diagnose the failure, '
                'fix it, and re-run the command until it passes before completing the task. '
                f'({remaining} further failed attempt(s) will release this requirement, '
                'for repositories whose tests were already failing.)'
            )
        if not summary.strip():
            raise ValueError('The completion summary cannot be empty.')

        cleaned: str = summary.strip()
        if not self._sandbox_available:
            cleaned = (
                f'{cleaned}\n\nWARNING: no sandbox was available in this session, so no command '
                'could be run and these changes were not verified at all; review the diff '
                'carefully.'
            )
        elif not self._last_verification_passed:
            cleaned = (
                f'{cleaned}\n\nWARNING: completed with a failing verification command after '
                f'{self._failed_verifications} failed attempt(s). The changes were not verified '
                'green; review the diff carefully.'
            )

        self._completion_summary = cleaned
        return self._completion_summary
