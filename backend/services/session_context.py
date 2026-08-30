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
    _changes_are_required: bool | None
    _passed_commands: list[str]
    _test_command: str
    _verification_waiver: str
    _pending_changes: list[str]
    _uncommitted_steps: list[str]
    _consecutive_commit_failures: int
    _step_modified_files: list[str]
    _step_summaries: list[str]
    _current_step: str | None
    _terminal_success_count: int

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
        self._changes_are_required = None
        # Step-scoped state, kept alongside the task-wide fields above rather
        # than replacing them. The whole-task counters feed `complete_task`, the
        # `DoneEvent` summary and the answering path, none of which know that a
        # plan has steps; resetting them per step would make the task look as
        # though only its last step had happened.
        self._step_modified_files = []
        self._step_summaries = []
        self._current_step = None
        self._terminal_success_count = 0
        # Review and merge both read only commits, so a step whose commit failed
        # has done work that neither will see. Remembering which ones is what
        # lets the final summary say so instead of implying it all landed.
        self._uncommitted_steps = []
        self._consecutive_commit_failures = 0
        self._pending_changes = []
        # Both survive from one step to the next: a test command the user had to
        # supply should be asked for once, not once per step.
        self._passed_commands = []
        self._test_command = ''
        self._verification_waiver = ''

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
    def terminal_success_count(self) -> int:
        """
        Count terminal tool calls that completed successfully in this session.

        The streaming execution loop snapshots this value before each coder run,
        then stops that run only when a terminal tool grows the count. Refused
        terminal-tool calls never reach the recording methods and therefore do
        not affect it.

        :returns: The monotonically increasing terminal-success count.
        """
        return self._terminal_success_count

    @property
    def completion_summary(self) -> str | None:
        """
        The summary the agent supplied when it signalled completion.

        :returns: The summary text, or None if completion has not been signalled.
        """
        return self._completion_summary

    @property
    def changes_are_required(self) -> bool | None:
        """
        Whether this task was classified as needing at least one file changed.

        None until the classification has been made. The completion gate treats
        None exactly as it treats True: an unclassified session has not earned
        the right to finish without changing anything, and defaulting the other
        way would turn a missing classification into a free pass.

        :returns: The verdict, or None if the task has not been classified.
        """
        return self._changes_are_required

    def record_classification(self, changes_are_required: bool) -> None:
        """
        Record what the classifier decided about this task.

        Set once, before any work begins, and read by the completion gate. It is
        recorded here rather than passed to the tool because the gate runs from
        inside a tool call, where the session is the only thing in reach.

        :param changes_are_required: True if the task needs at least one file changed.
        """
        self._changes_are_required = changes_are_required

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

        :returns: True if completion was accepted for a task needing no changes.
        """
        return self._completed_without_changes

    def accept_completion_without_changes(self, summary: str) -> str:
        """
        Accept completion for a task classified as requiring no file changes.

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
        self._terminal_success_count += 1
        return cleaned

    @property
    def step_summaries(self) -> Sequence[str]:
        """
        What each finished step reported, oldest first.

        :returns: The recorded step summaries.
        """
        return tuple(self._step_summaries)

    @property
    def current_step(self) -> str | None:
        """
        The step being worked on, if execution has begun.

        :returns: The current step's title, or None.
        """
        return self._current_step

    @property
    def step_has_modified_files(self) -> bool:
        """
        Whether the current step has changed anything yet.

        Distinct from `has_modified_files`, which is true for the rest of the
        task once any step has changed something.

        :returns: True if this step has modified at least one file.
        """
        return bool(self._step_modified_files)

    @property
    def uncommitted_steps(self) -> Sequence[str]:
        """
        Steps whose changes were never committed.

        :returns: The titles, oldest first.
        """
        return tuple(self._uncommitted_steps)

    @property
    def consecutive_commit_failures(self) -> int:
        """
        How many commits have failed in a row.

        Consecutive rather than total: a lock contended once and then released
        is not the same problem as an identity that will never be configured,
        and only the second is worth stopping the session over.

        :returns: The current run of failures.
        """
        return self._consecutive_commit_failures

    def record_commit_failure(self, step: str) -> int:
        """
        Record that a step's changes could not be committed.

        :param step: The step whose commit failed.
        :returns: How many commits have now failed in a row.
        """
        if step not in self._uncommitted_steps:
            self._uncommitted_steps.append(step)
        self._consecutive_commit_failures += 1
        return self._consecutive_commit_failures

    def forgive_commit_failures(self) -> None:
        """
        Clear the run of failures without claiming anything was committed.

        For a user who has been asked and chose to carry on regardless: the
        question should not be repeated at every remaining step, but the stranded
        steps stay recorded so the final summary still reports them.
        """
        self._consecutive_commit_failures = 0

    def record_commit_success(self) -> None:
        """
        Record that a commit succeeded.

        Clears every stranded step, not just the latest one, and takes no step
        argument for that reason: a commit stages everything outstanding, so a
        step whose own commit failed earlier has landed after all.
        """
        self._consecutive_commit_failures = 0
        self._uncommitted_steps.clear()

    @property
    def pending_change_summaries(self) -> Sequence[str]:
        """
        The per-edit descriptions gathered since the last commit.

        Collected so a step's single commit can say what it contains, now that
        the individual writes no longer each make one.

        :returns: The descriptions, oldest first.
        """
        return tuple(self._pending_changes)

    def note_pending_change(self, description: str) -> None:
        """
        Record what one edit did, for the step's eventual commit message.

        :param description: A one-line description of the edit.
        """
        cleaned: str = description.strip()
        if cleaned and cleaned not in self._pending_changes:
            self._pending_changes.append(cleaned)

    def clear_pending_changes(self) -> None:
        """
        Forget the gathered edit descriptions, once they have been committed.
        """
        self._pending_changes.clear()

    @property
    def test_command(self) -> str:
        """
        The verification command established for this plan, if any.

        :returns: The command, or an empty string if none was established.
        """
        return self._test_command

    @property
    def verification_waiver(self) -> str:
        """
        Why verification is not required, if it is not.

        :returns: The reason, or an empty string while verification still applies.
        """
        return self._verification_waiver

    @property
    def verification_required(self) -> bool:
        """
        Whether a change still has to be verified before a step can finish.

        :returns: True if verification is required.
        """
        return self._sandbox_available and not self._verification_waiver

    @property
    def passed_commands(self) -> Sequence[str]:
        """
        Commands that have actually been run and exited zero this session.

        :returns: The commands, in the order they first passed.
        """
        return tuple(self._passed_commands)

    def note_command_passed(self, command: str) -> None:
        """
        Record that a command was run and succeeded.

        :param command: The command, exactly as it was run.
        """
        cleaned: str = command.strip()
        if cleaned and cleaned not in self._passed_commands:
            self._passed_commands.append(cleaned)

    def record_test_command(self, command: str) -> str:
        """
        Establish the command that verifies work in this repository.

        Kept for the rest of the plan so that a command the user had to be asked
        for is asked for once rather than once per step.

        Only a command that has already been run and passed may be recorded, and
        it must match what was run exactly. Left unchecked, an agent will record
        a plausible-looking command it never tried — `npm install && vite build
        && npm run typecheck && npm run test` after only running `node -v` — and
        every later step then inherits a command that may not exist, reinstalls
        the world on each use, or names scripts the project does not define. The
        model is the one who decides which command is the right one; Loom's part
        is to insist it be a command that demonstrably runs.

        :param command: The command that was found to work.
        :returns: The command, as recorded.
        :raises ValueError: If the command is empty, or has not been run and passed.
        """
        cleaned: str = command.strip()
        if not cleaned:
            raise ValueError('The test command cannot be empty.')
        if cleaned not in self._passed_commands:
            raise ValueError(
                f'"{cleaned}" has not been run and passed in this session, so it cannot be '
                'recorded as the verification command. Run it with run_test_command first, '
                'exactly as you want it recorded, and record it once it passes.'
            )
        self._test_command = cleaned
        return cleaned

    def clear_test_command(self) -> None:
        """
        Forget the established command, so the next step works it out again.

        For a command that turned out to be wrong rather than a test that
        genuinely failed — a monorepo where one area tests differently, say.
        Silently reusing it would keep producing failures that look like the
        code's fault.
        """
        self._test_command = ''

    def waive_verification(self, reason: str) -> str:
        """
        Stop requiring verification for the rest of the plan.

        For a repository with no tests to run, or a user who has said to skip
        them. Waived rather than weakened, exactly as the missing-sandbox case
        is: the requirement still applies wherever it can be met, and the reason
        is recorded so the summary can say why nothing was verified.

        :param reason: Why verification no longer applies.
        :returns: The reason, as recorded.
        :raises ValueError: If no reason is given.
        """
        cleaned: str = reason.strip()
        if not cleaned:
            raise ValueError('A reason is required to skip verification.')
        self._verification_waiver = cleaned
        return cleaned

    def begin_step(self, title: str) -> str:
        """
        Start a step, clearing the per-step bookkeeping.

        The verification state is deliberately not reset. A step that changes
        nothing inherits the last verification, which is correct: the working
        tree it is looking at really was verified, and demanding a fresh command
        for an investigative step would make it unfinishable.

        :param title: The step being started.
        :returns: The title, as recorded.
        """
        self._step_modified_files = []
        self._current_step = title
        return title

    def record_step_completion(self, summary: str) -> str:
        """
        Record what a step accomplished.

        :param summary: The agent's account of the step.
        :returns: The recorded summary.
        :raises ValueError: If the summary is empty.
        """
        cleaned: str = summary.strip()
        if not cleaned:
            raise ValueError('The step summary cannot be empty.')
        self._step_summaries.append(cleaned)
        self._terminal_success_count += 1
        return cleaned

    def request_step_completion(self, summary: str) -> str:
        """
        Record a step as finished, enforcing the same gate as the whole task.

        The checks mirror `request_completion` deliberately: a step that changed
        a file without verifying it is exactly the failure the task-wide gate
        exists to catch, and catching it a step later means the agent has moved
        on before hearing about it.

        :param summary: The agent's account of the step.
        :returns: The recorded summary.
        :raises ValueError: If the step changed a file it has not verified, or the summary is empty.
        """
        if not self._step_modified_files:
            raise ValueError(
                'This step has not modified any file. If the step calls for a change, make it, '
                'verify it with run_test_command, and then complete the step. If the step was '
                'only ever investigative, use accept_step_without_changes instead.'
            )
        if self.verification_required and not self._verified_since_modification:
            raise ValueError(
                f'The last change to {self._modified_files[-1]} has not been verified yet. '
                'Run a relevant test, build or lint command with run_test_command, '
                'and only complete this step once it passes.'
            )
        if (
            self.verification_required
            and not self._last_verification_passed
            and not self.verification_allowance_exhausted
        ):
            remaining: int = self._max_failed_verifications - self._failed_verifications
            raise ValueError(
                'The most recent verification command failed. Diagnose the failure, fix it, '
                'and re-run the command until it passes before completing this step. '
                f'({remaining} further failed attempt(s) will release this requirement, '
                'for repositories whose tests were already failing.)'
            )
        return self.record_step_completion(summary)

    def accept_step_completion_without_changes(self, summary: str) -> str:
        """
        Record an investigative step as finished, having changed nothing.

        Separate from `accept_completion_without_changes`, which settles the
        whole task and would end it. A plan can legitimately contain a step that
        only reads — now more than before, since the planner can read the
        repository and will write such steps.

        :param summary: What the step found.
        :returns: The recorded summary.
        :raises ValueError: If the step did modify files, or the summary is empty.
        """
        if self._step_modified_files:
            raise ValueError(
                'This step modified '
                f'{", ".join(self._step_modified_files)}, so it did change something. '
                'Verify the change and complete the step normally instead.'
            )
        return self.record_step_completion(summary)

    def record_modification(self, path: str) -> int:
        """
        Record that the agent created or edited a file.

        :param path: The repository-relative path that was modified.
        :returns: The total number of modifications recorded so far.
        """
        self._step_modified_files.append(path)
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

    def record_rollup_completion(self, summary: str) -> str:
        """
        Settle the task's summary from the steps that were carried out.

        The per-step gate has already been applied to each step, so applying the
        task-wide gate again here would re-litigate decisions that were made when
        the agent was still in a position to act on them. What this records is
        the account of the work, not a fresh claim that it was done.

        The verification warnings `request_completion` appends still apply: they
        describe the state the working tree ended in, which is what the reviewer
        needs to know regardless of how the summary was assembled.

        :param summary: The combined account of every step.
        :returns: The recorded summary.
        :raises ValueError: If the summary is empty.
        """
        cleaned: str = summary.strip()
        if not cleaned:
            raise ValueError('The completion summary cannot be empty.')

        if self._uncommitted_steps:
            # The reviewer and the merge both read only commits, so this is not a
            # cosmetic note: without it the summary would describe work that
            # neither of them can see as though it had landed.
            stranded: str = ', '.join(self._uncommitted_steps)
            cleaned = (
                f'{cleaned}\n\nWARNING: the changes for {stranded} could not be committed. '
                'They are in the working tree but not in the history, so they are absent from '
                'the diff under review and will not be merged.'
            )

        if self._verification_waiver and self._modified_files:
            # Says why nothing was verified rather than leaving the reader to
            # infer it from an absence, which reads as an oversight.
            cleaned = (
                f'{cleaned}\n\nWARNING: these changes were not verified. '
                f'{self._verification_waiver}'
            )

        if not self._sandbox_available:
            cleaned = (
                f'{cleaned}\n\nWARNING: no sandbox was available in this session, so no command '
                'could be run and these changes were not verified at all; review the diff '
                'carefully.'
            )
        elif self.verification_required and self._modified_files and not self._last_verification_passed:
            cleaned = (
                f'{cleaned}\n\nWARNING: the last verification command did not pass after '
                f'{self._failed_verifications} failed attempt(s). The changes were not verified '
                'green; review the diff carefully.'
            )

        self._completion_summary = cleaned
        self._terminal_success_count += 1
        return self._completion_summary

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
        if self.verification_required and not self._verified_since_modification:
            raise ValueError(
                f'The last change to {self._modified_files[-1]} has not been verified yet. '
                'Run a relevant test, build or lint command with run_test_command, '
                'and only call complete_task once it passes.'
            )
        if (
            self.verification_required
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
        if self._verification_waiver and self._modified_files:
            # Says why nothing was verified rather than leaving the reader to
            # infer it from an absence, which reads as an oversight.
            cleaned = (
                f'{cleaned}\n\nWARNING: these changes were not verified. '
                f'{self._verification_waiver}'
            )

        if not self._sandbox_available:
            cleaned = (
                f'{cleaned}\n\nWARNING: no sandbox was available in this session, so no command '
                'could be run and these changes were not verified at all; review the diff '
                'carefully.'
            )
        elif self.verification_required and not self._last_verification_passed:
            cleaned = (
                f'{cleaned}\n\nWARNING: completed with a failing verification command after '
                f'{self._failed_verifications} failed attempt(s). The changes were not verified '
                'green; review the diff carefully.'
            )

        self._completion_summary = cleaned
        return self._completion_summary
