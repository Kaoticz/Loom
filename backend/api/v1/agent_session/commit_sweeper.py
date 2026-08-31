from __future__ import annotations

from typing import TYPE_CHECKING

from services.event_bus import CommitEvent, WarningEvent
from services.git_manager import CommitResult
from services.user_questions import StopRequested, UserQuestioner

if TYPE_CHECKING:
    from .agent_session import AgentSession


MAX_CONSECUTIVE_COMMIT_FAILURES: int = 3
"""
How many commits may fail in a row before the user is asked what to do.

Three rather than one: a single failure is often an index lock held for a moment
and clears by itself, and the retry inside the commit already covers that. Three
in a row is a configuration problem, and every further step past it adds work
that neither the review nor the merge can see.
"""

COMMIT_FAILURE_COMMITTED: str = 'I have committed the changes myself'
COMMIT_FAILURE_CONTINUE: str = 'Carry on without committing'
COMMIT_FAILURE_STOP: str = 'Stop the session'

COMMIT_FAILURE_OPTIONS: tuple[str, ...] = (
    COMMIT_FAILURE_COMMITTED,
    COMMIT_FAILURE_CONTINUE,
    COMMIT_FAILURE_STOP,
)
"""
What the user may do about commits that keep failing.

Committing by hand is offered first because it is usually the quickest fix and
Loom has no business guessing at someone's git configuration. The claim is
verified rather than trusted, so choosing it wrongly costs another question
rather than a lost step.
"""


class CommitSweeper:
    """
    Preserve committed visibility for work left by execution steps.

    :param session: The session that owns branch, event, and commit state.
    """

    _session: AgentSession

    def __init__(self, session: AgentSession) -> None:
        """
        Bind the sweeper to one session.

        :param session: The owning session.
        """
        self._session = session

    async def sweep_uncommitted(self) -> None:
        """
        Commit anything left in the working tree once every step has run.

        Steps commit their own work, but a step that ended without reporting
        completion does not, and neither does one whose commit failed. Left
        alone, that work would reach neither the reviewer nor the merge, both of
        which read only commits — the exact loss this phase exists to prevent.
        """
        if not self._session._branch_created:
            return
        if not await self._session._services.git.has_uncommitted_changes():
            return

        result: CommitResult = await self._session._services.git.commit_changes_with_retry(
            'Remaining changes not committed by any step'
        )

        if result.failed:
            await self._session._events.emit(
                WarningEvent(
                    content=(
                        f'Changes are left uncommitted and could not be swept up: {result.error}. '
                        'They will not appear in the review or the merge.'
                    )
                )
            )
            self._session._session.record_commit_failure(
                self._session._session.current_step or 'Remaining changes'
            )
            return

        if result.commit is None:
            return

        await self._session._events.emit(
            CommitEvent(
                commit_hash=result.commit.commit_hash,
                message=result.commit.message,
                files_changed=result.commit.files_changed,
            )
        )
        await self._session._events.emit(
            WarningEvent(
                content=(
                    'Some changes were not committed by the step that made them, so they have '
                    'been committed together to keep them visible to the review.'
                )
            )
        )

    async def resolve_commit_failures(self) -> None:
        """
        Stop and ask once commits have failed repeatedly.

        :raises StopRequested: If the user chooses to stop.
        """
        while self._session._session.consecutive_commit_failures >= MAX_CONSECUTIVE_COMMIT_FAILURES:
            failures: int = self._session._session.consecutive_commit_failures
            stranded: str = ', '.join(self._session._session.uncommitted_steps) or 'the current step'
            questioner: UserQuestioner | None = self._session._deps.questioner

            if questioner is None:
                await self._session._events.emit(
                    WarningEvent(
                        content=(
                            f'{failures} commits in a row have failed and there is no one to ask '
                            f'how to proceed. Work for {stranded} is uncommitted and will not be '
                            'reviewed or merged.'
                        )
                    )
                )
                return

            reply, _ = await questioner.ask(
                f'{failures} commits in a row have failed. Work for {stranded} is sitting '
                'uncommitted, and neither the review nor the merge can see it. How should '
                'Loom proceed?',
                list(COMMIT_FAILURE_OPTIONS),
            )
            answer: str = reply.strip().lower()

            if answer == COMMIT_FAILURE_STOP.lower() or answer == 'stop':
                raise StopRequested()

            # Keep the existing free-text classification intact. Phase 4 replaces
            # this fragile boundary without changing which answers are accepted.
            claimed_commit: bool = answer == COMMIT_FAILURE_COMMITTED.lower() or (
                'commit' in answer and 'without' not in answer and 'not ' not in answer
            )

            if claimed_commit:
                if await self._session._services.git.has_uncommitted_changes():
                    await self._session._events.emit(
                        WarningEvent(
                            content=(
                                'The working tree still has uncommitted changes, so nothing was '
                                'committed. Asking again.'
                            )
                        )
                    )
                    continue
                self._session._session.record_commit_success()
                self._session._session.clear_pending_changes()
                await self._session._announce('The changes were committed. Continuing with the plan.')
                return

            self._session._session.forgive_commit_failures()
            await self._session._events.emit(
                WarningEvent(
                    content=(
                        'Continuing without committing, at your request. Uncommitted work will '
                        'not appear in the review or the merge.'
                    )
                )
            )
            return
