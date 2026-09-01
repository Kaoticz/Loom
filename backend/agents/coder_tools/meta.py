from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic_ai import RunContext

from services.events import CommitEvent, WarningEvent
from services.git_manager import CommitResult
from services.session_context import SessionContext
from services.tool_reporting import report_tool
from services.user_questions import MAX_SUGGESTED_ANSWERS, UserQuestioner

from .deps import AgentDeps


async def ask_user(
    ctx: RunContext[AgentDeps], question: str, options: list[str] | None = None
) -> str:
    """
    Ask the user a question and wait for their answer.

    Use this when the task cannot be carried out correctly without something
    only the user can tell you: which of several things they meant, which
    approach they want, or a decision the repository does not settle. Suggest up
    to six answers if you can; the interface always offers the user the further
    option of typing their own, so a question with no suggestions is fine too.

    Do not use it to ask permission for work you have already been asked to do,
    and do not use it in place of reading the repository to find something out.
    :param question: The question to put to the user, in one or two sentences.
    :param options: Up to six suggested answers. The user can always type their own instead, so this may be left empty.
    """
    async with report_tool(ctx.deps.events, 'ask_user', {'question': question}) as report:
        questioner: UserQuestioner | None = ctx.deps.questioner
        if questioner is None:
            return report.fail(
                'There is no user to ask in this session, so this question cannot be put to '
                'anyone. Continue with the most reasonable assumption and say in your summary '
                'what you assumed.'
            )

        asked: str = question.strip()
        if not asked:
            return report.fail('ask_user was rejected: the question was empty.')

        reply, truncated = await questioner.ask(asked, options or [])
        if truncated:
            return report.succeed(
                f'Only the first {MAX_SUGGESTED_ANSWERS} suggested answers were shown; the rest '
                f'were dropped. Suggest no more than {MAX_SUGGESTED_ANSWERS} next time. '
                f'The user replied: {reply}'
            )
        return report.succeed(f'The user replied: {reply}')


async def complete_task(ctx: RunContext[AgentDeps], summary: str) -> str:
    """
    Signal that all the work for the task is finished.

    Call this exactly once, as the final action of your execution phase, with a
    summary of what you did. If the task required changing files, do not call it
    before you have modified at least one and verified the change with
    run_test_command. If the task only ever called for an answer — a summary, an
    explanation, an assessment — call it with that answer as the summary.
    :param summary: What you did across the whole task, in a sentence or two.
    """
    session: SessionContext = ctx.deps.session

    async with report_tool(ctx.deps.events, 'complete_task', {}) as report:
        if not session.has_modified_files:
            # Whether finishing without a change is legitimate was settled before
            # the task began, so there is no judgment to make here: a task
            # classified as needing changes and offering none is simply refused.
            if session.changes_are_required is not False:
                return report.fail(
                    'complete_task was rejected: this task was judged to require changing at '
                    'least one file, and nothing has been modified. Make the changes the task '
                    'requires, verify them with run_test_command, and only then call '
                    'complete_task.'
                )

            try:
                recorded_answer: str = session.accept_completion_without_changes(summary)
            except ValueError as error:
                return report.fail(f'complete_task was rejected: {error}')
            return report.succeed(f'Task marked complete with no file changes. Answer recorded: {recorded_answer}')

        try:
            recorded: str = session.request_completion(summary)
        except ValueError as error:
            return report.fail(f'complete_task was rejected: {error}')

        modified: str = ', '.join(session.modified_files)
        # Still commits. This is the whole-task path — an unparsed plan, or an
        # answering task with nothing to commit — and review reads only commits,
        # so skipping it here would leave the work invisible.
        note: str = await _commit_step(ctx, recorded)
        return report.succeed(
            f'Task marked complete. Summary recorded: {recorded} '
            f'(files modified: {modified}){note}'
        )


async def complete_step(ctx: RunContext[AgentDeps], summary: str) -> str:
    """
    Signal that the current plan step is finished.

    Call this once the step you were given is done and its changes are verified.
    Say what you actually did, in a sentence or two — the next step is told your
    summary rather than shown your work, so anything you leave out is lost to it.

    This finishes the step, not the task. Do not use it to report the whole plan
    as done, and do not call it for work belonging to a later step.
    :param summary: What you did for this step, in a sentence or two. The next step is told this and is not shown your work.
    """
    async with report_tool(ctx.deps.events, 'complete_step', {'summary': summary}) as report:
        session: SessionContext = ctx.deps.session
        try:
            recorded: str = session.request_step_completion(summary)
        except ValueError as error:
            return report.fail(f'complete_step was rejected: {error}')

        note: str = await _commit_step(ctx, recorded)
        return report.succeed(f'Step marked complete. Summary recorded: {recorded}{note}')


async def accept_step_without_changes(ctx: RunContext[AgentDeps], summary: str) -> str:
    """
    Finish a step that required no change to any file.

    Only for a step that was investigative from the start — finding where
    something lives, or confirming what the code already does. Report what you
    found, so the steps after this one can use it.

    Do not use it to get past a step you could not manage. If the step called
    for a change, make the change.
    :param summary: What you found out, in a sentence or two, for the steps that follow this one.
    """
    arguments: dict[str, Any] = {'summary': summary}

    async with report_tool(ctx.deps.events, 'accept_step_without_changes', arguments) as report:
        session: SessionContext = ctx.deps.session
        try:
            recorded: str = session.accept_step_completion_without_changes(summary)
        except ValueError as error:
            return report.fail(f'accept_step_without_changes was rejected: {error}')

        return report.succeed(f'Step marked complete with no file changes. Recorded: {recorded}')


async def _commit_step(ctx: RunContext[AgentDeps], summary: str) -> str:
    """
    Commit everything the finished step changed, as one commit.

    A failure is reported back to the agent and announced as a warning, not
    raised. The changes are on disk either way, so killing the session would
    lose the work along with the explanation — and the next step's commit stages
    everything outstanding, so a failure here may well be made good later.

    :param ctx: The agent run context.
    :param summary: The step's recorded summary, used as the commit subject.
    :returns: A note to append to the tool's reply, empty if the commit was clean.
    """
    session: SessionContext = ctx.deps.session
    # Falls back to the summary's first line rather than a fixed word, because
    # this also runs for a whole task, which has no step title. A commit subject
    # reading "Plan step" would tell a reader nothing at all.
    step: str = session.current_step or summary.strip().splitlines()[0][:72]
    body: Sequence[str] = session.pending_change_summaries
    message: str = f'{step}\n\n{summary}'
    if body:
        message += '\n\n' + '\n'.join(f'- {line}' for line in body)

    result: CommitResult = await ctx.deps.git.commit_changes_with_retry(message)

    if result.failed:
        failures: int = session.record_commit_failure(step)
        await ctx.deps.events.emit(
            WarningEvent(
                content=(
                    f'Could not commit the changes for "{step}" after two attempts: '
                    f'{result.error}. The changes are still in the working tree. '
                    'Carrying on; they will be included in the next commit that succeeds.'
                )
            )
        )
        return (
            f' (warning: the commit failed - {result.error}. Your changes are safe on disk '
            f'and will be picked up by a later commit. This is failure {failures} in a row; '
            'do not try to fix git yourself, carry on with the plan.)'
        )

    session.record_commit_success()
    session.clear_pending_changes()

    if result.commit is None:
        return ''

    await ctx.deps.events.emit(
        CommitEvent(
            commit_hash=result.commit.commit_hash,
            message=result.commit.message,
            files_changed=result.commit.files_changed,
        )
    )
    return f' (committed as {result.commit.commit_hash[:8]})'
