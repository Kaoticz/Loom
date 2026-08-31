from __future__ import annotations

from pydantic_ai import RunContext

from services.browser_tooling import browser_automation_refusal
from services.docker_sandbox import CommandResult
from services.tool_reporting import report_tool

from .deps import AgentDeps


async def run_test_command(ctx: RunContext[AgentDeps], command: str) -> str:
    """
    Run a test, build, or lint command inside the sandbox and return its output.
    :param command: The shell command to run inside the sandbox, for example "pytest -q tests/".
    """
    async with report_tool(ctx.deps.events, 'run_test_command', {'command': command}) as report:
        if not command.strip():
            return report.fail('No command was given. Pass the test, build or lint command to '
                               'run, for example "pytest -q tests/".')

        refusal: str | None = browser_automation_refusal(command)
        if refusal is not None:
            # Refused before reaching the sandbox rather than after: installing
            # a browser stack is slow and leaves the sandbox changed, so letting
            # it run and complaining afterwards would achieve neither.
            return report.fail(refusal)

        try:
            result: CommandResult = await ctx.deps.docker.run_test_command(command)
        except Exception as error:
            # The agent needs to know the sandbox is absent, not just see a
            # runtime error, or it will keep retrying a command that can never
            # run instead of adapting.
            return report.fail(
                'The sandbox is not available, so no command can be run in this session. '
                'Do not retry: continue making changes, and say clearly in your completion '
                f'summary that they could not be verified. (Reason: {error})'
            )

        if not result.success and _looks_uninstalled(result):
            # Deliberately not recorded as a failed verification. Nothing was
            # verified: the command never ran. Counting it would spend the
            # allowance that exists for a repository whose tests were already
            # failing, and once that is spent the completion gate releases — so
            # a missing tool would quietly become permission to stop verifying.
            return report.fail(
                f'The command did not run: the tooling it needs is not installed.\n\n'
                f'{result.output}\n\n'
                'Install it inside the sandbox and run the command again. Installing '
                'dependencies is ordinary work, not a detour. Look at what the repository '
                'says about itself — a lockfile, a manifest, a contributor guide — and use '
                'whatever it calls for. Do not move on to a different command instead, and '
                'do not treat this as a failing test.'
            )

        ctx.deps.session.record_verification(result.success)
        if result.success:
            # Remembered so `record_test_command` can insist on a command that
            # demonstrably runs, rather than one that merely looks right.
            ctx.deps.session.note_command_passed(command)
        status: str = 'passed' if result.success else f'FAILED with exit code {result.exit_code}'
        output: str = f'Command {status}.\n\n{result.output}'
        return report.succeed(output) if result.success else report.fail(output)


MISSING_TOOL_EXIT_CODES: frozenset[int] = frozenset({126, 127})
"""
Shell exit codes meaning the command could not be run at all.

127 is "command not found" and 126 is "found but not executable". Neither says
anything about the code under test.
"""

MISSING_TOOL_MESSAGES: tuple[str, ...] = (
    'command not found',
    'no such file or directory',
    'no module named',
    'is not recognized as an internal',
    'executable file not found',
)
"""
What a missing tool says when the exit code alone does not give it away.

`python -m pytest` with pytest absent exits 1, not 127, and says "No module
named pytest" — indistinguishable from a real failure by exit code, and the
single most likely way this arises in a Python repository.
"""


def _looks_uninstalled(result: CommandResult) -> bool:
    """
    Decide whether a command failed because its tooling is missing.

    Kept narrow on purpose. Treating a real test failure as a missing tool would
    tell the agent to install something instead of fixing a genuine defect, so
    the signal has to be an exit code that means "did not run" or a message the
    shell and interpreter only produce when something is absent.

    :param result: The outcome of the command.
    :returns: True if the command never really ran.
    """
    if result.exit_code in MISSING_TOOL_EXIT_CODES:
        return True

    output: str = result.output.lower()
    return any(message in output for message in MISSING_TOOL_MESSAGES)


async def record_test_command(ctx: RunContext[AgentDeps], command: str) -> str:
    """
    Remember the command that verifies work in this repository.

    Call this once you have found a command that actually runs — after setting
    up an environment, or after the user told you what to use. Later steps are
    given it, so nobody has to work it out or be asked again.

    If a remembered command later fails in a way that says the command itself is
    wrong rather than the code, call `forget_test_command` instead of persisting
    with it.
    :param command: The verification command, exactly as you already ran it with run_test_command and saw it pass. A command you have not run cannot be recorded.
    """
    async with report_tool(ctx.deps.events, 'record_test_command', {'command': command}) as report:
        try:
            recorded: str = ctx.deps.session.record_test_command(command)
        except ValueError as error:
            return report.fail(f'record_test_command was rejected: {error}')
        return report.succeed(f'Verification command for this plan: {recorded}')


async def forget_test_command(ctx: RunContext[AgentDeps], reason: str) -> str:
    """
    Discard the remembered verification command because it is wrong.

    For a command that does not fit the work in hand — a different area of a
    monorepo, a suite that does not cover what you changed — not for a test that
    ran correctly and reported a real failure. Say which of the two you believe
    happened; if you are not sure, you are probably looking at a real failure.
    :param reason: Why the remembered command does not fit the work in hand.
    """
    async with report_tool(ctx.deps.events, 'forget_test_command', {'reason': reason}) as report:
        ctx.deps.session.clear_test_command()
        return report.succeed(
            f'Forgotten. Work out how to verify this step, or ask the user. Reason: {reason}'
        )


async def skip_verification(ctx: RunContext[AgentDeps], reason: str) -> str:
    """
    Stop requiring verification for the rest of this plan.

    Only for two situations: the repository genuinely has no tests to run, or
    the user has told you to skip them. Not for a suite you could not get
    working and not for tests that fail — those are for `ask_user`.

    Every summary from here on says the work was unverified, and gives your
    reason, so make it accurate.
    :param reason: Why verification does not apply, for example that the repository has no tests. This is repeated in every summary from now on.
    """
    async with report_tool(ctx.deps.events, 'skip_verification', {'reason': reason}) as report:
        try:
            recorded: str = ctx.deps.session.waive_verification(reason)
        except ValueError as error:
            return report.fail(f'skip_verification was rejected: {error}')
        return report.succeed(f'Verification is no longer required for this plan: {recorded}')
