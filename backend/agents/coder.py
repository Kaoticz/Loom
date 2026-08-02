from __future__ import annotations

import os
from logging import Logger, getLogger
from typing import Any, ClassVar

from httpx import AsyncClient, HTTPError, Response
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, RunContext
from pydantic_ai.agent import AgentRunResult

from services.code_search import CodeSearchService, LexicalMatch, SymbolDefinition
from services.docker_sandbox import CommandResult, DockerSandboxManager
from services.embedding import EmbeddingService
from services.event_bus import CommitEvent, DiffEvent, EventBus
from services.git_manager import CommitInfo, GitManagerService
from services.tool_reporting import (
    ToolReport,
    describe_file_read,
    describe_symbol_lookup,
    report_tool,
)
from services.file_editor import EditResult, FileEditorService
from agents.adjudicator import (
    AdjudicationResult,
    adjudicator_agent,
    build_adjudication_prompt,
)
from services.model_resolution import resolve_model
from services.session_context import SessionContext

logger: Logger = getLogger(__name__)

MAX_COMMIT_SUMMARY_LENGTH: int = 72
WEB_SEARCH_RESULT_LIMIT: int = 5
WEB_SEARCH_SNIPPET_LENGTH: int = 200

CODER_SYSTEM_PROMPT: str = """You are Loom, an expert AI software engineer working inside a real Git repository.
You operate in a strict ReAct loop: Plan, Action, Observation. Follow the rules below exactly.

# 1. Plan format

When you are asked to produce a plan, respond with a numbered list of steps and nothing else.

- Every step must be a single, concrete, actionable instruction (for example
  "Add a `timeout` parameter to `run_test_command` in backend/services/docker_sandbox.py").
- Do not bundle several changes into one step.
- Do not include commentary, headings, or code blocks around the list.
- The final line of your plan must contain only the word PLAN_COMPLETE.

Example:

1. Read backend/services/git_manager.py to locate the merge_branch method.
2. Replace the hardcoded 'main' reference with the repository's default branch.
3. Run `python -m pytest tests/test_git_manager.py` to verify the change.
PLAN_COMPLETE

# 2. Understand before you change

Before modifying any code you must first understand it.

- Use `search_code` or `search_code_lexical` to locate the relevant area of the codebase.
- Use `get_symbol` to look up the definition of a class or function by name.
- Use `read_file` to read the exact current contents of any file you intend to change.

Never call `edit_file` on a file you have not read in this session. The search block you
pass to `edit_file` must be copied verbatim from what `read_file` returned, or the edit
will not match.

# 3. Verify every change (hard rule)

After every single call to `edit_file` or `create_file`, you MUST immediately call
`run_test_command` with a relevant test, build, or lint command before moving on to the
next step. This is a hard rule, not a suggestion. Never make two file modifications in a
row without a verification step between them.

If you do not know which test to run, run the narrowest command that would catch a
mistake in the file you just touched (for example the module's test file, a syntax check,
or a linter on that path).

# 4. Recover from errors, do not skip them

If a tool call fails or a test does not pass:

1. State what went wrong and why, in one or two sentences.
2. Propose a specific fix.
3. Apply the fix and re-run the verification command.

Never silently skip a failing step, and never move to the next step while a verification
command you just ran is still failing.

# 5. Finishing

When, and only when, the task is genuinely done, call the `complete_task` tool.

Most tasks require changing files. For those:

- Pass a summary of everything you changed.
- You must not call `complete_task` before you have made at least one file modification and
  verified it with a passing `run_test_command`.
- The exception is a session with no sandbox. If `run_test_command` reports that the sandbox is
  unavailable, no command can be run and none will become runnable later. Do not keep retrying
  it and do not treat the missing verification as a reason to carry on working. Finish the task,
  then complete it, saying plainly in your summary that the changes are unverified.

Some tasks require changing nothing at all: summarising code, explaining how something works,
assessing a design, or investigating whether a problem exists where the answer turns out to be
no. For those:

- Pass your answer itself as the summary. It is the whole product of the task, so make it
  complete: the reader sees this and nothing else.
- Do not invent a file change to satisfy the rule above. A task that asked a question is
  finished when the question is answered.
- Do not use this to escape work you found difficult. Whether the task genuinely required no
  changes is checked, and if it did require them you will be told to carry on.

In both cases, `complete_task` is the final action of your execution phase. After calling it,
stop. Do not produce another plan, do not call more tools, and do not keep reasoning.

# 6. Style

Match the conventions already present in the codebase you are editing. Keep edits
surgical: change what the task requires and nothing else."""


class AgentDeps(BaseModel):
    """
    Dependencies made available to every Coder agent tool.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    docker: DockerSandboxManager
    search: CodeSearchService
    editor: FileEditorService
    git: GitManagerService
    embedding: EmbeddingService
    events: EventBus
    session: SessionContext


coder_agent: Agent[AgentDeps, str] = Agent(
    deps_type=AgentDeps,
    output_type=str,
    system_prompt=CODER_SYSTEM_PROMPT,
)


@coder_agent.tool
async def search_code(ctx: RunContext[AgentDeps], query: str) -> str:
    """
    Search the codebase for code related to a natural language query.

    Runs a semantic search over the indexed embeddings and a literal text search
    in parallel, then merges the two, so that both conceptual matches and exact
    keyword matches are covered.
    """
    async with report_tool(ctx.deps.events, 'search_code', {'query': query}) as report:
        return report.succeed(await _search_code(ctx, query))


async def _search_code(ctx: RunContext[AgentDeps], query: str) -> str:
    """
    Run the merged semantic and lexical search.

    :param ctx: The agent run context.
    :param query: The natural language query.
    :returns: The rendered search results.
    """
    embedding: list[float] = await ctx.deps.embedding.embed(query)

    sections: list[str] = []
    seen: set[str] = set()
    notices: list[str] = []

    semantic: list[Any] = []
    try:
        semantic = await ctx.deps.search.search_code_semantic(
            embedding, ctx.deps.embedding.embedding_model
        )
    except Exception as error:
        # A vector backend failure must not cost us the lexical results, but it
        # is reported rather than swallowed so a broken index cannot go unnoticed.
        logger.warning('Semantic search failed for query %r: %s', query, error)
        notices.append(f'Semantic search is unavailable ({type(error).__name__}); showing text matches only.')

    for chunk in semantic:
        path: str = str(chunk.file.file_path)
        if path in seen:
            continue
        seen.add(path)
        sections.append(f'{path}:{chunk.start_line} (semantic)\n{chunk.chunk_text}')

    lexical: list[LexicalMatch] = await ctx.deps.search.search_code_lexical(query)
    for match in lexical:
        if match.file_path in seen:
            continue
        seen.add(match.file_path)
        sections.append(f'{match.render()} (lexical)')

    if not sections:
        notices.append(f'No code found for {query!r}. The repository may not be indexed yet.')
        return '\n'.join(notices)

    body: str = '\n\n'.join(sections)
    return '\n\n'.join([*notices, body]) if notices else body


@coder_agent.tool
async def search_code_lexical(ctx: RunContext[AgentDeps], keyword: str, limit: int = 10) -> str:
    """
    Search the codebase for an exact, case-insensitive piece of text.

    Use this when you know the literal string you are looking for, such as a
    function name, an error message, or a configuration key.
    """
    async with report_tool(
        ctx.deps.events, 'search_code_lexical', {'keyword': keyword, 'limit': limit}
    ) as report:
        matches: list[LexicalMatch] = await ctx.deps.search.search_code_lexical(keyword, limit=limit)
        if not matches:
            return report.succeed(f'No occurrences of {keyword!r} were found.')
        return report.succeed('\n\n'.join(match.render() for match in matches))


@coder_agent.tool
async def get_symbol(ctx: RunContext[AgentDeps], symbol: str) -> str:
    """
    Get the definition of a symbol (class, function).
    """
    async with report_tool(ctx.deps.events, 'get_symbol', {'symbol': symbol}) as report:
        found: list[SymbolDefinition] = await ctx.deps.search.get_symbol_definition(symbol)

        if not found:
            return report.succeed(
                f'No definition of {symbol!r} is in the index. It may be spelled differently, '
                'declared in a file that was not indexed, or not exist.'
            )

        # The agent gets the definitions; the reader gets a line saying how many.
        return report.succeed(
            '\n\n'.join(definition.render() for definition in found),
            observation=describe_symbol_lookup(symbol, len(found)),
        )


@coder_agent.tool
async def read_file(
    ctx: RunContext[AgentDeps],
    path: str,
    start_line: int | None = None,
    end_line: int | None = None
) -> str:
    """
    Read a file from the repository.

    Pass start_line and end_line to read only part of a large file. When you do,
    each line is prefixed with its line number for reference: those prefixes are
    not part of the file, so never include them in an edit_file search block.
    """
    arguments: dict[str, Any] = {'path': path}
    if start_line is not None:
        arguments['start_line'] = start_line
    if end_line is not None:
        arguments['end_line'] = end_line

    async with report_tool(ctx.deps.events, 'read_file', arguments) as report:
        try:
            contents: str = await ctx.deps.editor.read_file(path, start_line, end_line)
            # The agent gets the file; the reader gets a line saying it was read.
            return report.succeed(
                contents,
                observation=describe_file_read(path, contents, start_line, end_line),
            )
        except FileNotFoundError:
            return report.fail(f'File {path} does not exist.')
        except ValueError as error:
            return report.fail(f'Could not read {path}: {error}')


@coder_agent.tool
async def create_file(
    ctx: RunContext[AgentDeps],
    path: str,
    content: str,
    summary: str = ''
) -> str:
    """
    Create a new file in the repository and commit it.

    Give a short summary of why the file is being added, in a few words, so the
    commit history reads meaningfully.
    """
    async with report_tool(ctx.deps.events, 'create_file', {'path': path}) as report:
        original: str | None = await _read_base_version(ctx, path)
        try:
            await ctx.deps.editor.create_file(path, content)
        except ValueError as error:
            return report.fail(f'Could not create {path}: {error}')

        ctx.deps.session.record_modification(path)
        await _commit_and_report(ctx, path, _commit_message('Create', path, summary), original)
        return report.succeed(f'Created {path}.')


@coder_agent.tool
async def edit_file(
    ctx: RunContext[AgentDeps],
    path: str,
    search: str,
    replace: str,
    summary: str = ''
) -> str:
    """
    Edit a file surgically and commit changes.

    Give a short summary of what the edit accomplishes, in a few words, so the
    commit history reads meaningfully.
    """
    async with report_tool(ctx.deps.events, 'edit_file', {'path': path}) as report:
        original: str | None = await _read_base_version(ctx, path)
        try:
            result: EditResult = await ctx.deps.editor.edit_file(path, search, replace)
        except ValueError as error:
            return report.fail(f'Could not edit {path}: {error}')

        if not result.success:
            return report.fail(result.message)

        ctx.deps.session.record_modification(path)
        await _commit_and_report(ctx, path, _commit_message('Edit', path, summary), original)
        return report.succeed(result.message)


def _commit_message(verb: str, path: str, summary: str) -> str:
    """
    Build a commit message from the tool call's context.

    :param verb: What was done to the file, for example 'Edit' or 'Create'.
    :param path: Repository-relative path of the file.
    :param summary: The agent's short description of the change, if it gave one.
    :returns: The commit message.
    """
    cleaned: str = ' '.join(summary.split())[:MAX_COMMIT_SUMMARY_LENGTH].strip(' .')
    return f'{verb} {path}: {cleaned}' if cleaned else f'{verb} {path}'


async def _read_base_version(ctx: RunContext[AgentDeps], path: str) -> str | None:
    """
    Read a file as it stands on the branch the task was cut from.

    This runs before the file is modified, so the result does not depend on
    whether a separate task branch exists: committing to the base branch would
    otherwise overwrite the very version the diff is measured against.

    A misconfigured base branch is logged rather than raised, so that a problem
    with diff rendering cannot block the edit itself.

    A file that does not exist on the base branch reports an empty string, since
    that is the correct original side for a newly created file. None is reserved
    for "could not be determined", which suppresses the diff entirely: emitting
    an empty original in that case would show an edited file as newly created.

    :param ctx: The agent run context.
    :param path: Repository-relative path of the file.
    :returns: The contents on the base branch, '' if the file is new there, or None if unknown.
    """
    base: str = ctx.deps.git.base_branch or ctx.deps.git.default_branch
    try:
        content: str | None = await ctx.deps.git.get_file_at_branch(base, path)
    except ValueError as error:
        logger.warning('Could not read the base version of %s: %s', path, error)
        return None
    return content if content is not None else ''


async def _commit_and_report(
    ctx: RunContext[AgentDeps],
    path: str,
    message: str,
    original: str | None,
) -> None:
    """
    Commit a modified file and announce the commit and the resulting diff.

    :param ctx: The agent run context.
    :param path: Repository-relative path of the modified file.
    :param message: The commit message to use.
    :param original: The contents before the change, or None if they could not be determined.
    """
    commit: CommitInfo | None = await ctx.deps.git.commit_changes(message)
    if commit is not None:
        await ctx.deps.events.emit(
            CommitEvent(
                commit_hash=commit.commit_hash,
                message=commit.message,
                files_changed=commit.files_changed,
            )
        )

    if original is None:
        logger.warning('Skipping diff for %s: the base version is unknown.', path)
        return

    modified: str
    try:
        modified = await ctx.deps.editor.read_file(path)
    except (FileNotFoundError, ValueError) as error:
        # A file that cannot be read back as text, such as a binary blob, has no
        # diff to show. The edit itself already succeeded.
        logger.warning('Skipping diff for %s: %s', path, error)
        return

    await ctx.deps.events.emit(
        DiffEvent(file_path=path, original=original, modified=modified)
    )


@coder_agent.tool
async def run_test_command(ctx: RunContext[AgentDeps], command: str) -> str:
    """
    Run a test, build, or lint command inside the sandbox and return its output.
    """
    async with report_tool(ctx.deps.events, 'run_test_command', {'command': command}) as report:
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

        ctx.deps.session.record_verification(result.success)
        status: str = 'passed' if result.success else f'FAILED with exit code {result.exit_code}'
        output: str = f'Command {status}.\n\n{result.output}'
        return report.succeed(output) if result.success else report.fail(output)


@coder_agent.tool
async def web_search(ctx: RunContext[AgentDeps], query: str) -> str:
    """
    Search the web for information using SearXNG.

    Returns the top results as title, URL and a short snippet each.
    """
    async with report_tool(ctx.deps.events, 'web_search', {'query': query}) as report:
        return report.succeed(await _web_search(query))


async def _web_search(query: str) -> str:
    """
    Query SearXNG and render the top results.

    :param query: The search query.
    :returns: The rendered results, or an explanation of why there are none.
    """
    searxng_url: str = os.getenv('SEARXNG_URL', 'http://searxng:8080')

    payload: dict[str, Any]
    try:
        async with AsyncClient(timeout=20.0) as client:
            response: Response = await client.get(
                f'{searxng_url}/search', params={'q': query, 'format': 'json'}
            )
            if response.status_code != 200:
                return f'Web search failed: SearXNG returned HTTP {response.status_code}.'
            payload = response.json()
    except HTTPError as error:
        return f'Web search failed: could not reach SearXNG at {searxng_url}. {error}'
    except ValueError:
        return 'Web search failed: SearXNG returned a response that is not valid JSON.'

    results: Any = payload.get('results')
    if not isinstance(results, list) or not results:
        return f'No web results were found for {query!r}.'

    sections: list[str] = []
    for entry in results[:WEB_SEARCH_RESULT_LIMIT]:
        if not isinstance(entry, dict):
            continue
        title: str = str(entry.get('title', 'Untitled')).strip()
        url: str = str(entry.get('url', '')).strip()
        snippet: str = ' '.join(str(entry.get('content', '')).split())[:WEB_SEARCH_SNIPPET_LENGTH]
        sections.append(f'{title}\n{url}\n{snippet}' if snippet else f'{title}\n{url}')

    if not sections:
        return f'No usable web results were found for {query!r}.'
    return '\n\n'.join(sections)


@coder_agent.tool
async def complete_task(ctx: RunContext[AgentDeps], summary: str) -> str:
    """
    Signal that all the work for the task is finished.

    Call this exactly once, as the final action of your execution phase, with a
    summary of what you did. If the task required changing files, do not call it
    before you have modified at least one and verified the change with
    run_test_command. If the task only ever called for an answer — a summary, an
    explanation, an assessment — call it with that answer as the summary.
    """
    session: SessionContext = ctx.deps.session

    async with report_tool(ctx.deps.events, 'complete_task', {}) as report:
        if not session.has_modified_files:
            # Some tasks legitimately change nothing. Whether this was one of
            # them is not a judgment the gate can make from counting files, so
            # it is put to a subagent. A failure here is deliberately allowed to
            # escape: guessing either way is worse than stopping.
            decision: AdjudicationResult = await adjudicate_completion(session, summary)

            if decision.changes_were_required:
                return report.fail(
                    f'complete_task was rejected: {decision.reason} '
                    'Make the changes the task requires, verify them with run_test_command, '
                    'and only then call complete_task.'
                )

            recorded_answer: str = session.accept_completion_without_changes(summary)
            return report.succeed(
                f'Task marked complete with no file changes. {decision.reason} '
                f'Answer recorded: {recorded_answer}'
            )

        try:
            recorded: str = session.request_completion(summary)
        except ValueError as error:
            return report.fail(f'complete_task was rejected: {error}')

        modified: str = ', '.join(session.modified_files)
        return report.succeed(
            f'Task marked complete. Summary recorded: {recorded} (files modified: {modified})'
        )


async def adjudicate_completion(
    session: SessionContext, summary: str
) -> AdjudicationResult:
    """
    Ask whether the task could have been completed without changing any file.

    Consulted only when nothing was modified. Runs on the reviewer's model, since
    this is a judgment about the work rather than part of doing it.

    :param session: The session context, carrying the task and reviewer model.
    :param summary: The agent's account of what it did.
    :returns: The adjudicator's verdict.
    """
    verdict: AgentRunResult[AdjudicationResult] = await adjudicator_agent.run(
        build_adjudication_prompt(session.task, summary),
        model=resolve_model(session.reviewer_model),
    )
    return verdict.output
