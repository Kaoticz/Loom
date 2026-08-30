from __future__ import annotations

import os
from collections.abc import Sequence
from logging import Logger, getLogger
from typing import Any, ClassVar

from httpx import AsyncClient, HTTPError, Response
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, RunContext

from services.code_search import (
    DEFAULT_LEXICAL_MODE,
    CodeSearchService,
    CodeSearchOutcome,
    LexicalMatch,
    LexicalMatchMode,
    SymbolDefinition,
    describe_ignored_keywords,
    describe_matched_files,
    describe_no_lexical_matches,
    normalise_keywords,
    split_at_keyword_limit,
)
from services.browser_tooling import browser_automation_refusal
from services.repository_files import list_indexable_files
from services.docker_sandbox import CommandResult, DockerSandboxManager
from services.embedding import EmbeddingService
from services.event_bus import CommitEvent, DiffEvent, EventBus, WarningEvent
from services.git_manager import CommitResult, GitManagerService
from services.indexing import IndexingService
from services.tool_reporting import (
    ToolReport,
    ToolReport,
    describe_file_read,
    describe_symbol_lookup,
    report_tool,
)
from services.file_editor import EditResult, FileEditorService
from services.session_context import SessionContext
from services.user_questions import MAX_SUGGESTED_ANSWERS, UserQuestioner

logger: Logger = getLogger(__name__)

MAX_COMMIT_SUMMARY_LENGTH: int = 72
WEB_SEARCH_RESULT_LIMIT: int = 5
WEB_SEARCH_SNIPPET_LENGTH: int = 200

CODER_SYSTEM_PROMPT: str = """You are Loom, an expert AI software engineer working inside a real Git repository.
You operate in a strict ReAct loop: Plan, Action, Observation. Follow the rules below exactly.
You can create, edit, and permanently delete repository files when the task calls for it.

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
- `search_code_lexical` takes a list of keywords, not just one. When you are looking for
  several things — three function names, a class and the constant it uses — pass them all
  in a single call rather than making a call for each. By default it returns lines matching
  any of them; pass `mode="all"` to return only lines containing every one, bearing in mind
  that related symbols usually sit on different lines, so `all` often finds nothing. Each
  entry is one literal, so pass a phrase as a single entry. At most 10 keywords are
  searched for per call; ask for more and the surplus is reported back unsearched.
- Use `get_symbol` to look up the definition of a class or function by name.
- Use `read_file` to read the exact current contents of any file you intend to change.

Never call `edit_file` on a file you have not read in this session. The search block you
pass to `edit_file` must be copied verbatim from what `read_file` returned, or the edit
will not match.

# 3. Verify every change (hard rule)

After every single call to `edit_file`, `create_file`, or `delete_file`, you MUST immediately call
`run_test_command` with a relevant test, build, or lint command before moving on to the
next step. This is a hard rule, not a suggestion. Never make two file modifications in a
row without a verification step between them.

If you do not know which test to run, run the narrowest command that would catch a
mistake in the file you just touched (for example the module's test file, a syntax check,
or a linter on that path).

## Getting a working environment first

Before concluding that a command does not work, check whether the project has a
development environment and set one up if it does not. Look at what the repository says
about itself — a lockfile, a manifest, a contributor guide, a CI configuration — and use
whatever that tooling calls for. Installing dependencies, creating a virtual environment
or fetching modules inside the sandbox is ordinary work, not a detour, and a command that
fails only because nothing is installed is not a failing test.

Once you have a command that runs, call `record_test_command` with it. Later steps of this
plan are told what it was, so nobody works it out twice or is asked twice.

Record only a command you have actually run and watched pass, spelled exactly as you ran it.
Do not record a command you assume will work, and do not assemble a longer one out of parts
you have not tried together — every later step inherits whatever you record, so a command
that does not exist, or that reinstalls everything each time, costs the rest of the plan.
Checking versions is not verifying: `node -v` tells you the tool is there, not that the
project's tests run.

## When there is nothing to test

Some repositories have no tests at all. That is a normal state, not a problem to solve and
not a reason to ask the user anything. Do not invent a test suite, and do not fabricate a
command so that something has run. Call `skip_verification`, saying that the repository has
no tests, and carry on with the work.

## When verification is genuinely stuck

If a command keeps failing and you believe the cause is the command or the environment
rather than the code — it cannot find the tests, it fails identically before and after your
change, it complains about missing tooling you could not install — stop guessing and call
`ask_user`. Ask how the project's tests are meant to be run, or whether to skip verification
for this step, and say which of the two you think is happening.

Do not simply keep re-running a failing command until the failure allowance runs out. That
allowance exists for a repository whose tests were already failing, not as a way past a
command that was never right.

Tell the two cases apart honestly. A test that runs and reports a real problem in your
change is a failing test: fix the code. A command that errors before any test runs, or that
fails in exactly the same way on code you have not touched, is a wrong command. If you are
not sure which you are looking at, assume it is a real failure and fix the code.

If a command you were given earlier starts failing this way, call `forget_test_command`
before asking again, so a later step does not inherit it.

# 4. Recover from errors, do not skip them

If a tool call fails or a test does not pass:

1. State what went wrong and why, in one or two sentences.
2. Propose a specific fix.
3. Apply the fix and re-run the verification command.

Never silently skip a failing step, and never move to the next step while a verification
command you just ran is still failing.

# 4b. Asking the user

When the task cannot be carried out correctly without something only the user can tell you — which
of several things they meant, which of two reasonable approaches they want, a decision the
repository does not settle — call `ask_user`. It puts the question to the user and waits, and
returns what they said.

Suggest up to six answers when you can see what the likely ones are. The interface always offers
the user a further option of typing their own reply, so you neither need to add one nor can remove
it, and a question with no suggested answers at all is perfectly usable.

Ask when the answer changes what you would do. Do not ask permission for work you were already
asked to do, do not ask what you could find out by reading the repository, and do not ask a
question you could answer by making a reasonable assumption and saying so in your summary. Every
question stops the task until someone comes back to it.

# 5. Finishing

## When you were given one step of a plan

You will be told plainly when this is the case: the prompt names which step of how many you are
working on. Then the step is your whole job, and `complete_task` is not yours to call — the plan
is not finished until every step is, and the steps after yours have not run yet.

- Finish with `complete_step`, passing a summary of what you did. The next step is told your
  summary and is not shown your work, so anything you leave out is lost to it.
- If the step was investigative from the start — finding where something lives, confirming what
  the code already does — and required no change to any file, finish with
  `accept_step_without_changes` and report what you found. Do not invent a change to satisfy the
  verification rule.
- Do not use `accept_step_without_changes` to get past a step you could not manage. If the step
  called for a change, make the change.
- Do the work of your step and nothing belonging to a later one. A step that runs ahead leaves
  the next one repeating it or contradicting it.

Everything below about verification applies to your step exactly as it does to a whole task.

## When you were given a whole task

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
no. Whether this is such a task has already been decided before you were given it, and you will
be told plainly when it is. For those:

- Pass your answer itself as the summary. It is the whole product of the task, so make it
  complete: the reader sees this and nothing else.
- Do not invent a file change to satisfy the rule above. A task that asked a question is
  finished when the question is answered.
- Do not use this to escape work you found difficult. On any task you were not told is an
  answering task, completing without a modification is refused outright.
- If you were told this is an answering task but find that it cannot be answered without
  changing a file, do not change one. Say so in your summary and stop. The task will be
  replanned as ordinary work.

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
    indexing: IndexingService
    embedding: EmbeddingService
    events: EventBus
    session: SessionContext
    questioner: UserQuestioner | None = None
    """
    How a tool puts a question to the user, when the session supplies one.

    Optional because the tools are also driven directly by tests and by any
    caller with no user to ask; `ask_user` says so plainly rather than hanging.
    """


coder_agent: Agent[AgentDeps, str] = Agent(
    deps_type=AgentDeps,
    output_type=str,
    system_prompt=CODER_SYSTEM_PROMPT,
    retries=5,
)


@coder_agent.tool
async def search_code(ctx: RunContext[AgentDeps], query: str) -> str:
    """
    Search the codebase for code related to a natural language query.

    Runs a semantic search over the indexed embeddings and a literal text search
    in parallel, then merges the two, so that both conceptual matches and exact
    keyword matches are covered.
    :param query: What you are looking for, in plain words, for example "where retries are configured".
    """
    async with report_tool(ctx.deps.events, 'search_code', {'query': query}) as report:
        outcome: CodeSearchOutcome = await _search_code(ctx, query)

        # The agent gets the code; the reader gets the files it came from. The
        # snippets are what the agent asked for and what floods the feed.
        return report.succeed(
            outcome.rendered,
            observation=describe_matched_files(
                f'Searched for {query!r}', outcome.file_paths, outcome.notices
            ),
        )


async def _search_code(ctx: RunContext[AgentDeps], query: str) -> CodeSearchOutcome:
    """
    Run the merged semantic and lexical search.

    :param ctx: The agent run context.
    :param query: The natural language query.
    :returns: The rendered results, the files they came from, and any warnings.
    """
    if not query.strip():
        # Embedding an empty string raises, and there is nothing meaningful to
        # search for anyway. Refuse with an explanation rather than let the
        # exception abort the whole session.
        return CodeSearchOutcome(
            rendered='No query was given. Pass what you are looking for in plain '
            'words, for example "where retries are configured".',
            file_paths=[],
            notices=[],
        )

    embedding: list[float] = await ctx.deps.embedding.embed(query)

    sections: list[str] = []
    seen: set[str] = set()
    notices: list[str] = []
    ordered_paths: list[str] = []

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
        ordered_paths.append(path)
        sections.append(f'{path}:{chunk.start_line} (semantic)\n{chunk.chunk_text}')

    lexical: list[LexicalMatch] = await ctx.deps.search.search_code_lexical(query)
    for match in lexical:
        if match.file_path in seen:
            continue
        seen.add(match.file_path)
        ordered_paths.append(match.file_path)
        sections.append(f'{match.render()} (lexical)')

    if not sections:
        # Kept out of `notices`: the observation says as much in its own words,
        # and repeating it there would read as two separate complaints.
        empty: str = f'No code found for {query!r}. The repository may not be indexed yet.'
        return CodeSearchOutcome(
            rendered='\n'.join([*notices, empty]),
            file_paths=[],
            notices=notices,
        )

    body: str = '\n\n'.join(sections)
    return CodeSearchOutcome(
        rendered='\n\n'.join([*notices, body]) if notices else body,
        file_paths=ordered_paths,
        notices=notices,
    )


@coder_agent.tool
async def search_code_lexical(
    ctx: RunContext[AgentDeps],
    keywords: list[str],
    limit: int = 10,
    mode: LexicalMatchMode = DEFAULT_LEXICAL_MODE
) -> str:
    """
    Search the codebase for exact, case-insensitive text.

    Use this when you know the literal strings you are looking for, such as
    function names, an error message, or a configuration key.

    Pass as many keywords as you like in one call. Each entry is one literal,
    spaces included, so use a single entry to search for a phrase such as
    "connection refused", and separate entries to search for separate terms.

    `mode` decides how they combine, and defaults to "any":

    - "any" returns lines containing at least one of the keywords. Use it to
      look for several unrelated things at once, such as three function names.
    - "all" returns only lines containing every keyword. Use it to narrow down,
      remembering that related symbols usually sit on different lines, so this
      often finds nothing even when both keywords are present in the file.

    `limit` caps the total number of lines returned across all keywords, and at
    most 10 keywords are searched for in one call.
    :param keywords: The exact identifiers or literals to look for, as a list of separate strings.
    :param limit: How many results to return at most.
    :param mode: Whether a result must contain all of the keywords or any of them.
    """
    searched: list[str]
    ignored: list[str]
    searched, ignored = split_at_keyword_limit(normalise_keywords(keywords))
    arguments: dict[str, object] = {'keywords': searched, 'limit': limit, 'mode': mode.value}

    async with report_tool(ctx.deps.events, 'search_code_lexical', arguments) as report:
        if not searched:
            return report.succeed(
                'No keywords were given. Pass one or more non-empty strings, '
                'for example ["parse_plan", "render_plan"].'
            )

        matches: list[LexicalMatch] = await ctx.deps.search.search_code_lexical(
            searched, limit=limit, mode=mode
        )
        body: str = (
            '\n\n'.join(match.render() for match in matches)
            if matches
            else describe_no_lexical_matches(searched, mode)
        )

        # Prepended rather than appended: the agent needs to know part of its
        # request went unserved before it draws conclusions from the results.
        notice: str = describe_ignored_keywords(ignored)
        result: str = f'{notice}\n\n{body}' if notice else body

        # Unique, in order: a file with nine matching lines is one file to the
        # reader, however many times it appears in what the agent receives.
        matched_paths: list[str] = list(dict.fromkeys(match.file_path for match in matches))
        subject: str = 'Searched for ' + ', '.join(repr(keyword) for keyword in searched)

        return report.succeed(
            result,
            observation=describe_matched_files(
                subject, matched_paths, [notice] if notice else []
            ),
        )


@coder_agent.tool
async def get_symbol(ctx: RunContext[AgentDeps], symbol: str) -> str:
    """
    Get the definition of a symbol (class, function).
    :param symbol: The exact name of the function, class or method, without brackets or a module prefix.
    """
    async with report_tool(ctx.deps.events, 'get_symbol', {'symbol': symbol}) as report:
        if not symbol.strip():
            return report.fail('No symbol was given. Pass the exact name of the function, '
                               'class or method to look up, for example "find_user".')
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
    /,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    **extra: object
) -> str:
    """
    Read a file from the repository.

    Pass start_line and end_line to read only part of a large file. When you do,
    each line is prefixed with its line number for reference: those prefixes are
    not part of the file, so never include them in an edit_file search block.
    :param path: Path to the file, relative to the repository root, for example "src/app.py". An absolute path inside the repository is accepted and converted.
    :param start_line: First line to read, counting from one. Omit to read from the beginning.
    :param end_line: Last line to read, inclusive. Omit to read to the end.
    """
    arguments: dict[str, Any] = {'path': path}
    if start_line is not None:
        arguments['start_line'] = start_line
    if end_line is not None:
        arguments['end_line'] = end_line

    async with report_tool(ctx.deps.events, 'read_file', arguments) as report:
        path = _repository_path(ctx, path)
        try:
            contents: str = await ctx.deps.editor.read_file(path, start_line, end_line)
            # The agent gets the file; the reader gets a line saying it was read.
            return report.succeed(
                contents,
                observation=describe_file_read(path, contents, start_line, end_line),
            )
        except FileNotFoundError:
            return report.fail(
                f'File {path} does not exist. Use list_files to see what is there, or '
                'search_code to find where something lives.'
            )
        except ValueError as error:
            return report.fail(f'Could not read {path}: {error}')


@coder_agent.tool
async def create_file(
    ctx: RunContext[AgentDeps],
    /,
    path: str,
    content: str,
    summary: str = '',
    **extra: object
) -> str:
    """
    Create a new file in the repository.

    Give a short summary of why the file is being added, in a few words, so the
    commit history reads meaningfully.
    :param path: Path for the new file, relative to the repository root. An absolute path inside the repository is accepted and converted.
    :param content: The complete contents of the new file.
    :param summary: A few words on what this file is for, used in the commit message.
    """
    async with report_tool(ctx.deps.events, 'create_file', {'path': path}) as report:
        path = _repository_path(ctx, path)
        if not path:
            return report.fail('Could not create a file: no path was given. Pass a '
                               'repository-relative path, for example "src/app.py".')
        original: str | None = await _read_base_version(ctx, path)
        try:
            await ctx.deps.editor.create_file(path, content)
        except (ValueError, OSError) as error:
            return report.fail(f'Could not create {path}: {error}')

        ctx.deps.session.record_modification(path)
        await _report_diff(ctx, path, original)
        ctx.deps.session.note_pending_change(_commit_message('Create', path, summary))
        try:
            await ctx.deps.indexing.index_file(path)
        except Exception as error:
            return report.fail(
                f'Created {path}, but could not refresh its index: {error}. The file was written, '
                'but indexed searches may still show stale results.'
            )
        return report.succeed(f'Created {path}.' + _note_ignored(extra))


@coder_agent.tool
async def edit_file(
    ctx: RunContext[AgentDeps],
    /,
    path: str,
    search: str,
    replace: str,
    summary: str = '',
    **extra: object
) -> str:
    """
    Edit a file surgically.

    Give a short summary of what the edit accomplishes, in a few words, so the
    commit made when this step finishes reads meaningfully.
    :param path: Path to the file to change, relative to the repository root. An absolute path inside the repository is accepted and converted.
    :param search: The exact text to find, copied character for character from the file, including its indentation. Not a description, not a regular expression, not a line number. Include enough surrounding lines to make it appear only once.
    :param replace: The exact text to put in its place. Pass an empty string to delete the text found.
    :param summary: A few words on what the edit accomplishes, used in the commit message.
    """
    async with report_tool(ctx.deps.events, 'edit_file', {'path': path}) as report:
        path = _repository_path(ctx, path)
        if not path:
            return report.fail('Could not edit a file: no path was given. Pass a '
                               'repository-relative path, for example "src/app.py".')
        if not search.strip():
            return report.fail('Could not edit: a search block is required. Copy the exact '
                               'text to replace from the file, character for character.')
        original: str | None = await _read_base_version(ctx, path)
        try:
            result: EditResult = await ctx.deps.editor.edit_file(path, search, replace)
        except (ValueError, OSError) as error:
            return report.fail(f'Could not edit {path}: {error}')

        if not result.success:
            return report.fail(result.message)

        ctx.deps.session.record_modification(path)
        await _report_diff(ctx, path, original)
        ctx.deps.session.note_pending_change(_commit_message('Edit', path, summary))
        try:
            await ctx.deps.indexing.index_file(path)
        except Exception as error:
            return report.fail(
                f'{result.message} The file was changed, but its index could not be refreshed: '
                f'{error}. Indexed searches may still show stale results.'
            )
        return report.succeed(result.message + _note_ignored(extra))


@coder_agent.tool
async def delete_file(
    ctx: RunContext[AgentDeps],
    /,
    path: str,
    summary: str = '',
    **extra: object,
) -> str:
    """
    Permanently delete a file from the repository.

    This removes the file from disk and its indexed search records. It counts as
    a file change, so run a relevant verification command immediately afterwards.
    :param path: Path to delete, relative to the repository root. An absolute path inside the repository is accepted and converted.
    :param summary: A few words on why the file is being deleted, used in the commit message.
    """
    async with report_tool(ctx.deps.events, 'delete_file', {'path': path}) as report:
        path = _repository_path(ctx, path)
        if not path:
            return report.fail('Could not delete a file: no path was given. Pass a '
                               'repository-relative file path, for example "src/unused.py".')
        original: str | None = await _read_base_version(ctx, path)
        try:
            _ = await ctx.deps.editor.delete_file(path)
        except FileNotFoundError:
            return report.fail(f'Could not delete {path}: the file does not exist.')
        except (ValueError, OSError) as error:
            return report.fail(f'Could not delete {path}: {error}')

        ctx.deps.session.record_modification(path)
        await _report_diff(ctx, path, original)
        ctx.deps.session.note_pending_change(_commit_message('Delete', path, summary))
        try:
            await ctx.deps.indexing.drop_file(path)
        except Exception as error:
            return report.fail(
                f'Deleted {path}, but could not remove it from the index: {error}. The file is '
                'gone, but indexed searches may still show stale results.'
            )
        return report.succeed(f'Deleted {path}.' + _note_ignored(extra))


def _repository_path(ctx: RunContext[AgentDeps], path: str) -> str:
    """
    Reduce a path the model supplied to one relative to the repository root.

    Models routinely pass an absolute path. The editor already tolerates one
    inside the root, so the edit lands, but everything downstream — the modified
    file list, the commit message, the git lookup for the original contents —
    assumes a relative path and quietly misbehaves with an absolute one. Reducing
    it once here keeps that from happening in four places.

    A path outside the repository is returned untouched, so the editor still
    refuses it and says so, rather than this function inventing a path inside.

    :param ctx: The agent run context.
    :param path: The path as the model wrote it.
    :returns: The path relative to the repository root where possible.
    """
    cleaned: str = path.strip().replace('\\', os.sep)
    if not cleaned:
        return ''

    if not os.path.isabs(cleaned):
        # `.`, `./`, `./src` and `src/` all name the same things as `` and `src`.
        # A model that writes any of them means the obvious thing, and refusing
        # it teaches nothing except that the tool is fussy.
        collapsed: str = os.path.normpath(cleaned)
        return '' if collapsed in ('.', os.sep) else collapsed.strip(os.sep)

    root: str = os.path.abspath(ctx.deps.git.repo_path)
    resolved: str = os.path.abspath(cleaned)
    if resolved == root:
        return ''
    if resolved.startswith(root + os.sep):
        return os.path.relpath(resolved, root)
    return cleaned


def _note_ignored(extra: dict[str, object]) -> str:
    """
    Describe any arguments the tool did not recognise.

    Reported rather than silently dropped: the call is honoured because failing
    an otherwise valid edit over a stray argument costs a retry and teaches the
    model nothing, but saying which arguments went nowhere lets it stop sending
    them instead of assuming they took effect.

    :param extra: The unrecognised keyword arguments.
    :returns: A note to append to the tool's reply, empty when there were none.
    """
    if not extra:
        return ''
    named: str = ', '.join(sorted(extra))
    return (
        f' (ignored unknown argument(s): {named}. They do nothing here; '
        'leave them out next time.)'
    )


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


async def _report_diff(
    ctx: RunContext[AgentDeps],
    path: str,
    original: str | None,
) -> None:
    """
    Announce the diff a single file modification produced.

    No longer commits. A commit per write turned one step into a string of
    commits nobody chose the boundaries of, and a plan into a history that could
    not be read back a step at a time. The commit now happens once, when the
    step finishes. The diff still goes out per modification, because the live diff viewer
    shows the edit as it lands and has nothing to do with commit cadence.

    :param ctx: The agent run context.
    :param path: Repository-relative path of the modified file.
    :param original: The contents before the change, or None if they could not be determined.
    """
    if original is None:
        logger.warning('Skipping diff for %s: the base version is unknown.', path)
        return

    modified: str
    try:
        modified = await ctx.deps.editor.read_file(path)
    except FileNotFoundError:
        await ctx.deps.events.emit(DiffEvent(file_path=path, original=original, modified=''))
        return
    except ValueError as error:
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


@coder_agent.tool
async def web_search(ctx: RunContext[AgentDeps], query: str) -> str:
    """
    Search the web for information using SearXNG.

    Returns the top results as title, URL and a short snippet each.
    :param query: What to search the web for, in plain words.
    """
    async with report_tool(ctx.deps.events, 'web_search', {'query': query}) as report:
        if not query.strip():
            return report.fail('No query was given. Pass what you want to search the web '
                               'for, in plain words.')
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


@coder_agent.tool
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


@coder_agent.tool
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


@coder_agent.tool
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


@coder_agent.tool
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


@coder_agent.tool
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


@coder_agent.tool
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


MAX_LISTED_FILES: int = 500
"""
How many paths `list_files` will return.

A large repository would otherwise spend most of a small model's window on a
file listing, leaving no room to reason about what the listing said.
"""


@coder_agent.tool
async def list_files(
    ctx: RunContext[AgentDeps], /, path: str = '', **extra: object
) -> str:
    """
    List the files in the repository, optionally under one directory.

    Use this to see how the project is laid out before deciding what to read.
    Pass a directory to narrow the listing, or nothing to list the whole
    repository. Files the repository ignores are never listed.

    A directory that does not exist is not an error: you will be shown the
    directories that do exist at that level, so you can correct the path.
    :param path: Directory to list, relative to the repository root. Omit it to list the whole repository.
    """
    arguments: dict[str, Any] = {'path': path} if path else {}

    async with report_tool(ctx.deps.events, 'list_files', arguments) as report:
        root: str = ctx.deps.git.repo_path
        try:
            found: list[str] = list_indexable_files(root)
        except OSError as error:
            return report.fail(f'Could not list {root}: {error}')

        relative: list[str] = sorted(os.path.relpath(item, root) for item in found)
        if not relative:
            return report.succeed(
                'The repository contains no files that are not ignored.',
                observation='Listed 0 files.',
            )

        wanted: str = _repository_path(ctx, path)
        if not wanted:
            return _render_listing(report, relative, 'the repository')

        matching: list[str] = [
            item for item in relative if item == wanted or item.startswith(wanted + os.sep)
        ]
        if matching:
            return _render_listing(report, matching, wanted)

        # A miss is a legitimate answer to "what is in here", not a failure. The
        # useful reply is what does exist nearby, which lets the model correct
        # itself in one turn instead of guessing again — and guessing again is
        # what exhausts the tool's retry budget and ends the session.
        return report.succeed(
            f'Nothing in the repository matches "{path}".\n\n'
            f'These directories do exist:\n{_directories_near(relative, wanted)}',
            observation=f'No match for {path}.',
        )


def _render_listing(report: ToolReport, paths: list[str], where: str) -> str:
    """
    Report a listing, truncated if it would flood the context.

    :param report: The tool report to record into.
    :param paths: The repository-relative paths to list.
    :param where: What was listed, for the observation line.
    :returns: The recorded result.
    """
    shown: list[str] = paths[:MAX_LISTED_FILES]
    listing: str = '\n'.join(shown)
    if len(paths) > len(shown):
        listing += (
            f'\n... and {len(paths) - len(shown)} more not shown. '
            'Pass a directory to narrow the listing.'
        )
    return report.succeed(
        listing, observation=f'Listed {len(shown)} of {len(paths)} files in {where}.'
    )


def _directories_near(paths: list[str], wanted: str) -> str:
    """
    List the directories that exist alongside a path that does not.

    Shown at the depth that was asked for, so a wrong guess at the third level
    is answered with the third level rather than with the repository root.

    :param paths: Every repository-relative path.
    :param wanted: The path that matched nothing.
    :returns: A newline-separated list of directories.
    """
    depth: int = len(wanted.split(os.sep)) if wanted else 1
    parent: str = os.sep.join(wanted.split(os.sep)[:-1])

    candidates: set[str] = set()
    for item in paths:
        parts: list[str] = item.split(os.sep)
        if len(parts) <= depth - 1:
            continue
        candidate: str = os.sep.join(parts[:depth])
        if parent and not candidate.startswith(parent + os.sep):
            continue
        candidates.add(candidate)

    if not candidates and parent:
        # The parent is wrong too, so answering at that depth would repeat the
        # mistake. Fall back to the top level, which certainly exists.
        candidates = {item.split(os.sep)[0] for item in paths}

    return '\n'.join(sorted(candidates)[:MAX_LISTED_FILES]) or '(none)'




READ_ONLY_TOOLS: tuple[Any, ...] = (
    ask_user,
    search_code,
    search_code_lexical,
    get_symbol,
    read_file,
    list_files,
    web_search,
    complete_task,
)
"""
The tools a task needing no file changes is allowed to use.

Asking the user a question is included: it changes nothing, and a task that can
only be answered once an ambiguity is settled is better off asking than
guessing.

Everything absent from this list can alter the working tree: `create_file`,
`edit_file`, and `delete_file` write or remove files, and `run_test_command` executes an arbitrary
command inside a sandbox with the repository mounted. A task classified as
needing no changes has nothing to write and nothing to verify, so none of them
has a use it could put them to.
"""


answering_agent: Agent[AgentDeps, str] = Agent(
    deps_type=AgentDeps,
    output_type=str,
    system_prompt=CODER_SYSTEM_PROMPT,
    retries=5,
)
"""
The agent used for a task classified as requiring no file changes.

A separate agent rather than a filtered view of `coder_agent`, so that being
unable to write is structural: the tools were never registered, so there is no
switch to be left in the wrong position and no per-run state to get out of step
between concurrent sessions. Telling the agent not to write, as the prompt also
does, is worth having as well — but by the time an instruction is disobeyed the
repository has already changed, which is exactly what this prevents.
"""

for _read_only_tool in READ_ONLY_TOOLS:
    answering_agent.tool(_read_only_tool)


PLANNING_TOOLS: tuple[Any, ...] = (
    ask_user,
    search_code,
    search_code_lexical,
    get_symbol,
    read_file,
    list_files,
    web_search,
)
"""
The tools the planner is allowed to use.

A plan written without looking at the repository is a guess about a codebase the
planner has never seen, so the tools that let it look are exactly the ones it
needs. Every one of them only reads.

`ask_user` is included: an ambiguity that would change what steps to produce is
better settled before the plan is written than discovered halfway through
executing it.

`complete_task` is absent, unlike in `READ_ONLY_TOOLS`: the planner's job ends
when it returns a plan, and it has no task to mark complete.
Nothing here can alter the working tree, which is the point — planning happens
before the user has approved anything.
"""
