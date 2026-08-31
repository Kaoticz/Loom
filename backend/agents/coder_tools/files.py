from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from functools import partial
from logging import Logger, getLogger
from typing import Any

from pydantic_ai import RunContext

from services.event_bus import DiffEvent
from services.file_editor import EditResult
from services.repository_files import list_indexable_files
from services.tool_reporting import ToolReport, describe_file_read, report_tool

from .deps import AgentDeps

logger: Logger = getLogger('agents.coder')

MAX_COMMIT_SUMMARY_LENGTH: int = 72


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
    return await _mutate_file(
        ctx,
        tool_name='create_file',
        path=path,
        summary=summary,
        extra=extra,
        mutation=partial(ctx.deps.editor.create_file, content=content),
        reindex=ctx.deps.indexing.index_file,
        commit_verb='Create',
        completed_verb='Created',
        path_error='Could not create a file: no path was given. Pass a '
                   'repository-relative path, for example "src/app.py".',
        mutation_error_prefix='Could not create',
        index_failure_suffix='but could not refresh its index: {error}. The file was written, '
                             'but indexed searches may still show stale results.',
    )


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
    search_error: str | None = None
    if not search.strip():
        search_error = ('Could not edit: a search block is required. Copy the exact '
                        'text to replace from the file, character for character.')

    return await _mutate_file(
        ctx,
        tool_name='edit_file',
        path=path,
        summary=summary,
        extra=extra,
        mutation=partial(
            ctx.deps.editor.edit_file,
            search_block=search,
            replace_block=replace,
        ),
        reindex=ctx.deps.indexing.index_file,
        commit_verb='Edit',
        completed_verb='Edited',
        path_error='Could not edit a file: no path was given. Pass a '
                   'repository-relative path, for example "src/app.py".',
        mutation_error_prefix='Could not edit',
        index_failure_suffix='The file was changed, but its index could not be refreshed: {error}. '
                             'Indexed searches may still show stale results.',
        validation_error=search_error,
    )


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
    return await _mutate_file(
        ctx,
        tool_name='delete_file',
        path=path,
        summary=summary,
        extra=extra,
        mutation=ctx.deps.editor.delete_file,
        reindex=ctx.deps.indexing.drop_file,
        commit_verb='Delete',
        completed_verb='Deleted',
        path_error='Could not delete a file: no path was given. Pass a '
                   'repository-relative file path, for example "src/unused.py".',
        mutation_error_prefix='Could not delete',
        index_failure_suffix='but could not remove it from the index: {error}. The file is '
                             'gone, but indexed searches may still show stale results.',
        missing_file_error='Could not delete {path}: the file does not exist.',
    )


async def _mutate_file(
    ctx: RunContext[AgentDeps],
    tool_name: str,
    path: str,
    summary: str,
    extra: dict[str, object],
    mutation: Callable[[str], Awaitable[object]],
    reindex: Callable[[str], Awaitable[object]],
    commit_verb: str,
    completed_verb: str,
    path_error: str,
    mutation_error_prefix: str,
    index_failure_suffix: str,
    validation_error: str | None = None,
    missing_file_error: str | None = None,
) -> str:
    """
    Apply a file mutation and report its modification lifecycle.

    :param ctx: The agent run context.
    :param tool_name: The name reported for the tool invocation.
    :param path: The model-supplied path to mutate.
    :param summary: The commit-summary fragment supplied by the model.
    :param extra: Unrecognised arguments to report back to the model.
    :param mutation: The editor operation to apply to the resolved path.
    :param reindex: The index operation to apply after the mutation.
    :param commit_verb: The verb to use in the pending commit summary.
    :param completed_verb: The past-tense verb to use in successful responses.
    :param path_error: The response to use when no path was supplied.
    :param mutation_error_prefix: The response prefix for editor errors.
    :param index_failure_suffix: The formatted response suffix for index failures.
    :param validation_error: An operation-specific validation response, if invalid.
    :param missing_file_error: The formatted response for a missing file, if applicable.
    :returns: The reported result of the mutation.
    """
    async with report_tool(ctx.deps.events, tool_name, {'path': path}) as report:
        path = _repository_path(ctx, path)
        if not path:
            return report.fail(path_error)
        if validation_error is not None:
            return report.fail(validation_error)

        original: str | None = await _read_base_version(ctx, path)
        try:
            result: object = await mutation(path)
        except FileNotFoundError as error:
            if missing_file_error is not None:
                return report.fail(missing_file_error.format(path=path))
            return report.fail(f'{mutation_error_prefix} {path}: {error}')
        except (ValueError, OSError) as error:
            return report.fail(f'{mutation_error_prefix} {path}: {error}')

        if isinstance(result, EditResult) and not result.success:
            return report.fail(result.message)

        message: str = result.message if isinstance(result, EditResult) else f'{completed_verb} {path}.'
        ctx.deps.session.record_modification(path)
        await _report_diff(ctx, path, original)
        ctx.deps.session.note_pending_change(_commit_message(commit_verb, path, summary))
        try:
            await reindex(path)
        except Exception as error:
            return report.fail(f'{message} {index_failure_suffix.format(error=error)}')
        return report.succeed(message + _note_ignored(extra))


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


MAX_LISTED_FILES: int = 500
"""
How many paths `list_files` will return.

A large repository would otherwise spend most of a small model's window on a
file listing, leaving no room to reason about what the listing said.
"""


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
