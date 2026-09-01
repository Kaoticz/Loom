from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from services.file_change import FileChange

DEFAULT_REVIEW_BUDGET_CHARACTERS: int = 24_000


def render_change_summary(changes: Sequence[FileChange]) -> str:
    """
    Render the list of changed files with their line counts.

    :param changes: The files the task branch has changed.
    :returns: One line per file, or a note that nothing changed.
    """
    if not changes:
        return '(no files changed)'
    return '\n'.join(
        f'  {change.path} (+{change.insertions}/-{change.deletions})' for change in changes
    )


async def collect_review_diffs(
    changes: Sequence[FileChange],
    fetch: Callable[[str], Awaitable[str]],
    budget_characters: int = DEFAULT_REVIEW_BUDGET_CHARACTERS,
) -> tuple[dict[str, str], list[str]]:
    """
    Fetch as many per-file diffs as the review budget allows.

    Files are visited smallest first, so one enormous generated file cannot
    crowd out every reviewable change. Once a diff no longer fits, the remaining
    files are larger still and are recorded as omitted without being fetched,
    which avoids reading diffs that would only be discarded.

    :param changes: The files the task branch has changed.
    :param fetch: A callable returning the diff for one path.
    :param budget_characters: Roughly how much diff content may be gathered.
    :returns: The diffs that fit, keyed by path, and the paths that did not.
    """
    ordered: list[FileChange] = sorted(changes, key=lambda item: item.total_changed)

    diffs: dict[str, str] = {}
    omitted: list[str] = []
    remaining: int = max(0, budget_characters)

    for index, change in enumerate(ordered):
        diff: str = await fetch(change.path)
        if not diff:
            continue

        cost: int = len(diff) + len(change.path) + 12
        if cost > remaining:
            omitted.extend(later.path for later in ordered[index:])
            break

        diffs[change.path] = diff
        remaining -= cost

    return diffs, omitted


def build_review_prompt(
    task: str,
    changes: Sequence[FileChange],
    diffs: dict[str, str],
    omitted: Sequence[str] = (),
    previous_feedback: str | None = None,
    budget_characters: int = DEFAULT_REVIEW_BUDGET_CHARACTERS,
) -> str:
    """
    Assemble a review prompt that fits within a context budget.

    The full list of changed files is always included, because the reviewer must
    know the shape of the change even when it cannot read all of it. Anything
    left out is named explicitly, so an omission is visible rather than looking
    like an unchanged file.

    :param task: The original task description given to the coder.
    :param changes: The files the task branch has changed.
    :param diffs: The per-file diffs that fit the budget, keyed by path.
    :param omitted: Paths whose diffs were left out.
    :param previous_feedback: What this reviewer said last time, on a re-review.
    :param budget_characters: Roughly how much of the diff may be included.
    :returns: The prompt to send to the reviewer agent.
    """
    header: str = (
        f'Original task:\n{task}\n\n'
        f'Files changed ({len(changes)}):\n{render_change_summary(changes)}\n'
    )

    if previous_feedback:
        header += (
            '\nYou reviewed an earlier version of this work and asked for the following '
            f'changes:\n{previous_feedback}\n'
            'Check specifically whether each point has been addressed, and do not repeat '
            'a point that has been resolved.\n'
        )

    included: list[str] = []
    omitted_paths: list[str] = list(omitted)
    remaining: int = max(0, budget_characters - len(header))

    # Smallest first, so a single enormous file cannot crowd out everything else.
    for change in sorted(changes, key=lambda item: item.total_changed):
        diff: str = diffs.get(change.path, '')
        if not diff:
            continue

        block: str = f'\n--- {change.path} ---\n{diff}'
        if len(block) > remaining:
            omitted_paths.append(change.path)
            continue

        included.append(block)
        remaining -= len(block)

    omitted = omitted_paths
    sections: list[str] = [header]
    if included:
        sections.append('\nDiff:\n' + ''.join(included))
    else:
        sections.append('\nNo diff content could be included within the review budget.')

    if omitted:
        sections.append(
            '\n\nThe following files changed but their diffs were too large to include: '
            + ', '.join(sorted(omitted))
            + '. Judge them by the summary above and say so if you cannot assess them.'
        )

    return ''.join(sections)
