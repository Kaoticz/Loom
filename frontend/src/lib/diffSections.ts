/**
 * Ordering and expansion state for the diff accordions.
 *
 * Kept apart from the component so both can be exercised directly: the ordering
 * has awkward cases around case and directory depth, and the expansion rules
 * have to survive diffs arriving while the reader has sections open.
 */

/**
 * Order file paths alphabetically.
 *
 * Compared case insensitively, because a reader scanning for `README.md` does
 * not expect capitals to sort into their own group ahead of everything else.
 * Paths differing only in case fall back to a code-unit comparison, so the order
 * is total and does not depend on the input order or the host's locale.
 *
 * @param paths The file paths to order.
 * @returns A new array, sorted.
 */
export function orderFilePaths(paths: readonly string[]): string[] {
    return [...paths].sort((left, right) => {
        const lowerLeft = left.toLowerCase();
        const lowerRight = right.toLowerCase();
        if (lowerLeft < lowerRight) return -1;
        if (lowerLeft > lowerRight) return 1;
        if (left < right) return -1;
        if (left > right) return 1;
        return 0;
    });
}

/**
 * Carry expansion state across a change to the set of changed files.
 *
 * A file the reader opened stays open when further diffs arrive; a file that has
 * only just appeared starts collapsed. Paths no longer present are dropped, so
 * the set cannot grow without bound over a long session.
 *
 * @param expanded The paths currently expanded.
 * @param paths The paths that now have diffs.
 * @returns The expansion set to keep.
 */
export function retainExpanded(
    expanded: ReadonlySet<string>,
    paths: readonly string[]
): Set<string> {
    const present = new Set(paths);
    return new Set([...expanded].filter((path) => present.has(path)));
}

/**
 * Toggle one path's expansion, leaving every other path alone.
 *
 * @param expanded The paths currently expanded.
 * @param path The path being acted on.
 * @returns The new expansion set.
 */
export function toggleExpanded(expanded: ReadonlySet<string>, path: string): Set<string> {
    const next = new Set(expanded);
    if (!next.delete(path)) next.add(path);
    return next;
}
