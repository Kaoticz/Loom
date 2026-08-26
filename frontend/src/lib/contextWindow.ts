/**
 * The context window settings, which control how much history each agent keeps.
 *
 * Bounds mirror the backend's, which enforces them again because the value
 * arrives from the client. Duplicating them here is for the person typing, not
 * for safety: a control that lets you enter a number the server will silently
 * change is worse than one that will not let you enter it.
 */

/** Smallest window worth honouring; below this there is no room for a prompt and a reply. */
export const MIN_CONTEXT_WINDOW = 2048;

/** Largest window offered. Beyond this no locally-hosted model is plausibly running. */
export const MAX_CONTEXT_WINDOW = 1000000;

/** The window offered when someone turns Auto off, matching the backend's own default. */
export const DEFAULT_CONTEXT_WINDOW = 32768;

/**
 * Interpret what was typed into a context window box.
 *
 * An empty or unreadable box means the person is mid-edit rather than asking
 * for the default, so the default stands in until they finish. Only the Auto
 * control selects the default, because that is an explicit choice.
 *
 * @param raw The raw value from the input.
 * @returns The context window to store.
 */
export function readContextWindow(raw: string): number {
    const parsed = Number.parseInt(raw, 10);
    if (Number.isNaN(parsed)) return DEFAULT_CONTEXT_WINDOW;
    return Math.max(MIN_CONTEXT_WINDOW, Math.min(parsed, MAX_CONTEXT_WINDOW));
}
