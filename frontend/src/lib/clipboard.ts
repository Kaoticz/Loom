/**
 * Copying text to the clipboard.
 *
 * Two things can go wrong and they are indistinguishable to the reader, so they
 * are reported the same way: the API is missing entirely, which is what happens
 * outside a secure context, or it is present and the write is refused. Neither
 * is exceptional enough to throw over — the caller's only sensible response to
 * either is to say the copy did not happen.
 */

/** Where a copy attempt has got to. */
export type CopyState = 'idle' | 'copied' | 'failed';

/**
 * The glyph shown on the copy control for a given result.
 *
 * A function rather than an inline conditional because TypeScript narrows a
 * variable to whatever it was last assigned, and the state is only ever
 * reassigned inside an async callback it cannot follow.
 *
 * @param state Where the copy attempt has got to.
 * @returns The character to show.
 */
export function copyGlyph(state: CopyState): string {
    if (state === 'copied') return '✓';
    if (state === 'failed') return '✕';
    return '⮺';
}

/** How long a copy result stays on screen before the control returns to normal. */
export const COPY_FEEDBACK_MS = 1600;

/**
 * Put text on the clipboard.
 *
 * @param text The text to copy.
 * @returns True if the text reached the clipboard.
 */
export async function copyText(text: string): Promise<boolean> {
    const clipboard = typeof navigator === 'undefined' ? undefined : navigator.clipboard;
    if (typeof clipboard?.writeText !== 'function') return false;

    try {
        await clipboard.writeText(text);
        return true;
    } catch {
        return false;
    }
}
