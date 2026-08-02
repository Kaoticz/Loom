/**
 * The chunk size setting, which controls how much code each embedding covers.
 *
 * Bounds mirror the backend's, which enforces them again because the value
 * arrives from the client. Duplicating them here is for the person typing, not
 * for safety: a control that lets you enter a number the server will silently
 * change is worse than one that will not let you enter it.
 */

/** Fewest lines a chunk may hold; below this it carries too little meaning. */
export const MIN_CHUNK_LINES = 10;

/** Most lines a chunk may hold; above this no embedding model will take it. */
export const MAX_CHUNK_LINES = 1000;

/** The size offered when someone turns Auto off, matching the backend's own default. */
export const DEFAULT_CHUNK_LINES = 150;

/**
 * Interpret what was typed into the chunk size box.
 *
 * An empty or unreadable box means the person is mid-edit rather than asking
 * for automatic sizing, so the default stands in until they finish. Only the
 * Auto control selects automatic sizing, because that is an explicit choice.
 *
 * @param raw The raw value from the input.
 * @returns The chunk size to store.
 */
export function readChunkLines(raw: string): number {
    const parsed = Number.parseInt(raw, 10);
    if (Number.isNaN(parsed)) return DEFAULT_CHUNK_LINES;
    return Math.max(MIN_CHUNK_LINES, Math.min(parsed, MAX_CHUNK_LINES));
}
