/**
 * Deciding whether a scrolling feed should follow new content.
 *
 * The rule lives here rather than in the component so it can be exercised with
 * real numbers. A component test cannot do that: jsdom implements no layout, so
 * every scroll measurement it reports is zero.
 */

/**
 * How far from the bottom still counts as being at the bottom, in pixels.
 *
 * Sub-pixel rounding, fractional device pixel ratios and a partly visible last
 * message all leave a reader who is plainly at the bottom a few pixels short of
 * it. Without a tolerance the feed would stop following for reasons invisible to
 * the person reading it.
 */
export const SCROLL_BOTTOM_TOLERANCE_PX = 32;

/** A scrollable element's position, as the three numbers the decision needs. */
export interface ScrollMetrics {
    /** How far the content has been scrolled down. */
    scrollTop: number;
    /** The full height of the content. */
    scrollHeight: number;
    /** The visible height of the container. */
    clientHeight: number;
}

/**
 * How far the bottom of the content sits below the bottom of the viewport.
 *
 * @param metrics The container's current position.
 * @returns The remaining distance in pixels; zero or less when fully scrolled.
 */
export function distanceFromBottom(metrics: ScrollMetrics): number {
    return metrics.scrollHeight - metrics.scrollTop - metrics.clientHeight;
}

/**
 * Whether the reader is at, or near enough to, the bottom of the feed.
 *
 * Content shorter than its container counts as at the bottom: there is nothing
 * to scroll, so new content should be followed. Negative distances, which
 * happen during elastic overscroll, count too.
 *
 * @param metrics The container's current position.
 * @param tolerance How many pixels short of the bottom still counts.
 * @returns True when the feed should follow new content.
 */
export function isAtBottom(
    metrics: ScrollMetrics,
    tolerance: number = SCROLL_BOTTOM_TOLERANCE_PX
): boolean {
    return distanceFromBottom(metrics) <= tolerance;
}
