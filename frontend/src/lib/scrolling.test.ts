import { describe, expect, it } from 'vitest';
import {
    SCROLL_BOTTOM_TOLERANCE_PX,
    distanceFromBottom,
    isAtBottom,
    type ScrollMetrics
} from './scrolling';

/** A container 400px tall showing 1000px of content, scrolled to a given offset. */
function scrolledTo(scrollTop: number): ScrollMetrics {
    return { scrollTop, scrollHeight: 1000, clientHeight: 400 };
}

describe('distanceFromBottom', () => {
    it('is zero when fully scrolled', () => {
        expect(distanceFromBottom(scrolledTo(600))).toBe(0);
    });

    it('grows as the reader scrolls up', () => {
        expect(distanceFromBottom(scrolledTo(500))).toBe(100);
        expect(distanceFromBottom(scrolledTo(0))).toBe(600);
    });
});

describe('isAtBottom', () => {
    it('counts an exactly bottomed feed', () => {
        expect(isAtBottom(scrolledTo(600))).toBe(true);
    });

    it('counts a feed a few pixels short of the bottom', () => {
        // Requirement 4.1: being slightly short still counts as at the bottom.
        expect(isAtBottom(scrolledTo(599))).toBe(true);
        expect(isAtBottom(scrolledTo(600 - SCROLL_BOTTOM_TOLERANCE_PX))).toBe(true);
    });

    it('does not count a feed scrolled beyond the tolerance', () => {
        expect(isAtBottom(scrolledTo(600 - SCROLL_BOTTOM_TOLERANCE_PX - 1))).toBe(false);
    });

    it('does not count a feed scrolled well up', () => {
        // Requirement 4.2: a reader who has scrolled up is left alone.
        expect(isAtBottom(scrolledTo(0))).toBe(false);
        expect(isAtBottom(scrolledTo(200))).toBe(false);
    });

    it('counts content shorter than its container', () => {
        // Nothing to scroll, so new content should be followed.
        expect(isAtBottom({ scrollTop: 0, scrollHeight: 120, clientHeight: 400 })).toBe(true);
    });

    it('counts an empty feed', () => {
        expect(isAtBottom({ scrollTop: 0, scrollHeight: 0, clientHeight: 0 })).toBe(true);
    });

    it('counts elastic overscroll past the bottom', () => {
        // Momentum scrolling can report a position beyond the end.
        expect(isAtBottom(scrolledTo(640))).toBe(true);
    });

    it('tolerates sub-pixel positions', () => {
        // Fractional device pixel ratios leave fractional offsets behind.
        expect(isAtBottom(scrolledTo(599.5))).toBe(true);
        expect(isAtBottom({ scrollTop: 600, scrollHeight: 1000.4, clientHeight: 400 })).toBe(true);
    });

    it('honours an explicit tolerance', () => {
        expect(isAtBottom(scrolledTo(590), 20)).toBe(true);
        expect(isAtBottom(scrolledTo(590), 5)).toBe(false);
    });

    it('treats a zero tolerance as requiring the exact bottom', () => {
        expect(isAtBottom(scrolledTo(600), 0)).toBe(true);
        expect(isAtBottom(scrolledTo(599), 0)).toBe(false);
    });

    it('uses a tolerance small enough not to swallow a real scroll', () => {
        // Generous enough for rounding, but well under one message's height, so
        // deliberately scrolling up by a line still stops the feed following.
        expect(SCROLL_BOTTOM_TOLERANCE_PX).toBeGreaterThan(0);
        expect(SCROLL_BOTTOM_TOLERANCE_PX).toBeLessThan(64);
    });
});
