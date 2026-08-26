import { describe, expect, it } from 'vitest';
import {
    DEFAULT_CONTEXT_WINDOW,
    MAX_CONTEXT_WINDOW,
    MIN_CONTEXT_WINDOW,
    readContextWindow
} from './contextWindow';

describe('readContextWindow', () => {
    it('keeps a plausible number as typed', () => {
        expect(readContextWindow('32768')).toBe(32768);
    });

    it('treats an unreadable box as mid-edit rather than as a choice', () => {
        // Only the Auto control selects the default, because that is explicit.
        expect(readContextWindow('')).toBe(DEFAULT_CONTEXT_WINDOW);
        expect(readContextWindow('lots')).toBe(DEFAULT_CONTEXT_WINDOW);
    });

    it('clamps to the bounds the backend will enforce', () => {
        expect(readContextWindow('1')).toBe(MIN_CONTEXT_WINDOW);
        expect(readContextWindow('-5000')).toBe(MIN_CONTEXT_WINDOW);
        expect(readContextWindow('999999999')).toBe(MAX_CONTEXT_WINDOW);
    });
});
