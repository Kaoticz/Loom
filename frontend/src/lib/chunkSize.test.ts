import { describe, expect, it } from 'vitest';
import { DEFAULT_CHUNK_LINES, MAX_CHUNK_LINES, MIN_CHUNK_LINES, readChunkLines } from './chunkSize';

describe('readChunkLines', () => {
    it('takes a number that was typed', () => {
        expect(readChunkLines('80')).toBe(80);
    });

    it('holds a value below the minimum up to it', () => {
        expect(readChunkLines('2')).toBe(MIN_CHUNK_LINES);
    });

    it('holds a value above the maximum down to it', () => {
        expect(readChunkLines('99999')).toBe(MAX_CHUNK_LINES);
    });

    it('stands in the default for an empty box', () => {
        // Mid-edit, not a request for automatic sizing: only the Auto control
        // chooses that, because it is an explicit choice.
        expect(readChunkLines('')).toBe(DEFAULT_CHUNK_LINES);
    });

    it('stands in the default for something unreadable', () => {
        expect(readChunkLines('abc')).toBe(DEFAULT_CHUNK_LINES);
    });

    it('never returns a value the backend would silently change', () => {
        for (const raw of ['', '0', '-40', '1', '10', '150', '1000', '5000', 'x']) {
            const value = readChunkLines(raw);
            expect(value).toBeGreaterThanOrEqual(MIN_CHUNK_LINES);
            expect(value).toBeLessThanOrEqual(MAX_CHUNK_LINES);
            expect(Number.isInteger(value)).toBe(true);
        }
    });

    it('offers a default inside its own bounds', () => {
        expect(DEFAULT_CHUNK_LINES).toBeGreaterThanOrEqual(MIN_CHUNK_LINES);
        expect(DEFAULT_CHUNK_LINES).toBeLessThanOrEqual(MAX_CHUNK_LINES);
    });
});
