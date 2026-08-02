import { afterEach, describe, expect, it } from 'vitest';
import { copyGlyph, copyText } from './clipboard';

/** Install a clipboard that either accepts or refuses writes. */
function withClipboard(writeText: (text: string) => Promise<void>) {
    Object.defineProperty(navigator, 'clipboard', {
        value: { writeText },
        configurable: true
    });
}

afterEach(() => {
    // jsdom provides no clipboard, which is the state to restore.
    Reflect.deleteProperty(navigator, 'clipboard');
});

describe('copyText', () => {
    it('reports failure when the clipboard is unavailable', async () => {
        // Requirement 12.5. jsdom implements no clipboard, which is the same
        // condition a browser presents outside a secure context.
        expect(navigator.clipboard).toBeUndefined();

        expect(await copyText('anything')).toBe(false);
    });

    it('reports failure when the write is refused', async () => {
        withClipboard(async () => {
            throw new Error('permission denied');
        });

        expect(await copyText('anything')).toBe(false);
    });

    it('reports failure when the clipboard exists but cannot write', async () => {
        Object.defineProperty(navigator, 'clipboard', { value: {}, configurable: true });

        expect(await copyText('anything')).toBe(false);
    });

    it('copies the text and reports success', async () => {
        const written: string[] = [];
        withClipboard(async (text) => {
            written.push(text);
        });

        expect(await copyText('the summary')).toBe(true);
        expect(written).toEqual(['the summary']);
    });

    it('copies text verbatim, including newlines', async () => {
        const written: string[] = [];
        withClipboard(async (text) => {
            written.push(text);
        });

        await copyText('first line\nsecond line');

        expect(written[0]).toBe('first line\nsecond line');
    });

    it('does not throw when the clipboard rejects', async () => {
        withClipboard(async () => {
            throw new Error('nope');
        });

        await expect(copyText('anything')).resolves.toBe(false);
    });
});

describe('copyGlyph', () => {
    it('offers the copy character when nothing has happened', () => {
        expect(copyGlyph('idle')).toBe('⮺');
    });

    it('confirms a copy', () => {
        expect(copyGlyph('copied')).toBe('✓');
    });

    it('marks a failure distinctly from a success', () => {
        // Requirement 12.5: a failure must be indicated, not silently ignored.
        expect(copyGlyph('failed')).not.toBe(copyGlyph('copied'));
        expect(copyGlyph('failed')).not.toBe(copyGlyph('idle'));
    });
});
