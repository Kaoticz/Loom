import { describe, expect, it } from 'vitest';
import { render, waitFor } from '@testing-library/svelte';
import DiffEditor from './DiffEditor.svelte';

/**
 * Tests for the diff editor itself.
 *
 * Monaco could not be loaded in this environment at all, so its wrapper was
 * necessarily untested. CodeMirror constructs under jsdom, so the editor can at
 * least be shown to mount, render both sides of the diff, and stay read only.
 * Layout and highlighting still cannot be asserted: jsdom measures nothing.
 */
describe('DiffEditor', () => {
    it('renders both sides of the diff', async () => {
        const { getByTestId } = render(DiffEditor, {
            path: 'src/main.py',
            original: 'first_line = 1',
            modified: 'first_line = 2'
        });

        await waitFor(() => {
            const text = getByTestId('diff-editor').textContent ?? '';
            expect(text).toContain('first_line = 1');
            expect(text).toContain('first_line = 2');
        });
    });

    it('is not editable', async () => {
        const { getByTestId } = render(DiffEditor, {
            path: 'src/main.py',
            original: 'a',
            modified: 'b'
        });

        await waitFor(() => {
            const editors = getByTestId('diff-editor').querySelectorAll('.cm-content');
            expect(editors.length).toBeGreaterThan(0);
            for (const editor of editors) {
                expect(editor.getAttribute('contenteditable')).not.toBe('true');
            }
        });
    });

    it('mounts for a file it cannot highlight', async () => {
        const { getByTestId } = render(DiffEditor, {
            path: 'notes.unknownext',
            original: 'plain text',
            modified: 'plain text changed'
        });

        await waitFor(() => {
            expect(getByTestId('diff-editor').textContent).toContain('plain text');
        });
    });
});
