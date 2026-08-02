import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, waitFor } from '@testing-library/svelte';
import { tick } from 'svelte';
import DiffEditor from './DiffEditor.svelte';
import { loadLanguage } from '$lib/editor/languages';

/**
 * The rebuild race.
 *
 * Building the view is asynchronous because the parser is fetched first, so two
 * rebuilds started in quick succession can finish in either order. The loader is
 * mocked here so that order can be controlled: without controlling it the test
 * would pass whether or not the guard exists, since parsers are cached and
 * normally resolve immediately.
 */
vi.mock('$lib/editor/languages', async (importOriginal) => ({
    ...(await importOriginal<typeof import('$lib/editor/languages')>()),
    loadLanguage: vi.fn()
}));

const mockedLoad = vi.mocked(loadLanguage);

describe('DiffEditor rebuild ordering', () => {
    beforeEach(() => {
        mockedLoad.mockReset();
    });

    it('does not rebuild the view when an older build finishes last', async () => {
        const finish: (() => void)[] = [];
        mockedLoad.mockImplementation(
            () => new Promise((resolve) => finish.push(() => resolve(null)))
        );

        const { getByTestId, rerender } = render(DiffEditor, {
            path: 'old.py',
            original: 'OLD CONTENT',
            modified: 'OLD CONTENT changed'
        });
        await tick();

        await rerender({
            path: 'new.rs',
            original: 'NEW CONTENT',
            modified: 'NEW CONTENT changed'
        });
        await tick();

        expect(finish.length).toBe(2);

        // The newer build completes first and installs its view.
        finish[1]();
        await waitFor(() => {
            expect(getByTestId('diff-editor').querySelector('.cm-mergeView')).not.toBeNull();
        });
        const installed = getByTestId('diff-editor').querySelector('.cm-mergeView');

        // Then the older one catches up. It must abandon itself: tearing the
        // view down and rebuilding would discard a correct view and rebuild it
        // with the previous file's language applied to the current content.
        finish[0]();
        await tick();
        await tick();

        const now = getByTestId('diff-editor').querySelector('.cm-mergeView');
        expect(now).toBe(installed);
        expect(installed?.isConnected).toBe(true);
        expect(getByTestId('diff-editor').textContent).toContain('NEW CONTENT');
    });

    it('shows the newest file when builds finish in order', async () => {
        const finish: (() => void)[] = [];
        mockedLoad.mockImplementation(
            () => new Promise((resolve) => finish.push(() => resolve(null)))
        );

        const { getByTestId, rerender } = render(DiffEditor, {
            path: 'old.py',
            original: 'OLD CONTENT',
            modified: 'OLD CONTENT changed'
        });
        await tick();
        await rerender({
            path: 'new.rs',
            original: 'NEW CONTENT',
            modified: 'NEW CONTENT changed'
        });
        await tick();

        finish[0]();
        await tick();
        finish[1]();
        await tick();

        await waitFor(() => {
            expect(getByTestId('diff-editor').textContent).toContain('NEW CONTENT');
        });
    });
});
