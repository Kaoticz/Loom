import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';

import DirectoryPicker from '$lib/components/DirectoryPicker.svelte';

/**
 * Choosing a repository from the filesystem.
 *
 * This had no tests of any kind, and it writes the repository path that the rest
 * of the application is scoped to — the index that gets built, searched, and
 * deleted all follow from what is chosen here. A picker that reported the wrong
 * path would not fail visibly; it would simply point everything at the wrong
 * project.
 */
describe('directory picker', () => {
    /**
     * Answer the listing endpoint with a directory tree.
     *
     * @param tree Directory names keyed by the path being listed.
     */
    function respondWith(tree: Record<string, string[]>) {
        vi.stubGlobal(
            'fetch',
            vi.fn(async (url: string) => {
                const path = decodeURIComponent(String(url).split('path=')[1] ?? '');
                const listing = tree[path];

                if (listing === undefined) {
                    return { ok: false, json: async () => ({}) } as unknown as Response;
                }
                return { ok: true, json: async () => listing } as unknown as Response;
            })
        );
    }

    beforeEach(() => {
        vi.unstubAllGlobals();
    });

    it('lists the directories at the path it was opened on', async () => {
        respondWith({ '/repos': ['alpha', 'beta'] });

        render(DirectoryPicker, { initialPath: '/repos', onclose: () => {}, onselect: () => {} });

        expect(await screen.findByText('alpha')).toBeInTheDocument();
        expect(screen.getByText('beta')).toBeInTheDocument();
    });

    it('returns the path that is actually showing, not the one it started on', async () => {
        // The failure worth guarding: reporting the opening path after the user
        // has navigated would point the whole application at the wrong project,
        // silently and plausibly.
        respondWith({ '/repos': ['alpha'], '/repos/alpha': [] });
        const chosen: string[] = [];

        render(DirectoryPicker, {
            initialPath: '/repos',
            onclose: () => {},
            onselect: (path: string) => chosen.push(path)
        });

        await fireEvent.click(await screen.findByText('alpha'));
        await waitFor(() => expect(screen.getByText(/\/repos\/alpha/)).toBeInTheDocument());
        await fireEvent.click(screen.getByRole('button', { name: /select current/i }));

        expect(chosen).toEqual(['/repos/alpha']);
    });

    it('reports a failed listing rather than showing an empty directory', async () => {
        // An empty list and a failed request look identical to a reader, and one
        // of them means the directory is empty while the other means nothing is
        // known at all.
        respondWith({});

        render(DirectoryPicker, { initialPath: '/repos', onclose: () => {}, onselect: () => {} });

        expect(await screen.findByText(/failed to load directories/i)).toBeInTheDocument();
    });

    it('leaves the path alone when a listing fails', async () => {
        // Moving to a directory that could not be read would strand the reader
        // somewhere the picker cannot show them.
        respondWith({ '/repos': ['alpha'] });

        render(DirectoryPicker, { initialPath: '/repos', onclose: () => {}, onselect: () => {} });
        await screen.findByText('alpha');

        await fireEvent.click(screen.getByText('alpha'));

        await waitFor(() => expect(screen.getByText(/failed to load/i)).toBeInTheDocument());
        expect(screen.getByText(/\/repos$/)).toBeInTheDocument();
    });

    it('closes without choosing anything when cancelled', async () => {
        respondWith({ '/repos': [] });
        let closed = false;
        const chosen: string[] = [];

        render(DirectoryPicker, {
            initialPath: '/repos',
            onclose: () => (closed = true),
            onselect: (path: string) => chosen.push(path)
        });

        await fireEvent.click(await screen.findByRole('button', { name: /cancel/i }));

        expect(closed).toBe(true);
        expect(chosen).toEqual([]);
    });
});
