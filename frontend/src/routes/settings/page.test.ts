import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { tick } from 'svelte';
import Settings from './+page.svelte';
import { agentStore } from '$lib/stores/agent';
import { DEFAULT_CHUNK_LINES, MAX_CHUNK_LINES, MIN_CHUNK_LINES } from '$lib/chunkSize';

vi.mock('$app/environment', () => ({ browser: true }));
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));

/**
 * Tests for the settings a person actually manipulates.
 *
 * The chunk size logic is covered on its own in `chunkSize.test.ts`; what these
 * establish is that the controls are wired to it — that turning Auto off puts a
 * real number into the configuration, and that what a model reports about itself
 * reaches the screen.
 */

/** Reply to the endpoints the settings page calls on mount. */
function respondWith(description: unknown, models: string[] = ['nomic-embed-text']) {
    vi.stubGlobal(
        'fetch',
        vi.fn(async (url: string) => {
            const body = String(url).includes('/description') ? description : models;
            return { ok: true, json: async () => body } as unknown as Response;
        })
    );
}

beforeEach(() => {
    localStorage.clear();
    respondWith({ dimensions: 768, context_length: 2048 });
    agentStore.updateConfig({ chunkLines: null });
});

/** Render the settings page and let its mount-time requests settle. */
async function renderSettings() {
    const rendered = render(Settings);
    await tick();
    await tick();
    return rendered;
}

describe('chunk size control', () => {
    it('starts on Auto', async () => {
        await renderSettings();

        expect(screen.getByRole('checkbox', { name: /auto/i })).toBeChecked();
        expect(screen.getByLabelText('Chunk Size')).toBeDisabled();
    });

    it('offers a real number once Auto is turned off', async () => {
        await renderSettings();

        await fireEvent.click(screen.getByRole('checkbox', { name: /auto/i }));

        // Leaving it null here would make the box look editable while still
        // meaning automatic.
        const box = screen.getByLabelText('Chunk Size') as HTMLInputElement;
        expect(box).toBeEnabled();
        expect(box.value).toBe(String(DEFAULT_CHUNK_LINES));
    });

    it('returns to Auto when the box is ticked again', async () => {
        await renderSettings();
        const auto = screen.getByRole('checkbox', { name: /auto/i });

        await fireEvent.click(auto);
        await fireEvent.click(auto);

        expect(screen.getByLabelText('Chunk Size')).toBeDisabled();
    });

    it('states the bounds the backend will enforce', async () => {
        await renderSettings();

        const box = screen.getByLabelText('Chunk Size');
        expect(box).toHaveAttribute('min', String(MIN_CHUNK_LINES));
        expect(box).toHaveAttribute('max', String(MAX_CHUNK_LINES));
    });

    it('explains that a set size can fail and that changing it re-indexes', async () => {
        await renderSettings();

        // Both are consequences a person should know before choosing a number.
        expect(screen.getByText(/indexing fails/i)).toBeInTheDocument();
        expect(screen.getByText(/re-indexes the repository/i)).toBeInTheDocument();
    });
});

describe('embedding model description', () => {
    it('reports what the model says about itself', async () => {
        await renderSettings();

        const description = screen.getByTestId('embedding-model-description');
        expect(description.textContent).toContain('768');
        expect(description.textContent).toContain('2048');
    });

    it('says so when the model reports nothing', async () => {
        respondWith({ dimensions: null, context_length: null });

        await renderSettings();

        expect(screen.getByTestId('embedding-model-description').textContent).toMatch(
            /does not report/i
        );
    });

    it('does not claim a width when only the capacity is known', async () => {
        respondWith({ dimensions: null, context_length: 8192 });

        await renderSettings();

        const description = screen.getByTestId('embedding-model-description');
        expect(description.textContent).toContain('8192');
        expect(description.textContent).toMatch(/unreported/i);
    });

    it('survives a description that cannot be fetched', async () => {
        vi.stubGlobal(
            'fetch',
            vi.fn(async (url: string) => {
                if (String(url).includes('/description')) throw new Error('offline');
                return { ok: true, json: async () => ['nomic-embed-text'] } as unknown as Response;
            })
        );

        await renderSettings();

        // A missing description must not take the settings page down with it.
        expect(screen.getByTestId('embedding-model-description')).toBeInTheDocument();
        expect(screen.getByLabelText('Chunk Size')).toBeInTheDocument();
    });
});

describe('removing indexed data', () => {
    /**
     * Reply to every endpoint the page calls, with a controllable index status.
     *
     * @param status What `/index/status` should report.
     * @param onDelete What `DELETE /index` should return, or an Error to throw.
     */
    function respondWithIndex(status: unknown, onDelete?: () => unknown) {
        vi.stubGlobal(
            'fetch',
            vi.fn(async (url: string, init?: RequestInit) => {
                const target = String(url);
                if (init?.method === 'DELETE') {
                    const result = onDelete?.() ?? {
                        project: '/repos/mine',
                        embeddings_deleted: 4,
                        files_deleted: 2,
                        symbols_deleted: 3,
                        message:
                            'Removed 2 indexed files, 3 symbols and 4 embeddings. The project is re-indexed when a session next starts.'
                    };
                    if (result instanceof Error) throw result;
                    return { ok: true, json: async () => result } as unknown as Response;
                }
                if (target.includes('/index/status')) {
                    return { ok: true, json: async () => status } as unknown as Response;
                }
                if (target.includes('/description')) {
                    return {
                        ok: true,
                        json: async () => ({ dimensions: 768, context_length: 2048 })
                    } as unknown as Response;
                }
                return { ok: true, json: async () => ['nomic-embed-text'] } as unknown as Response;
            })
        );
    }

    const indexed = {
        project: '/repos/mine',
        indexed_files: 2,
        can_delete: true,
        blocked_reason: ''
    };

    /**
     * Read the explanation the remove button points at.
     *
     * Read by id rather than by text: it is the element the button references
     * with `aria-describedby`, so this asserts on the thing a screen reader
     * would actually announce.
     */
    function reason(): string {
        return document.getElementById('index-delete-reason')?.textContent?.trim() ?? '';
    }

    /** The remove button, which stays clickable even when marked disabled. */
    function removeButton(): HTMLElement {
        return screen.getByRole('button', { name: /remove indexed data/i });
    }

    /** Whether a DELETE has been sent. */
    function deleteWasSent(): boolean {
        const calls = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls;
        return calls.some(([, init]) => (init as RequestInit | undefined)?.method === 'DELETE');
    }

    it('says how much is indexed and offers to remove it', async () => {
        respondWithIndex(indexed);
        await renderSettings();

        await waitFor(() => expect(reason()).toMatch(/2 files are indexed/i));
        expect(removeButton()).toHaveAttribute('aria-disabled', 'false');
    });

    it('counts a single file as one file', async () => {
        respondWithIndex({ ...indexed, indexed_files: 1 });
        await renderSettings();

        // Sloppy wording in something read constantly is worth one assertion.
        await waitFor(() => expect(reason()).toMatch(/1 file is indexed/i));
    });

    it('asks for confirmation before removing anything', async () => {
        respondWithIndex(indexed);
        await renderSettings();
        await waitFor(() => expect(reason()).toMatch(/2 files are indexed/i));

        await fireEvent.click(removeButton());

        // The confirmation is identified by its own action, not by prose that
        // also appears in the section's description.
        expect(screen.getByRole('button', { name: /yes, remove it/i })).toBeInTheDocument();
        expect(deleteWasSent()).toBe(false);
    });

    it('reports what was removed, and that re-indexing waits for a session', async () => {
        respondWithIndex(indexed);
        await renderSettings();
        await waitFor(() => expect(reason()).toMatch(/2 files are indexed/i));

        await fireEvent.click(removeButton());
        await fireEvent.click(screen.getByRole('button', { name: /yes, remove it/i }));

        const outcome = await screen.findByRole('status');
        expect(outcome).toHaveTextContent(/removed 2 indexed files/i);
        expect(outcome).toHaveTextContent(/session next starts/i);
    });

    it('does not act when disabled, and says why where it can be read', async () => {
        // The state the backend reports while a session is running.
        respondWithIndex({
            project: '/repos/mine',
            indexed_files: 2,
            can_delete: false,
            blocked_reason: 'A session is running. Finish or stop the session first.'
        });
        await renderSettings();

        // The reason is on the page rather than only in a tooltip, because
        // keyboard and touch users never hover.
        await waitFor(() => expect(reason()).toMatch(/a session is running/i));

        const button = removeButton();
        expect(button).toHaveAttribute('aria-disabled', 'true');
        expect(button).toHaveAttribute('aria-describedby', 'index-delete-reason');

        // Asserted as behaviour, not as an attribute: the button stays clickable
        // on purpose so its explanation stays reachable, so what matters is that
        // clicking it does nothing.
        await fireEvent.click(button);

        expect(screen.queryByRole('button', { name: /yes, remove it/i })).not.toBeInTheDocument();
        expect(deleteWasSent()).toBe(false);
    });

    it('follows the repository path rather than describing the previous one', async () => {
        // Two repositories with different indexes. The section must describe
        // whichever is currently entered, because the remove button acts on that
        // one — if they disagree, the count shown is about a project the button
        // will not touch, and nothing says so.
        vi.stubGlobal(
            'fetch',
            vi.fn(async (url: string) => {
                const target = String(url);
                if (target.includes('/index/status')) {
                    const forSecond = target.includes('second');
                    return {
                        ok: true,
                        json: async () => ({
                            project: forSecond ? '/repos/second' : '/repos/first',
                            indexed_files: forSecond ? 7 : 2,
                            can_delete: true,
                            blocked_reason: ''
                        })
                    } as unknown as Response;
                }
                if (target.includes('/repos/validate')) {
                    return { ok: true, json: async () => ({ valid: true }) } as unknown as Response;
                }
                if (target.includes('/description')) {
                    return {
                        ok: true,
                        json: async () => ({ dimensions: 768, context_length: 2048 })
                    } as unknown as Response;
                }
                return { ok: true, json: async () => ['nomic-embed-text'] } as unknown as Response;
            })
        );

        agentStore.updateConfig({ repoPath: '/repos/first' });
        await renderSettings();
        await waitFor(() => expect(reason()).toMatch(/2 files are indexed/i));

        const input = screen.getByLabelText(/repository path/i);
        await fireEvent.input(input, { target: { value: '/repos/second' } });
        await fireEvent.blur(input);

        await waitFor(() => expect(reason()).toMatch(/7 files are indexed/i));
    });

    it('surfaces a failure rather than looking as though it worked', async () => {
        respondWithIndex(indexed, () => new Error('the database is unreachable'));
        await renderSettings();
        await waitFor(() => expect(reason()).toMatch(/2 files are indexed/i));

        await fireEvent.click(removeButton());
        await fireEvent.click(screen.getByRole('button', { name: /yes, remove it/i }));

        expect(await screen.findByRole('alert')).toHaveTextContent(/was not removed/i);
        expect(screen.queryByRole('status')).not.toBeInTheDocument();
    });
});
