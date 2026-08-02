import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/svelte';
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
