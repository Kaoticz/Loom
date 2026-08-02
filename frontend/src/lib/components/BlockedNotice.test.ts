import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/svelte';
import BlockedNotice from './BlockedNotice.svelte';

describe('BlockedNotice', () => {
    it('explains that another tab holds the session', () => {
        render(BlockedNotice, { reason: 'another_tab', onretry: () => {} });

        expect(screen.getByText(/already running in another tab/)).toBeInTheDocument();
        expect(screen.getByText(/Browsers do not allow one tab/)).toBeInTheDocument();
    });

    it('explains a session that is still closing', () => {
        render(BlockedNotice, { reason: 'stale', onretry: () => {} });

        expect(screen.getByText(/still closing/)).toBeInTheDocument();
        expect(screen.getByText(/Retrying usually works/)).toBeInTheDocument();
    });

    it('offers a retry either way', () => {
        render(BlockedNotice, { reason: 'stale', onretry: () => {} });

        expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
    });

    it('asks its caller to retry, rather than reaching for the store', async () => {
        const onretry = vi.fn();
        render(BlockedNotice, { reason: 'stale', onretry });

        await fireEvent.click(screen.getByRole('button', { name: 'Retry' }));

        expect(onretry).toHaveBeenCalledOnce();
    });
});
