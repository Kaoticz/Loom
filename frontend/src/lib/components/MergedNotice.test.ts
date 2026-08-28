import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import MergedNotice from './MergedNotice.svelte';

describe('MergedNotice', () => {
    it('tells the reader to reload without offering an action', () => {
        render(MergedNotice);

        expect(
            screen.getByText('The task is done. Reload the page to start another task.')
        ).toBeInTheDocument();
        expect(screen.queryByRole('button')).not.toBeInTheDocument();
    });
});
