import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import ChatHeader from './ChatHeader.svelte';

describe('ChatHeader', () => {
    it('reports the connection state', () => {
        render(ChatHeader, { connection: 'connected', status: 'idle' });

        expect(screen.getByText('connected')).toBeInTheDocument();
    });

    it('reports the session status in words rather than identifiers', () => {
        render(ChatHeader, { connection: 'connected', status: 'waiting_approval' });

        expect(screen.getByText('waiting approval')).toBeInTheDocument();
    });

    it('distinguishes the connection states visually', () => {
        const { container, unmount } = render(ChatHeader, {
            connection: 'connected',
            status: 'idle'
        });
        const connected = container.querySelector('span[title]')?.className;
        unmount();

        render(ChatHeader, { connection: 'disconnected', status: 'idle' });
        const disconnected = screen.getByTitle(/disconnected/).className;

        expect(disconnected).not.toBe(connected);
    });
});
