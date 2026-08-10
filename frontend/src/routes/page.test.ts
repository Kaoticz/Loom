import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import { tick } from 'svelte';
import Page from './+page.svelte';
import { agentStore } from '$lib/stores/agent';

vi.mock('$app/environment', () => ({ browser: true }));
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));

/**
 * Page-level tests, covering the one thing no component test can see: where
 * things sit relative to each other.
 *
 * Requirement 6.8 is about position, and position is a property of the page
 * rather than of the banner, so it cannot be established by rendering the
 * banner alone.
 */

/** A stand-in for the browser's WebSocket, matching the store's usage. */
class FakeSocket {
    static latest: FakeSocket;

    readyState = 1;
    sent: string[] = [];
    onopen?: () => void;
    onmessage?: (event: { data: string }) => void;
    onclose?: () => void;
    onerror?: (error: unknown) => void;

    constructor(public url: string) {
        FakeSocket.latest = this;
    }

    send(data: string): void {
        this.sent.push(data);
    }

    close(): void {
        this.readyState = 3;
    }

    /** Push one server message through the socket. */
    deliver(payload: object): void {
        this.onmessage?.({ data: JSON.stringify(payload) });
    }
}

/** Where the layout connects the session; the page itself no longer does. */
const WS_URL = 'ws://test/agent-session';

let session = 0;

beforeEach(() => {
    localStorage.clear();
    vi.stubGlobal('WebSocket', FakeSocket);
    // The store is a singleton shared by every test in this file, so a session
    // left open by the previous one would make the next connection a no-op.
    agentStore.disconnect();
    // The page fetches the model list on mount; an empty list is a valid answer.
    vi.stubGlobal(
        'fetch',
        vi.fn(async () => ({ ok: true, json: async () => [] }) as unknown as Response)
    );

    session += 1;
    agentStore.updateConfig({ repoPath: `/test/page-${session}` });
});

/**
 * Render the page against an open session.
 *
 * The connection is established here rather than by the page, because since
 * requirement 1.1 it belongs to the layout: the page is one of several views
 * that come and go while the session stays.
 */
function renderPage() {
    agentStore.connect(WS_URL);
    const socket = FakeSocket.latest;
    socket?.onopen?.();
    const rendered = render(Page);
    return { rendered, socket };
}

/** Report the sandbox as unusable. */
async function reportSandboxDown(socket: FakeSocket): Promise<void> {
    socket.deliver({
        type: 'sandbox_unavailable',
        stage: 'runtime_connectivity',
        reason: 'Cannot connect to the Docker daemon'
    });
    await tick();
}

describe('sandbox banner placement', () => {
    it('shows no banner while the sandbox is usable', () => {
        renderPage();

        expect(screen.queryByTestId('sandbox-banner')).not.toBeInTheDocument();
    });

    it('shows the banner once the sandbox is reported unusable', async () => {
        const { socket } = renderPage();

        await reportSandboxDown(socket);

        expect(screen.getByTestId('sandbox-banner')).toBeInTheDocument();
        expect(screen.getByText(/could not reach the container runtime/i)).toBeInTheDocument();
    });

    it('places the banner below the top bar and above the chat', async () => {
        const { rendered, socket } = renderPage();
        await reportSandboxDown(socket);

        const banner = screen.getByTestId('sandbox-banner');
        const header = rendered.container.querySelector('header');
        const main = rendered.container.querySelector('main');

        // Requirement 6.8. DOCUMENT_POSITION_FOLLOWING means the argument comes
        // after the node it is compared against.
        expect(header).not.toBeNull();
        expect(main).not.toBeNull();
        expect(header?.compareDocumentPosition(banner)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
        expect(banner.compareDocumentPosition(main as Node)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    });

    it('spans the page rather than sitting inside a column', async () => {
        const { rendered, socket } = renderPage();
        await reportSandboxDown(socket);

        const banner = screen.getByTestId('sandbox-banner');
        const main = rendered.container.querySelector('main');

        // Inside `main` it would be confined to the chat or the diff column.
        expect(main?.contains(banner)).toBe(false);
        expect(banner.className).toContain('w-full');
    });

    it('hides the banner once it is dismissed, and does not bring it back', async () => {
        const { socket } = renderPage();
        await reportSandboxDown(socket);

        agentStore.dismissSandboxNotice();
        await tick();
        expect(screen.queryByTestId('sandbox-banner')).not.toBeInTheDocument();

        // Requirement 6.5: a second report must not resurrect it.
        socket.deliver({
            type: 'sandbox_unavailable',
            stage: 'container_start',
            reason: 'and it will not start either'
        });
        await tick();

        expect(screen.queryByTestId('sandbox-banner')).not.toBeInTheDocument();
    });
});

describe('model list failures', () => {
    it('tells the user when the model list fails validation', async () => {
        // Requirement 15.6. A silently empty dropdown reads as a broken app;
        // the reason has to reach the person looking at it, not just the console.
        vi.stubGlobal(
            'fetch',
            vi.fn(
                async () =>
                    ({ ok: true, json: async () => ({ not: 'a list' }) }) as unknown as Response
            )
        );

        renderPage();

        const message = await screen.findByTestId('models-error');
        expect(message.textContent).toContain('model list');
    });

    it('tells the user when the model list cannot be fetched at all', async () => {
        vi.stubGlobal(
            'fetch',
            vi.fn(async () => {
                throw new Error('Network unreachable');
            })
        );

        renderPage();

        expect((await screen.findByTestId('models-error')).textContent).toContain(
            'Network unreachable'
        );
    });

    it('shows nothing when the model list loads', async () => {
        renderPage();
        await tick();

        expect(screen.queryByTestId('models-error')).not.toBeInTheDocument();
    });
});
