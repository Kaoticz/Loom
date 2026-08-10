import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import { createRawSnippet, tick, type Component } from 'svelte';
import { get } from 'svelte/store';
import Layout from './+layout.svelte';
import Page from './+page.svelte';
import Settings from './settings/+page.svelte';
import { agentStore } from '$lib/stores/agent';

vi.mock('$app/environment', () => ({ browser: true }));
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));

/**
 * Tests for the one thing the session lifecycle depends on: where it lives.
 *
 * Requirement 1.1 is not a property of any single view. It is the claim that
 * moving between views leaves the session alone, which can only be established
 * by mounting and unmounting views around a layout that stays put — exactly
 * what SvelteKit does on a client-side navigation, where the layout component
 * is kept and only the page beneath it is swapped. There is one `+layout.svelte`
 * in this application and every route sits under it, so it is mounted once for
 * the life of the tab.
 *
 * The components are imported statically and share the one real store, as the
 * other route tests do. Re-importing them per test would hand the components a
 * second copy of the Svelte runtime, which cannot mount.
 */

/** A stand-in for the browser's WebSocket, matching the store's usage. */
class FakeSocket {
    static latest: FakeSocket | null = null;
    static instances: FakeSocket[] = [];

    readyState = 1;
    sent: string[] = [];
    onopen?: () => void;
    onmessage?: (event: { data: string }) => void;
    onclose?: () => void;
    onerror?: (error: unknown) => void;

    constructor(public url: string) {
        FakeSocket.latest = this;
        FakeSocket.instances.push(this);
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

/** A child slot for the layout, standing in for whichever route is current. */
const emptyChildren = createRawSnippet(() => ({
    render: () => '<div data-testid="route-outlet"></div>'
}));

let session = 0;

beforeEach(() => {
    vi.unstubAllGlobals();
    vi.stubGlobal('WebSocket', FakeSocket);
    // Both routes fetch on mount: a model list, and on Settings a description of
    // the embedding model. Answering both keeps a failure in the log meaningful.
    vi.stubGlobal(
        'fetch',
        vi.fn(async (url: string) => {
            const body = String(url).includes('/description')
                ? { dimensions: 768, context_length: 2048 }
                : [];
            return { ok: true, json: async () => body } as unknown as Response;
        })
    );

    // The store is a singleton shared by every test in this file. Ending the
    // session and changing the repository clears what the previous test left in
    // it, which is how the other route tests isolate themselves.
    agentStore.disconnect();
    session += 1;
    agentStore.updateConfig({ repoPath: `/test/layout-${session}` });

    FakeSocket.latest = null;
    FakeSocket.instances = [];
});

/** Mount the layout, which is what establishes the session. */
function startApp() {
    const mounted = render(Layout, { props: { children: emptyChildren } });
    FakeSocket.latest?.onopen?.();
    return mounted;
}

/**
 * Mount a route, then leave it, as a client-side navigation would.
 *
 * The layout is untouched throughout, which is the whole point: it is the
 * component SvelteKit keeps.
 */
async function visit(route: Component): Promise<void> {
    const view = render(route);
    await tick();
    view.unmount();
    await tick();
}

describe('the session lifecycle', () => {
    it('opens exactly one session when the application starts', async () => {
        startApp();

        expect(FakeSocket.instances).toHaveLength(1);
        expect(FakeSocket.latest?.url).toContain('agent-session');
        expect(get(agentStore).connection).toBe('connected');
    });

    it('keeps the session open across navigation between views', async () => {
        // Requirement 1.1. Connecting on the main view's mount and disconnecting
        // on its destroy meant a trip to Settings ended the session, and the
        // session built on the way back could be refused as busy while the
        // previous one was still being reaped.
        startApp();

        await visit(Page);
        await visit(Settings);
        await visit(Page);

        expect(FakeSocket.instances).toHaveLength(1);
        expect(get(agentStore).connection).toBe('connected');
    });

    it('preserves the feed and a running task across navigation', async () => {
        startApp();
        FakeSocket.latest?.deliver({
            type: 'system',
            content: 'Executing step 1',
            status: 'executing'
        });

        await visit(Page);
        await visit(Settings);
        await visit(Page);

        const state = get(agentStore);
        expect(state.messages.map((message) => message.content)).toContain('Executing step 1');
        expect(state.status).toBe('executing');
    });

    it('shows no notice about another tab under any navigation sequence', async () => {
        // The symptom this plan starts from: one tab, no sibling anywhere, and a
        // notice claiming the session was held elsewhere.
        startApp();

        await visit(Settings);
        await visit(Page);
        render(Page);
        await tick();

        expect(get(agentStore).blocked).toBeNull();
        expect(screen.queryByText(/already running in another tab/i)).not.toBeInTheDocument();
        expect(screen.queryByText(/previous session is still closing/i)).not.toBeInTheDocument();
    });

    it('establishes one session when Settings is the first view seen', async () => {
        // Arriving at /settings directly and moving to the main view afterwards
        // must not leave the application either sessionless or doubly connected.
        startApp();

        await visit(Settings);
        await visit(Page);

        expect(FakeSocket.instances).toHaveLength(1);
        expect(get(agentStore).connection).toBe('connected');
    });
});
