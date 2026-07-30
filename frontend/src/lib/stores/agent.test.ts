import { beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';

vi.mock('$app/environment', () => ({ browser: true }));

/** The store's real type, so state assertions are checked rather than `unknown`. */
type AgentStore = (typeof import('./agent'))['agentStore'];

/** The newest chat message, asserting one exists. */
function latest(store: AgentStore) {
    const messages = get(store).messages;
    const message = messages[messages.length - 1];
    if (!message) throw new Error('no message was recorded');
    return message;
}

/**
 * A stand-in for the browser's WebSocket.
 *
 * The store assigns its handlers as properties, so tests drive it by invoking
 * those handlers directly and inspect what the store sent back.
 */
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

    close(): void {}

    /** Push one server message through the socket. */
    deliver(payload: object): void {
        this.onmessage?.({ data: JSON.stringify(payload) });
    }

    /** The most recently sent client message, parsed. */
    get lastSent(): Record<string, unknown> {
        return JSON.parse(this.sent[this.sent.length - 1]);
    }
}

/** Import the store fresh, connected to a new fake socket. */
async function connected(): Promise<{ store: AgentStore; socket: FakeSocket }> {
    localStorage.clear();
    vi.resetModules();
    vi.stubGlobal('WebSocket', FakeSocket);

    const store = (await import('./agent')).agentStore;
    store.connect('ws://test/agent-session');
    const socket = FakeSocket.latest;
    socket.onopen?.();
    return { store, socket };
}

describe('connection state', () => {
    it('reports connected once the socket opens', async () => {
        const { store } = await connected();
        expect(get(store).connection).toBe('connected');
    });

    it('reports disconnected when the socket closes', async () => {
        const { store, socket } = await connected();
        vi.stubGlobal('setTimeout', vi.fn());
        socket.onclose?.();
        expect(get(store).connection).toBe('disconnected');
    });

    it('schedules a reconnect after closing', async () => {
        const { socket } = await connected();
        const scheduled = vi.fn();
        vi.stubGlobal('setTimeout', scheduled);

        socket.onclose?.();

        expect(scheduled).toHaveBeenCalledOnce();
        expect(scheduled.mock.calls[0][1]).toBe(3000);
    });
});

describe('message handling', () => {
    it('renders a plan from its steps and asks for approval', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'plan', steps: ['Read the file', 'Edit it'], requires_approval: true });

        const state = get(store);
        expect(state.status).toBe('waiting_approval');
        expect(state.requiresAction).toBe(true);
        expect(state.actionType).toBe('approve_plan');
        expect(latest(store).content).toBe('1. Read the file\n2. Edit it');
    });

    it('summarises a tool action rather than showing nothing', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'action', tool_name: 'edit_file', arguments: { path: 'a.py' } });

        expect(latest(store).content).toBe('edit_file(path=a.py)');
    });

    it('shows a tool result for an observation', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'observation', tool_name: 'edit_file', result: 'Edited a.py', status: 'success' });

        expect(latest(store).content).toBe('Edited a.py');
    });

    it('shows a commit with a short hash', async () => {
        const { store, socket } = await connected();

        socket.deliver({
            type: 'commit',
            commit_hash: '3f2a1b9c4d5e6f70',
            message: 'Edit a.py',
            files_changed: ['a.py']
        });

        expect(latest(store).content).toBe('Edit a.py (3f2a1b9)');
    });

    it('populates both sides of a diff', async () => {
        const { store, socket } = await connected();

        socket.deliver({
            type: 'diff',
            file_path: 'a.py',
            original: 'value = 1\n',
            modified: 'value = 2\n',
            truncated: false
        });

        const state = get(store);
        expect(state.fileDiffs['a.py']).toEqual({
            original: 'value = 1\n',
            modified: 'value = 2\n',
            truncated: false
        });
        expect(state.files).toContain('a.py');
        expect(state.selectedFile).toBe('a.py');
    });

    it('keeps diffs out of the chat feed', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'diff', file_path: 'a.py', original: '', modified: 'x', truncated: false });

        expect(get(store).messages).toHaveLength(0);
    });

    it('requests rework when the reviewer rejects', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'review', approved: false, comments: 'Needs a guard clause.' });

        const state = get(store);
        expect(state.status).toBe('waiting_review_approval');
        expect(state.actionType).toBe('approve_review');
        expect(latest(store).content).toBe('Needs a guard clause.');
    });

    it('distinguishes the merge prompt from the merge confirmation', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'done', summary: 'Added the function.', branch_name: 'ai-task-1', total_commits: 2 });
        expect(get(store).status).toBe('completed');
        expect(get(store).actionType).toBe('approve_merge');

        socket.deliver({ type: 'done', summary: 'Merged ai-task-1 into main.', branch_name: 'ai-task-1', total_commits: 2 });
        expect(get(store).status).toBe('merged');
        expect(get(store).requiresAction).toBe(false);
    });

    it('marks the session failed on an error', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'error', content: 'The session failed.' });

        expect(get(store).status).toBe('failed');
        expect(get(store).requiresAction).toBe(false);
    });

    it('derives progress status from indexing thoughts', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'thought', content: 'Indexing [3/10]: a.py' });
        expect(get(store).status).toBe('indexing');

        socket.deliver({ type: 'thought', content: 'Executing plan...' });
        expect(get(store).status).toBe('executing');

        socket.deliver({ type: 'thought', content: 'Reworking based on the feedback received.' });
        expect(get(store).status).toBe('re_working');
    });
});

describe('sending decisions', () => {
    it('sends a rejection reason back to the agent', async () => {
        const { store, socket } = await connected();

        store.approve(false, 'You forgot the migration.');

        expect(socket.lastSent).toEqual({ approved: false, reason: 'You forgot the migration.' });
        expect(get(store).requiresAction).toBe(false);
    });

    it('sends explicit actions for the iteration cap prompt', async () => {
        const { store, socket } = await connected();

        store.sendAction('abort');

        expect(socket.lastSent).toEqual({ action: 'abort', reason: '' });
    });
});

describe('configuration persistence', () => {
    beforeEach(() => localStorage.clear());

    it('persists ordinary settings', async () => {
        const { store } = await connected();

        store.updateConfig({ coderModel: 'ollama:qwen' });

        expect(JSON.parse(localStorage.getItem('loom_config')!).coderModel).toBe('ollama:qwen');
    });

    it('never writes secret values to storage', async () => {
        const { store } = await connected();

        store.updateConfig({ secrets: [{ key: 'API_KEY', value: 'super-secret' }] });

        const stored = localStorage.getItem('loom_config')!;
        expect(stored).not.toContain('super-secret');
        expect(stored).not.toContain('API_KEY');
        expect(JSON.parse(stored).secrets).toBeUndefined();
        // The value stays usable in memory for the session that entered it.
        expect(get(store).config.secrets[0].value).toBe('super-secret');
    });

    it('starts with no secrets even when storage holds some', async () => {
        localStorage.setItem(
            'loom_config',
            JSON.stringify({ coderModel: 'ollama:x', secrets: [{ key: 'LEAKED', value: 'v' }] })
        );
        vi.resetModules();
        vi.stubGlobal('WebSocket', FakeSocket);

        const store = (await import('./agent')).agentStore;

        expect(get(store).config.secrets).toEqual([]);
        expect(get(store).config.coderModel).toBe('ollama:x');
    });
});


describe('a session already running elsewhere', () => {
    it('shows a stale message until a sibling confirms', async () => {
        // No BroadcastChannel means no sibling can answer, so the refusal stays
        // provisional and the user is offered a retry.
        vi.stubGlobal('BroadcastChannel', undefined);
        const { store, socket } = await connected();

        socket.deliver({
            type: 'session_busy',
            content: 'A Loom session is already running in another tab.',
            active_session_id: 'abc123'
        });

        expect(get(store).blocked).toBe('stale');
        expect(latest(store).content).toContain('already running');
    });

    it('does not present the refusal as a failure', async () => {
        vi.stubGlobal('BroadcastChannel', undefined);
        const { store, socket } = await connected();

        socket.deliver({ type: 'session_busy', content: 'busy', active_session_id: 'x' });

        expect(get(store).status).not.toBe('failed');
    });

    it('stops reconnecting once refused', async () => {
        vi.stubGlobal('BroadcastChannel', undefined);
        const { store, socket } = await connected();
        socket.deliver({ type: 'session_busy', content: 'busy', active_session_id: 'x' });

        const scheduled = vi.fn();
        vi.stubGlobal('setTimeout', scheduled);
        socket.onclose?.();

        // Reconnecting would loop against a backend that will keep refusing.
        expect(scheduled).not.toHaveBeenCalled();
        expect(get(store).connection).toBe('disconnected');
    });

    it('clears the block and reconnects on retry', async () => {
        vi.stubGlobal('BroadcastChannel', undefined);
        const { store, socket } = await connected();
        socket.deliver({ type: 'session_busy', content: 'busy', active_session_id: 'x' });
        expect(get(store).blocked).toBe('stale');

        store.retryConnection();

        expect(get(store).blocked).toBeNull();
        expect(get(store).messages).toHaveLength(0);
    });
});
