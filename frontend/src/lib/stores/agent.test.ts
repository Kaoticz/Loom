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

        socket.deliver({
            type: 'plan',
            steps: ['Read the file', 'Edit it'],
            requires_approval: true
        });

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

        socket.deliver({
            type: 'observation',
            tool_name: 'edit_file',
            result: 'Edited a.py',
            status: 'success'
        });

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
    });

    it('keeps diffs out of the chat feed', async () => {
        const { store, socket } = await connected();

        socket.deliver({
            type: 'diff',
            file_path: 'a.py',
            original: '',
            modified: 'x',
            truncated: false
        });

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

        socket.deliver({
            type: 'done',
            summary: 'Added the function.',
            branch_name: 'ai-task-1',
            total_commits: 2
        });
        expect(get(store).status).toBe('completed');
        expect(get(store).actionType).toBe('approve_merge');

        socket.deliver({
            type: 'done',
            summary: 'Merged ai-task-1 into main.',
            branch_name: 'ai-task-1',
            total_commits: 2
        });
        expect(get(store).status).toBe('merged');
        expect(get(store).requiresAction).toBe(false);
    });

    it('marks the session failed on an error', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'error', content: 'The session failed.' });

        expect(get(store).status).toBe('failed');
        expect(get(store).requiresAction).toBe(false);
    });

    it('takes progress status from the phase the backend states', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'system', content: 'Indexing repository...', status: 'indexing' });
        expect(get(store).status).toBe('indexing');

        socket.deliver({ type: 'system', content: 'Executing plan...', status: 'executing' });
        expect(get(store).status).toBe('executing');

        socket.deliver({
            type: 'system',
            content: 'Reworking based on the feedback received.',
            status: 're_working'
        });
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

describe('system messages', () => {
    const PROGRESS_KEY = 'indexing_progress';

    it('appends a system message that names no slot', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'system', content: 'Starting sandbox...' });

        expect(get(store).messages).toHaveLength(1);
        expect(latest(store).type).toBe('system');
        expect(latest(store).content).toBe('Starting sandbox...');
    });

    it('replaces a keyed message in place rather than appending', async () => {
        const { store, socket } = await connected();

        socket.deliver({
            type: 'system',
            content: 'a.py',
            key: PROGRESS_KEY,
            processed: 1,
            total: 3
        });
        socket.deliver({
            type: 'system',
            content: 'b.py',
            key: PROGRESS_KEY,
            processed: 2,
            total: 3
        });
        socket.deliver({
            type: 'system',
            content: 'c.py',
            key: PROGRESS_KEY,
            processed: 3,
            total: 3
        });

        expect(get(store).messages).toHaveLength(1);
        expect(latest(store).content).toBe('[3/3] c.py');
    });

    it('keeps one progress message however many updates arrive', async () => {
        const { store, socket } = await connected();

        for (let processed = 0; processed <= 500; processed += 1) {
            socket.deliver({
                type: 'system',
                content: `file_${processed}.py`,
                key: PROGRESS_KEY,
                processed,
                total: 500
            });
        }

        expect(get(store).messages).toHaveLength(1);
        expect(latest(store).content).toBe('[500/500] file_500.py');
    });

    it('renders the counter from the counts rather than the prose', async () => {
        const { store, socket } = await connected();

        socket.deliver({
            type: 'system',
            content: 'Indexed src/app.py',
            key: PROGRESS_KEY,
            processed: 7,
            total: 42
        });

        expect(latest(store).content).toBe('[7/42] Indexed src/app.py');
    });

    it('omits the counter when a system message carries no counts', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'system', content: 'Merging...', key: 'stage' });

        expect(latest(store).content).toBe('Merging...');
    });

    it('resolves an empty repository at zero of zero', async () => {
        const { store, socket } = await connected();

        socket.deliver({
            type: 'system',
            content: 'Indexing complete.',
            key: PROGRESS_KEY,
            processed: 0,
            total: 0
        });

        expect(get(store).messages).toHaveLength(1);
        expect(latest(store).content).toBe('[0/0] Indexing complete.');
    });

    it('keeps distinct slots independent', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'system', content: 'one', key: 'first' });
        socket.deliver({ type: 'system', content: 'two', key: 'second' });
        socket.deliver({ type: 'system', content: 'one again', key: 'first' });

        const messages = get(store).messages;
        expect(messages).toHaveLength(2);
        expect(messages[0].content).toBe('one again');
        expect(messages[1].content).toBe('two');
    });

    it('does not disturb surrounding messages when replacing a slot', async () => {
        const { store, socket } = await connected();

        socket.deliver({
            type: 'system',
            content: 'a.py',
            key: PROGRESS_KEY,
            processed: 1,
            total: 2
        });
        socket.deliver({ type: 'thought', content: 'considering' });
        socket.deliver({
            type: 'system',
            content: 'b.py',
            key: PROGRESS_KEY,
            processed: 2,
            total: 2
        });

        const messages = get(store).messages;
        expect(messages).toHaveLength(2);
        expect(messages[0].content).toBe('[2/2] b.py');
        expect(messages[1].type).toBe('thought');
    });
});

describe('thought coalescing', () => {
    it('grows one message for a continuous run of reasoning', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'thought', content: 'First.', stream_id: 's:1' });
        socket.deliver({ type: 'thought', content: 'Second.', stream_id: 's:1' });
        socket.deliver({ type: 'thought', content: 'Third.', stream_id: 's:1' });

        expect(get(store).messages).toHaveLength(1);
        expect(latest(store).content).toBe('First. Second. Third.');
    });

    it('starts a new message for a new run', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'thought', content: 'Before.', stream_id: 's:1' });
        socket.deliver({ type: 'thought', content: 'After.', stream_id: 's:2' });

        const messages = get(store).messages;
        expect(messages).toHaveLength(2);
        expect(messages[0].content).toBe('Before.');
        expect(messages[1].content).toBe('After.');
    });

    it("renders the user's sequence as prompt, thought, action, thought, observation", async () => {
        const { store, socket } = await connected();

        store.sendPrompt('do the thing');
        socket.deliver({ type: 'thought', content: 'One.', stream_id: 's:1' });
        socket.deliver({ type: 'thought', content: 'Two.', stream_id: 's:1' });
        socket.deliver({ type: 'thought', content: 'Three.', stream_id: 's:1' });
        socket.deliver({ type: 'action', tool_name: 'edit_file', arguments: { path: 'a.py' } });
        socket.deliver({ type: 'thought', content: 'Four.', stream_id: 's:2' });
        socket.deliver({ type: 'observation', tool_name: 'edit_file', result: 'written' });

        const messages = get(store).messages;
        expect(messages.map((message) => message.type)).toEqual([
            'prompt',
            'thought',
            'action',
            'thought',
            'observation'
        ]);
        expect(messages[1].content).toBe('One. Two. Three.');
        expect(messages[3].content).toBe('Four.');
    });

    it('does not coalesce thoughts that carry no run identifier', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'thought', content: 'One.' });
        socket.deliver({ type: 'thought', content: 'Two.' });

        expect(get(store).messages).toHaveLength(2);
    });

    it('leaves other repeated message types separate', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'observation', tool_name: 'run_command', result: 'a' });
        socket.deliver({ type: 'observation', tool_name: 'run_command', result: 'b' });
        socket.deliver({ type: 'commit', message: 'one', commit_hash: 'abc1234' });
        socket.deliver({ type: 'commit', message: 'two', commit_hash: 'def5678' });

        expect(get(store).messages).toHaveLength(4);
    });

    it('does not let model reasoning drive the session status', async () => {
        const { store, socket } = await connected();
        const before = get(store).status;

        socket.deliver({ type: 'thought', content: 'Merging the branch.', stream_id: 's:1' });

        // The backend owns the status; reasoning that merely mentions merging
        // must not move it.
        expect(get(store).status).toBe(before);
    });

    it('caps a run of reasoning that never ends', async () => {
        const { store, socket } = await connected();
        const sentence = 'a'.repeat(1_000);

        for (let index = 0; index < 40; index += 1) {
            socket.deliver({ type: 'thought', content: sentence, stream_id: 's:1' });
        }

        expect(get(store).messages).toHaveLength(1);
        const { content } = latest(store);
        expect(content).toContain('truncated');
        expect(content.length).toBeLessThan(40 * sentence.length);
    });

    it('stops growing a run once it has been capped', async () => {
        const { store, socket } = await connected();
        const sentence = 'b'.repeat(1_000);

        for (let index = 0; index < 40; index += 1) {
            socket.deliver({ type: 'thought', content: sentence, stream_id: 's:1' });
        }
        const capped = latest(store).content;

        socket.deliver({ type: 'thought', content: 'and more', stream_id: 's:1' });

        expect(latest(store).content).toBe(capped);
    });

    it('keeps a run separate from an interleaved progress slot', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'thought', content: 'Before.', stream_id: 's:1' });
        socket.deliver({
            type: 'system',
            content: 'a.py',
            key: 'indexing_progress',
            processed: 1,
            total: 2
        });
        socket.deliver({ type: 'thought', content: 'After.', stream_id: 's:2' });

        expect(get(store).messages.map((message) => message.type)).toEqual([
            'thought',
            'system',
            'thought'
        ]);
    });
});

describe('inbound validation', () => {
    it('accepts a message carrying fields it does not know about', async () => {
        const { store, socket } = await connected();

        socket.deliver({
            type: 'thought',
            content: 'Considering.',
            stream_id: 's:1',
            invented_by_a_newer_backend: { nested: true }
        });

        expect(get(store).messages).toHaveLength(1);
        expect(latest(store).type).toBe('thought');
        expect(latest(store).content).toBe('Considering.');
    });

    it('reports a message of an unknown type as an error in the feed', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'not_a_real_message', content: 'hello' });

        expect(latest(store).type).toBe('error');
        expect(latest(store).content).toContain('could not be understood');
    });

    it('reports a message missing a required field rather than applying it', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'commit', message: 'no hash here' });

        expect(get(store).messages).toHaveLength(1);
        expect(latest(store).type).toBe('error');
    });

    it('reports a payload that is not valid JSON', async () => {
        const { store, socket } = await connected();

        socket.onmessage?.({ data: '{ not json' });

        expect(latest(store).type).toBe('error');
        expect(latest(store).content).toContain('not valid JSON');
    });

    it('keeps working after rejecting an invalid message', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'nonsense' });
        socket.deliver({ type: 'thought', content: 'Still here.', stream_id: 's:1' });

        const messages = get(store).messages;
        expect(messages.map((message) => message.type)).toEqual(['error', 'thought']);
        expect(get(store).connection).toBe('connected');
    });

    it('applies the schema defaults the backend would have sent', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'plan', steps: ['One', 'Two'] });

        expect(get(store).status).toBe('waiting_approval');
        expect(latest(store).content).toBe('1. One\n2. Two');
    });

    it('does not let an invalid message leave the interface stuck', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'done' });

        expect(get(store).status).not.toBe('completed');
        expect(get(store).requiresAction).toBe(false);
    });
});

describe('validation error coalescing', () => {
    it('collapses repeats of the same fault into one counted message', async () => {
        const { store, socket } = await connected();

        for (let index = 0; index < 200; index += 1) {
            socket.deliver({ type: 'commit', message: 'no hash' });
        }

        const messages = get(store).messages;
        expect(messages).toHaveLength(1);
        expect(messages[0].type).toBe('error');
        expect(messages[0].count).toBe(200);
        expect(messages[0].content).toContain('200 occurrences');
    });

    it('reads as a single fault the first time', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'commit', message: 'no hash' });

        expect(latest(store).content).not.toContain('occurrences');
        expect(latest(store).count).toBe(1);
    });

    it('keeps distinct faults in separate slots', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'commit', message: 'no hash' });
        socket.deliver({ type: 'review', approved: true });
        socket.deliver({ type: 'commit', message: 'still no hash' });

        const errors = get(store).messages.filter((message) => message.type === 'error');
        expect(errors).toHaveLength(2);
        expect(errors[0].count).toBe(2);
        expect(errors[1].count).toBe(1);
    });

    it('caps how many distinct faults get their own slot', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'commit', message: 'a' });
        socket.deliver({ type: 'review', approved: true });
        socket.deliver({ type: 'observation', tool_name: 'x' });
        socket.deliver({ type: 'done' });
        socket.deliver({ type: 'diff', file_path: 'a.py' });
        socket.deliver({ type: 'action' });
        socket.deliver({ type: 'system' });

        const errors = get(store).messages.filter((message) => message.type === 'error');
        expect(errors).toHaveLength(6);
        const overflow = errors[errors.length - 1];
        expect(overflow.content).toContain('Further messages');
        expect(overflow.count).toBe(2);
    });

    it('does not disturb real messages while counting up', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'commit', message: 'no hash' });
        socket.deliver({ type: 'thought', content: 'Working.', stream_id: 's:1' });
        socket.deliver({ type: 'commit', message: 'no hash again' });

        const messages = get(store).messages;
        expect(messages.map((message) => message.type)).toEqual(['error', 'thought']);
        expect(messages[0].count).toBe(2);
        expect(messages[1].content).toBe('Working.');
    });

    it('counts malformed JSON in its own slot', async () => {
        const { store, socket } = await connected();

        socket.onmessage?.({ data: '{ nope' });
        socket.onmessage?.({ data: '{ nope again' });

        const errors = get(store).messages.filter((message) => message.type === 'error');
        expect(errors).toHaveLength(1);
        expect(errors[0].content).toContain('not valid JSON');
        expect(errors[0].count).toBe(2);
    });
});

describe('system status announcements', () => {
    it('takes the status the backend states', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'system', content: 'Generating plan...', status: 'planning' });

        expect(get(store).status).toBe('planning');
    });

    it('trusts the stated status over the wording', async () => {
        const { store, socket } = await connected();

        // Wording that the old prose matching would have read as 'merging'.
        socket.deliver({ type: 'system', content: 'Merging is next.', status: 'reviewing' });

        expect(get(store).status).toBe('reviewing');
    });

    it('leaves the status alone for an announcement that claims no phase', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'system', content: 'Generating plan...', status: 'planning' });
        socket.deliver({
            type: 'system',
            content: 'The following secrets are available inside the sandbox.'
        });

        expect(get(store).status).toBe('planning');
    });

    it('appends announcements rather than replacing, since they carry no key', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'system', content: 'Indexing repository...', status: 'indexing' });
        socket.deliver({ type: 'system', content: 'Generating plan...', status: 'planning' });
        socket.deliver({ type: 'system', content: 'Starting review...', status: 'reviewing' });

        const messages = get(store).messages;
        expect(messages).toHaveLength(3);
        expect(messages.every((message) => message.type === 'system')).toBe(true);
    });

    it('still updates the progress slot in place alongside announcements', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'system', content: 'Indexing repository...', status: 'indexing' });
        socket.deliver({
            type: 'system',
            content: 'a.py',
            key: 'indexing_progress',
            processed: 1,
            total: 2
        });
        socket.deliver({
            type: 'system',
            content: 'b.py',
            key: 'indexing_progress',
            processed: 2,
            total: 2
        });

        const messages = get(store).messages;
        expect(messages).toHaveLength(2);
        expect(messages[1].content).toBe('[2/2] b.py');
    });

    it('rejects a status the frontend does not know', async () => {
        const { store, socket } = await connected();

        socket.deliver({
            type: 'system',
            content: 'Doing something new...',
            status: 'teleporting'
        });

        expect(latest(store).type).toBe('error');
    });

    it('ignores the wording when no status is stated', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'system', content: 'Generating plan...', status: 'planning' });
        socket.deliver({ type: 'system', content: 'Merging is mentioned here.' });

        // There is no prose fallback: an announcement that claims no phase
        // leaves the status where it was.
        expect(get(store).status).toBe('planning');
    });
});

describe('message identity', () => {
    it('keeps one identity while a run of reasoning grows', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'thought', content: 'First.', stream_id: 's:1' });
        const original = get(store).messages[0].id;

        socket.deliver({ type: 'thought', content: 'Second.', stream_id: 's:1' });
        socket.deliver({ type: 'thought', content: 'Third.', stream_id: 's:1' });

        // The object is replaced on each update; the identity must outlive it,
        // or the rendered element is destroyed and rebuilt every sentence.
        expect(get(store).messages[0].id).toBe(original);
    });

    it('keeps one identity while a progress slot counts up', async () => {
        const { store, socket } = await connected();

        socket.deliver({
            type: 'system',
            content: 'a',
            key: 'indexing_progress',
            processed: 1,
            total: 3
        });
        const original = get(store).messages[0].id;

        socket.deliver({
            type: 'system',
            content: 'b',
            key: 'indexing_progress',
            processed: 2,
            total: 3
        });
        socket.deliver({
            type: 'system',
            content: 'c',
            key: 'indexing_progress',
            processed: 3,
            total: 3
        });

        expect(get(store).messages[0].id).toBe(original);
    });

    it('keeps one identity while a validation fault counts up', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'commit', message: 'no hash' });
        const original = get(store).messages[0].id;

        socket.deliver({ type: 'commit', message: 'still no hash' });

        expect(get(store).messages[0].id).toBe(original);
    });

    it('gives every message a distinct identity', async () => {
        const { store, socket } = await connected();

        store.sendPrompt('do the thing');
        socket.deliver({ type: 'thought', content: 'One.', stream_id: 's:1' });
        socket.deliver({ type: 'action', tool_name: 'edit_file', arguments: {} });
        socket.deliver({ type: 'thought', content: 'Two.', stream_id: 's:2' });
        socket.deliver({ type: 'system', content: 'Merging...', status: 'merging' });

        const ids = get(store).messages.map((message) => message.id);
        expect(new Set(ids).size).toBe(ids.length);
    });

    it('never reissues an identity already in the feed', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'thought', content: 'Before.', stream_id: 's:1' });
        const existing = get(store).messages.map((message) => message.id);

        store.sendPrompt('a new task');
        socket.deliver({ type: 'thought', content: 'After.', stream_id: 's:2' });

        const added = get(store)
            .messages.map((message) => message.id)
            .filter((id) => !existing.includes(id));
        expect(added).toHaveLength(2);
        expect(added.every((id) => Math.max(...existing) < id)).toBe(true);
    });
});

describe('sandbox availability', () => {
    it('records the reason without adding a chat message', async () => {
        const { store, socket } = await connected();

        socket.deliver({
            type: 'sandbox_unavailable',
            stage: 'runtime_connectivity',
            reason: 'no docker'
        });

        // Requirement 6.1: a banner, not a chat message.
        expect(get(store).messages).toHaveLength(0);
        expect(get(store).sandboxUnavailable?.reason).toBe('no docker');
        expect(get(store).sandboxUnavailable?.stage).toBe('runtime_connectivity');
    });

    it('starts with no sandbox problem and nothing dismissed', async () => {
        const { store } = await connected();

        expect(get(store).sandboxUnavailable).toBeNull();
        expect(get(store).sandboxNoticeDismissed).toBe(false);
    });

    it('does not stack a second notification', async () => {
        const { store, socket } = await connected();

        socket.deliver({
            type: 'sandbox_unavailable',
            stage: 'runtime_connectivity',
            reason: 'unreachable'
        });
        socket.deliver({
            type: 'sandbox_unavailable',
            stage: 'container_start',
            reason: 'failed to start'
        });

        // Requirement 6.5: one predicament, one banner.
        expect(get(store).messages).toHaveLength(0);
        expect(get(store).sandboxUnavailable?.reason).toBe('failed to start');
    });

    it('stays dismissed once dismissed', async () => {
        const { store, socket } = await connected();

        socket.deliver({
            type: 'sandbox_unavailable',
            stage: 'runtime_connectivity',
            reason: 'unreachable'
        });
        store.dismissSandboxNotice();
        socket.deliver({
            type: 'sandbox_unavailable',
            stage: 'container_start',
            reason: 'failed to start'
        });

        // Requirement 6.5: it must not reappear for the same session.
        expect(get(store).sandboxNoticeDismissed).toBe(true);
    });

    it('rejects a notice that carries no reason', async () => {
        const { store, socket } = await connected();

        socket.deliver({ type: 'sandbox_unavailable', stage: 'image_build' });

        expect(get(store).sandboxUnavailable).toBeNull();
        expect(latest(store).type).toBe('error');
    });
});

describe('stopping', () => {
    it('asks the backend to interrupt', async () => {
        const { store, socket } = await connected();

        store.stop();

        expect(socket.lastSent).toEqual({ action: 'stop' });
    });

    it('does not claim the session has stopped', async () => {
        const { store, socket } = await connected();
        socket.deliver({ type: 'system', content: 'Executing plan...', status: 'executing' });

        store.stop();

        // The backend decides when it has actually stopped. Guessing would show
        // idle while the agent was still finishing a step.
        expect(get(store).status).toBe('executing');
    });
});

describe('after stopping', () => {
    it('leaves the working state when the backend says it has stopped', async () => {
        const { store, socket } = await connected();
        socket.deliver({ type: 'system', content: 'Executing plan...', status: 'executing' });

        socket.deliver({
            type: 'system',
            content: 'Stopped at your request. Everything done so far has been kept.',
            status: 'cancelled'
        });

        // Requirement 8.7: the interface has to come out of the working state,
        // or the prompt box goes on offering to stop work that already has.
        expect(get(store).status).toBe('cancelled');
    });
});
