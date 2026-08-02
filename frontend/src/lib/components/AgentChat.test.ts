import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/svelte';
import { tick } from 'svelte';
import AgentChat from './AgentChat.svelte';
import { agentStore } from '$lib/stores/agent';

vi.mock('$app/environment', () => ({ browser: true }));

/**
 * Tests of the chat as a whole, rather than of its parts.
 *
 * Every component here is covered on its own, but that leaves the joins between
 * them unexamined — and the joins are where the interesting claims live. The
 * text typed into the input has to reach the approval panel and then the agent,
 * and no test of either end can show that it does.
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

let session = 0;

beforeEach(() => {
    localStorage.clear();
    vi.stubGlobal('WebSocket', FakeSocket);

    // The store is a module singleton, so one test's outstanding decision would
    // otherwise still be outstanding in the next. Pointing it at a different
    // repository is the store's own way of archiving a task and starting clean.
    session += 1;
    agentStore.updateConfig({ repoPath: `/test/repo-${session}` });
});

/**
 * Render the chat against a freshly connected store.
 *
 * The store is a module singleton, so each test reconnects it rather than
 * reimporting: reimporting would give the test a different instance from the
 * one the components hold.
 */
function renderChat() {
    agentStore.connect('ws://test/agent-session');
    const socket = FakeSocket.latest;
    socket.onopen?.();

    const rendered = render(AgentChat);
    return { socket, rendered };
}

/** Put the session in front of a plan awaiting approval, and let it render. */
async function awaitPlan(socket: FakeSocket): Promise<void> {
    socket.deliver({ type: 'plan', steps: ['Read the file', 'Edit it'], requires_approval: true });
    await tick();
}

describe('composing and deciding', () => {
    it('sends what was typed in the prompt box as the rejection reason', async () => {
        const { socket } = renderChat();
        await awaitPlan(socket);

        await fireEvent.input(screen.getByRole('textbox', { name: 'Task prompt' }), {
            target: { value: 'split step two into smaller steps' }
        });
        await fireEvent.click(screen.getByRole('button', { name: 'Reject & Replan' }));

        // The whole chain: input, to the chat, to the panel, to the agent.
        expect(socket.lastSent).toEqual({
            approved: false,
            reason: 'split step two into smaller steps'
        });
    });

    it('clears the box once the decision is sent', async () => {
        const { socket } = renderChat();
        await awaitPlan(socket);
        const box = screen.getByRole('textbox', { name: 'Task prompt' }) as HTMLTextAreaElement;

        await fireEvent.input(box, { target: { value: 'not this way' } });
        await fireEvent.click(screen.getByRole('button', { name: 'Reject & Replan' }));

        expect(box.value).toBe('');
    });

    it('approves without sending what was typed', async () => {
        const { socket } = renderChat();
        await awaitPlan(socket);

        await fireEvent.input(screen.getByRole('textbox', { name: 'Task prompt' }), {
            target: { value: 'idle musing, not a rejection' }
        });
        await fireEvent.click(screen.getByRole('button', { name: 'Approve Plan' }));

        expect(socket.lastSent).toEqual({ approved: true, reason: '' });
    });

    it('offers the prompt box as the place to type a reason', async () => {
        const { socket } = renderChat();
        await awaitPlan(socket);

        // Requirement 9.6a: one input, and it says what it is for.
        expect(screen.getAllByRole('textbox')).toHaveLength(1);
        expect(screen.getByRole('textbox', { name: 'Task prompt' })).toHaveAttribute(
            'placeholder',
            expect.stringContaining('reason')
        );
    });

    it('refuses to send a new task while the plan is undecided', async () => {
        const { socket } = renderChat();
        await awaitPlan(socket);
        const before = socket.sent.length;

        await fireEvent.input(screen.getByRole('textbox', { name: 'Task prompt' }), {
            target: { value: 'a completely different task' }
        });
        await fireEvent.keyDown(screen.getByRole('textbox', { name: 'Task prompt' }), {
            key: 'Enter'
        });

        expect(socket.sent).toHaveLength(before);
        expect(screen.getByRole('button', { name: 'Send Message' })).toBeDisabled();
    });

    it('sends a new task when nothing is outstanding', async () => {
        const { socket } = renderChat();

        await fireEvent.input(screen.getByRole('textbox', { name: 'Task prompt' }), {
            target: { value: 'add a health endpoint' }
        });
        await fireEvent.click(screen.getByRole('button', { name: 'Send Message' }));

        expect(socket.lastSent).toMatchObject({ prompt: 'add a health endpoint' });
    });

    it('keeps composed text while the agent works', async () => {
        const { socket } = renderChat();

        await fireEvent.input(screen.getByRole('textbox', { name: 'Task prompt' }), {
            target: { value: 'the next thing I want' }
        });
        socket.deliver({ type: 'system', content: 'Executing plan...', status: 'executing' });
        await tick();

        const box = screen.getByRole('textbox', { name: 'Task prompt' }) as HTMLTextAreaElement;
        expect(box.value).toBe('the next thing I want');
        expect(box).toBeEnabled();
        expect(screen.getByRole('button', { name: 'Stop the agent' })).toBeInTheDocument();
    });
});
