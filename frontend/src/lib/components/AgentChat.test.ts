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

    close(): void {
        this.readyState = 3;
    }

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
    agentStore.disconnect();
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
    socket.deliver({
        type: 'plan',
        steps: [{ title: 'Read the file' }, { title: 'Edit it' }],
        requires_approval: true
    });
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

    it('confirms before overriding a rejected review and merging', async () => {
        const { socket } = renderChat();
        socket.deliver({ type: 'review', approved: false, comments: 'Add an empty-case guard.' });
        await tick();

        await fireEvent.click(screen.getByRole('button', { name: 'Overrule reviewer and merge' }));

        expect(
            screen.getByText(
                'The current changes will be merged into Main, and the task will be marked as complete.'
            )
        ).toBeInTheDocument();
        expect(socket.sent).toHaveLength(0);

        await fireEvent.click(screen.getByRole('button', { name: 'Merge into Main' }));

        expect(socket.lastSent).toEqual({ action: 'merge', reason: '' });
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

describe('answering a question from the agent', () => {
    /** Put the session in front of an outstanding question, and let it render. */
    async function askQuestion(socket: FakeSocket, options: string[] = []): Promise<void> {
        // A question arrives while the agent is working, which is what makes
        // requirement 4.8 delicate: the interface is mid-task and still has to
        // accept an answer.
        socket.deliver({ type: 'system', content: 'Executing plan...', status: 'executing' });
        socket.deliver({ type: 'question', question: 'Which retry policy?', options });
        await tick();
    }

    /** The prompt box, which is the only text input in the interface. */
    function box(): HTMLElement {
        return screen.getByRole('textbox', { name: 'Task prompt' });
    }

    it('shows the question with the answers the agent suggested', async () => {
        const { socket } = renderChat();

        await askQuestion(socket, ['The HTTP client', 'The database pool']);

        expect(screen.getByTestId('question-panel')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'The HTTP client' })).toBeInTheDocument();
    });

    it('sends a chosen answer straight to the agent', async () => {
        const { socket } = renderChat();
        await askQuestion(socket, ['The HTTP client', 'The database pool']);

        await fireEvent.click(screen.getByRole('button', { name: 'The database pool' }));

        expect(socket.lastSent).toEqual({ action: 'reply', reply: 'The database pool' });
    });

    it('says the prompt box now sends a reply', async () => {
        // Requirement 4.9a: the box has changed meaning, and a control that
        // changes silently is worse than one that says so.
        const { socket } = renderChat();

        await askQuestion(socket);

        expect(box()).toHaveAttribute('placeholder', "Type your reply to the agent's question...");
    });

    it('sends a typed reply as a reply rather than a new task', async () => {
        // Requirement 4.8 and 4.9. The session is working, so the box would
        // ordinarily refuse to send at all; a question has to lift that, and
        // what it sends must be the answer, not a fresh task.
        const { socket } = renderChat();
        await askQuestion(socket);

        await fireEvent.input(box(), { target: { value: 'The one in the HTTP client' } });
        await fireEvent.keyDown(box(), { key: 'Enter' });

        expect(socket.lastSent).toEqual({
            action: 'reply',
            reply: 'The one in the HTTP client'
        });
    });

    it('withholds a new task while the question is outstanding', async () => {
        // The other half of 4.8: nothing typed can start a second task, because
        // everything typed is an answer until the question is gone.
        const { socket } = renderChat();
        await askQuestion(socket);

        await fireEvent.input(box(), { target: { value: 'do something else entirely' } });
        await fireEvent.keyDown(box(), { key: 'Enter' });

        expect(socket.lastSent).not.toHaveProperty('prompt');
        expect(socket.lastSent.action).toBe('reply');
    });

    it('puts both the question and the answer in the feed', async () => {
        // Requirement 4.5.
        const { socket } = renderChat();
        await askQuestion(socket, ['The HTTP client']);

        await fireEvent.click(screen.getByRole('button', { name: 'The HTTP client' }));
        await tick();

        expect(screen.getByText('Which retry policy?')).toBeInTheDocument();
        expect(screen.getAllByText('The HTTP client').length).toBeGreaterThan(0);
    });

    it('takes the question away once it has been answered', async () => {
        const { socket } = renderChat();
        await askQuestion(socket, ['The HTTP client']);

        await fireEvent.click(screen.getByRole('button', { name: 'The HTTP client' }));
        await tick();

        expect(screen.queryByTestId('question-panel')).not.toBeInTheDocument();
    });

    it('puts the cursor in the box when the reader chooses to type', async () => {
        // Requirement 4.4: choosing the free-reply option has to lead somewhere.
        const { socket } = renderChat();
        await askQuestion(socket, ['The HTTP client']);

        await fireEvent.click(screen.getByTestId('free-reply-option'));
        await tick();

        expect(document.activeElement).toBe(box());
    });

    it('stops accepting anything once the question is answered', async () => {
        // The lifting of requirement 4.8's rule lasts exactly as long as the
        // question. The agent is still working afterwards, so the box goes back
        // to withholding what is typed rather than sending it as a task.
        const { socket } = renderChat();
        await askQuestion(socket, ['The HTTP client']);
        await fireEvent.click(screen.getByRole('button', { name: 'The HTTP client' }));
        await tick();
        const afterAnswering = socket.sent.length;

        expect(box()).toHaveAttribute(
            'placeholder',
            'Write the next task while the agent works...'
        );

        await fireEvent.input(box(), { target: { value: 'a brand new task' } });
        await fireEvent.keyDown(box(), { key: 'Enter' });

        expect(socket.sent).toHaveLength(afterAnswering);
    });
});
