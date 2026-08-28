import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/svelte';
import ChatInput from './ChatInput.svelte';
import type { TaskStatus } from '$lib/stores/agent';

/** Render the input in a given session status. */
function renderInput(status: TaskStatus) {
    const onsend = vi.fn();
    const onstop = vi.fn();
    return { onsend, onstop, ...render(ChatInput, { status, onsend, onstop }) };
}

describe('ChatInput', () => {
    it('offers Send when the agent is idle', () => {
        renderInput('idle');

        expect(screen.getByRole('button', { name: 'Send Message' })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /stop/i })).not.toBeInTheDocument();
    });

    it.each(['indexing', 'planning', 'executing', 'reviewing', 're_working', 'merging'] as const)(
        'offers Stop instead of Send while %s',
        (status) => {
            // Requirements 8.1 and 8.2.
            renderInput(status);

            expect(screen.getByRole('button', { name: /stop/i })).toBeInTheDocument();
            expect(screen.queryByRole('button', { name: 'Send Message' })).not.toBeInTheDocument();
        }
    );

    it.each(['waiting_approval', 'completed', 'failed'] as const)(
        'offers Send while %s, since the agent is not working',
        (status) => {
            renderInput(status);

            expect(screen.getByRole('button', { name: 'Send Message' })).toBeInTheDocument();
        }
    );

    it('sends what was typed', async () => {
        const { onsend } = renderInput('idle');

        await fireEvent.input(screen.getByRole('textbox'), { target: { value: 'do the thing' } });
        await fireEvent.click(screen.getByRole('button', { name: 'Send Message' }));

        expect(onsend).toHaveBeenCalledWith('do the thing');
    });

    it('does not send an empty prompt', async () => {
        const { onsend } = renderInput('idle');

        await fireEvent.click(screen.getByRole('button', { name: 'Send Message' }));

        expect(onsend).not.toHaveBeenCalled();
    });

    it('clears the box after sending', async () => {
        renderInput('idle');
        const box = screen.getByRole('textbox') as HTMLTextAreaElement;

        await fireEvent.input(box, { target: { value: 'do the thing' } });
        await fireEvent.click(screen.getByRole('button', { name: 'Send Message' }));

        expect(box.value).toBe('');
    });

    it('asks to stop when Stop is pressed', async () => {
        const { onstop } = renderInput('executing');

        await fireEvent.click(screen.getByRole('button', { name: /stop/i }));

        expect(onstop).toHaveBeenCalledOnce();
    });

    it('does not send on Enter while the agent is working', async () => {
        const { onsend } = renderInput('executing');

        await fireEvent.input(screen.getByRole('textbox'), { target: { value: 'queued?' } });
        await fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' });

        expect(onsend).not.toHaveBeenCalled();
    });

    it('sends on Enter when idle, but not on Shift+Enter', async () => {
        const { onsend } = renderInput('idle');
        const box = screen.getByRole('textbox');

        await fireEvent.input(box, { target: { value: 'go' } });
        await fireEvent.keyDown(box, { key: 'Enter', shiftKey: true });
        expect(onsend).not.toHaveBeenCalled();

        await fireEvent.keyDown(box, { key: 'Enter' });
        expect(onsend).toHaveBeenCalledWith('go');
    });
});

describe('ChatInput after a stop', () => {
    it('offers Send again once the session is cancelled', () => {
        // Requirement 8.7: a new task can be started.
        renderInput('cancelled');

        expect(screen.getByRole('button', { name: 'Send Message' })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /stop/i })).not.toBeInTheDocument();
    });

    it('sends a new prompt after a stop', async () => {
        const { onsend } = renderInput('cancelled');

        await fireEvent.input(screen.getByRole('textbox'), { target: { value: 'next task' } });
        await fireEvent.click(screen.getByRole('button', { name: 'Send Message' }));

        expect(onsend).toHaveBeenCalledWith('next task');
    });
});

describe('ChatInput enablement', () => {
    /** Render with a decision outstanding. */
    function renderPending(status: TaskStatus, pendingAction: string | null) {
        const onsend = vi.fn();
        const onstop = vi.fn();
        return { onsend, onstop, ...render(ChatInput, { status, pendingAction, onsend, onstop }) };
    }

    it('leaves the box usable while a decision is outstanding', () => {
        // Requirement 9.1a: composing is always allowed.
        renderPending('waiting_approval', 'approve_plan');

        expect(screen.getByRole('textbox')).toBeEnabled();
    });

    it('leaves the box usable while the agent works', () => {
        renderPending('executing', null);

        expect(screen.getByRole('textbox')).toBeEnabled();
    });

    it('keeps typed text while the agent starts working', async () => {
        // The point of the override: text half composed is never taken away.
        const { rerender } = render(ChatInput, {
            status: 'idle' as TaskStatus,
            pendingAction: null,
            onsend: vi.fn(),
            onstop: vi.fn()
        });
        const box = screen.getByRole('textbox') as HTMLTextAreaElement;
        await fireEvent.input(box, { target: { value: 'half a thought' } });

        await rerender({ status: 'executing' as TaskStatus });

        expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('half a thought');
        expect(screen.getByRole('textbox')).toBeEnabled();
    });

    it('says the box is for a reason while a decision is outstanding', () => {
        // Requirement 9.3a and 9.6a.
        renderPending('waiting_approval', 'approve_plan');

        expect(screen.getByRole('textbox')).toHaveAttribute(
            'placeholder',
            expect.stringContaining('reason')
        );
    });

    it('explains that Enter will not send while the agent works', () => {
        // Requirement 9.4a: the Stop button occupies the slot, so there is no
        // disabled Send control to convey this.
        renderPending('executing', null);

        expect(screen.getByText(/Stop it to send this/)).toBeInTheDocument();
    });

    it('shows no such hint when sending is available', () => {
        renderPending('idle', null);

        expect(screen.queryByText(/Stop it to send this/)).not.toBeInTheDocument();
    });

    it('does not send on Enter while a decision is outstanding', async () => {
        const { onsend } = renderPending('waiting_approval', 'approve_plan');

        await fireEvent.input(screen.getByRole('textbox'), { target: { value: 'a reason' } });
        await fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' });

        expect(onsend).not.toHaveBeenCalled();
    });

    it('shows Send as disabled while a decision is outstanding', () => {
        // Requirement 9.5a: refusal is visible where a Send control exists.
        renderPending('waiting_approval', 'approve_plan');

        expect(screen.getByRole('button', { name: 'Send Message' })).toBeDisabled();
    });

    it('disables the input after a merge', () => {
        renderPending('merged', null);

        expect(screen.getByRole('textbox')).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Send Message' })).toBeDisabled();
    });
});
