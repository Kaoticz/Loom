import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/svelte';
import ApprovalPanel from './ApprovalPanel.svelte';

/** Render the panel for a given decision, optionally with composed text. */
function renderPanel(actionType: string, reason: string = '') {
    const onapprove = vi.fn();
    const onreject = vi.fn();
    return {
        onapprove,
        onreject,
        ...render(ApprovalPanel, { actionType, reason, onapprove, onreject })
    };
}

describe('ApprovalPanel', () => {
    it('asks the question belonging to the decision', () => {
        renderPanel('approve_merge');

        expect(screen.getByText('The work is complete.')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Merge into Main' })).toBeInTheDocument();
    });

    it('offers both answers straight away', () => {
        renderPanel('approve_plan');

        expect(screen.getByRole('button', { name: 'Approve Plan' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Reject & Replan' })).toBeInTheDocument();
    });

    it('has no text input of its own', () => {
        // Requirement 9.6a: the prompt box is the single place to type, so a
        // second input here would leave it ambiguous which one is read.
        renderPanel('approve_plan');

        expect(screen.queryByRole('textbox')).not.toBeInTheDocument();
    });

    it('says where the reason goes', () => {
        renderPanel('approve_plan');

        expect(screen.getByText(/box below/)).toBeInTheDocument();
    });

    it('reports an approval', async () => {
        const { onapprove, onreject } = renderPanel('approve_plan');

        await fireEvent.click(screen.getByRole('button', { name: 'Approve Plan' }));

        expect(onapprove).toHaveBeenCalledOnce();
        expect(onreject).not.toHaveBeenCalled();
    });

    it('reports a rejection', async () => {
        const { onapprove, onreject } = renderPanel('approve_plan');

        await fireEvent.click(screen.getByRole('button', { name: 'Reject & Replan' }));

        expect(onreject).toHaveBeenCalledOnce();
        expect(onapprove).not.toHaveBeenCalled();
    });

    it('sends the composed text as the rejection reason', async () => {
        // Requirement 9.6a: the prompt box is the reason input, so what was
        // typed there has to reach the agent.
        const { onreject } = renderPanel('approve_plan', 'try a different approach');

        await fireEvent.click(screen.getByRole('button', { name: 'Reject & Replan' }));

        expect(onreject).toHaveBeenCalledWith('try a different approach');
    });

    it('trims the composed text before sending it', async () => {
        const { onreject } = renderPanel('approve_plan', '  padded  ');

        await fireEvent.click(screen.getByRole('button', { name: 'Reject & Replan' }));

        expect(onreject).toHaveBeenCalledWith('padded');
    });

    it('rejects with no reason when nothing was composed', async () => {
        const { onreject } = renderPanel('approve_plan');

        await fireEvent.click(screen.getByRole('button', { name: 'Reject & Replan' }));

        expect(onreject).toHaveBeenCalledWith('');
    });
});
