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

/** The reject button for a decision, by its label. */
function rejectButton(actionType: string): HTMLElement {
    const labels: Record<string, string> = {
        approve_plan: 'Reject & Replan',
        approve_merge: 'Reject & Continue'
    };
    return screen.getByRole('button', { name: labels[actionType] });
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
        const { onapprove, onreject } = renderPanel('approve_plan', 'the plan misses a step');

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

    it('does not reject when nothing was composed', async () => {
        // Requirement 3.1. This replaces the previous behaviour, where the same
        // click sent an empty reason and the agent was told nothing at all.
        const { onreject } = renderPanel('approve_plan');

        await fireEvent.click(screen.getByRole('button', { name: 'Reject & Replan' }));

        expect(onreject).not.toHaveBeenCalled();
    });

    it('does not reject when only whitespace was composed', async () => {
        const { onreject } = renderPanel('approve_plan', '   \n  ');

        await fireEvent.click(screen.getByRole('button', { name: 'Reject & Replan' }));

        expect(onreject).not.toHaveBeenCalled();
    });

    it.each(['approve_plan', 'approve_merge'])(
        'marks reject unavailable for %s while nothing is composed',
        (actionType) => {
            // Requirement 3.4: all three decisions send the text as feedback,
            // so all three need a reason.
            renderPanel(actionType);

            expect(rejectButton(actionType)).toHaveAttribute('aria-disabled', 'true');
        }
    );

    it.each(['approve_plan', 'approve_merge'])(
        'makes reject available for %s as soon as a reason is typed',
        (actionType) => {
            renderPanel(actionType, 'not good enough');

            expect(rejectButton(actionType)).toHaveAttribute('aria-disabled', 'false');
        }
    );

    it('never gates approval on the composed text', async () => {
        // Requirement 3.3: approving discards what was typed, and is always
        // available.
        const { onapprove } = renderPanel('approve_plan');

        const approve = screen.getByRole('button', { name: 'Approve Plan' });
        expect(approve).not.toHaveAttribute('aria-disabled', 'true');

        await fireEvent.click(approve);

        expect(onapprove).toHaveBeenCalledOnce();
    });

    it('says why reject is unavailable without being asked', () => {
        // Requirement 3.2, first half. Hovering is not available to keyboard or
        // touch users, so the reason cannot only be reachable that way.
        renderPanel('approve_plan');

        expect(screen.getByText(/type a reason in the box below first/i)).toBeInTheDocument();
    });

    it('points the reject control at that explanation', () => {
        renderPanel('approve_plan');

        const described = rejectButton('approve_plan').getAttribute('aria-describedby');
        expect(described).toBeTruthy();
        expect(document.getElementById(described as string)?.textContent).toMatch(
            /type a reason in the box below first/i
        );
    });

    it('explains itself the moment the pointer arrives', async () => {
        // Requirement 3.2, second half.
        renderPanel('approve_plan');
        expect(screen.queryByTestId('reject-tooltip')).not.toBeInTheDocument();

        await fireEvent.mouseEnter(rejectButton('approve_plan'));

        expect(screen.getByTestId('reject-tooltip')).toBeInTheDocument();
    });

    it('stops explaining itself when the pointer leaves', async () => {
        renderPanel('approve_plan');
        await fireEvent.mouseEnter(rejectButton('approve_plan'));

        await fireEvent.mouseLeave(rejectButton('approve_plan'));

        expect(screen.queryByTestId('reject-tooltip')).not.toBeInTheDocument();
    });

    it('explains itself on focus, for anyone not using a pointer', async () => {
        renderPanel('approve_plan');

        await fireEvent.focus(rejectButton('approve_plan'));

        expect(screen.getByTestId('reject-tooltip')).toBeInTheDocument();
    });

    it('does not explain itself once a reason has been typed', async () => {
        renderPanel('approve_plan', 'the plan misses a step');

        await fireEvent.mouseEnter(rejectButton('approve_plan'));

        expect(screen.queryByTestId('reject-tooltip')).not.toBeInTheDocument();
    });

    it('does not rely on the native title attribute to explain itself', () => {
        // Requirement 3.2a: browsers delay `title` by around a second and will
        // not be told otherwise, so it cannot satisfy "instantly". The Send
        // button in ChatInput uses that pattern; copying it here would look
        // right and behave wrongly.
        renderPanel('approve_plan');

        expect(rejectButton('approve_plan')).not.toHaveAttribute('title');
    });

    it('leaves the reject control able to receive a pointer', () => {
        // Requirement 3.2a: a truly `disabled` button emits no pointer events
        // in most browsers, so the hover explanation would never appear. jsdom
        // dispatches those events regardless, which is exactly why the absence
        // of the attribute is asserted rather than trusted to a hover test.
        renderPanel('approve_plan');

        expect(rejectButton('approve_plan')).not.toBeDisabled();
    });

    it('uses no animation to signal the empty state', () => {
        // Requirement 3.5: a shake says nothing to a screen reader and can be
        // unpleasant for readers sensitive to motion.
        renderPanel('approve_plan');

        const classes = rejectButton('approve_plan').className;
        expect(classes).not.toMatch(/animate-/);
        expect(classes).not.toMatch(/shake/);
    });
});
