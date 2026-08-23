import { describe, expect, it } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/svelte';

import PlanSteps from '$lib/components/PlanSteps.svelte';

/**
 * The two shapes a plan step arrives in, and what each should become.
 *
 * A step with a description is worth collapsing; one without is not, and giving
 * it a control that opens onto nothing is worse than giving it none.
 */
describe('plan steps', () => {
    const withDescriptions = [
        { title: 'Add a timeout', description: 'In run_test_command, defaulting to 30 seconds.' },
        { title: 'Run the suite', description: 'Confirm nothing else regressed.' }
    ];

    it('shows every title at once and no description', async () => {
        render(PlanSteps, { steps: withDescriptions });

        // The point of the list is that a whole plan can be taken in at a
        // glance, which is lost if every description is already showing.
        expect(screen.getByText('Add a timeout')).toBeInTheDocument();
        expect(screen.getByText('Run the suite')).toBeInTheDocument();
        expect(screen.queryByText(/defaulting to 30 seconds/)).not.toBeInTheDocument();
    });

    it('opens one step without opening the others', async () => {
        render(PlanSteps, { steps: withDescriptions });

        await fireEvent.click(screen.getByRole('button', { name: /add a timeout/i }));

        expect(screen.getByText(/defaulting to 30 seconds/)).toBeInTheDocument();
        expect(screen.queryByText(/nothing else regressed/)).not.toBeInTheDocument();
    });

    it('closes again when clicked a second time', async () => {
        render(PlanSteps, { steps: withDescriptions });
        const toggle = screen.getByRole('button', { name: /add a timeout/i });

        await fireEvent.click(toggle);
        await fireEvent.click(toggle);

        expect(screen.queryByText(/defaulting to 30 seconds/)).not.toBeInTheDocument();
    });

    it('announces whether a step is open', async () => {
        render(PlanSteps, { steps: withDescriptions });
        const toggle = screen.getByRole('button', { name: /add a timeout/i });

        // The state has to be in the markup, not only in the chevron, or it is
        // invisible to anything that is not looking at the screen.
        expect(toggle).toHaveAttribute('aria-expanded', 'false');
        await fireEvent.click(toggle);
        expect(toggle).toHaveAttribute('aria-expanded', 'true');
    });

    it('gives a step with no description no control at all', async () => {
        render(PlanSteps, {
            steps: [
                { title: 'Read the file', description: '' },
                { title: 'Edit it', description: 'Change the timeout constant.' }
            ]
        });

        // Present as text, absent as a button: a control promising content it
        // does not have is worse than no control.
        expect(screen.getByText('Read the file')).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /read the file/i })).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: /edit it/i })).toBeInTheDocument();
    });

    it('numbers the steps in the order given', async () => {
        render(PlanSteps, { steps: withDescriptions });

        const list = screen.getByTestId('plan-steps');
        const text = list.textContent ?? '';

        expect(text.indexOf('1.')).toBeLessThan(text.indexOf('2.'));
        expect(text.indexOf('Add a timeout')).toBeLessThan(text.indexOf('Run the suite'));
    });

    it('keeps a long description inside the feed rather than stretching it', async () => {
        const long = 'A very long explanation. '.repeat(60);
        render(PlanSteps, { steps: [{ title: 'Do the thing', description: long }] });

        await fireEvent.click(screen.getByRole('button', { name: /do the thing/i }));

        // `whitespace-pre-wrap` wraps rather than forcing the feed wider, which
        // would push the horizontal scrolling problem onto every other message.
        const body = screen.getByText(long.trim());
        expect(body.className).toContain('whitespace-pre-wrap');
    });
});
