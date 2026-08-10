import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/svelte';
import QuestionPanel from './QuestionPanel.svelte';

/** Render the panel with the answers the agent suggested. */
function renderPanel(question: string, options: string[] = []) {
    const onanswer = vi.fn();
    const onfreereply = vi.fn();
    return {
        onanswer,
        onfreereply,
        ...render(QuestionPanel, { question, options, onanswer, onfreereply })
    };
}

describe('QuestionPanel', () => {
    it('shows the question the agent asked', () => {
        renderPanel('Which retry policy did you mean?');

        expect(screen.getByText('Which retry policy did you mean?')).toBeInTheDocument();
    });

    it('offers each answer the agent suggested', () => {
        renderPanel('Which one?', ['The HTTP client', 'The database pool']);

        expect(screen.getByRole('button', { name: 'The HTTP client' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'The database pool' })).toBeInTheDocument();
    });

    it('sends exactly the answer that was chosen', async () => {
        // Requirement 4.4: the agent receives the option, not an index or a
        // paraphrase of it.
        const { onanswer } = renderPanel('Which one?', ['The HTTP client', 'The database pool']);

        await fireEvent.click(screen.getByRole('button', { name: 'The database pool' }));

        expect(onanswer).toHaveBeenCalledExactlyOnceWith('The database pool');
    });

    it('always offers the option of replying freely', () => {
        // Requirement 4.3: the interface adds this, so it is present whatever
        // the agent supplied.
        renderPanel('Which one?', ['The HTTP client']);

        expect(screen.getByTestId('free-reply-option')).toBeInTheDocument();
    });

    it('offers the free reply even when the agent suggested nothing', () => {
        // Requirement 4.7: the free-reply option alone is a usable interface.
        renderPanel('How should this behave?');

        expect(screen.getByTestId('free-reply-option')).toBeInTheDocument();
        expect(screen.getAllByRole('button')).toHaveLength(1);
    });

    it('cannot have the free reply suppressed by what the agent sends', () => {
        // The agent has no way to remove it, including by supplying something
        // that looks like it.
        renderPanel('Which one?', ['Something else — type my own reply']);

        expect(screen.getByTestId('free-reply-option')).toBeInTheDocument();
        expect(screen.getAllByRole('button')).toHaveLength(2);
    });

    it('asks for a typed reply rather than sending one itself', async () => {
        // Choosing it hands over to the prompt box; it is not an answer.
        const { onanswer, onfreereply } = renderPanel('Which one?', ['A']);

        await fireEvent.click(screen.getByTestId('free-reply-option'));

        expect(onfreereply).toHaveBeenCalledOnce();
        expect(onanswer).not.toHaveBeenCalled();
    });
});
