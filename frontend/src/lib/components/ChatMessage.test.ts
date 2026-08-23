import { afterEach, describe, expect, it } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/svelte';
import ChatMessage from './ChatMessage.svelte';
import type { Message } from '$lib/stores/agent';

/** Build a message of a given type for rendering. */
function message(overrides: Partial<Message> = {}): Message {
    return {
        id: 1,
        type: 'thought',
        content: 'Considering.',
        timestamp: new Date(),
        ...overrides
    };
}

describe('ChatMessage', () => {
    it('labels the message with its type', () => {
        render(ChatMessage, { message: message({ type: 'system', content: 'Indexing...' }) });

        expect(screen.getByText('system')).toBeInTheDocument();
    });

    it('renders prose types as markdown', () => {
        render(ChatMessage, { message: message({ content: '**bold**' }) });

        expect(screen.getByText('bold').tagName).toBe('STRONG');
    });

    it('renders literal types verbatim, without markdown', () => {
        // An underscored path would be mangled if this went through markdown.
        render(ChatMessage, {
            message: message({ type: 'observation', content: 'read my_file_name.py' })
        });

        expect(screen.getByText('read my_file_name.py')).toBeInTheDocument();
    });
});

describe('ChatMessage timestamp', () => {
    it('shows the time the message arrived', () => {
        const at = new Date('2026-03-04T14:32:07Z');
        const { container } = render(ChatMessage, { message: message({ timestamp: at }) });

        const time = container.querySelector('time');
        expect(time).not.toBeNull();
        expect(time?.textContent?.trim()).toBe(
            at.toLocaleTimeString(undefined, {
                hour: '2-digit',
                minute: '2-digit'
            })
        );
    });

    it('carries a machine readable timestamp', () => {
        const at = new Date('2026-03-04T14:32:07Z');
        const { container } = render(ChatMessage, { message: message({ timestamp: at }) });

        expect(container.querySelector('time')?.getAttribute('datetime')).toBe(at.toISOString());
    });

    it('keeps the type label alongside the time', () => {
        const { getByText, container } = render(ChatMessage, {
            message: message({ type: 'commit', content: 'work' })
        });

        expect(getByText('commit')).toBeInTheDocument();
        expect(container.querySelector('time')).not.toBeNull();
    });
});

describe('ChatMessage copy control', () => {
    /** Install a clipboard that either accepts or refuses writes. */
    function withClipboard(writeText: (text: string) => Promise<void>) {
        Object.defineProperty(navigator, 'clipboard', {
            value: { writeText },
            configurable: true
        });
    }

    afterEach(() => {
        Reflect.deleteProperty(navigator, 'clipboard');
    });

    it.each(['prompt', 'done'] as const)('offers a copy control on a %s message', (type) => {
        // Requirement 12.1.
        render(ChatMessage, { message: message({ type, content: 'text' }) });

        expect(screen.getByRole('button', { name: 'Copy message text' })).toBeInTheDocument();
    });

    it.each(['thought', 'action', 'observation', 'commit', 'error', 'system', 'plan'] as const)(
        'offers no copy control on a %s message',
        (type) => {
            // Requirement 12.3: a control on every bubble would be noise.
            render(ChatMessage, { message: message({ type, content: 'text' }) });

            expect(
                screen.queryByRole('button', { name: 'Copy message text' })
            ).not.toBeInTheDocument();
        }
    );

    it("copies the message's own text", async () => {
        // Requirement 12.2.
        const written: string[] = [];
        withClipboard(async (text) => {
            written.push(text);
        });
        render(ChatMessage, { message: message({ type: 'done', content: 'The work is done.' }) });

        await fireEvent.click(screen.getByRole('button', { name: 'Copy message text' }));

        expect(written).toEqual(['The work is done.']);
    });

    it('confirms that the copy happened', async () => {
        // Requirement 12.4.
        withClipboard(async () => {});
        render(ChatMessage, { message: message({ type: 'prompt', content: 'do the thing' }) });

        await fireEvent.click(screen.getByRole('button', { name: 'Copy message text' }));

        expect(await screen.findByText('Copied')).toBeInTheDocument();
    });

    it('says so when the clipboard is unavailable', async () => {
        // Requirement 12.5, exercised without any stubbing: jsdom has no
        // clipboard, exactly as a browser outside a secure context does not.
        render(ChatMessage, { message: message({ type: 'prompt', content: 'do the thing' }) });

        await fireEvent.click(screen.getByRole('button', { name: 'Copy message text' }));

        expect(await screen.findByText('Copy failed')).toBeInTheDocument();
    });

    it('says so when the clipboard refuses', async () => {
        withClipboard(async () => {
            throw new Error('permission denied');
        });
        render(ChatMessage, { message: message({ type: 'done', content: 'summary' }) });

        await fireEvent.click(screen.getByRole('button', { name: 'Copy message text' }));

        expect(await screen.findByText('Copy failed')).toBeInTheDocument();
    });

    it('shows nothing until the control is used', () => {
        render(ChatMessage, { message: message({ type: 'done', content: 'summary' }) });

        expect(screen.queryByText('Copied')).not.toBeInTheDocument();
        expect(screen.queryByText('Copy failed')).not.toBeInTheDocument();
    });

    it('announces the result to assistive technology', async () => {
        withClipboard(async () => {});
        render(ChatMessage, { message: message({ type: 'done', content: 'summary' }) });

        await fireEvent.click(screen.getByRole('button', { name: 'Copy message text' }));

        // The glyph alone conveys nothing to a reader who cannot see it change.
        expect(screen.getByRole('status')).toHaveTextContent('Copied');
    });
});

describe('plan rendering', () => {
    /**
     * Build a plan message as the store would have produced it.
     *
     * @param format Whether the plan arrived structured or as free text.
     * @param steps The steps it carries.
     */
    function planMessage(
        format: 'structured' | 'text',
        steps: { title: string; description?: string }[]
    ) {
        return {
            id: 1,
            type: 'plan' as const,
            content: steps.map((step, index) => `${index + 1}. ${step.title}`).join('\n'),
            timestamp: new Date('2026-01-01T12:00:00Z'),
            planFormat: format,
            planSteps: steps.map((step) => ({
                title: step.title,
                description: step.description ?? ''
            }))
        };
    }

    it('gives a structured plan collapsible steps', async () => {
        render(ChatMessage, {
            message: planMessage('structured', [
                { title: 'Add a timeout', description: 'In run_test_command.' }
            ])
        });

        expect(screen.getByRole('button', { name: /add a timeout/i })).toBeInTheDocument();
    });

    it('renders a text plan as text, with no controls', async () => {
        // The steps are present either way, so a test that only checked for the
        // titles would pass against accordions here. What matters is that a plan
        // recovered from free text is not dressed up as one that was not: the
        // descriptions it would promise do not exist.
        render(ChatMessage, {
            message: planMessage('text', [{ title: 'Read the file' }, { title: 'Edit it' }])
        });

        expect(screen.queryByTestId('plan-steps')).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /read the file/i })).not.toBeInTheDocument();
        // The step text is still shown; markdown turns the numbering into a
        // list marker, so the digit is not itself text to search for.
        expect(screen.getByText('Read the file')).toBeInTheDocument();
        expect(screen.getByText('Edit it')).toBeInTheDocument();
    });
});
