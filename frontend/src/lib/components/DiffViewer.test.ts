import { describe, expect, it } from 'vitest';
import { fireEvent, render, waitFor, within } from '@testing-library/svelte';
import DiffViewer from './DiffViewer.svelte';
import type { FileDiff } from '$lib/stores/agent';

/** Build a diff for a file. */
function diff(overrides: Partial<FileDiff> = {}): FileDiff {
    return { original: 'before', modified: 'after', truncated: false, ...overrides };
}

/** Render the viewer over a set of paths. */
function renderViewer(diffs: Record<string, FileDiff>) {
    return render(DiffViewer, { diffs });
}

/** The section headings, in the order they appear. */
function headings(container: HTMLElement): string[] {
    return [...container.querySelectorAll('button[aria-expanded]')].map((button) =>
        (button.textContent ?? '').trim().replace(/^›\s*/, '')
    );
}

describe('DiffViewer accordions', () => {
    it('shows one collapsible section per changed file, headed by its path', () => {
        const { container } = renderViewer({ 'src/a.py': diff(), 'src/b.py': diff() });

        // Requirement 5.1.
        const buttons = container.querySelectorAll('button[aria-expanded]');
        expect(buttons).toHaveLength(2);
        expect(headings(container)).toContain('src/a.py');
    });

    it('starts every section collapsed', () => {
        const { container } = renderViewer({ 'a.py': diff(), 'b.py': diff() });

        // Requirement 5.2.
        for (const button of container.querySelectorAll('button[aria-expanded]')) {
            expect(button.getAttribute('aria-expanded')).toBe('false');
        }
    });

    it('orders sections alphabetically by path', () => {
        const { container } = renderViewer({
            'src/zebra.py': diff(),
            'docs/readme.md': diff(),
            'src/apple.py': diff()
        });

        // Requirement 5.3.
        expect(headings(container)).toEqual(['docs/readme.md', 'src/apple.py', 'src/zebra.py']);
    });

    it('expands only the section acted on', async () => {
        const { getByRole } = renderViewer({ 'a.py': diff(), 'b.py': diff() });

        await fireEvent.click(getByRole('button', { name: /a\.py/ }));

        // Requirement 5.4.
        expect(getByRole('button', { name: /a\.py/ }).getAttribute('aria-expanded')).toBe('true');
        expect(getByRole('button', { name: /b\.py/ }).getAttribute('aria-expanded')).toBe('false');
    });

    it('allows several sections open at once', async () => {
        const { getByRole } = renderViewer({ 'a.py': diff(), 'b.py': diff() });

        await fireEvent.click(getByRole('button', { name: /a\.py/ }));
        await fireEvent.click(getByRole('button', { name: /b\.py/ }));

        expect(getByRole('button', { name: /a\.py/ }).getAttribute('aria-expanded')).toBe('true');
        expect(getByRole('button', { name: /b\.py/ }).getAttribute('aria-expanded')).toBe('true');
    });

    it('collapses again when the heading is clicked twice', async () => {
        const { getByRole } = renderViewer({ 'a.py': diff() });

        await fireEvent.click(getByRole('button', { name: /a\.py/ }));
        await fireEvent.click(getByRole('button', { name: /a\.py/ }));

        expect(getByRole('button', { name: /a\.py/ }).getAttribute('aria-expanded')).toBe('false');
    });

    it('keeps a section open when new diffs arrive, and adds new files collapsed', async () => {
        const { getByRole, rerender } = renderViewer({ 'a.py': diff() });

        await fireEvent.click(getByRole('button', { name: /a\.py/ }));
        await rerender({ diffs: { 'a.py': diff(), 'b.py': diff() } });

        // Requirement 5.5.
        expect(getByRole('button', { name: /a\.py/ }).getAttribute('aria-expanded')).toBe('true');
        expect(getByRole('button', { name: /b\.py/ }).getAttribute('aria-expanded')).toBe('false');
    });

    it('keeps a section open when its own diff is updated', async () => {
        const { getByRole, rerender } = renderViewer({ 'a.py': diff({ modified: 'first' }) });

        await fireEvent.click(getByRole('button', { name: /a\.py/ }));
        await rerender({ diffs: { 'a.py': diff({ modified: 'second' }) } });

        expect(getByRole('button', { name: /a\.py/ }).getAttribute('aria-expanded')).toBe('true');
    });

    it('creates no editor for a collapsed section', () => {
        const { container } = renderViewer({ 'a.py': diff(), 'b.py': diff() });

        // Requirement 5.7: an editor per file up front would be wasteful.
        expect(container.querySelectorAll('[data-testid="diff-editor"]')).toHaveLength(0);
    });

    it('creates an editor only for the section that is open', async () => {
        const { container, getByRole } = renderViewer({ 'a.py': diff(), 'b.py': diff() });

        await fireEvent.click(getByRole('button', { name: /a\.py/ }));

        await waitFor(() => {
            expect(container.querySelectorAll('[data-testid="diff-editor"]')).toHaveLength(1);
        });
    });

    it('disposes of the editor when a section closes', async () => {
        const { container, getByRole } = renderViewer({ 'a.py': diff() });

        await fireEvent.click(getByRole('button', { name: /a\.py/ }));
        await waitFor(() => {
            expect(container.querySelectorAll('[data-testid="diff-editor"]')).toHaveLength(1);
        });

        await fireEvent.click(getByRole('button', { name: /a\.py/ }));

        expect(container.querySelectorAll('[data-testid="diff-editor"]')).toHaveLength(0);
    });

    it('shows the truncation notice inside the affected section only', async () => {
        const { container, getByRole } = renderViewer({
            'big.py': diff({ truncated: true }),
            'small.py': diff()
        });

        await fireEvent.click(getByRole('button', { name: /big\.py/ }));
        await fireEvent.click(getByRole('button', { name: /small\.py/ }));

        // Requirement 5.6.
        const sections = container.querySelectorAll('section');
        const big = [...sections].find((section) => section.textContent?.includes('big.py'));
        const small = [...sections].find((section) => section.textContent?.includes('small.py'));
        expect(
            within(big as HTMLElement).getByText(/too large to show in full/)
        ).toBeInTheDocument();
        expect(small?.textContent).not.toContain('too large to show in full');
    });

    it('says so when nothing has been modified', () => {
        const { getByText } = renderViewer({});

        expect(getByText('No files modified yet')).toBeInTheDocument();
    });
});

describe('DiffSection panel identity', () => {
    it('gives colliding paths distinct panel ids', () => {
        // `src/a.py` and `src-a.py` sanitise to the same string, so an id derived
        // from the path would leave `aria-controls` pointing at the wrong panel.
        const { container } = renderViewer({ 'src/a.py': diff(), 'src-a.py': diff() });

        const controls = [...container.querySelectorAll('button[aria-controls]')].map((button) =>
            button.getAttribute('aria-controls')
        );

        expect(new Set(controls).size).toBe(controls.length);
    });

    it('points each heading at its own panel', async () => {
        const { container, getByRole } = renderViewer({ 'a.py': diff(), 'b.py': diff() });

        await fireEvent.click(getByRole('button', { name: /a\.py/ }));

        const heading = getByRole('button', { name: /a\.py/ });
        const target = heading.getAttribute('aria-controls');
        expect(container.querySelector(`#${target}`)).not.toBeNull();
    });
});
