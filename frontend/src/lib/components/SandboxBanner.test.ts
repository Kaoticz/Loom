import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/svelte';
import SandboxBanner from './SandboxBanner.svelte';

describe('SandboxBanner', () => {
    it('names the step that failed', () => {
        render(SandboxBanner, {
            stage: 'image_build',
            reason: 'bad Dockerfile',
            ondismiss: () => {}
        });

        // Requirement 7.7: causes must be distinguishable.
        expect(screen.getByText(/image could not be built/)).toBeInTheDocument();
    });

    it('distinguishes an unreachable runtime from a failed build', () => {
        const { unmount } = render(SandboxBanner, {
            stage: 'runtime_connectivity',
            reason: 'no daemon',
            ondismiss: () => {}
        });
        const unreachable = screen.getByRole('status').textContent;
        unmount();

        render(SandboxBanner, { stage: 'image_build', reason: 'no daemon', ondismiss: () => {} });

        expect(screen.getByRole('status').textContent).not.toBe(unreachable);
    });

    it('explains the consequence, not just the failure', () => {
        render(SandboxBanner, {
            stage: 'runtime_connectivity',
            reason: 'no docker',
            ondismiss: () => {}
        });

        // Requirement 6.6.
        expect(screen.getByText(/Files can still be edited/)).toBeInTheDocument();
        expect(screen.getByText(/no command can be run to verify them/)).toBeInTheDocument();
    });

    it('shows the underlying reason', () => {
        render(SandboxBanner, {
            stage: 'runtime_connectivity',
            reason: 'Cannot connect to the Docker daemon',
            ondismiss: () => {}
        });

        expect(screen.getByText('Cannot connect to the Docker daemon')).toBeInTheDocument();
    });

    it('offers a close control', async () => {
        const ondismiss = vi.fn();
        render(SandboxBanner, { stage: 'runtime_connectivity', reason: 'no docker', ondismiss });

        await fireEvent.click(screen.getByRole('button', { name: /dismiss/i }));

        // Requirement 6.4: dismissal is explicit.
        expect(ondismiss).toHaveBeenCalledOnce();
    });

    it('announces itself to assistive technology', () => {
        render(SandboxBanner, {
            stage: 'runtime_connectivity',
            reason: 'no docker',
            ondismiss: () => {}
        });

        expect(screen.getByRole('status')).toBeInTheDocument();
    });
});
