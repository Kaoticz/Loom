import { describe, expect, it } from 'vitest';
import { taskStatusSchema } from './schemas';
import { taskStatusStyle, taskStatusStyles } from './taskPresentation';

describe('task status styling', () => {
    it('styles every status the backend can report', () => {
        // Requirement 8.9: an unstyled status renders as though the task had
        // never started, which is exactly wrong for a cancelled one.
        for (const status of taskStatusSchema.options) {
            expect(taskStatusStyles, `no style for '${status}'`).toHaveProperty(status);
        }
    });

    it('distinguishes cancelled from failed', () => {
        // Work the user stopped is not work that broke.
        expect(taskStatusStyles.cancelled).not.toBe(taskStatusStyles.failed);
    });

    it('distinguishes cancelled from a task that never started', () => {
        expect(taskStatusStyles.cancelled).not.toBe(taskStatusStyles.initialized);
    });

    it('distinguishes an answered task from a completed or merged one', () => {
        // Requirement 2.8. `completed` here means the reviewer approved and a
        // merge is pending; `merged` means it happened. An answered task did
        // neither, and a badge shared with either would say it had.
        expect(taskStatusStyles.answered).not.toBe(taskStatusStyles.completed);
        expect(taskStatusStyles.answered).not.toBe(taskStatusStyles.merged);
        expect(taskStatusStyles.answered).not.toBe(taskStatusStyles.failed);
        expect(taskStatusStyles.answered).not.toBe(taskStatusStyles.initialized);
    });

    it('falls back for a status it does not recognise', () => {
        expect(taskStatusStyle('invented_by_a_newer_backend')).toBe(taskStatusStyles.initialized);
    });

    it('returns the real style for a known status', () => {
        expect(taskStatusStyle('cancelled')).toBe(taskStatusStyles.cancelled);
    });
});
