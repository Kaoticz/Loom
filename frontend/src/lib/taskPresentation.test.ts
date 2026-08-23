import { describe, expect, it } from 'vitest';
import { taskStatusSchema } from './schemas';
import { planEntrySteps, taskStatusStyle, taskStatusStyles } from './taskPresentation';

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

describe('plan entries in task history', () => {
    it('reads plans stored as objects, with their descriptions', () => {
        const lines = planEntrySteps({
            format: 'structured',
            steps: [{ title: 'Add a timeout', description: 'In run_test_command.' }]
        });

        expect(lines).toEqual(['Add a timeout — In run_test_command.']);
    });

    it('still reads plans stored as plain strings', () => {
        // Entries written before plans carried descriptions are still in the
        // database. Reading only the new shape filtered them out entirely, so an
        // older task's plan rendered as nothing rather than as anything a reader
        // could question.
        expect(planEntrySteps({ steps: ['Read the file', 'Edit it'] })).toEqual([
            'Read the file',
            'Edit it'
        ]);
    });

    it('omits a description that was never given', () => {
        expect(planEntrySteps({ steps: [{ title: 'Run the suite', description: '' }] })).toEqual([
            'Run the suite'
        ]);
    });

    it('skips a step with no usable title rather than yielding a blank line', () => {
        expect(planEntrySteps({ steps: [{ description: 'orphaned' }, { title: 'Real' }] })).toEqual(
            ['Real']
        );
    });

    it('survives content that is not a list of steps at all', () => {
        expect(planEntrySteps({})).toEqual([]);
        expect(planEntrySteps({ steps: 'not a list' })).toEqual([]);
    });
});
