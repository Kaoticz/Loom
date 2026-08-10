import type { TaskStatus } from '$lib/schemas';

/**
 * Appearance of task statuses in the history view.
 *
 * Typed as a total map over `TaskStatus`, so a status added to the protocol
 * fails to compile here rather than falling through to the unknown-status
 * styling and rendering as though the task had never started.
 */
export const taskStatusStyles: Record<TaskStatus, string> = {
    initialized: 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-200',
    running: 'bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-200',
    completed: 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-200',
    merged: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200',
    // Deliberately unlike `completed` and `merged`: this task produced an answer
    // rather than a diff, so nothing was merged and nothing is waiting to be.
    answered: 'bg-sky-100 text-sky-800 dark:bg-sky-900/40 dark:text-sky-200',
    failed: 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-200',
    // Deliberately unlike `failed`: the user stopped this, it did not break.
    cancelled: 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200'
};

/**
 * Styling for a status, falling back for anything unrecognised.
 *
 * A backend newer than this frontend can report a status not listed above, and
 * an unstyled badge is better than a broken page.
 *
 * @param status The status reported for the task.
 * @returns The classes to apply to the status badge.
 */
export function taskStatusStyle(status: string): string {
    return taskStatusStyles[status as TaskStatus] ?? taskStatusStyles.initialized;
}
