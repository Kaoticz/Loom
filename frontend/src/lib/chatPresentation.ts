import type { ConnectionState, MessageType, TaskStatus } from '$lib/stores/agent';
import type { SandboxStage } from '$lib/schemas';

/**
 * Presentation tables for the chat feed.
 *
 * These are plain data rather than component-local constants so they can be
 * asserted against directly — a message type added to the protocol without a
 * style here would otherwise only show up by looking at the running interface.
 */

/**
 * Types whose content is prose and is rendered as markdown.
 *
 * Tool output, commit lines and system announcements are literal text;
 * rendering them as markdown would mangle command output, file paths and diff
 * fragments.
 */
export const MARKDOWN_TYPES: MessageType[] = [
    'prompt',
    'question',
    'reply',
    'thought',
    'plan',
    'review',
    'done',
    'error',
    'warning'
];

/** The dot beside the connection label. */
export const connectionStyles: Record<ConnectionState, string> = {
    connected: 'bg-green-500',
    connecting: 'bg-yellow-500 animate-pulse',
    disconnected: 'bg-red-500'
};

/** Per-type appearance of a message bubble. */
export const typeStyles: Record<MessageType, string> = {
    prompt: 'bg-blue-50 border-blue-200 text-blue-800 self-end',
    thought: 'bg-gray-50 border-gray-200 text-gray-600 italic',
    action: 'bg-yellow-50 border-yellow-200 text-yellow-800 font-mono text-xs',
    observation: 'bg-green-50 border-green-200 text-green-800 font-mono text-xs',
    plan: 'bg-purple-50 border-purple-200 text-purple-800 dark:border-purple-800 dark:bg-purple-950/30 dark:text-purple-200',
    review: 'bg-orange-50 border-orange-200 text-orange-800',
    commit: 'bg-slate-50 border-slate-200 text-slate-600 font-mono text-xs',
    done: 'bg-green-100 border-green-300 text-green-900 font-bold',
    error: 'bg-red-50 border-red-200 text-red-800',
    // Amber rather than red: something failed but the run is still going, and
    // red would tell the user their session had died when it has not.
    warning: 'bg-amber-50 border-amber-300 text-amber-900',
    // A refusal, not a fault: distinct from `error` so a busy slot does not look
    // like the run failed.
    session_busy: 'bg-amber-50 border-amber-200 text-amber-900',
    // The agent asking rather than telling, and the answer it was given. The
    // reply sits with the prompt on the right, because both are the user
    // speaking.
    question: 'bg-teal-50 border-teal-300 text-teal-900',
    reply: 'bg-blue-50 border-blue-200 text-blue-800 self-end',
    // Deliberately unlike `thought`: system messages are Loom talking about the
    // session, not the model reasoning, and should read as such.
    system: 'bg-indigo-50 border-indigo-200 text-indigo-800 text-xs tracking-wide'
};

/** Wording of the approval panel, by the decision being asked for. */
export const actionLabels: Record<string, { title: string; approve: string; reject: string }> = {
    approve_plan: {
        title: 'Approve this plan?',
        approve: 'Approve Plan',
        reject: 'Reject & Replan'
    },
    approve_review: {
        title: 'The reviewer asked for changes.',
        approve: 'Accept Anyway',
        reject: 'Send Back for Rework'
    },
    approve_merge: {
        title: 'The work is complete.',
        approve: 'Merge into Main',
        reject: 'Reject & Continue'
    }
};

/**
 * The time a message arrived, for display beside its type.
 *
 * Hours and minutes only: a chat feed is read for sequence, and seconds would be
 * noise. The locale is a parameter so the caller can be explicit in tests while
 * the interface follows whatever the reader's browser is set to.
 *
 * @param at When the message arrived.
 * @param locale The locale to format for, or undefined for the browser's own.
 * @returns The formatted time.
 */
export function formatMessageTime(at: Date, locale?: string): string {
    return at.toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit' });
}

/**
 * What each sandbox failure stage means, in the reader's terms.
 *
 * Naming the step is the point: a container runtime that is not running calls
 * for something entirely different from a Dockerfile that will not build, and
 * previously both arrived as the same sentence.
 */
export const sandboxStageLabels: Record<SandboxStage, string> = {
    runtime_connectivity: 'Loom could not reach the container runtime.',
    path_discovery: 'Loom could not work out where the repository lives on the host.',
    image_build: 'The sandbox image could not be built.',
    image_pull: 'The sandbox image could not be downloaded.',
    container_start: 'The sandbox container could not be started.',
    readiness_probe: 'The sandbox container started but did not respond.'
};

/**
 * Whether the agent is working, as opposed to waiting for the user.
 *
 * Requirement 8.2 defines this as the states the agent moves through under its
 * own steam. The waiting states are excluded deliberately: a session paused for
 * an approval has already stopped, and offering to stop it again would be
 * meaningless.
 */
export function isWorking(status: TaskStatus): boolean {
    return (
        status === 'indexing' ||
        status === 'classifying' ||
        status === 'planning' ||
        status === 'answering' ||
        status === 'executing' ||
        status === 'reviewing' ||
        status === 're_working' ||
        status === 'merging'
    );
}

/**
 * Whether a typed prompt can be sent as a new task.
 *
 * Withheld while the agent is working, since the session is not listening for a
 * new instruction, and while any decision is outstanding, since what it is
 * waiting for is that decision. The box itself stays enabled either way: a
 * reader composing the next task should not have it greyed out mid-sentence.
 *
 * @param status Where the session currently is.
 * @param pendingAction The decision being asked for, if any.
 * @returns True when a typed prompt may be sent.
 */
export function canSendPrompt(status: TaskStatus, pendingAction: string | null): boolean {
    return !isWorking(status) && pendingAction === null;
}

/**
 * Whether the prompt box may be sent at all, task or reply.
 *
 * Requirement 4.8: a new task is withheld while a question is outstanding, but
 * the reply must not be, or the interface would be asking for an answer while
 * disabling the only means of giving one.
 *
 * @param status Where the session currently is.
 * @param pendingAction The decision being asked for, if any.
 * @param answeringQuestion Whether the agent is waiting on an answer.
 * @returns True when the box's contents may be sent.
 */
export function canSubmitPrompt(
    status: TaskStatus,
    pendingAction: string | null,
    answeringQuestion: boolean
): boolean {
    return answeringQuestion || canSendPrompt(status, pendingAction);
}

/**
 * What the prompt box is currently for.
 *
 * The box never goes dead, so the placeholder is what tells the reader whether
 * they are writing a task, a reason, or something that cannot be sent yet.
 *
 * @param status Where the session currently is.
 * @param pendingAction The decision being asked for, if any.
 * @returns The placeholder to show.
 */
export function promptPlaceholder(
    status: TaskStatus,
    pendingAction: string | null,
    answeringQuestion: boolean = false
): string {
    // Requirement 4.9a: the box sends a reply rather than a task while a
    // question is outstanding, and a control that silently changes what it does
    // is worse than one that says so.
    if (answeringQuestion) {
        return "Type your reply to the agent's question...";
    }
    if (pendingAction !== null) {
        return 'Add a reason for rejecting, or leave blank...';
    }
    if (isWorking(status)) {
        return 'Write the next task while the agent works...';
    }
    return 'Describe a task for the agent...';
}

/**
 * Message types whose text is worth offering to copy.
 *
 * A prompt is what the reader wrote and may want to reuse or adjust; a `done`
 * message is the result they came for. Everything else is running commentary,
 * and a control on every bubble would be noise.
 */
export const COPYABLE_TYPES: MessageType[] = ['prompt', 'done'];
