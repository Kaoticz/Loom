import { browser } from '$app/environment';
import type { CommandAction, PlanFormat, PlanStep, SandboxStage } from '$lib/schemas';
import { get, writable } from 'svelte/store';
import { createAgentEventReducer } from './agent-events';
import { createSessionPresence } from './session-presence';
import { createSessionSocket } from './session-socket';

export type MessageType =
    | 'prompt'
    | 'thought'
    | 'action'
    | 'observation'
    | 'plan'
    | 'review'
    | 'commit'
    | 'done'
    | 'error'
    | 'warning'
    | 'compaction'
    | 'session_busy'
    | 'question'
    | 'reply'
    | 'system';

export interface Message {
    /**
     * Identity of this message within the session, stable across updates.
     *
     * A message that updates in place is replaced by a new object each time.
     * Keeping this identity stable prevents a keyed rendered element from being
     * rebuilt, losing selected text and restarting transitions.
     */
    id: number;
    type: MessageType;
    content: string;
    /**
     * The steps of a plan, when this message is one.
     *
     * Kept beside `content`: a plan that arrived as free text is still shown as text.
     */
    planSteps?: PlanStep[];
    /** Whether the plan arrived structured, and so how it should be shown. */
    planFormat?: PlanFormat;
    timestamp: Date;
    /**
     * The feed slot this message occupies.
     *
     * A keyed message is replaced in place for progress updates; unkeyed messages
     * are always appended.
     */
    key?: string;
    /**
     * The run of reasoning this message holds.
     *
     * Thoughts sharing an identifier stream into one message. The backend retires
     * that identifier on every non-thought event, separating reasoning around work.
     */
    streamId?: string;
    /** How many times this message's validation-fault slot has fired. */
    count?: number;
}

export type TaskStatus =
    | 'idle'
    | 'indexing'
    | 'classifying'
    | 'planning'
    | 'waiting_approval'
    | 'executing'
    | 'answering'
    | 'reviewing'
    | 're_working'
    | 'completed'
    | 'answered'
    | 'merging'
    | 'merged'
    | 'failed'
    | 'cancelled';

export type ConnectionState = 'connecting' | 'connected' | 'disconnected';

/**
 * Why this tab cannot start a session.
 *
 * `another_tab` means a sibling answered and holds it. `stale` means the backend
 * refused while a closed tab's socket may still be reaped, so retrying is useful.
 */
export type BlockedReason = 'another_tab' | 'stale' | null;

export type ActionType = 'approve_plan' | 'approve_merge' | 'review_rejected' | null;

/** A question the agent is waiting on an answer to. */
export interface PendingQuestion {
    question: string;
    options: string[];
}

export interface FileDiff {
    original: string;
    modified: string;
    truncated: boolean;
}

export type Theme = 'light' | 'dark' | 'system';

export interface AgentConfig {
    coderModel: string;
    reviewerModel: string;
    embeddingModel: string;
    /**
     * Lines per chunk when indexing, or null to size chunks automatically.
     *
     * A chosen number is honoured exactly. Automatic is the only mode that
     * recovers when an embedding model refuses a chunk.
     */
    chunkLines: number | null;
    /**
     * Context window for the coder agent, or null for the default.
     *
     * Coder and planner share a model setting but not a shape of work, so their
     * budgets remain separate.
     */
    coderContextWindow: number | null;
    /** Context window for the planner agent, or null for the default. */
    plannerContextWindow: number | null;
    repoPath: string;
    dockerfile: string;
    secrets: { key: string; value: string }[];
    theme: Theme;
}

export interface AgentState {
    messages: Message[];
    status: TaskStatus;
    fileDiffs: Record<string, FileDiff>;
    requiresAction: boolean;
    actionType: ActionType;
    connection: ConnectionState;
    blocked: BlockedReason;
    /** Why the sandbox cannot be used; state keeps this persistent constraint out of the feed. */
    sandboxUnavailable: { stage: SandboxStage; reason: string } | null;
    /** A replaceable context reading, rather than a chat line for every request. */
    contextUsage: { usedTokens: number; windowTokens: number; resolved: boolean } | null;
    /** Whether the reader has dismissed the sandbox notice for this session. */
    sandboxNoticeDismissed: boolean;
    /** The question the interface acts on, separate from the feed's record of the exchange. */
    pendingQuestion: PendingQuestion | null;
    config: AgentConfig;
}

const defaultConfig: AgentConfig = {
    coderModel: 'ollama:llama3',
    reviewerModel: 'ollama:llama3',
    embeddingModel: 'nomic-embed-text',
    chunkLines: null,
    coderContextWindow: null,
    plannerContextWindow: null,
    repoPath: '/tmp/repo',
    dockerfile: 'FROM python:3.14-slim\n\nWORKDIR /workspace\n',
    secrets: [],
    theme: 'system'
};

const savedConfig =
    browser && typeof localStorage !== 'undefined' ? localStorage.getItem('loom_config') : null;
const initialState: AgentState = {
    messages: [],
    status: 'idle',
    fileDiffs: {},
    requiresAction: false,
    actionType: null,
    connection: 'disconnected',
    blocked: null,
    sandboxUnavailable: null,
    contextUsage: null,
    sandboxNoticeDismissed: false,
    pendingQuestion: null,
    // Secrets are deliberately never persisted, so merge saved settings into the
    // defaults rather than replacing the configuration wholesale.
    config: savedConfig
        ? { ...defaultConfig, ...JSON.parse(savedConfig), secrets: [] }
        : defaultConfig
};

function createAgentStore() {
    const { subscribe, update } = writable<AgentState>(initialState);
    const presence = createSessionPresence(() => get({ subscribe }).connection === 'connected');
    const events = createAgentEventReducer(update, presence.siblingHoldsSession);
    const connection = createSessionSocket({
        receive: events.receive,
        setConnection: (state: ConnectionState): void => {
            update((current) => ({
                ...current,
                connection: state,
                ...(state === 'connected' ? { blocked: null } : {})
            }));
        },
        beforeConnect: presence.stop,
        connected: (): void => {
            presence.stop();
            presence.announce();
        },
        disconnected: presence.stop,
        closed: (scheduleReconnect: () => void): void => {
            update((state) => {
                if (state.blocked !== null) return { ...state, connection: 'disconnected' };
                scheduleReconnect();
                return { ...state, connection: 'disconnected' };
            });
        }
    });

    return {
        subscribe,
        updateConfig: (config: Partial<AgentConfig>): void => {
            update((state) => {
                const repoChanged =
                    config.repoPath !== undefined && config.repoPath !== state.config.repoPath;
                const nextConfig: AgentConfig = { ...state.config, ...config };

                if (browser && typeof localStorage !== 'undefined') {
                    // Omit secrets at this single persistence boundary rather than
                    // relying on every future write site to blank them correctly.
                    const { secrets, ...persistable } = nextConfig;
                    localStorage.setItem('loom_config', JSON.stringify(persistable));
                }

                if (repoChanged) {
                    return {
                        ...state,
                        config: nextConfig,
                        messages: [],
                        status: 'idle',
                        fileDiffs: {},
                        requiresAction: false,
                        actionType: null
                    };
                }

                return { ...state, config: nextConfig };
            });
        },
        /**
         * Answer the question the agent is waiting on.
         *
         * The reply is recorded as well as sent so the exchange reads as a
         * conversation rather than an unanswered question.
         *
         * @param reply - What the user chose or typed.
         * @returns Nothing.
         */
        answerQuestion: (reply: string): void => {
            const answer = reply.trim();
            if (!answer) return;
            update((state) => ({
                ...state,
                pendingQuestion: null,
                messages: [
                    ...state.messages,
                    {
                        id: events.nextMessageId(),
                        type: 'reply',
                        content: answer,
                        timestamp: new Date()
                    }
                ]
            }));
            connection.send(JSON.stringify({ action: 'reply', reply: answer }));
        },
        sendPrompt: (prompt: string): void => {
            const state = get({ subscribe });
            update((current) => ({
                ...current,
                messages: [
                    ...current.messages,
                    {
                        id: events.nextMessageId(),
                        type: 'prompt',
                        content: prompt,
                        timestamp: new Date()
                    }
                ],
                status: 'indexing'
            }));
            connection.send(
                JSON.stringify({
                    prompt,
                    coder_model: state.config.coderModel,
                    reviewer_model: state.config.reviewerModel,
                    embedding_model: state.config.embeddingModel,
                    chunk_lines: state.config.chunkLines,
                    coder_context_window: state.config.coderContextWindow ?? 0,
                    planner_context_window: state.config.plannerContextWindow ?? 0,
                    repo_path: state.config.repoPath,
                    dockerfile: state.config.dockerfile,
                    secrets: state.config.secrets
                })
            );
        },
        approve: (approved: boolean, reason: string = ''): void => {
            update((state) => ({ ...state, requiresAction: false, actionType: null }));
            connection.send(JSON.stringify({ approved, reason }));
        },
        sendAction: (action: CommandAction, reason: string = ''): void => {
            update((state) => ({ ...state, requiresAction: false, actionType: null }));
            connection.send(JSON.stringify({ action, reason }));
        },
        /**
         * Ask the backend to interrupt the session.
         *
         * The backend decides when it has stopped; claiming idle here could be
         * wrong while it finishes a step.
         *
         * @returns Nothing.
         */
        stop: (): void => {
            connection.send(JSON.stringify({ action: 'stop' }));
        },
        /**
         * Dismiss the sandbox notice for the rest of the session.
         *
         * @returns Nothing.
         */
        dismissSandboxNotice: (): void => {
            update((state) => ({ ...state, sandboxNoticeDismissed: true }));
        },
        retryConnection: (): void => {
            if (!connection.hasUrl()) return;
            update((state) => ({ ...state, blocked: null, messages: [] }));
            // The refused socket might not yet be reaped. Teardown must precede
            // retry so the idempotency guard does not decline the requested retry.
            connection.retry();
        },
        /**
         * End the session deliberately without scheduling a reconnect.
         *
         * This differs from a dropped socket: its stale close handler is silent
         * after teardown, so the deliberate close cannot schedule a retry.
         *
         * @returns Nothing.
         */
        disconnect: (): void => {
            connection.teardown();
            update((state) => ({ ...state, connection: 'disconnected' }));
        },
        connect: connection.connect
    };
}

/** The shared agent session state and action facade. */
export const agentStore = createAgentStore();
