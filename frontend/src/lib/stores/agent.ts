import { writable, get } from 'svelte/store';
import { browser } from '$app/environment';
import {
    agentMessageSchema,
    type ActionMessagePayload,
    type AgentMessagePayload,
    type SystemMessagePayload,
    type CommandAction,
    type SandboxStage,
    type ThoughtMessagePayload
} from '$lib/schemas';

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
    | 'session_busy'
    | 'system';

export interface Message {
    /**
     * Identity of this message within the session, stable across updates.
     *
     * A message that updates in place — a growing run of reasoning, a progress
     * slot, a counted validation fault — is replaced by a new object each time.
     * Keying the rendered list on object identity would therefore destroy and
     * rebuild the element on every update, losing any text selected inside it
     * and restarting its transitions. The identity has to outlive the object.
     */
    id: number;
    type: MessageType;
    content: string;
    timestamp: Date;
    /**
     * The feed slot this message occupies.
     *
     * A keyed message is replaced in place when another arrives bearing the same
     * key, which is how one long running activity reports progress without
     * appending an entry per update. Unkeyed messages are always appended.
     */
    key?: string;
    /**
     * The run of reasoning this message holds.
     *
     * Thoughts sharing a stream identifier are joined into this one message as
     * they stream in. The backend retires the identifier as soon as it emits
     * anything that is not a thought, so reasoning either side of a tool call
     * lands in separate messages.
     */
    streamId?: string;
    /**
     * How many times this message's slot has fired.
     *
     * Set on validation errors, which count repeats of one fault rather than
     * appending a line each time.
     */
    count?: number;
}

export type TaskStatus =
    | 'idle'
    | 'indexing'
    | 'planning'
    | 'waiting_approval'
    | 'executing'
    | 'reviewing'
    | 'waiting_review_approval'
    | 're_working'
    | 'completed'
    | 'merging'
    | 'merged'
    | 'failed'
    | 'cancelled';

export type ConnectionState = 'connecting' | 'connected' | 'disconnected';

/**
 * Why this tab cannot start a session.
 *
 * `another_tab` means a sibling tab answered and holds it. `stale` means the
 * backend refused but no sibling replied, which happens while a closed tab's
 * socket is still being reaped, and is worth retrying.
 */
export type BlockedReason = 'another_tab' | 'stale' | null;

export type ActionType = 'approve_plan' | 'approve_review' | 'approve_merge' | null;

export interface FileDiff {
    original: string;
    modified: string;
    truncated: boolean;
}

export type Theme = 'light' | 'dark' | 'system';

/**
 * The most text one coalesced run of reasoning will retain.
 *
 * A model that reasons without pause would otherwise grow a single message
 * without limit. The cap mirrors the treatment of oversized diffs: keep the
 * beginning, and say plainly that the rest was dropped.
 */
const MAX_THOUGHT_CHARACTERS = 20_000;

/** Prefix marking a feed slot that counts occurrences of one validation fault. */
const VALIDATION_PREFIX = 'validation:';

/** The slot absorbing further distinct faults once `MAX_VALIDATION_SLOTS` is reached. */
const VALIDATION_OVERFLOW_KEY = `${VALIDATION_PREFIX}(other)`;

/**
 * How many distinct validation faults are reported individually.
 *
 * Beyond this, further distinct faults are counted in one overflow slot. A
 * backend that has drifted wholesale should not be able to fill the feed.
 */
const MAX_VALIDATION_SLOTS = 5;

/** Appended to a run of reasoning that hit `MAX_THOUGHT_CHARACTERS`. */
const THOUGHT_TRUNCATION_NOTICE =
    '\n... (truncated: this run of reasoning grew too long to display in full)';

export interface AgentConfig {
    coderModel: string;
    reviewerModel: string;
    embeddingModel: string;
    /**
     * Lines per chunk when indexing, or null to size chunks automatically.
     *
     * Automatic is the default and the only mode that recovers from a chunk the
     * embedding model will not accept. A number is honoured exactly: if the
     * model refuses a chunk that size, indexing fails and says so, rather than
     * quietly using a different size and leaving the setting looking effective.
     */
    chunkLines: number | null;
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
    /**
     * Why the sandbox cannot be used, or null while it can.
     *
     * Held as state rather than added to the feed: the constraint lasts for the
     * session, and a line in the feed would scroll away from it.
     */
    sandboxUnavailable: { stage: SandboxStage; reason: string } | null;
    /** Whether the reader has dismissed the sandbox notice for this session. */
    sandboxNoticeDismissed: boolean;
    config: AgentConfig;
}

const defaultConfig: AgentConfig = {
    coderModel: 'ollama:llama3',
    reviewerModel: 'ollama:llama3',
    embeddingModel: 'nomic-embed-text',
    chunkLines: null,
    repoPath: '/tmp/repo',
    dockerfile: 'FROM python:3.14-slim\n\nWORKDIR /workspace\n',
    secrets: [],
    theme: 'system'
};

const savedConfig = browser ? localStorage.getItem('loom_config') : null;
const initialState: AgentState = {
    messages: [],
    status: 'idle',
    fileDiffs: {},
    requiresAction: false,
    actionType: null,
    connection: 'disconnected',
    blocked: null,
    sandboxUnavailable: null,
    sandboxNoticeDismissed: false,
    // Secrets are deliberately never persisted, so merge the saved config over
    // the defaults rather than replacing them.
    config: savedConfig
        ? { ...defaultConfig, ...JSON.parse(savedConfig), secrets: [] }
        : defaultConfig
};

function createAgentStore() {
    const { subscribe, update } = writable<AgentState>(initialState);
    let socket: WebSocket | null = null;
    let presence: BroadcastChannel | null = null;
    // Remembered so a retry does not need the caller to supply it again.
    let lastUrl = '';
    let reconnectTimeout: number;

    let messagesCreated = 0;

    /**
     * Issue the next message identity.
     *
     * Monotonic rather than contiguous: identities are never reused, so a feed
     * that has been reset cannot collide with what was rendered before it.
     */
    function nextMessageId(): number {
        messagesCreated += 1;
        return messagesCreated;
    }

    const SESSION_CHANNEL = 'loom_session';

    /** Answer sibling tabs so they can tell a real tab from a stale socket. */
    function announcePresence(): BroadcastChannel | null {
        if (!browser || typeof BroadcastChannel === 'undefined') return null;

        const channel = new BroadcastChannel(SESSION_CHANNEL);
        channel.onmessage = (event) => {
            // Only a tab that actually holds a session should answer.
            if (event.data?.type === 'who_has_session') {
                const held = get({ subscribe }).connection === 'connected';
                if (held) channel.postMessage({ type: 'i_have_session' });
            }
        };
        return channel;
    }

    /** Ask whether any sibling tab holds the session. Resolves false on silence. */
    function siblingHoldsSession(timeoutMs = 300): Promise<boolean> {
        if (!browser || typeof BroadcastChannel === 'undefined') return Promise.resolve(false);

        return new Promise((resolve) => {
            const channel = new BroadcastChannel(SESSION_CHANNEL);
            let settled = false;

            const finish = (answer: boolean) => {
                if (settled) return;
                settled = true;
                channel.close();
                resolve(answer);
            };

            channel.onmessage = (event) => {
                if (event.data?.type === 'i_have_session') finish(true);
            };
            channel.postMessage({ type: 'who_has_session' });
            setTimeout(() => finish(false), timeoutMs);
        });
    }

    function connect(url: string) {
        lastUrl = url;
        update((state) => ({ ...state, connection: 'connecting' }));
        socket = new WebSocket(url);

        socket.onopen = () => {
            update((state) => ({ ...state, connection: 'connected', blocked: null }));
            presence?.close();
            presence = announcePresence();
        };

        socket.onmessage = (event) => {
            receive(event.data);
        };

        socket.onclose = () => {
            update((state) => {
                // A refused connection is a decision, not a fault: reconnecting
                // would loop against a backend that will keep refusing.
                if (state.blocked !== null) {
                    return { ...state, connection: 'disconnected' };
                }
                reconnectTimeout = window.setTimeout(() => connect(url), 3000);
                return { ...state, connection: 'disconnected' };
            });
        };

        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            update((state) => ({ ...state, connection: 'disconnected' }));
        };
    }

    /** One-line summary of a tool call, for the chat feed. */
    function describeAction(data: ActionMessagePayload): string {
        const args = data.arguments;
        const detail = Object.entries(args)
            .map(([key, value]) => `${key}=${String(value)}`)
            .join(', ');
        return detail ? `${data.tool_name}(${detail})` : `${data.tool_name}()`;
    }

    /**
     * Join a further unit of reasoning onto the text already displayed.
     *
     * The backend releases reasoning a sentence at a time with surrounding
     * whitespace stripped, so the units are rejoined with a single space.
     */
    function joinThought(existing: string, addition: string): string {
        if (!existing) return addition;
        if (existing.endsWith(THOUGHT_TRUNCATION_NOTICE)) return existing;
        if (!addition) return existing;

        const joined = `${existing} ${addition}`;
        if (joined.length <= MAX_THOUGHT_CHARACTERS) return joined;
        return `${joined.slice(0, MAX_THOUGHT_CHARACTERS)}${THOUGHT_TRUNCATION_NOTICE}`;
    }

    /**
     * Apply a thought, growing the run of reasoning it continues or starting one.
     *
     * Only thoughts coalesce. Every other kind of message is a discrete event a
     * reader wants to see on its own, and the backend enforces that by retiring
     * the stream identifier whenever it emits anything else.
     */
    function applyThought(state: AgentState, data: ThoughtMessagePayload): AgentState {
        const content = data.content;
        const streamId = data.stream_id ? data.stream_id : undefined;

        // The model's reasoning never drives session status; the backend owns it.
        const withStatus: AgentState = state;

        const slot =
            streamId === undefined
                ? -1
                : withStatus.messages.findIndex((existing) => existing.streamId === streamId);

        if (slot === -1) {
            const message: Message = {
                id: nextMessageId(),
                type: 'thought',
                content,
                timestamp: new Date(),
                streamId
            };
            return { ...withStatus, messages: [...withStatus.messages, message] };
        }

        // Keep the original timestamp: this is the same run of reasoning growing,
        // not a new message.
        const messages = [...withStatus.messages];
        const existing = messages[slot];
        messages[slot] = {
            ...existing,
            content: joinThought(existing.content, content)
        };
        return { ...withStatus, messages };
    }

    /**
     * Text for a system message, prefixed with a counter when it reports progress.
     *
     * The counts arrive as numbers precisely so the counter need not be parsed
     * back out of the prose.
     */
    function describeSystem(data: SystemMessagePayload): string {
        const { content, processed, total } = data;
        if (typeof processed !== 'number' || typeof total !== 'number') return content;
        return content ? `[${processed}/${total}] ${content}` : `[${processed}/${total}]`;
    }

    /** Apply a system message, replacing the slot it names or appending if it names none. */
    function applySystem(state: AgentState, data: SystemMessagePayload): AgentState {
        const key = data.key ? data.key : undefined;

        // The backend states the phase outright. There is deliberately no
        // fallback to reading it out of the wording: that inference moved the
        // status on any announcement merely mentioning a sandbox or a merge, and
        // it broke silently whenever an announcement was reworded.
        const base: AgentState = data.status ? { ...state, status: data.status } : state;
        const message: Message = {
            id: nextMessageId(),
            type: 'system',
            content: describeSystem(data),
            timestamp: new Date(),
            key
        };

        if (key === undefined) {
            return { ...base, messages: [...base.messages, message] };
        }

        const slot = base.messages.findIndex((existing) => existing.key === key);
        if (slot === -1) {
            return { ...base, messages: [...base.messages, message] };
        }

        // Replace the content but keep the original timestamp, so an updating
        // message stays where it first appeared rather than jumping to the end.
        const messages = [...base.messages];
        messages[slot] = { ...messages[slot], content: message.content };
        return { ...base, messages };
    }

    /**
     * Validate a raw socket payload and hand it on, or report it as an error.
     *
     * This is the trust boundary for the WebSocket. A payload that is not valid
     * JSON, or that does not match any known message, is surfaced in the feed
     * and logged with the offending text. Silently discarding a message the
     * backend deliberately sent is precisely the failure this guards against, so
     * it is made visible rather than swallowed.
     */
    function receive(raw: unknown): void {
        let decoded: unknown;
        try {
            decoded = JSON.parse(typeof raw === 'string' ? raw : String(raw));
        } catch (error) {
            console.error('Discarded an agent message that was not valid JSON.', { raw, error });
            reportInvalid('not valid JSON');
            return;
        }

        const result = agentMessageSchema.safeParse(decoded);
        if (!result.success) {
            console.error('Discarded an agent message that failed validation.', {
                payload: decoded,
                issues: result.error.issues
            });
            const detail = result.error.issues
                .map((issue) => `${issue.path.join('.') || '(root)'}: ${issue.message}`)
                .join('; ');
            reportInvalid(detail);
            return;
        }

        handleMessage(result.data);
    }

    /** Wording for a validation slot, given how many times it has now fired. */
    function validationText(key: string, count: number): string {
        const base =
            key === VALIDATION_OVERFLOW_KEY
                ? 'Further messages could not be understood; the details are in the browser console.'
                : `A message could not be understood (${key.slice(VALIDATION_PREFIX.length)}).`;
        return count === 1 ? base : `${base} (${count} occurrences)`;
    }

    /** Add a validation slot to the feed. */
    function openValidationSlot(state: AgentState, key: string): AgentState {
        return {
            ...state,
            messages: [
                ...state.messages,
                {
                    id: nextMessageId(),
                    type: 'error',
                    content: validationText(key, 1),
                    timestamp: new Date(),
                    key,
                    count: 1
                }
            ]
        };
    }

    /** Record one more occurrence against a validation slot already in the feed. */
    function bumpValidationSlot(state: AgentState, slot: number): AgentState {
        const messages = [...state.messages];
        const existing = messages[slot];
        const count = (existing.count ?? 1) + 1;
        // The original timestamp is kept: this is the first occurrence being
        // counted up, not a new fault.
        messages[slot] = {
            ...existing,
            content: validationText(existing.key ?? '', count),
            count
        };
        return { ...state, messages };
    }

    /**
     * Put a validation failure in the feed, collapsing repeats of the same fault.
     *
     * A protocol mismatch is systematic rather than isolated: if the backend
     * renames a field, every message of that type fails, and appending one error
     * apiece would bury the session under hundreds of identical lines. Each
     * distinct fault therefore occupies one slot that counts up, and the number
     * of distinct slots is itself capped so that a wholesale mismatch cannot
     * flood the feed either.
     *
     * @param detail What was wrong, used both as the slot key and in the text.
     */
    function reportInvalid(detail: string): void {
        update((state) => {
            const key = `${VALIDATION_PREFIX}${detail}`;

            const existing = state.messages.findIndex((message) => message.key === key);
            if (existing !== -1) return bumpValidationSlot(state, existing);

            const distinct = state.messages.filter((message) =>
                message.key?.startsWith(VALIDATION_PREFIX)
            ).length;
            if (distinct < MAX_VALIDATION_SLOTS) return openValidationSlot(state, key);

            const overflow = state.messages.findIndex(
                (message) => message.key === VALIDATION_OVERFLOW_KEY
            );
            if (overflow !== -1) return bumpValidationSlot(state, overflow);
            return openValidationSlot(state, VALIDATION_OVERFLOW_KEY);
        });
    }

    function handleMessage(data: AgentMessagePayload) {
        update((state) => {
            const nextState: AgentState = { ...state };

            // Diffs drive the diff viewer rather than the chat feed, so they are
            // applied to state without also appearing as a message.
            if (data.type === 'diff') {
                if (data.file_path) {
                    nextState.fileDiffs = {
                        ...nextState.fileDiffs,
                        [data.file_path]: {
                            original: data.original,
                            modified: data.modified,
                            truncated: data.truncated
                        }
                    };
                }
                return nextState;
            }

            // Shown as a banner, so it never joins the feed. Once dismissed it
            // stays dismissed: a later failure describes the same predicament.
            if (data.type === 'sandbox_unavailable') {
                return {
                    ...nextState,
                    sandboxUnavailable: { stage: data.stage, reason: data.reason }
                };
            }

            // System messages may replace a message already in the feed, so they
            // cannot go through the append path below.
            if (data.type === 'system') {
                return applySystem(nextState, data);
            }

            // A thought may grow the message already holding its run of reasoning,
            // so it cannot go through the append path below either.
            if (data.type === 'thought') {
                return applyThought(nextState, data);
            }

            const newMessage: Message = {
                id: nextMessageId(),
                type: data.type,
                content: '',
                timestamp: new Date()
            };

            switch (data.type) {
                case 'action':
                    newMessage.content = describeAction(data);
                    break;
                case 'observation':
                    newMessage.content = data.result;
                    break;
                case 'commit':
                    newMessage.content = `${data.message} (${data.commit_hash.slice(0, 7)})`;
                    break;
                case 'plan':
                    newMessage.content = data.steps
                        .map((step, index) => `${index + 1}. ${step}`)
                        .join('\n');
                    if (data.requires_approval) {
                        nextState.status = 'waiting_approval';
                        nextState.requiresAction = true;
                        nextState.actionType = 'approve_plan';
                    } else {
                        nextState.status = 'executing';
                    }
                    break;
                case 'review':
                    newMessage.content = data.comments;
                    if (data.approved) {
                        nextState.requiresAction = false;
                        nextState.actionType = null;
                    } else {
                        nextState.status = 'waiting_review_approval';
                        nextState.requiresAction = true;
                        nextState.actionType = 'approve_review';
                    }
                    break;
                case 'done':
                    newMessage.content = data.summary;
                    // The backend sends 'done' twice: once when the reviewer has
                    // approved and the merge decision is pending, and again once
                    // the merge has happened.
                    if (state.actionType === 'approve_merge') {
                        nextState.status = 'merged';
                        nextState.requiresAction = false;
                        nextState.actionType = null;
                    } else {
                        nextState.status = 'completed';
                        nextState.requiresAction = true;
                        nextState.actionType = 'approve_merge';
                    }
                    break;
                case 'error':
                    newMessage.content = data.content;
                    nextState.status = 'failed';
                    nextState.requiresAction = false;
                    nextState.actionType = null;
                    break;
                case 'session_busy':
                    // The refusal is authoritative, but only a sibling's reply
                    // proves a real tab holds it: a closed tab's socket may not
                    // have been reaped yet, which is worth retrying.
                    newMessage.content = data.content;
                    nextState.blocked = 'stale';
                    void siblingHoldsSession().then((held) => {
                        update((current) => ({
                            ...current,
                            blocked: held ? 'another_tab' : 'stale'
                        }));
                    });
                    break;
            }

            nextState.messages = [...state.messages, newMessage];
            return nextState;
        });
    }

    return {
        subscribe,
        updateConfig: (config: Partial<AgentConfig>) => {
            update((state) => {
                const repoChanged =
                    config.repoPath !== undefined && config.repoPath !== state.config.repoPath;
                const nextConfig = { ...state.config, ...config };

                if (browser) {
                    // Secrets are omitted rather than blanked. Omission is enforced
                    // at this single point; a blanking transform would have to be
                    // repeated correctly at every future write site.
                    const { secrets, ...persistable } = nextConfig;
                    localStorage.setItem('loom_config', JSON.stringify(persistable));
                }

                if (repoChanged) {
                    // Archive current task by resetting state
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
        sendPrompt: (prompt: string) => {
            const state = get({ subscribe });
            update((state) => ({
                ...state,
                messages: [
                    ...state.messages,
                    {
                        id: nextMessageId(),
                        type: 'prompt',
                        content: prompt,
                        timestamp: new Date()
                    }
                ],
                status: 'indexing'
            }));
            socket?.send(
                JSON.stringify({
                    prompt,
                    coder_model: state.config.coderModel,
                    reviewer_model: state.config.reviewerModel,
                    embedding_model: state.config.embeddingModel,
                    chunk_lines: state.config.chunkLines,
                    repo_path: state.config.repoPath,
                    dockerfile: state.config.dockerfile,
                    secrets: state.config.secrets
                })
            );
        },
        approve: (approved: boolean, reason: string = '') => {
            update((state) => ({ ...state, requiresAction: false, actionType: null }));
            socket?.send(JSON.stringify({ approved, reason }));
        },
        sendAction: (action: CommandAction, reason: string = '') => {
            update((state) => ({ ...state, requiresAction: false, actionType: null }));
            socket?.send(JSON.stringify({ action, reason }));
        },
        /**
         * Ask the backend to interrupt the session.
         *
         * The status is not changed here. The backend decides when it has
         * actually stopped, and says so; guessing would leave the interface
         * claiming idle while the agent was still finishing a step.
         */
        stop: () => {
            socket?.send(JSON.stringify({ action: 'stop' }));
        },

        /** Dismiss the sandbox notice for the rest of the session. */
        dismissSandboxNotice: () => {
            update((state) => ({ ...state, sandboxNoticeDismissed: true }));
        },

        retryConnection: () => {
            if (!lastUrl) return;
            update((state) => ({ ...state, blocked: null, messages: [] }));
            connect(lastUrl);
        },
        disconnect: () => {
            clearTimeout(reconnectTimeout);
            socket?.close();
        },
        connect
    };
}

export const agentStore = createAgentStore();
