import { writable, get } from 'svelte/store';
import { z } from 'zod';
import { browser } from '$app/environment';
import {
    agentMessageSchema,
    type ActionMessagePayload,
    type AgentMessagePayload,
    type SystemMessagePayload,
    type CommandAction,
    type PlanFormat,
    type PlanStep,
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
     * A message that updates in place — a growing run of reasoning, a progress
     * slot, a counted validation fault — is replaced by a new object each time.
     * Keying the rendered list on object identity would therefore destroy and
     * rebuild the element on every update, losing any text selected inside it
     * and restarting its transitions. The identity has to outlive the object.
     */
    id: number;
    type: MessageType;
    content: string;
    /**
     * The steps of a plan, when this message is one.
     *
     * Kept beside `content` rather than replacing it: a plan that arrived as
     * free text is shown as text, and the rendered form is what it shows.
     */
    planSteps?: PlanStep[];
    /** Whether the plan arrived structured, and so how it should be shown. */
    planFormat?: PlanFormat;
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
 * `another_tab` means a sibling tab answered and holds it. `stale` means the
 * backend refused but no sibling replied, which happens while a closed tab's
 * socket is still being reaped, and is worth retrying.
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

/** Tools that confirm a plan step has finished successfully. */
const STEP_COMPLETION_TOOLS = new Set(['complete_step', 'accept_step_without_changes']);

/**
 * The values a WebSocket reports as its `readyState`, named.
 *
 * Spelled out rather than read from `WebSocket.CONNECTING` and friends, because
 * the tests replace the global with a stand-in that has no static constants, and
 * a comparison against `undefined` would silently never match.
 */
const SocketReadyState = {
    connecting: 0,
    open: 1,
    closing: 2,
    closed: 3
} as const;

/**
 * What one tab says to another about who is holding the session.
 *
 * `from` identifies the store instance that sent it, and is what makes the
 * exchange trustworthy: `BroadcastChannel` delivers a message to every other
 * channel on the origin — including further channels belonging to the tab that
 * sent it — and the event itself does not say where it came from. Without an
 * identity a tab hears its own query, answers it, and concludes that a sibling
 * holds the session.
 */
const presenceMessageSchema = z.object({
    type: z.enum(['who_has_session', 'i_have_session']),
    from: z.string()
});

type PresenceMessage = z.infer<typeof presenceMessageSchema>;

/**
 * Mint an identifier for one store instance.
 *
 * @returns An identifier no other store instance is expected to share.
 */
function createInstanceId(): string {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
        return crypto.randomUUID();
    }
    return `loom-${Math.random().toString(36).slice(2)}-${Date.now().toString(36)}`;
}

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
    /**
     * Context window in tokens budgeted for the coder agent, or null for the default.
     *
     * Separate from the planner's. The two agents share one model setting, but
     * they do not share a shape of work: the coder runs a long tool-heavy loop
     * whose history grows with every read and test, while the planner makes one
     * structured call. Budgeting them together would hold one of them wrong.
     */
    coderContextWindow: number | null;
    /** Context window in tokens budgeted for the planner agent, or null for the default. */
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
    /**
     * Why the sandbox cannot be used, or null while it can.
     *
     * Held as state rather than added to the feed: the constraint lasts for the
     * session, and a line in the feed would scroll away from it.
     */
    sandboxUnavailable: { stage: SandboxStage; reason: string } | null;
    /**
     * How full the model's context is, or null before the first reading.
     *
     * State rather than a feed entry: it is replaced on every request, and a
     * line per request would bury the conversation in a number that is only
     * worth a glance. `resolved` is false when the window is a fallback rather
     * than the model's own, so the reading can be shown as the guess it is.
     */
    contextUsage: { usedTokens: number; windowTokens: number; resolved: boolean } | null;
    /** Whether the reader has dismissed the sandbox notice for this session. */
    sandboxNoticeDismissed: boolean;
    /**
     * The question the agent is waiting on, or null when it is not waiting.
     *
     * Held as state rather than read back out of the feed: the feed keeps the
     * exchange as a record, while this is what the interface acts on, and
     * deriving one from the other would tie the two together for no gain.
     */
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

    /**
     * Who this store instance is, for the length of its life.
     *
     * One instance per tab, so this identifies the tab as far as the presence
     * protocol is concerned. Every message it sends carries it, and any message
     * bearing it came from here and is disregarded.
     */
    const instanceId: string = createInstanceId();

    /** Read a presence message, or null if it was not one. */
    function readPresence(data: unknown): PresenceMessage | null {
        const result = presenceMessageSchema.safeParse(data);
        if (!result.success) return null;
        // Anything carrying this instance's identity was posted by one of this
        // tab's own channels, and says nothing about any sibling.
        if (result.data.from === instanceId) return null;
        return result.data;
    }

    /** Answer sibling tabs so they can tell a real tab from a stale socket. */
    function announcePresence(): BroadcastChannel | null {
        if (!browser || typeof BroadcastChannel === 'undefined') return null;

        const channel = new BroadcastChannel(SESSION_CHANNEL);
        channel.onmessage = (event: MessageEvent<unknown>) => {
            const message = readPresence(event.data);
            if (message === null || message.type !== 'who_has_session') return;

            // Only a tab that actually holds a session should answer.
            const held = get({ subscribe }).connection === 'connected';
            if (!held) return;

            const reply: PresenceMessage = { type: 'i_have_session', from: instanceId };
            channel.postMessage(reply);
        };
        return channel;
    }

    /**
     * Stop answering presence queries.
     *
     * Called wherever a session ends or is replaced, so that a tab never leaves
     * behind a channel still willing to speak for a session it no longer has.
     */
    function stopAnnouncing(): void {
        presence?.close();
        presence = null;
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

            channel.onmessage = (event: MessageEvent<unknown>) => {
                const message = readPresence(event.data);
                if (message === null || message.type !== 'i_have_session') return;
                finish(true);
            };

            const query: PresenceMessage = { type: 'who_has_session', from: instanceId };
            channel.postMessage(query);
            setTimeout(() => finish(false), timeoutMs);
        });
    }

    /** Whether a socket is already open, or on its way to being open. */
    function sessionIsLive(): boolean {
        if (socket === null) return false;
        return (
            socket.readyState === SocketReadyState.connecting ||
            socket.readyState === SocketReadyState.open
        );
    }

    /**
     * End the current socket without letting it speak for the session afterwards.
     *
     * @returns Nothing; the socket reference is cleared as part of closing.
     */
    function teardown(): void {
        clearTimeout(reconnectTimeout);
        stopAnnouncing();
        const closing = socket;
        // Cleared before closing, so the handlers still attached to the old
        // socket recognise themselves as superseded and stay quiet.
        socket = null;
        closing?.close();
    }

    function connect(url: string) {
        // Requirement 1.2. Connecting is idempotent: the layout establishes the
        // session once, but nothing stops a second call arriving, and a second
        // socket would be refused as busy by a backend that is already talking
        // to the first one.
        if (sessionIsLive()) return;

        lastUrl = url;
        stopAnnouncing();
        update((state) => ({ ...state, connection: 'connecting' }));

        const opened = new WebSocket(url);
        socket = opened;

        opened.onopen = () => {
            if (socket !== opened) return;
            update((state) => ({ ...state, connection: 'connected', blocked: null }));
            stopAnnouncing();
            presence = announcePresence();
        };

        opened.onmessage = (event) => {
            if (socket !== opened) return;
            receive(event.data);
        };

        opened.onclose = () => {
            // A socket that has already been replaced must not report a
            // disconnection, schedule a reconnect, or otherwise speak for the
            // session that succeeded it.
            if (socket !== opened) return;

            // The session is over, so this tab no longer answers for it.
            stopAnnouncing();

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

        opened.onerror = (error) => {
            if (socket !== opened) return;
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

            // Shown as a gauge, so it never joins the feed.
            if (data.type === 'context_usage') {
                return {
                    ...nextState,
                    contextUsage: {
                        usedTokens: data.used_tokens,
                        windowTokens: data.window_tokens,
                        resolved: data.resolved
                    }
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

            if (
                data.type === 'observation' &&
                data.status === 'success' &&
                STEP_COMPLETION_TOOLS.has(data.tool_name)
            ) {
                // A step summary carries the useful outcome forward; retaining its
                // individual tool calls would only bury the next step's progress.
                return {
                    ...nextState,
                    messages: state.messages.filter(
                        (message) => message.type !== 'action' && message.type !== 'observation'
                    )
                };
            }

            // The question joins the feed and also becomes the state the
            // interface acts on, so it cannot go through the append path alone.
            if (data.type === 'question') {
                nextState.pendingQuestion = {
                    question: data.question,
                    options: [...data.options]
                };
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
                    newMessage.planSteps = data.steps;
                    newMessage.planFormat = data.format;
                    newMessage.content = data.steps
                        .map((step, index) => `${index + 1}. ${step.title}`)
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
                        // A rejected verdict is a user decision: they can send the
                        // work back or deliberately merge despite the review.
                        nextState.requiresAction = true;
                        nextState.actionType = 'review_rejected';
                    }
                    break;
                case 'done':
                    newMessage.content = data.summary;
                    // A task that changed nothing carries no branch, because
                    // there is none: it produced an answer. Offering a merge
                    // would promise a diff that does not exist.
                    if (!data.branch_name) {
                        nextState.status = 'answered';
                        nextState.requiresAction = false;
                        nextState.actionType = null;
                        break;
                    }
                    if (data.merged) {
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
                case 'warning':
                    // Deliberately leaves `status` alone. The session is still
                    // running, and marking it failed would say otherwise.
                    newMessage.content = data.content;
                    break;
                case 'compaction':
                    newMessage.content =
                        'Loom shortened the conversation to fit the context window ' +
                        `(estimated ${data.tokens_before.toLocaleString()} to ` +
                        `${data.tokens_after.toLocaleString()} tokens).`;
                    break;
                case 'question':
                    newMessage.content = data.question;
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

                if (browser && typeof localStorage !== 'undefined') {
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
        /**
         * Answer the question the agent is waiting on.
         *
         * The reply is recorded in the feed as well as sent, so the exchange
         * reads as a conversation rather than the question sitting there
         * unanswered.
         *
         * @param reply - What the user chose or typed.
         */
        answerQuestion: (reply: string) => {
            const answer = reply.trim();
            if (!answer) return;
            update((state) => ({
                ...state,
                pendingQuestion: null,
                messages: [
                    ...state.messages,
                    {
                        id: nextMessageId(),
                        type: 'reply',
                        content: answer,
                        timestamp: new Date()
                    }
                ]
            }));
            socket?.send(JSON.stringify({ action: 'reply', reply: answer }));
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
                    coder_context_window: state.config.coderContextWindow ?? 0,
                    planner_context_window: state.config.plannerContextWindow ?? 0,
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
            // The refused socket may not have been reaped yet, and the
            // idempotency guard would otherwise read it as a live session and
            // decline to try again — which is the retry doing nothing.
            teardown();
            connect(lastUrl);
        },
        /**
         * End the session deliberately.
         *
         * Distinct from a socket that drops: no reconnect is scheduled, because
         * closing was the point. The state is set here rather than left to the
         * socket's own `onclose`, which stays silent once superseded.
         */
        disconnect: () => {
            teardown();
            update((state) => ({ ...state, connection: 'disconnected' }));
        },
        connect
    };
}

export const agentStore = createAgentStore();
