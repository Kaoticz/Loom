import {
    agentMessageSchema,
    type ActionMessagePayload,
    type AgentMessagePayload,
    type SystemMessagePayload,
    type ThoughtMessagePayload
} from '$lib/schemas';
import type { AgentState, Message } from './agent';

/** A reasoning run is capped so an unbroken stream cannot grow without limit. */
const MAX_THOUGHT_CHARACTERS = 20_000;
/** Prefix marking a feed slot that counts one validation fault. */
const VALIDATION_PREFIX = 'validation:';
/** The slot absorbing further distinct faults after the individual-slot cap. */
const VALIDATION_OVERFLOW_KEY = `${VALIDATION_PREFIX}(other)`;
/** A wholesale protocol mismatch must not fill the feed with distinct errors. */
const MAX_VALIDATION_SLOTS = 5;
/** Appended to a reasoning run that reaches its display cap. */
const THOUGHT_TRUNCATION_NOTICE =
    '\n... (truncated: this run of reasoning grew too long to display in full)';
/** Tools that confirm a plan step has successfully finished. */
const STEP_COMPLETION_TOOLS = new Set(['complete_step', 'accept_step_without_changes']);

type UpdateAgentState = (updater: (state: AgentState) => AgentState) => void;

export interface AgentEventReducer {
    nextMessageId(): number;
    receive(raw: unknown): void;
}

/**
 * Create the validated inbound-event reducer for an agent store.
 *
 * @param update - Applies one state transition to the owning Svelte store.
 * @param siblingHoldsSession - Looks up whether a sibling tab owns a refused session.
 * @returns The inbound payload receiver and shared message identity generator.
 */
export function createAgentEventReducer(
    update: UpdateAgentState,
    siblingHoldsSession: () => Promise<boolean>
): AgentEventReducer {
    let messagesCreated = 0;

    /** Monotonic identities are never reused after resetting the visible feed. */
    function nextMessageId(): number {
        messagesCreated += 1;
        return messagesCreated;
    }

    function describeAction(data: ActionMessagePayload): string {
        const detail = Object.entries(data.arguments)
            .map(([key, value]) => `${key}=${String(value)}`)
            .join(', ');
        return detail ? `${data.tool_name}(${detail})` : `${data.tool_name}()`;
    }

    function joinThought(existing: string, addition: string): string {
        // Thought units are stripped by the backend, so rejoin them with one space.
        if (!existing) return addition;
        if (existing.endsWith(THOUGHT_TRUNCATION_NOTICE)) return existing;
        if (!addition) return existing;

        const joined = `${existing} ${addition}`;
        if (joined.length <= MAX_THOUGHT_CHARACTERS) return joined;
        return `${joined.slice(0, MAX_THOUGHT_CHARACTERS)}${THOUGHT_TRUNCATION_NOTICE}`;
    }

    function applyThought(state: AgentState, data: ThoughtMessagePayload): AgentState {
        const content = data.content;
        const streamId = data.stream_id ? data.stream_id : undefined;
        // Only thoughts coalesce; all other event types remain discrete feed events.
        const slot =
            streamId === undefined
                ? -1
                : state.messages.findIndex((existing) => existing.streamId === streamId);

        if (slot === -1) {
            const message: Message = {
                id: nextMessageId(),
                type: 'thought',
                content,
                timestamp: new Date(),
                streamId
            };
            return { ...state, messages: [...state.messages, message] };
        }

        // This is the same reasoning run growing, not a new event, so retain its timestamp.
        const messages = [...state.messages];
        const existing = messages[slot];
        messages[slot] = { ...existing, content: joinThought(existing.content, content) };
        return { ...state, messages };
    }

    function describeSystem(data: SystemMessagePayload): string {
        const { content, processed, total } = data;
        if (typeof processed !== 'number' || typeof total !== 'number') return content;
        return content ? `[${processed}/${total}] ${content}` : `[${processed}/${total}]`;
    }

    function applySystem(state: AgentState, data: SystemMessagePayload): AgentState {
        const key = data.key ? data.key : undefined;
        // The backend states the phase. Inferring it from prose breaks whenever
        // wording changes or merely mentions a phase.
        const base: AgentState = data.status ? { ...state, status: data.status } : state;
        const message: Message = {
            id: nextMessageId(),
            type: 'system',
            content: describeSystem(data),
            timestamp: new Date(),
            key
        };

        if (key === undefined) return { ...base, messages: [...base.messages, message] };

        const slot = base.messages.findIndex((existing) => existing.key === key);
        if (slot === -1) return { ...base, messages: [...base.messages, message] };

        // Keep the first timestamp so an updating progress slot stays in place.
        const messages = [...base.messages];
        messages[slot] = { ...messages[slot], content: message.content };
        return { ...base, messages };
    }

    function validationText(key: string, count: number): string {
        const base =
            key === VALIDATION_OVERFLOW_KEY
                ? 'Further messages could not be understood; the details are in the browser console.'
                : `A message could not be understood (${key.slice(VALIDATION_PREFIX.length)}).`;
        return count === 1 ? base : `${base} (${count} occurrences)`;
    }

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

    function bumpValidationSlot(state: AgentState, slot: number): AgentState {
        const messages = [...state.messages];
        const existing = messages[slot];
        const count = (existing.count ?? 1) + 1;
        // Count the first occurrence rather than creating another timestamped error.
        messages[slot] = {
            ...existing,
            content: validationText(existing.key ?? '', count),
            count
        };
        return { ...state, messages };
    }

    function reportInvalid(detail: string): void {
        // Repeated protocol faults occupy one counted slot, and distinct slots are
        // capped so backend drift cannot bury real session events.
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

    function handleMessage(data: AgentMessagePayload): void {
        update((state) => {
            const nextState: AgentState = { ...state };

            // Diffs drive their viewer and never join the chat feed.
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

            // This persistent constraint is a banner, not an event to scroll away.
            if (data.type === 'sandbox_unavailable') {
                return {
                    ...nextState,
                    sandboxUnavailable: { stage: data.stage, reason: data.reason }
                };
            }

            // Context usage is a replaceable gauge, not a feed entry per request.
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

            // Both branches can update an existing slot instead of always appending.
            if (data.type === 'system') return applySystem(nextState, data);
            if (data.type === 'thought') return applyThought(nextState, data);

            if (
                data.type === 'observation' &&
                data.status === 'success' &&
                STEP_COMPLETION_TOOLS.has(data.tool_name)
            ) {
                // Step summaries preserve the outcome; individual tool calls would
                // only bury the next step's progress.
                return {
                    ...nextState,
                    messages: state.messages.filter(
                        (message) => message.type !== 'action' && message.type !== 'observation'
                    )
                };
            }

            if (data.type === 'question') {
                // The feed records the question while this state drives the UI.
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
                        // A rejected review is a user decision: rework or merge anyway.
                        nextState.requiresAction = true;
                        nextState.actionType = 'review_rejected';
                    }
                    break;
                case 'done':
                    newMessage.content = data.summary;
                    if (!data.branch_name) {
                        // A task that changed nothing has no branch and no merge choice.
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
                    // A warning leaves the session running; it is not a failure.
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
                    // The refusal is authoritative, but only a sibling reply proves
                    // it is not a stale socket still being reaped. Do not couple this
                    // asynchronous lookup to subsequent events.
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

    function receive(raw: unknown): void {
        // The raw WebSocket payload is a trust boundary: malformed JSON and schema
        // mismatches stay visible in the feed and console rather than disappearing.
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

    return { nextMessageId, receive };
}
