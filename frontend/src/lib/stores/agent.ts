import { writable, get } from 'svelte/store';
import { browser } from '$app/environment';

export type MessageType =
    | 'prompt'
    | 'thought'
    | 'action'
    | 'observation'
    | 'plan'
    | 'review'
    | 'commit'
    | 'done'
    | 'error';

export interface Message {
    type: MessageType;
    content: string;
    timestamp: Date;
    metadata?: any;
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
    | 'failed';

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

export interface AgentConfig {
    coderModel: string;
    reviewerModel: string;
    embeddingModel: string;
    repoPath: string;
    dockerfile: string;
    secrets: { key: string; value: string }[];
    theme: Theme;
}

export interface AgentState {
    messages: Message[];
    status: TaskStatus;
    fileDiffs: Record<string, FileDiff>;
    files: string[];
    selectedFile: string | null;
    requiresAction: boolean;
    actionType: ActionType;
    connection: ConnectionState;
    blocked: BlockedReason;
    config: AgentConfig;
}

const defaultConfig: AgentConfig = {
    coderModel: 'ollama:llama3',
    reviewerModel: 'ollama:llama3',
    embeddingModel: 'nomic-embed-text',
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
    files: [],
    selectedFile: null,
    requiresAction: false,
    actionType: null,
    connection: 'disconnected',
    blocked: null,
    // Secrets are deliberately never persisted, so merge the saved config over
    // the defaults rather than replacing them.
    config: savedConfig ? { ...defaultConfig, ...JSON.parse(savedConfig), secrets: [] } : defaultConfig
};

function createAgentStore() {
    const { subscribe, set, update } = writable<AgentState>(initialState);
    let socket: WebSocket | null = null;
    let presence: BroadcastChannel | null = null;
    // Remembered so a retry does not need the caller to supply it again.
    let lastUrl = '';
    let reconnectTimeout: number;

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
        update(state => ({ ...state, connection: 'connecting' }));
        socket = new WebSocket(url);

        socket.onopen = () => {
            update((state) => ({ ...state, connection: 'connected', blocked: null }));
            presence?.close();
            presence = announcePresence();
        };

        socket.onmessage = (event) => {
            handleMessage(JSON.parse(event.data));
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
            update(state => ({ ...state, connection: 'disconnected' }));
        };
    }

    /** Status implied by a thought, or null when it says nothing about progress. */
    function statusFromThought(content: string): TaskStatus | null {
        if (content.includes('Indexing')) return 'indexing';
        if (content.includes('sandbox')) return 'indexing';
        if (content.includes('Generating plan')) return 'planning';
        if (content.includes('Executing plan')) return 'executing';
        if (content.includes('Starting review')) return 'reviewing';
        if (content.includes('Reworking')) return 're_working';
        if (content.includes('Merging')) return 'merging';
        return null;
    }

    /** One-line summary of a tool call, for the chat feed. */
    function describeAction(data: any): string {
        const args = data.arguments ?? {};
        const detail = Object.entries(args)
            .map(([key, value]) => `${key}=${String(value)}`)
            .join(', ');
        return detail ? `${data.tool_name}(${detail})` : `${data.tool_name}()`;
    }

    function handleMessage(data: any) {
        update(state => {
            let nextState: AgentState = { ...state };

            // Diffs drive the diff viewer rather than the chat feed, so they are
            // applied to state without also appearing as a message.
            if (data.type === 'diff') {
                if (data.file_path) {
                    nextState.fileDiffs = {
                        ...nextState.fileDiffs,
                        [data.file_path]: {
                            original: data.original ?? '',
                            modified: data.modified ?? '',
                            truncated: data.truncated ?? false
                        }
                    };
                    if (!nextState.files.includes(data.file_path)) {
                        nextState.files = [...nextState.files, data.file_path];
                    }
                    if (!nextState.selectedFile) {
                        nextState.selectedFile = data.file_path;
                    }
                }
                return nextState;
            }

            const newMessage: Message = {
                type: data.type,
                content: data.content ?? '',
                timestamp: new Date(),
                metadata: data
            };

            switch (data.type) {
                case 'thought': {
                    const implied = statusFromThought(data.content ?? '');
                    if (implied) nextState.status = implied;
                    break;
                }
                case 'action':
                    newMessage.content = describeAction(data);
                    break;
                case 'observation':
                    newMessage.content = data.result ?? '';
                    break;
                case 'commit':
                    newMessage.content = `${data.message} (${String(data.commit_hash).slice(0, 7)})`;
                    break;
                case 'plan':
                    newMessage.content = (data.steps ?? [])
                        .map((step: string, index: number) => `${index + 1}. ${step}`)
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
                    newMessage.content = data.comments ?? '';
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
                    newMessage.content = data.summary ?? '';
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
                    nextState.status = 'failed';
                    nextState.requiresAction = false;
                    nextState.actionType = null;
                    break;
                case 'session_busy':
                    // The refusal is authoritative, but only a sibling's reply
                    // proves a real tab holds it: a closed tab's socket may not
                    // have been reaped yet, which is worth retrying.
                    newMessage.content = data.content ?? '';
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
            update(state => {
                const repoChanged = config.repoPath !== undefined && config.repoPath !== state.config.repoPath;
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
                        files: [],
                        selectedFile: null,
                        requiresAction: false,
                        actionType: null
                    };
                }
                
                return { ...state, config: nextConfig };
            });
        },
        sendPrompt: (prompt: string) => {
            const state = get({ subscribe });
            update(state => ({
                ...state,
                messages: [...state.messages, { type: 'prompt', content: prompt, timestamp: new Date() }],
                status: 'indexing'
            }));
            socket?.send(JSON.stringify({
                prompt,
                coder_model: state.config.coderModel,
                reviewer_model: state.config.reviewerModel,
                embedding_model: state.config.embeddingModel,
                repo_path: state.config.repoPath,
                dockerfile: state.config.dockerfile,
                secrets: state.config.secrets
            }));
        },
        approve: (approved: boolean, reason: string = '') => {
            update(state => ({ ...state, requiresAction: false, actionType: null }));
            socket?.send(JSON.stringify({ approved, reason }));
        },
        sendAction: (action: 'approve' | 'reject' | 'merge' | 'abort' | 'continue' | 'auto_approve', reason: string = '') => {
            update(state => ({ ...state, requiresAction: false, actionType: null }));
            socket?.send(JSON.stringify({ action, reason }));
        },
        retryConnection: () => {
            if (!lastUrl) return;
            update((state) => ({ ...state, blocked: null, messages: [] }));
            connect(lastUrl);
        },
        selectFile: (file: string) => {
            update(state => ({ ...state, selectedFile: file }));
        },
        disconnect: () => {
            clearTimeout(reconnectTimeout);
            socket?.close();
        },
        connect
    };
}

export const agentStore = createAgentStore();
