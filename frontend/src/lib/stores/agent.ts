import { writable, get } from 'svelte/store';
import { browser } from '$app/environment';

export type MessageType = 'prompt' | 'thought' | 'action' | 'observation' | 'plan' | 'review' | 'done' | 'error';

export interface Message {
    type: MessageType;
    content: string;
    timestamp: Date;
    metadata?: any;
}

export type TaskStatus = 'idle' | 'indexing' | 'planning' | 'waiting_approval' | 'executing' | 'reviewing' | 'completed' | 'failed';

export interface FileDiff {
    original: string;
    modified: string;
}

export interface AgentConfig {
    coderModel: string;
    reviewerModel: string;
    embeddingModel: string;
    repoPath: string;
    dockerfile: string;
    secrets: { key: string; value: string }[];
    theme: 'light' | 'dark' | 'system';
}

export interface AgentState {
    messages: Message[];
    status: TaskStatus;
    fileDiffs: Record<string, FileDiff>;
    files: string[];
    selectedFile: string | null;
    requiresAction: boolean;
    actionType: 'approve_plan' | 'approve_review' | null;
    config: AgentConfig;
}

const defaultConfig: AgentConfig = {
    coderModel: 'ollama:llama3',
    reviewerModel: 'ollama:llama3',
    embeddingModel: 'llama3',
    repoPath: '',
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
    config: savedConfig ? JSON.parse(savedConfig) : defaultConfig
};

function createAgentStore() {
    const { subscribe, set, update } = writable<AgentState>(initialState);
    let socket: WebSocket | null = null;
    let reconnectTimeout: number;

    function connect(url: string) {
        socket = new WebSocket(url);

        socket.onopen = () => {
            console.log('WebSocket connected');
        };

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleMessage(data);
        };

        socket.onclose = () => {
            console.log('WebSocket disconnected');
            reconnectTimeout = window.setTimeout(() => connect(url), 3000);
        };

        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    function handleMessage(data: any) {
        update(state => {
            const newMessage: Message = {
                type: data.type,
                content: data.content,
                timestamp: new Date(),
                metadata: data.metadata
            };

            let nextState = { ...state, messages: [...state.messages, newMessage] };

            switch (data.type) {
                case 'thought':
                    if (data.content.includes('Indexing')) nextState.status = 'indexing';
                    else if (data.content.includes('Generating plan')) nextState.status = 'planning';
                    else if (data.content.includes('Executing plan')) nextState.status = 'executing';
                    else if (data.content.includes('Starting review')) nextState.status = 'reviewing';
                    break;
                case 'plan':
                    nextState.status = 'waiting_approval';
                    nextState.requiresAction = true;
                    nextState.actionType = 'approve_plan';
                    break;
                case 'review':
                    nextState.status = 'reviewing';
                    nextState.requiresAction = !data.approved;
                    nextState.actionType = data.approved ? null : 'approve_review';
                    break;
                case 'done':
                    nextState.status = 'completed';
                    break;
                case 'error':
                    nextState.status = 'failed';
                    break;
                case 'diff':
                    // Assume data.content is the new content for data.file
                    if (data.file) {
                        const fileDiff = nextState.fileDiffs[data.file] || { original: '', modified: '' };
                        nextState.fileDiffs[data.file] = {
                            ...fileDiff,
                            modified: data.content
                        };
                        if (!nextState.files.includes(data.file)) {
                            nextState.files = [...nextState.files, data.file];
                        }
                        if (!nextState.selectedFile) {
                            nextState.selectedFile = data.file;
                        }
                    }
                    break;
            }

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
                    localStorage.setItem('loom_config', JSON.stringify(nextConfig));
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
                repo_path: state.config.repoPath
            }));
        },
        approve: (approved: boolean) => {
            update(state => ({ ...state, requiresAction: false, actionType: null }));
            socket?.send(JSON.stringify({ approved }));
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
