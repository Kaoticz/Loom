import { writable, get } from 'svelte/store';

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

export interface AgentState {
    messages: Message[];
    status: TaskStatus;
    fileDiffs: Record<string, FileDiff>;
    files: string[];
    selectedFile: string | null;
    requiresAction: boolean;
    actionType: 'approve_plan' | 'approve_review' | null;
}

const initialState: AgentState = {
    messages: [],
    status: 'idle',
    fileDiffs: {},
    files: [],
    selectedFile: null,
    requiresAction: false,
    actionType: null
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
        sendPrompt: (prompt: string) => {
            update(state => ({
                ...state,
                messages: [...state.messages, { type: 'prompt', content: prompt, timestamp: new Date() }],
                status: 'indexing'
            }));
            socket?.send(JSON.stringify({ prompt }));
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
