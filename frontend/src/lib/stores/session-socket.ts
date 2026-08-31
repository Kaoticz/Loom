/**
 * The named WebSocket ready states this factory needs.
 *
 * Tests replace the global with a stand-in without static constants, so reading
 * `WebSocket.CONNECTING` or `WebSocket.OPEN` would compare against `undefined`.
 */
const SocketReadyState = {
    connecting: 0,
    open: 1
} as const;

export interface SessionSocket {
    connect(url: string): void;
    teardown(): void;
    retry(): void;
    hasUrl(): boolean;
    send(data: string): void;
}

interface SessionSocketOptions {
    receive(raw: unknown): void;
    setConnection(connection: 'connecting' | 'connected' | 'disconnected'): void;
    beforeConnect(): void;
    connected(): void;
    disconnected(): void;
    closed(scheduleReconnect: () => void): void;
}

/**
 * Create the WebSocket lifecycle manager for one agent session.
 *
 * @param options - State and lifecycle callbacks supplied by the agent store.
 * @returns The connection, teardown, retry, and send operations for the session.
 */
export function createSessionSocket(options: SessionSocketOptions): SessionSocket {
    let socket: WebSocket | null = null;
    // Remembered so retrying does not need the caller to supply the URL again.
    let lastUrl = '';
    let reconnectTimeout: number | undefined;

    function sessionIsLive(): boolean {
        if (socket === null) return false;
        return (
            socket.readyState === SocketReadyState.connecting ||
            socket.readyState === SocketReadyState.open
        );
    }

    function teardown(): void {
        clearTimeout(reconnectTimeout);
        options.disconnected();
        const closing = socket;
        // Clear first so handlers on the closing socket know they were superseded.
        socket = null;
        closing?.close();
    }

    function connect(url: string): void {
        // A second connection would be refused as busy by a backend already
        // talking to the first, so establishing the layout connection is idempotent.
        if (sessionIsLive()) return;

        lastUrl = url;
        options.beforeConnect();
        options.setConnection('connecting');

        const opened = new WebSocket(url);
        socket = opened;

        opened.onopen = (): void => {
            // A replacement socket alone may report its session state.
            if (socket !== opened) return;
            options.setConnection('connected');
            options.connected();
        };

        opened.onmessage = (event: MessageEvent<unknown>): void => {
            // Never let a superseded socket feed a newer session.
            if (socket !== opened) return;
            options.receive(event.data);
        };

        opened.onclose = (): void => {
            // A superseded socket must not report disconnection or schedule a retry.
            if (socket !== opened) return;

            // The session is over, so this tab no longer answers presence queries.
            options.disconnected();
            options.closed(() => {
                // A refusal is a decision, not a fault: the store declines this
                // callback when blocked rather than looping against the backend.
                reconnectTimeout = window.setTimeout(() => connect(url), 3000);
            });
        };

        opened.onerror = (error: Event): void => {
            if (socket !== opened) return;
            console.error('WebSocket error:', error);
            options.setConnection('disconnected');
        };
    }

    function retry(): void {
        if (!lastUrl) return;
        teardown();
        connect(lastUrl);
    }

    function hasUrl(): boolean {
        return Boolean(lastUrl);
    }

    function send(data: string): void {
        // Preserve direct outbound sends: callers intentionally do not gate on
        // readyState, matching the existing action facade's behavior.
        socket?.send(data);
    }

    return { connect, teardown, retry, hasUrl, send };
}
