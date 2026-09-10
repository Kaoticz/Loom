import { browser } from '$app/environment';
import { z } from 'zod';

const SESSION_CHANNEL = 'loom_session';

/**
 * What one tab says to another about who holds the session.
 *
 * BroadcastChannel has no sender identity and reaches other channels from this
 * tab too. `from` prevents a tab from answering its own query and mistaking that
 * reply for a sibling holding the session.
 */
const presenceMessageSchema = z.object({
    type: z.enum(['who_has_session', 'i_have_session']),
    from: z.string()
});

type PresenceMessage = z.infer<typeof presenceMessageSchema>;

export interface SessionPresence {
    announce(): void;
    stop(): void;
    siblingHoldsSession(timeoutMs?: number): Promise<boolean>;
}

/**
 * Create the per-tab cross-tab session-presence protocol.
 *
 * @param holdsSession - Reads whether this store instance currently holds the session.
 * @returns The presence announcer, stopper, and sibling query functions.
 */
export function createSessionPresence(holdsSession: () => boolean): SessionPresence {
    let presence: BroadcastChannel | null = null;
    /** This store instance's identity for its entire lifetime. */
    const instanceId: string = createInstanceId();

    function readPresence(data: unknown): PresenceMessage | null {
        const result = presenceMessageSchema.safeParse(data);
        if (!result.success || result.data.from === instanceId) return null;
        return result.data;
    }

    function announce(): void {
        if (!browser || typeof BroadcastChannel === 'undefined') return;

        const channel = new BroadcastChannel(SESSION_CHANNEL);
        channel.onmessage = (event: MessageEvent<unknown>) => {
            const message = readPresence(event.data);
            if (message === null || message.type !== 'who_has_session') return;
            // Only a tab that actually holds a session may answer.
            if (!holdsSession()) return;

            const reply: PresenceMessage = { type: 'i_have_session', from: instanceId };
            channel.postMessage(reply);
        };
        presence = channel;
    }

    function stop(): void {
        // A channel must not outlive the session it would otherwise answer for.
        presence?.close();
        presence = null;
    }

    function siblingHoldsSession(timeoutMs = 300): Promise<boolean> {
        if (!browser || typeof BroadcastChannel === 'undefined') return Promise.resolve(false);

        return new Promise((resolve) => {
            const channel = new BroadcastChannel(SESSION_CHANNEL);
            let settled = false;

            const finish = (answer: boolean): void => {
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
            // Post before scheduling the 300ms silence window; that ordering is
            // the protocol's complete query/close handoff.
            channel.postMessage(query);
            setTimeout(() => finish(false), timeoutMs);
        });
    }

    return { announce, stop, siblingHoldsSession };
}

function createInstanceId(): string {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
        return crypto.randomUUID();
    }
    return `loom-${Math.random().toString(36).slice(2)}-${Date.now().toString(36)}`;
}
