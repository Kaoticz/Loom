import { describe, expect, it } from 'vitest';
import { agentMessageSchema } from './schemas';
import {
    MARKDOWN_TYPES,
    actionLabels,
    connectionStyles,
    formatMessageTime,
    canSendPrompt,
    promptPlaceholder,
    isWorking,
    typeStyles
} from './chatPresentation';

describe('chat presentation', () => {
    it('styles every message type that reaches the feed', () => {
        // Types the store consumes into state rather than rendering as a
        // bubble: `diff` feeds the diff viewer, `sandbox_unavailable` the banner.
        const CONSUMED_WITHOUT_DISPLAY = ['diff', 'sandbox_unavailable', 'context_usage'];

        // The union's own discriminator values, so a new message type added to
        // the protocol fails here rather than rendering unstyled.
        const delivered = agentMessageSchema.options
            .map((option) => option.shape.type.value as string)
            .filter((type) => !CONSUMED_WITHOUT_DISPLAY.includes(type));

        for (const type of delivered) {
            expect(typeStyles, `no style for '${type}'`).toHaveProperty(type);
        }
    });

    it('styles the locally generated prompt type too', () => {
        expect(typeStyles).toHaveProperty('prompt');
    });

    it('distinguishes system from thought', () => {
        expect(typeStyles.system).not.toBe(typeStyles.thought);
    });

    it('distinguishes a busy session from a failure', () => {
        expect(typeStyles.session_busy).not.toBe(typeStyles.error);
    });

    it('distinguishes compaction from system announcements and warnings', () => {
        expect(typeStyles.compaction).not.toBe(typeStyles.system);
        expect(typeStyles.compaction).not.toBe(typeStyles.warning);
        expect(MARKDOWN_TYPES).not.toContain('compaction');
    });

    it('renders only prose types as markdown', () => {
        // Tool output, commits and system announcements contain file paths and
        // command output that markdown would mangle.
        expect(MARKDOWN_TYPES).not.toContain('action');
        expect(MARKDOWN_TYPES).not.toContain('observation');
        expect(MARKDOWN_TYPES).not.toContain('commit');
        expect(MARKDOWN_TYPES).not.toContain('system');
    });

    it('covers every connection state', () => {
        expect(Object.keys(connectionStyles).sort()).toEqual([
            'connected',
            'connecting',
            'disconnected'
        ]);
    });

    it('gives every decision a full set of labels', () => {
        for (const [action, labels] of Object.entries(actionLabels)) {
            expect(labels.title, action).toBeTruthy();
            expect(labels.approve, action).toBeTruthy();
            expect(labels.reject, action).toBeTruthy();
        }
    });
});

describe('formatMessageTime', () => {
    const at = new Date(2026, 2, 4, 14, 32, 7);

    it('shows hours and minutes, not seconds', () => {
        expect(formatMessageTime(at, 'en-GB')).toBe('14:32');
    });

    it('pads a single digit hour', () => {
        expect(formatMessageTime(new Date(2026, 2, 4, 9, 5), 'en-GB')).toBe('09:05');
    });

    it('follows the locale it is given', () => {
        expect(formatMessageTime(at, 'en-US')).toMatch(/2:32\s?PM/i);
    });

    it('crosses midnight without wrapping to 24', () => {
        expect(formatMessageTime(new Date(2026, 2, 4, 0, 1), 'en-GB')).toBe('00:01');
    });
});

describe('sandboxStageLabels', () => {
    it('labels every stage the backend can report', async () => {
        const { sandboxStageSchema } = await import('./schemas');
        const { sandboxStageLabels } = await import('./chatPresentation');

        for (const stage of sandboxStageSchema.options) {
            expect(sandboxStageLabels, `no label for '${stage}'`).toHaveProperty(stage);
        }
    });

    it('gives each stage its own wording', async () => {
        const { sandboxStageLabels } = await import('./chatPresentation');
        const wordings = Object.values(sandboxStageLabels);

        // Requirement 7.7: the causes must actually read differently.
        expect(new Set(wordings).size).toBe(wordings.length);
    });
});

describe('isWorking', () => {
    it.each([
        'indexing',
        // Both phases the read-only route added. The agent is running under its
        // own steam in each, so a stop must be on offer and a typed prompt must
        // not be sent into a session that is not listening for one.
        'classifying',
        'planning',
        'executing',
        'answering',
        'reviewing',
        're_working',
        'merging'
    ] as const)('counts %s as working', (status) => {
        expect(isWorking(status)).toBe(true);
    });

    it.each([
        'idle',
        'waiting_approval',
        'completed',
        // The answer has been given and the session is over, so a new task can
        // be typed straight away.
        'answered',
        'merged',
        'failed'
    ] as const)('does not count %s as working', (status) => {
        // Requirement 8.2: waiting for the user is not working.
        expect(isWorking(status)).toBe(false);
    });
});

describe('isWorking after a stop', () => {
    it('does not count a cancelled session as working', () => {
        // Requirement 8.7: Stop must turn back into Send.
        expect(isWorking('cancelled')).toBe(false);
    });
});

describe('canSendPrompt', () => {
    it.each(['indexing', 'planning', 'executing', 'reviewing', 're_working', 'merging'] as const)(
        'withholds sending while %s',
        (status) => {
            // Requirement 9.2a.
            expect(canSendPrompt(status, null)).toBe(false);
        }
    );

    it('withholds sending while any decision is outstanding', () => {
        expect(canSendPrompt('waiting_approval', 'approve_plan')).toBe(false);
        expect(canSendPrompt('completed', 'approve_merge')).toBe(false);
    });

    it('allows sending when the agent is idle', () => {
        expect(canSendPrompt('idle', null)).toBe(true);
    });

    it('allows sending once a task has finished', () => {
        expect(canSendPrompt('completed', null)).toBe(true);
        expect(canSendPrompt('failed', null)).toBe(true);
        expect(canSendPrompt('cancelled', null)).toBe(true);
    });

    it('withholds sending after a merge', () => {
        expect(canSendPrompt('merged', null)).toBe(false);
    });
});

describe('promptPlaceholder', () => {
    it('invites a task when idle', () => {
        expect(promptPlaceholder('idle', null)).toMatch(/task/i);
    });

    it('invites a reason when a decision is outstanding', () => {
        // Requirement 9.3a: the box doubles as the rejection reason input.
        expect(promptPlaceholder('waiting_approval', 'approve_plan')).toMatch(/reason/i);
    });

    it('says the next task can be written while the agent works', () => {
        expect(promptPlaceholder('executing', null)).toMatch(/while the agent works/i);
    });

    it('always says something', () => {
        for (const status of ['idle', 'executing', 'completed'] as const) {
            expect(promptPlaceholder(status, null).length).toBeGreaterThan(0);
        }
    });

    it('tells the reader to reload after a merge', () => {
        expect(promptPlaceholder('merged', null)).toMatch(/reload/i);
    });
});

describe('isWorking after a stop', () => {
    it('does not count a cancelled session as working', () => {
        // Requirement 8.7: Stop must turn back into Send.
        expect(isWorking('cancelled')).toBe(false);
    });
});

describe('canSendPrompt', () => {
    it.each(['indexing', 'planning', 'executing', 'reviewing', 're_working', 'merging'] as const)(
        'withholds sending while %s',
        (status) => {
            // Requirement 9.2a: a prompt sent now has nowhere to go.
            expect(canSendPrompt(status, null)).toBe(false);
        }
    );

    it.each(['approve_plan', 'approve_merge'] as const)(
        'withholds sending while %s is outstanding',
        (action) => {
            // The session is waiting to be told approve or reject, not handed
            // something new.
            expect(canSendPrompt('waiting_approval', action)).toBe(false);
        }
    );

    it('allows sending when the agent is idle', () => {
        expect(canSendPrompt('idle', null)).toBe(true);
    });

    it('allows sending once a task has ended', () => {
        expect(canSendPrompt('completed', null)).toBe(true);
        expect(canSendPrompt('failed', null)).toBe(true);
        expect(canSendPrompt('cancelled', null)).toBe(true);
    });
});

describe('promptPlaceholder', () => {
    it('invites a task when idle', () => {
        expect(promptPlaceholder('idle', null)).toMatch(/describe a task/i);
    });

    it('invites a reason when a decision is outstanding', () => {
        // Requirement 9.6a: this box is where the rejection reason goes.
        expect(promptPlaceholder('waiting_approval', 'approve_plan')).toMatch(/reason/i);
    });

    it('explains that composing is allowed while the agent works', () => {
        expect(promptPlaceholder('executing', null)).toMatch(/while the agent works/i);
    });

    it('prefers the decision wording over the working wording', () => {
        // Both can be true at once; the outstanding decision is the more useful
        // thing to say.
        expect(promptPlaceholder('reviewing', 'approve_merge')).toMatch(/reason/i);
    });
});

describe('warnings versus errors', () => {
    it('styles a warning differently from an error', () => {
        // A warning means the run is still going. Rendering it like an error
        // tells the user their session has died when it has not.
        expect(typeStyles.warning).not.toBe(typeStyles.error);
        expect(typeStyles.warning).not.toContain('red');
    });
});
