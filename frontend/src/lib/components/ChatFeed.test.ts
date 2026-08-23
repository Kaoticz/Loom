import { describe, expect, it } from 'vitest';
import { render } from '@testing-library/svelte';
import { tick } from 'svelte';
import ChatFeed from './ChatFeed.svelte';
import type { Message } from '$lib/stores/agent';

/**
 * Wiring tests for the feed's scroll following.
 *
 * jsdom performs no layout, so every scroll measurement it reports is zero and
 * the container's metrics have to be stubbed. Crucially `scrollHeight` is
 * derived from how many messages are actually in the DOM rather than fixed: it
 * has to grow as a message is added, because the difficulty this requirement
 * turns on is that measuring after the new message has rendered gives the wrong
 * answer. A fixed height would leave that mistake invisible.
 *
 * The stub also models `scroll-behavior: smooth`, which the feed sets. An
 * assignment to `scrollTop` on such an element does not move it: it starts an
 * animation, and the property reports an intermediate position until that
 * animation finishes. Only `scrollTo` with `behavior: 'instant'` moves it at
 * once. Modelling this is what lets these tests see a defect that would
 * otherwise need a real browser — a stub that applied every assignment
 * immediately reported success while the real feed sat thousands of pixels from
 * the bottom.
 *
 * Whether the arithmetic is right is settled in `scrolling.test.ts`.
 */

const VIEWPORT_HEIGHT = 400;
const MESSAGE_HEIGHT = 200;
const INITIAL_MESSAGES = 5;

/**
 * How far an in-flight smooth scroll has travelled when the next message lands.
 *
 * The exact figure does not matter, only that it is short of the target, which
 * is what a reader sees when messages arrive faster than the animation runs.
 */
const SMOOTH_STEP_PX = 40;

/** Where `scrollTop` sits when a feed of this many messages is fully scrolled. */
function bottomFor(messageCount: number): number {
    return messageCount * MESSAGE_HEIGHT - VIEWPORT_HEIGHT;
}

let issued = 0;

/** Build a feed message with a fresh identity, as the store would. */
function message(content: string): Message {
    issued += 1;
    return { id: issued, type: 'thought', content, timestamp: new Date() };
}

/**
 * Render a feed whose scroll position is controllable.
 *
 * @param scrollTop Where the reader has scrolled to.
 * @returns The rendered component and a live view of its scroll position.
 */
function renderFeed(scrollTop: number) {
    const messages: Message[] = Array.from({ length: INITIAL_MESSAGES }, (_, index) =>
        message(`message ${index}`)
    );
    const rendered = render(ChatFeed, {
        messages: [...messages],
        onapprove: () => {},
        onreject: () => {},
        onretry: () => {},
        onanswer: () => {},
        onfreereply: () => {}
    });

    const feed = rendered.getByTestId('chat-feed');
    let position = scrollTop;

    Object.defineProperty(feed, 'clientHeight', {
        value: VIEWPORT_HEIGHT,
        configurable: true
    });
    // Derived from what is currently rendered, so it grows with the content at
    // exactly the moment the DOM does.
    Object.defineProperty(feed, 'scrollHeight', {
        get: () => feed.children.length * MESSAGE_HEIGHT,
        configurable: true
    });
    Object.defineProperty(feed, 'scrollTop', {
        get: () => position,
        // An assignment on a smoothly scrolling element begins an animation
        // rather than moving it, so the position only creeps toward the target.
        // Real browsers also clamp to `scrollHeight - clientHeight`; that is
        // left out because it is not what these tests turn on, and modelling it
        // would change what every assertion below has to expect.
        set: (next: number) => {
            position = Math.min(next, position + SMOOTH_STEP_PX);
        },
        configurable: true
    });
    // jsdom implements no scrolling methods at all, so this has to be supplied
    // alongside the properties above.
    feed.scrollTo = ((options: ScrollToOptions) => {
        if (options.behavior === 'instant') {
            position = options.top ?? position;
            return;
        }
        position = Math.min(options.top ?? position, position + SMOOTH_STEP_PX);
    }) as HTMLElement['scrollTo'];

    /** Deliver another message, as the store would. */
    async function arrive(content: string): Promise<void> {
        messages.push(message(content));
        // A new array identity, because props are compared by reference.
        await rendered.rerender({
            messages: [...messages],
            onapprove: () => {},
            onreject: () => {}
        });
        await tick();
    }

    return {
        rendered,
        arrive,
        scrollTo: (next: number) => {
            position = next;
        },
        position: () => position,
        contentHeight: () => feed.scrollHeight
    };
}

describe('ChatFeed scroll following', () => {
    it('follows a new message when the reader is at the bottom', async () => {
        const feed = renderFeed(bottomFor(INITIAL_MESSAGES));

        await feed.arrive('newest');

        // The taller content, not the height measured before it arrived.
        expect(feed.contentHeight()).toBe((INITIAL_MESSAGES + 1) * MESSAGE_HEIGHT);
        expect(feed.position()).toBe(feed.contentHeight());
    });

    it('follows when the reader is a few pixels short of the bottom', async () => {
        // Requirement 4.1: slightly short still counts as at the bottom.
        const feed = renderFeed(bottomFor(INITIAL_MESSAGES) - 10);

        await feed.arrive('newest');

        expect(feed.position()).toBe(feed.contentHeight());
    });

    it('keeps following across a run of messages once at the bottom', async () => {
        const feed = renderFeed(bottomFor(INITIAL_MESSAGES));

        await feed.arrive('second');
        await feed.arrive('third');

        expect(feed.position()).toBe(feed.contentHeight());
    });

    it('keeps following when messages arrive faster than a smooth scroll runs', async () => {
        // The defect this guards against: the feed scrolls smoothly, so an
        // assignment to `scrollTop` leaves the position short of the bottom
        // while the animation runs. Measuring that intermediate position before
        // the next message reads as "the reader has scrolled up", and following
        // stops for the rest of the session. Arrivals here are not settled
        // between, which is what a streaming agent actually produces.
        const feed = renderFeed(bottomFor(INITIAL_MESSAGES));

        await feed.arrive('second');
        await feed.arrive('third');
        await feed.arrive('fourth');
        await feed.arrive('fifth');

        expect(feed.position()).toBe(feed.contentHeight());
    });

    it('leaves the viewport alone when the reader has scrolled up', async () => {
        // Requirement 4.2: 500px from the bottom, well outside the tolerance.
        const feed = renderFeed(100);

        await feed.arrive('second');

        expect(feed.position()).toBe(100);
    });

    it('does not move the viewport across several arriving messages', async () => {
        const feed = renderFeed(100);

        await feed.arrive('second');
        await feed.arrive('third');
        await feed.arrive('fourth');

        expect(feed.position()).toBe(100);
    });

    it('resumes following once the reader returns to the bottom', async () => {
        // Requirement 4.3, and 4.4: the decision is re-measured per message
        // rather than latched, so returning to the bottom is enough.
        const feed = renderFeed(100);

        await feed.arrive('second');
        expect(feed.position()).toBe(100);

        feed.scrollTo(bottomFor(INITIAL_MESSAGES + 1));
        await feed.arrive('third');

        expect(feed.position()).toBe(feed.contentHeight());
    });

    it('renders every message it is given', () => {
        const feed = renderFeed(bottomFor(INITIAL_MESSAGES));

        expect(feed.rendered.getByText('message 0')).toBeInTheDocument();
        expect(feed.rendered.getByText('message 4')).toBeInTheDocument();
    });
});
