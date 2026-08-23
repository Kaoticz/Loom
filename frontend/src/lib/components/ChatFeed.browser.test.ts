// @vitest-environment node
//
// The real DOM here is the browser's, not jsdom's. This file also starts a Vite
// server in-process, and esbuild cannot run inside jsdom: its Uint8Array comes
// from another realm, which esbuild detects and refuses to work with.
import { describe, expect, it } from 'vitest';
import { BROWSER_BIN_VAR, browserUnavailableReason, withBrowserFeed } from '$lib/testing/browser';
import { SCROLL_BOTTOM_TOLERANCE_PX } from '$lib/scrolling';

/**
 * The parts of the feed's scrolling that only a real browser can settle.
 *
 * `ChatFeed.test.ts` covers the decision wiring, and covers it well, but it
 * cannot observe layout: jsdom reports every scroll measurement as zero, so a
 * test there cannot tell whether the feed is a scroll container at all, nor
 * whether an animation left the reader short of the bottom. Both of those have
 * gone wrong in this component, so they are checked here against a browser that
 * has actually laid the page out.
 *
 * These skip unless a browser is available — see `$lib/testing/browser`.
 */

const skipReason = await browserUnavailableReason();

describe.skipIf(skipReason !== null)(
    `ChatFeed scrolling in a real browser${skipReason ? ` (skipped: ${skipReason})` : ''}`,
    () => {
        /** Fill the feed past its viewport, as a working session quickly does. */
        async function fill(
            send: (payload: Record<string, unknown>) => Promise<void>,
            count: number,
            from = 1
        ) {
            for (let index = from; index < from + count; index += 1) {
                await send({
                    type: 'observation',
                    tool_name: 'read_file',
                    result:
                        `Observation ${index}: ` +
                        'the quick brown fox jumps over the lazy dog. '.repeat(12)
                });
            }
        }

        it('overflows rather than growing, and never scrolls the page', async () => {
            const metrics = await withBrowserFeed(async ({ send, metrics }) => {
                await fill(send, 20);
                return metrics();
            });

            // If the feed grew instead of overflowing, its content would be
            // pushed past the viewport and out of reach entirely.
            expect(metrics.isScrollContainer).toBe(true);
            expect(metrics.scrollHeight).toBeGreaterThan(metrics.clientHeight);
            expect(metrics.pageScrolls).toBe(false);
        }, 120_000);

        it('follows a run of messages arriving faster than a scroll animation', async () => {
            // The defect: the feed scrolls smoothly, so assigning `scrollTop`
            // begins an animation and reports an intermediate position while it
            // runs. Measuring that before the next message reads as "scrolled
            // up", and following stops for the rest of the session.
            const metrics = await withBrowserFeed(async ({ send, metrics }) => {
                await fill(send, 20);
                return metrics();
            });

            expect(metrics.distanceFromBottom).toBeLessThanOrEqual(SCROLL_BOTTOM_TOLERANCE_PX);
        }, 120_000);

        it('does not move the viewport while the reader is scrolled up', async () => {
            const { parked, afterwards } = await withBrowserFeed(
                async ({ send, metrics, scrollFeedTo }) => {
                    await fill(send, 20);
                    await scrollFeedTo(500);
                    const parked = (await metrics()).scrollTop;
                    await fill(send, 8, 21);
                    return { parked, afterwards: (await metrics()).scrollTop };
                }
            );

            expect(parked).toBe(500);
            // Not "close to": the viewport must not move at all, not even briefly.
            expect(afterwards).toBe(parked);
        }, 120_000);

        it('resumes following once the reader returns to the bottom', async () => {
            const metrics = await withBrowserFeed(async ({ send, metrics, scrollFeedTo }) => {
                await fill(send, 20);
                await scrollFeedTo(500);
                await fill(send, 5, 21);
                await scrollFeedTo('bottom');
                await fill(send, 5, 26);
                return metrics();
            });

            expect(metrics.distanceFromBottom).toBeLessThanOrEqual(SCROLL_BOTTOM_TOLERANCE_PX);
        }, 120_000);

        it('does not move the reader when a plan step is expanded', async () => {
            // Phase 1 established that following is driven by messages arriving.
            // Expanding a step changes the feed's height without adding a
            // message, so the reader's position has to survive it — otherwise
            // the feed jumps under someone who is reading. The argument was
            // sound and never measured, which is what this measures.
            const { before, after } = await withBrowserFeed(
                async ({ send, metrics, scrollFeedTo, expandFirstPlanStep }) => {
                    await fill(send, 20);
                    await send({
                        type: 'plan',
                        format: 'structured',
                        steps: [
                            { title: 'First step', description: 'A long description. '.repeat(40) },
                            { title: 'Second step', description: 'Another one. '.repeat(40) }
                        ],
                        requires_approval: false
                    });

                    await scrollFeedTo(400);
                    const before = (await metrics()).scrollTop;
                    await expandFirstPlanStep();
                    return { before, after: (await metrics()).scrollTop };
                }
            );

            expect(before).toBe(400);
            // Not "close to": the viewport must not move at all.
            expect(after).toBe(before);
        }, 120_000);

        it('scrolls to the end of a message taller than the viewport', async () => {
            // "At the bottom" has to mean the end of the content, not the top of
            // the last message, or a long message leaves the newest text unread.
            const metrics = await withBrowserFeed(async ({ send, metrics }) => {
                await fill(send, 10);
                await send({
                    type: 'observation',
                    tool_name: 'read_file',
                    result: 'A very long observation: ' + 'lorem ipsum dolor sit amet. '.repeat(400)
                });
                return metrics();
            });

            expect(metrics.distanceFromBottom).toBeLessThanOrEqual(SCROLL_BOTTOM_TOLERANCE_PX);
        }, 120_000);
    }
);

describe('browser test availability', () => {
    it('names the variable that enables the browser tests', () => {
        // A skipped suite that does not say how to unskip it tends to stay
        // skipped, so the reason is asserted to mention the variable.
        expect(BROWSER_BIN_VAR).toBe('LOOM_BROWSER_BIN');
        expect(skipReason === null || skipReason.includes(BROWSER_BIN_VAR)).toBe(true);
    });
});
