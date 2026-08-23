/**
 * Support for tests that need a real browser.
 *
 * Layout is the one thing jsdom cannot provide, so a handful of behaviours can
 * only be checked against a browser that has actually laid the page out. Those
 * tests are optional: they run when a browser is available and skip with a
 * stated reason when it is not, in the same way the backend's suites skip
 * unless `TEST_DATABASE_URL` and `AST_GREP_BIN` are set.
 *
 * Two things have to be present:
 *
 * - `playwright-core`, declared as an optional dependency. It is the driver
 *   only: unlike `playwright` it has no install hook and downloads no browsers.
 * - `LOOM_BROWSER_BIN`, an absolute path to a Chromium-family executable. The
 *   binary is deliberately not a dependency — it is large, and a machine that
 *   wants these tests generally has one already.
 *
 * `playwright-core` is imported through a specifier Vite cannot see, because
 * Vite resolves imports when it transforms this file: a static specifier for a
 * package that is not installed fails the entire test file before any skip
 * check can run. For the same reason nothing here imports its *types* either,
 * so `npm run check` stays clean without it. The narrow interfaces below are
 * the whole surface these tests depend on.
 *
 * `process` is declared rather than taken from `@types/node`, which this
 * project does not depend on. Widening the type surface of the whole codebase
 * to support an optional test would be a poor trade.
 */
declare const process: { env: Record<string, string | undefined> };

/** Where to find a browser, if these tests are wanted. */
export const BROWSER_BIN_VAR = 'LOOM_BROWSER_BIN';

interface PlaywrightRoute {
    fulfill(options: { status: number; contentType: string; body: string }): Promise<void>;
}

interface PlaywrightSocket {
    send(data: string): void;
}

interface PlaywrightPage {
    route(url: string, handler: (route: PlaywrightRoute) => unknown): Promise<void>;
    routeWebSocket(
        matcher: () => boolean,
        handler: (socket: PlaywrightSocket) => void
    ): Promise<void>;
    goto(url: string, options?: { waitUntil?: 'networkidle' }): Promise<unknown>;
    waitForSelector(selector: string): Promise<unknown>;
    waitForTimeout(milliseconds: number): Promise<void>;
    evaluate<Result>(fn: () => Result): Promise<Result>;
    evaluate<Result, Argument>(fn: (arg: Argument) => Result, arg: Argument): Promise<Result>;
}

interface PlaywrightBrowser {
    newPage(options?: { viewport: { width: number; height: number } }): Promise<PlaywrightPage>;
    close(): Promise<void>;
}

interface PlaywrightModule {
    chromium: {
        launch(options: { executablePath: string; args?: string[] }): Promise<PlaywrightBrowser>;
    };
}

// Held in a variable so the specifier is opaque to Vite's static analysis; the
// comment stops it trying to resolve the import at transform time.
const PLAYWRIGHT_SPECIFIER = 'playwright-core';

/**
 * Load the driver.
 *
 * @returns The module, or null when it is not installed.
 */
async function loadPlaywright(): Promise<PlaywrightModule | null> {
    try {
        return (await import(/* @vite-ignore */ PLAYWRIGHT_SPECIFIER)) as PlaywrightModule;
    } catch {
        return null;
    }
}

/**
 * Whether anything stands in the way of running the browser tests.
 *
 * @returns The reason to skip, or null when they can run.
 */
export async function browserUnavailableReason(): Promise<string | null> {
    const executablePath = process.env[BROWSER_BIN_VAR];

    if (!executablePath) {
        return `set ${BROWSER_BIN_VAR} to a Chromium executable to run these`;
    }

    if ((await loadPlaywright()) === null) {
        return `${BROWSER_BIN_VAR} is set but playwright-core is not installed; run \`npm install playwright-core\``;
    }

    // A path that does not point at a browser is deliberately not a skip. Having
    // set the variable, the reader means to run these, and a silent skip would
    // read as "passed".
    return null;
}

/** A page driving the running application, plus the means to feed it messages. */
export interface BrowserFixture {
    /** Deliver a message over the mocked agent socket, as the backend would. */
    send: (payload: Record<string, unknown>) => Promise<void>;
    /** Read the feed's scroll position, measured by the browser itself. */
    metrics: () => Promise<FeedMetrics>;
    /** Move the feed's scroll position, as a reader would. */
    scrollFeedTo: (top: number | 'bottom') => Promise<void>;
    /** Open the first collapsible plan step, as a reader would. */
    expandFirstPlanStep: () => Promise<void>;
}

/** What the browser reports about the feed, with layout actually applied. */
export interface FeedMetrics {
    scrollTop: number;
    scrollHeight: number;
    clientHeight: number;
    /** How far the bottom of the content sits below the bottom of the viewport. */
    distanceFromBottom: number;
    /** False when the element never overflows, so it cannot scroll at all. */
    isScrollContainer: boolean;
    /** True when the document itself scrolls, which it never should. */
    pageScrolls: boolean;
}

/**
 * Run a function against the real application in a real browser.
 *
 * A development server is started in-process and the agent socket is mocked, so
 * the page runs exactly as shipped without needing a backend behind it.
 *
 * @param run What to do with the page once it is ready.
 * @returns Whatever `run` returned.
 */
export async function withBrowserFeed<T>(run: (fixture: BrowserFixture) => Promise<T>): Promise<T> {
    const executablePath = process.env[BROWSER_BIN_VAR];
    const playwright = await loadPlaywright();

    if (!executablePath || playwright === null) {
        throw new Error('withBrowserFeed needs a browser; check browserUnavailableReason first');
    }

    const { createServer } = await import('vite');
    const server = await createServer({
        server: { port: 0, strictPort: false },
        mode: 'development',
        logLevel: 'error'
    });
    await server.listen();

    const address = server.resolvedUrls?.local?.[0];

    if (!address) {
        await server.close();
        throw new Error('the development server reported no address');
    }

    const browser = await playwright.chromium.launch({
        executablePath,
        args: ['--no-sandbox', '--disable-gpu']
    });

    try {
        const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });

        // The model list is irrelevant here, and its failure would paint an
        // error into the header that has nothing to do with what is measured.
        await page.route('**/api/v1/ollama/models', (route) =>
            route.fulfill({ status: 200, contentType: 'application/json', body: '[]' })
        );

        let socket: PlaywrightSocket | null = null;
        await page.routeWebSocket(
            () => true,
            (ws) => {
                socket = ws;
            }
        );

        await page.goto(address, { waitUntil: 'networkidle' });
        await page.waitForSelector('[data-testid="chat-feed"]');

        // The socket is opened from `onMount`, which can land after the page is
        // otherwise idle.
        for (let attempt = 0; attempt < 50 && socket === null; attempt += 1) {
            await page.waitForTimeout(100);
        }

        if (socket === null) {
            throw new Error('the page never opened an agent socket');
        }

        const openSocket: PlaywrightSocket = socket;

        return await run({
            send: async (payload) => {
                openSocket.send(JSON.stringify(payload));
                // Long enough for the message to be applied, short enough that a
                // run of them still outpaces a smooth scroll animation, which is
                // the condition the following behaviour has to survive.
                await page.waitForTimeout(60);
            },
            scrollFeedTo: async (top) => {
                await page.evaluate((target: number | 'bottom') => {
                    const feed = document.querySelector('[data-testid="chat-feed"]');
                    if (!feed) throw new Error('the feed is not present');
                    // Instant, because this stands in for a reader dragging the
                    // scrollbar or spinning a wheel, and `scroll-behavior:
                    // smooth` does not animate input like that — only
                    // programmatic scrolls. Animating it here would leave the
                    // position still moving when the next message arrived and
                    // make the harness, rather than the component, the thing
                    // under test.
                    feed.scrollTo({
                        top: target === 'bottom' ? feed.scrollHeight : target,
                        behavior: 'instant'
                    });
                }, top);
                await page.waitForTimeout(250);
            },
            expandFirstPlanStep: async () => {
                await page.evaluate(() => {
                    const step = document.querySelector<HTMLElement>(
                        '[data-testid="plan-steps"] button[aria-expanded="false"]'
                    );
                    if (!step) throw new Error('no collapsed plan step to open');
                    step.click();
                });
                await page.waitForTimeout(250);
            },
            metrics: async () => {
                // Any animation still in flight would otherwise be measured part
                // way through, which is the very fault under test.
                await page.waitForTimeout(600);
                return page.evaluate((): FeedMetrics => {
                    const feed = document.querySelector('[data-testid="chat-feed"]');
                    if (!feed) throw new Error('the feed is not present');
                    const doc = document.documentElement;
                    return {
                        scrollTop: Math.round(feed.scrollTop),
                        scrollHeight: feed.scrollHeight,
                        clientHeight: feed.clientHeight,
                        distanceFromBottom: Math.round(
                            feed.scrollHeight - feed.scrollTop - feed.clientHeight
                        ),
                        isScrollContainer: feed.scrollHeight > feed.clientHeight,
                        pageScrolls: doc.scrollHeight > doc.clientHeight
                    };
                });
            }
        });
    } finally {
        await browser.close();
        await server.close();
    }
}
