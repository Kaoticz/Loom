import { describe, expect, it } from 'vitest';
import { get } from 'svelte/store';
import { isDarkTheme, prefersDark } from './theme';

describe('isDarkTheme', () => {
    it('is dark when dark is chosen, whatever the system says', () => {
        expect(isDarkTheme('dark', false)).toBe(true);
        expect(isDarkTheme('dark', true)).toBe(true);
    });

    it('is light when light is chosen, whatever the system says', () => {
        expect(isDarkTheme('light', true)).toBe(false);
        expect(isDarkTheme('light', false)).toBe(false);
    });

    it('follows the system when system is chosen', () => {
        expect(isDarkTheme('system', true)).toBe(true);
        expect(isDarkTheme('system', false)).toBe(false);
    });
});

describe('prefersDark', () => {
    it('falls back to light where media queries are unavailable', () => {
        // jsdom implements no media queries, which is also the server side case.
        expect(get(prefersDark)).toBe(false);
    });

    it('reports what the media query says, and follows changes', () => {
        const listeners: ((event: MediaQueryListEvent) => void)[] = [];
        const query = {
            matches: true,
            addEventListener: (_: string, handler: (event: MediaQueryListEvent) => void) =>
                listeners.push(handler),
            removeEventListener: () => {}
        };
        const original = window.matchMedia;
        // @ts-expect-error jsdom provides no matchMedia to spy on.
        window.matchMedia = () => query;

        try {
            let seen: boolean | undefined;
            const stop = prefersDark.subscribe((value) => {
                seen = value;
            });
            expect(seen).toBe(true);

            listeners[0]({ matches: false } as MediaQueryListEvent);
            expect(seen).toBe(false);

            stop();
        } finally {
            window.matchMedia = original;
        }
    });
});
