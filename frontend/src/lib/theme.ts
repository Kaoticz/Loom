import { readable } from 'svelte/store';
import { browser } from '$app/environment';
import type { Theme } from '$lib/stores/agent';

/**
 * Resolving the configured theme into an actual appearance.
 *
 * Two places need to know whether the interface is currently dark: the layout,
 * which sets the class Tailwind keys off, and the diff editor, which has to be
 * told outright because CodeMirror styles itself rather than reading CSS
 * classes. Deriving that answer twice invites the two to disagree, and reading
 * it back off the DOM makes the reader race whoever writes the class. So it is
 * computed once, from the setting, and shared.
 */

const DARK_QUERY = '(prefers-color-scheme: dark)';

/**
 * Whether the interface should be dark.
 *
 * @param theme The configured preference.
 * @param prefersDark Whether the operating system asks for a dark appearance.
 * @returns True when dark styling applies.
 */
export function isDarkTheme(theme: Theme, prefersDark: boolean): boolean {
    if (theme === 'dark') return true;
    if (theme === 'light') return false;
    return prefersDark;
}

/**
 * Whether the operating system currently asks for a dark appearance.
 *
 * Falls back to light where the query is unavailable, which covers server side
 * rendering and test environments that implement no media queries.
 */
export const prefersDark = readable(false, (set) => {
    if (!browser || typeof window.matchMedia !== 'function') return;

    const query = window.matchMedia(DARK_QUERY);
    set(query.matches);

    const handler = (event: MediaQueryListEvent) => set(event.matches);
    query.addEventListener('change', handler);
    return () => query.removeEventListener('change', handler);
});
