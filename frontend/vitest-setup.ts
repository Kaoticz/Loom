import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/svelte';
import { afterEach } from 'vitest';

// Node 26 exposes an unavailable storage global unless it is given a backing
// file. Tests need ordinary browser storage semantics, not that process-level
// opt-in, so provide the small in-memory implementation jsdom would normally.
if (typeof localStorage === 'undefined') {
    const values = new Map<string, string>();
    const storage: Storage = {
        get length(): number {
            return values.size;
        },
        clear(): void {
            values.clear();
        },
        getItem(key: string): string | null {
            return values.get(key) ?? null;
        },
        key(index: number): string | null {
            return [...values.keys()][index] ?? null;
        },
        removeItem(key: string): void {
            values.delete(key);
        },
        setItem(key: string, value: string): void {
            values.set(key, value);
        }
    };
    Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: storage });
}

// jsdom implements no scrolling methods, so a component that calls one throws
// rather than quietly doing nothing. This gives every element the smallest
// honest implementation: it moves the element, with no animation, which is as
// much as an environment without layout can model. A test that cares how the
// scroll was requested — instantly or smoothly — replaces this on the element
// it is exercising, and that own property takes precedence over this one.
// Guarded because setup also runs for files that opt into the node
// environment, where there is no DOM at all.
if (typeof Element !== 'undefined' && !Element.prototype.scrollTo) {
    Element.prototype.scrollTo = function scrollTo(
        this: Element,
        optionsOrX?: ScrollToOptions | number,
        y?: number
    ): void {
        const top = typeof optionsOrX === 'number' ? y : optionsOrX?.top;
        if (typeof top === 'number') this.scrollTop = top;
    } as Element['scrollTo'];
}

// Components mounted by a test are torn down after it, so one test cannot see
// another's DOM.
afterEach(() => {
    cleanup();
});
