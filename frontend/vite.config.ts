import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

export default defineConfig({
    plugins: [sveltekit()],
    // Svelte ships separate server and browser builds. Component tests mount
    // real components, so the browser build is the one to resolve.
    resolve: {
        conditions: ['browser']
    },
    test: {
        // The store reads `browser` from $app/environment and touches localStorage,
        // so tests need a DOM rather than a bare node environment.
        environment: 'jsdom',
        include: ['src/**/*.test.ts'],
        setupFiles: ['./vitest-setup.ts'],
        globals: true
    }
});
