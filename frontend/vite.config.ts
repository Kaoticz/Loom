import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

export default defineConfig({
	plugins: [sveltekit()],
	test: {
		// The store reads `browser` from $app/environment and touches localStorage,
		// so tests need a DOM rather than a bare node environment.
		environment: 'jsdom',
		include: ['src/**/*.test.ts'],
		globals: true
	}
});
