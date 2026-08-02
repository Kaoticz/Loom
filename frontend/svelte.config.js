import adapter from '@sveltejs/adapter-auto';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
    preprocess: vitePreprocess(),
    // Runes everywhere. Without this, Svelte 5 infers the mode per component and
    // a stray `export let` or `$:` would silently compile in legacy mode.
    compilerOptions: {
        runes: true
    },
    kit: {
        adapter: adapter()
    }
};

export default config;
