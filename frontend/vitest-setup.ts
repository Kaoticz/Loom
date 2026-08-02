import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/svelte';
import { afterEach } from 'vitest';

// Components mounted by a test are torn down after it, so one test cannot see
// another's DOM.
afterEach(() => {
    cleanup();
});
