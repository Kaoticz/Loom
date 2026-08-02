<script lang="ts">
    import { onMount, untrack } from 'svelte';
    import { directoryListingSchema, parseResponse } from '$lib/schemas';

    let {
        initialPath = '/repo',
        onclose,
        onselect
    }: {
        initialPath?: string;
        onclose: () => void;
        onselect: (path: string) => void;
    } = $props();

    // Seeded from the prop, then owned locally as the user navigates. Reading it
    // untracked states that the one-time read is deliberate: reopening the picker
    // is what should change where it starts, not a prop change while it is open.
    let currentPath = $state(untrack(() => initialPath));
    let directories: string[] = $state([]);
    let isLoading = $state(false);
    let error = $state('');

    onMount(() => {
        loadDirectories(currentPath);
    });

    const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

    async function loadDirectories(path: string) {
        isLoading = true;
        error = '';
        try {
            const response = await fetch(
                `${apiBase}/api/v1/repos/list?path=${encodeURIComponent(path)}`
            );
            if (!response.ok) throw new Error('Failed to load directories');
            directories = parseResponse(
                directoryListingSchema,
                await response.json(),
                'directory listing'
            );
            currentPath = path;
        } catch (e: unknown) {
            error = e instanceof Error ? e.message : 'Failed to load directories';
        } finally {
            isLoading = false;
        }
    }

    function navigateUp() {
        const parts = currentPath.split('/').filter(Boolean);
        parts.pop();
        const nextPath = '/' + parts.join('/');
        loadDirectories(nextPath);
    }

    function selectDirectory(name: string) {
        const nextPath = currentPath === '/' ? `/${name}` : `${currentPath}/${name}`;
        loadDirectories(nextPath);
    }

    function confirmSelection() {
        onselect(currentPath);
    }
</script>

<div
    class="flex max-h-[80vh] w-full max-w-md flex-col overflow-hidden rounded-lg bg-white shadow-xl dark:bg-gray-800"
>
    <div
        class="flex items-center justify-between border-b border-gray-200 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-900/50"
    >
        <h3 class="font-bold dark:text-white">Select Directory</h3>
        <button
            onclick={() => onclose()}
            class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-white"
            >&times;</button
        >
    </div>

    <div
        class="flex items-center gap-2 border-b border-gray-200 bg-gray-100 p-4 dark:border-gray-700 dark:bg-gray-900/30"
    >
        <button
            onclick={navigateUp}
            class="rounded p-1 hover:bg-gray-200 dark:hover:bg-gray-700"
            title="Up"
        >
            <svg
                xmlns="http://www.w3.org/2000/svg"
                class="h-4 w-4"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"><path d="M11 17l-5-5 5-5M18 17l-5-5 5-5" /></svg
            >
        </button>
        <div class="truncate font-mono text-xs dark:text-gray-300" title={currentPath}>
            {currentPath}
        </div>
    </div>

    <div class="min-h-[200px] flex-1 overflow-y-auto p-2">
        {#if isLoading}
            <div class="flex justify-center p-8">
                <div class="h-8 w-8 animate-spin rounded-full border-b-2 border-blue-600"></div>
            </div>
        {:else if error}
            <div class="p-4 text-center text-sm text-red-500">{error}</div>
        {:else}
            <div class="space-y-1">
                {#each directories as dir}
                    <button
                        onclick={() => selectDirectory(dir)}
                        class="group flex w-full items-center gap-3 rounded px-3 py-2 text-left transition-colors hover:bg-blue-50 dark:hover:bg-blue-900/20"
                    >
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            class="h-5 w-5 fill-current text-yellow-500"
                            viewBox="0 0 24 24"
                            ><path
                                d="M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"
                            /></svg
                        >
                        <span
                            class="text-sm group-hover:text-blue-600 dark:text-gray-200 dark:group-hover:text-blue-400"
                            >{dir}</span
                        >
                    </button>
                {/each}
                {#if directories.length === 0}
                    <div class="p-8 text-center text-sm italic text-gray-400">
                        No subdirectories found
                    </div>
                {/if}
            </div>
        {/if}
    </div>

    <div
        class="flex justify-end gap-3 border-t border-gray-200 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-900/50"
    >
        <button
            onclick={() => onclose()}
            class="px-4 py-2 text-sm text-gray-600 hover:text-gray-800 dark:text-gray-400 dark:hover:text-white"
            >Cancel</button
        >
        <button
            onclick={confirmSelection}
            class="rounded bg-blue-600 px-4 py-2 text-sm text-white shadow-sm transition-all hover:bg-blue-700 active:scale-95"
            >Select Current</button
        >
    </div>
</div>
