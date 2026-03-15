<script lang="ts">
    import { createEventDispatcher, onMount } from 'svelte';

    const dispatch = createEventDispatcher();
    export let initialPath = '/';

    let currentPath = initialPath;
    let directories: string[] = [];
    let isLoading = false;
    let error = '';

    onMount(() => {
        loadDirectories(currentPath);
    });

    const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

    async function loadDirectories(path: string) {
        isLoading = true;
        error = '';
        try {
            const response = await fetch(`${apiBase}/api/v1/repos/list?path=${encodeURIComponent(path)}`);
            if (!response.ok) throw new Error('Failed to load directories');
            directories = await response.json();
            currentPath = path;
        } catch (e: any) {
            error = e.message;
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
        dispatch('select', currentPath);
    }
</script>

<div class="bg-white dark:bg-gray-800 rounded-lg shadow-xl overflow-hidden flex flex-col max-h-[80vh] w-full max-w-md">
    <div class="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center bg-gray-50 dark:bg-gray-900/50">
        <h3 class="font-bold dark:text-white">Select Directory</h3>
        <button on:click={() => dispatch('close')} class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-white">&times;</button>
    </div>

    <div class="p-4 bg-gray-100 dark:bg-gray-900/30 border-b border-gray-200 dark:border-gray-700 flex items-center gap-2">
        <button on:click={navigateUp} class="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded" title="Up">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 17l-5-5 5-5M18 17l-5-5 5-5"/></svg>
        </button>
        <div class="text-xs font-mono truncate dark:text-gray-300" title={currentPath}>{currentPath}</div>
    </div>

    <div class="flex-1 overflow-y-auto p-2 min-h-[200px]">
        {#if isLoading}
            <div class="flex justify-center p-8">
                <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            </div>
        {:else if error}
            <div class="p-4 text-red-500 text-sm text-center">{error}</div>
        {:else}
            <div class="space-y-1">
                {#each directories as dir}
                    <button 
                        on:click={() => selectDirectory(dir)}
                        class="w-full text-left px-3 py-2 rounded hover:bg-blue-50 dark:hover:bg-blue-900/20 flex items-center gap-3 transition-colors group"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5 text-yellow-500 fill-current" viewBox="0 0 24 24"><path d="M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>
                        <span class="text-sm dark:text-gray-200 group-hover:text-blue-600 dark:group-hover:text-blue-400">{dir}</span>
                    </button>
                {/each}
                {#if directories.length === 0}
                    <div class="p-8 text-center text-gray-400 text-sm italic">No subdirectories found</div>
                {/if}
            </div>
        {/if}
    </div>

    <div class="p-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 flex justify-end gap-3">
        <button on:click={() => dispatch('close')} class="px-4 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-white">Cancel</button>
        <button on:click={confirmSelection} class="px-4 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 shadow-sm transition-all active:scale-95">Select Current</button>
    </div>
</div>
