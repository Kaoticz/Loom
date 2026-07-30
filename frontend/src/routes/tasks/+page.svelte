<script lang="ts">
    import { onMount } from 'svelte';

    const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

    interface TaskSummary {
        id: string;
        status: string;
        prompt: string;
        repo_path: string;
        branch_name: string;
        summary: string;
        created_at: string;
    }

    let tasks: TaskSummary[] = [];
    let total = 0;
    let offset = 0;
    let loading = true;
    let error = '';

    const LIMIT = 20;

    const statusStyles: Record<string, string> = {
        initialized: 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-200',
        running: 'bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-200',
        completed: 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-200',
        merged: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200',
        failed: 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-200'
    };

    async function load(nextOffset: number) {
        loading = true;
        error = '';
        try {
            const response = await fetch(`${apiBase}/api/v1/tasks?offset=${nextOffset}&limit=${LIMIT}`);
            if (!response.ok) throw new Error(`Server returned ${response.status}`);
            const data = await response.json();
            tasks = data.items;
            total = data.total;
            offset = data.offset;
        } catch (e) {
            error = e instanceof Error ? e.message : String(e);
        } finally {
            loading = false;
        }
    }

    onMount(() => load(0));
</script>

<div class="p-6 max-w-4xl mx-auto">
    <div class="flex items-center justify-between mb-6">
        <h1 class="text-2xl font-bold dark:text-white">Task History</h1>
        <a href="/" class="text-sm text-blue-600 dark:text-blue-400 hover:underline">Back to agent</a>
    </div>

    {#if loading}
        <p class="text-gray-500 dark:text-gray-400">Loading tasks...</p>
    {:else if error}
        <p class="text-red-600 dark:text-red-400">Could not load tasks: {error}</p>
    {:else if tasks.length === 0}
        <p class="text-gray-500 dark:text-gray-400">No tasks have been run yet.</p>
    {:else}
        <ul class="space-y-2">
            {#each tasks as task}
                <li>
                    <a
                        href="/tasks/{task.id}"
                        class="block p-4 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-blue-400 dark:hover:border-blue-500 bg-white dark:bg-gray-800 transition-colors"
                    >
                        <div class="flex items-center justify-between gap-4">
                            <span class="font-medium dark:text-gray-100 truncate">{task.prompt}</span>
                            <span class="text-xs px-2 py-1 rounded shrink-0 {statusStyles[task.status] ?? statusStyles.initialized}">
                                {task.status}
                            </span>
                        </div>
                        <div class="text-xs text-gray-500 dark:text-gray-400 mt-1">
                            {new Date(task.created_at).toLocaleString()}
                            {#if task.branch_name} &middot; {task.branch_name}{/if}
                        </div>
                    </a>
                </li>
            {/each}
        </ul>

        <div class="flex items-center justify-between mt-6 text-sm">
            <button
                disabled={offset === 0}
                on:click={() => load(Math.max(0, offset - LIMIT))}
                class="px-3 py-1 rounded border border-gray-300 dark:border-gray-600 disabled:opacity-40 dark:text-gray-200"
            >
                Previous
            </button>
            <span class="text-gray-500 dark:text-gray-400">
                {offset + 1}&ndash;{Math.min(offset + LIMIT, total)} of {total}
            </span>
            <button
                disabled={offset + LIMIT >= total}
                on:click={() => load(offset + LIMIT)}
                class="px-3 py-1 rounded border border-gray-300 dark:border-gray-600 disabled:opacity-40 dark:text-gray-200"
            >
                Next
            </button>
        </div>
    {/if}
</div>
