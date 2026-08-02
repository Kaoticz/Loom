<script lang="ts">
    import { onMount } from 'svelte';
    import { parseResponse, taskListSchema, type Task } from '$lib/schemas';
    import { taskStatusStyle } from '$lib/taskPresentation';

    const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

    let tasks: Task[] = $state([]);
    let total = $state(0);
    let offset = $state(0);
    let loading = $state(true);
    let error = $state('');

    const LIMIT = 20;

    async function load(nextOffset: number) {
        loading = true;
        error = '';
        try {
            const response = await fetch(
                `${apiBase}/api/v1/tasks?offset=${nextOffset}&limit=${LIMIT}`
            );
            if (!response.ok) throw new Error(`Server returned ${response.status}`);
            const data = parseResponse(taskListSchema, await response.json(), 'task list');
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

<div class="mx-auto max-w-4xl p-6">
    <div class="mb-6 flex items-center justify-between">
        <h1 class="text-2xl font-bold dark:text-white">Task History</h1>
        <a href="/" class="text-sm text-blue-600 hover:underline dark:text-blue-400"
            >Back to agent</a
        >
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
                        class="block rounded-xl border border-gray-200 bg-white p-4 transition-colors hover:border-blue-400 dark:border-gray-700 dark:bg-gray-800 dark:hover:border-blue-500"
                    >
                        <div class="flex items-center justify-between gap-4">
                            <span class="truncate font-medium dark:text-gray-100"
                                >{task.prompt}</span
                            >
                            <span
                                class="shrink-0 rounded px-2 py-1 text-xs {taskStatusStyle(
                                    task.status
                                )}"
                            >
                                {task.status}
                            </span>
                        </div>
                        <div class="mt-1 text-xs text-gray-500 dark:text-gray-400">
                            {new Date(task.created_at).toLocaleString()}
                            {#if task.branch_name}
                                &middot; {task.branch_name}{/if}
                        </div>
                    </a>
                </li>
            {/each}
        </ul>

        <div class="mt-6 flex items-center justify-between text-sm">
            <button
                disabled={offset === 0}
                onclick={() => load(Math.max(0, offset - LIMIT))}
                class="rounded border border-gray-300 px-3 py-1 disabled:opacity-40 dark:border-gray-600 dark:text-gray-200"
            >
                Previous
            </button>
            <span class="text-gray-500 dark:text-gray-400">
                {offset + 1}&ndash;{Math.min(offset + LIMIT, total)} of {total}
            </span>
            <button
                disabled={offset + LIMIT >= total}
                onclick={() => load(offset + LIMIT)}
                class="rounded border border-gray-300 px-3 py-1 disabled:opacity-40 dark:border-gray-600 dark:text-gray-200"
            >
                Next
            </button>
        </div>
    {/if}
</div>
