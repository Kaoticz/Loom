<script lang="ts">
    import { onMount } from 'svelte';

    const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
    import { page } from '$app/stores';
    import {
        parseResponse,
        taskDetailSchema,
        type TaskDetail,
        type TaskLogEntry
    } from '$lib/schemas';

    let detail: TaskDetail | null = $state(null);
    let loading = $state(true);
    let error = $state('');

    const entryStyles: Record<string, string> = {
        plan: 'border-purple-300 dark:border-purple-700',
        tool_call: 'border-yellow-300 dark:border-yellow-700',
        commit: 'border-slate-300 dark:border-slate-600',
        review: 'border-orange-300 dark:border-orange-700',
        rejection: 'border-red-300 dark:border-red-700',
        status_change: 'border-blue-300 dark:border-blue-700',
        summary: 'border-gray-300 dark:border-gray-600'
    };

    /**
     * Read a string out of a log entry's free-form content.
     *
     * The content of a log entry varies by entry type and is not described by a
     * schema, so each field is narrowed at the point of use rather than cast.
     */
    function text(content: Record<string, unknown>, key: string): string {
        const value = content[key];
        return typeof value === 'string' ? value : '';
    }

    /** Read the plan steps out of a log entry's content. */
    function steps(content: Record<string, unknown>): string[] {
        const value = content.steps;
        if (!Array.isArray(value)) return [];
        return value.filter((step): step is string => typeof step === 'string');
    }

    /** One-line description of a log entry, derived from its type. */
    function describe(entry: TaskLogEntry): string {
        const c = entry.content;
        switch (entry.entry_type) {
            case 'plan':
                return steps(c)
                    .map((step, index) => `${index + 1}. ${step}`)
                    .join('\n');
            case 'tool_call':
                return `${text(c, 'tool')} [${text(c, 'status')}] ${text(c, 'result')}`;
            case 'commit':
                return `${text(c, 'message')} (${text(c, 'hash').slice(0, 7)})`;
            case 'review':
                return `${c.approved ? 'Approved' : 'Rejected'}: ${text(c, 'comments')}`;
            case 'rejection':
                return `User rejected the ${text(c, 'stage')}: ${text(c, 'reason')}`;
            case 'status_change':
                return `Status became ${text(c, 'status')}`;
            case 'summary':
                return `${String(c.collapsed_entries)} earlier entries collapsed: ${JSON.stringify(c.counts)}`;
            default:
                return JSON.stringify(c);
        }
    }

    onMount(async () => {
        try {
            const response = await fetch(`${apiBase}/api/v1/tasks/${$page.params.id}`);
            if (!response.ok) throw new Error(`Server returned ${response.status}`);
            detail = parseResponse(taskDetailSchema, await response.json(), 'task');
        } catch (e) {
            error = e instanceof Error ? e.message : String(e);
        } finally {
            loading = false;
        }
    });
</script>

<div class="mx-auto max-w-4xl p-6">
    <a href="/tasks" class="text-sm text-blue-600 hover:underline dark:text-blue-400"
        >&larr; All tasks</a
    >

    {#if loading}
        <p class="mt-4 text-gray-500 dark:text-gray-400">Loading task...</p>
    {:else if error}
        <p class="mt-4 text-red-600 dark:text-red-400">Could not load this task: {error}</p>
    {:else if detail}
        <h1 class="mb-1 mt-4 text-xl font-bold dark:text-white">{detail.task.prompt}</h1>
        <div class="mb-4 text-sm text-gray-500 dark:text-gray-400">
            {detail.task.status}
            {#if detail.task.branch_name}
                &middot; {detail.task.branch_name}{/if}
            &middot; {new Date(detail.task.created_at).toLocaleString()}
        </div>

        {#if detail.task.summary}
            <div
                class="mb-6 whitespace-pre-wrap rounded-lg bg-green-50 p-3 text-sm text-green-900 dark:bg-green-900/20 dark:text-green-200"
            >
                {detail.task.summary}
            </div>
        {/if}

        <h2 class="mb-2 font-semibold dark:text-gray-100">Action log</h2>
        {#if detail.log.length === 0}
            <p class="text-sm text-gray-500 dark:text-gray-400">
                Nothing was recorded for this task.
            </p>
        {:else}
            <ol class="space-y-2">
                {#each detail.log as entry}
                    <li
                        class="rounded-lg border-l-4 bg-white p-3 dark:bg-gray-800 {entryStyles[
                            entry.entry_type
                        ] ?? entryStyles.summary}"
                    >
                        <div class="mb-1 text-[10px] uppercase tracking-widest text-gray-400">
                            {entry.entry_type.replaceAll('_', ' ')}
                        </div>
                        <div class="whitespace-pre-wrap text-sm dark:text-gray-200">
                            {describe(entry)}
                        </div>
                    </li>
                {/each}
            </ol>
        {/if}
    {/if}
</div>
