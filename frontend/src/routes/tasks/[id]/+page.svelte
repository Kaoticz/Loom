<script lang="ts">
    import { onMount } from 'svelte';

    const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
    import { page } from '$app/stores';

    interface LogEntry {
        entry_type: string;
        content: Record<string, unknown>;
        created_at: string;
    }

    let detail: any = null;
    let loading = true;
    let error = '';

    const entryStyles: Record<string, string> = {
        plan: 'border-purple-300 dark:border-purple-700',
        tool_call: 'border-yellow-300 dark:border-yellow-700',
        commit: 'border-slate-300 dark:border-slate-600',
        review: 'border-orange-300 dark:border-orange-700',
        rejection: 'border-red-300 dark:border-red-700',
        status_change: 'border-blue-300 dark:border-blue-700',
        summary: 'border-gray-300 dark:border-gray-600'
    };

    /** One-line description of a log entry, derived from its type. */
    function describe(entry: LogEntry): string {
        const c = entry.content as any;
        switch (entry.entry_type) {
            case 'plan':
                return (c.steps ?? []).map((s: string, i: number) => `${i + 1}. ${s}`).join('\n');
            case 'tool_call':
                return `${c.tool} [${c.status}] ${c.result ?? ''}`;
            case 'commit':
                return `${c.message} (${String(c.hash ?? '').slice(0, 7)})`;
            case 'review':
                return `${c.approved ? 'Approved' : 'Rejected'}: ${c.comments ?? ''}`;
            case 'rejection':
                return `User rejected the ${c.stage}: ${c.reason ?? ''}`;
            case 'status_change':
                return `Status became ${c.status}`;
            case 'summary':
                return `${c.collapsed_entries} earlier entries collapsed: ${JSON.stringify(c.counts)}`;
            default:
                return JSON.stringify(c);
        }
    }

    onMount(async () => {
        try {
            const response = await fetch(`${apiBase}/api/v1/tasks/${$page.params.id}`);
            if (!response.ok) throw new Error(`Server returned ${response.status}`);
            detail = await response.json();
        } catch (e) {
            error = e instanceof Error ? e.message : String(e);
        } finally {
            loading = false;
        }
    });
</script>

<div class="p-6 max-w-4xl mx-auto">
    <a href="/tasks" class="text-sm text-blue-600 dark:text-blue-400 hover:underline">&larr; All tasks</a>

    {#if loading}
        <p class="mt-4 text-gray-500 dark:text-gray-400">Loading task...</p>
    {:else if error}
        <p class="mt-4 text-red-600 dark:text-red-400">Could not load this task: {error}</p>
    {:else if detail}
        <h1 class="text-xl font-bold mt-4 mb-1 dark:text-white">{detail.task.prompt}</h1>
        <div class="text-sm text-gray-500 dark:text-gray-400 mb-4">
            {detail.task.status}
            {#if detail.task.branch_name} &middot; {detail.task.branch_name}{/if}
            &middot; {new Date(detail.task.created_at).toLocaleString()}
        </div>

        {#if detail.task.summary}
            <div class="p-3 mb-6 rounded-lg bg-green-50 dark:bg-green-900/20 text-green-900 dark:text-green-200 text-sm whitespace-pre-wrap">
                {detail.task.summary}
            </div>
        {/if}

        <h2 class="font-semibold mb-2 dark:text-gray-100">Action log</h2>
        {#if detail.log.length === 0}
            <p class="text-gray-500 dark:text-gray-400 text-sm">Nothing was recorded for this task.</p>
        {:else}
            <ol class="space-y-2">
                {#each detail.log as entry}
                    <li class="p-3 rounded-lg border-l-4 bg-white dark:bg-gray-800 {entryStyles[entry.entry_type] ?? entryStyles.summary}">
                        <div class="text-[10px] uppercase tracking-widest text-gray-400 mb-1">
                            {entry.entry_type.replaceAll('_', ' ')}
                        </div>
                        <div class="text-sm whitespace-pre-wrap dark:text-gray-200">{describe(entry)}</div>
                    </li>
                {/each}
            </ol>
        {/if}
    {/if}
</div>
