<script lang="ts">
    import type { BlockedReason } from '$lib/stores/agent';

    let {
        reason,
        onretry
    }: {
        reason: BlockedReason;
        onretry: () => void;
    } = $props();
</script>

<div
    class="space-y-3 rounded-2xl border-2 border-dashed border-amber-400 bg-amber-50 p-4 dark:border-amber-500 dark:bg-amber-900/20"
>
    <p class="text-sm font-semibold text-amber-900 dark:text-amber-200">
        {#if reason === 'another_tab'}
            A Loom session is already running in another tab.
        {:else}
            A previous session is still closing.
        {/if}
    </p>
    <p class="text-xs text-amber-800 dark:text-amber-300">
        {#if reason === 'another_tab'}
            Switch to that tab to continue, or close it and retry here. Browsers do not allow one
            tab to focus or close another.
        {:else}
            No other tab answered, so its connection is probably still being cleaned up. Retrying
            usually works within a few seconds.
        {/if}
    </p>
    <button
        onclick={onretry}
        class="rounded-lg bg-amber-600 px-4 py-2 text-sm text-white shadow-md transition-all hover:bg-amber-700 active:scale-95"
    >
        Retry
    </button>
</div>
