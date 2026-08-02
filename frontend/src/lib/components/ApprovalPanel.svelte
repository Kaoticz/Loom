<script lang="ts">
    import { actionLabels } from '$lib/chatPresentation';

    let {
        actionType,
        reason = '',
        onapprove,
        onreject
    }: {
        actionType: string;
        reason?: string;
        onapprove: () => void;
        onreject: (reason: string) => void;
    } = $props();

    const labels = $derived(actionLabels[actionType]);
</script>

<div
    class="space-y-3 rounded-2xl border-2 border-dashed border-blue-400 bg-blue-50 p-4 dark:border-blue-500 dark:bg-blue-900/20"
>
    <p class="text-sm font-semibold text-blue-800 dark:text-blue-300">{labels.title}</p>

    <p class="text-xs text-blue-700 dark:text-blue-400">
        {#if reason.trim()}
            Rejecting will send what you have typed above as the reason.
        {:else}
            To reject with a reason, type it in the box below first.
        {/if}
    </p>

    <div class="flex gap-2">
        <button
            onclick={onapprove}
            class="rounded-lg bg-blue-600 px-4 py-2 text-sm text-white shadow-md transition-all hover:bg-blue-700 active:scale-95"
        >
            {labels.approve}
        </button>
        <button
            onclick={() => onreject(reason.trim())}
            class="rounded-lg border border-blue-600 bg-white px-4 py-2 text-sm text-blue-600 transition-all hover:bg-blue-50 active:scale-95 dark:border-blue-400 dark:bg-gray-800 dark:text-blue-400 dark:hover:bg-blue-900/30"
        >
            {labels.reject}
        </button>
    </div>
</div>
