<script lang="ts">
    import type { SandboxStage } from '$lib/schemas';
    import { sandboxStageLabels } from '$lib/chatPresentation';

    let {
        stage,
        reason,
        ondismiss
    }: {
        stage: SandboxStage;
        reason: string;
        ondismiss: () => void;
    } = $props();
</script>

<div
    role="status"
    data-testid="sandbox-banner"
    class="flex w-full items-start gap-3 border-b border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-900/30 dark:text-amber-200"
>
    <span aria-hidden="true" class="mt-0.5 shrink-0">⚠</span>
    <div class="min-w-0 flex-1">
        <p class="font-semibold">{sandboxStageLabels[stage]}</p>
        <p class="mt-0.5">Files can still be edited, but no command can be run to verify them.</p>
        <p class="mt-1 break-words font-mono text-xs opacity-75">{reason}</p>
    </div>
    <button
        type="button"
        onclick={ondismiss}
        aria-label="Dismiss sandbox warning"
        class="shrink-0 rounded px-2 py-0.5 text-lg leading-none transition-colors hover:bg-amber-200 dark:hover:bg-amber-800/50"
    >
        ×
    </button>
</div>
