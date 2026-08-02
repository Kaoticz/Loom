<script lang="ts" module>
    let sectionsCreated = 0;
</script>

<script lang="ts">
    import type { FileDiff } from '$lib/stores/agent';
    import DiffEditor from './DiffEditor.svelte';

    let {
        path,
        diff,
        expanded = false,
        dark = false,
        ontoggle
    }: {
        path: string;
        diff: FileDiff;
        expanded?: boolean;
        dark?: boolean;
        ontoggle: (path: string) => void;
    } = $props();

    // Derived from an instance count rather than from the path: two paths that
    // differ only in characters the sanitiser strips would otherwise share an
    // id, leaving `aria-controls` pointing at the wrong panel.
    const panelId = `diff-panel-${(sectionsCreated += 1)}`;
</script>

<section class="border-b border-gray-200 dark:border-[#333]">
    <h3>
        <button
            type="button"
            onclick={() => ontoggle(path)}
            aria-expanded={expanded}
            aria-controls={panelId}
            class="flex w-full items-center gap-2 px-4 py-2 text-left text-sm transition-colors hover:bg-gray-100 dark:hover:bg-[#2a2d2e]"
        >
            <span
                class="text-gray-400 transition-transform dark:text-gray-500 {expanded
                    ? 'rotate-90'
                    : ''}"
                aria-hidden="true">›</span
            >
            <span class="truncate font-mono text-gray-700 dark:text-gray-300">{path}</span>
            {#if diff.truncated}
                <span
                    class="ml-auto shrink-0 rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-amber-800 dark:bg-amber-900/40 dark:text-amber-300"
                >
                    truncated
                </span>
            {/if}
        </button>
    </h3>

    {#if expanded}
        <div id={panelId}>
            {#if diff.truncated}
                <div
                    class="border-y border-amber-200 bg-amber-50 px-4 py-2 text-xs text-amber-800 dark:border-amber-800 dark:bg-amber-900/20 dark:text-amber-300"
                >
                    This file is too large to show in full. The diff below is truncated.
                </div>
            {/if}
            <!-- Mounted only while open, so a change touching many files does not
                 build an editor for every one of them. -->
            <div class="max-h-[60vh] overflow-auto">
                <DiffEditor {path} original={diff.original} modified={diff.modified} {dark} />
            </div>
        </div>
    {/if}
</section>
