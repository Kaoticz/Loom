<script lang="ts">
    import type { FileDiff } from '$lib/stores/agent';
    import { orderFilePaths, retainExpanded, toggleExpanded } from '$lib/diffSections';
    import DiffSection from './DiffSection.svelte';

    let {
        diffs,
        dark = false
    }: {
        diffs: Record<string, FileDiff>;
        dark?: boolean;
    } = $props();

    let expanded = $state(new Set<string>());

    const paths = $derived(orderFilePaths(Object.keys(diffs)));

    // A file the reader opened stays open as further diffs arrive; one that has
    // just appeared starts collapsed.
    $effect(() => {
        const retained = retainExpanded(expanded, paths);
        if (retained.size !== expanded.size) expanded = retained;
    });

    function toggle(path: string) {
        expanded = toggleExpanded(expanded, path);
    }
</script>

<div class="flex h-full flex-col overflow-y-auto bg-white transition-colors dark:bg-[#1e1e1e]">
    {#if paths.length === 0}
        <div class="px-4 py-2 text-sm italic text-gray-400 dark:text-gray-500">
            No files modified yet
        </div>
    {:else}
        {#each paths as path (path)}
            <DiffSection
                {path}
                diff={diffs[path]}
                expanded={expanded.has(path)}
                {dark}
                ontoggle={toggle}
            />
        {/each}
    {/if}
</div>
