<script lang="ts">
    import { untrack } from 'svelte';
    import {
        indexDeletionSchema,
        indexStatusSchema,
        parseResponse,
        SchemaValidationError
    } from '$lib/schemas';
    import type { IndexStatus } from '$lib/schemas';

    let {
        apiBase,
        repoPath,
        refreshRequest
    }: {
        apiBase: string;
        repoPath: string;
        refreshRequest: number;
    } = $props();

    let indexStatus: IndexStatus | null = $state(null);
    // Two separate errors on purpose. The status check runs again in the
    // deletion's `finally`, so a single field would let a successful re-check
    // wipe the message explaining that the deletion had just failed — leaving
    // the control looking as though it had worked.
    let statusError = $state('');
    let indexError = $state('');
    let indexOutcome = $state('');
    let confirmingDelete = $state(false);
    let deleting = $state(false);

    const canDeleteIndex = $derived.by(() => {
        const status = indexStatus;
        return status !== null && status.can_delete && !deleting;
    });

    const indexReason = $derived.by(() => {
        const status = indexStatus;
        if (deleting) return 'Removing the index...';
        if (status === null) return 'Checking what is indexed for this project...';
        if (!status.can_delete) return status.blocked_reason;

        const files = status.indexed_files;
        return `${files} ${files === 1 ? 'file is' : 'files are'} indexed for this repository.`;
    });

    /** Ask what is indexed for the repository whose path was last validated. */
    async function refreshIndexStatus(): Promise<void> {
        statusError = '';
        if (!repoPath) {
            indexStatus = null;
            return;
        }
        try {
            const response = await fetch(
                `${apiBase}/api/v1/index/status?path=${encodeURIComponent(repoPath)}`
            );
            if (!response.ok) throw new Error(`The server returned ${response.status}`);
            indexStatus = parseResponse(indexStatusSchema, await response.json(), 'index status');
        } catch (error) {
            indexStatus = null;
            statusError =
                error instanceof SchemaValidationError
                    ? error.message
                    : `Could not check what is indexed: ${error instanceof Error ? error.message : error}`;
        }
    }

    /** Remove the current repository's index after the visible confirmation. */
    async function deleteIndex(): Promise<void> {
        if (!canDeleteIndex) return;

        confirmingDelete = false;
        deleting = true;
        indexError = '';
        indexOutcome = '';

        try {
            const response = await fetch(
                `${apiBase}/api/v1/index?path=${encodeURIComponent(repoPath)}`,
                { method: 'DELETE' }
            );
            if (!response.ok) {
                const detail = await response
                    .json()
                    .then((body) => body?.detail)
                    .catch(() => null);
                throw new Error(detail || `The server returned ${response.status}`);
            }
            const result = parseResponse(
                indexDeletionSchema,
                await response.json(),
                'index deletion'
            );
            indexOutcome = result.message;
        } catch (error) {
            indexError = `The index was not removed: ${error instanceof Error ? error.message : error}`;
        } finally {
            deleting = false;
            await refreshIndexStatus();
        }
    }

    // The parent increments this only on mount and repository validation. The
    // path is deliberately read untracked, so typing does not request per key.
    $effect(() => {
        void refreshRequest;
        void untrack(() => refreshIndexStatus());
    });
</script>

<section class="rounded-lg bg-white p-6 shadow dark:bg-gray-800">
    <h2 class="mb-4 text-xl font-semibold text-gray-900 dark:text-white">Indexed data</h2>

    <p class="mb-4 text-sm text-gray-600 dark:text-gray-400">
        Removing the index deletes the stored files, symbols and embeddings for this repository, for
        every embedding model. Other repositories are left alone. This cannot be undone except by
        indexing again, which happens when a session next starts rather than immediately.
    </p>

    <!-- The reason stays visible so keyboard and touch users can reach it. -->
    <p id="index-delete-reason" class="mb-3 text-sm text-gray-700 dark:text-gray-300">
        {indexReason}
    </p>

    {#if confirmingDelete}
        <div
            class="mb-3 rounded border border-red-300 bg-red-50 p-3 dark:border-red-800 dark:bg-red-900/20"
        >
            <p class="mb-2 text-sm text-red-800 dark:text-red-200">
                Remove the index for <span class="font-mono">{repoPath}</span>? This cannot be
                undone.
            </p>
            <div class="flex gap-2">
                <button
                    onclick={deleteIndex}
                    class="rounded bg-red-600 px-3 py-1.5 text-sm text-white hover:bg-red-700"
                >
                    Yes, remove it
                </button>
                <button
                    onclick={() => (confirmingDelete = false)}
                    class="rounded border border-gray-300 px-3 py-1.5 text-sm text-gray-700 dark:border-gray-600 dark:text-gray-300"
                >
                    Cancel
                </button>
            </div>
        </div>
    {:else}
        <button
            onclick={() => canDeleteIndex && (confirmingDelete = true)}
            aria-disabled={!canDeleteIndex}
            aria-describedby="index-delete-reason"
            class="rounded px-4 py-2 text-sm transition-all {canDeleteIndex
                ? 'bg-red-600 text-white hover:bg-red-700 active:scale-95'
                : 'cursor-not-allowed bg-red-600 text-white opacity-50'}"
        >
            Remove indexed data
        </button>
    {/if}

    {#if indexOutcome}
        <p class="mt-3 text-sm text-green-700 dark:text-green-400" role="status">{indexOutcome}</p>
    {/if}
    {#if indexError}
        <p class="mt-3 text-sm text-red-600 dark:text-red-400" role="alert">{indexError}</p>
    {/if}
    {#if statusError}
        <p class="mt-3 text-sm text-amber-700 dark:text-amber-400">{statusError}</p>
    {/if}
</section>
