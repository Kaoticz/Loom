<script lang="ts">
    import { onMount } from 'svelte';
    import { agentStore } from '$lib/stores/agent';
    import type { Theme } from '$lib/stores/agent';
    import { goto } from '$app/navigation';
    import DirectoryPicker from '$lib/components/DirectoryPicker.svelte';
    import {
        DEFAULT_CHUNK_LINES,
        MAX_CHUNK_LINES,
        MIN_CHUNK_LINES,
        readChunkLines
    } from '$lib/chunkSize';
    import {
        DEFAULT_CONTEXT_WINDOW,
        MAX_CONTEXT_WINDOW,
        MIN_CONTEXT_WINDOW,
        readContextWindow
    } from '$lib/contextWindow';
    import {
        indexDeletionSchema,
        indexStatusSchema,
        modelDescriptionSchema,
        ollamaModelsSchema,
        parseResponse,
        repoValidationSchema,
        SchemaValidationError
    } from '$lib/schemas';
    import type { IndexStatus, ModelDescription } from '$lib/schemas';

    const THEMES: Theme[] = ['light', 'dark', 'system'];

    let availableModels: string[] = $state([]);
    let modelsError = $state('');
    let described: ModelDescription | null = $state(null);

    /**
     * Ask what the selected embedding model reports about itself.
     *
     * Both values come from the model's metadata rather than from using it, so
     * this does not load the model.
     */
    async function describeEmbeddingModel(name: string) {
        described = null;
        if (!name) return;
        try {
            const response = await fetch(
                `${apiBase}/api/v1/ollama/models/${encodeURIComponent(name)}/description`
            );
            described = parseResponse(
                modelDescriptionSchema,
                await response.json(),
                'model description'
            );
        } catch (e) {
            console.error('Failed to describe the embedding model', e);
            described = { dimensions: null, context_length: null };
        }
    }

    $effect(() => {
        void describeEmbeddingModel(config.embeddingModel);
    });
    let showPicker = $state(false);
    let repoError = $state('');
    let isValidating = $state(false);

    let config = $state({ ...$agentStore.config });

    const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

    // --- Indexed data ----------------------------------------------------

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

    /**
     * Whether removing the index is available.
     *
     * False while the status is unknown, so the control cannot be used before
     * the server has said whether it may be.
     */
    const canDeleteIndex = $derived.by(() => {
        // Bound to a local first: inside a `$derived` expression the checker
        // narrows the state variable to its initial `null` and then reports the
        // rest as unreachable, which a plain local avoids.
        const status = indexStatus;
        return status !== null && status.can_delete && !deleting;
    });

    /**
     * Why removal is unavailable, in the server's words.
     *
     * Taken from the response rather than composed here: the server decides
     * whether a session is running or nothing is indexed, and an explanation
     * written separately would eventually disagree with it.
     */
    const indexReason = $derived.by(() => {
        const status = indexStatus;
        if (deleting) return 'Removing the index...';
        if (status === null) return 'Checking what is indexed for this project...';
        if (!status.can_delete) return status.blocked_reason;

        const files = status.indexed_files;
        return `${files} ${files === 1 ? 'file is' : 'files are'} indexed for this repository.`;
    });

    /**
     * Ask what is indexed for the repository currently entered.
     *
     * Called again after a deletion so the control settles into its disabled
     * state rather than continuing to offer work that is already done.
     */
    async function refreshIndexStatus() {
        statusError = '';
        if (!config.repoPath) {
            indexStatus = null;
            return;
        }
        try {
            const response = await fetch(
                `${apiBase}/api/v1/index/status?path=${encodeURIComponent(config.repoPath)}`
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

    /**
     * Remove the index for the repository currently entered.
     *
     * Guarded rather than relying on the button being unusable, because the
     * button is only marked disabled to assistive technology and stays
     * clickable so that its reason remains reachable.
     */
    async function deleteIndex() {
        // Belt and braces, and honestly unreachable today: confirming clears
        // `confirmingDelete` below, which unmounts the button that calls this, so
        // no second click can arrive. It is kept because the operation is
        // destructive and should not depend on a button being unmounted for its
        // safety — but no test covers it, because none can without contriving a
        // call the interface cannot make. The guard that is tested is the one on
        // the button that opens the confirmation.
        if (!canDeleteIndex) return;

        confirmingDelete = false;
        deleting = true;
        indexError = '';
        indexOutcome = '';

        try {
            const response = await fetch(
                `${apiBase}/api/v1/index?path=${encodeURIComponent(config.repoPath)}`,
                { method: 'DELETE' }
            );
            if (!response.ok) {
                // A refusal carries its reason in the body; anything else at
                // least carries its status. Either way this must not be able to
                // look like a success.
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

    onMount(async () => {
        try {
            const response = await fetch(`${apiBase}/api/v1/ollama/models`);
            availableModels = parseResponse(
                ollamaModelsSchema,
                await response.json(),
                'model list'
            );
        } catch (e) {
            // Logged for whoever is debugging, and shown to whoever is waiting:
            // an empty dropdown with no explanation looks like the app is broken.
            console.error('Failed to fetch models', e);
            modelsError = e instanceof Error ? e.message : 'The model list could not be loaded.';
        }

        // Independent of the model list: a failure there must not leave the
        // index section stuck reporting that it is still checking.
        await refreshIndexStatus();
    });

    async function validateRepo() {
        // The index section describes whichever repository is entered, so it has
        // to follow the path as well. Without this the section kept reporting
        // the previous repository's count while the remove button acted on the
        // new one — the two could disagree about which project was even meant,
        // and nothing would have said so.
        //
        // Hung off the same trigger as validation rather than off the value, so
        // that typing a path does not fire a request per keystroke.
        await refreshIndexStatus();

        if (!config.repoPath) return;
        isValidating = true;
        repoError = '';
        try {
            const response = await fetch(
                `${apiBase}/api/v1/repos/validate?path=${encodeURIComponent(config.repoPath)}`
            );
            const data = parseResponse(
                repoValidationSchema,
                await response.json(),
                'repository check'
            );
            if (!data.valid) {
                repoError = 'Invalid git repository';
            }
        } catch (e: unknown) {
            // Naming validation as the cause, rather than reporting the generic
            // failure a network error would produce.
            repoError =
                e instanceof SchemaValidationError ? e.message : 'Failed to validate repository';
        } finally {
            isValidating = false;
        }
    }

    function save() {
        agentStore.updateConfig(config);
        goto('/');
    }

    function addSecret() {
        config.secrets = [...config.secrets, { key: '', value: '' }];
    }

    function removeSecret(index: number) {
        config.secrets = config.secrets.filter((_, i) => i !== index);
    }

    function updateThemeInstant(t: Theme) {
        config.theme = t;
        agentStore.updateConfig({ theme: t });
    }
</script>

<div
    class="min-h-screen bg-gray-50 px-4 py-8 transition-colors duration-200 dark:bg-gray-900 sm:px-6 lg:px-8"
>
    <div class="mx-auto max-w-3xl">
        <div class="mb-8 flex items-center justify-between">
            <h1 class="text-3xl font-bold text-gray-900 dark:text-white">Settings</h1>
            <button
                onclick={() => goto('/')}
                class="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white"
            >
                Back to Chat
            </button>
        </div>

        <div class="space-y-6">
            <!-- Model Selection -->
            <section class="rounded-lg bg-white p-6 shadow dark:bg-gray-800">
                <h2 class="mb-4 text-xl font-semibold text-gray-900 dark:text-white">AI Models</h2>
                {#if modelsError}
                    <p
                        role="status"
                        data-testid="models-error"
                        class="mb-4 rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-300"
                    >
                        {modelsError}
                    </p>
                {/if}
                <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <div>
                        <label
                            for="coderModel"
                            class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
                            >Coder Model</label
                        >
                        <select
                            id="coderModel"
                            bind:value={config.coderModel}
                            disabled={availableModels.length === 0}
                            class="w-full rounded border border-gray-300 bg-white p-2 text-gray-900 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                        >
                            {#if availableModels.length === 0}
                                <option>No models available</option>
                            {:else}
                                {#each availableModels as model}
                                    <option value="ollama:{model}">{model}</option>
                                {/each}
                            {/if}
                        </select>
                    </div>
                    <div>
                        <label
                            for="reviewerModel"
                            class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
                            >Reviewer Model</label
                        >
                        <select
                            id="reviewerModel"
                            bind:value={config.reviewerModel}
                            disabled={availableModels.length === 0}
                            class="w-full rounded border border-gray-300 bg-white p-2 text-gray-900 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                        >
                            {#if availableModels.length === 0}
                                <option>No models available</option>
                            {:else}
                                {#each availableModels as model}
                                    <option value="ollama:{model}">{model}</option>
                                {/each}
                            {/if}
                        </select>
                    </div>
                    <div>
                        <label
                            for="coderContextWindow"
                            class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
                            >Coder Context Window</label
                        >
                        <p class="mb-1 text-xs text-gray-500 dark:text-gray-400">
                            How many tokens of history the coder may hold before older tool results
                            are trimmed. Set this to your model's actual window. Too high and the
                            run overflows mid-task; too low only trims sooner than needed.
                        </p>
                        <div class="flex items-center gap-3">
                            <label class="flex items-center gap-2 text-sm dark:text-gray-300">
                                <input
                                    type="checkbox"
                                    aria-label="Automatic coder context window"
                                    checked={config.coderContextWindow === null}
                                    onchange={(event) => {
                                        config.coderContextWindow = event.currentTarget.checked
                                            ? null
                                            : DEFAULT_CONTEXT_WINDOW;
                                    }}
                                />
                                Auto
                            </label>
                            <input
                                id="coderContextWindow"
                                type="number"
                                min={MIN_CONTEXT_WINDOW}
                                max={MAX_CONTEXT_WINDOW}
                                step="1024"
                                disabled={config.coderContextWindow === null}
                                value={config.coderContextWindow ?? DEFAULT_CONTEXT_WINDOW}
                                oninput={(event) => {
                                    config.coderContextWindow = readContextWindow(
                                        event.currentTarget.value
                                    );
                                }}
                                class="w-32 rounded border border-gray-300 bg-white p-2 text-gray-900 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                            />
                            <span class="text-xs text-gray-500 dark:text-gray-400">tokens</span>
                        </div>
                    </div>
                    <div>
                        <label
                            for="plannerContextWindow"
                            class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
                            >Planner Context Window</label
                        >
                        <p class="mb-1 text-xs text-gray-500 dark:text-gray-400">
                            The same budget for the planning agent, which runs on the coder's model
                            but does far less: one structured call, no tools. Held separately so a
                            tighter planning budget does not constrain the coder.
                        </p>
                        <div class="flex items-center gap-3">
                            <label class="flex items-center gap-2 text-sm dark:text-gray-300">
                                <input
                                    type="checkbox"
                                    aria-label="Automatic planner context window"
                                    checked={config.plannerContextWindow === null}
                                    onchange={(event) => {
                                        config.plannerContextWindow = event.currentTarget.checked
                                            ? null
                                            : DEFAULT_CONTEXT_WINDOW;
                                    }}
                                />
                                Auto
                            </label>
                            <input
                                id="plannerContextWindow"
                                type="number"
                                min={MIN_CONTEXT_WINDOW}
                                max={MAX_CONTEXT_WINDOW}
                                step="1024"
                                disabled={config.plannerContextWindow === null}
                                value={config.plannerContextWindow ?? DEFAULT_CONTEXT_WINDOW}
                                oninput={(event) => {
                                    config.plannerContextWindow = readContextWindow(
                                        event.currentTarget.value
                                    );
                                }}
                                class="w-32 rounded border border-gray-300 bg-white p-2 text-gray-900 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                            />
                            <span class="text-xs text-gray-500 dark:text-gray-400">tokens</span>
                        </div>
                    </div>
                    <div>
                        <label
                            for="embeddingModel"
                            class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
                            >Embedding Model</label
                        >
                        <p class="mb-1 text-xs text-gray-500 dark:text-gray-400">
                            Must be an embedding-capable model (e.g. <code>nomic-embed-text</code>,
                            <code>mxbai-embed-large</code>). Chat models such as <code>llama3</code> will
                            not produce usable embeddings.
                        </p>
                        <select
                            id="embeddingModel"
                            bind:value={config.embeddingModel}
                            disabled={availableModels.length === 0}
                            class="w-full rounded border border-gray-300 bg-white p-2 text-gray-900 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                        >
                            {#if availableModels.length === 0}
                                <option>No models available</option>
                            {:else}
                                {#each availableModels as model}
                                    <option value={model}>{model}</option>
                                {/each}
                            {/if}
                        </select>
                        <p
                            data-testid="embedding-model-description"
                            class="mt-1 text-xs text-gray-500 dark:text-gray-400"
                        >
                            {#if described === null}
                                Reading what this model reports about itself...
                            {:else if described.dimensions === null && described.context_length === null}
                                This model does not report its vector width or capacity.
                            {:else}
                                {described.dimensions ?? 'An unreported number of'} dimensions,
                                {described.context_length ?? 'an unreported'} token capacity.
                            {/if}
                        </p>
                    </div>

                    <div>
                        <label
                            for="chunkLines"
                            class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
                            >Chunk Size</label
                        >
                        <p class="mb-1 text-xs text-gray-500 dark:text-gray-400">
                            How many lines of code each embedding covers. Larger chunks carry more
                            context but match less precisely. Leave on Auto unless you are tuning
                            search quality: Auto shrinks any chunk the model refuses, while a set
                            size is used exactly and indexing fails if the model will not take it.
                            Changing this re-indexes the repository.
                        </p>
                        <div class="flex items-center gap-3">
                            <label class="flex items-center gap-2 text-sm dark:text-gray-300">
                                <input
                                    type="checkbox"
                                    aria-label="Automatic chunk size"
                                    checked={config.chunkLines === null}
                                    onchange={(event) => {
                                        config.chunkLines = event.currentTarget.checked
                                            ? null
                                            : DEFAULT_CHUNK_LINES;
                                    }}
                                />
                                Auto
                            </label>
                            <input
                                id="chunkLines"
                                type="number"
                                min={MIN_CHUNK_LINES}
                                max={MAX_CHUNK_LINES}
                                step="10"
                                disabled={config.chunkLines === null}
                                value={config.chunkLines ?? DEFAULT_CHUNK_LINES}
                                oninput={(event) => {
                                    config.chunkLines = readChunkLines(event.currentTarget.value);
                                }}
                                class="w-32 rounded border border-gray-300 bg-white p-2 text-gray-900 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                            />
                            <span class="text-xs text-gray-500 dark:text-gray-400">lines</span>
                        </div>
                    </div>
                </div>
            </section>

            <!-- Repository Configuration -->
            <section class="rounded-lg bg-white p-6 shadow dark:bg-gray-800">
                <h2 class="mb-4 text-xl font-semibold text-gray-900 dark:text-white">
                    Project Repository
                </h2>
                <div>
                    <label
                        for="repoPath"
                        class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
                        >Repository Path</label
                    >
                    <div class="flex gap-2">
                        <input
                            id="repoPath"
                            type="text"
                            bind:value={config.repoPath}
                            onblur={validateRepo}
                            placeholder="/path/to/your/git/repo"
                            class="flex-1 border p-2 {repoError
                                ? 'border-red-500'
                                : 'border-gray-300 dark:border-gray-600'} rounded bg-white text-gray-900 dark:bg-gray-700 dark:text-white"
                        />
                        <button
                            onclick={() => (showPicker = true)}
                            class="flex items-center gap-2 rounded border border-gray-300 bg-gray-100 px-4 py-2 text-gray-700 transition-colors hover:bg-gray-200 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600"
                        >
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                class="h-4 w-4"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                stroke-width="2"
                                ><path
                                    d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"
                                /></svg
                            >
                            Browse
                        </button>
                    </div>
                    {#if repoError}
                        <p class="mt-1 text-sm text-red-500">{repoError}</p>
                    {/if}
                    {#if isValidating}
                        <p class="mt-1 text-sm text-gray-500">Validating...</p>
                    {/if}
                </div>
            </section>

            <!-- Environment Configuration -->
            <section class="rounded-lg bg-white p-6 shadow dark:bg-gray-800">
                <h2 class="mb-4 text-xl font-semibold text-gray-900 dark:text-white">
                    Environment
                </h2>

                <div class="mb-6">
                    <label
                        for="dockerfile"
                        class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
                        >Dockerfile</label
                    >
                    <textarea
                        id="dockerfile"
                        bind:value={config.dockerfile}
                        class="h-32 w-full rounded border border-gray-300 bg-white p-2 font-mono text-sm text-gray-900 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                    ></textarea>
                </div>

                <div role="group" aria-labelledby="secrets-heading">
                    <div class="mb-2 flex items-center justify-between">
                        <span
                            id="secrets-heading"
                            class="block text-sm font-medium text-gray-700 dark:text-gray-300"
                            >Secrets / Env Vars</span
                        >
                        <button
                            onclick={addSecret}
                            class="rounded bg-blue-600 px-2 py-1 text-sm text-white hover:bg-blue-700"
                        >
                            + Add
                        </button>
                    </div>
                    <div class="space-y-2">
                        {#each config.secrets as secret, i}
                            <div class="flex gap-2">
                                <input
                                    type="text"
                                    bind:value={secret.key}
                                    placeholder="KEY"
                                    aria-label="Secret {i + 1} name"
                                    class="flex-1 rounded border border-gray-300 bg-white p-2 text-gray-900 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                                />
                                <input
                                    type="password"
                                    bind:value={secret.value}
                                    placeholder="VALUE"
                                    aria-label="Secret {i + 1} value"
                                    class="flex-1 rounded border border-gray-300 bg-white p-2 text-gray-900 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                                />
                                <button
                                    onclick={() => removeSecret(i)}
                                    aria-label="Remove secret {i + 1}"
                                    class="px-2 text-red-500 hover:text-red-700"
                                >
                                    &times;
                                </button>
                            </div>
                        {/each}
                    </div>
                </div>
            </section>

            <!-- Indexed data -->
            <section class="rounded-lg bg-white p-6 shadow dark:bg-gray-800">
                <h2 class="mb-4 text-xl font-semibold text-gray-900 dark:text-white">
                    Indexed data
                </h2>

                <p class="mb-4 text-sm text-gray-600 dark:text-gray-400">
                    Removing the index deletes the stored files, symbols and embeddings for this
                    repository, for every embedding model. Other repositories are left alone. This
                    cannot be undone except by indexing again, which happens when a session next
                    starts rather than immediately.
                </p>

                <!--
                    The reason is always on the page rather than only in a tooltip.
                    A disabled control that explains itself on hover alone tells
                    keyboard and touch users nothing, which is what the approval
                    panel solves the same way.
                -->
                <!--
                    One text node on purpose. Interpolating the count next to
                    the wording split the sentence across nodes, which reads
                    the same but cannot be found as a sentence — by a screen
                    reader or by a test.
                -->
                <p id="index-delete-reason" class="mb-3 text-sm text-gray-700 dark:text-gray-300">
                    {indexReason}
                </p>

                {#if confirmingDelete}
                    <div
                        class="mb-3 rounded border border-red-300 bg-red-50 p-3 dark:border-red-800 dark:bg-red-900/20"
                    >
                        <p class="mb-2 text-sm text-red-800 dark:text-red-200">
                            Remove the index for <span class="font-mono">{config.repoPath}</span>?
                            This cannot be undone.
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
                    <!--
                        Marked disabled to assistive technology rather than with
                        the `disabled` attribute, which would drop the button out
                        of the tab order and take its explanation with it. The
                        action is refused in the handler instead.
                    -->
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
                    <p class="mt-3 text-sm text-green-700 dark:text-green-400" role="status">
                        {indexOutcome}
                    </p>
                {/if}

                {#if indexError}
                    <p class="mt-3 text-sm text-red-600 dark:text-red-400" role="alert">
                        {indexError}
                    </p>
                {/if}

                {#if statusError}
                    <p class="mt-3 text-sm text-amber-700 dark:text-amber-400">
                        {statusError}
                    </p>
                {/if}
            </section>

            <!-- Theme Selection -->
            <section class="rounded-lg bg-white p-6 shadow dark:bg-gray-800">
                <h2 class="mb-4 text-xl font-semibold text-gray-900 dark:text-white">Appearance</h2>
                <div class="flex gap-4">
                    {#each THEMES as t}
                        <button
                            onclick={() => updateThemeInstant(t)}
                            class="flex-1 rounded border px-4 py-2 {config.theme === t
                                ? 'border-blue-600 bg-blue-600 text-white'
                                : 'border-gray-300 bg-white text-gray-700 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-300'}"
                        >
                            {t.charAt(0).toUpperCase() + t.slice(1)}
                        </button>
                    {/each}
                </div>
            </section>

            {#if showPicker}
                <div
                    class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm"
                >
                    <DirectoryPicker
                        initialPath={config.repoPath || '/repo'}
                        onclose={() => (showPicker = false)}
                        onselect={(path) => {
                            config.repoPath = path;
                            showPicker = false;
                            validateRepo();
                        }}
                    />
                </div>
            {/if}

            <div class="mt-8 flex justify-end gap-4">
                <button
                    onclick={() => goto('/')}
                    class="px-6 py-2 text-gray-600 hover:text-gray-800 dark:text-gray-400 dark:hover:text-white"
                >
                    Cancel
                </button>
                <button
                    onclick={save}
                    class="rounded bg-blue-600 px-6 py-2 text-white shadow transition hover:bg-blue-700"
                >
                    Save Changes
                </button>
            </div>
        </div>
    </div>
</div>
