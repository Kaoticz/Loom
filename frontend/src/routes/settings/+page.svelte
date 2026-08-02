<script lang="ts">
    import { onMount } from 'svelte';
    import { agentStore } from '$lib/stores/agent';
    import type { Theme } from '$lib/stores/agent';
    import { goto } from '$app/navigation';
    import DirectoryPicker from '$lib/components/DirectoryPicker.svelte';
    import {
        ollamaModelsSchema,
        parseResponse,
        repoValidationSchema,
        SchemaValidationError
    } from '$lib/schemas';

    const THEMES: Theme[] = ['light', 'dark', 'system'];

    let availableModels: string[] = $state([]);
    let modelsError = $state('');
    let showPicker = $state(false);
    let repoError = $state('');
    let isValidating = $state(false);

    let config = $state({ ...$agentStore.config });

    const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

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
    });

    async function validateRepo() {
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
