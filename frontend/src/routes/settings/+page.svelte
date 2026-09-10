<script lang="ts">
    import { onMount } from 'svelte';
    import { goto } from '$app/navigation';
    import DirectoryPicker from '$lib/components/DirectoryPicker.svelte';
    import { agentStore } from '$lib/stores/agent';
    import type { AgentConfig, Theme } from '$lib/stores/agent';
    import {
        ollamaModelsSchema,
        parseResponse,
        repoValidationSchema,
        SchemaValidationError
    } from '$lib/schemas';
    import IndexStatusPanel from './IndexStatusPanel.svelte';
    import ModelSettingsPanel from './ModelSettingsPanel.svelte';
    import SecretsEditor from './SecretsEditor.svelte';

    const THEMES: Theme[] = ['light', 'dark', 'system'];
    const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

    let availableModels: string[] = $state([]);
    let modelsError = $state('');
    let showPicker = $state(false);
    let repoError = $state('');
    let isValidating = $state(false);
    let indexRefreshRequest = $state(0);
    let config = $state({ ...$agentStore.config });

    onMount(async () => {
        try {
            const response = await fetch(`${apiBase}/api/v1/ollama/models`);
            availableModels = parseResponse(
                ollamaModelsSchema,
                await response.json(),
                'model list'
            );
        } catch (error) {
            console.error('Failed to fetch models', error);
            modelsError =
                error instanceof Error ? error.message : 'The model list could not be loaded.';
        }
    });

    async function validateRepo(): Promise<void> {
        // Index status follows validation, not each keystroke in the path input.
        indexRefreshRequest += 1;
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
            if (!data.valid) repoError = 'Invalid git repository';
        } catch (error: unknown) {
            repoError =
                error instanceof SchemaValidationError
                    ? error.message
                    : 'Failed to validate repository';
        } finally {
            isValidating = false;
        }
    }

    function updateModelSettings(
        patch: Partial<
            Pick<
                AgentConfig,
                | 'coderModel'
                | 'reviewerModel'
                | 'embeddingModel'
                | 'coderContextWindow'
                | 'plannerContextWindow'
                | 'chunkLines'
            >
        >
    ): void {
        Object.assign(config, patch);
    }

    function save(): void {
        agentStore.updateConfig(config);
        goto('/');
    }

    function updateThemeInstant(theme: Theme): void {
        config.theme = theme;
        agentStore.updateConfig({ theme });
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
                >Back to Chat</button
            >
        </div>

        <div class="space-y-6">
            <ModelSettingsPanel
                {apiBase}
                {availableModels}
                {modelsError}
                settings={config}
                onchange={updateModelSettings}
            />

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
                    {#if repoError}<p class="mt-1 text-sm text-red-500">{repoError}</p>{/if}
                    {#if isValidating}<p class="mt-1 text-sm text-gray-500">Validating...</p>{/if}
                </div>
            </section>

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
                <SecretsEditor
                    secrets={config.secrets}
                    onchange={(secrets) => (config.secrets = secrets)}
                />
            </section>

            <IndexStatusPanel
                {apiBase}
                repoPath={config.repoPath}
                refreshRequest={indexRefreshRequest}
            />

            <section class="rounded-lg bg-white p-6 shadow dark:bg-gray-800">
                <h2 class="mb-4 text-xl font-semibold text-gray-900 dark:text-white">Appearance</h2>
                <div class="flex gap-4">
                    {#each THEMES as theme}
                        <button
                            onclick={() => updateThemeInstant(theme)}
                            class="flex-1 rounded border px-4 py-2 {config.theme === theme
                                ? 'border-blue-600 bg-blue-600 text-white'
                                : 'border-gray-300 bg-white text-gray-700 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-300'}"
                            >{theme.charAt(0).toUpperCase() + theme.slice(1)}</button
                        >
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
                    >Cancel</button
                >
                <button
                    onclick={save}
                    class="rounded bg-blue-600 px-6 py-2 text-white shadow transition hover:bg-blue-700"
                    >Save Changes</button
                >
            </div>
        </div>
    </div>
</div>
