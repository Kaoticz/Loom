<script lang="ts">
    import { onMount } from 'svelte';
    import { agentStore } from '$lib/stores/agent';
    import { goto } from '$app/navigation';
    import DirectoryPicker from '$lib/components/DirectoryPicker.svelte';

    let availableModels: string[] = [];
    let showPicker = false;
    let repoError = '';
    let isValidating = false;

    let config = { ...$agentStore.config };

    const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

    onMount(async () => {
        try {
            const response = await fetch(`${apiBase}/api/v1/ollama/models`);
            availableModels = await response.json();
        } catch (e) {
            console.error('Failed to fetch models', e);
        }
    });

    async function validateRepo() {
        if (!config.repoPath) return;
        isValidating = true;
        repoError = '';
        try {
            const response = await fetch(`${apiBase}/api/v1/repos/validate?path=${encodeURIComponent(config.repoPath)}`);
            const data = await response.json();
            if (!data.valid) {
                repoError = data.error || 'Invalid git repository';
            }
        } catch (e) {
            repoError = 'Failed to validate repository';
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

    function updateThemeInstant(t: 'light' | 'dark' | 'system') {
        config.theme = t;
        agentStore.updateConfig({ theme: t });
    }
</script>

<div class="min-h-screen bg-gray-50 dark:bg-gray-900 py-8 px-4 sm:px-6 lg:px-8 transition-colors duration-200">
    <div class="max-w-3xl mx-auto">
        <div class="flex items-center justify-between mb-8">
            <h1 class="text-3xl font-bold text-gray-900 dark:text-white">Settings</h1>
            <button 
                on:click={() => goto('/')}
                class="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
            >
                Back to Chat
            </button>
        </div>

        <div class="space-y-6">
            <!-- Model Selection -->
            <section class="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
                <h2 class="text-xl font-semibold mb-4 text-gray-900 dark:text-white">AI Models</h2>
                <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Coder Model</label>
                        <select 
                            bind:value={config.coderModel}
                            disabled={availableModels.length === 0}
                            class="w-full p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
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
                        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Reviewer Model</label>
                        <select 
                            bind:value={config.reviewerModel}
                            disabled={availableModels.length === 0}
                            class="w-full p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
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
                        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Embedding Model</label>
                        <select 
                            bind:value={config.embeddingModel}
                            disabled={availableModels.length === 0}
                            class="w-full p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
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
            <section class="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
                <h2 class="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Project Repository</h2>
                <div>
                    <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Repository Path</label>
                    <div class="flex gap-2">
                        <input 
                            type="text" 
                            bind:value={config.repoPath}
                            on:blur={validateRepo}
                            placeholder="/path/to/your/git/repo"
                            class="flex-1 p-2 border {repoError ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'} rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                        />
                        <button 
                            on:click={() => showPicker = true}
                            class="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors flex items-center gap-2"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
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
            <section class="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
                <h2 class="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Environment</h2>
                
                <div class="mb-6">
                    <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Dockerfile</label>
                    <textarea
                        bind:value={config.dockerfile}
                        class="w-full h-32 p-2 font-mono text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    ></textarea>
                </div>

                <div>
                    <div class="flex justify-between items-center mb-2">
                        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300">Secrets / Env Vars</label>
                        <button
                            on:click={addSecret}
                            class="text-sm bg-blue-600 text-white px-2 py-1 rounded hover:bg-blue-700"
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
                                    class="flex-1 p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                                />
                                <input
                                    type="password"
                                    bind:value={secret.value}
                                    placeholder="VALUE"
                                    class="flex-1 p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                                />
                                <button
                                    on:click={() => removeSecret(i)}
                                    class="text-red-500 hover:text-red-700 px-2"
                                >
                                    &times;
                                </button>
                            </div>
                        {/each}
                    </div>
                </div>
            </section>

            <!-- Theme Selection -->
            <section class="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
                <h2 class="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Appearance</h2>
                <div class="flex gap-4">
                    {#each ['light', 'dark', 'system'] as t}
                        <button
                            on:click={() => updateThemeInstant(t)}
                            class="flex-1 py-2 px-4 rounded border {config.theme === t ? 'bg-blue-600 text-white border-blue-600' : 'bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 border-gray-300 dark:border-gray-600'}"
                        >
                            {t.charAt(0).toUpperCase() + t.slice(1)}
                        </button>
                    {/each}
                </div>
            </section>

            {#if showPicker}
                <div class="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
                    <DirectoryPicker 
                        initialPath={config.repoPath || '/'}
                        on:close={() => showPicker = false}
                        on:select={(e) => {
                            config.repoPath = e.detail;
                            showPicker = false;
                            validateRepo();
                        }}
                    />
                </div>
            {/if}

            <div class="flex justify-end gap-4 mt-8">
                <button
                    on:click={() => goto('/')}
                    class="px-6 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-white"
                >
                    Cancel
                </button>
                <button
                    on:click={save}
                    class="px-6 py-2 bg-blue-600 text-white rounded shadow hover:bg-blue-700 transition"
                >
                    Save Changes
                </button>
            </div>
        </div>
    </div>
</div>
