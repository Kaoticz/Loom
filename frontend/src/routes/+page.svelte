<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import AgentChat from '$lib/components/AgentChat.svelte';
    import DiffViewer from '$lib/components/DiffViewer.svelte';
    import SandboxBanner from '$lib/components/SandboxBanner.svelte';
    import { isDarkTheme, prefersDark } from '$lib/theme';
    import { agentStore } from '$lib/stores/agent';
    import { goto } from '$app/navigation';
    import { ollamaModelsSchema, parseResponse } from '$lib/schemas';
    import '../app.css';

    const dark = $derived(isDarkTheme($agentStore.config.theme, $prefersDark));

    let availableModels: string[] = $state([]);
    let modelsError = $state('');

    onMount(async () => {
        const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
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
        const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/api/v1/agent-session';
        agentStore.connect(wsUrl);
    });

    onDestroy(() => {
        agentStore.disconnect();
    });

    function handleModelChange(e: Event) {
        const target = e.target as HTMLSelectElement;
        agentStore.updateConfig({ coderModel: target.value });
    }
</script>

<div
    class="flex h-screen flex-col overflow-hidden bg-gray-100 transition-colors duration-200 dark:bg-gray-900"
>
    <!-- Navbar -->
    <header
        class="z-10 flex h-14 shrink-0 items-center justify-between border-b border-gray-200 bg-white px-6 shadow-sm dark:border-gray-700 dark:bg-gray-800"
    >
        <div class="flex items-center gap-6">
            <div class="flex items-center gap-2">
                <div
                    class="flex h-8 w-8 items-center justify-center rounded bg-blue-600 font-bold text-white"
                >
                    L
                </div>
                <h1 class="text-xl font-bold tracking-tight dark:text-white">
                    Loom <span class="text-blue-600">AI</span>
                </h1>
            </div>

            <div class="hidden items-center gap-2 text-sm text-gray-500 dark:text-gray-400 md:flex">
                <span class="font-medium">Project:</span>
                <span
                    class="max-w-[200px] truncate rounded bg-gray-100 px-2 py-0.5 dark:bg-gray-700"
                    title={$agentStore.config.repoPath}
                >
                    {$agentStore.config.repoPath
                        ? $agentStore.config.repoPath.split('/').pop()
                        : 'No Project Selected'}
                </span>
            </div>
        </div>

        <div class="flex items-center gap-4">
            {#if modelsError}
                <span
                    role="status"
                    data-testid="models-error"
                    class="max-w-[280px] truncate text-xs text-red-600 dark:text-red-400"
                    title={modelsError}
                >
                    {modelsError}
                </span>
            {/if}
            <div class="flex items-center gap-2">
                <label
                    for="coderModel"
                    class="text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400"
                    >Coder:</label
                >
                <select
                    id="coderModel"
                    value={$agentStore.config.coderModel}
                    onchange={handleModelChange}
                    class="rounded border border-gray-300 bg-gray-50 px-2 py-1 text-sm outline-none focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                >
                    {#if availableModels.length === 0}
                        <option>No models</option>
                    {:else}
                        {#each availableModels as model}
                            <option value="ollama:{model}">{model}</option>
                        {/each}
                    {/if}
                </select>
            </div>

            <button
                onclick={() => goto('/tasks')}
                class="rounded-full p-2 text-gray-600 transition-colors hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
                title="Task history"
            >
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    class="h-5 w-5"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    ><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"
                    ></polyline></svg
                >
            </button>

            <button
                onclick={() => goto('/settings')}
                class="rounded-full p-2 text-gray-600 transition-colors hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
                title="Settings"
            >
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    class="h-5 w-5"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    ><path
                        d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"
                    ></path><circle cx="12" cy="12" r="3"></circle></svg
                >
            </button>

            <div class="flex items-center gap-2">
                <div
                    class="h-2 w-2 rounded-full {$agentStore.status === 'idle'
                        ? 'bg-gray-400'
                        : 'animate-pulse bg-green-500'}"
                ></div>
                <span class="font-mono text-xs text-gray-500">{$agentStore.status}</span>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <!-- Requirement 6.8: full width, below the top bar and above the feed. -->
    {#if $agentStore.sandboxUnavailable && !$agentStore.sandboxNoticeDismissed}
        <SandboxBanner
            stage={$agentStore.sandboxUnavailable.stage}
            reason={$agentStore.sandboxUnavailable.reason}
            ondismiss={() => agentStore.dismissSandboxNotice()}
        />
    {/if}

    <main class="flex flex-1 overflow-hidden">
        <div class="w-1/3 min-w-[400px]">
            <AgentChat />
        </div>
        <div class="relative flex-1 bg-[#1e1e1e]">
            <DiffViewer diffs={$agentStore.fileDiffs} {dark} />
        </div>
    </main>
</div>

<style>
    :global(body) {
        margin: 0;
        padding: 0;
        font-family:
            -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
</style>
