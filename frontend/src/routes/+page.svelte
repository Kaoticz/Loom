<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import AgentChat from '$lib/components/AgentChat.svelte';
    import DiffViewer from '$lib/components/DiffViewer.svelte';
    import { agentStore } from '$lib/stores/agent';
    import { goto } from '$app/navigation';
    import '../app.css';

    let availableModels: string[] = [];

    onMount(async () => {
        const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
        try {
            const response = await fetch(`${apiBase}/api/v1/ollama/models`);
            availableModels = await response.json();
        } catch (e) {
            console.error('Failed to fetch models', e);
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

<div class="h-screen flex flex-col overflow-hidden bg-gray-100 dark:bg-gray-900 transition-colors duration-200">
    <!-- Navbar -->
    <header class="h-14 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between px-6 shrink-0 shadow-sm z-10">
        <div class="flex items-center gap-6">
            <div class="flex items-center gap-2">
                <div class="w-8 h-8 bg-blue-600 rounded flex items-center justify-center text-white font-bold">L</div>
                <h1 class="text-xl font-bold tracking-tight dark:text-white">Loom <span class="text-blue-600">AI</span></h1>
            </div>
            
            <div class="hidden md:flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
                <span class="font-medium">Project:</span>
                <span class="bg-gray-100 dark:bg-gray-700 px-2 py-0.5 rounded truncate max-w-[200px]" title={$agentStore.config.repoPath}>
                    {$agentStore.config.repoPath ? $agentStore.config.repoPath.split('/').pop() : 'No Project Selected'}
                </span>
            </div>
        </div>
        
        <div class="flex items-center gap-4">
            <div class="flex items-center gap-2">
                <label for="coderModel" class="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Coder:</label>
                <select 
                    id="coderModel"
                    value={$agentStore.config.coderModel}
                    on:change={handleModelChange}
                    class="text-sm bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-2 py-1 dark:text-white focus:ring-2 focus:ring-blue-500 outline-none"
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
                on:click={() => goto('/settings')}
                class="p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full transition-colors"
                title="Settings"
            >
                <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"></path><circle cx="12" cy="12" r="3"></circle></svg>
            </button>

            <div class="flex items-center gap-2">
                <div class="w-2 h-2 rounded-full {$agentStore.status === 'idle' ? 'bg-gray-400' : 'bg-green-500 animate-pulse'}"></div>
                <span class="text-xs font-mono text-gray-500">{$agentStore.status}</span>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="flex-1 flex overflow-hidden">
        <div class="w-1/3 min-w-[400px]">
            <AgentChat />
        </div>
        <div class="flex-1 relative bg-[#1e1e1e]">
            <DiffViewer />
        </div>
    </main>

</div>

<style>
    :global(body) {
        margin: 0;
        padding: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
</style>
