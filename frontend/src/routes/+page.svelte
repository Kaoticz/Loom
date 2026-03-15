<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import AgentChat from '$lib/components/AgentChat.svelte';
    import DiffViewer from '$lib/components/DiffViewer.svelte';
    import ConfigView from '$lib/components/ConfigView.svelte';
    import { agentStore } from '$lib/stores/agent';
    import '../app.css';

    let showConfig = false;
    let dockerfile = 'FROM python:3.14-slim\n\nWORKDIR /workspace\n';
    let secrets: { key: string; value: string }[] = [];

    onMount(() => {
        const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/api/v1/agent-session';
        agentStore.connect(wsUrl);
    });

    onDestroy(() => {
        agentStore.disconnect();
    });

    function handleConfigSave(event: CustomEvent) {
        dockerfile = event.detail.dockerfile;
        secrets = event.detail.secrets;
        showConfig = false;
        // In a real app, we'd send this to the backend
        console.log('Saved config:', { dockerfile, secrets });
    }
</script>

<div class="h-screen flex flex-col overflow-hidden bg-gray-100">
    <!-- Navbar -->
    <header class="h-14 bg-white border-b border-gray-200 flex items-center justify-between px-6 shrink-0">
        <div class="flex items-center gap-2">
            <div class="w-8 h-8 bg-blue-600 rounded flex items-center justify-center text-white font-bold">L</div>
            <h1 class="text-xl font-bold tracking-tight">Loom <span class="text-blue-600">AI</span></h1>
        </div>
        
        <div class="flex items-center gap-4">
            <button 
                on:click={() => showConfig = true}
                class="text-sm font-medium text-gray-600 hover:text-blue-600 transition-colors"
            >
                Environment Settings
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

    <!-- Modals -->
    {#if showConfig}
        <div class="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <ConfigView 
                {dockerfile} 
                {secrets} 
                on:close={() => showConfig = false}
                on:save={handleConfigSave}
            />
        </div>
    {/if}
</div>

<style>
    :global(body) {
        margin: 0;
        padding: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
</style>
