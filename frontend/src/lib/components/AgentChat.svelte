<script lang="ts">
    import { agentStore } from '$lib/stores/agent';
    import { afterUpdate } from 'svelte';

    let promptInput = '';
    let chatContainer: HTMLElement;

    function handleSend() {
        if (promptInput.trim()) {
            agentStore.sendPrompt(promptInput);
            promptInput = '';
        }
    }

    afterUpdate(() => {
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    });

    const typeStyles = {
        prompt: 'bg-blue-50 border-blue-200 text-blue-800 self-end',
        thought: 'bg-gray-50 border-gray-200 text-gray-600 italic',
        action: 'bg-yellow-50 border-yellow-200 text-yellow-800 font-mono text-xs',
        observation: 'bg-green-50 border-green-200 text-green-800 font-mono text-xs',
        plan: 'bg-purple-50 border-purple-200 text-purple-800',
        review: 'bg-orange-50 border-orange-200 text-orange-800',
        done: 'bg-green-100 border-green-300 text-green-900 font-bold',
        error: 'bg-red-50 border-red-200 text-red-800'
    };
</script>

<div class="flex flex-col h-full bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 transition-colors duration-200">
    <!-- Header -->
    <div class="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center shadow-sm">
        <h2 class="font-bold text-lg dark:text-white">Agent Chat</h2>
        <span class="text-xs px-2 py-1 rounded bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 uppercase tracking-wider font-semibold">
            {$agentStore.status.replace('_', ' ')}
        </span>
    </div>

    <!-- Messages -->
    <div 
        bind:this={chatContainer}
        class="flex-1 overflow-y-auto p-4 space-y-4 flex flex-col scroll-smooth"
    >
        {#each $agentStore.messages as message}
            <div class="max-w-[90%] p-3 rounded-2xl border {typeStyles[message.type] || 'bg-gray-100 dark:bg-gray-700 border-gray-200 dark:border-gray-600'} shadow-sm transform transition-all hover:shadow-md">
                <div class="text-[10px] opacity-60 mb-1 uppercase font-bold tracking-widest">{message.type}</div>
                <div class="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</div>
            </div>
        {/each}

        {#if $agentStore.requiresAction}
            <div class="p-4 border-2 border-dashed border-blue-400 dark:border-blue-500 rounded-2xl bg-blue-50 dark:bg-blue-900/20 space-y-3 animate-pulse-slow">
                <p class="text-sm font-semibold text-blue-800 dark:text-blue-300">User Action Required:</p>
                <div class="flex gap-2">
                    <button 
                        on:click={() => agentStore.approve(true)}
                        class="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 shadow-md transition-all active:scale-95"
                    >
                        Approve
                    </button>
                    <button 
                        on:click={() => agentStore.approve(false)}
                        class="px-4 py-2 bg-white dark:bg-gray-800 text-blue-600 dark:text-blue-400 border border-blue-600 dark:border-blue-400 rounded-lg text-sm hover:bg-blue-50 dark:hover:bg-blue-900/30 transition-all active:scale-95"
                    >
                        Reject
                    </button>
                </div>
            </div>
        {/if}
    </div>

    <!-- Input -->
    <div class="p-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
        <div class="flex gap-2 bg-white dark:bg-gray-700 rounded-xl p-2 border border-gray-300 dark:border-gray-600 shadow-inner focus-within:ring-2 focus-within:ring-blue-500 transition-all">
            <textarea
                bind:value={promptInput}
                on:keydown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())}
                placeholder="Describe a task for the agent..."
                class="flex-1 p-2 bg-transparent dark:text-white outline-none resize-none h-20 text-sm"
            ></textarea>
            <button
                on:click={handleSend}
                disabled={$agentStore.status !== 'idle' && $agentStore.status !== 'completed' && $agentStore.status !== 'failed'}
                class="self-end p-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md active:scale-95"
                title="Send Message"
                aria-label="Send Message"
            >
                <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
            </button>
        </div>
    </div>
</div>

<style>
    @keyframes pulse-slow {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }
    .animate-pulse-slow {
        animation: pulse-slow 3s infinite ease-in-out;
    }
</style>
