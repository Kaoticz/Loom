<script lang="ts">
    import { agentStore } from '$lib/stores/agent';
    import type { MessageType, ConnectionState } from '$lib/stores/agent';
    import { afterUpdate } from 'svelte';
    import { renderMarkdown } from '$lib/markdown';

    // Tool output and commit lines are literal text; rendering them as markdown
    // would mangle command output and diff fragments.
    const MARKDOWN_TYPES: MessageType[] = ['prompt', 'thought', 'plan', 'review', 'done', 'error'];

    let promptInput = '';
    let rejectionReason = '';
    let showRejectionBox = false;
    let chatContainer: HTMLElement;

    function reject() {
        agentStore.approve(false, rejectionReason.trim());
        rejectionReason = '';
        showRejectionBox = false;
    }

    function approve() {
        agentStore.approve(true);
        rejectionReason = '';
        showRejectionBox = false;
    }

    const connectionStyles: Record<ConnectionState, string> = {
        connected: 'bg-green-500',
        connecting: 'bg-yellow-500 animate-pulse',
        disconnected: 'bg-red-500'
    };

    const actionLabels: Record<string, { title: string; approve: string; reject: string }> = {
        approve_plan: {
            title: 'Approve this plan?',
            approve: 'Approve Plan',
            reject: 'Reject & Replan'
        },
        approve_review: {
            title: 'The reviewer asked for changes.',
            approve: 'Accept Anyway',
            reject: 'Send Back for Rework'
        },
        approve_merge: {
            title: 'The work is complete.',
            approve: 'Merge into Main',
            reject: 'Reject & Continue'
        }
    };

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

    const typeStyles: Record<MessageType, string> = {
        prompt: 'bg-blue-50 border-blue-200 text-blue-800 self-end',
        thought: 'bg-gray-50 border-gray-200 text-gray-600 italic',
        action: 'bg-yellow-50 border-yellow-200 text-yellow-800 font-mono text-xs',
        observation: 'bg-green-50 border-green-200 text-green-800 font-mono text-xs',
        plan: 'bg-purple-50 border-purple-200 text-purple-800',
        review: 'bg-orange-50 border-orange-200 text-orange-800',
        commit: 'bg-slate-50 border-slate-200 text-slate-600 font-mono text-xs',
        done: 'bg-green-100 border-green-300 text-green-900 font-bold',
        error: 'bg-red-50 border-red-200 text-red-800'
    };
</script>

<div class="flex flex-col h-full bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 transition-colors duration-200">
    <!-- Header -->
    <div class="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center shadow-sm">
        <div class="flex items-center gap-2">
            <h2 class="font-bold text-lg dark:text-white">Agent Chat</h2>
            <span
                class="w-2 h-2 rounded-full {connectionStyles[$agentStore.connection]}"
                title="Connection: {$agentStore.connection}"
            ></span>
            <span class="text-xs text-gray-500 dark:text-gray-400">{$agentStore.connection}</span>
        </div>
        <span class="text-xs px-2 py-1 rounded bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 uppercase tracking-wider font-semibold">
            {$agentStore.status.replaceAll('_', ' ')}
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
                {#if MARKDOWN_TYPES.includes(message.type)}
                    <div class="markdown-body text-sm leading-relaxed">{@html renderMarkdown(message.content)}</div>
                {:else}
                    <div class="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</div>
                {/if}
            </div>
        {/each}

        {#if $agentStore.blocked}
            <div class="p-4 border-2 border-dashed border-amber-400 dark:border-amber-500 rounded-2xl bg-amber-50 dark:bg-amber-900/20 space-y-3">
                <p class="text-sm font-semibold text-amber-900 dark:text-amber-200">
                    {#if $agentStore.blocked === 'another_tab'}
                        A Loom session is already running in another tab.
                    {:else}
                        A previous session is still closing.
                    {/if}
                </p>
                <p class="text-xs text-amber-800 dark:text-amber-300">
                    {#if $agentStore.blocked === 'another_tab'}
                        Switch to that tab to continue, or close it and retry here. Browsers do
                        not allow one tab to focus or close another.
                    {:else}
                        No other tab answered, so its connection is probably still being cleaned
                        up. Retrying usually works within a few seconds.
                    {/if}
                </p>
                <button
                    on:click={() => agentStore.retryConnection()}
                    class="px-4 py-2 bg-amber-600 text-white rounded-lg text-sm hover:bg-amber-700 shadow-md transition-all active:scale-95"
                >
                    Retry
                </button>
            </div>
        {/if}

        {#if $agentStore.requiresAction && $agentStore.actionType}
            {@const labels = actionLabels[$agentStore.actionType]}
            <div class="p-4 border-2 border-dashed border-blue-400 dark:border-blue-500 rounded-2xl bg-blue-50 dark:bg-blue-900/20 space-y-3">
                <p class="text-sm font-semibold text-blue-800 dark:text-blue-300">{labels.title}</p>

                {#if showRejectionBox}
                    <textarea
                        bind:value={rejectionReason}
                        placeholder="What should be done differently? This is sent to the agent."
                        rows="3"
                        class="w-full text-sm p-2 rounded-lg border border-blue-300 dark:border-blue-600 bg-white dark:bg-gray-800 dark:text-gray-100"
                    ></textarea>
                {/if}

                <div class="flex gap-2">
                    <button
                        on:click={approve}
                        class="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 shadow-md transition-all active:scale-95"
                    >
                        {labels.approve}
                    </button>
                    {#if showRejectionBox}
                        <button
                            on:click={reject}
                            class="px-4 py-2 bg-white dark:bg-gray-800 text-blue-600 dark:text-blue-400 border border-blue-600 dark:border-blue-400 rounded-lg text-sm hover:bg-blue-50 dark:hover:bg-blue-900/30 transition-all active:scale-95"
                        >
                            Send Feedback
                        </button>
                    {:else}
                        <button
                            on:click={() => (showRejectionBox = true)}
                            class="px-4 py-2 bg-white dark:bg-gray-800 text-blue-600 dark:text-blue-400 border border-blue-600 dark:border-blue-400 rounded-lg text-sm hover:bg-blue-50 dark:hover:bg-blue-900/30 transition-all active:scale-95"
                        >
                            {labels.reject}
                        </button>
                    {/if}
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

.markdown-body :global(p) {
        margin-bottom: 0.5rem;
    }
    .markdown-body :global(p:last-child) {
        margin-bottom: 0;
    }
    .markdown-body :global(ul),
    .markdown-body :global(ol) {
        margin: 0.25rem 0 0.5rem 1.25rem;
        list-style: revert;
    }
    .markdown-body :global(code) {
        background: rgba(127, 127, 127, 0.18);
        padding: 0.05rem 0.25rem;
        border-radius: 0.25rem;
        font-size: 0.85em;
    }
    .markdown-body :global(pre) {
        background: rgba(127, 127, 127, 0.15);
        padding: 0.6rem;
        border-radius: 0.5rem;
        overflow-x: auto;
        margin: 0.5rem 0;
    }
    .markdown-body :global(pre code) {
        background: none;
        padding: 0;
    }
    .markdown-body :global(a) {
        text-decoration: underline;
    }
</style>
