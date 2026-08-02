<script lang="ts">
    import { agentStore } from '$lib/stores/agent';
    import ChatFeed from './ChatFeed.svelte';
    import ChatHeader from './ChatHeader.svelte';
    import ChatInput from './ChatInput.svelte';

    /**
     * What the reader has typed.
     *
     * Owned here rather than by the input, because it doubles as the reason for
     * rejecting a plan and the approval panel has to be able to read it.
     */
    let promptText = $state('');

    const pendingAction = $derived($agentStore.requiresAction ? $agentStore.actionType : null);

    function approve() {
        agentStore.approve(true);
        promptText = '';
    }

    /**
     * Reject with the reason the panel gathered.
     *
     * The reason is taken from the callback rather than read back out of
     * `promptText`, so there is one path from the box to the agent. Two would
     * agree today and diverge the first time either changed.
     */
    function reject(reason: string) {
        agentStore.approve(false, reason);
        promptText = '';
    }
</script>

<div
    class="flex h-full flex-col border-r border-gray-200 bg-white transition-colors duration-200 dark:border-gray-700 dark:bg-gray-800"
>
    <ChatHeader connection={$agentStore.connection} status={$agentStore.status} />
    <ChatFeed
        messages={$agentStore.messages}
        blocked={$agentStore.blocked}
        actionType={pendingAction}
        reason={promptText}
        onapprove={approve}
        onreject={reject}
        onretry={() => agentStore.retryConnection()}
    />
    <ChatInput
        status={$agentStore.status}
        {pendingAction}
        bind:prompt={promptText}
        onsend={(prompt) => agentStore.sendPrompt(prompt)}
        onstop={() => agentStore.stop()}
    />
</div>
