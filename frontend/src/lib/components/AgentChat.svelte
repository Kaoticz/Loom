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

    /** The question the agent is waiting on, or null when it is not waiting. */
    const question = $derived($agentStore.pendingQuestion);

    /** Bumped when the reader chooses to type their own reply. */
    let focusRequest = $state(0);

    /**
     * Send what is in the box, as a reply if one is being waited on.
     *
     * The box is the only text input in the interface, so what it means depends
     * on what the session is waiting for. Deciding that here keeps it in the one
     * component that reads the store.
     *
     * @param text - What the reader typed.
     */
    function submit(text: string) {
        if (question !== null) {
            agentStore.answerQuestion(text);
            return;
        }
        agentStore.sendPrompt(text);
    }

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
        status={$agentStore.status}
        reason={promptText}
        onapprove={approve}
        onreject={reject}
        onretry={() => agentStore.retryConnection()}
        {question}
        onanswer={(answer) => agentStore.answerQuestion(answer)}
        onfreereply={() => (focusRequest += 1)}
    />
    <ChatInput
        status={$agentStore.status}
        {pendingAction}
        answeringQuestion={question !== null}
        {focusRequest}
        bind:prompt={promptText}
        onsend={submit}
        onstop={() => agentStore.stop()}
    />
</div>
