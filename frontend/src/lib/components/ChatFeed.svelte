<script lang="ts">
    import type { BlockedReason, Message, PendingQuestion, TaskStatus } from '$lib/stores/agent';
    import { isAtBottom } from '$lib/scrolling';
    import ApprovalPanel from './ApprovalPanel.svelte';
    import BlockedNotice from './BlockedNotice.svelte';
    import ChatMessage from './ChatMessage.svelte';
    import MergedNotice from './MergedNotice.svelte';
    import QuestionPanel from './QuestionPanel.svelte';

    let {
        messages,
        blocked = null,
        actionType = null,
        status = 'idle',
        reason = '',
        question = null,
        onapprove,
        onreject,
        onretry,
        onanswer,
        onfreereply
    }: {
        messages: Message[];
        blocked?: BlockedReason;
        /** The question the agent is waiting on, or null if it is not. */
        question?: PendingQuestion | null;
        actionType?: string | null;
        status?: TaskStatus;
        /** The text being composed, which the approval panel offers as a reason. */
        reason?: string;
        onapprove: () => void;
        onreject: (reason: string) => void;
        onretry: () => void;
        onanswer: (answer: string) => void;
        onfreereply: () => void;
    } = $props();

    let container: HTMLElement | undefined = $state();

    /**
     * Whether the feed was at the bottom just before the newest message landed.
     *
     * Deliberately a plain variable rather than state: it is a measurement
     * handed from the pre-update pass to the post-update one, not something the
     * interface renders.
     */
    let wasAtBottom = true;

    // The decision has to be taken before the new message is in the DOM. Once it
    // has been added, `scrollHeight` has already grown and a reader who was at
    // the bottom would measure as scrolled up, so the feed would never follow.
    $effect.pre(() => {
        // Referenced so this runs for each new message, not only on mount.
        void messages;
        wasAtBottom = container ? isAtBottom(container) : true;
    });

    // Requirement 4.4: measured afresh every time, so scrolling back to the
    // bottom resumes following without any flag needing to be reset.
    //
    // The follow must not animate. This element sets `scroll-behavior: smooth`,
    // so assigning `scrollTop` starts an animation instead of moving the feed,
    // and `scrollTop` reports an intermediate position until that animation
    // finishes. The next message's `$effect.pre` would read that intermediate
    // position, conclude the reader had scrolled away, and stop following for
    // good. That is not hypothetical: with messages arriving faster than the
    // animation completed, the feed latched off after the first one and sat
    // thousands of pixels from the bottom for the rest of the session.
    //
    // `behavior: 'instant'` overrides the stylesheet and moves the element
    // synchronously, so the next measurement sees where the feed really is.
    $effect(() => {
        void messages;
        if (wasAtBottom && container) {
            container.scrollTo({ top: container.scrollHeight, behavior: 'instant' });
        }
    });
</script>

<div
    bind:this={container}
    data-testid="chat-feed"
    class="flex flex-1 flex-col space-y-4 overflow-y-auto scroll-smooth p-4"
>
    {#each messages as message (message.id)}
        <ChatMessage {message} />
    {/each}

    {#if blocked}
        <BlockedNotice reason={blocked} {onretry} />
    {/if}

    {#if actionType}
        <ApprovalPanel {actionType} {reason} {onapprove} {onreject} />
    {/if}

    {#if status === 'merged' && !actionType}
        <MergedNotice />
    {/if}

    {#if question}
        <QuestionPanel
            question={question.question}
            options={question.options}
            {onanswer}
            {onfreereply}
        />
    {/if}
</div>
