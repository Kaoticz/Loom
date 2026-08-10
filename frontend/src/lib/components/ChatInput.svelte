<script lang="ts">
    import type { TaskStatus } from '$lib/stores/agent';
    import { canSubmitPrompt, isWorking, promptPlaceholder } from '$lib/chatPresentation';

    let {
        prompt = $bindable(''),
        status,
        pendingAction = null,
        answeringQuestion = false,
        focusRequest = 0,
        onsend,
        onstop
    }: {
        prompt?: string;
        status: TaskStatus;
        pendingAction?: string | null;
        /** Whether the agent is waiting on an answer, which the box now sends. */
        answeringQuestion?: boolean;
        /**
         * Bumped to ask for focus.
         *
         * A number rather than a method, so the box stays prop-driven: choosing
         * the free-reply option has to put the cursor where the reply is typed,
         * and the panel offering that option cannot reach in here itself.
         */
        focusRequest?: number;
        onsend: (prompt: string) => void;
        onstop: () => void;
    } = $props();

    const working = $derived(isWorking(status));
    const canSend = $derived(canSubmitPrompt(status, pendingAction, answeringQuestion));
    const placeholder = $derived(promptPlaceholder(status, pendingAction, answeringQuestion));

    let box: HTMLTextAreaElement | undefined = $state();

    $effect(() => {
        if (focusRequest > 0) box?.focus();
    });

    function send() {
        if (!canSend || !prompt.trim()) return;
        onsend(prompt);
        prompt = '';
    }

    function handleKeydown(event: KeyboardEvent) {
        if (event.key !== 'Enter' || event.shiftKey) return;
        event.preventDefault();
        send();
    }
</script>

<div class="border-t border-gray-200 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-800/50">
    <div
        class="flex gap-2 rounded-xl border border-gray-300 bg-white p-2 shadow-inner transition-all focus-within:ring-2 focus-within:ring-blue-500 dark:border-gray-600 dark:bg-gray-700"
    >
        <textarea
            bind:this={box}
            bind:value={prompt}
            onkeydown={handleKeydown}
            {placeholder}
            aria-label="Task prompt"
            class="h-20 flex-1 resize-none bg-transparent p-2 text-sm outline-none dark:text-white"
        ></textarea>
        {#if working}
            <button
                onclick={onstop}
                class="self-end rounded-lg bg-red-600 p-3 text-white shadow-md transition-all hover:bg-red-700 active:scale-95"
                title="Stop the agent"
                aria-label="Stop the agent"
            >
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    class="h-5 w-5"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                    aria-hidden="true"
                >
                    <rect x="6" y="6" width="12" height="12" rx="1.5" />
                </svg>
            </button>
        {:else}
            <button
                onclick={send}
                disabled={!canSend}
                class="self-end rounded-lg bg-blue-600 p-3 text-white shadow-md transition-all hover:bg-blue-700 active:scale-95 disabled:cursor-not-allowed disabled:bg-gray-400 disabled:hover:bg-gray-400"
                title={canSend ? 'Send Message' : 'Answer the question below first'}
                aria-label="Send Message"
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
                    aria-hidden="true"
                >
                    <line x1="22" y1="2" x2="11" y2="13"></line>
                    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                </svg>
            </button>
        {/if}
    </div>

    {#if working}
        <!-- While working the slot holds Stop, so there is no Send control to
             show as disabled. Without this, pressing Enter would do nothing with
             nothing to look at. -->
        <p class="mt-2 text-xs text-gray-500 dark:text-gray-400">
            The agent is working. Stop it to send this.
        </p>
    {/if}
</div>
