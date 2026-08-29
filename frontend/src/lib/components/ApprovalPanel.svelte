<script lang="ts">
    import { actionLabels } from '$lib/chatPresentation';

    let {
        actionType,
        reason = '',
        onapprove,
        onreject,
        onrework = () => {},
        onoverride = () => {}
    }: {
        actionType: string;
        reason?: string;
        onapprove: () => void;
        onreject: (reason: string) => void;
        onrework?: () => void;
        onoverride?: () => void;
    } = $props();

    const labels = $derived(actionLabels[actionType]);

    /**
     * Whether a reason has been composed, which is what rejecting requires.
     *
     * Whitespace does not count: a reason made of spaces tells the agent as
     * little as no reason at all, and it is the agent's only account of what
     * was wrong.
     */
    const canReject = $derived(reason.trim().length > 0);

    /**
     * Whether the reject control is currently explaining itself.
     *
     * Held in state rather than left to the native `title` attribute, which
     * browsers delay by around a second and will not be told otherwise. The
     * requirement is that the reason appears the moment the pointer arrives,
     * and focus is offered the same courtesy for anyone not using a pointer.
     */
    let explaining = $state(false);
    let confirmingOverride = $state(false);

    /**
     * Reject with the composed reason, or refuse if there is none.
     *
     * The guard is what makes the control unavailable, since the button is only
     * marked disabled rather than being disabled outright.
     */
    function reject(): void {
        if (!canReject) return;
        onreject(reason.trim());
    }

    /** Reveal the irreversible reviewer-override confirmation. */
    function requestOverride(): void {
        confirmingOverride = true;
    }

    /** Merge the reviewed branch despite its rejected verdict. */
    function confirmOverride(): void {
        onoverride();
    }

    // Unique per panel, so the button's description points at this panel's own
    // explanation rather than one left in the document by another.
    const explanationId = $props.id();
</script>

<div
    class="space-y-3 rounded-2xl border-2 border-dashed border-blue-400 bg-blue-50 p-4 dark:border-blue-500 dark:bg-blue-900/20"
>
    <p class="text-sm font-semibold text-blue-800 dark:text-blue-300">{labels.title}</p>

    <!--
        Always present, never revealed by an interaction. This is the half of
        requirement 3.2 that serves keyboard and touch users, who never hover,
        and it is why the disabled button carries no other explanation of
        itself.
    -->
    {#if actionType !== 'review_rejected'}
        <p id={explanationId} class="text-xs text-blue-700 dark:text-blue-400">
            {#if canReject}
                Rejecting will send what you have typed above as the reason.
            {:else}
                To reject, type a reason in the box below first. It is all the agent will be told
                about what was wrong.
            {/if}
        </p>
    {/if}

    <div class="flex gap-2">
        <button
            onclick={actionType === 'review_rejected' ? onrework : onapprove}
            class="rounded-lg bg-blue-600 px-4 py-2 text-sm text-white shadow-md transition-all hover:bg-blue-700 active:scale-95"
        >
            {labels.approve}
        </button>

        {#if actionType === 'review_rejected'}
            <button
                onclick={requestOverride}
                class="rounded-lg border border-red-600 bg-white px-4 py-2 text-sm text-red-700 transition-all hover:bg-red-50 active:scale-95 dark:border-red-400 dark:bg-gray-800 dark:text-red-300 dark:hover:bg-red-950/30"
            >
                {labels.reject}
            </button>
        {:else}
            <!--
            Marked disabled to assistive technology rather than with the
            `disabled` attribute. A truly disabled button emits no pointer
            events in most browsers and drops out of the tab order, so the
            reason for it being unavailable could be reached by neither hover
            nor keyboard — which is precisely what requirement 3.2 is about.
            The click is refused in the handler instead.
        -->
            <span class="relative inline-block">
                <button
                    onclick={reject}
                    onmouseenter={() => (explaining = !canReject)}
                    onmouseleave={() => (explaining = false)}
                    onfocus={() => (explaining = !canReject)}
                    onblur={() => (explaining = false)}
                    aria-disabled={!canReject}
                    aria-describedby={explanationId}
                    class="rounded-lg border border-blue-600 bg-white px-4 py-2 text-sm text-blue-600 transition-all dark:border-blue-400 dark:bg-gray-800 dark:text-blue-400 {canReject
                        ? 'hover:bg-blue-50 active:scale-95 dark:hover:bg-blue-900/30'
                        : 'cursor-not-allowed opacity-50'}"
                >
                    {labels.reject}
                </button>

                {#if explaining}
                    <!--
                    Hidden from assistive technology on purpose: it repeats the
                    explanation above, which the button already points at, and
                    hearing the same sentence twice is worse than once.
                -->
                    <span
                        role="tooltip"
                        aria-hidden="true"
                        data-testid="reject-tooltip"
                        class="absolute bottom-full left-1/2 z-10 mb-2 w-56 -translate-x-1/2 rounded-lg bg-gray-900 px-3 py-2 text-xs text-white shadow-lg dark:bg-gray-700"
                    >
                        Type a reason in the box below first.
                    </span>
                {/if}
            </span>
        {/if}
    </div>

    {#if confirmingOverride}
        <div
            role="alert"
            class="space-y-3 rounded-lg border border-amber-400 bg-amber-50 p-3 text-sm text-amber-950 dark:border-amber-500 dark:bg-amber-950/30 dark:text-amber-100"
        >
            <p>
                The current changes will be merged into Main, and the task will be marked as
                complete.
            </p>
            <div class="flex gap-2">
                <button
                    onclick={confirmOverride}
                    class="rounded-lg bg-red-700 px-3 py-2 text-sm text-white transition-colors hover:bg-red-800"
                >
                    Merge into Main
                </button>
                <button
                    onclick={() => (confirmingOverride = false)}
                    class="rounded-lg border border-amber-700 px-3 py-2 text-sm text-amber-950 transition-colors hover:bg-amber-100 dark:border-amber-300 dark:text-amber-100 dark:hover:bg-amber-900/40"
                >
                    Cancel
                </button>
            </div>
        </div>
    {/if}
</div>
