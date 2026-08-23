<script lang="ts">
    import type { PlanStep } from '$lib/schemas';

    interface Props {
        steps: PlanStep[];
    }

    const { steps }: Props = $props();

    /**
     * Which steps are open, by index.
     *
     * Collapsed by default: the value of the list is that a whole plan can be
     * taken in at once, which is lost if every description is showing.
     */
    let open = $state(new Set<number>());

    /**
     * Show or hide one step's description.
     *
     * @param index Which step to toggle.
     */
    function toggle(index: number) {
        // Reassigned rather than mutated: a Set changed in place is the same
        // object, and nothing would re-render.
        const next = new Set(open);
        if (next.has(index)) {
            next.delete(index);
        } else {
            next.add(index);
        }
        open = next;
    }
</script>

<ol class="space-y-1" data-testid="plan-steps">
    {#each steps as step, index (index)}
        <li class="rounded border border-gray-200 dark:border-gray-700">
            {#if step.description}
                <!--
                    A real button, so it is reachable by keyboard and announces
                    its state. `aria-expanded` is the state; the chevron is
                    decoration and is hidden from assistive technology.
                -->
                <button
                    type="button"
                    onclick={() => toggle(index)}
                    aria-expanded={open.has(index)}
                    aria-controls={`plan-step-${index}`}
                    class="flex w-full items-start gap-2 px-3 py-2 text-left text-sm hover:bg-gray-50 dark:hover:bg-gray-700/50"
                >
                    <span class="font-mono text-xs text-gray-500 dark:text-gray-400"
                        >{index + 1}.</span
                    >
                    <span class="flex-1 text-gray-900 dark:text-gray-100">{step.title}</span>
                    <span aria-hidden="true" class="text-gray-400">
                        {open.has(index) ? '−' : '+'}
                    </span>
                </button>

                {#if open.has(index)}
                    <div
                        id={`plan-step-${index}`}
                        class="whitespace-pre-wrap border-t border-gray-200 px-3 py-2 pl-8 text-sm text-gray-700 dark:border-gray-700 dark:text-gray-300"
                    >
                        {step.description}
                    </div>
                {/if}
            {:else}
                <!--
                    No description, so nothing to expand. Rendered as a plain row
                    rather than a control that opens onto nothing — a button
                    promising content it does not have is worse than no button.
                -->
                <div class="flex items-start gap-2 px-3 py-2 text-sm">
                    <span class="font-mono text-xs text-gray-500 dark:text-gray-400"
                        >{index + 1}.</span
                    >
                    <span class="flex-1 text-gray-900 dark:text-gray-100">{step.title}</span>
                </div>
            {/if}
        </li>
    {/each}
</ol>
