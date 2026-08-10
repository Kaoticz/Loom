<script lang="ts">
    let {
        question,
        options = [],
        onanswer,
        onfreereply
    }: {
        question: string;
        options?: string[];
        onanswer: (answer: string) => void;
        onfreereply: () => void;
    } = $props();
</script>

<div
    class="space-y-3 rounded-2xl border-2 border-dashed border-teal-400 bg-teal-50 p-4 dark:border-teal-500 dark:bg-teal-900/20"
    data-testid="question-panel"
>
    <p class="text-sm font-semibold text-teal-900 dark:text-teal-200">{question}</p>

    <div class="flex flex-col gap-2">
        {#each options as option (option)}
            <button
                onclick={() => onanswer(option)}
                class="rounded-lg border border-teal-600 bg-white px-4 py-2 text-left text-sm text-teal-800 transition-all hover:bg-teal-100 active:scale-95 dark:border-teal-400 dark:bg-gray-800 dark:text-teal-200 dark:hover:bg-teal-900/30"
            >
                {option}
            </button>
        {/each}

        <!--
            Always present, whatever the agent supplied and whether it supplied
            anything at all. The agent neither adds this nor can remove it: a
            question with no way out but the answers someone else thought of is
            not a question.
        -->
        <button
            onclick={onfreereply}
            data-testid="free-reply-option"
            class="rounded-lg border border-dashed border-teal-500 bg-transparent px-4 py-2 text-left text-sm text-teal-700 transition-all hover:bg-teal-100 active:scale-95 dark:border-teal-400 dark:text-teal-300 dark:hover:bg-teal-900/30"
        >
            Something else — type my own reply
        </button>
    </div>
</div>
