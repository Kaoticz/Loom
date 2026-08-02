<script lang="ts">
    import type { Message } from '$lib/stores/agent';
    import { onDestroy } from 'svelte';
    import {
        COPYABLE_TYPES,
        MARKDOWN_TYPES,
        formatMessageTime,
        typeStyles
    } from '$lib/chatPresentation';
    import { COPY_FEEDBACK_MS, type CopyState, copyGlyph, copyText } from '$lib/clipboard';
    import { renderMarkdown } from '$lib/markdown';

    let { message }: { message: Message } = $props();

    const copyable = $derived(COPYABLE_TYPES.includes(message.type));

    let copyState: CopyState = $state('idle');
    let resetTimer: ReturnType<typeof setTimeout> | undefined;

    const copyLabel = $derived(copyGlyph(copyState));

    async function copy() {
        copyState = (await copyText(message.content)) ? 'copied' : 'failed';

        // The result is a moment's reassurance, not a permanent change of state.
        clearTimeout(resetTimer);
        resetTimer = setTimeout(() => (copyState = 'idle'), COPY_FEEDBACK_MS);
    }

    onDestroy(() => clearTimeout(resetTimer));
</script>

<div
    class="max-w-[90%] rounded-2xl border p-3 {typeStyles[message.type] ||
        'border-gray-200 bg-gray-100 dark:border-gray-600 dark:bg-gray-700'} transform shadow-sm transition-all hover:shadow-md"
>
    <div class="mb-1 flex items-baseline justify-between gap-3 text-[10px] opacity-60">
        <span class="font-bold uppercase tracking-widest">{message.type}</span>
        <time
            datetime={message.timestamp.toISOString()}
            title={message.timestamp.toLocaleString()}
            class="shrink-0 font-mono tabular-nums"
        >
            {formatMessageTime(message.timestamp)}
        </time>
    </div>
    {#if MARKDOWN_TYPES.includes(message.type)}
        <div class="markdown-body text-sm leading-relaxed">
            <!-- renderMarkdown runs its output through DOMPurify, so the markup
                 inserted here has already been sanitised. -->
            <!-- eslint-disable-next-line svelte/no-at-html-tags -->
            {@html renderMarkdown(message.content)}
        </div>
    {:else}
        <div class="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</div>
    {/if}

    {#if copyable}
        <div class="mt-2 flex items-center gap-2">
            <button
                type="button"
                onclick={copy}
                aria-label="Copy message text"
                title="Copy message text"
                class="rounded px-1.5 py-0.5 text-xs leading-none opacity-60 transition-opacity hover:opacity-100"
            >
                {copyLabel}
            </button>
            <!-- Announced as well as shown: the glyph alone conveys nothing to a
                 reader who cannot see it change. -->
            <span role="status" class="text-[10px] opacity-70">
                {#if copyState === 'copied'}
                    Copied
                {:else if copyState === 'failed'}
                    Copy failed
                {/if}
            </span>
        </div>
    {/if}
</div>

<style>
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
