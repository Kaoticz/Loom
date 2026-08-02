<script lang="ts">
    import type { ConnectionState, TaskStatus } from '$lib/stores/agent';
    import { connectionStyles } from '$lib/chatPresentation';

    let {
        connection,
        status
    }: {
        connection: ConnectionState;
        status: TaskStatus;
    } = $props();
</script>

<div
    class="flex items-center justify-between border-b border-gray-200 p-4 shadow-sm dark:border-gray-700"
>
    <div class="flex items-center gap-2">
        <h2 class="text-lg font-bold dark:text-white">Agent Chat</h2>
        <span
            class="h-2 w-2 rounded-full {connectionStyles[connection]}"
            title="Connection: {connection}"
        ></span>
        <span class="text-xs text-gray-500 dark:text-gray-400">{connection}</span>
    </div>
    <span
        class="rounded bg-gray-100 px-2 py-1 text-xs font-semibold uppercase tracking-wider text-gray-600 dark:bg-gray-700 dark:text-gray-300"
    >
        {status.replaceAll('_', ' ')}
    </span>
</div>
