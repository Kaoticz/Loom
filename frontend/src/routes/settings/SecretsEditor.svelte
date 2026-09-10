<script lang="ts">
    import type { AgentConfig } from '$lib/stores/agent';

    type Secret = AgentConfig['secrets'][number];

    let {
        secrets,
        onchange
    }: {
        secrets: Secret[];
        onchange: (secrets: Secret[]) => void;
    } = $props();

    function addSecret(): void {
        onchange([...secrets, { key: '', value: '' }]);
    }

    function removeSecret(index: number): void {
        onchange(secrets.filter((_, currentIndex) => currentIndex !== index));
    }

    function updateSecret(index: number, patch: Partial<Secret>): void {
        onchange(
            secrets.map((secret, currentIndex) =>
                currentIndex === index ? { ...secret, ...patch } : secret
            )
        );
    }
</script>

<div role="group" aria-labelledby="secrets-heading">
    <div class="mb-2 flex items-center justify-between">
        <span
            id="secrets-heading"
            class="block text-sm font-medium text-gray-700 dark:text-gray-300"
            >Secrets / Env Vars</span
        >
        <button
            onclick={addSecret}
            class="rounded bg-blue-600 px-2 py-1 text-sm text-white hover:bg-blue-700"
        >
            + Add
        </button>
    </div>
    <div class="space-y-2">
        {#each secrets as secret, index}
            <div class="flex gap-2">
                <input
                    type="text"
                    value={secret.key}
                    oninput={(event) => updateSecret(index, { key: event.currentTarget.value })}
                    placeholder="KEY"
                    aria-label="Secret {index + 1} name"
                    class="flex-1 rounded border border-gray-300 bg-white p-2 text-gray-900 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                />
                <input
                    type="password"
                    value={secret.value}
                    oninput={(event) => updateSecret(index, { value: event.currentTarget.value })}
                    placeholder="VALUE"
                    aria-label="Secret {index + 1} value"
                    class="flex-1 rounded border border-gray-300 bg-white p-2 text-gray-900 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                />
                <button
                    onclick={() => removeSecret(index)}
                    aria-label="Remove secret {index + 1}"
                    class="px-2 text-red-500 hover:text-red-700"
                >
                    &times;
                </button>
            </div>
        {/each}
    </div>
</div>
