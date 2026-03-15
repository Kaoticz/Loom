<script lang="ts">
    import { createEventDispatcher } from 'svelte';

    const dispatch = createEventDispatcher();

    export let dockerfile: string = 'FROM python:3.14-slim\n\nWORKDIR /workspace\n';
    export let secrets: { key: string; value: string }[] = [];

    function addSecret() {
        secrets = [...secrets, { key: '', value: '' }];
    }

    function removeSecret(index: number) {
        secrets = secrets.filter((_, i) => i !== index);
    }

    function save() {
        dispatch('save', { dockerfile, secrets });
    }
</script>

<div class="p-6 bg-white rounded-lg shadow-xl max-w-2xl mx-auto">
    <h2 class="text-2xl font-bold mb-6">Environment Configuration</h2>

    <div class="mb-6">
        <label class="block text-sm font-medium text-gray-700 mb-2">Dockerfile</label>
        <textarea
            bind:value={dockerfile}
            class="w-full h-48 p-3 font-mono text-sm border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            placeholder="FROM python:3.14-slim..."
        ></textarea>
    </div>

    <div class="mb-6">
        <div class="flex justify-between items-center mb-2">
            <label class="block text-sm font-medium text-gray-700">Secrets / Environment Variables</label>
            <button
                on:click={addSecret}
                class="text-sm bg-blue-500 text-white px-2 py-1 rounded hover:bg-blue-600 transition"
            >
                + Add Secret
            </button>
        </div>
        
        <div class="space-y-3">
            {#each secrets as secret, i}
                <div class="flex gap-2">
                    <input
                        type="text"
                        bind:value={secret.key}
                        placeholder="KEY"
                        class="flex-1 p-2 border border-gray-300 rounded-md text-sm"
                    />
                    <input
                        type="password"
                        bind:value={secret.value}
                        placeholder="VALUE"
                        class="flex-1 p-2 border border-gray-300 rounded-md text-sm"
                    />
                    <button
                        on:click={() => removeSecret(i)}
                        class="text-red-500 hover:text-red-700 px-2"
                    >
                        &times;
                    </button>
                </div>
            {/each}
        </div>
    </div>

    <div class="flex justify-end gap-3">
        <button
            on:click={() => dispatch('close')}
            class="px-4 py-2 text-gray-600 hover:text-gray-800"
        >
            Cancel
        </button>
        <button
            on:click={save}
            class="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
        >
            Save Configuration
        </button>
    </div>
</div>
