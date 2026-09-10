<script lang="ts">
    import {
        DEFAULT_CHUNK_LINES,
        MAX_CHUNK_LINES,
        MIN_CHUNK_LINES,
        readChunkLines
    } from '$lib/chunkSize';
    import { modelDescriptionSchema, parseResponse } from '$lib/schemas';
    import type { ModelDescription } from '$lib/schemas';
    import type { AgentConfig } from '$lib/stores/agent';
    import ContextWindowControls from './ContextWindowControls.svelte';

    type ModelSettings = Pick<
        AgentConfig,
        | 'coderModel'
        | 'reviewerModel'
        | 'embeddingModel'
        | 'coderContextWindow'
        | 'plannerContextWindow'
        | 'chunkLines'
    >;

    let {
        apiBase,
        availableModels,
        modelsError,
        settings,
        onchange
    }: {
        apiBase: string;
        availableModels: string[];
        modelsError: string;
        settings: ModelSettings;
        onchange: (patch: Partial<ModelSettings>) => void;
    } = $props();

    let described: ModelDescription | null = $state(null);

    async function describeEmbeddingModel(name: string): Promise<void> {
        described = null;
        if (!name) return;
        try {
            const response = await fetch(
                `${apiBase}/api/v1/ollama/models/${encodeURIComponent(name)}/description`
            );
            described = parseResponse(
                modelDescriptionSchema,
                await response.json(),
                'model description'
            );
        } catch (error) {
            console.error('Failed to describe the embedding model', error);
            described = { dimensions: null, context_length: null };
        }
    }

    $effect(() => {
        void describeEmbeddingModel(settings.embeddingModel);
    });
</script>

<section class="rounded-lg bg-white p-6 shadow dark:bg-gray-800">
    <h2 class="mb-4 text-xl font-semibold text-gray-900 dark:text-white">AI Models</h2>
    {#if modelsError}
        <p
            role="status"
            data-testid="models-error"
            class="mb-4 rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-300"
        >
            {modelsError}
        </p>
    {/if}
    <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div>
            <label
                for="coderModel"
                class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
                >Coder Model</label
            >
            <select
                id="coderModel"
                value={settings.coderModel}
                onchange={(event) => onchange({ coderModel: event.currentTarget.value })}
                disabled={availableModels.length === 0}
                class="w-full rounded border border-gray-300 bg-white p-2 text-gray-900 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
            >
                {#if availableModels.length === 0}<option>No models available</option
                    >{:else}{#each availableModels as model}<option value="ollama:{model}"
                            >{model}</option
                        >{/each}{/if}
            </select>
        </div>
        <div>
            <label
                for="reviewerModel"
                class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
                >Reviewer Model</label
            >
            <select
                id="reviewerModel"
                value={settings.reviewerModel}
                onchange={(event) => onchange({ reviewerModel: event.currentTarget.value })}
                disabled={availableModels.length === 0}
                class="w-full rounded border border-gray-300 bg-white p-2 text-gray-900 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
            >
                {#if availableModels.length === 0}<option>No models available</option
                    >{:else}{#each availableModels as model}<option value="ollama:{model}"
                            >{model}</option
                        >{/each}{/if}
            </select>
        </div>
        <ContextWindowControls {apiBase} {settings} {onchange} />
        <div>
            <label
                for="embeddingModel"
                class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
                >Embedding Model</label
            >
            <p class="mb-1 text-xs text-gray-500 dark:text-gray-400">
                Must be an embedding-capable model (e.g. <code>nomic-embed-text</code>,
                <code>mxbai-embed-large</code>). Chat models such as <code>llama3</code> will not produce
                usable embeddings.
            </p>
            <select
                id="embeddingModel"
                value={settings.embeddingModel}
                onchange={(event) => onchange({ embeddingModel: event.currentTarget.value })}
                disabled={availableModels.length === 0}
                class="w-full rounded border border-gray-300 bg-white p-2 text-gray-900 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
            >
                {#if availableModels.length === 0}<option>No models available</option
                    >{:else}{#each availableModels as model}<option value={model}>{model}</option
                        >{/each}{/if}
            </select>
            <p
                data-testid="embedding-model-description"
                class="mt-1 text-xs text-gray-500 dark:text-gray-400"
            >
                {#if described === null}Reading what this model reports about itself...{:else if described.dimensions === null && described.context_length === null}This
                    model does not report its vector width or capacity.{:else}{described.dimensions ??
                        'An unreported number of'} dimensions, {described.context_length ??
                        'an unreported'} token capacity.{/if}
            </p>
        </div>
        <div>
            <label
                for="chunkLines"
                class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
                >Chunk Size</label
            >
            <p class="mb-1 text-xs text-gray-500 dark:text-gray-400">
                How many lines of code each embedding covers. Larger chunks carry more context but
                match less precisely. Leave on Auto unless you are tuning search quality: Auto
                shrinks any chunk the model refuses, while a set size is used exactly and indexing
                fails if the model will not take it. Changing this re-indexes the repository.
            </p>
            <div class="flex items-center gap-3">
                <label class="flex items-center gap-2 text-sm dark:text-gray-300"
                    ><input
                        type="checkbox"
                        aria-label="Automatic chunk size"
                        checked={settings.chunkLines === null}
                        onchange={(event) =>
                            onchange({
                                chunkLines: event.currentTarget.checked ? null : DEFAULT_CHUNK_LINES
                            })}
                    /> Auto</label
                >
                <input
                    id="chunkLines"
                    type="number"
                    min={MIN_CHUNK_LINES}
                    max={MAX_CHUNK_LINES}
                    step="10"
                    disabled={settings.chunkLines === null}
                    value={settings.chunkLines ?? DEFAULT_CHUNK_LINES}
                    oninput={(event) =>
                        onchange({ chunkLines: readChunkLines(event.currentTarget.value) })}
                    class="w-32 rounded border border-gray-300 bg-white p-2 text-gray-900 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                />
                <span class="text-xs text-gray-500 dark:text-gray-400">lines</span>
            </div>
        </div>
    </div>
</section>
