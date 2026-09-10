<script lang="ts">
    import {
        DEFAULT_CONTEXT_WINDOW,
        MAX_CONTEXT_WINDOW,
        MIN_CONTEXT_WINDOW,
        readContextWindow
    } from '$lib/contextWindow';
    import { fetchContextWindowDetection } from '$lib/contextWindowDetection';
    import type { ContextWindowDetection } from '$lib/schemas';
    import type { AgentConfig } from '$lib/stores/agent';

    type ContextWindowSettings = Pick<
        AgentConfig,
        'coderModel' | 'coderContextWindow' | 'plannerContextWindow'
    >;

    let {
        apiBase,
        settings,
        onchange
    }: {
        apiBase: string;
        settings: ContextWindowSettings;
        onchange: (patch: Partial<ContextWindowSettings>) => void;
    } = $props();

    let contextWindowDetection: ContextWindowDetection | null = $state(null);
    let contextWindowDetectionRequest = 0;

    const automaticContextWindow = $derived.by(() => {
        const detection = contextWindowDetection;
        if (detection?.status !== 'detected' || detection.context_window === null) {
            return DEFAULT_CONTEXT_WINDOW;
        }
        return detection.context_window;
    });

    async function detectContextWindow(model: string): Promise<void> {
        const request = ++contextWindowDetectionRequest;
        if (!model.startsWith('ollama:')) {
            contextWindowDetection = { context_window: null, status: 'not_applicable' };
            return;
        }

        contextWindowDetection = null;
        try {
            const detection = await fetchContextWindowDetection(apiBase, { model });
            if (request !== contextWindowDetectionRequest) return;
            contextWindowDetection = detection;
        } catch (error) {
            if (request !== contextWindowDetectionRequest) return;
            console.error('Failed to detect the context window', error);
            contextWindowDetection = { context_window: null, status: 'failed' };
        }
    }

    $effect(() => {
        void detectContextWindow(settings.coderModel);
    });
</script>

<div>
    <label
        for="coderContextWindow"
        class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
        >Coder Context Window</label
    >
    <p class="mb-1 text-xs text-gray-500 dark:text-gray-400">
        How many tokens of history the coder may hold before older tool results are trimmed. Set
        this to your model's actual window. Too high and the run overflows mid-task; too low only
        trims sooner than needed.
    </p>
    <div class="flex items-center gap-3">
        <label class="flex items-center gap-2 text-sm dark:text-gray-300"
            ><input
                type="checkbox"
                aria-label="Automatic coder context window"
                checked={settings.coderContextWindow === null}
                onchange={(event) =>
                    onchange({
                        coderContextWindow: event.currentTarget.checked
                            ? null
                            : automaticContextWindow
                    })}
            /> Auto</label
        >
        <input
            id="coderContextWindow"
            type="number"
            min={MIN_CONTEXT_WINDOW}
            max={MAX_CONTEXT_WINDOW}
            step="1024"
            disabled={settings.coderContextWindow === null}
            value={settings.coderContextWindow ?? automaticContextWindow}
            oninput={(event) =>
                onchange({ coderContextWindow: readContextWindow(event.currentTarget.value) })}
            class="w-32 rounded border border-gray-300 bg-white p-2 text-gray-900 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
        />
        <span class="text-xs text-gray-500 dark:text-gray-400">tokens</span>
    </div>
</div>

<div>
    <label
        for="plannerContextWindow"
        class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
        >Planner Context Window</label
    >
    <p class="mb-1 text-xs text-gray-500 dark:text-gray-400">
        The same budget for the planning agent, which runs on the coder's model but does far less:
        one structured call, no tools. Held separately so a tighter planning budget does not
        constrain the coder.
    </p>
    <div class="flex items-center gap-3">
        <label class="flex items-center gap-2 text-sm dark:text-gray-300"
            ><input
                type="checkbox"
                aria-label="Automatic planner context window"
                checked={settings.plannerContextWindow === null}
                onchange={(event) =>
                    onchange({
                        plannerContextWindow: event.currentTarget.checked
                            ? null
                            : automaticContextWindow
                    })}
            /> Auto</label
        >
        <input
            id="plannerContextWindow"
            type="number"
            min={MIN_CONTEXT_WINDOW}
            max={MAX_CONTEXT_WINDOW}
            step="1024"
            disabled={settings.plannerContextWindow === null}
            value={settings.plannerContextWindow ?? automaticContextWindow}
            oninput={(event) =>
                onchange({ plannerContextWindow: readContextWindow(event.currentTarget.value) })}
            class="w-32 rounded border border-gray-300 bg-white p-2 text-gray-900 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
        />
        <span class="text-xs text-gray-500 dark:text-gray-400">tokens</span>
    </div>
</div>

<div class="sm:col-span-2">
    {#if contextWindowDetection === null}<p
            data-testid="context-window-detection-loading"
            class="text-xs text-gray-500 dark:text-gray-400"
        >
            Detecting the selected model's context window...
        </p>{:else if contextWindowDetection.status === 'failed'}<p
            data-testid="context-window-detection-error"
            class="rounded border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700 dark:border-amber-800 dark:bg-amber-900/20 dark:text-amber-300"
        >
            Could not detect the selected model's context window. Showing the default of {DEFAULT_CONTEXT_WINDOW}
            tokens.
        </p>{/if}
</div>
