<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import { agentStore } from '$lib/stores/agent';
    
    let editorContainer: HTMLElement;
    let diffEditor: any;
    let monaco: any;

    async function initMonaco() {
        if (typeof window !== 'undefined') {
            monaco = await import('monaco-editor');
            
            const isDark = document.documentElement.classList.contains('dark');
            diffEditor = monaco.editor.createDiffEditor(editorContainer, {
                originalEditable: false,
                readOnly: true,
                automaticLayout: true,
                theme: isDark ? 'vs-dark' : 'vs',
                renderSideBySide: true
            });

            updateDiff();
        }
    }

    function updateDiff() {
        if (!diffEditor || !monaco || !$agentStore.selectedFile) return;

        const fileDiff = $agentStore.fileDiffs[$agentStore.selectedFile];
        if (!fileDiff) return;

        // Get extension to determine language
        const extension = $agentStore.selectedFile.split('.').pop();
        let language = 'plaintext';
        if (extension === 'ts' || extension === 'js') language = 'javascript';
        else if (extension === 'svelte') language = 'html';
        else if (extension === 'py') language = 'python';
        else if (extension === 'css') language = 'css';
        else if (extension === 'json') language = 'json';

        const originalModel = monaco.editor.createModel(fileDiff.original, language);
        const modifiedModel = monaco.editor.createModel(fileDiff.modified, language);

        diffEditor.setModel({
            original: originalModel,
            modified: modifiedModel
        });
    }

    $: if ($agentStore.selectedFile || $agentStore.fileDiffs) {
        updateDiff();
    }

    $: if (monaco && diffEditor && $agentStore.config.theme) {
        const isDark = document.documentElement.classList.contains('dark');
        monaco.editor.setTheme(isDark ? 'vs-dark' : 'vs');
    }

    onMount(() => {
        initMonaco();
    });

    onDestroy(() => {
        if (diffEditor) {
            diffEditor.dispose();
        }
    });
</script>

<div class="flex flex-col h-full bg-white dark:bg-[#1e1e1e] transition-colors duration-200">
    <!-- File Selector -->
    <div class="flex bg-gray-50 dark:bg-[#252526] border-b border-gray-200 dark:border-[#333] overflow-x-auto no-scrollbar">
        {#each $agentStore.files as file}
            <button
                on:click={() => agentStore.selectFile(file)}
                class="px-4 py-2 text-sm border-r border-gray-200 dark:border-[#333] whitespace-nowrap transition-colors {$agentStore.selectedFile === file ? 'bg-white dark:bg-[#1e1e1e] text-blue-600 dark:text-blue-400 font-medium' : 'text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-[#2a2d2e]'}"
            >
                {file.split('/').pop()}
            </button>
        {/each}
        {#if $agentStore.files.length === 0}
            <div class="px-4 py-2 text-sm text-gray-400 dark:text-gray-500 italic">No files modified yet</div>
        {/if}
    </div>

    <!-- Editor -->
    <div bind:this={editorContainer} class="flex-1"></div>
</div>

<style>
    .no-scrollbar::-webkit-scrollbar {
        display: none;
    }
    .no-scrollbar {
        -ms-overflow-style: none;
        scrollbar-width: none;
    }
</style>
