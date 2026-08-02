<script lang="ts">
    import { onDestroy } from 'svelte';
    import { MergeView } from '@codemirror/merge';
    import { EditorState, type Extension } from '@codemirror/state';
    import { EditorView } from '@codemirror/view';
    import { oneDark } from '@codemirror/theme-one-dark';
    import { languageIdForPath, loadLanguage } from '$lib/editor/languages';

    let {
        path,
        original,
        modified,
        dark = false
    }: {
        path: string;
        original: string;
        modified: string;
        dark?: boolean;
    } = $props();

    let host: HTMLElement | undefined = $state();
    let view: MergeView | null = null;

    /**
     * Which rebuild is the current one.
     *
     * Building is asynchronous, because the parser is fetched first. Two
     * rebuilds started in quick succession can therefore finish in either
     * order, and without this an older one finishing last would install the
     * previous file's diff over the newer one.
     */
    let currentBuild = 0;

    /** Extensions shared by both panes: read only, since this only ever displays. */
    function baseExtensions(language: Extension | null): Extension[] {
        const extensions: Extension[] = [
            EditorView.editable.of(false),
            EditorState.readOnly.of(true),
            EditorView.lineWrapping
        ];
        if (language) extensions.push(language);
        if (dark) extensions.push(oneDark);
        return extensions;
    }

    /**
     * Build the merge view for the current file.
     *
     * The language is resolved and fetched before the view is constructed, so
     * the panes are never rendered once unhighlighted and then again with
     * highlighting.
     */
    async function build(): Promise<void> {
        if (!host) return;

        const build = (currentBuild += 1);
        const languageId = languageIdForPath(path);
        let language: Extension | null = null;
        if (languageId) {
            try {
                language = await loadLanguage(languageId);
            } catch (error) {
                // A missing parser is not worth failing the diff over; the file
                // is still perfectly readable without highlighting.
                console.warn(`No highlighting available for ${languageId}.`, error);
            }
        }

        // Abandon this build if the component went away, or if a newer build
        // started while the parser was loading.
        if (!host || build !== currentBuild) return;

        view?.destroy();
        view = new MergeView({
            parent: host,
            a: { doc: original, extensions: baseExtensions(language) },
            b: { doc: modified, extensions: baseExtensions(language) },
            highlightChanges: true,
            gutter: true
        });
    }

    $effect(() => {
        // Rebuilt whenever the file or its content changes.
        void [path, original, modified, dark];
        void build();
    });

    onDestroy(() => {
        view?.destroy();
        view = null;
    });
</script>

<div bind:this={host} data-testid="diff-editor" class="h-full overflow-auto text-sm"></div>
