<script lang="ts">
    import { agentStore } from '$lib/stores/agent';
    import { isDarkTheme, prefersDark } from '$lib/theme';
    import '../app.css';

    let { children } = $props();

    const dark = $derived(isDarkTheme($agentStore.config.theme, $prefersDark));

    // The class Tailwind keys its dark variants off. Following `dark` rather
    // than the setting means a change of operating system appearance is picked
    // up too, without a media query listener of its own here.
    $effect(() => {
        document.documentElement.classList.toggle('dark', dark);
    });
</script>

{@render children()}
