<script lang="ts">
    import { onMount } from 'svelte';
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

    // Requirement 1.1: the session belongs to the application, not to one route.
    // The layout is mounted once and survives client-side navigation, so a socket
    // opened here is not torn down by moving to Settings and back — which used to
    // end the session, and then have the new one refused as busy while the old
    // one was still being reaped.
    //
    // `onMount` never runs during server-side rendering, which is the same guard
    // the store's own `browser` check exists for. It is also the reason there is
    // no matching teardown: the layout is destroyed only when the document is,
    // and the socket goes with it.
    onMount(() => {
        const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/api/v1/agent-session';
        agentStore.connect(wsUrl);
    });
</script>

{@render children()}
