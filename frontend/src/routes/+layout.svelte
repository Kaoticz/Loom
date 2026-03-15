<script lang="ts">
  import { onMount } from 'svelte';
  import { agentStore } from '$lib/stores/agent';
  import "../app.css";
  
  let { children } = $props();

  function updateTheme() {
    const theme = $agentStore.config.theme;
    const isDark = theme === 'dark' || (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
    
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }

  onMount(() => {
    updateTheme();
    
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = () => {
      if ($agentStore.config.theme === 'system') {
        updateTheme();
      }
    };
    
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  });

  // Watch for theme changes in store
  $effect(() => {
    $agentStore.config.theme;
    updateTheme();
  });
</script>

{@render children()}
