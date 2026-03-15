<script lang="ts">
  let prompt: string = '';
  
  async function submitTask() {
    const baseUrl: string = import.meta.env.VITE_API_BASE_URL || '';
    const res = await fetch(`${baseUrl}/api/v1/tasks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });
    const data = await res.json();
    console.log(data);
  }
</script>

<main class="p-8">
  <h1 class="text-2xl font-bold mb-4">Loom AI Agent</h1>
  <textarea 
    bind:value={prompt} 
    placeholder="What should I do?" 
    class="w-full p-2 border rounded mb-4"
  ></textarea>
  <button 
    on:click={submitTask} 
    class="bg-blue-500 text-white px-4 py-2 rounded"
  >
    Submit Task
  </button>
</main>
