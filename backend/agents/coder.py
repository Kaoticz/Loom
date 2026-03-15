from __future__ import annotations
from pydantic_ai import Agent, RunContext
from typing import Annotated, Any, ClassVar
from pydantic import BaseModel, Field
from services.docker_sandbox import DockerSandboxManager
from services.code_search import CodeSearchService
from services.file_editor import FileEditorService
from services.git_manager import GitManagerService
import os

from pydantic import ConfigDict

class AgentDeps(BaseModel):
    """
    Dependencies for the Coder agent.
    """
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)
    
    docker: DockerSandboxManager
    search: CodeSearchService
    editor: FileEditorService
    git: GitManagerService

coder_agent: Agent[AgentDeps, str] = Agent(
    'ollama:llama3', # Default model, will be overridden
    deps_type=AgentDeps,
    system_prompt=(
        'You are an expert AI software engineer. Your goal is to complete the given task '
        'by following a ReAct loop: Plan, Action, Observation. '
        '1. Plan: Analyze the request and search the codebase. Output a step-by-step plan. '
        '2. Action: Use the provided tools to modify files or run tests. '
        '3. Observation: Verify your changes by running tests in the sandbox. '
        'Always start by indexing the project and reading relevant files.'
    )
)

@coder_agent.tool
async def search_code(ctx: RunContext[AgentDeps], query: str) -> str:
    """
    Search for code semantically.
    """
    # NOTE: To make this production-ready, we should call an embedding service.
    # We can use httpx to call the local Ollama instance for embeddings.
    import httpx
    import os
    
    # We'll need a way to pass the embedding model. For now, we can use an environment variable or a default.
    # Ideally, it should be in ctx.deps.
    embedding_model = os.getenv('EMBEDDING_MODEL', 'llama3')
    ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://ollama:11434')
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f'{ollama_url}/api/embeddings',
                json={'model': embedding_model, 'prompt': query}
            )
            embedding = resp.json().get('embedding', [0.0] * 768)
    except Exception:
        embedding = [0.0] * 768

    results = await ctx.deps.search.search_code_semantic(embedding) 
    return str(results)

@coder_agent.tool
async def get_symbol(ctx: RunContext[AgentDeps], symbol: str) -> str:
    """
    Get the definition of a symbol (class, function).
    """
    results = await ctx.deps.search.get_symbol_definition(symbol)
    return str(results)

@coder_agent.tool
async def edit_file(ctx: RunContext[AgentDeps], path: str, search: str, replace: str) -> str:
    """
    Edit a file surgically and commit changes.
    """
    success = await ctx.deps.editor.edit_file(path, search, replace)
    if success:
        await ctx.deps.git.commit_changes(f'Edit {path}')
        return 'Success'
    return 'Failed'

@coder_agent.tool
async def run_test(ctx: RunContext[AgentDeps], command: str) -> str:
    """
    Run a test command in the sandbox.
    """
    return await ctx.deps.docker.run_test_command(command)

@coder_agent.tool
async def web_search(ctx: RunContext[AgentDeps], query: str) -> str:
    """
    Search the web for information using SearXNG.
    """
    import httpx
    searxng_url = os.getenv('SEARXNG_URL', 'http://searxng:8080')
    async with httpx.AsyncClient() as client:
        response = await client.get(f'{searxng_url}/search', params={'q': query, 'format': 'json'})
        return str(response.json())
