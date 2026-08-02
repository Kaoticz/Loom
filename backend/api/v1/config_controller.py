from fastapi import APIRouter, HTTPException
from typing import Any
from httpx import AsyncClient, Response
from git import Repo, InvalidGitRepositoryError
import os

from services.embedding import EmbeddingService, ModelDescription

router: APIRouter = APIRouter()

@router.get('/ollama/models')
async def list_ollama_models() -> list[str]:
    """
    List models available in the local Ollama instance.
    """
    ollama_url: str = os.getenv('OLLAMA_BASE_URL', 'http://ollama:11434')
    try:
        async with AsyncClient() as client:
            resp: Response = await client.get(f'{ollama_url}/api/tags')
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            models: list[str] = [model['name'] for model in data.get('models', [])]
            return models
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'Error fetching Ollama models: {str(e)}'
        )

@router.get('/ollama/models/{name:path}/description')
async def describe_ollama_model(name: str) -> dict[str, int | None]:
    """
    Report what an embedding model says about itself.

    Answers the two questions a person choosing a model actually has: how wide
    are its vectors, and how much text will it take. Both come from the model's
    metadata rather than from using it, so selecting a model in the interface
    does not load it.

    Either value may be null, because a model need not describe itself and
    neither answer is required for indexing to work.
    """
    ollama_url: str = os.getenv('OLLAMA_BASE_URL', 'http://ollama:11434')
    service: EmbeddingService = EmbeddingService(ollama_url, embedding_model=name)
    try:
        described: ModelDescription = await service.describe_model()
    finally:
        await service.aclose()

    return {'dimensions': described.dimensions, 'context_length': described.context_length}


@router.get('/repos/validate')
async def validate_repo(path: str) -> dict[str, Any]:
    """
    Validate if the given path is a valid git repository.
    """
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail='Path does not exist')

    try:
        Repo(path)
        return {'valid': True}
    except InvalidGitRepositoryError:
        raise HTTPException(status_code=400, detail='Not a valid git repository')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/repos/list')
async def list_directories(path: str = '/') -> list[str]:
    """
    List subdirectories of a given path.
    """
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='Path not found')

    try:
        dirs: list[str] = []
        for name in os.listdir(path):
            full_path: str = os.path.join(path, name)
            if os.path.isdir(full_path):
                dirs.append(name)
        return sorted(dirs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
