from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import httpx
import os
import git

router = APIRouter()

@router.get('/ollama/models')
async def list_ollama_models() -> List[str]:
    """
    List models available in the local Ollama instance.
    """
    ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://ollama:11434')
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f'{ollama_url}/api/tags')
            resp.raise_for_status()
            data = resp.json()
            return [model['name'] for model in data.get('models', [])]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching Ollama models: {str(e)}"
        )

@router.get('/repos/validate')
async def validate_repo(path: str) -> Dict[str, Any]:
    """
    Validate if the given path is a valid git repository.
    """
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail="Path does not exist")
    
    try:
        git.Repo(path)
        return {'valid': True}
    except git.InvalidGitRepositoryError:
        raise HTTPException(status_code=400, detail="Not a valid git repository")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/repos/list')
async def list_directories(path: str = '/') -> List[str]:
    """
    List subdirectories of a given path.
    """
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Path not found")
    
    try:
        dirs = []
        for name in os.listdir(path):
            full_path = os.path.join(path, name)
            if os.path.isdir(full_path):
                dirs.append(name)
        return sorted(dirs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
