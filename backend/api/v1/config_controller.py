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
        print(f"Error fetching Ollama models: {e}")
        return []

@router.get('/repos/validate')
async def validate_repo(path: str) -> Dict[str, Any]:
    """
    Validate if the given path is a valid git repository.
    """
    if not os.path.exists(path):
        return {'valid': False, 'error': 'Path does not exist'}
    
    try:
        git.Repo(path)
        return {'valid': True}
    except git.InvalidGitRepositoryError:
        return {'valid': False, 'error': 'Not a valid git repository'}
    except Exception as e:
        return {'valid': False, 'error': str(e)}

@router.get('/repos/list')
async def list_directories(path: str = '/') -> List[str]:
    """
    List subdirectories of a given path.
    """
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Path not found")
    
    try:
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
