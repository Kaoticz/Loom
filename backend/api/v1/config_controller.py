from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from httpx import AsyncClient, Response
from git import Repo, InvalidGitRepositoryError
import os

from dishka.integrations.fastapi import FromDishka, inject

from schemas.context_window_detection_request import ContextWindowDetectionRequest
from schemas.context_window_detection_response import ContextWindowDetectionResponse
from schemas.context_window_detection_status import ContextWindowDetectionStatus
from schemas.index_deletion_response import IndexDeletionResponse
from schemas.index_status_response import IndexStatusResponse
from services.embedding import EmbeddingService
from services.model_description import ModelDescription
from services.index_deletion import (
    IndexDeletionSummary,
    count_project_index,
    delete_project_index,
)
from services.projects import normalise_project_path
from services.model_resolution import measure_ollama_context_window
from services.session_registry import SessionRegistry

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


@router.get('/ollama/context-window', response_model=ContextWindowDetectionResponse)
async def detect_ollama_context_window(
    request: Annotated[ContextWindowDetectionRequest, Query()],
) -> ContextWindowDetectionResponse:
    """Detect the configured Ollama model's effective context window."""
    if not request.model.startswith('ollama:'):
        return ContextWindowDetectionResponse(
            context_window=None,
            status=ContextWindowDetectionStatus.not_applicable,
        )

    try:
        context_window: int | None = await measure_ollama_context_window(request.model)
    except Exception:
        context_window = None

    if context_window is None:
        return ContextWindowDetectionResponse(
            context_window=None,
            status=ContextWindowDetectionStatus.failed,
        )
    return ContextWindowDetectionResponse(
        context_window=context_window,
        status=ContextWindowDetectionStatus.detected,
    )


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


SESSION_ACTIVE_REASON: str = (
    'A session is running. Deleting the index while an agent is working would leave its '
    'searches returning nothing mid-task, so finish or stop the session first.'
)
NOTHING_INDEXED_REASON: str = 'Nothing is indexed for this project yet.'


@router.get('/index/status', response_model=IndexStatusResponse)
@inject
async def index_status(
    path: str,
    registry: FromDishka[SessionRegistry],
) -> IndexStatusResponse:
    """
    Report what is indexed for a project, and whether it can be removed.

    The path comes from the caller rather than the environment. The settings
    page is where a project is chosen, so it is the only thing that knows which
    project the reader means; a backend reading its own configuration would
    answer about whichever repository it was started with.
    """
    project: str = normalise_project_path(path)
    indexed: int = await count_project_index(project)

    # Deliberately not `is_busy`: that is true from the moment a tab connects,
    # which would block deletion on an idle session that has been given nothing
    # to do.
    if registry.has_active_task:
        return IndexStatusResponse(
            project=project,
            indexed_files=indexed,
            can_delete=False,
            blocked_reason=SESSION_ACTIVE_REASON,
        )

    if indexed == 0:
        return IndexStatusResponse(
            project=project,
            indexed_files=0,
            can_delete=False,
            blocked_reason=NOTHING_INDEXED_REASON,
        )

    return IndexStatusResponse(project=project, indexed_files=indexed, can_delete=True)


@router.delete('/index', response_model=IndexDeletionResponse)
@inject
async def delete_index(
    path: str,
    registry: FromDishka[SessionRegistry],
) -> IndexDeletionResponse:
    """
    Remove everything indexed for a project.

    Refused while a task is running: deleting the index underneath a working
    agent does not raise anything it can report, it simply makes every search
    come back empty, which reads as an unindexed repository rather than as a
    deliberate act. A session that is merely connected is not working, so it
    does not refuse.

    Re-indexing is not started here. It happens when a session next begins, and
    the interface says so.
    """
    if registry.has_active_task:
        raise HTTPException(status_code=409, detail=SESSION_ACTIVE_REASON)

    project: str = normalise_project_path(path)

    try:
        summary: IndexDeletionSummary = await delete_project_index(project)
    except Exception as error:
        # Reported rather than swallowed: a control that failed must not be able
        # to look as though it succeeded.
        raise HTTPException(
            status_code=500, detail=f'The index could not be deleted: {error}'
        ) from error

    if summary.files_deleted == 0:
        message: str = 'There was nothing indexed for this project, so nothing was removed.'
    else:
        message = (
            f'Removed {summary.files_deleted} indexed '
            f'{"file" if summary.files_deleted == 1 else "files"}, '
            f'{summary.symbols_deleted} symbols and {summary.embeddings_deleted} embeddings. '
            'The project is re-indexed when a session next starts.'
        )

    return IndexDeletionResponse(
        project=project,
        embeddings_deleted=summary.embeddings_deleted,
        files_deleted=summary.files_deleted,
        symbols_deleted=summary.symbols_deleted,
        message=message,
    )
