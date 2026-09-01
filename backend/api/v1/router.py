from fastapi import APIRouter
from dishka.integrations.fastapi import FromDishka, inject
from api.v1.tasks_controller import TaskController
from api.v1.agent_socket import ws_router
from api.v1.config_controller import router as config_router
from schemas.health_response import HealthResponse
from schemas.task_request import TaskRequest
from schemas.task_detail_response import TaskDetailResponse
from schemas.task_list_response import TaskListResponse
from schemas.task_patch_request import TaskPatchRequest
from schemas.task_response import TaskResponse

api_router: APIRouter = APIRouter()
api_router.include_router(ws_router)
api_router.include_router(config_router)

@api_router.get('/health', response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Report whether the service can serve requests.
    """
    from fastapi import HTTPException
    from tortoise import Tortoise

    try:
        await Tortoise.get_connection('default').execute_query('SELECT 1')
    except Exception as error:
        # A process that cannot reach its database can accept connections but
        # cannot answer any real request, so it must not report itself healthy.
        raise HTTPException(
            status_code=503, detail=f'The database is unreachable: {error}'
        ) from error

    return HealthResponse(status='ok', database=True)

@api_router.post('/tasks', response_model=TaskResponse)
@inject
async def create_task(
    body: TaskRequest,
    controller: FromDishka[TaskController],
) -> TaskResponse:
    """
    Create a new AI task.
    """
    return await controller.create_task(body)

@api_router.get('/tasks', response_model=TaskListResponse)
@inject
async def list_tasks(
    controller: FromDishka[TaskController],
    offset: int = 0,
    limit: int = 20,
) -> TaskListResponse:
    """
    List previous tasks, newest first.
    """
    return await controller.list_tasks(offset=offset, limit=limit)

@api_router.get('/tasks/{task_id}', response_model=TaskDetailResponse)
@inject
async def get_task(
    task_id: str,
    controller: FromDishka[TaskController],
) -> TaskDetailResponse:
    """
    Fetch one task and its action log.
    """
    return await controller.get_task(task_id)

@api_router.patch('/tasks/{task_id}', response_model=TaskResponse)
@inject
async def update_task(
    task_id: str,
    body: TaskPatchRequest,
    controller: FromDishka[TaskController],
) -> TaskResponse:
    """
    Update a task's status or summary.
    """
    return await controller.update_task(task_id, body)
