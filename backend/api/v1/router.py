from fastapi import APIRouter
from dishka.integrations.fastapi import FromDishka, inject
from api.v1.tasks_controller import TaskController
from api.v1.agent_socket import ws_router
from schemas.task_request import TaskRequest
from schemas.task_response import TaskResponse

api_router: APIRouter = APIRouter()
api_router.include_router(ws_router)

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
