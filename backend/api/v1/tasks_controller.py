from uuid import UUID, uuid4

from database.task_status import TaskStatus
from database.models.task import Task
from database.models.task_memory import TaskMemory
from fastapi import HTTPException
from schemas.task_request import TaskRequest
from schemas.task_detail_response import TaskDetailResponse
from schemas.task_list_response import TaskListResponse
from schemas.task_log_entry_response import TaskLogEntryResponse
from schemas.task_patch_request import TaskPatchRequest
from schemas.task_response import TaskResponse

MAX_PAGE_SIZE: int = 100


def to_response(task: Task) -> TaskResponse:
    """
    Render a task record as its API representation.

    :param task: The task to render.
    :returns: The response model.
    """
    return TaskResponse(
        id=str(task.id),
        status=task.status,
        prompt=task.prompt,
        repo_path=task.repo_path,
        branch_name=task.branch_name,
        summary=task.summary,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


class TaskController:
    """
    Controller for managing agent tasks.
    """

    async def create_task(self, body: TaskRequest) -> TaskResponse:
        """
        Create a new AI task.

        The record is registered here, but no work starts: the agent runs over
        the WebSocket endpoint, which adopts this task when given its id.

        :param body: The task request body.
        :returns: The created task information.
        """
        task: Task = await Task.create(
            id=uuid4(),
            prompt=body.prompt,
            repo_path=body.repo_path,
            coder_model=body.coder_model,
            reviewer_model=body.reviewer_model,
            coder_context_window=body.coder_context_window,
            planner_context_window=body.planner_context_window,
            status=TaskStatus.initialized,
        )
        return to_response(task)

    async def list_tasks(self, offset: int = 0, limit: int = 20) -> TaskListResponse:
        """
        List tasks, newest first.

        :param offset: How many tasks to skip.
        :param limit: How many tasks to return.
        :returns: A page of tasks and the total count.
        :raises HTTPException: If the pagination arguments are out of range.
        """
        if offset < 0:
            raise HTTPException(status_code=422, detail='offset cannot be negative')
        if limit < 1 or limit > MAX_PAGE_SIZE:
            raise HTTPException(
                status_code=422, detail=f'limit must be between 1 and {MAX_PAGE_SIZE}'
            )

        total: int = await Task.all().count()
        tasks: list[Task] = (
            await Task.all().order_by('-created_at').offset(offset).limit(limit)
        )
        return TaskListResponse(
            items=[to_response(task) for task in tasks],
            total=total,
            offset=offset,
            limit=limit,
        )

    async def get_task(self, task_id: str) -> TaskDetailResponse:
        """
        Fetch one task together with its action log.

        :param task_id: The identifier of the task.
        :returns: The task and its recorded log, oldest entry first.
        :raises HTTPException: If the identifier is malformed or no such task exists.
        """
        task: Task = await self._require_task(task_id)
        entries: list[TaskMemory] = (
            await TaskMemory.filter(task=task).order_by('created_at', 'id')
        )

        return TaskDetailResponse(
            task=to_response(task),
            coder_model=task.coder_model,
            reviewer_model=task.reviewer_model,
            coder_context_window=task.coder_context_window,
            planner_context_window=task.planner_context_window,
            log=[
                TaskLogEntryResponse(
                    entry_type=entry.entry_type,
                    content=entry.content,
                    created_at=entry.created_at,
                )
                for entry in entries
            ],
        )

    async def update_task(self, task_id: str, body: TaskPatchRequest) -> TaskResponse:
        """
        Update a task's status or summary.

        :param task_id: The identifier of the task.
        :param body: The fields to change.
        :returns: The updated task.
        :raises HTTPException: If the identifier is malformed, no such task exists, or nothing was supplied.
        """
        task: Task = await self._require_task(task_id)

        if body.status is None and body.summary is None:
            raise HTTPException(status_code=422, detail='Supply a status or a summary to update')

        if body.status is not None:
            task.status = body.status
        if body.summary is not None:
            task.summary = body.summary

        await task.save()
        return to_response(task)

    async def _require_task(self, task_id: str) -> Task:
        """
        Load a task by identifier.

        :param task_id: The identifier of the task.
        :returns: The task.
        :raises HTTPException: If the identifier is malformed or no such task exists.
        """
        try:
            identifier: UUID = UUID(task_id)
        except ValueError:
            raise HTTPException(status_code=404, detail=f'No task with id {task_id}') from None

        task: Task | None = await Task.get_or_none(id=identifier)
        if task is None:
            raise HTTPException(status_code=404, detail=f'No task with id {task_id}')
        return task
