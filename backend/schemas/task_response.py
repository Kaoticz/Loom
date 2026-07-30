from datetime import datetime
from typing import Annotated, Any

from database.enums import TaskLogEntryType, TaskStatus
from pydantic import BaseModel, Field

class TaskResponse(BaseModel):
    """
    Schema for a task response.
    """
    id: Annotated[str, Field(description='The task identifier.')]
    status: Annotated[TaskStatus, Field(description='Where the task currently stands.')]
    prompt: Annotated[str, Field(description='The instruction the agent was given.')]
    repo_path: Annotated[str, Field(description='Repository the work was carried out in.')]
    branch_name: Annotated[str, Field(description='Branch the commits were written to.')]
    summary: Annotated[str, Field(description='What the agent reported having accomplished.')]
    created_at: Annotated[datetime, Field(description='When the task was created.')]
    updated_at: Annotated[datetime, Field(description='When the task was last modified.')]


class TaskLogEntryResponse(BaseModel):
    """
    Schema for a single entry in a task's action log.
    """
    entry_type: Annotated[TaskLogEntryType, Field(description='What kind of event this records.')]
    content: Annotated[dict[str, Any], Field(description='The details of the event.')]
    created_at: Annotated[datetime, Field(description='When the entry was recorded.')]


class TaskDetailResponse(BaseModel):
    """
    Schema for a task together with its action log.
    """
    task: Annotated[TaskResponse, Field(description='The task itself.')]
    coder_model: Annotated[str, Field(description='Model used to plan and execute the work.')]
    reviewer_model: Annotated[str, Field(description='Model used to review the work.')]
    log: Annotated[list[TaskLogEntryResponse], Field(default_factory=list, description='The recorded action log, oldest first.')]


class TaskListResponse(BaseModel):
    """
    Schema for a page of tasks.
    """
    items: Annotated[list[TaskResponse], Field(default_factory=list, description='The tasks on this page, newest first.')]
    total: Annotated[int, Field(ge=0, description='How many tasks exist in total.')]
    offset: Annotated[int, Field(ge=0, description='The offset this page starts at.')]
    limit: Annotated[int, Field(ge=1, description='The page size that was applied.')]


class TaskPatchRequest(BaseModel):
    """
    Schema for updating a task.
    """
    status: Annotated[TaskStatus | None, Field(description='The new status, if it is changing.')] = None
    summary: Annotated[str | None, Field(description='The new summary, if it is changing.')] = None
