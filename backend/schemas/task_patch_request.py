from typing import Annotated

from database.task_status import TaskStatus
from pydantic import BaseModel, Field


class TaskPatchRequest(BaseModel):
    """
    Schema for updating a task.
    """
    status: Annotated[TaskStatus | None, Field(description='The new status, if it is changing.')] = None
    summary: Annotated[str | None, Field(description='The new summary, if it is changing.')] = None
