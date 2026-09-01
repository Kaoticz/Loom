from typing import Annotated

from pydantic import BaseModel, Field

from .task_response import TaskResponse


class TaskListResponse(BaseModel):
    """
    Schema for a page of tasks.
    """
    items: Annotated[list[TaskResponse], Field(default_factory=list, description='The tasks on this page, newest first.')]
    total: Annotated[int, Field(ge=0, description='How many tasks exist in total.')]
    offset: Annotated[int, Field(ge=0, description='The offset this page starts at.')]
    limit: Annotated[int, Field(ge=1, description='The page size that was applied.')]
