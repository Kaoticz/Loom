from typing import Annotated

from pydantic import BaseModel, Field

from .task_log_entry_response import TaskLogEntryResponse
from .task_response import TaskResponse


class TaskDetailResponse(BaseModel):
    """
    Schema for a task together with its action log.
    """
    task: Annotated[TaskResponse, Field(description='The task itself.')]
    coder_model: Annotated[str, Field(description='Model used to plan and execute the work.')]
    reviewer_model: Annotated[str, Field(description='Model used to review the work.')]
    coder_context_window: Annotated[int, Field(ge=0, description='Context window in tokens budgeted for the coder agent. Zero means the default was used.')]
    planner_context_window: Annotated[int, Field(ge=0, description='Context window in tokens budgeted for the planner agent. Zero means the default was used.')]
    log: Annotated[list[TaskLogEntryResponse], Field(default_factory=list, description='The recorded action log, oldest first.')]
