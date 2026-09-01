from datetime import datetime
from typing import Annotated

from database.task_status import TaskStatus
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
