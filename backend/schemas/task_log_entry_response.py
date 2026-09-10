from datetime import datetime
from typing import Annotated, Any

from database.task_log_entry_type import TaskLogEntryType
from pydantic import BaseModel, Field


class TaskLogEntryResponse(BaseModel):
    """
    Schema for a single entry in a task's action log.
    """
    entry_type: Annotated[TaskLogEntryType, Field(description='What kind of event this records.')]
    content: Annotated[dict[str, Any], Field(description='The details of the event.')]
    created_at: Annotated[datetime, Field(description='When the entry was recorded.')]
