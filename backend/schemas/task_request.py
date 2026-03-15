from typing import Annotated
from pydantic import BaseModel, Field

class TaskRequest(BaseModel):
    """
    Schema for a new task request.
    """
    prompt: Annotated[str, Field(min_length=1)]
