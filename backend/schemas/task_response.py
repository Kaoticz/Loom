from typing import Annotated
from pydantic import BaseModel, Field

class TaskResponse(BaseModel):
    """
    Schema for a task response.
    """
    id: Annotated[str, Field()]
    status: Annotated[str, Field()]
