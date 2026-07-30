from typing import Annotated
from pydantic import BaseModel, Field

class TaskRequest(BaseModel):
    """
    Schema for a new task request.
    """
    prompt: Annotated[str, Field(min_length=1, description='The instruction for the agent.')]
    repo_path: Annotated[str, Field(description='Repository the work should be carried out in.')] = ''
    coder_model: Annotated[str, Field(description='Model used to plan and execute the work.')] = ''
    reviewer_model: Annotated[str, Field(description='Model used to review the work.')] = ''
