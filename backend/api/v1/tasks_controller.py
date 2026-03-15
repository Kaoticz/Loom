from schemas.task_request import TaskRequest
from schemas.task_response import TaskResponse

class TaskController:
    """
    Controller for managing agent tasks.
    """
    
    async def create_task(self, body: TaskRequest) -> TaskResponse:
        """
        Create a new AI task.

        :param body: The task request body.
        :returns: The created task information.
        """
        # Logic for initializing git branch, spawning sandbox, etc. will go here.
        return TaskResponse(id='task-123', status='initialized')
