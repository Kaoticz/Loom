from datetime import datetime
from uuid import UUID

from database.enums import TaskStatus
from tortoise.fields import CharEnumField, CharField, DatetimeField, Field, TextField, UUIDField
from tortoise.models import Model

class Task(Model):
    """
    A unit of work the agent was asked to carry out.

    Separate from the action log so that status can be filtered, sorted and
    updated without rewriting the log.
    """
    id: Field[UUID] = UUIDField(primary_key=True)
    """
    The unique identifier for the task.
    """

    prompt: Field[str] = TextField()
    """
    The instruction the user gave the agent.
    """

    repo_path: Field[str] = CharField(max_length=1024, default='')
    """
    The repository the task was carried out in.
    """

    coder_model: Field[str] = CharField(max_length=255, default='')
    """
    The model used to plan and execute the work.
    """

    reviewer_model: Field[str] = CharField(max_length=255, default='')
    """
    The model used to review the work.
    """

    branch_name: Field[str] = CharField(max_length=255, default='')
    """
    The branch the agent's commits were written to.
    """

    status: TaskStatus = CharEnumField(TaskStatus, max_length=32, default=TaskStatus.initialized, index=True)
    """
    Where the task currently stands.
    """

    summary: Field[str] = TextField(default='')
    """
    What the agent reported having accomplished.
    """

    created_at: Field[datetime] = DatetimeField(auto_now_add=True, index=True)
    """
    When the task was created.
    """

    updated_at: Field[datetime] = DatetimeField(auto_now=True)
    """
    When the task was last modified.
    """

    class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]
        table: str = 'tasks'
