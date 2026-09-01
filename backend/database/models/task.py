from datetime import datetime
from uuid import UUID

from database.task_status import TaskStatus
from tortoise.fields import (
    CharEnumField,
    CharField,
    DatetimeField,
    Field,
    IntField,
    TextField,
    UUIDField,
)
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

    coder_context_window: Field[int] = IntField(default=0)
    """
    Context window, in tokens, budgeted for the coder agent.

    Zero means the user configured nothing and the default was used. Recorded as
    the raw setting rather than the resolved number so that a later change to
    the default is visible in old tasks as "unset", not as the old default.
    """

    planner_context_window: Field[int] = IntField(default=0)
    """
    Context window, in tokens, budgeted for the planner agent.

    Separate from the coder's at the user's direction. The two agents share one
    model setting today, so a deployment where they differ is one where the user
    wants the planner held to a tighter budget than the coder.
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
