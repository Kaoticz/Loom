from datetime import datetime
from typing import Any

from database.task_log_entry_type import TaskLogEntryType
from database.models.task import Task
from tortoise.fields import (
    CharEnumField,
    DatetimeField,
    Field,
    ForeignKeyField,
    ForeignKeyRelation,
    IntField,
    JSONField,
    OnDelete,
)
from tortoise.models import Model

class TaskMemory(Model):
    """
    An append-only record of what happened during a task.

    One row per event rather than one row per task, so recording an action is a
    single insert instead of rewriting the whole log.
    """
    id: Field[int] = IntField(primary_key=True)
    """
    The unique identifier for the log entry.
    """

    task: ForeignKeyRelation[Task] = ForeignKeyField(
        'models.Task',
        related_name='log_entries',
        on_delete=OnDelete.CASCADE
    )
    """
    The task this entry belongs to.
    """

    entry_type: TaskLogEntryType = CharEnumField(TaskLogEntryType, max_length=32)
    """
    What kind of event this entry records.
    """

    content: Field[dict[str, Any]] = JSONField(default=dict)
    """
    The details of the event, shaped according to its type.
    """

    created_at: Field[datetime] = DatetimeField(auto_now_add=True)
    """
    When the entry was recorded.
    """

    class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]
        table: str = 'task_memory'
        indexes: list[tuple[str, ...]] = [('task',)]
