from tortoise import fields, models
from typing import Any

class TaskMemory(models.Model):
    """
    Stores agent task logs and tool outputs.
    """
    id: Any = fields.IntField(pk=True)
    task_id: Any = fields.CharField(max_length=255)
    log_content: Any = fields.TextField()
    created_at: Any = fields.DatetimeField(auto_now_add=True)

    class Meta: # type: ignore
        table: str = 'task_memory'
