from tortoise import fields, models
from typing import Any

class File(models.Model):
    """
    Stores file information from the repository.
    """
    id: Any = fields.IntField(pk=True)
    file_path: Any = fields.CharField(max_length=1024, unique=True)
    raw_content: Any = fields.TextField()
    hash: Any = fields.CharField(max_length=64)
    last_indexed: Any = fields.DatetimeField(auto_now=True)

    class Meta: # type: ignore
        table: str = 'files'
