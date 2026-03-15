from tortoise import fields, models
from typing import Any

class Symbol(models.Model):
    """
    Stores AST symbols (functions, classes) found in files.
    """
    id: Any = fields.IntField(pk=True)
    file: Any = fields.ForeignKeyField('models.File', related_name='symbols')
    symbol_name: Any = fields.CharField(max_length=255)
    kind: Any = fields.CharField(max_length=50)  # e.g., 'function', 'class'
    line_start: Any = fields.IntField()
    line_end: Any = fields.IntField()

    class Meta: # type: ignore
        table: str = 'symbols'
