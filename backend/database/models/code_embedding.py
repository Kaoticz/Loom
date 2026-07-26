from tortoise import fields, models
from tortoise_embeddings import VectorField
from typing import Any

class CodeEmbedding(models.Model):
    """
    Stores semantic embeddings for code chunks.
    """
    id: Any = fields.IntField(pk=True)
    file: Any = fields.ForeignKeyField('models.File', related_name='embeddings')
    chunk_text: Any = fields.TextField()
    embedding: Any = VectorField()
    start_line: Any = fields.IntField()

    class Meta: # type: ignore
        table: str = 'code_embeddings'
