from database.models.embedding_model import EmbeddingModel
from database.models.file import File
from tortoise.fields import Field, ForeignKeyField, ForeignKeyRelation, IntField, OnDelete, TextField
from tortoise.models import Model
from tortoise_embeddings import VectorField

class CodeEmbedding(Model):
    """
    Stores semantic embeddings for code chunks.
    """
    id: Field[int] = IntField(primary_key=True)
    """
    The unique identifier for the embedding.
    """

    model: ForeignKeyRelation[EmbeddingModel] = ForeignKeyField(
        'models.EmbeddingModel',
        related_name='embeddings',
        on_delete=OnDelete.CASCADE
    )
    """
    The embedding model that produced this vector.
    """

    file: ForeignKeyRelation[File] = ForeignKeyField(
        'models.File',
        related_name='embeddings',
        on_delete=OnDelete.CASCADE
    )
    """
    The file the chunk was taken from.
    """

    chunk_text: Field[str] = TextField()
    """
    The source text of the chunk that was embedded.
    """

    embedding: Field[list[float]] = VectorField()
    """
    The vector representation of the chunk.
    """

    start_line: Field[int] = IntField()
    """
    The 1-based line in the file where the chunk begins.
    """

    class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]
        table: str = 'code_embeddings'
        indexes: list[tuple[str, ...]] = [('file',), ('model',)]
