from datetime import datetime
from tortoise.fields import CharField, DatetimeField, Field, IntField
from tortoise.models import Model

class EmbeddingModel(Model):
    """
    Registry of the embedding models used to index the repository.

    Recording each model's vector width is what makes mixed-dimension corruption
    impossible: embeddings reference the model that produced them, and searches
    are scoped to a single model, so every vector in a comparison is the same
    width by construction.
    """
    id: Field[int] = IntField(primary_key=True)
    """
    The unique identifier for the embedding model.
    """

    model_name: Field[str] = CharField(max_length=100, unique=True)
    """
    The name of the model as known to the provider, for example 'nomic-embed-text'.
    """

    dimensions: Field[int] = IntField()
    """
    The number of components in the vectors this model produces.
    """

    chunk_lines: Field[int] = IntField(default=0)
    """
    Lines per chunk used when these embeddings were made, or 0 for automatic.

    Recorded because a change to it invalidates every vector the model produced:
    the vectors remain comparable with one another, but they no longer describe
    the chunking that was asked for, so the setting would appear to do nothing.
    """

    provider: Field[str] = CharField(max_length=64, default='ollama')
    """
    The service that hosts the model.
    """

    created_at: Field[datetime] = DatetimeField(auto_now_add=True)
    """
    The timestamp when the model was first registered.
    """

    class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]
        table: str = 'embedding_models'
