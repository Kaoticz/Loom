class EmbeddingError(RuntimeError):
    """
    Raised when an embedding could not be generated.

    A failure to embed is always surfaced rather than being masked with a zero
    vector, because a zero vector silently poisons every downstream similarity
    search.
    """
