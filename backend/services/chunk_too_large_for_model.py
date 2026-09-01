from services.embedding_error import EmbeddingError


class ChunkTooLargeForModel(EmbeddingError):
    """
    Raised when a chosen chunk size does not fit the chosen embedding model.

    Only reachable when the chunk size was set by hand. On automatic sizing an
    over-long chunk is divided until it fits, but a chosen size is a decision,
    and quietly overriding it would leave the setting appearing to work while
    doing nothing.
    """
