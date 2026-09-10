from services.embedding_error import EmbeddingError


class EmbeddingInputTooLong(EmbeddingError):
    """
    Raised when the text sent to embed is longer than the model will accept.

    Distinguished from other failures because the caller can do something about
    it: a shorter piece of the same text will usually succeed, whereas an
    unreachable service will not improve on retry.
    """
