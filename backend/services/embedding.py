from __future__ import annotations

from typing import Any

from httpx import AsyncClient, HTTPError, Response

DEFAULT_EMBEDDING_MODEL: str = 'nomic-embed-text'
DIMENSION_PROBE_TEXT: str = 'dimension probe'


class EmbeddingError(RuntimeError):
    """
    Raised when an embedding could not be generated.

    A failure to embed is always surfaced rather than being masked with a zero
    vector, because a zero vector silently poisons every downstream similarity
    search.
    """


class EmbeddingService:
    """
    Generates vector embeddings through a local Ollama instance.
    """

    _ollama_url: str
    _embedding_model: str
    _timeout: float
    _client: AsyncClient | None
    _owns_client: bool

    def __init__(
        self,
        ollama_url: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        timeout: float = 30.0,
        client: AsyncClient | None = None,
    ) -> None:
        """
        Create an embedding service.

        :param ollama_url: Base URL of the Ollama instance, without a trailing path.
        :param embedding_model: Name of an embedding-capable Ollama model.
        :param timeout: Request timeout in seconds.
        :param client: An existing HTTP client to reuse. When omitted, the service owns its own.
        """
        self._ollama_url = ollama_url.rstrip('/')
        self._embedding_model = embedding_model
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

    @property
    def ollama_url(self) -> str:
        """
        Base URL of the configured Ollama instance.

        :returns: The base URL.
        """
        return self._ollama_url

    @property
    def embedding_model(self) -> str:
        """
        Name of the model used to generate embeddings.

        :returns: The model name.
        """
        return self._embedding_model

    @embedding_model.setter
    def embedding_model(self, value: str) -> None:
        """
        Change the model used to generate embeddings.

        :param value: Name of an embedding-capable Ollama model.
        :raises ValueError: If the model name is empty.
        """
        if not value.strip():
            raise ValueError('The embedding model name cannot be empty.')
        self._embedding_model = value

    def _get_client(self) -> AsyncClient:
        """
        Return the HTTP client, creating an owned one on first use.

        :returns: The HTTP client to issue requests with.
        """
        if self._client is None:
            self._client = AsyncClient(timeout=self._timeout)
            self._owns_client = True
        return self._client

    async def embed(self, text: str) -> list[float]:
        """
        Generate an embedding for the given text.

        :param text: The text to embed.
        :returns: The embedding vector.
        :raises EmbeddingError: If the text is empty, Ollama is unreachable, or the response is malformed.
        """
        if not text.strip():
            raise EmbeddingError('Cannot generate an embedding for empty text.')

        client: AsyncClient = self._get_client()
        response: Response
        try:
            response = await client.post(
                f'{self._ollama_url}/api/embeddings',
                json={'model': self._embedding_model, 'prompt': text},
                timeout=self._timeout,
            )
        except HTTPError as error:
            raise EmbeddingError(
                f'Could not reach Ollama at {self._ollama_url} to generate an embedding. '
                f'Is Ollama running? Original error: {error}'
            ) from error

        if response.status_code != 200:
            raise EmbeddingError(
                f'Ollama returned HTTP {response.status_code} while embedding with model '
                f'{self._embedding_model!r}: {response.text[:200]}'
            )

        payload: dict[str, Any]
        try:
            payload = response.json()
        except ValueError as error:
            raise EmbeddingError('Ollama returned a response that is not valid JSON.') from error

        embedding: Any = payload.get('embedding')
        if not isinstance(embedding, list) or len(embedding) == 0:
            raise EmbeddingError(
                f'Ollama returned no embedding for model {self._embedding_model!r}. '
                'Make sure the model is an embedding model (for example nomic-embed-text) '
                'and that it has been pulled.'
            )

        return [float(value) for value in embedding]

    async def detect_dimensions(self) -> int:
        """
        Determine how many components the configured model's vectors have.

        The width is discovered by embedding a short probe string, since Ollama
        does not report it separately.

        :returns: The number of components in this model's vectors.
        :raises EmbeddingError: If the model cannot be reached or returns nothing usable.
        """
        return len(await self.embed(DIMENSION_PROBE_TEXT))

    async def aclose(self) -> None:
        """
        Release the HTTP client if this service owns it.
        """
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
