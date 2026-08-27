from __future__ import annotations

from typing import Annotated, Any

from httpx import AsyncClient, HTTPError, Response
from pydantic import BaseModel, Field
from services.ollama_model_metadata import (
    ARCHITECTURE_KEY,
    CONTEXT_LENGTH_SUFFIX,
    context_length_from_model_info,
    positive_int,
)

DEFAULT_EMBEDDING_MODEL: str = 'nomic-embed-text'
DIMENSION_PROBE_TEXT: str = 'dimension probe'

EMBEDDING_LENGTH_SUFFIX: str = '.embedding_length'


class EmbeddingError(RuntimeError):
    """
    Raised when an embedding could not be generated.

    A failure to embed is always surfaced rather than being masked with a zero
    vector, because a zero vector silently poisons every downstream similarity
    search.
    """


class ModelDescription(BaseModel):
    """
    What a model reports about itself, for showing to the user.

    Both values are optional because a model need not describe itself, and
    neither is required for indexing to work. They exist so that a person
    choosing between models can see what they are choosing.
    """

    dimensions: int | None = Field(default=None, description='Vector width the model reports, or None if it does not say.')
    context_length: int | None = Field(default=None, description='Tokens the model accepts, or None if it does not say.')


class EmbeddingInputTooLong(EmbeddingError):
    """
    Raised when the text sent to embed is longer than the model will accept.

    Distinguished from other failures because the caller can do something about
    it: a shorter piece of the same text will usually succeed, whereas an
    unreachable service will not improve on retry.
    """


class ChunkTooLargeForModel(EmbeddingError):
    """
    Raised when a chosen chunk size does not fit the chosen embedding model.

    Only reachable when the chunk size was set by hand. On automatic sizing an
    over-long chunk is divided until it fits, but a chosen size is a decision,
    and quietly overriding it would leave the setting appearing to work while
    doing nothing.
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
            detail: str = response.text[:200]
            if _is_length_complaint(detail):
                raise EmbeddingInputTooLong(
                    f'Model {self._embedding_model!r} rejected a {len(text)} character chunk as '
                    f'longer than its context allows: {detail}'
                )
            raise EmbeddingError(
                f'Ollama returned HTTP {response.status_code} while embedding with model '
                f'{self._embedding_model!r}: {detail}'
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

    async def describe_model(self) -> ModelDescription:
        """
        Read what the configured model reports about itself.

        One request answers both questions, and neither requires the model to be
        loaded, so this is cheap enough to run when a user merely picks a model
        from a list. Probing for the width by embedding a string would load it.

        Keys are named for the model's architecture, so the architecture has to
        be read first. Anything unexpected yields None for that value rather than
        an error: these are for display and for avoiding futile work, and the
        caller copes without them.

        Note that the context length is in tokens while chunks are measured in
        lines, and the ratio between them varies by tokenizer. Two models can
        report the same context length and disagree about whether the same text
        fits.

        :returns: What the model reports, with None for anything it does not say.
        """
        info: dict[str, Any] | None = await self._model_info()
        if info is None:
            return ModelDescription()

        architecture: Any = info.get(ARCHITECTURE_KEY)
        if not isinstance(architecture, str) or not architecture:
            return ModelDescription()

        return ModelDescription(
            dimensions=positive_int(info.get(f'{architecture}{EMBEDDING_LENGTH_SUFFIX}')),
            context_length=context_length_from_model_info(info),
        )

    async def detect_context_length(self) -> int | None:
        """
        Discover how many tokens the configured model will accept.

        :returns: The context length in tokens, or None if it cannot be read.
        """
        return (await self.describe_model()).context_length

    async def _model_info(self) -> dict[str, Any] | None:
        """
        Fetch the model's metadata block from Ollama.

        :returns: The metadata, or None if it could not be read.
        """
        client: AsyncClient = self._get_client()
        try:
            response: Response = await client.post(
                f'{self._ollama_url}/api/show',
                json={'model': self._embedding_model},
                timeout=self._timeout,
            )
        except HTTPError:
            return None

        if response.status_code != 200:
            return None

        try:
            payload: dict[str, Any] = response.json()
        except ValueError:
            return None

        info: Any = payload.get('model_info')
        return info if isinstance(info, dict) else None

    async def aclose(self) -> None:
        """
        Release the HTTP client if this service owns it.
        """
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


def _is_length_complaint(detail: str) -> bool:
    """
    Judge whether an error from Ollama is about the input being too long.

    Matched on the message text because the API offers nothing better: the
    status code is the same 500 it returns for anything else. Deliberately
    forgiving about wording, and a false negative merely costs the caller the
    chance to retry with something shorter.

    :param detail: The response body Ollama returned.
    :returns: True if the message describes an over-long input.
    """
    lowered: str = detail.lower()
    return 'context' in lowered and ('exceed' in lowered or 'too long' in lowered)
