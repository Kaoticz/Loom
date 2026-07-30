from __future__ import annotations

import pytest
from httpx import AsyncClient, ConnectError, MockTransport, Request, Response

from services.embedding import DEFAULT_EMBEDDING_MODEL, EmbeddingError, EmbeddingService


def service_returning(handler: object) -> EmbeddingService:
    """
    Build an embedding service backed by the given mock handler.

    :param handler: A callable accepting a request and returning a response.
    :returns: The embedding service.
    """
    transport: MockTransport = MockTransport(handler)  # type: ignore[arg-type]
    return EmbeddingService('http://ollama:11434/', client=AsyncClient(transport=transport))


async def test_returns_the_embedding() -> None:
    """
    A well formed response yields the embedding vector.
    """
    service: EmbeddingService = service_returning(
        lambda request: Response(200, json={'embedding': [0.1, 0.2, 0.3]})
    )
    assert await service.embed('some code') == [0.1, 0.2, 0.3]


def test_configuration_defaults() -> None:
    """
    The base URL is normalised and the default model is embedding capable.
    """
    service: EmbeddingService = service_returning(lambda request: Response(200, json={'embedding': [1.0]}))
    assert service.ollama_url == 'http://ollama:11434'
    assert service.embedding_model == DEFAULT_EMBEDDING_MODEL == 'nomic-embed-text'


@pytest.mark.parametrize(
    'handler',
    [
        lambda request: Response(500, text='boom'),
        lambda request: Response(200, json={}),
        lambda request: Response(200, json={'embedding': []}),
        lambda request: Response(200, text='not json'),
    ],
    ids=['non_200', 'missing_embedding', 'empty_embedding', 'invalid_json'],
)
async def test_malformed_responses_raise(handler: object) -> None:
    """
    Any unusable response raises rather than returning a degenerate vector.
    """
    service: EmbeddingService = service_returning(handler)
    with pytest.raises(EmbeddingError):
        await service.embed('x')


async def test_unreachable_ollama_raises_with_guidance() -> None:
    """
    A connection failure is reported with an actionable message.
    """
    def unreachable(request: Request) -> Response:
        raise ConnectError('connection refused')

    service: EmbeddingService = service_returning(unreachable)
    with pytest.raises(EmbeddingError, match='Is Ollama running'):
        await service.embed('x')


async def test_empty_text_raises() -> None:
    """
    Embedding blank text is rejected before any request is made.
    """
    service: EmbeddingService = service_returning(lambda request: Response(200, json={'embedding': [1.0]}))
    with pytest.raises(EmbeddingError):
        await service.embed('   ')


def test_empty_model_name_rejected() -> None:
    """
    The embedding model cannot be set to a blank name.
    """
    service: EmbeddingService = service_returning(lambda request: Response(200, json={'embedding': [1.0]}))
    with pytest.raises(ValueError):
        service.embedding_model = ''
