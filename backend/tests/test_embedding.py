from __future__ import annotations

import pytest
from httpx import AsyncClient, ConnectError, MockTransport, Request, Response

from services.embedding import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingService,
)
from services.embedding_error import EmbeddingError
from services.embedding_input_too_long import EmbeddingInputTooLong


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


# The metadata Ollama actually returns for snowflake-arctic-embed2, trimmed to
# the keys that matter. Note the context length is 8192, the same as
# nomic-embed-text: the models differ in how densely they tokenise, not in how
# much they will hold.
ARCTIC_MODEL_INFO: dict[str, object] = {
    'general.architecture': 'bert',
    'bert.context_length': 8192,
    'bert.embedding_length': 1024,
    'tokenizer.ggml.model': 't5',
    'tokenizer.ggml.mask_token_id': 250001,
}


async def test_the_context_length_is_read_from_the_model_metadata() -> None:
    """
    Ollama reports the limit under a key named for the model's architecture.
    """

    def handler(request: Request) -> Response:
        assert request.url.path == '/api/show'
        return Response(200, json={'model_info': ARCTIC_MODEL_INFO})

    assert await service_returning(handler).detect_context_length() == 8192


# The metadata Ollama actually returns for nomic-embed-text, trimmed. Note that
# its context is 2048 against arctic's 8192: the two models differ in both
# directions, so neither can stand in for the other.
NOMIC_MODEL_INFO: dict[str, object] = {
    'general.architecture': 'nomic-bert',
    'nomic-bert.context_length': 2048,
    'nomic-bert.embedding_length': 768,
    'tokenizer.ggml.model': 'bert',
    'tokenizer.ggml.mask_token_id': 103,
}


async def test_the_architecture_prefix_is_followed_rather_than_assumed() -> None:
    """
    The key is not always `bert.context_length`.

    Reading `general.architecture` first is the whole point: nomic-embed-text
    reports under `nomic-bert`, so a hard-coded prefix would return nothing for
    the project's own default model.
    """

    def handler(request: Request) -> Response:
        return Response(200, json={'model_info': NOMIC_MODEL_INFO})

    described = await service_returning(handler).describe_model()

    assert described.context_length == 2048
    assert described.dimensions == 768


async def test_two_real_models_are_described_differently() -> None:
    """
    Neither model can stand in for the other, in either direction.

    arctic-embed2 holds four times as much as nomic-embed-text while producing
    wider vectors, which is the opposite of what its failure to index first
    suggested. The values are read, never assumed.
    """

    def arctic(request: Request) -> Response:
        return Response(200, json={'model_info': ARCTIC_MODEL_INFO})

    def nomic(request: Request) -> Response:
        return Response(200, json={'model_info': NOMIC_MODEL_INFO})

    described_arctic = await service_returning(arctic).describe_model()
    described_nomic = await service_returning(nomic).describe_model()

    assert (described_arctic.context_length, described_arctic.dimensions) == (8192, 1024)
    assert (described_nomic.context_length, described_nomic.dimensions) == (2048, 768)


@pytest.mark.parametrize(
    'payload',
    [
        {},
        {'model_info': None},
        {'model_info': {}},
        {'model_info': {'general.architecture': 'bert'}},
        {'model_info': {'general.architecture': '', 'bert.context_length': 8192}},
        {'model_info': {'general.architecture': 'bert', 'bert.context_length': 'many'}},
        {'model_info': {'general.architecture': 'bert', 'bert.context_length': 0}},
    ],
    ids=['no info', 'null info', 'empty info', 'no length', 'no architecture', 'not a number', 'zero'],
)
async def test_unreadable_metadata_yields_no_answer(payload: dict[str, object]) -> None:
    """
    Knowing the limit is an optimisation, so an unusable answer is not an error.
    """

    def handler(request: Request) -> Response:
        return Response(200, json=payload)

    assert await service_returning(handler).detect_context_length() is None


async def test_an_unreachable_model_yields_no_answer() -> None:
    """
    A model that cannot be described still has to be usable.
    """

    def handler(request: Request) -> Response:
        raise ConnectError('no route to host')

    assert await service_returning(handler).detect_context_length() is None


async def test_a_length_rejection_is_distinguishable_from_other_failures() -> None:
    """
    The caller can retry a shorter chunk, but cannot fix an unreachable service.
    """

    def handler(request: Request) -> Response:
        return Response(500, json={'error': 'the input length exceeds the context length'})

    with pytest.raises(EmbeddingInputTooLong):
        await service_returning(handler).embed('a very long chunk')


async def test_other_server_errors_are_not_treated_as_length_problems() -> None:
    """
    Splitting the input would not help, and would hide the real fault.
    """

    def handler(request: Request) -> Response:
        return Response(500, json={'error': 'model not found, try pulling it first'})

    with pytest.raises(EmbeddingError) as raised:
        await service_returning(handler).embed('anything')

    assert not isinstance(raised.value, EmbeddingInputTooLong)


async def test_a_model_describes_its_width_and_capacity_in_one_call() -> None:
    """
    Both answers come from metadata, so choosing a model does not load it.

    Probing for the width by embedding a string would load the model into
    memory, which is an unpleasant thing to do when someone merely opens a
    dropdown.
    """
    calls: list[str] = []

    def handler(request: Request) -> Response:
        calls.append(request.url.path)
        return Response(200, json={'model_info': ARCTIC_MODEL_INFO})

    described = await service_returning(handler).describe_model()

    assert described.dimensions == 1024
    assert described.context_length == 8192
    assert calls == ['/api/show'], 'describing a model should not embed anything'


async def test_a_model_that_says_nothing_is_described_as_unknown() -> None:
    """
    Neither value is required for indexing, so absence is not an error.
    """

    def handler(request: Request) -> Response:
        return Response(200, json={'model_info': {'general.architecture': 'bert'}})

    described = await service_returning(handler).describe_model()

    assert described.dimensions is None
    assert described.context_length is None


async def test_a_boolean_is_not_mistaken_for_a_measurement() -> None:
    """
    `True` is an int in Python, and would otherwise read as a width of one.
    """

    def handler(request: Request) -> Response:
        return Response(
            200,
            json={'model_info': {'general.architecture': 'bert', 'bert.embedding_length': True}},
        )

    assert (await service_returning(handler).describe_model()).dimensions is None
