import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.v1 import config_controller
from schemas.context_window import (
    ContextWindowDetectionRequest,
    ContextWindowDetectionResponse,
    ContextWindowDetectionStatus,
)


async def test_context_window_endpoint_reports_a_detected_modelfile_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A measured Modelfile override is returned as a usable detected value.
    """

    async def probe(model_name: str) -> int | None:
        """
        Return the explicit Modelfile context window.

        :param model_name: The model identifier requested by the endpoint.
        :returns: The detected Modelfile override.
        """
        assert model_name == 'ollama:llama3'
        return 8192

    monkeypatch.setattr(config_controller, 'measure_ollama_context_window', probe)

    response = await config_controller.detect_ollama_context_window(
        ContextWindowDetectionRequest(model='ollama:llama3')
    )

    assert response.context_window == 8192
    assert response.status is ContextWindowDetectionStatus.detected


async def test_context_window_endpoint_reports_architecture_fallback_as_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Architecture metadata is still a successful known context-window value.
    """

    async def probe(model_name: str) -> int | None:
        """
        Return the architecture-derived context window.

        :param model_name: The model identifier requested by the endpoint.
        :returns: The detected architecture context window.
        """
        assert model_name == 'ollama:llama3'
        return 131_072

    monkeypatch.setattr(config_controller, 'measure_ollama_context_window', probe)

    response = await config_controller.detect_ollama_context_window(
        ContextWindowDetectionRequest(model='ollama:llama3')
    )

    assert response.context_window == 131_072
    assert response.status is ContextWindowDetectionStatus.detected


async def test_context_window_endpoint_reports_an_unavailable_ollama_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An unavailable Ollama produces a structured fallback instead of an error.
    """

    async def probe(model_name: str) -> int | None:
        """
        Simulate an unavailable Ollama context-window probe.

        :param model_name: The model identifier requested by the endpoint.
        :returns: None because detection failed.
        """
        assert model_name == 'ollama:llama3'
        return None

    monkeypatch.setattr(config_controller, 'measure_ollama_context_window', probe)

    response = await config_controller.detect_ollama_context_window(
        ContextWindowDetectionRequest(model='ollama:llama3')
    )

    assert response.context_window is None
    assert response.status is ContextWindowDetectionStatus.failed


async def test_context_window_endpoint_skips_non_ollama_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A foreign provider has no detection attempt or failure state.
    """

    async def probe(model_name: str) -> int | None:
        """
        Fail if a non-Ollama model reaches the Ollama probe.

        :param model_name: The model identifier that must not be probed.
        :returns: Never returns because the probe must not be called.
        :raises AssertionError: Always, if the endpoint probes a foreign model.
        """
        raise AssertionError(f'Unexpected probe for {model_name}')

    monkeypatch.setattr(config_controller, 'measure_ollama_context_window', probe)

    response = await config_controller.detect_ollama_context_window(
        ContextWindowDetectionRequest(model='openai:gpt-4o')
    )

    assert response.context_window is None
    assert response.status is ContextWindowDetectionStatus.not_applicable


def test_context_window_route_validates_the_query_and_response_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    FastAPI exposes the typed request and response shape to browser callers.
    """

    async def probe(model_name: str) -> int | None:
        """
        Return a known context window for the route contract test.

        :param model_name: The model identifier requested by the endpoint.
        :returns: The detected context window.
        """
        assert model_name == 'ollama:llama3'
        return 8192

    monkeypatch.setattr(config_controller, 'measure_ollama_context_window', probe)
    app: FastAPI = FastAPI()
    app.include_router(config_controller.router)

    with TestClient(app) as client:
        result = client.get('/ollama/context-window', params={'model': 'ollama:llama3'})

    assert result.status_code == 200
    response: ContextWindowDetectionResponse = ContextWindowDetectionResponse.model_validate(
        result.json()
    )
    assert response.context_window == 8192
    assert response.status is ContextWindowDetectionStatus.detected
