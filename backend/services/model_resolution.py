from __future__ import annotations

import os
from typing import Any

from httpx import AsyncClient, HTTPError, Response
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings

from services.compaction import resolve_context_window
from pydantic_ai.providers.openai import OpenAIProvider
from services.ollama_model_metadata import context_length_from_model_info

OLLAMA_BASE_URL: str = 'http://ollama:11434'
MODEL_METADATA_TIMEOUT_SECONDS: float = 2.0


def model_settings(
    context_window: int | None, *, allow_subminimum: bool = False
) -> OpenAIChatModelSettings:
    """
    Build the per-run settings that keep a model inside its window.

    `max_tokens` is deliberately not set. It was, twice, and each time the
    resulting failure quoted whichever number had been chosen — which made it
    look causal when it never was. "Exceeded before any response was generated"
    means the prompt had already filled the model's real context window, leaving
    nothing to generate into; that happens whatever the reply cap is. Setting it
    only decided which number appeared in the message. The provider's own default
    is left in place, and the prompt is kept small enough to answer in by
    `AgentSession`, which narrows its working window whenever this failure shows
    the configured one to be optimistic.

    `num_ctx` tells Ollama how large a window to actually allocate. Ollama
    defaults it low, often 4096, regardless of what the model could hold. Until
    this was sent, Loom budgeted compaction against the configured window while
    the model ran with a much smaller one, so compaction sat far below its target
    and never fired while the real window overflowed. It goes through
    `extra_body` because `num_ctx` is Ollama's own option rather than part of the
    OpenAI schema; a provider that does not know it ignores it.

    :param context_window: The configured or measured window in tokens, or None for the default.
    :param allow_subminimum: Preserve a positive model measurement below Loom's usual floor.
    :returns: The settings to pass to a run.
    """
    window: int = _window_for_consumer(context_window, allow_subminimum)
    return OpenAIChatModelSettings(extra_body={'options': {'num_ctx': window}})


async def measure_ollama_context_window(
    model_name: str,
    *,
    client: AsyncClient | None = None,
    timeout: float = MODEL_METADATA_TIMEOUT_SECONDS,
) -> int | None:
    """
    Read the serving context length for an Ollama model.

    :param model_name: The configured model identifier.
    :param client: Optional HTTP client, primarily for callers that inject a transport.
    :param timeout: Maximum seconds to wait for Ollama's metadata response.
    :returns: The measured context length, or None when it cannot be determined.
    """
    if not model_name.startswith('ollama:'):
        return None

    owns_client: bool = client is None
    request_client: AsyncClient = client or AsyncClient()
    try:
        response: Response = await request_client.post(
            f'{_ollama_base_url()}/api/show',
            json={'model': _ollama_model_name(model_name)},
            timeout=timeout,
        )
    except HTTPError:
        return None
    finally:
        if owns_client:
            await request_client.aclose()

    if response.status_code != 200:
        return None

    try:
        payload: Any = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None

    model_info: Any = payload.get('model_info')
    if not isinstance(model_info, dict):
        return None
    return context_length_from_model_info(model_info)


def effective_context_window(configured: int | None, measured: int | None) -> int:
    """
    Choose the effective context-window ceiling.

    :param configured: The user's requested window, or None for the default.
    :param measured: The window Ollama reported, or None when unavailable.
    :returns: The effective window to use for a run.
    """
    if configured is None or configured <= 0:
        if measured is not None:
            return measured
        return resolve_context_window(configured)

    configured_window: int = resolve_context_window(configured)
    if measured is None:
        return configured_window
    return min(configured_window, measured)


def resolve_model(model_name: str) -> OpenAIChatModel | str:
    """
    Resolve a configured model name into something an agent can run.

    Ollama is reached through its OpenAI-compatible endpoint, so those names are
    turned into a configured client. Anything else is passed through untouched
    for Pydantic AI to interpret.

    Lives here rather than on the session because the session is not always in
    reach of the caller: the controller resolves models for the coder, the
    reviewer and the classifier alike.

    :param model_name: The configured model identifier.
    :returns: An Ollama-backed model, or the name itself for other providers.
    """
    if not model_name.startswith('ollama:'):
        return model_name

    base_url: str = _ollama_base_url()
    return OpenAIChatModel(
        model_name=_ollama_model_name(model_name),
        provider=OpenAIProvider(base_url=f'{base_url}/v1', api_key='ollama'),
    )


def _ollama_base_url() -> str:
    """
    Return the configured Ollama endpoint without a trailing slash.

    :returns: The Ollama base URL.
    """
    return os.getenv('OLLAMA_BASE_URL', OLLAMA_BASE_URL).rstrip('/')


def _ollama_model_name(model_name: str) -> str:
    """
    Strip only the Ollama provider prefix from a configured model identifier.

    :param model_name: The configured Ollama model identifier.
    :returns: The Ollama model name, preserving any tag.
    """
    return model_name.split(':', 1)[1]


def _window_for_consumer(context_window: int | None, allow_subminimum: bool) -> int:
    """
    Resolve a requested window while preserving a known small model limit.

    :param context_window: The requested window, or None for the default.
    :param allow_subminimum: Whether a positive window may remain below Loom's floor.
    :returns: The window suitable for a provider request.
    """
    if allow_subminimum and context_window is not None and context_window > 0:
        return context_window
    return resolve_context_window(context_window)
