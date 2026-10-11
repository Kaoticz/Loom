from __future__ import annotations

import os
from typing import Any

from httpx import AsyncClient, HTTPError, Response

from services.ollama_model_metadata import context_length_from_model_info, num_ctx_from_parameters
from services.ollama_native_model import OllamaNativeModel, OllamaNativeModelSettings

OLLAMA_BASE_URL: str = 'http://ollama:11434'
MODEL_METADATA_TIMEOUT_SECONDS: float = 2.0


def model_settings(
    context_window: int | None, *, allow_subminimum: bool = False
) -> OllamaNativeModelSettings:
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

    The window travels as the required `num_ctx`, which `OllamaNativeModel`
    reads before each call. The model compares it against what Ollama currently
    has loaded and only sends `options.num_ctx` when a reload to a larger
    window is genuinely needed, so alternating consumers never thrash the
    server down to a smaller window. A provider that does not know the key
    ignores it.

    :param context_window: The configured or measured window in tokens, or None for the default.
    :param allow_subminimum: Preserve a positive model measurement below Loom's usual floor.
    :returns: The settings to pass to a run.
    """
    window: int = _window_for_consumer(context_window, allow_subminimum)
    return OllamaNativeModelSettings(ollama_required_num_ctx=window)


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

    override: int | None = num_ctx_from_parameters(payload.get('parameters'))
    if override is not None:
        return override

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
    # `compaction` uses this module's shared consumer helper, so defer its
    # broader resolution import until both modules have initialized.
    from services.compaction import resolve_context_window

    if configured is None or configured <= 0:
        if measured is not None:
            return measured
        return resolve_context_window(configured)

    configured_window: int = resolve_context_window(configured)
    if measured is None:
        return configured_window
    return min(configured_window, measured)


def resolve_model(model_name: str) -> OllamaNativeModel | str:
    """
    Resolve a configured model name into something an agent can run.

    Ollama is reached through its native API, so those names are turned into
    a native model. Anything else is passed through untouched for Pydantic AI
    to interpret.

    Lives here rather than on the session because the session is not always in
    reach of the caller: the controller resolves models for the coder, the
    reviewer and the classifier alike.

    :param model_name: The configured model identifier.
    :returns: An Ollama-backed model, or the name itself for other providers.
    """
    if not model_name.startswith('ollama:'):
        return model_name

    return OllamaNativeModel(
        model_name=_ollama_model_name(model_name),
        base_url=_ollama_base_url(),
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
    # `compaction` imports this helper, so its broader window resolution remains
    # deferred until the helper is called after both modules have initialized.
    from services.compaction import resolve_context_window

    return resolve_context_window(context_window)
