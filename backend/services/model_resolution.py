from __future__ import annotations

import os

from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings

from services.compaction import resolve_context_window, response_allowance
from pydantic_ai.providers.openai import OpenAIProvider


def model_settings(context_window: int | None) -> OpenAIChatModelSettings:
    """
    Build the per-run settings that keep a model inside its window.

    Two things are set, and both matter for a locally-hosted model.

    `max_tokens` bounds the reply. Without it the provider's own default applies,
    and if the prompt has filled the window there is no room left to generate
    into — which does not fail loudly but returns an empty response, surfacing as
    "token limit exceeded before any response was generated".

    `num_ctx` tells Ollama how large a window to actually allocate. Ollama
    defaults it low, often 4096, regardless of what the model could hold. Until
    this was sent, Loom budgeted compaction against the configured window while
    the model ran with a much smaller one, so compaction sat far below its target
    and never fired while the real window overflowed. It goes through
    `extra_body` because `num_ctx` is Ollama's own option rather than part of the
    OpenAI schema; a provider that does not know it ignores it.

    :param context_window: The configured window in tokens, or None for the default.
    :returns: The settings to pass to a run.
    """
    return OpenAIChatModelSettings(
        max_tokens=response_allowance(context_window),
        extra_body={'options': {'num_ctx': resolve_context_window(context_window)}},
    )


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

    base_url: str = os.getenv('OLLAMA_BASE_URL', 'http://ollama:11434')
    return OpenAIChatModel(
        model_name=model_name.split(':', 1)[1],
        provider=OpenAIProvider(base_url=f'{base_url}/v1', api_key='ollama'),
    )
