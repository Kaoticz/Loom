from __future__ import annotations

import os

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


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
