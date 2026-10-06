from __future__ import annotations

from typing import Any, override

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.profiles import ModelProfileSpec
from pydantic_ai.settings import ModelSettings


class OllamaNativeModel(Model[Any]):
    """
    An Ollama model reached through its native `/api/chat` endpoint.

    Ollama's OpenAI-compatible endpoint silently drops `num_ctx`, so Ollama
    traffic goes through this model instead of `OpenAIChatModel`. Request
    handling lands in Phase 3 (non-streaming) and Phase 4 (streaming); until
    then this class only identifies the model to run.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        *,
        settings: ModelSettings | None = None,
        profile: ModelProfileSpec | None = None,
    ) -> None:
        """
        Create a native Ollama model.

        :param model_name: The Ollama model name, without any provider prefix.
        :param base_url: The Ollama base URL, without a trailing slash.
        :param settings: Default model settings for this model instance.
        :param profile: The model profile to use.
        """
        self._model_name: str = model_name
        self._base_url: str = base_url.rstrip('/')
        super().__init__(settings=settings, profile=profile)

    @property
    @override
    def model_name(self) -> str:
        """
        Return the Ollama model name.

        :returns: The model name, without any provider prefix.
        """
        return self._model_name

    @property
    @override
    def system(self) -> str:
        """
        Return the model provider identifier.

        :returns: The provider name.
        """
        return 'ollama'

    @property
    @override
    def base_url(self) -> str:
        """
        Return the Ollama base URL this model talks to.

        :returns: The base URL, without a trailing slash.
        """
        return self._base_url

    @override
    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """
        Make a non-streaming request to the model.

        Unimplemented until Phase 3; present only so the class is concrete.

        :param messages: The conversation history for the request.
        :param model_settings: Per-request model settings, if any.
        :param model_request_parameters: Tool and output configuration for the request.
        :returns: Never returns.
        :raises NotImplementedError: Always, until Phase 3 implements it.
        """
        raise NotImplementedError('OllamaNativeModel.request() is implemented in Phase 3.')


def decide_num_ctx(loaded_context_size: int | None, required_window: int) -> int | None:
    """
    Decide which `num_ctx` value to send on an Ollama chat request.

    Ollama allocates its KV cache at model-load time, so sending a `num_ctx`
    different from what is already loaded forces a model reload. A smaller
    window never needs to force one: running with a larger loaded window than
    required still satisfies the caller.

    :param loaded_context_size: The context size the model is currently loaded with, or None when unknown.
    :param required_window: The context window this call requires.
    :returns: The `num_ctx` to send, or None when no override is needed.
    """
    if loaded_context_size is None:
        return required_window
    if loaded_context_size >= required_window:
        return None
    return required_window
