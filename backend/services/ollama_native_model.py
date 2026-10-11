from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, cast, override

from ollama import AsyncClient, ChatResponse
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    FinishReason,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse, check_allow_model_requests
from pydantic_ai.profiles import ModelProfileSpec
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage


class OllamaNativeModelSettings(ModelSettings, total=False):
    """
    Per-request settings read by OllamaNativeModel.

    A plain `ModelSettings` extension, so providers that do not know the
    Ollama-specific keys ignore them, exactly as they ignored the old
    OpenAI-shaped settings this replaces.
    """

    ollama_required_num_ctx: int
    """The context window the call requires.

    A ceiling for the request, not the value sent: before each call the model
    compares this against what Ollama currently has loaded and only sends
    `options.num_ctx` when a reload to a larger window is genuinely needed.
    """


class OllamaNativeModel(Model[Any]):
    """
    An Ollama model reached through its native `/api/chat` endpoint.

    Ollama's OpenAI-compatible endpoint silently drops `num_ctx`, so Ollama
    traffic goes through this model instead of `OpenAIChatModel`.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        *,
        client: AsyncClient | None = None,
        settings: ModelSettings | None = None,
        profile: ModelProfileSpec | None = None,
    ) -> None:
        """
        Create a native Ollama model.

        :param model_name: The Ollama model name, without any provider prefix.
        :param base_url: The Ollama base URL, without a trailing slash.
        :param client: An Ollama client to reuse, primarily for tests that inject a mock transport.
        :param settings: Default model settings for this model instance.
        :param profile: The model profile to use.
        """
        self._model_name: str = model_name
        self._base_url: str = base_url.rstrip('/')
        self._client: AsyncClient | None = client
        if profile is None:
            profile = {
                'supports_json_schema_output': True,
                'default_structured_output_mode': 'native',
            }
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

        :param messages: The conversation history for the request.
        :param model_settings: Per-request model settings, if any.
        :param model_request_parameters: Tool and output configuration for the request.
        :returns: The model's response.
        :raises UnexpectedModelBehavior: If a message part cannot be sent to Ollama.
        """
        check_allow_model_requests()
        merged_settings, model_request_parameters = self.prepare_request(
            model_settings, model_request_parameters
        )
        instruction_parts: list[InstructionPart] | None = self._get_instruction_parts(
            messages, model_request_parameters
        )
        ollama_messages: list[dict[str, Any]] = _request_messages(messages, instruction_parts)
        tools: list[dict[str, Any]] | None = _request_tools(model_request_parameters)
        output_format: dict[str, Any] | None = _request_format(model_request_parameters)
        if self._client is not None:
            options: dict[str, Any] | None = await _request_options(
                self._client, self._model_name, merged_settings
            )
            response: ChatResponse = await self._client.chat(
                model=self._model_name,
                messages=ollama_messages,
                tools=tools,
                format=output_format,
                options=options,
            )
        else:
            async with AsyncClient(host=self._base_url) as client:
                options = await _request_options(client, self._model_name, merged_settings)
                response = await client.chat(
                    model=self._model_name,
                    messages=ollama_messages,
                    tools=tools,
                    format=output_format,
                    options=options,
                )
        return _response_to_model_response(
            response,
            model_name=self._model_name,
            system=self.system,
            base_url=self._base_url,
        )

    @override
    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        """
        Make a streaming request to the model.

        :param messages: The conversation history for the request.
        :param model_settings: Per-request model settings, if any.
        :param model_request_parameters: Tool and output configuration for the request.
        :param run_context: The run context, unused.
        :returns: A generator yielding the streamed response.
        :raises UnexpectedModelBehavior: If a message part cannot be sent to Ollama.
        """
        check_allow_model_requests()
        merged_settings, model_request_parameters = self.prepare_request(
            model_settings, model_request_parameters
        )
        instruction_parts: list[InstructionPart] | None = self._get_instruction_parts(
            messages, model_request_parameters
        )
        ollama_messages: list[dict[str, Any]] = _request_messages(messages, instruction_parts)
        tools: list[dict[str, Any]] | None = _request_tools(model_request_parameters)
        output_format: dict[str, Any] | None = _request_format(model_request_parameters)
        if self._client is not None:
            options: dict[str, Any] | None = await _request_options(
                self._client, self._model_name, merged_settings
            )
            chat_stream: AsyncIterator[ChatResponse] = await self._client.chat(
                model=self._model_name,
                messages=ollama_messages,
                tools=tools,
                format=output_format,
                options=options,
                stream=True,
            )
            yield OllamaNativeStreamedResponse(
                model_request_parameters,
                chat_stream,
                self._model_name,
                self.system,
                self._base_url,
            )
        else:
            async with AsyncClient(host=self._base_url) as client:
                options = await _request_options(client, self._model_name, merged_settings)
                chat_stream = await client.chat(
                    model=self._model_name,
                    messages=ollama_messages,
                    tools=tools,
                    format=output_format,
                    options=options,
                    stream=True,
                )
                yield OllamaNativeStreamedResponse(
                    model_request_parameters,
                    chat_stream,
                    self._model_name,
                    self.system,
                    self._base_url,
                )


@dataclass
class OllamaNativeStreamedResponse(StreamedResponse):
    """
    A streamed `/api/chat` response mapped to pydantic_ai stream events.

    Ollama streams text as per-line content fragments and reports tool calls
    whole, in a single chunk, rather than as incremental deltas.
    """

    _chunk_stream: AsyncIterator[ChatResponse]
    _model_name: str
    _provider_name: str
    _provider_url: str
    _timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    @override
    def model_name(self) -> str:
        """
        Return the model name.

        :returns: The model name.
        """
        return self._model_name

    @property
    @override
    def provider_name(self) -> str:
        """
        Return the provider name.

        :returns: The provider name.
        """
        return self._provider_name

    @property
    @override
    def provider_url(self) -> str:
        """
        Return the provider URL.

        :returns: The provider URL.
        """
        return self._provider_url

    @property
    @override
    def timestamp(self) -> datetime:
        """
        Return when streaming started locally.

        :returns: The stream start time.
        """
        return self._timestamp

    @override
    async def close_stream(self) -> None:
        """
        Close the underlying chat stream, releasing its connection.
        """
        if isinstance(self._chunk_stream, AsyncGenerator):
            await self._chunk_stream.aclose()

    @override
    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """
        Translate native chat chunks into pydantic_ai stream events.

        :returns: An iterator of stream events.
        :raises UnexpectedModelBehavior: If the stream ends without a final chunk.
        """
        seen_done: bool = False
        async for chunk in self._chunk_stream:
            if chunk.model:
                self._model_name = chunk.model
            if chunk.prompt_eval_count is not None:
                self._usage.input_tokens = chunk.prompt_eval_count
            if chunk.eval_count is not None:
                self._usage.output_tokens = chunk.eval_count
            if chunk.done_reason is not None and chunk.done_reason in _FINISH_REASON_MAP:
                self.finish_reason = _FINISH_REASON_MAP[  # pyright: ignore[reportUnannotatedClassAttribute]
                    chunk.done_reason
                ]
            if chunk.done:
                seen_done = True
            if chunk.message.content:
                for text_event in self._parts_manager.handle_text_delta(
                    vendor_part_id='content', content=chunk.message.content
                ):
                    if text_event is not None:
                        yield text_event
            for index, tool_call in enumerate(chunk.message.tool_calls or []):
                tool_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=index,
                    tool_name=tool_call.function.name,
                    args=dict(tool_call.function.arguments),
                )
                if tool_event is not None:
                    yield tool_event
        if not seen_done:
            raise UnexpectedModelBehavior('Ollama ended the chat stream without a final chunk.')


def _request_messages(
    messages: list[ModelMessage],
    instruction_parts: list[InstructionPart] | None,
) -> list[dict[str, Any]]:
    """
    Map conversation history to native Ollama chat messages.

    :param messages: The conversation history for the request.
    :param instruction_parts: Structured instructions to send as leading system messages, if any.
    :returns: The Ollama message payloads.
    :raises UnexpectedModelBehavior: If a part has no native Ollama representation.
    """
    ollama_messages: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for request_part in message.parts:
                if isinstance(request_part, SystemPromptPart):
                    ollama_messages.append({'role': 'system', 'content': request_part.content})
                elif isinstance(request_part, UserPromptPart):
                    if not isinstance(request_part.content, str):
                        raise UnexpectedModelBehavior(
                            'OllamaNativeModel does not support multimodal user prompts.'
                        )
                    ollama_messages.append({'role': 'user', 'content': request_part.content})
                elif isinstance(request_part, ToolReturnPart):
                    ollama_messages.append(
                        {
                            'role': 'tool',
                            'content': request_part.model_response_str(),
                            'tool_name': request_part.tool_name,
                        }
                    )
                elif isinstance(request_part, RetryPromptPart):
                    if request_part.tool_name is None:
                        ollama_messages.append(
                            {'role': 'user', 'content': request_part.model_response()}
                        )
                    else:
                        ollama_messages.append(
                            {
                                'role': 'tool',
                                'content': request_part.model_response(),
                                'tool_name': request_part.tool_name,
                            }
                        )
                else:
                    raise UnexpectedModelBehavior(
                        f'OllamaNativeModel cannot send {type(request_part).__name__} to Ollama.'
                    )
        else:
            texts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for response_part in message.parts:
                if isinstance(response_part, TextPart):
                    texts.append(response_part.content)
                elif isinstance(response_part, ToolCallPart):
                    tool_calls.append(
                        {
                            'function': {
                                'name': response_part.tool_name,
                                'arguments': response_part.args_as_dict(),
                            }
                        }
                    )
                elif isinstance(response_part, ThinkingPart):
                    continue
                else:
                    raise UnexpectedModelBehavior(
                        f'OllamaNativeModel cannot send {type(response_part).__name__} to Ollama.'
                    )
            if texts or tool_calls:
                assistant_message: dict[str, Any] = {'role': 'assistant'}
                if texts:
                    assistant_message['content'] = ''.join(texts)
                if tool_calls:
                    assistant_message['tool_calls'] = tool_calls
                ollama_messages.append(assistant_message)
    if instruction_parts:
        system_messages: list[dict[str, Any]] = [
            {'role': 'system', 'content': part.content} for part in instruction_parts
        ]
        insert_at: int = next(
            (index for index, item in enumerate(ollama_messages) if item.get('role') != 'system'),
            len(ollama_messages),
        )
        ollama_messages[insert_at:insert_at] = system_messages
    return ollama_messages


def _request_tools(model_request_parameters: ModelRequestParameters) -> list[dict[str, Any]] | None:
    """
    Map declared tools to native Ollama tool definitions.

    :param model_request_parameters: Tool and output configuration for the request.
    :returns: The Ollama tool payloads, or None when the request declares no tools.
    """
    tools: list[dict[str, Any]] = [
        {
            'type': 'function',
            'function': {
                'name': tool.name,
                'description': tool.description or '',
                'parameters': tool.parameters_json_schema,
            },
        }
        for tool in model_request_parameters.declared_tool_defs.values()
    ]
    if not tools:
        return None
    return tools


def _request_format(model_request_parameters: ModelRequestParameters) -> dict[str, Any] | None:
    """
    Read the native JSON schema for structured output, if any.

    :param model_request_parameters: Tool and output configuration for the request.
    :returns: The schema to send as Ollama's `format`, or None for unstructured requests.
    """
    if model_request_parameters.output_mode != 'native':
        return None
    output_object = model_request_parameters.output_object
    if output_object is None:
        return None
    return dict(output_object.json_schema)


def _required_window(model_settings: ModelSettings | None) -> int | None:
    """
    Read the required context window from per-request settings, if any.

    :param model_settings: The settings the call was made with.
    :returns: The required window, or None when no requirement was provided.
    """
    if model_settings is None:
        return None
    window: int | None = cast(OllamaNativeModelSettings, model_settings).get(
        'ollama_required_num_ctx'
    )
    if window is None or window <= 0:
        return None
    return window


async def _loaded_context_size(client: AsyncClient, model_name: str) -> int | None:
    """
    Read the context size Ollama currently has loaded for the model, if any.

    :param client: The Ollama client to query.
    :param model_name: The Ollama model name.
    :returns: The loaded context size, or None when the model is not running.
    """
    running = await client.ps()
    for entry in running.models:
        if entry.model == model_name or entry.name == model_name:
            return entry.context_length
    return None


async def _request_options(
    client: AsyncClient, model_name: str, model_settings: ModelSettings | None
) -> dict[str, Any] | None:
    """
    Decide the Ollama `options` for a call, applying the never-reload-down rule.

    A `/api/ps` failure propagates: proceeding with a guessed window could
    force a needless reload or silently run under-window.

    :param client: The Ollama client to query for loaded models.
    :param model_name: The Ollama model name.
    :param model_settings: The settings the call was made with.
    :returns: The `options` payload, or None when no override is needed.
    """
    required_window: int | None = _required_window(model_settings)
    if required_window is None:
        return None
    loaded_size: int | None = await _loaded_context_size(client, model_name)
    num_ctx: int | None = decide_num_ctx(loaded_size, required_window)
    if num_ctx is None:
        return None
    return {'num_ctx': num_ctx}


_FINISH_REASON_MAP: dict[str, FinishReason] = {'stop': 'stop', 'length': 'length'}


def _response_to_model_response(
    response: ChatResponse, *, model_name: str, system: str, base_url: str
) -> ModelResponse:
    """
    Map a native `/api/chat` response to a pydantic_ai response.

    Ollama reports tool arguments as a JSON object rather than an
    OpenAI-style JSON-encoded string; the mapping passes the object through
    untouched, which is the shape pydantic_ai's tool machinery accepts.

    :param response: The native Ollama response.
    :param model_name: The model name to report when the response names none.
    :param system: The provider name to report.
    :param base_url: The provider URL to report.
    :returns: The pydantic_ai response.
    """
    parts: list[ModelResponsePart] = []
    if response.message.content:
        parts.append(TextPart(content=response.message.content))
    for tool_call in response.message.tool_calls or []:
        parts.append(
            ToolCallPart(tool_name=tool_call.function.name, args=dict(tool_call.function.arguments))
        )
    usage: RequestUsage = RequestUsage(
        input_tokens=response.prompt_eval_count or 0,
        output_tokens=response.eval_count or 0,
    )
    finish_reason: FinishReason | None = None
    if response.done_reason is not None:
        finish_reason = _FINISH_REASON_MAP.get(response.done_reason)
    return ModelResponse(
        parts=parts,
        usage=usage,
        model_name=response.model or model_name,
        timestamp=datetime.now(timezone.utc),
        provider_name=system,
        provider_url=base_url,
        finish_reason=finish_reason,
    )


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
