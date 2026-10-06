import json
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest
from httpx import ConnectError, MockTransport, Request, Response
from ollama import AsyncClient as OllamaAsyncClient
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.tools import ToolDefinition

from services.ollama_native_model import OllamaNativeModel, decide_num_ctx


@asynccontextmanager
async def _mocked_model(handler: Callable[[Request], Response]) -> AsyncGenerator[OllamaNativeModel]:
    """
    Build a model whose Ollama traffic is served by the given handler.

    :param handler: The mock transport handler answering `/api/chat`.
    :returns: An iterator yielding the model.
    """
    client: OllamaAsyncClient = OllamaAsyncClient(
        host='http://ollama:11434', transport=MockTransport(handler)
    )
    try:
        yield OllamaNativeModel('qwen2.5-coder', 'http://ollama:11434', client=client)
    finally:
        await client.close()  # type: ignore[no-untyped-call]


def _chat_response(content: str) -> dict[str, Any]:
    """
    Build a minimal native `/api/chat` response body.

    :param content: The assistant message text.
    :returns: The response payload.
    """
    return {
        'model': 'qwen2.5-coder',
        'created_at': '2026-10-06T00:00:00Z',
        'message': {'role': 'assistant', 'content': content},
        'done': True,
        'done_reason': 'stop',
        'prompt_eval_count': 10,
        'eval_count': 5,
    }


async def test_plain_text_reply_becomes_a_text_part() -> None:
    """
    The common case: Ollama answers with text, pydantic_ai sees text.
    """
    seen_bodies: list[dict[str, Any]] = []

    def handler(request: Request) -> Response:
        assert request.url.path == '/api/chat'
        seen_bodies.append(json.loads(request.content))
        return Response(200, json=_chat_response('done'))

    model: OllamaNativeModel
    async with _mocked_model(handler) as model:
        response = await model.request(
            [ModelRequest(parts=[UserPromptPart(content='fix it')])],
            None,
            ModelRequestParameters(),
        )

    assert len(response.parts) == 1
    part = response.parts[0]
    assert isinstance(part, TextPart)
    assert part.content == 'done'
    assert response.model_name == 'qwen2.5-coder'
    assert seen_bodies[0]['model'] == 'qwen2.5-coder'
    assert seen_bodies[0]['messages'] == [{'role': 'user', 'content': 'fix it'}]


async def test_token_counts_reach_usage() -> None:
    """
    Compaction budgets against provider-reported counts, so they must arrive.
    """

    def handler(request: Request) -> Response:
        return Response(200, json=_chat_response('done'))

    async with _mocked_model(handler) as model:
        response = await model.request(
            [ModelRequest(parts=[UserPromptPart(content='fix it')])],
            None,
            ModelRequestParameters(),
        )

    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5


async def test_structured_output_travels_as_a_native_format_schema() -> None:
    """
    Structured output constrains generation through `format`, not tools.
    """
    seen_bodies: list[dict[str, Any]] = []
    schema: dict[str, Any] = {
        'type': 'object',
        'properties': {'verdict': {'type': 'string'}},
        'required': ['verdict'],
    }

    def handler(request: Request) -> Response:
        seen_bodies.append(json.loads(request.content))
        return Response(200, json=_chat_response('{"verdict": "changes"}'))

    async with _mocked_model(handler) as model:
        response = await model.request(
            [ModelRequest(parts=[UserPromptPart(content='classify this')])],
            None,
            ModelRequestParameters(
                output_mode='native',
                output_object=OutputObjectDefinition(json_schema=schema),
            ),
        )

    assert seen_bodies[0]['format'] == schema
    assert seen_bodies[0]['tools'] == []
    assert len(response.parts) == 1
    part = response.parts[0]
    assert isinstance(part, TextPart)
    assert part.content == '{"verdict": "changes"}'


async def test_object_tool_arguments_become_a_tool_call_part() -> None:
    """
    Ollama reports tool arguments as an object, not a JSON-encoded string.
    """
    seen_bodies: list[dict[str, Any]] = []

    def handler(request: Request) -> Response:
        seen_bodies.append(json.loads(request.content))
        return Response(
            200,
            json={
                'model': 'qwen2.5-coder',
                'created_at': '2026-10-06T00:00:00Z',
                'message': {
                    'role': 'assistant',
                    'content': '',
                    'tool_calls': [
                        {'function': {'name': 'read_file', 'arguments': {'path': 'main.py'}}}
                    ],
                },
                'done': True,
                'done_reason': 'stop',
            },
        )

    tool: ToolDefinition = ToolDefinition(
        name='read_file',
        description='Read a file',
        parameters_json_schema={
            'type': 'object',
            'properties': {'path': {'type': 'string'}},
            'required': ['path'],
        },
    )
    async with _mocked_model(handler) as model:
        response = await model.request(
            [ModelRequest(parts=[UserPromptPart(content='read it')])],
            None,
            ModelRequestParameters(function_tools=[tool]),
        )

    assert seen_bodies[0]['tools'] == [
        {
            'type': 'function',
            'function': {
                'name': 'read_file',
                'description': 'Read a file',
                'parameters': {
                    'type': 'object',
                    'properties': {'path': {'type': 'string'}},
                    'required': ['path'],
                },
            },
        }
    ]
    assert len(response.parts) == 1
    part = response.parts[0]
    assert isinstance(part, ToolCallPart)
    assert part.tool_name == 'read_file'
    assert part.args == {'path': 'main.py'}


async def test_empty_object_arguments_become_an_empty_tool_call() -> None:
    """
    A tool call with no arguments is valid, not an error.
    """

    def handler(request: Request) -> Response:
        return Response(
            200,
            json={
                'model': 'qwen2.5-coder',
                'created_at': '2026-10-06T00:00:00Z',
                'message': {
                    'role': 'assistant',
                    'content': '',
                    'tool_calls': [{'function': {'name': 'list_files', 'arguments': {}}}],
                },
                'done': True,
                'done_reason': 'stop',
            },
        )

    async with _mocked_model(handler) as model:
        response = await model.request(
            [ModelRequest(parts=[UserPromptPart(content='list them')])],
            None,
            ModelRequestParameters(),
        )

    assert len(response.parts) == 1
    part = response.parts[0]
    assert isinstance(part, ToolCallPart)
    assert part.args == {}


async def test_a_non_200_response_raises_without_fallback() -> None:
    """
    A failed chat call must surface, never degrade silently.
    """

    def handler(request: Request) -> Response:
        return Response(500, text='boom')

    async with _mocked_model(handler) as model:
        with pytest.raises(Exception):
            await model.request(
                [ModelRequest(parts=[UserPromptPart(content='fix it')])],
                None,
                ModelRequestParameters(),
            )


async def test_a_malformed_response_body_raises() -> None:
    """
    An unparseable chat reply must surface, never degrade silently.
    """

    def handler(request: Request) -> Response:
        return Response(200, content=b'not json')

    async with _mocked_model(handler) as model:
        with pytest.raises(Exception):
            await model.request(
                [ModelRequest(parts=[UserPromptPart(content='fix it')])],
                None,
                ModelRequestParameters(),
            )


async def test_a_transport_error_raises() -> None:
    """
    An unreachable Ollama must surface, never degrade silently.
    """

    def handler(request: Request) -> Response:
        raise ConnectError('connection refused', request=request)

    async with _mocked_model(handler) as model:
        with pytest.raises(Exception):
            await model.request(
                [ModelRequest(parts=[UserPromptPart(content='fix it')])],
                None,
                ModelRequestParameters(),
            )


async def test_tool_history_round_trips_through_tool_role_messages() -> None:
    """
    A follow-up request replays the assistant tool call and its result.
    """
    seen_bodies: list[dict[str, Any]] = []

    def handler(request: Request) -> Response:
        seen_bodies.append(json.loads(request.content))
        return Response(200, json=_chat_response('all read'))

    async with _mocked_model(handler) as model:
        await model.request(
            [
                ModelRequest(parts=[UserPromptPart(content='read it')]),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='read_file', args={'path': 'main.py'})]
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='read_file',
                            content='file contents',
                            tool_call_id='call-1',
                        )
                    ]
                ),
            ],
            None,
            ModelRequestParameters(),
        )

    assert seen_bodies[0]['messages'] == [
        {'role': 'user', 'content': 'read it'},
        {
            'role': 'assistant',
            'tool_calls': [{'function': {'name': 'read_file', 'arguments': {'path': 'main.py'}}}],
        },
        {'role': 'tool', 'content': 'file contents', 'tool_name': 'read_file'},
    ]


def test_nothing_loaded_requests_the_required_window() -> None:
    """
    Without a running instance there is nothing to preserve.
    """
    assert decide_num_ctx(None, 32_768) == 32_768


def test_a_satisfied_window_sends_no_override() -> None:
    """
    A loaded window at or above the requirement must not force a reload.
    """
    assert decide_num_ctx(32_768, 32_768) is None
    assert decide_num_ctx(65_536, 32_768) is None


def test_a_smaller_loaded_window_requests_the_larger_window() -> None:
    """
    Only a genuinely larger requirement justifies a reload.
    """
    assert decide_num_ctx(4_096, 32_768) == 32_768
