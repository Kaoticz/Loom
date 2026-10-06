import json
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest
from httpx import ConnectError, MockTransport, Request, Response
from ollama import AsyncClient as OllamaAsyncClient
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
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


def _ndjson(lines: list[dict[str, Any]]) -> bytes:
    """
    Encode native chat chunks as a streaming response body.

    :param lines: One payload per streamed line.
    :returns: The NDJSON body.
    """
    return '\n'.join(json.dumps(line) for line in lines).encode()


def _stream_chunk(content: str, *, done: bool = False) -> dict[str, Any]:
    """
    Build one native streamed `/api/chat` line.

    :param content: The content fragment for the line.
    :param done: Whether this line ends the stream.
    :returns: The line payload.
    """
    line: dict[str, Any] = {
        'model': 'qwen2.5-coder',
        'created_at': '2026-10-06T00:00:00Z',
        'message': {'role': 'assistant', 'content': content},
        'done': done,
    }
    if done:
        line['done_reason'] = 'stop'
        line['prompt_eval_count'] = 10
        line['eval_count'] = 5
    return line


async def test_streamed_text_arrives_as_start_and_delta_events() -> None:
    """
    Incremental content maps to the start/delta sequence the session consumes.
    """

    def handler(request: Request) -> Response:
        return Response(
            200, content=_ndjson([_stream_chunk('Hel'), _stream_chunk('lo'), _stream_chunk('', done=True)])
        )

    async with _mocked_model(handler) as model:
        async with model.request_stream(
            [ModelRequest(parts=[UserPromptPart(content='fix it')])],
            None,
            ModelRequestParameters(),
        ) as streamed:
            events: list[ModelResponseStreamEvent] = [event async for event in streamed]
            final: ModelResponse = streamed.get()

    starts: list[PartStartEvent] = [event for event in events if isinstance(event, PartStartEvent)]
    text_deltas: list[TextPartDelta] = [
        event.delta
        for event in events
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta)
    ]
    assert len(starts) == 1
    assert isinstance(starts[0].part, TextPart)
    assert starts[0].part.content == 'Hel'
    assert ''.join(delta.content_delta for delta in text_deltas) == 'lo'
    assert len(final.parts) == 1
    assert isinstance(final.parts[0], TextPart)
    assert final.parts[0].content == 'Hello'
    assert final.usage.input_tokens == 10
    assert final.usage.output_tokens == 5


async def test_streamed_tool_calls_arrive_whole_in_one_chunk() -> None:
    """
    Ollama reports streamed tool calls complete, not as incremental deltas.
    """

    def handler(request: Request) -> Response:
        return Response(
            200,
            content=_ndjson(
                [
                    _stream_chunk(''),
                    {
                        'model': 'qwen2.5-coder',
                        'created_at': '2026-10-06T00:00:00Z',
                        'message': {
                            'role': 'assistant',
                            'content': '',
                            'tool_calls': [
                                {'function': {'name': 'read_file', 'arguments': {'path': 'main.py'}}}
                            ],
                        },
                        'done': False,
                    },
                    _stream_chunk('', done=True),
                ]
            ),
        )

    async with _mocked_model(handler) as model:
        async with model.request_stream(
            [ModelRequest(parts=[UserPromptPart(content='read it')])],
            None,
            ModelRequestParameters(),
        ) as streamed:
            events: list[ModelResponseStreamEvent] = [event async for event in streamed]
            final: ModelResponse = streamed.get()

    starts: list[PartStartEvent] = [event for event in events if isinstance(event, PartStartEvent)]
    assert len(starts) == 1
    assert isinstance(starts[0].part, ToolCallPart)
    assert starts[0].part.tool_name == 'read_file'
    assert starts[0].part.args == {'path': 'main.py'}
    assert len(final.parts) == 1
    assert isinstance(final.parts[0], ToolCallPart)
    assert final.parts[0].args == {'path': 'main.py'}


async def test_a_failed_stream_connect_raises() -> None:
    """
    A stream that never starts must surface, never read as empty.
    """

    def handler(request: Request) -> Response:
        return Response(500, text='boom')

    async with _mocked_model(handler) as model:
        async with model.request_stream(
            [ModelRequest(parts=[UserPromptPart(content='fix it')])],
            None,
            ModelRequestParameters(),
        ) as streamed:
            with pytest.raises(Exception):
                async for _ in streamed:
                    pass


async def test_a_malformed_stream_line_raises() -> None:
    """
    A stream with an unparseable line must surface mid-stream.
    """

    def handler(request: Request) -> Response:
        return Response(200, content=_ndjson([_stream_chunk('Hel')]) + b'\nnot json')

    async with _mocked_model(handler) as model:
        async with model.request_stream(
            [ModelRequest(parts=[UserPromptPart(content='fix it')])],
            None,
            ModelRequestParameters(),
        ) as streamed:
            with pytest.raises(Exception):
                async for _ in streamed:
                    pass


async def test_a_stream_without_a_final_chunk_raises() -> None:
    """
    A stream cut off before completion must not read as a complete response.
    """

    def handler(request: Request) -> Response:
        return Response(200, content=_ndjson([_stream_chunk('Hel'), _stream_chunk('lo')]))

    async with _mocked_model(handler) as model:
        async with model.request_stream(
            [ModelRequest(parts=[UserPromptPart(content='fix it')])],
            None,
            ModelRequestParameters(),
        ) as streamed:
            with pytest.raises(UnexpectedModelBehavior):
                async for _ in streamed:
                    pass


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
