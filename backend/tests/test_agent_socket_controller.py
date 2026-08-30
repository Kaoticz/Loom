from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import WebSocketDisconnect

from api.v1.agent_socket_controller import AgentSocketController
from services.code_search import CodeSearchService
from services.docker_sandbox import DockerSandboxManager
from services.file_editor import FileEditorService
from services.git_manager import GitManagerService
from services.indexing import IndexingService
from services.session_registry import SessionRegistry
from tests.conftest import TEST_REPO_PATH, stub_embedding_service


class RecordingSocket:
    """
    A WebSocket stand-in that records what the controller does to it.
    """

    inbound: list[str]
    sent: list[dict[str, Any]]
    accepted: bool
    closed: bool
    disconnect_after: int | None
    _reads: int

    def __init__(self, inbound: list[str], disconnect_after: int | None = None) -> None:
        """
        Create a recording socket.

        :param inbound: Messages the client will send, in order.
        :param disconnect_after: Raise a disconnect once this many reads have happened.
        """
        self.inbound = list(inbound)
        self.sent = []
        self.accepted = False
        self.closed = False
        self.disconnect_after = disconnect_after
        self._reads = 0

    async def accept(self) -> None:
        """
        Record that the connection was accepted.
        """
        self.accepted = True

    async def receive_text(self) -> str:
        """
        Return the next scripted message.

        :returns: The message text.
        :raises WebSocketDisconnect: When the scripted disconnect point is reached.
        """
        self._reads += 1
        if self.disconnect_after is not None and self._reads > self.disconnect_after:
            raise WebSocketDisconnect(code=1000)
        if not self.inbound:
            raise WebSocketDisconnect(code=1000)
        return self.inbound.pop(0)

    async def send_json(self, payload: dict[str, Any]) -> None:
        """
        Record a message sent to the client.

        :param payload: The serialised message.
        """
        self.sent.append(payload)

    async def close(self) -> None:
        """
        Record that the connection was closed.
        """
        self.closed = True


def build_controller(tmp_path: Path) -> tuple[AgentSocketController, list[str]]:
    """
    Assemble a controller whose services are harmless in tests.

    :param tmp_path: A scratch directory to root the services at.
    :returns: The controller and the list recording sandbox teardown calls.
    """
    teardown: list[str] = []

    docker: DockerSandboxManager = DockerSandboxManager(repo_path=str(tmp_path))

    async def stop_sandbox() -> None:
        teardown.append('stopped')

    docker.stop_sandbox = stop_sandbox  # type: ignore[method-assign]

    indexing: Any = MagicMock(spec=IndexingService)

    async def index(progress_callback: Any = None) -> None:
        return None

    indexing.index_repository = index

    controller: AgentSocketController = AgentSocketController(
        docker=docker,
        search=CodeSearchService(repo_path=TEST_REPO_PATH),
        editor=FileEditorService(root_path=str(tmp_path)),
        git=GitManagerService(repo_path=str(tmp_path)),
        indexing=indexing,
        embedding=stub_embedding_service(),
        registry=SessionRegistry(),
    )
    return controller, teardown


async def test_socket_is_closed_when_the_session_ends(tmp_path: Path) -> None:
    """
    The connection is closed once the session finishes.

    Leaving it open strands the client waiting for messages that never come.
    """
    controller, teardown = build_controller(tmp_path)
    socket: RecordingSocket = RecordingSocket([json.dumps({'prompt': ''})])

    await controller.handle_session(socket)  # type: ignore[arg-type]

    assert socket.accepted is True
    assert socket.closed is True
    assert teardown == ['stopped']


async def test_failure_is_reported_before_closing(tmp_path: Path) -> None:
    """
    A session that cannot start tells the client why.
    """
    controller, _ = build_controller(tmp_path)
    socket: RecordingSocket = RecordingSocket([json.dumps({'prompt': ''})])

    await controller.handle_session(socket)  # type: ignore[arg-type]

    kinds: list[str] = [message.get('type', '') for message in socket.sent]
    assert 'error' in kinds
    assert socket.closed is True


async def test_disconnect_mid_session_still_cleans_up(tmp_path: Path) -> None:
    """
    A client that vanishes does not leak the sandbox.
    """
    controller, teardown = build_controller(tmp_path)
    socket: RecordingSocket = RecordingSocket([], disconnect_after=0)

    await controller.handle_session(socket)  # type: ignore[arg-type]

    assert teardown == ['stopped']
    assert socket.closed is True


async def test_close_failure_does_not_propagate(tmp_path: Path) -> None:
    """
    A socket that refuses to close does not turn into an unhandled error.
    """
    controller, teardown = build_controller(tmp_path)
    socket: RecordingSocket = RecordingSocket([json.dumps({'prompt': ''})])

    async def refuse() -> None:
        raise RuntimeError('already closed')

    socket.close = refuse  # type: ignore[method-assign]

    await controller.handle_session(socket)  # type: ignore[arg-type]
    assert teardown == ['stopped']


async def test_sandbox_teardown_failure_does_not_prevent_close(tmp_path: Path) -> None:
    """
    A sandbox that will not stop still leaves the connection closed.
    """
    controller, _ = build_controller(tmp_path)

    async def failing_stop() -> None:
        raise RuntimeError('docker is unreachable')

    controller.docker.stop_sandbox = failing_stop  # type: ignore[method-assign]
    socket: RecordingSocket = RecordingSocket([json.dumps({'prompt': ''})])

    await controller.handle_session(socket)  # type: ignore[arg-type]
    assert socket.closed is True


async def test_a_second_connection_is_refused(tmp_path: Path) -> None:
    """
    While a session runs, another connection is told rather than starting one.
    """
    controller, _ = build_controller(tmp_path)
    await controller.registry.acquire('already-running')

    socket: RecordingSocket = RecordingSocket([json.dumps({'prompt': 'x'})])
    await controller.handle_session(socket)  # type: ignore[arg-type]

    assert len(socket.sent) == 1
    message = socket.sent[0]
    assert message['type'] == 'session_busy'
    assert 'another tab' in message['content']
    assert message['active_session_id'] == 'already-running'
    assert socket.closed is True


async def test_a_refused_connection_leaves_the_slot_alone(tmp_path: Path) -> None:
    """
    Refusing a connection must not release the running session's claim.
    """
    controller, teardown = build_controller(tmp_path)
    await controller.registry.acquire('already-running')

    await controller.handle_session(RecordingSocket([json.dumps({'prompt': 'x'})]))  # type: ignore[arg-type]

    assert controller.registry.active_session_id == 'already-running'
    # The refused connection has no sandbox of its own to tear down.
    assert teardown == []


async def test_the_slot_is_released_when_a_session_ends(tmp_path: Path) -> None:
    """
    A finished session frees the slot for the next one.
    """
    controller, _ = build_controller(tmp_path)
    socket: RecordingSocket = RecordingSocket([json.dumps({'prompt': ''})])

    await controller.handle_session(socket)  # type: ignore[arg-type]

    assert controller.registry.is_busy is False


async def test_the_slot_is_released_even_when_a_session_crashes(tmp_path: Path) -> None:
    """
    A crash must not leave the slot held, or no session could ever start again.
    """
    controller, _ = build_controller(tmp_path)

    async def exploding_index(progress_callback: Any = None) -> None:
        raise RuntimeError('indexing blew up')

    controller.indexing.index_repository = exploding_index  # type: ignore[method-assign]
    socket: RecordingSocket = RecordingSocket([
        json.dumps({'prompt': 'do something', 'repo_path': str(tmp_path)})
    ])

    await controller.handle_session(socket)  # type: ignore[arg-type]

    assert controller.registry.is_busy is False
