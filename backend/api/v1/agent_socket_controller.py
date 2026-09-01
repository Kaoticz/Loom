from asyncio import Event, Lock
from api.v1.agent_session import AgentSession
from fastapi import WebSocket, WebSocketDisconnect
from agents.coder import AgentDeps
from schemas.agent_messages import SessionBusyMessage, to_message
from services.code_search import CodeSearchService
from services.conversation_history import ConversationHistory
from services.docker_sandbox import DockerSandboxManager
from services.embedding import EmbeddingService
from services.events import AgentEvent, ErrorEvent, EventBus
from services.file_editor import FileEditorService
from services.git_manager import GitManagerService
from services.indexing import IndexingService
from services.session_context import SessionContext
from services.session_fsm import SessionStateMachine
from services.session_registry import SessionRegistry
from services.session_limits import IterationLimits
import logging
import traceback
import uuid

logger = logging.getLogger(__name__)


class AgentSocketController:
    """
    Controller for handling agent WebSocket sessions.
    """
    docker: DockerSandboxManager
    search: CodeSearchService
    editor: FileEditorService
    git: GitManagerService
    indexing: IndexingService
    embedding: EmbeddingService
    registry: SessionRegistry

    def __init__(
        self,
        docker: DockerSandboxManager,
        search: CodeSearchService,
        editor: FileEditorService,
        git: GitManagerService,
        indexing: IndexingService,
        embedding: EmbeddingService,
        registry: SessionRegistry,
    ) -> None:
        """
        Create the controller.

        :param docker: Sandbox manager used to run verification commands.
        :param search: Code retrieval service.
        :param editor: File editing service.
        :param git: Repository manager.
        :param indexing: Repository indexer.
        :param embedding: Embedding backend.
        :param registry: Tracks which session currently holds the single slot.
        """
        self.docker = docker
        self.search = search
        self.editor = editor
        self.git = git
        self.indexing = indexing
        self.embedding = embedding
        self.registry = registry

    async def handle_session(self, websocket: WebSocket) -> None:
        """
        Handle a WebSocket session for real-time agent interaction.

        The session itself is driven by a state machine in `AgentSession`. This
        method owns only the connection: wiring the event bus to the socket,
        reporting failures, and cleaning up afterwards.

        :param websocket: The client connection.
        """
        await websocket.accept()

        session_id: str = str(uuid.uuid4())[:8]

        if not await self.registry.acquire(session_id):
            # Two sessions on one repository would both branch and commit to it.
            await websocket.send_json(
                SessionBusyMessage(
                    content=(
                        'A Loom session is already running in another tab. Switch to it to '
                        'continue, or close it and retry here.'
                    ),
                    active_session_id=self.registry.active_session_id or '',
                ).model_dump(mode='json')
            )
            await websocket.close()
            return

        websocket_lock: Lock = Lock()
        events: EventBus = EventBus(session_id=session_id)
        connection_lost: Event = Event()

        async def forward_event(event: AgentEvent) -> None:
            """
            Forward an event emitted during the session to the client.

            A send that fails means the client has gone. The bus deliberately
            isolates handler failures, so the flag is what lets the session
            notice; without it the agent would keep working for nobody.

            :param event: The event to serialise and send.
            """
            async with websocket_lock:
                try:
                    await websocket.send_json(to_message(event).model_dump(mode='json'))
                except Exception:
                    connection_lost.set()
                    raise

        events.subscribe(forward_event)

        session: SessionContext = SessionContext()
        deps: AgentDeps = AgentDeps(
            docker=self.docker,
            search=self.search,
            editor=self.editor,
            git=self.git,
            indexing=self.indexing,
            embedding=self.embedding,
            events=events,
            session=session
        )

        driver: AgentSession = AgentSession(
            websocket=websocket,
            services=self,
            session_id=session_id,
            events=events,
            deps=deps,
            session=session,
            history=ConversationHistory(),
            machine=SessionStateMachine(),
            limits=IterationLimits.from_environment(),
            connection_lost=connection_lost
        )

        try:
            await driver.run()
        except WebSocketDisconnect:
            logger.info('The client disconnected from session %s.', session_id)
        except Exception as error:
            logger.error('Session %s failed: %s', session_id, traceback.format_exc())
            try:
                await events.emit(ErrorEvent(content=f'The session failed: {error}'))
            except Exception:
                logger.warning('Could not report the failure of session %s.', session_id)
            try:
                await driver.mark_failed()
            except Exception:
                logger.warning('Could not mark the task for session %s as failed.', session_id)
        finally:
            await self.registry.release(session_id)
            events.clear()
            try:
                await self.docker.stop_sandbox()
            except Exception:
                logger.warning('Could not stop the sandbox for session %s.', session_id)

            # Leaving the socket open would keep the client waiting for messages
            # that will never arrive.
            try:
                await websocket.close()
            except Exception:
                logger.debug('Session %s was already disconnected.', session_id)
