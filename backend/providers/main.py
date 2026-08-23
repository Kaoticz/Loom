from dishka import Provider, Scope, provide
from typing import Annotated, Any
from pydantic import Field
from api.v1.tasks_controller import TaskController
from api.v1.agent_socket_controller import AgentSocketController
from services.docker_sandbox import DockerSandboxManager
from services.code_search import CodeSearchService
from services.file_editor import FileEditorService
from services.git_manager import GitManagerService
from services.indexing import IndexingService
from collections.abc import AsyncIterator
from services.embedding import EmbeddingService, DEFAULT_EMBEDDING_MODEL
from services.session_registry import SessionRegistry
import os

class MainProvider(Provider):
    """
    Main dependency provider for Dishka.
    """
    # REQUEST scope gives every WebSocket connection its own service instances.
    # These services carry per-session state (repository path, sandbox container,
    # embedding model), so sharing them across sessions lets one session
    # repoint another mid-run.
    scope: Any = Scope.SESSION

    @provide
    def get_task_controller(self) -> TaskController:
        """
        Provide an instance of TaskController.
        """
        return TaskController()

    @provide
    def get_agent_socket_controller(
        self,
        docker: DockerSandboxManager,
        search: CodeSearchService,
        editor: FileEditorService,
        git: GitManagerService,
        indexing: IndexingService,
        embedding: EmbeddingService,
        registry: SessionRegistry,
    ) -> AgentSocketController:
        """
        Provide an instance of AgentSocketController with injected services.
        """
        return AgentSocketController(docker, search, editor, git, indexing, embedding, registry)

    @provide
    def get_docker_sandbox_manager(self) -> DockerSandboxManager:
        """
        Provide an instance of DockerSandboxManager.
        """
        repo_path: str = os.getenv('REPO_PATH', '/tmp/repo')
        return DockerSandboxManager(repo_path=repo_path)

    @provide
    def get_code_search_service(self) -> CodeSearchService:
        """
        Provide an instance of CodeSearchService, bound to the current project.
        """
        repo_path: str = os.getenv('REPO_PATH', '/tmp/repo')
        return CodeSearchService(repo_path=repo_path)

    @provide
    def get_file_editor_service(self) -> FileEditorService:
        """
        Provide an instance of FileEditorService.
        """
        root_path: str = os.getenv('REPO_PATH', '/tmp/repo')
        return FileEditorService(root_path=root_path)

    @provide
    def get_git_manager_service(self) -> GitManagerService:
        """
        Provide an instance of GitManagerService.
        """
        repo_path: str = os.getenv('REPO_PATH', '/tmp/repo')
        return GitManagerService(repo_path=repo_path)

    @provide(scope=Scope.APP)
    def get_session_registry(self) -> SessionRegistry:
        """
        Provide the process-wide registry of the active session.

        Deliberately application scoped: a registry created per connection could
        never see the session held by another connection.
        """
        return SessionRegistry()

    @provide
    async def get_embedding_service(self) -> AsyncIterator[EmbeddingService]:
        """
        Provide an instance of EmbeddingService, closed when the session ends.

        The service owns an HTTP connection pool. Now that one is created per
        session rather than per process, leaving it open would leak sockets for
        every connection the user makes.
        """
        ollama_url: str = os.getenv('OLLAMA_BASE_URL', 'http://ollama:11434')
        embedding_model: str = os.getenv('EMBEDDING_MODEL', DEFAULT_EMBEDDING_MODEL)
        service: EmbeddingService = EmbeddingService(
            ollama_url=ollama_url, embedding_model=embedding_model
        )
        try:
            yield service
        finally:
            await service.aclose()

    @provide
    def get_indexing_service(self, embedding: EmbeddingService) -> IndexingService:
        """
        Provide an instance of IndexingService.
        """
        repo_path: str = os.getenv('REPO_PATH', '/tmp/repo')
        return IndexingService(repo_path=repo_path, embedding=embedding)
