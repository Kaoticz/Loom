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
import os

class MainProvider(Provider):
    """
    Main dependency provider for Dishka.
    """
    scope: Any = Scope.APP

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
    ) -> AgentSocketController:
        """
        Provide an instance of AgentSocketController with injected services.
        """
        return AgentSocketController(docker, search, editor, git, indexing)

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
        Provide an instance of CodeSearchService.
        """
        return CodeSearchService()

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

    @provide
    def get_indexing_service(self) -> IndexingService:
        """
        Provide an instance of IndexingService.
        """
        repo_path: str = os.getenv('REPO_PATH', '/tmp/repo')
        return IndexingService(repo_path=repo_path)
