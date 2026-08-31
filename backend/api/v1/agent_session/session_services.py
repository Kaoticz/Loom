from typing import Protocol

from services.code_search import CodeSearchService
from services.docker_sandbox import DockerSandboxManager
from services.embedding import EmbeddingService
from services.file_editor import FileEditorService
from services.git_manager import GitManagerService
from services.indexing import IndexingService
from services.session_registry import SessionRegistry


class SessionServices(Protocol):
    """
    Define the services a session needs from its controller.

    Declared structurally so the session can be typed against the controller
    without importing it, which would be circular. Typing this as `Any` would
    silence the checkers on every service access.
    """

    docker: DockerSandboxManager
    editor: FileEditorService
    git: GitManagerService
    indexing: IndexingService
    embedding: EmbeddingService
    search: CodeSearchService
    registry: SessionRegistry
