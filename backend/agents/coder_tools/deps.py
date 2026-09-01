from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from services.code_search import CodeSearchService
from services.docker_sandbox import DockerSandboxManager
from services.embedding import EmbeddingService
from services.events import EventBus
from services.file_editor import FileEditorService
from services.git_manager import GitManagerService
from services.indexing import IndexingService
from services.session_context import SessionContext
from services.user_questioner import UserQuestioner


class AgentDeps(BaseModel):
    """
    Dependencies made available to every Coder agent tool.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    docker: DockerSandboxManager
    search: CodeSearchService
    editor: FileEditorService
    git: GitManagerService
    indexing: IndexingService
    embedding: EmbeddingService
    events: EventBus
    session: SessionContext
    questioner: UserQuestioner | None = None
    """
    How a tool puts a question to the user, when the session supplies one.

    Optional because the tools are also driven directly by tests and by any
    caller with no user to ask; `ask_user` says so plainly rather than hanging.
    """
