from __future__ import annotations

from typing import Annotated, cast

from database.models.code_embedding import CodeEmbedding
from database.models.file import File
from database.models.symbol import Symbol
from pydantic import BaseModel, Field


class IndexDeletionSummary(BaseModel):
    """
    What a deletion removed.

    Reported rather than assumed, so the interface can say what happened instead
    of claiming a fixed outcome.
    """

    embeddings_deleted: Annotated[int, Field(description='Embeddings removed.')]
    files_deleted: Annotated[int, Field(description='Indexed files removed.')]
    symbols_deleted: Annotated[int, Field(description='Symbols removed with those files.')]


async def count_project_index(project: str) -> int:
    """
    Count the files indexed for one project.

    Counted per project and not per embedding model, to match what deletion
    removes. A count that asked about one model could report nothing to delete
    while the project still held another model's index, and the control would be
    disabled over data that is about to be thrown away by the next deletion.

    :param project: The normalised project path.
    :returns: How many files are indexed for the project.
    """
    return await File.filter(project=project).count()


async def delete_project_index(project: str) -> IndexDeletionSummary:
    """
    Remove everything indexed for one project.

    Every embedding model's work for this project goes, not only the one
    currently configured. That is the user's decision, taken for simplicity: a
    project's index is one thing to delete rather than several, and keeping a
    file because some other model still points at it made the outcome depend on
    history the reader cannot see.

    Other projects are untouched, which is the scoping that matters here.

    Safe to repeat: a second call finds nothing and reports zeroes rather than
    failing, so a doubled click cannot produce an error.

    :param project: The normalised project path.
    :returns: What was removed.
    """
    # Resolved to identifiers first. A filter that crosses a relation compiles
    # to a DELETE with a join, which Postgres refuses outright.
    file_ids: list[int] = cast(
        'list[int]', await File.filter(project=project).values_list('id', flat=True)
    )

    if not file_ids:
        return IndexDeletionSummary(embeddings_deleted=0, files_deleted=0, symbols_deleted=0)

    # Counted before deleting, because the cascade takes them and there would be
    # nothing left to count afterwards.
    embeddings: int = await CodeEmbedding.filter(file_id__in=file_ids).count()
    symbols: int = await Symbol.filter(file_id__in=file_ids).count()
    files: int = await File.filter(id__in=file_ids).delete()

    return IndexDeletionSummary(
        embeddings_deleted=embeddings, files_deleted=files, symbols_deleted=symbols
    )
