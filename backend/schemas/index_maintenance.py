from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


class IndexStatusResponse(BaseModel):
    """
    What is currently indexed for a project.

    Answers the two things the settings control needs: whether to offer itself
    at all, and what it would remove if used.
    """

    project: Annotated[str, Field(description='The normalised path identifying the project.')]
    indexed_files: Annotated[int, Field(ge=0, description='How many files are indexed for it.')]
    can_delete: Annotated[
        bool, Field(description='Whether deletion is available right now.')
    ]
    blocked_reason: Annotated[
        str,
        Field(
            description=(
                'Why deletion is unavailable, in words meant for the reader. Empty when it is '
                'available. Sent from here rather than composed in the interface so that the '
                'explanation cannot drift from the condition that produced it.'
            )
        ),
    ] = ''


class IndexDeletionResponse(BaseModel):
    """
    The outcome of a deletion, in terms the interface can report directly.
    """

    project: Annotated[str, Field(description='The project that was cleared.')]
    embeddings_deleted: Annotated[int, Field(ge=0)]
    files_deleted: Annotated[int, Field(ge=0)]
    symbols_deleted: Annotated[int, Field(ge=0)]
    message: Annotated[
        str,
        Field(
            description=(
                'A sentence describing what happened, including the case where there was '
                'nothing to remove.'
            )
        ),
    ]
