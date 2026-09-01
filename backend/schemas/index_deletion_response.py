from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


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
