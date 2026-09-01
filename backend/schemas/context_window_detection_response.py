from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from .context_window_detection_status import ContextWindowDetectionStatus


class ContextWindowDetectionResponse(BaseModel):
    """
    The outcome of attempting to detect a model's context window.
    """

    context_window: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description='The detected context window, or null when unavailable or inapplicable.',
        ),
    ]
    status: Annotated[ContextWindowDetectionStatus, Field(description='How the value was resolved.')]
