from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class ContextWindowDetectionStatus(str, Enum):
    """
    How a context-window detection request was resolved.
    """

    detected = 'detected'
    failed = 'failed'
    not_applicable = 'not_applicable'


class ContextWindowDetectionRequest(BaseModel):
    """
    The model whose context-window setting should be detected.
    """

    model: Annotated[str, Field(min_length=1, description='The configured model identifier.')]


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
