from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


class ContextWindowDetectionRequest(BaseModel):
    """
    The model whose context-window setting should be detected.
    """

    model: Annotated[str, Field(min_length=1, description='The configured model identifier.')]
