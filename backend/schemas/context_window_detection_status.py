from __future__ import annotations

from enum import Enum


class ContextWindowDetectionStatus(str, Enum):
    """
    How a context-window detection request was resolved.
    """

    detected = 'detected'
    failed = 'failed'
    not_applicable = 'not_applicable'
