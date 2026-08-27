from __future__ import annotations

from typing import Any

ARCHITECTURE_KEY: str = 'general.architecture'
CONTEXT_LENGTH_SUFFIX: str = '.context_length'


def positive_int(value: object) -> int | None:
    """
    Return a positive integer, rejecting booleans and other values.

    :param value: The value read from provider metadata.
    :returns: The positive integer, or None when the value is unusable.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def context_length_from_model_info(model_info: dict[str, Any]) -> int | None:
    """
    Read a model's context length from Ollama's architecture-keyed metadata.

    :param model_info: The `model_info` object returned by `/api/show`.
    :returns: The reported context length, or None when it is absent or malformed.
    """
    architecture: Any = model_info.get(ARCHITECTURE_KEY)
    if not isinstance(architecture, str) or not architecture:
        return None
    return positive_int(model_info.get(f'{architecture}{CONTEXT_LENGTH_SUFFIX}'))
