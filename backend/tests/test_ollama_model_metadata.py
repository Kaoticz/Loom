from typing import Any

import pytest

from services.ollama_model_metadata import num_ctx_from_parameters


@pytest.mark.parametrize(
    ('parameters', 'expected'),
    [
        ('temperature 0.7\nnum_ctx 8192', 8192),
        ('num_ctx    131072\ntop_p 0.9', 131_072),
        ('temperature 0.7', None),
        ('', None),
        (None, None),
        ('num_ctx', None),
        ('num_ctx zero', None),
        ('num_ctx 0', None),
        ('num_ctx -8192', None),
        ('num_ctx 8192 extra', None),
    ],
)
def test_num_ctx_from_parameters_accepts_only_a_positive_integer(
    parameters: Any, expected: int | None
) -> None:
    """
    Only a well-formed explicit context parameter counts as an override.

    :param parameters: The serialized Ollama parameter text to parse.
    :param expected: The expected override, if any.
    """
    assert num_ctx_from_parameters(parameters) == expected
