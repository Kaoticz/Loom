from __future__ import annotations

from protocol_export import ARTEFACT_PATH, protocol_vocabulary, render_artefact


def test_the_artefact_on_disk_is_current() -> None:
    """
    The committed artefact matches the enums as they stand.

    Changing an enum without regenerating would let the frontend fall behind
    silently, which is the drift this artefact exists to catch. Run
    `python scripts/export_protocol.py` and commit the result.
    """
    assert ARTEFACT_PATH.exists(), f'{ARTEFACT_PATH} is missing; run scripts/export_protocol.py'
    assert ARTEFACT_PATH.read_text() == render_artefact(), (
        'The protocol artefact is stale. Run scripts/export_protocol.py and commit the result.'
    )


def test_every_vocabulary_is_non_empty() -> None:
    """
    An empty vocabulary would make the frontend's matching assertion vacuous.
    """
    for name, values in protocol_vocabulary().items():
        assert values, f'{name} exported no values'


def test_values_are_unique_within_a_vocabulary() -> None:
    """
    Two members sharing a value would make one of them unreachable over the wire.
    """
    for name, values in protocol_vocabulary().items():
        assert len(values) == len(set(values)), f'{name} contains duplicate values'
