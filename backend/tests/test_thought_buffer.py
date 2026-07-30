from __future__ import annotations

from collections.abc import Sequence

import pytest

from services.thought_buffer import ThoughtBuffer


def test_partial_text_is_withheld() -> None:
    """
    Tokens are held back until they form something worth sending.
    """
    buffer: ThoughtBuffer = ThoughtBuffer()
    assert buffer.add('I will ', 0.0) == []
    assert buffer.add('start by ', 0.05) == []
    assert buffer.pending == 'I will start by '


def test_sentence_boundary_releases_a_thought() -> None:
    """
    A completed sentence is sent as one message.
    """
    buffer: ThoughtBuffer = ThoughtBuffer()
    buffer.add('First I will read the file. ', 0.0)
    assert buffer.add('Then', 0.01) == []

    released: list[str] = buffer.add(' I will edit it. ', 0.02)
    assert released == ['Then I will edit it.']


def test_paragraph_break_releases_a_thought() -> None:
    """
    A blank line ends a thought even without sentence punctuation.
    """
    buffer: ThoughtBuffer = ThoughtBuffer()
    assert buffer.add('Step one\n\n', 0.0) == ['Step one']


def test_multiple_boundaries_release_in_order() -> None:
    """
    A chunk containing several boundaries yields several thoughts, in order.
    """
    buffer: ThoughtBuffer = ThoughtBuffer()
    released: list[str] = buffer.add('One. Two. Three and counting', 0.0)
    assert released == ['One.', 'Two.']
    assert buffer.pending == 'Three and counting'


def test_debounce_releases_text_that_has_waited() -> None:
    """
    Text with no boundary is still sent once the debounce interval passes.

    Otherwise a model that writes a long unpunctuated passage would leave the
    user staring at nothing.
    """
    buffer: ThoughtBuffer = ThoughtBuffer(debounce_seconds=0.2)
    assert buffer.add('thinking about it', 1.0) == []
    assert buffer.add(' some more', 1.1) == []
    assert buffer.add(' and more', 1.25) == ['thinking about it some more and more']
    assert buffer.is_empty


def test_debounce_does_not_fire_early() -> None:
    """
    Text is not released before the interval has elapsed.
    """
    buffer: ThoughtBuffer = ThoughtBuffer(debounce_seconds=0.2)
    assert buffer.add('partial', 5.0) == []
    assert buffer.add(' text', 5.19) == []


def test_flush_returns_the_remainder() -> None:
    """
    Whatever is left when the stream ends is still delivered.
    """
    buffer: ThoughtBuffer = ThoughtBuffer()
    buffer.add('a trailing fragment', 0.0)
    assert buffer.flush() == 'a trailing fragment'
    assert buffer.flush() is None


def test_whitespace_only_content_is_not_sent() -> None:
    """
    Blank chunks never become empty chat messages.
    """
    buffer: ThoughtBuffer = ThoughtBuffer(debounce_seconds=0.0)
    assert buffer.add('   \n  ', 0.0) == []
    assert buffer.flush() is None


def test_token_stream_produces_few_messages() -> None:
    """
    A realistic token stream collapses into a handful of readable thoughts.

    The previous implementation sent one message per chunk, flooding the client.
    """
    stream: str = (
        'I ,will ,read ,the ,file ,first. ,'
        'Then ,I ,will ,apply ,the ,edit. ,'
        'Finally ,I ,will ,run ,the ,tests.'
    )
    tokens: Sequence[str] = stream.split(',')

    buffer: ThoughtBuffer = ThoughtBuffer(debounce_seconds=10.0)
    thoughts: list[str] = []
    for index, token in enumerate(tokens):
        thoughts.extend(buffer.add(token, index * 0.01))
    remaining: str | None = buffer.flush()
    if remaining is not None:
        thoughts.append(remaining)

    assert len(tokens) == 18
    assert thoughts == [
        'I will read the file first.',
        'Then I will apply the edit.',
        'Finally I will run the tests.',
    ]


def test_negative_debounce_rejected() -> None:
    """
    A negative debounce interval is invalid.
    """
    with pytest.raises(ValueError):
        ThoughtBuffer(debounce_seconds=-1.0)
