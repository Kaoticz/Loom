from __future__ import annotations

import pytest

from services.chunking import (
    Chunk,
    SymbolSpan,
    build_chunks,
    extend_over_decorators,
    window_chunks,
)


def spans(*pairs: tuple[int, int]) -> list[SymbolSpan]:
    """
    Build symbol spans from start and end pairs.

    :param pairs: The 1-based inclusive line ranges.
    :returns: The spans.
    """
    return [SymbolSpan(start_line=start, end_line=end) for start, end in pairs]


def covers(chunks: list[Chunk], total: int) -> bool:
    """
    Check that chunks tile every line exactly once, in order.

    :param chunks: The chunks to check.
    :param total: How many lines the file has.
    :returns: True if the chunks form a contiguous cover with no gaps or overlaps.
    """
    cursor: int = 1
    for chunk in chunks:
        if chunk.start_line != cursor:
            return False
        cursor = chunk.end_line + 1
    return cursor == total + 1


def test_no_symbols_falls_back_to_windows() -> None:
    """
    A file with no recognised symbols is split into fixed windows.

    Languages with no registered patterns must still be searchable.
    """
    lines: list[str] = [f'line {index}' for index in range(1, 26)]
    chunks: list[Chunk] = build_chunks(lines, [], max_lines=10)

    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 10), (11, 20), (21, 25)]
    assert covers(chunks, 25)


def test_symbols_are_never_split_across_chunks() -> None:
    """
    No chunk boundary falls inside a definition.

    Fixed windows cut roughly a quarter of real definitions in half, so a chunk
    ends up holding the tail of one function and the head of the next.
    """
    lines: list[str] = [f'line {index}' for index in range(1, 61)]
    symbols: list[SymbolSpan] = spans((5, 20), (25, 45), (48, 58))

    chunks: list[Chunk] = build_chunks(lines, symbols, max_lines=25)

    for symbol in symbols:
        holder = [c for c in chunks if c.start_line <= symbol.start_line <= c.end_line]
        assert len(holder) == 1
        assert holder[0].end_line >= symbol.end_line, f'{symbol} was split'


def test_gaps_between_symbols_are_still_covered() -> None:
    """
    Imports and module level code belong to no symbol but stay searchable.
    """
    lines: list[str] = [f'line {index}' for index in range(1, 41)]
    chunks: list[Chunk] = build_chunks(lines, spans((10, 20), (30, 38)), max_lines=15)

    assert covers(chunks, 40)
    assert chunks[0].start_line == 1


def test_short_symbols_are_packed_together() -> None:
    """
    Consecutive small definitions share a chunk.

    A chunk per symbol would produce many embeddings with almost no signal;
    a third of real symbols are only a few lines long.
    """
    lines: list[str] = [f'line {index}' for index in range(1, 21)]
    symbols: list[SymbolSpan] = spans((1, 3), (4, 6), (7, 9), (10, 12))

    chunks: list[Chunk] = build_chunks(lines, symbols, max_lines=40)

    # All four definitions share one chunk; the trailing lines that belong to no
    # symbol form their own.
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 12)
    assert len(chunks) == 2
    assert covers(chunks, 20)


def test_packing_respects_the_size_budget() -> None:
    """
    Packing stops before a chunk grows past the limit.
    """
    lines: list[str] = [f'line {index}' for index in range(1, 41)]
    symbols: list[SymbolSpan] = spans((1, 10), (11, 20), (21, 30), (31, 40))

    chunks: list[Chunk] = build_chunks(lines, symbols, max_lines=20)

    assert all(chunk.length <= 20 for chunk in chunks)
    assert covers(chunks, 40)


def test_nested_symbols_are_not_stored_twice() -> None:
    """
    A method is covered by its class, not emitted separately as well.

    Chunking both would embed every method twice and make search return
    overlapping results.
    """
    lines: list[str] = [f'line {index}' for index in range(1, 31)]
    symbols: list[SymbolSpan] = spans((1, 20), (3, 8), (10, 18))

    chunks: list[Chunk] = build_chunks(lines, symbols, max_lines=25)

    assert covers(chunks, 30)
    assert any(c.start_line == 1 and c.end_line >= 20 for c in chunks)


def test_oversized_class_is_split_into_its_methods() -> None:
    """
    A class too large to embed whole is divided at its methods.

    Splitting at arbitrary offsets instead would lose exactly the structure that
    makes the result readable.
    """
    lines: list[str] = [f'line {index}' for index in range(1, 101)]
    symbols: list[SymbolSpan] = spans((1, 100), (5, 30), (32, 60), (62, 95))

    chunks: list[Chunk] = build_chunks(lines, symbols, max_lines=40)

    boundaries: set[int] = {c.start_line for c in chunks}
    assert {5, 32, 62}.issubset(boundaries), f'methods not used as boundaries: {boundaries}'
    assert covers(chunks, 100)


def test_oversized_symbol_without_children_falls_back_to_windows() -> None:
    """
    A very long function with nothing nested inside is windowed.
    """
    lines: list[str] = [f'line {index}' for index in range(1, 101)]
    chunks: list[Chunk] = build_chunks(lines, spans((1, 100)), max_lines=30)

    assert all(chunk.length <= 30 for chunk in chunks)
    assert covers(chunks, 100)


def test_decorators_are_kept_with_their_definition() -> None:
    """
    A decorated definition's chunk includes its decorators.

    ast-grep reports the range as starting at `def`, so the decorator would
    otherwise be separated from the function it describes, losing what
    distinguishes a fixture from a plain function.
    """
    lines: list[str] = [
        'import pytest',
        '',
        '@pytest.fixture',
        '@another',
        'def thing():',
        '    return 1',
    ]
    chunks: list[Chunk] = build_chunks(lines, spans((5, 6)), max_lines=40)

    holder = [c for c in chunks if c.start_line <= 5 <= c.end_line][0]
    assert holder.start_line <= 3, 'decorators were left out of the chunk'
    assert '@pytest.fixture' in holder.text(lines)


def test_decorator_extension_stops_at_ordinary_code() -> None:
    """
    Only decorators directly above a definition are absorbed into it.
    """
    lines: list[str] = ['x = 1', '@decorator', 'def thing():', '    pass']
    extended: SymbolSpan = extend_over_decorators(
        SymbolSpan(start_line=3, end_line=4), lines
    )

    assert (extended.start_line, extended.end_line) == (2, 4)


def test_chunk_text_matches_its_range() -> None:
    """
    A chunk renders exactly the lines it claims.
    """
    lines: list[str] = ['alpha', 'beta', 'gamma', 'delta']
    assert Chunk(start_line=2, end_line=3).text(lines) == 'beta\ngamma'


@pytest.mark.parametrize('total', [0, 1, 7, 80, 81, 200])
def test_every_line_is_covered_for_any_file_size(total: int) -> None:
    """
    Chunks tile the whole file whatever its length.

    A line that falls outside every chunk is invisible to semantic search.
    """
    lines: list[str] = [f'line {index}' for index in range(1, total + 1)]
    chunks: list[Chunk] = build_chunks(lines, spans((2, 4)) if total > 4 else [])

    assert covers(chunks, total) if total else chunks == []


def test_empty_run_yields_no_windows() -> None:
    """
    An empty range produces nothing rather than a zero-length chunk.
    """
    assert window_chunks(5, 4, 10) == []


def test_zero_chunk_size_is_rejected() -> None:
    """
    A chunk must be allowed at least one line.
    """
    with pytest.raises(ValueError):
        build_chunks(['a'], [], max_lines=0)


def test_symbols_beyond_the_file_are_ignored() -> None:
    """
    A stale symbol range pointing past the end of the file is discarded.

    Symbols are re-extracted when a file changes, but a range must never be
    trusted to still fit the content it is applied to.
    """
    lines: list[str] = ['a', 'b', 'c']
    chunks: list[Chunk] = build_chunks(lines, spans((10, 20)), max_lines=10)

    assert covers(chunks, 3)
