from __future__ import annotations

from collections.abc import Sequence
from services.chunk import Chunk
from services.symbol_span import SymbolSpan

DEFAULT_MAX_CHUNK_LINES: int = 150


def extend_over_decorators(span: SymbolSpan, lines: Sequence[str]) -> SymbolSpan:
    """
    Grow a span upwards to include the decorators attached to it.

    ast-grep reports a decorated definition as beginning at `def` or `class`,
    excluding the decorators above it. A chunk built from that range would drop
    exactly the context that distinguishes a fixture from a plain function, or a
    property from a method.

    :param span: The span reported for the symbol.
    :param lines: The file's lines, without terminators.
    :returns: The span, extended over any decorators immediately above it.
    """
    start: int = span.start_line
    while start > 1:
        candidate: str = lines[start - 2].strip()
        if not candidate.startswith('@'):
            break
        start -= 1

    if start == span.start_line:
        return span
    return SymbolSpan(start_line=start, end_line=span.end_line)


def window_chunks(start_line: int, end_line: int, max_lines: int) -> list[Chunk]:
    """
    Split a plain run of lines into fixed size chunks.

    Used for the parts of a file that belong to no symbol, and for files whose
    language has no symbol patterns registered.

    :param start_line: First line of the run, 1-based.
    :param end_line: Last line of the run, inclusive.
    :param max_lines: The greatest number of lines a chunk may hold.
    :returns: The chunks covering the run, empty if the run is empty.
    """
    if end_line < start_line:
        return []

    chunks: list[Chunk] = []
    cursor: int = start_line
    while cursor <= end_line:
        last: int = min(cursor + max_lines - 1, end_line)
        chunks.append(Chunk(start_line=cursor, end_line=last))
        cursor = last + 1
    return chunks


def build_chunks(
    lines: Sequence[str],
    symbols: Sequence[SymbolSpan],
    max_lines: int = DEFAULT_MAX_CHUNK_LINES,
) -> list[Chunk]:
    """
    Divide a file into chunks that respect symbol boundaries.

    Fixed line windows cut roughly a quarter of real definitions in half, so a
    chunk routinely holds the tail of one function and the head of the next.
    Aligning to symbols keeps each chunk to something a reader would recognise.

    Symbols are packed together rather than emitted one per chunk: many
    definitions are only a few lines long, and a chunk each would produce
    embeddings with almost no signal. Nested symbols are covered by their parent
    rather than stored twice, unless the parent is too large to embed whole, in
    which case its children become the chunks.

    :param lines: The file's lines, without terminators.
    :param symbols: The spans of the symbols found in the file, in any order.
    :param max_lines: The greatest number of lines a chunk may hold.
    :returns: Chunks covering every line of the file, in order.
    :raises ValueError: If the chunk size is not positive.
    """
    if max_lines < 1:
        raise ValueError('A chunk must be allowed at least one line.')

    total: int = len(lines)
    if total == 0:
        return []

    usable: list[SymbolSpan] = [
        extend_over_decorators(span, lines)
        for span in symbols
        if span.start_line <= total and span.end_line <= total and span.end_line >= span.start_line
    ]
    if not usable:
        return window_chunks(1, total, max_lines)

    top_level: list[SymbolSpan] = sorted(
        (span for span in usable if not any(other.contains(span) for other in usable)),
        key=lambda span: (span.start_line, -span.end_line),
    )

    chunks: list[Chunk] = []
    cursor: int = 1
    index: int = 0

    while index < len(top_level):
        span: SymbolSpan = top_level[index]

        # Anything before this symbol belongs to no symbol: imports, constants,
        # module level statements. It still has to be searchable.
        if span.start_line > cursor:
            chunks.extend(window_chunks(cursor, span.start_line - 1, max_lines))
            cursor = span.start_line

        if span.length > max_lines:
            chunks.extend(_split_large_symbol(span, usable, max_lines))
            cursor = span.end_line + 1
            index += 1
            continue

        # Pack as many following symbols as fit, so short definitions do not
        # each become their own thin embedding.
        last: int = span.end_line
        index += 1
        while index < len(top_level):
            candidate: SymbolSpan = top_level[index]
            if candidate.end_line - span.start_line + 1 > max_lines:
                break
            last = candidate.end_line
            index += 1

        chunks.append(Chunk(start_line=span.start_line, end_line=last))
        cursor = last + 1

    if cursor <= total:
        chunks.extend(window_chunks(cursor, total, max_lines))

    return chunks


def _split_large_symbol(
    parent: SymbolSpan,
    symbols: Sequence[SymbolSpan],
    max_lines: int,
) -> list[Chunk]:
    """
    Break a symbol that is too large to embed whole.

    Its own nested definitions are preferred as the split points, since a class
    divided into its methods is far more useful than one divided at arbitrary
    line offsets. A symbol with no children falls back to fixed windows.

    :param parent: The oversized symbol.
    :param symbols: Every symbol found in the file.
    :param max_lines: The greatest number of lines a chunk may hold.
    :returns: The chunks covering the parent's range.
    """
    children: list[SymbolSpan] = sorted(
        (span for span in symbols if parent.contains(span)),
        key=lambda span: span.start_line,
    )
    if not children:
        return window_chunks(parent.start_line, parent.end_line, max_lines)

    # Keep only the outermost children, so a method inside a nested class is not
    # emitted separately from that class.
    outermost: list[SymbolSpan] = [
        span for span in children if not any(other.contains(span) for other in children)
    ]

    chunks: list[Chunk] = []
    cursor: int = parent.start_line
    for child in outermost:
        if child.start_line > cursor:
            chunks.extend(window_chunks(cursor, child.start_line - 1, max_lines))
        if child.length > max_lines:
            chunks.extend(window_chunks(child.start_line, child.end_line, max_lines))
        else:
            chunks.append(Chunk(start_line=child.start_line, end_line=child.end_line))
        cursor = child.end_line + 1

    if cursor <= parent.end_line:
        chunks.extend(window_chunks(cursor, parent.end_line, max_lines))

    return chunks
