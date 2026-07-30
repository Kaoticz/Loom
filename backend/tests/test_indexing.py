from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from database.models.embedding_model import EmbeddingModel
from database.models.file import File
from database.models.symbol import Symbol
from services.indexing import IndexingService
from tests.conftest import ast_grep_available, stub_embedding_service

SAMPLE_SOURCE: str = '''import os


class Alpha:
    def one(self):
        return 1


@decorator
class Beta(Alpha, object):
    def two(
        self,
        x,
    ):
        return x


def standalone(a, b):
    return a + b
'''


@pytest.fixture
def indexing(repo: str) -> IndexingService:
    """
    Provide an indexing service with a stubbed embedding backend.

    :param repo: The repository root.
    :returns: The indexing service.
    """
    return IndexingService(repo_path=repo, embedding=MagicMock())


def test_ast_grep_invocation_uses_a_file_path(indexing: IndexingService) -> None:
    """
    ast-grep is given a path so it infers the language and reads real content.

    Piping through stdin silently scans the working directory unless both
    `--stdin` and `--lang` are supplied, which is easy to get wrong.
    """
    captured: list[list[str]] = []

    def fake_run(command: Any, **kwargs: Any) -> Any:
        captured.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout='', stderr='')

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(subprocess, 'run', fake_run)
        IndexingService._run_ast_grep('ast-grep', 'rule-document', '/tmp/sample.py')

    command: list[str] = captured[0]
    assert command[command.index('--inline-rules') + 1] == 'rule-document'
    assert command[-1] == '/tmp/sample.py'
    assert '--stdin' not in command
    assert '--lang' not in command


def test_symbol_name_extraction() -> None:
    """
    The captured `$NAME` metavariable is used, not the whole matched body.
    """
    extract = IndexingService._extract_symbol_name
    assert extract({'metaVariables': {'single': {'NAME': {'text': 'Beta'}}}}) == 'Beta'
    assert extract({'text': 'class X: pass'}) is None
    assert extract({'metaVariables': {'single': {}}}) is None
    assert extract({'metaVariables': {'single': {'NAME': {'text': '  '}}}}) is None


@pytest.mark.parametrize(
    'match,expected',
    [
        ({'range': {'start': {'line': 0}, 'end': {'line': 4}}}, (1, 5)),
        ({'range': {'start': {'line': 9}, 'end': {'line': 14}}}, (10, 15)),
        ({}, None),
        ({'range': {'end': {'line': 3}}}, None),
        ({'range': {'start': {'line': 'x'}, 'end': {'line': 3}}}, None),
        ({'range': {'start': {'line': True}, 'end': {'line': 3}}}, None),
        ({'range': {'start': {'line': -1}, 'end': {'line': 3}}}, None),
        ({'range': {'start': {'line': 9}, 'end': {'line': 2}}}, None),
        ({'range': {'start': {}, 'end': {}}}, None),
    ],
    ids=[
        'converts_to_1_based', 'converts_mid_file', 'missing_range', 'missing_start',
        'non_integer', 'bool_line', 'negative_line', 'inverted_range', 'empty_positions',
    ],
)
def test_line_range_extraction(match: dict[str, Any], expected: tuple[int, int] | None) -> None:
    """
    Line ranges are normalised to 1-based, and unusable ranges are rejected.

    Defaulting a missing range to zero would fabricate a plausible-looking
    line 1 that is indistinguishable from a real result.
    """
    assert IndexingService._extract_line_range(match) == expected


@pytest.mark.usefixtures('sqlite_db')
async def test_symbols_are_indexed_from_the_file(
    indexing: IndexingService, repo: str, ast_grep_bin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Symbols come from the file's own contents with correct 1-based line numbers.
    """
    if not ast_grep_available(ast_grep_bin):
        pytest.skip('ast-grep is not installed')
    monkeypatch.setenv('AST_GREP_BIN', ast_grep_bin)

    path: Path = Path(repo, 'sample.py')
    path.write_text(SAMPLE_SOURCE)
    file_obj: File = await File.create(file_path='sample.py', raw_content=SAMPLE_SOURCE, hash='h')

    await indexing._index_symbols(file_obj, str(path))

    rows: list[Symbol] = await Symbol.all().order_by('line_start')
    found: dict[str, str] = {row.symbol_name: row.kind for row in rows}

    assert found == {
        'Alpha': 'class',
        'one': 'function',
        'Beta': 'class',
        'two': 'function',
        'standalone': 'function',
    }

    lines: list[str] = SAMPLE_SOURCE.splitlines()
    for row in rows:
        assert row.symbol_name in lines[row.line_start - 1], row.symbol_name


@pytest.mark.usefixtures('sqlite_db')
async def test_classes_with_and_without_bases_are_indexed_once(
    indexing: IndexingService, repo: str, ast_grep_bin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The two class patterns are disjoint, so no class is recorded twice.
    """
    if not ast_grep_available(ast_grep_bin):
        pytest.skip('ast-grep is not installed')
    monkeypatch.setenv('AST_GREP_BIN', ast_grep_bin)

    source: str = 'class Plain:\n    pass\n\n\nclass Derived(Plain):\n    pass\n'
    path: Path = Path(repo, 'classes.py')
    path.write_text(source)
    file_obj: File = await File.create(file_path='classes.py', raw_content=source, hash='h')

    await indexing._index_symbols(file_obj, str(path))

    names: list[str] = [row.symbol_name for row in await Symbol.all()]
    assert sorted(names) == ['Derived', 'Plain']


def test_explicit_paths_bypass_ignore_rules(tmp_path: Path, ast_grep_bin: str) -> None:
    """
    A gitignored file is still indexed when its path is named explicitly.

    Ignore rules only apply when ast-grep walks a directory, so nothing is
    silently skipped by the per-file invocation.
    """
    if not ast_grep_available(ast_grep_bin):
        pytest.skip('ast-grep is not installed')

    subprocess.run(['git', 'init', '-q', str(tmp_path)], capture_output=True, check=True)
    Path(tmp_path, '.gitignore').write_text('ignored.py\n')
    ignored: Path = tmp_path / 'ignored.py'
    ignored.write_text('class Ignored:\n    pass\n')

    from services.symbol_patterns import build_rules

    process: subprocess.CompletedProcess[str] = IndexingService._run_ast_grep(
        ast_grep_bin, build_rules('python'), str(ignored)
    )
    assert 'Ignored' in process.stdout


@pytest.mark.usefixtures('sqlite_db')
async def test_registers_the_embedding_model_on_first_use(repo: str) -> None:
    """
    The configured model is recorded along with the width of its vectors.
    """
    service: IndexingService = IndexingService(
        repo_path=repo, embedding=stub_embedding_service([0.1, 0.2, 0.3])
    )

    model: EmbeddingModel = await service.resolve_embedding_model()

    assert model.model_name == 'nomic-embed-text'
    assert model.dimensions == 3
    assert model.provider == 'ollama'
    assert await EmbeddingModel.all().count() == 1


@pytest.mark.usefixtures('sqlite_db')
async def test_reuses_an_already_registered_model(repo: str) -> None:
    """
    Re-indexing with the same model reuses its registry entry.
    """
    service: IndexingService = IndexingService(
        repo_path=repo, embedding=stub_embedding_service([0.1, 0.2, 0.3])
    )

    first: EmbeddingModel = await service.resolve_embedding_model()
    second: EmbeddingModel = await service.resolve_embedding_model()

    assert first.id == second.id
    assert await EmbeddingModel.all().count() == 1


@pytest.mark.usefixtures('sqlite_db')
async def test_wrongly_sized_vector_is_refused(repo: str) -> None:
    """
    A vector that does not match the registered width is never stored.

    Storing it would leave a chunk that cannot be compared with the rest of the
    index, which only surfaces later as a failed search.
    """
    from services.embedding import EmbeddingError

    model: EmbeddingModel = await EmbeddingModel.create(model_name='nomic-embed-text', dimensions=768)
    service: IndexingService = IndexingService(
        repo_path=repo, embedding=stub_embedding_service([0.1, 0.2, 0.3])
    )
    file_obj: File = await File.create(file_path='a.py', raw_content='print(1)\n', hash='h')

    with pytest.raises(EmbeddingError, match='3-dimension vector'):
        await service._index_embeddings(file_obj, model)


@pytest.mark.usefixtures('sqlite_db')
async def test_annotated_definitions_are_indexed(
    indexing: IndexingService, repo: str, ast_grep_bin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Functions carrying return annotations are found.

    Spelling the body out in the pattern as 'def $NAME($$$): $$$' silently fails
    to match any annotated function, so in a typed codebase the symbol table
    ends up nearly empty and `get_symbol` finds nothing.
    """
    if not ast_grep_available(ast_grep_bin):
        pytest.skip('ast-grep is not installed')
    monkeypatch.setenv('AST_GREP_BIN', ast_grep_bin)

    source: str = (
        'def plain(a):\n'
        '    return a\n'
        '\n'
        '\n'
        'def annotated(a: int) -> str:\n'
        '    return str(a)\n'
        '\n'
        '\n'
        'async def async_annotated(b: int) -> None:\n'
        '    return None\n'
        '\n'
        '\n'
        'class Derived(Base, metaclass=Meta):\n'
        '    async def method(self, x: int) -> bool:\n'
        '        return True\n'
    )
    path: Path = Path(repo, 'annotated.py')
    path.write_text(source)
    file_obj: File = await File.create(file_path='annotated.py', raw_content=source, hash='h')

    await indexing._index_symbols(file_obj, str(path))

    found: dict[str, str] = {row.symbol_name: row.kind for row in await Symbol.all()}
    assert found == {
        'plain': 'function',
        'annotated': 'function',
        'async_annotated': 'function',
        'method': 'function',
        'Derived': 'class',
    }


@pytest.mark.usefixtures('sqlite_db')
async def test_symbol_range_covers_the_whole_definition(
    indexing: IndexingService, repo: str, ast_grep_bin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A symbol's range spans its body, not just its signature line.
    """
    if not ast_grep_available(ast_grep_bin):
        pytest.skip('ast-grep is not installed')
    monkeypatch.setenv('AST_GREP_BIN', ast_grep_bin)

    source: str = 'def widget(a: int) -> int:\n    x = 1\n    y = 2\n    return x + y + a\n'
    path: Path = Path(repo, 'widget.py')
    path.write_text(source)
    file_obj: File = await File.create(file_path='widget.py', raw_content=source, hash='h')

    await indexing._index_symbols(file_obj, str(path))

    symbol: Symbol = (await Symbol.all())[0]
    assert (symbol.line_start, symbol.line_end) == (1, 4)


@pytest.mark.usefixtures('sqlite_db')
async def test_derived_classes_are_recorded_once(
    indexing: IndexingService, repo: str, ast_grep_bin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A single class pattern covers plain and derived classes without duplicating.
    """
    if not ast_grep_available(ast_grep_bin):
        pytest.skip('ast-grep is not installed')
    monkeypatch.setenv('AST_GREP_BIN', ast_grep_bin)

    source: str = 'class Plain:\n    pass\n\n\nclass Derived(Plain):\n    pass\n'
    path: Path = Path(repo, 'classes2.py')
    path.write_text(source)
    file_obj: File = await File.create(file_path='classes2.py', raw_content=source, hash='h')

    await indexing._index_symbols(file_obj, str(path))

    names: list[str] = sorted(row.symbol_name for row in await Symbol.all())
    assert names == ['Derived', 'Plain']


@pytest.mark.usefixtures('sqlite_db')
async def test_process_file_handles_each_kind_of_file(
    indexing: IndexingService, repo: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    New, unchanged, updated, binary and empty files are each handled.

    Every one must report completion, or progress can never reach the total.
    """
    from asyncio import Semaphore
    from unittest.mock import MagicMock

    from services.indexing_progress import ProgressReporter

    monkeypatch.setattr(
        IndexingService, '_index_symbols',
        lambda self, file_obj, file_path: _noop()
    )
    monkeypatch.setattr(
        IndexingService, '_index_embeddings',
        lambda self, file_obj, model: _noop()
    )

    written: dict[str, Path] = {}
    for name, content in (
        ('fresh.py', 'x = 1\n'),
        ('empty.py', '   \n'),
    ):
        path = Path(repo, name)
        path.write_text(content)
        written[name] = path

    binary: Path = Path(repo, 'blob.bin')
    binary.write_bytes(b'\x00\x01\x02')
    written['blob.bin'] = binary

    reporter: ProgressReporter = ProgressReporter(3, None)
    model: Any = MagicMock()

    for path in written.values():
        await indexing._process_file(str(path), repo, Semaphore(1), model, reporter)

    assert reporter.completed == 3, 'skipped files must still be counted'
    assert await File.all().count() == 1, 'only the real source file is indexed'
    assert (await File.all())[0].file_path == 'fresh.py'


@pytest.mark.usefixtures('sqlite_db')
async def test_unchanged_file_is_not_reindexed(
    indexing: IndexingService, repo: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A file whose content is unchanged is not symbol-scanned or re-embedded.

    Re-embedding an unchanged file wastes an Ollama round trip per chunk.
    """
    from asyncio import Semaphore
    from unittest.mock import MagicMock

    from services.indexing_progress import ProgressReporter

    calls: list[str] = []
    monkeypatch.setattr(
        IndexingService, '_index_symbols',
        lambda self, file_obj, file_path: _record(calls, 'symbols')
    )
    monkeypatch.setattr(
        IndexingService, '_index_embeddings',
        lambda self, file_obj, model: _record(calls, 'embeddings')
    )

    path: Path = Path(repo, 'stable.py')
    path.write_text('x = 1\n')
    model: Any = MagicMock()

    await indexing._process_file(str(path), repo, Semaphore(1), model, ProgressReporter(1, None))
    assert calls == ['symbols', 'embeddings']

    calls.clear()
    await indexing._process_file(str(path), repo, Semaphore(1), model, ProgressReporter(1, None))
    assert calls == [], 'an unchanged file was indexed again'


async def _noop() -> None:
    """
    Stand in for an indexing step that should do nothing.
    """
    return None


async def _record(calls: list[str], name: str) -> None:
    """
    Record that an indexing step ran.

    :param calls: Where to record the call.
    :param name: The step's name.
    """
    calls.append(name)


@pytest.mark.parametrize(
    'value,expected',
    [(None, 1024 * 1024), ('2048', 2048), ('0', 0), ('-1', -1), ('enormous', 1024 * 1024)],
    ids=['default', 'configured', 'unlimited_zero', 'unlimited_negative', 'unusable'],
)
def test_size_limit_is_configurable(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: int
) -> None:
    """
    The size limit is configurable, and can be turned off entirely.
    """
    from services.indexing import max_file_size_bytes

    monkeypatch.delenv('LOOM_MAX_FILE_SIZE_BYTES', raising=False)
    if value is not None:
        monkeypatch.setenv('LOOM_MAX_FILE_SIZE_BYTES', value)

    assert max_file_size_bytes() == expected


@pytest.mark.usefixtures('sqlite_db')
async def test_oversized_files_are_skipped(
    indexing: IndexingService, repo: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A file past the limit is skipped, and still counted as processed.

    The whole file is held in memory and stored in a column, so a large
    generated file would otherwise risk the process.
    """
    from asyncio import Semaphore
    from unittest.mock import MagicMock

    from services.indexing_progress import ProgressReporter

    monkeypatch.setenv('LOOM_MAX_FILE_SIZE_BYTES', '100')
    monkeypatch.setattr(IndexingService, '_index_symbols', lambda self, f, p: _noop())
    monkeypatch.setattr(IndexingService, '_index_embeddings', lambda self, f, m: _noop())

    big: Path = Path(repo, 'big.py')
    big.write_text('x = 1\n' * 200)
    small: Path = Path(repo, 'small.py')
    small.write_text('y = 2\n')

    reporter: ProgressReporter = ProgressReporter(2, None)
    for path in (big, small):
        await indexing._process_file(str(path), repo, Semaphore(1), MagicMock(), reporter)

    assert reporter.completed == 2, 'a skipped file must still count'
    assert [f.file_path for f in await File.all()] == ['small.py']


@pytest.mark.usefixtures('sqlite_db')
async def test_unlimited_size_indexes_everything(
    indexing: IndexingService, repo: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Setting the limit to zero indexes files of any size.
    """
    from asyncio import Semaphore
    from unittest.mock import MagicMock

    from services.indexing_progress import ProgressReporter

    monkeypatch.setenv('LOOM_MAX_FILE_SIZE_BYTES', '0')
    monkeypatch.setattr(IndexingService, '_index_symbols', lambda self, f, p: _noop())
    monkeypatch.setattr(IndexingService, '_index_embeddings', lambda self, f, m: _noop())

    big: Path = Path(repo, 'big.py')
    big.write_text('x = 1\n' * 500)

    await indexing._process_file(
        str(big), repo, Semaphore(1), MagicMock(), ProgressReporter(1, None)
    )

    assert await File.all().count() == 1
