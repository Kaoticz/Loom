from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from tortoise import Tortoise

from services.code_search import CodeSearchService
from tests.conftest import TEST_PROJECT, TEST_REPO_PATH
from tests.test_code_search import OTHER_PROJECT

pytestmark = pytest.mark.usefixtures('require_database')


# `db` now lives in conftest, so more than one module can use it.


@pytest.mark.usefixtures('db')
async def test_vectors_insert_and_rank_by_distance(search: CodeSearchService) -> None:
    """
    Embeddings store as vectors and semantic search orders by distance.
    """
    from database.models.code_embedding import CodeEmbedding
    from database.models.embedding_model import EmbeddingModel
    from database.models.file import File

    model: EmbeddingModel = await EmbeddingModel.create(model_name='probe-3', dimensions=3)
    users: File = await File.create(project=TEST_PROJECT, file_path='repo/users.py', raw_content='def find_user(): pass\n', hash='h1')
    orders: File = await File.create(project=TEST_PROJECT, file_path='repo/orders.py', raw_content='def place_order(): pass\n', hash='h2')
    await CodeEmbedding.create(model=model, file=users, chunk_text='def find_user(uid): ...', embedding=[1.0, 0.0, 0.0], start_line=7)
    await CodeEmbedding.create(model=model, file=orders, chunk_text='def place_order(o): ...', embedding=[0.0, 1.0, 0.0], start_line=3)

    near: list[CodeEmbedding] = await search.search_code_semantic([0.9, 0.1, 0.0], 'probe-3', limit=2)
    assert len(near) == 2
    assert near[0].file.file_path == 'repo/users.py'
    assert near[0].start_line == 7
    assert 'find_user' in near[0].chunk_text

    far: list[CodeEmbedding] = await search.search_code_semantic([0.0, 0.9, 0.1], 'probe-3', limit=1)
    assert far[0].file.file_path == 'repo/orders.py'


@pytest.mark.usefixtures('db')
async def test_unknown_model_returns_no_results(search: CodeSearchService) -> None:
    """
    Searching with a model that was never registered yields nothing, not an error.
    """
    assert await search.search_code_semantic([0.1, 0.2, 0.3], 'never-registered') == []


@pytest.mark.usefixtures('db')
async def test_mixed_dimensions_no_longer_break_search(search: CodeSearchService) -> None:
    """
    Vectors from a differently sized model are excluded rather than fatal.

    A dimensionless column accepts any width, so before embeddings were scoped
    to a model this combination failed every query with
    'different vector dimensions'.
    """
    from database.models.code_embedding import CodeEmbedding
    from database.models.embedding_model import EmbeddingModel
    from database.models.file import File

    small: EmbeddingModel = await EmbeddingModel.create(model_name='small-3', dimensions=3)
    large: EmbeddingModel = await EmbeddingModel.create(model_name='large-4', dimensions=4)
    file_obj: File = await File.create(project=TEST_PROJECT, file_path='a.py', raw_content='x', hash='h')

    await CodeEmbedding.create(model=small, file=file_obj, chunk_text='three', embedding=[1.0, 0.0, 0.0], start_line=1)
    await CodeEmbedding.create(model=large, file=file_obj, chunk_text='four', embedding=[1.0, 0.0, 0.0, 0.5], start_line=2)

    results: list[CodeEmbedding] = await search.search_code_semantic([0.9, 0.1, 0.0], 'small-3', limit=5)
    assert len(results) == 1
    assert results[0].chunk_text == 'three'

    other: list[CodeEmbedding] = await search.search_code_semantic([0.9, 0.1, 0.0, 0.2], 'large-4', limit=5)
    assert len(other) == 1
    assert other[0].chunk_text == 'four'


@pytest.mark.usefixtures('db')
async def test_deleting_a_model_cascades_to_its_embeddings(search: CodeSearchService) -> None:
    """
    Removing a model removes the embeddings that only it can interpret.
    """
    from database.models.code_embedding import CodeEmbedding
    from database.models.embedding_model import EmbeddingModel
    from database.models.file import File

    model: EmbeddingModel = await EmbeddingModel.create(model_name='doomed', dimensions=3)
    file_obj: File = await File.create(project=TEST_PROJECT, file_path='a.py', raw_content='x', hash='h')
    await CodeEmbedding.create(model=model, file=file_obj, chunk_text='c', embedding=[1.0, 0.0, 0.0], start_line=1)

    await model.delete()
    assert await CodeEmbedding.all().count() == 0


@pytest.mark.usefixtures('db')
async def test_deleting_a_file_cascades_to_its_embeddings() -> None:
    """
    Removing a file removes its chunks' embeddings.
    """
    from database.models.code_embedding import CodeEmbedding
    from database.models.embedding_model import EmbeddingModel
    from database.models.file import File

    model: EmbeddingModel = await EmbeddingModel.create(model_name='keeper', dimensions=3)
    file_obj: File = await File.create(project=TEST_PROJECT, file_path='a.py', raw_content='x', hash='h')
    await CodeEmbedding.create(model=model, file=file_obj, chunk_text='c', embedding=[1.0, 0.0, 0.0], start_line=1)

    await file_obj.delete()
    assert await CodeEmbedding.all().count() == 0
    assert await EmbeddingModel.all().count() == 1


@pytest.mark.usefixtures('db')
async def test_ilike_is_case_insensitive_on_postgres(search: CodeSearchService) -> None:
    """
    Lexical search is case-insensitive on Postgres, where LIKE is not.

    Asserted against PostgreSQL, whose LIKE is case-sensitive, so this can only be
    proven against a real Postgres instance.
    """
    from database.models.file import File

    await File.create(project=TEST_PROJECT, file_path='repo/users.py', raw_content='def find_user(): pass\n', hash='h1')
    hits = await search.search_code_lexical('FIND_USER')
    assert len(hits) > 0
    assert hits[0].file_path == 'repo/users.py'


@pytest.mark.usefixtures('db')
async def test_changed_vector_width_discards_stale_embeddings(repo: str) -> None:
    """
    A model that starts producing a different width has its entry corrected.

    Its existing embeddings can no longer be compared with anything it produces,
    so they are discarded rather than left to poison future searches.
    """
    from database.models.code_embedding import CodeEmbedding
    from database.models.embedding_model import EmbeddingModel
    from database.models.file import File
    from services.indexing import IndexingService
    from tests.conftest import stub_embedding_service

    model: EmbeddingModel = await EmbeddingModel.create(model_name='nomic-embed-text', dimensions=4)
    file_obj: File = await File.create(project=TEST_PROJECT, file_path='a.py', raw_content='x', hash='h')
    await CodeEmbedding.create(
        model=model, file=file_obj, chunk_text='stale', embedding=[1.0, 0.0, 0.0, 0.5], start_line=1
    )

    service: IndexingService = IndexingService(
        repo_path=repo, embedding=stub_embedding_service([0.1, 0.2, 0.3])
    )
    refreshed: EmbeddingModel = await service.resolve_embedding_model()

    assert refreshed.dimensions == 3
    assert await CodeEmbedding.all().count() == 0
    assert await EmbeddingModel.all().count() == 1


@pytest.mark.usefixtures('db')
async def test_indexing_a_repository_end_to_end(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Indexing a real directory populates files, symbols and embeddings.

    Exercises the whole pipeline against a real database rather than stubs,
    which is where the ast-grep and vector-binding faults were hiding.
    """
    import os
    from pathlib import Path

    from database.models.code_embedding import CodeEmbedding
    from database.models.file import File
    from database.models.symbol import Symbol
    from services.indexing import IndexingService
    from tests.conftest import ast_grep_available, stub_embedding_service

    binary = os.getenv('AST_GREP_BIN', 'ast-grep')
    if not ast_grep_available(binary):
        pytest.skip('ast-grep is not installed')
    monkeypatch.setenv('AST_GREP_BIN', binary)

    repo: Path = tmp_path_factory.mktemp('index_repo')
    for index in range(6):
        (repo / f'mod_{index}.py').write_text(
            f'class C{index}:\n    def m{index}(self):\n        return {index}\n'
        )
    (repo / 'binary.bin').write_bytes(b'\x00\x01\x02')
    (repo / 'empty.py').write_text('   \n')
    (repo / '.venv').mkdir()
    (repo / '.venv' / 'ignored.py').write_text('class ShouldNotAppear:\n    pass\n')

    updates: list[tuple[int, int, str]] = []

    async def progress(current: int, total: int, message: str) -> None:
        updates.append((current, total, message))

    service: IndexingService = IndexingService(
        repo_path=str(repo), embedding=stub_embedding_service([0.1, 0.2, 0.3])
    )
    await service.index_repository(progress_callback=progress)

    # Progress never goes backwards and always reaches the total.
    counts: list[int] = [current for current, _, _ in updates]
    assert counts == sorted(counts)
    assert updates[-1][0] == updates[-1][1]

    # Phase detail reaches the client, and throttling keeps the volume sane.
    # Which specific phases survive throttling is timing dependent, so this only
    # asserts that phase reporting happens at all; the exact messages are
    # covered by the reporter's own tests, where the clock is controlled.
    messages: str = ' '.join(message for _, _, message in updates)
    assert 'symbols' in messages or 'embeddings' in messages
    assert 'Indexing complete' in messages
    assert len(updates) < 20

    assert await File.all().count() == 6, 'binary and empty files must be skipped'
    assert await File.filter(file_path__contains='.venv').count() == 0
    assert await Symbol.all().count() == 12, 'a class and a method per file'
    assert await CodeEmbedding.all().count() == 6


@pytest.mark.usefixtures('db')
async def test_changed_file_replaces_its_old_symbols(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Editing a file discards the symbols and embeddings from its previous version.

    Leaving them behind would let search return definitions that no longer exist.
    Needs the full schema, since re-indexing clears the embeddings table.
    """
    from asyncio import Semaphore
    from pathlib import Path
    from typing import Any
    from unittest.mock import MagicMock

    from database.models.file import File
    from database.models.symbol import Symbol
    from services.indexing import IndexingService
    from services.indexing_progress import ProgressReporter
    from tests.conftest import stub_embedding_service

    async def noop(*args: Any, **kwargs: Any) -> None:
        return None

    async def one_symbol(self: IndexingService, file_obj: File, file_path: str) -> None:
        await Symbol.create(
            file=file_obj, symbol_name=file_obj.raw_content.strip(),
            kind='function', line_start=1, line_end=1
        )

    monkeypatch.setattr(IndexingService, '_index_embeddings', noop)
    monkeypatch.setattr(IndexingService, '_index_symbols', one_symbol)

    repo: Path = tmp_path_factory.mktemp('changing')
    path: Path = repo / 'changing.py'
    indexing: IndexingService = IndexingService(
        repo_path=str(repo), embedding=stub_embedding_service([0.1, 0.2, 0.3])
    )
    model: Any = MagicMock()

    path.write_text('first\n')
    await indexing._process_file(str(path), str(repo), Semaphore(1), model, ProgressReporter(1, None))
    assert [s.symbol_name for s in await Symbol.all()] == ['first']

    path.write_text('second\n')
    await indexing._process_file(str(path), str(repo), Semaphore(1), model, ProgressReporter(1, None))
    assert [s.symbol_name for s in await Symbol.all()] == ['second']


@pytest.mark.usefixtures('db')
async def test_semantic_search_never_returns_another_project(
    search: CodeSearchService,
) -> None:
    """
    The semantic path is scoped like every other read.

    Tested separately from the lexical paths on purpose: scoping one and not
    another produces no error, only another project's code appearing in the
    results, which is worse than the collision this all exists to prevent.
    """
    from database.models.code_embedding import CodeEmbedding
    from database.models.embedding_model import EmbeddingModel
    from database.models.file import File

    model: EmbeddingModel = await EmbeddingModel.create(model_name='probe-3', dimensions=3)
    mine: File = await File.create(
        project=TEST_PROJECT, file_path='src/main.py',
        raw_content='def shared_name(): pass\n', hash='h1',
    )
    theirs: File = await File.create(
        project=OTHER_PROJECT, file_path='src/main.py',
        raw_content='def shared_name(): pass\n', hash='h2',
    )
    await CodeEmbedding.create(
        model=model, file=mine, chunk_text='MINE', embedding=[1.0, 0.0, 0.0], start_line=1
    )
    await CodeEmbedding.create(
        model=model, file=theirs, chunk_text='THEIRS', embedding=[1.0, 0.0, 0.0], start_line=1
    )

    found: list[CodeEmbedding] = await search.search_code_semantic([1.0, 0.0, 0.0], 'probe-3', limit=10)

    assert [row.chunk_text for row in found] == ['MINE']
