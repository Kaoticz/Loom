from __future__ import annotations

import pytest

from database.models.code_embedding import CodeEmbedding
from database.models.embedding_model import EmbeddingModel
from database.models.file import File
from database.models.symbol import Symbol
from services.index_deletion import (
    IndexDeletionSummary,
    count_project_index,
    delete_project_index,
)
from services.projects import normalise_project_path

import os

pytestmark = pytest.mark.usefixtures('require_database')

MINE: str = normalise_project_path('/repos/mine')
THEIRS: str = normalise_project_path('/repos/theirs')


async def seed(project: str, model: EmbeddingModel, path: str = 'src/main.py') -> File:
    """
    Index one file for a project under one model, with a symbol and an embedding.

    :param project: The project the file belongs to.
    :param model: The embedding model to store the embedding under.
    :param path: The repository-relative path.
    :returns: The file that was created.
    """
    file_obj: File = await File.create(
        project=project, file_path=path, raw_content='def thing():\n    pass\n', hash='h'
    )
    await Symbol.create(
        file=file_obj, symbol_name='thing', kind='function', line_start=1, line_end=2
    )
    await CodeEmbedding.create(
        model=model, file=file_obj, chunk_text='def thing()', embedding=[0.1, 0.2, 0.3], start_line=1
    )
    return file_obj


@pytest.mark.usefixtures('db')
async def test_deleting_one_project_leaves_the_other_untouched() -> None:
    """
    Seeded with two projects on purpose.

    A test seeding one project would pass against an implementation that
    deleted everything, which is the mistake most worth catching here.
    """
    model: EmbeddingModel = await EmbeddingModel.create(model_name='probe-3', dimensions=3)
    await seed(MINE, model)
    await seed(THEIRS, model)

    summary: IndexDeletionSummary = await delete_project_index(MINE)

    assert summary.embeddings_deleted == 1
    assert summary.files_deleted == 1
    assert summary.symbols_deleted == 1

    assert await File.filter(project=MINE).count() == 0
    assert await File.filter(project=THEIRS).count() == 1
    assert await Symbol.filter(file__project=THEIRS).count() == 1
    assert await CodeEmbedding.filter(file__project=THEIRS).count() == 1


@pytest.mark.usefixtures('db')
async def test_every_models_index_for_the_project_is_removed() -> None:
    """
    A project's index is one thing, whichever model produced each part of it.

    Decided by the user in favour of simplicity: keeping a file because some
    other model still pointed at it made the outcome depend on history the
    reader could not see. The consequence is deliberate — re-indexing under any
    model rebuilds what it needs.
    """
    current: EmbeddingModel = await EmbeddingModel.create(model_name='probe-3', dimensions=3)
    other: EmbeddingModel = await EmbeddingModel.create(model_name='other-3', dimensions=3)

    file_obj: File = await seed(MINE, current)
    await CodeEmbedding.create(
        model=other, file=file_obj, chunk_text='other', embedding=[0.4, 0.5, 0.6], start_line=1
    )
    # A second project holds the other model too, and must survive.
    await seed(THEIRS, other, path='src/other.py')

    summary: IndexDeletionSummary = await delete_project_index(MINE)

    assert summary.embeddings_deleted == 2, 'both models\' embeddings for this project'
    assert summary.files_deleted == 1
    assert await File.filter(project=MINE).count() == 0
    assert await CodeEmbedding.filter(file__project=MINE).count() == 0

    # Scoped where it counts: the other project keeps everything.
    assert await File.filter(project=THEIRS).count() == 1
    assert await CodeEmbedding.filter(file__project=THEIRS).count() == 1


@pytest.mark.usefixtures('db')
async def test_deleting_twice_reports_zeroes_rather_than_failing() -> None:
    """
    A doubled click must not produce a confusing error.
    """
    model: EmbeddingModel = await EmbeddingModel.create(model_name='probe-3', dimensions=3)
    await seed(MINE, model)

    first: IndexDeletionSummary = await delete_project_index(MINE)
    second: IndexDeletionSummary = await delete_project_index(MINE)

    assert first.files_deleted == 1
    assert second.embeddings_deleted == 0
    assert second.files_deleted == 0
    assert second.symbols_deleted == 0


@pytest.mark.usefixtures('db')
async def test_a_project_with_nothing_indexed_removes_nothing() -> None:
    """
    Deleting where there is no index is a no-op, not an error.
    """
    model: EmbeddingModel = await EmbeddingModel.create(model_name='probe-3', dimensions=3)
    await seed(MINE, model)

    summary: IndexDeletionSummary = await delete_project_index(
        normalise_project_path('/repos/nowhere')
    )

    assert summary.embeddings_deleted == 0
    assert summary.files_deleted == 0
    assert await File.filter(project=MINE).count() == 1


@pytest.mark.usefixtures('db')
async def test_counting_is_scoped_to_the_project() -> None:
    """
    The count decides whether the control is offered at all.
    """
    model: EmbeddingModel = await EmbeddingModel.create(model_name='probe-3', dimensions=3)
    await seed(MINE, model)
    await seed(THEIRS, model)

    assert await count_project_index(MINE) == 1
    assert await count_project_index(THEIRS) == 1
    assert await count_project_index(normalise_project_path('/repos/nowhere')) == 0
