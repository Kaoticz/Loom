from __future__ import annotations

import pytest

from database.models.file import File
from database.models.task import Task
from tests.conftest import TEST_PROJECT, managed_tables

pytestmark = pytest.mark.usefixtures('require_database')

"""
That one test cannot see another's rows.

Isolation is by truncation of a table list, and that list was hand-written until
it disagreed with the schema: `tasks` was missing, so task rows survived into the
next test with nothing to show it. These tests exist because that failure is
invisible — nothing raises, a later test simply sees rows it did not create.

Written as ordered pairs, one leaving rows behind and the next requiring none,
because that is the only way to observe a fixture failing to clean up.
"""


@pytest.mark.usefixtures('db')
async def test_every_declared_table_is_managed() -> None:
    """
    The truncation list is derived from the models, so it cannot fall behind.

    Asserted on a table that was genuinely missed rather than on the mechanism,
    so this fails if the derivation is ever replaced by a literal list again.
    """
    tables: list[str] = managed_tables()

    assert 'tasks' in tables
    assert 'files' in tables
    assert 'code_embeddings' in tables


@pytest.mark.usefixtures('db')
async def test_leaves_rows_behind_for_the_next_test() -> None:
    """
    Deliberately dirty: the test after this one requires a clean database.
    """
    await Task.create(prompt='left behind', repo_path='/repos/mine')
    await File.create(
        project=TEST_PROJECT, file_path='left/behind.py', raw_content='x\n', hash='h'
    )

    assert await Task.all().count() == 1
    assert await File.all().count() == 1


@pytest.mark.usefixtures('db')
async def test_sees_none_of_the_previous_test_s_rows() -> None:
    """
    The pair to the test above. Without truncation of `tasks`, this found one.
    """
    assert await Task.all().count() == 0
    assert await File.all().count() == 0
