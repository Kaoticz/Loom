from __future__ import annotations

import os
from collections.abc import AsyncIterator
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
import pytest_asyncio

from database.setup import SchemaSetupError, apply_migrations, ensure_vector_extension

pytestmark = pytest.mark.skipif(
    not os.getenv('TEST_DATABASE_URL'),
    reason='migrations run against Postgres; set TEST_DATABASE_URL to run',
)

SCRATCH_DATABASE: str = 'loom_migration_test'
EXPECTED_TABLES: set[str] = {
    'aerich',
    'code_embeddings',
    'embedding_models',
    'files',
    'symbols',
    'task_memory',
    'tasks',
}


def dsn_for(database: str) -> str:
    """
    Rewrite the test connection string to point at another database.

    :param database: The database name to target.
    :returns: The rewritten connection string.
    """
    parts = urlsplit(os.environ['TEST_DATABASE_URL'])
    return urlunsplit((parts.scheme, parts.netloc, f'/{database}', '', ''))


@pytest_asyncio.fixture
async def scratch_database() -> AsyncIterator[str]:
    """
    Provide an empty database that migrations can be applied to.

    :returns: An iterator yielding the connection string of the scratch database.
    """
    admin = await asyncpg.connect(dsn_for('postgres'))
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {SCRATCH_DATABASE}')
        await admin.execute(f'CREATE DATABASE {SCRATCH_DATABASE}')
    finally:
        await admin.close()

    yield dsn_for(SCRATCH_DATABASE)

    admin = await asyncpg.connect(dsn_for('postgres'))
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {SCRATCH_DATABASE}')
    finally:
        await admin.close()


def config_for(dsn: str) -> dict[str, object]:
    """
    Build a Tortoise configuration pointing at a specific database.

    :param dsn: The connection string to use.
    :returns: The Tortoise configuration.
    """
    return {
        'connections': {'default': dsn},
        'apps': {
            'models': {
                'models': ['database.models', 'aerich.models'],
                'default_connection': 'default',
            }
        },
    }


async def table_names(dsn: str) -> set[str]:
    """
    List the tables present in a database.

    :param dsn: The connection string of the database to inspect.
    :returns: The table names in the public schema.
    """
    connection = await asyncpg.connect(dsn)
    try:
        rows = await connection.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )
    finally:
        await connection.close()
    return {row['tablename'] for row in rows}


async def test_migrations_build_the_whole_schema(scratch_database: str) -> None:
    """
    Applying migrations to an empty database produces the full schema.

    The schema is no longer generated from the models, so the migrations have to
    be sufficient on their own.
    """
    await ensure_vector_extension(scratch_database)
    applied: list[str] = await apply_migrations(config_for(scratch_database))

    assert len(applied) == 3
    assert applied == sorted(applied), 'migrations must apply in order'
    assert await table_names(scratch_database) == EXPECTED_TABLES


async def test_applying_twice_is_a_no_op(scratch_database: str) -> None:
    """
    A second run applies nothing and leaves the schema intact.

    Startup runs migrations every time, so re-applying has to be harmless.
    """
    await ensure_vector_extension(scratch_database)
    await apply_migrations(config_for(scratch_database))

    assert await apply_migrations(config_for(scratch_database)) == []
    assert await table_names(scratch_database) == EXPECTED_TABLES


async def test_vector_extension_is_enabled(scratch_database: str) -> None:
    """
    The pgvector extension is created before any migration needs it.

    The first migration declares a `vector` column, which cannot be created
    without the extension already present.
    """
    await ensure_vector_extension(scratch_database)

    connection = await asyncpg.connect(scratch_database)
    try:
        version = await connection.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
    finally:
        await connection.close()

    assert version is not None


async def test_enabling_the_extension_twice_is_safe(scratch_database: str) -> None:
    """
    Enabling the extension is idempotent, since startup does it every time.
    """
    await ensure_vector_extension(scratch_database)
    await ensure_vector_extension(scratch_database)


async def test_unreachable_database_is_reported_clearly() -> None:
    """
    A database that cannot be reached produces an explanatory error.

    Starting up against a schema that could not be verified would be worse than
    failing outright.
    """
    with pytest.raises(SchemaSetupError, match='pgvector'):
        await ensure_vector_extension('postgres://nobody:nobody@127.0.0.1:1/nothing')


async def test_broken_migration_directory_is_reported() -> None:
    """
    A missing migration directory fails loudly rather than silently skipping.
    """
    with pytest.raises(SchemaSetupError, match='No migration directory'):
        await apply_migrations(
            config_for(dsn_for('postgres')), location='does/not/exist'
        )
