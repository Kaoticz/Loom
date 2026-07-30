from __future__ import annotations

from logging import Logger, getLogger
from pathlib import Path
from typing import Any

import asyncpg
from aerich import Command

logger: Logger = getLogger(__name__)

DEFAULT_MIGRATIONS_APP: str = 'models'
DEFAULT_MIGRATIONS_LOCATION: str = 'database/migrations'


class SchemaSetupError(RuntimeError):
    """
    Raised when the database schema could not be brought up to date.
    """


async def ensure_vector_extension(dsn: str) -> None:
    """
    Make sure the pgvector extension exists before any migration needs it.

    The first migration creates a `vector` column, so the extension has to be in
    place before it runs. `tortoise-embeddings` happens to inject the same
    statement through its DDL patches, but relying on that leaves a hard schema
    dependency invisible in this project's own code.

    :param dsn: The database connection string.
    :raises SchemaSetupError: If the extension cannot be created.
    """
    connection: Any = None
    try:
        connection = await asyncpg.connect(dsn)
        await connection.execute('CREATE EXTENSION IF NOT EXISTS vector;')
    except Exception as error:
        raise SchemaSetupError(
            f'Could not enable the pgvector extension: {error}. The database user may '
            'lack permission to create extensions.'
        ) from error
    finally:
        if connection is not None:
            await connection.close()


async def apply_migrations(
    tortoise_config: dict[str, Any],
    app_name: str = DEFAULT_MIGRATIONS_APP,
    location: str = DEFAULT_MIGRATIONS_LOCATION,
) -> list[str]:
    """
    Bring the database schema up to date by applying pending migrations.

    Migrations are the single source of truth for the schema. Generating it from
    the models instead would create missing tables but never alter existing
    ones, so an upgraded installation would silently run against an out-of-date
    schema, and the recorded migration history would never match reality.

    Applying an already-current database is a no-op.

    :param tortoise_config: The Tortoise settings, including the connection.
    :param app_name: The Tortoise app the migrations belong to.
    :param location: The directory holding the migration files.
    :returns: The names of the migrations that were applied, oldest first.
    :raises SchemaSetupError: If the migrations are missing or a migration fails.
    """
    directory: Path = Path(location) / app_name
    if not directory.is_dir():
        raise SchemaSetupError(
            f'No migration directory at {directory}. The schema is defined entirely by '
            'migrations, so starting without them would leave the database empty. Check that '
            'the migrations were included in the deployment.'
        )

    available: list[Path] = sorted(
        path for path in directory.glob('*.py') if not path.name.startswith('__')
    )
    if not available:
        raise SchemaSetupError(
            f'The migration directory {directory} contains no migrations, so the schema '
            'cannot be created.'
        )

    # A Command closes Tortoise down when it exits, so state left by an earlier
    # call has to be cleared and re-established before another one can start.
    await _reset_tortoise_state()
    await _prime_tortoise(tortoise_config)

    try:
        async with Command(tortoise_config, app_name, location) as command:
            applied: list[str] = await command.upgrade(run_in_transaction=True)
    except Exception as error:
        raise SchemaSetupError(
            f'Applying database migrations failed: {error}. The application will not start '
            'against a schema it cannot verify.'
        ) from error

    if applied:
        logger.info('Applied %d database migration(s): %s', len(applied), ', '.join(applied))
    else:
        logger.info('The database schema is already up to date.')

    return applied


async def _reset_tortoise_state() -> None:
    """
    Return Tortoise to an uninitialised state.

    Aerich's `Command` initialises and then tears down Tortoise itself, which
    leaves configuration behind that prevents a second initialisation in the
    same process.
    """
    from tortoise import Tortoise
    from tortoise.backends.base.executor import EXECUTOR_CACHE

    await Tortoise.close_connections()
    await Tortoise._reset_apps()
    EXECUTOR_CACHE.clear()


async def _prime_tortoise(tortoise_config: dict[str, Any]) -> None:
    """
    Establish Tortoise configuration so a migration command can initialise.

    Aerich's `Command` initialises Tortoise itself, but a command that ran
    earlier in the process leaves state behind that stops the next one from
    doing so. Initialising explicitly first restores it.

    :param tortoise_config: The Tortoise settings.
    """
    from tortoise import Tortoise

    await Tortoise.init(config=tortoise_config)


async def prepare_database(
    dsn: str,
    tortoise_config: dict[str, Any],
    app_name: str = DEFAULT_MIGRATIONS_APP,
    location: str = DEFAULT_MIGRATIONS_LOCATION,
) -> list[str]:
    """
    Enable the required extensions and migrate the schema, in that order.

    :param dsn: The database connection string.
    :param tortoise_config: The Tortoise settings.
    :param app_name: The Tortoise app the migrations belong to.
    :param location: The directory holding the migration files.
    :returns: The names of the migrations that were applied.
    :raises SchemaSetupError: If the extension or a migration fails.
    """
    await ensure_vector_extension(dsn)
    return await apply_migrations(tortoise_config, app_name, location)
