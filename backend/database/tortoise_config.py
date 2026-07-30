import os
from typing import Any

from environment import load_environment

# aerich resolves this module by dotted path and never runs main, so the file
# has to be loaded here too or migrations would ignore it.
load_environment()
from database import vector_client as vector_client  # noqa: F401  installs the pgvector client

DEFAULT_DATABASE_URL: str = 'postgres://loom:loom_password@localhost:5432/loom'


def database_url() -> str:
    """
    Read the configured database connection string.

    :returns: The connection string, or the local development default.
    """
    return os.getenv('DATABASE_URL', DEFAULT_DATABASE_URL)


def build_config() -> dict[str, Any]:
    """
    Build the Tortoise configuration from the current environment.

    Read at call time rather than at import. A module level constant freezes
    whatever DATABASE_URL happened to be set when the module was first imported,
    which makes the connection depend on import order rather than configuration.

    :returns: The Tortoise settings.
    """
    return {
        'connections': {'default': database_url()},
        'apps': {
            'models': {
                'models': ['database.models', 'aerich.models'],
                'default_connection': 'default',
            }
        },
    }


# aerich resolves its configuration through a dotted path to a dict, so this has
# to stay a module attribute. Application code should call build_config().
DATABASE_URL: str = database_url()
TORTOISE_ORM: dict[str, Any] = build_config()
