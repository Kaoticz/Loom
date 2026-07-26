import os
from typing import Any
import tortoise.backends.asyncpg
from tortoise_embeddings import VectorAsyncpgDBClient

# Override the client class for the asyncpg backend to support pgvector
tortoise.backends.asyncpg.client_class = VectorAsyncpgDBClient

DATABASE_URL: str = os.getenv('DATABASE_URL', 'postgres://loom:loom_password@localhost:5432/loom_db')

TORTOISE_ORM: dict[str, Any] = {
    'connections': {'default': DATABASE_URL},
    'apps': {
        'models': {
            'models': ['database.models', 'aerich.models'],
            'default_connection': 'default',
        }
    },
}
