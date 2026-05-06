import os
from typing import Any

DATABASE_URL: str = os.getenv('DATABASE_URL', 'postgres://loom:loom_password@localhost:5432/loom')

TORTOISE_ORM: dict[str, Any] = {
    'connections': {'default': DATABASE_URL},
    'apps': {
        'models': {
            'models': ['database.models', 'aerich.models'],
            'default_connection': 'default',
        }
    },
}
