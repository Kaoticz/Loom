import os

DATABASE_URL = os.getenv('DATABASE_URL', 'postgres://loom:loom_password@localhost:5432/loom')

TORTOISE_ORM = {
    'connections': {'default': DATABASE_URL},
    'apps': {
        'models': {
            'models': ['database.models', 'aerich.models'],
            'default_connection': 'default',
        }
    },
}
