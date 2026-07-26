# Loom Backend

Loom is a local agentic coder assistant. This is the backend component, built with FastAPI and TortoiseORM.

## Setup

### Prerequisites

- Python 3.14+
- PostgreSQL with `pgvector` extension

### Installation

1. Install the dependencies:

```bash
pip install .
```

### Database Migrations

This project uses `aerich` for database migrations.

1. Initialize migrations (already done in the repository):

```bash
aerich init -t database.tortoise_config.TORTOISE_ORM --location database/migrations
```

2. Apply migrations to the database:

```bash
aerich upgrade
```

3. To generate new migrations after model changes:

```bash
aerich migrate --name your_migration_name
```

## Maintenance

### Static Analysis

Run `mypy` to check for type errors:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
mypy . --ignore-missing-imports --explicit-package-bases
```

### Starting the Server

Run the FastAPI application with `uvicorn`:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Dependencies

- **FastAPI**: Web framework.
- **TortoiseORM**: Async ORM.
- **tortoise-embeddings**: `pgvector` support for TortoiseORM.
- **pydantic-ai**: Agentic AI framework.
- **Dishka**: Dependency injection.
- **GitPython**: Git operations.
- **Docker SDK**: Sandbox environment management.
