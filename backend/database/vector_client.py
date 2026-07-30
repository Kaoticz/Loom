import tortoise.backends.asyncpg
from tortoise_embeddings import VectorAsyncpgDBClient

# Registering the pgvector-aware client is what teaches asyncpg to encode and
# decode vector columns. Without it, inserting an embedding fails with
# 'expected str, got list'. It lives in its own module, separate from the
# connection settings, so that installing it does not force DATABASE_URL to be
# read before the caller has had a chance to set it.
tortoise.backends.asyncpg.client_class = VectorAsyncpgDBClient
