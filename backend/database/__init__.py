# Installs the pgvector-aware asyncpg client. Doing it here means any entry
# point that touches a model gets working vector support, rather than only those
# that happen to import the Tortoise configuration first.
from database import vector_client as vector_client
