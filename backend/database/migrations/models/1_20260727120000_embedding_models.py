from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "embedding_models" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "model_name" VARCHAR(100) NOT NULL UNIQUE,
            "dimensions" INT NOT NULL,
            "provider" VARCHAR(64) NOT NULL DEFAULT 'ollama',
            "created_at" TIMESTAMPTZ NOT NULL
        );
        COMMENT ON TABLE "embedding_models" IS 'Registry of the embedding models used to index the repository.';
        -- Existing embeddings predate the registry, so nothing records which model
        -- produced them or how wide their vectors are. They cannot be attributed
        -- retroactively and are discarded; the next index rebuilds them.
        DELETE FROM "code_embeddings";
        ALTER TABLE "code_embeddings" ADD COLUMN "model_id" INT NOT NULL
            REFERENCES "embedding_models" ("id") ON DELETE CASCADE;
        CREATE INDEX IF NOT EXISTS "idx_code_embeddings_model_id" ON "code_embeddings" ("model_id");
        CREATE INDEX IF NOT EXISTS "idx_code_embeddings_file_id" ON "code_embeddings" ("file_id");
        CREATE INDEX IF NOT EXISTS "idx_symbols_file_id" ON "symbols" ("file_id");
        CREATE INDEX IF NOT EXISTS "idx_symbols_symbol_name" ON "symbols" ("symbol_name");
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP INDEX IF EXISTS "idx_symbols_symbol_name";
        DROP INDEX IF EXISTS "idx_symbols_file_id";
        DROP INDEX IF EXISTS "idx_code_embeddings_file_id";
        DROP INDEX IF EXISTS "idx_code_embeddings_model_id";
        ALTER TABLE "code_embeddings" DROP COLUMN "model_id";
        DROP TABLE IF EXISTS "embedding_models";
    """
