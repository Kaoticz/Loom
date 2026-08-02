from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    """
    Record the chunk size each model's embeddings were built with.

    Existing rows take 0, meaning automatic, which is what every index built
    before this column existed used.
    """
    return """
        ALTER TABLE "embedding_models"
            ADD COLUMN IF NOT EXISTS "chunk_lines" INT NOT NULL DEFAULT 0;
        COMMENT ON COLUMN "embedding_models"."chunk_lines" IS
            'Lines per chunk used for these embeddings, or 0 for automatic sizing.';
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    """
    Drop the recorded chunk size.
    """
    return """
        ALTER TABLE "embedding_models" DROP COLUMN IF EXISTS "chunk_lines";
    """
