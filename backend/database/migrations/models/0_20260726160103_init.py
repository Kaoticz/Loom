from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "files" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "file_path" VARCHAR(1024) NOT NULL UNIQUE,
            "raw_content" TEXT NOT NULL,
            "hash" VARCHAR(64) NOT NULL,
            "last_indexed" TIMESTAMPTZ NOT NULL
        );
        COMMENT ON TABLE "files" IS 'Stores file information from the repository.';
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS "code_embeddings" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "chunk_text" TEXT NOT NULL,
            "embedding" vector NOT NULL,
            "start_line" INT NOT NULL,
            "file_id" INT NOT NULL REFERENCES "files" ("id") ON DELETE CASCADE
        );
        COMMENT ON TABLE "code_embeddings" IS 'Stores semantic embeddings for code chunks.';
        CREATE TABLE IF NOT EXISTS "symbols" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "symbol_name" VARCHAR(255) NOT NULL,
            "kind" VARCHAR(50) NOT NULL,
            "line_start" INT NOT NULL,
            "line_end" INT NOT NULL,
            "file_id" INT NOT NULL REFERENCES "files" ("id") ON DELETE CASCADE
        );
        COMMENT ON TABLE "symbols" IS 'Stores AST symbols (functions, classes) found in files.';
        CREATE TABLE IF NOT EXISTS "task_memory" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "task_id" VARCHAR(255) NOT NULL,
            "log_content" TEXT NOT NULL,
            "created_at" TIMESTAMPTZ NOT NULL
        );
        COMMENT ON TABLE "task_memory" IS 'Stores agent task logs and tool outputs.';
        CREATE TABLE IF NOT EXISTS "aerich" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "version" VARCHAR(255) NOT NULL,
            "app" VARCHAR(100) NOT NULL,
            "content" JSONB NOT NULL
        );
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        """


MODELS_STATE = (
    "eJztmV1v2zYUhv8KoasE6ILES9yid47jrFkbe4iFrmhRELR0bAuWSFekmhhd/vtISrK+VS"
    "lrMtvQVaLzwZAPj8hXJz8Mj9ng8pOh/DHyZmDbDl0Yb9EPgxIP5C/lAa+QQdbrxK0Mgsxc"
    "nWFJE4Y4VvvIjAufWEK658TlIE02cMt31sJhVCVNBfOBIw4eocKxUJKP5sxHakxkLQO64i"
    "dqQJtZcsRwrm1zA+p8CwALtgCxBF+O8OWrNDvUhgfg8eN6hecOuHYGhmOrAbQdi81a226o"
    "uNaBalozbDE38GgSvN6IJaPbaIcKZV0ABZ8IUMMLP1BAaOC6EcKYUTjTJCScYirHhjkJXI"
    "VVZYcTSGwGxuOJiacjE2OjgDzOSJGMTBajarvkVLle/UJN4bfe2fnr8ze/98/fyBA9za3l"
    "9WP4pxMwYaLGMzaNR+0ngoQRmnECVW8NFvAginBNaS2nm83KUZbTz1OOmdZhjg0J56RaXw"
    "B0DUVz9MlUI3ucf3OVYfxxcDd8N7g7uh18OtaeTeT5MBn/EYcz+dKFL+V4+GFyqTciAQ/p"
    "9znL/SNY8qUqJ59Jy4EfMwqHQ37NuFj4+sH4ronkCHJBfIFdJ1x0w3Mhm/Tz82EP+P2aIy"
    "LhOndcwK0O21RGRzQiqq6x+ar0zFW4inSv5T3qLOh72GjIN3JGhFplZRqpgutomD2D+xhX"
    "T2xNblSf3G/v/XRRybXLFYPQqx8OpsPB1cjQhGfEWt0T38YZ1MrDeixn2cYWXV7Py1sIJQ"
    "sNR61CzTlNvUSjxbtRLc3UgtoJMpWBHCp1lEeUC8195iGpm5AP8nh0ZNSmUpE1Tu4k2W5J"
    "Ml34ayKWRbbDJanQBZmkXyPI/l/IHnnALtCFXNFbdHbaO69hGssxFXack16Rrxc5s1edOn"
    "DkHxZAW8nfXFqnf5+gf5eEtyrwOP4gYGeru9+ktvvVld0v1LVLuMDx2gqQryQo4XhQDjqf"
    "mwNuR8kn8S/7iN8HYk+ou4kOtbrav7kdTc3B7V+ZF+BqYI6Up5cp/th61M9t1XYQ9PeN+Q"
    "6pR/R5Mh5pvKnPnG2c+dlQcyKBYJiye0zsNKPYHM++IDWrlVHJ9ycvFshllHv9/g5cLR6q"
    "VWihN7VntVAlR7Ofmhtvxtz/SGqqBzkkRM+psyNaJUo74VittVMb1lhtD6YmivLQ0Tyglv"
    "LzV8iSJyIHfozmLKC2lNRaWle3Qp8yTqfBd0uDh/uH9WMLkZJLO0Ct0ru4aCBWZFSlWtG+"
    "rFxZyaW1wRzHHyDfi9MGeC9OK+kqV04LOhSwbnu2OCeySV1Tr6RNqhFBWeHWU4XS2u2YGl"
    "3ruWs9Py/cPW89m4SvbsFj/sYokcUpb600FjIOe0lgU3ks50QFUtnIZQv5LDWsYMxFLBDr"
    "QFTr4UaJnQDeLQGsq6SMbLUqS6UcoDB7FuEr34en9J9zaQdB+6X7z5YPCg4mJeTrG6TZzK"
    "49uivt0ej4TXVHNbU27dHnvL0H4DvWsuzmjjy1tzZJYn52YVfvc3fJ7tYl+x18HrWMm16y"
    "qZSDOPZf4JJVL1ULwlH4AdI9O23SXJJRNf9DL7SXKuXLn9PJuOICrZQutmMJ9A9yHb6P3+"
    "41cBWMegmTVyu5u08NcFn2bf+Sl9njv8Pc+wc="
)
