from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "embedding_models" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "model_name" VARCHAR(100) NOT NULL UNIQUE,
            "dimensions" INT NOT NULL,
            "chunk_lines" INT NOT NULL,
            "provider" VARCHAR(64) NOT NULL,
            "created_at" TIMESTAMPTZ NOT NULL
        );
        COMMENT ON TABLE "embedding_models" IS 'Registry of the embedding models used to index the repository.';
        CREATE TABLE IF NOT EXISTS "files" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "project" VARCHAR(1024) NOT NULL,
            "file_path" VARCHAR(1024) NOT NULL,
            "raw_content" TEXT NOT NULL,
            "hash" VARCHAR(64) NOT NULL,
            "last_indexed" TIMESTAMPTZ NOT NULL,
            CONSTRAINT "uid_files_project_d1fe26" UNIQUE ("project", "file_path")
        );
        CREATE INDEX IF NOT EXISTS "idx_files_project_78d309" ON "files" ("project");
        COMMENT ON TABLE "files" IS 'Stores file information from the repository.';
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS "code_embeddings" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "chunk_text" TEXT NOT NULL,
            "embedding" vector NOT NULL,
            "start_line" INT NOT NULL,
            "file_id" INT NOT NULL REFERENCES "files" ("id") ON DELETE CASCADE,
            "model_id" INT NOT NULL REFERENCES "embedding_models" ("id") ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS "idx_code_embedd_file_id_50aabe" ON "code_embeddings" ("file_id");
        CREATE INDEX IF NOT EXISTS "idx_code_embedd_model_i_0af20a" ON "code_embeddings" ("model_id");
        COMMENT ON TABLE "code_embeddings" IS 'Stores semantic embeddings for code chunks.';
        CREATE TABLE IF NOT EXISTS "symbols" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "symbol_name" VARCHAR(255) NOT NULL,
            "kind" VARCHAR(50) NOT NULL,
            "line_start" INT NOT NULL,
            "line_end" INT NOT NULL,
            "file_id" INT NOT NULL REFERENCES "files" ("id") ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS "idx_symbols_symbol__fba61f" ON "symbols" ("symbol_name");
        CREATE INDEX IF NOT EXISTS "idx_symbols_file_id_dc645c" ON "symbols" ("file_id");
        COMMENT ON TABLE "symbols" IS 'Stores AST symbols (functions, classes) found in files.';
        CREATE TABLE IF NOT EXISTS "tasks" (
            "id" UUID NOT NULL PRIMARY KEY,
            "prompt" TEXT NOT NULL,
            "repo_path" VARCHAR(1024) NOT NULL,
            "coder_model" VARCHAR(255) NOT NULL,
            "reviewer_model" VARCHAR(255) NOT NULL,
            "coder_context_window" INT NOT NULL,
            "planner_context_window" INT NOT NULL,
            "branch_name" VARCHAR(255) NOT NULL,
            "status" VARCHAR(32) NOT NULL,
            "summary" TEXT NOT NULL,
            "created_at" TIMESTAMPTZ NOT NULL,
            "updated_at" TIMESTAMPTZ NOT NULL
        );
        CREATE INDEX IF NOT EXISTS "idx_tasks_status_449504" ON "tasks" ("status");
        CREATE INDEX IF NOT EXISTS "idx_tasks_created_99edd2" ON "tasks" ("created_at");
        COMMENT ON COLUMN "tasks"."status" IS 'initialized: initialized\nrunning: running\ncancelled: cancelled\ncompleted: completed\nanswered: answered\nmerged: merged\nfailed: failed';
        COMMENT ON TABLE "tasks" IS 'A unit of work the agent was asked to carry out.';
        CREATE TABLE IF NOT EXISTS "task_memory" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "entry_type" VARCHAR(32) NOT NULL,
            "content" JSONB NOT NULL,
            "created_at" TIMESTAMPTZ NOT NULL,
            "task_id" UUID NOT NULL REFERENCES "tasks" ("id") ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS "idx_task_memory_task_id_3a4438" ON "task_memory" ("task_id");
        COMMENT ON COLUMN "task_memory"."entry_type" IS 'tool_call: tool_call\ncommit: commit\nreview: review\nplan: plan\nrejection: rejection\nstatus_change: status_change\nsummary: summary';
        COMMENT ON TABLE "task_memory" IS 'An append-only record of what happened during a task.';
        CREATE TABLE IF NOT EXISTS "aerich" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "version" VARCHAR(255) NOT NULL,
            "app" VARCHAR(100) NOT NULL,
            "content" JSONB NOT NULL
        );"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        """


MODELS_STATE = (
    "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"
    "jo1v8yu/8flbTr8w=="
)
