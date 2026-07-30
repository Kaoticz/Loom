from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "tasks" (
            "id" UUID NOT NULL PRIMARY KEY,
            "prompt" TEXT NOT NULL,
            "repo_path" VARCHAR(1024) NOT NULL DEFAULT '',
            "coder_model" VARCHAR(255) NOT NULL DEFAULT '',
            "reviewer_model" VARCHAR(255) NOT NULL DEFAULT '',
            "branch_name" VARCHAR(255) NOT NULL DEFAULT '',
            "status" VARCHAR(32) NOT NULL DEFAULT 'initialized',
            "summary" TEXT NOT NULL DEFAULT '',
            "created_at" TIMESTAMPTZ NOT NULL,
            "updated_at" TIMESTAMPTZ NOT NULL
        );
        COMMENT ON TABLE "tasks" IS 'A unit of work the agent was asked to carry out.';
        CREATE INDEX IF NOT EXISTS "idx_tasks_status" ON "tasks" ("status");
        CREATE INDEX IF NOT EXISTS "idx_tasks_created_at" ON "tasks" ("created_at");

        -- The previous task_memory held a free-text log keyed by an opaque string
        -- with no task record behind it. Nothing referenced those rows, and they
        -- cannot be attached to a task retroactively, so the table is rebuilt.
        DROP TABLE IF EXISTS "task_memory";
        CREATE TABLE IF NOT EXISTS "task_memory" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "task_id" UUID NOT NULL REFERENCES "tasks" ("id") ON DELETE CASCADE,
            "entry_type" VARCHAR(32) NOT NULL,
            "content" JSONB NOT NULL,
            "created_at" TIMESTAMPTZ NOT NULL
        );
        COMMENT ON TABLE "task_memory" IS 'An append-only record of what happened during a task.';
        CREATE INDEX IF NOT EXISTS "idx_task_memory_task_id" ON "task_memory" ("task_id");
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "task_memory";
        DROP TABLE IF EXISTS "tasks";
        CREATE TABLE IF NOT EXISTS "task_memory" (
            "id" SERIAL NOT NULL PRIMARY KEY,
            "task_id" VARCHAR(255) NOT NULL,
            "log_content" TEXT NOT NULL,
            "created_at" TIMESTAMPTZ NOT NULL
        );
    """
