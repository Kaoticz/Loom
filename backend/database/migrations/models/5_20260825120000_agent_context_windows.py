from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    """
    Record the context window budgeted for each agent on a task.

    Two columns rather than one: the coder and the planner are budgeted
    separately, so a deployment can hold the planner to a tighter window than
    the long tool-heavy coder loop.

    Existing rows take 0, meaning the user configured nothing and the default
    applied. The raw setting is stored rather than the resolved number so that
    changing the default later reads as "unset" on old tasks instead of
    silently claiming they used a value they never saw.

    Written to be re-runnable: every statement is guarded, so applying this
    twice neither fails nor duplicates.
    """
    return """
        ALTER TABLE "tasks"
            ADD COLUMN IF NOT EXISTS "coder_context_window" INT NOT NULL DEFAULT 0;
        COMMENT ON COLUMN "tasks"."coder_context_window" IS
            'Context window in tokens budgeted for the coder agent, or 0 for the default.';

        ALTER TABLE "tasks"
            ADD COLUMN IF NOT EXISTS "planner_context_window" INT NOT NULL DEFAULT 0;
        COMMENT ON COLUMN "tasks"."planner_context_window" IS
            'Context window in tokens budgeted for the planner agent, or 0 for the default.';
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    """
    Drop the recorded context windows.

    Nothing is lost that cannot be re-entered: the columns record a setting the
    user supplies with each task, not anything derived from the work itself.
    """
    return """
        ALTER TABLE "tasks" DROP COLUMN IF EXISTS "planner_context_window";
        ALTER TABLE "tasks" DROP COLUMN IF EXISTS "coder_context_window";
    """
