from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


async def upgrade(db: BaseDBAsyncClient) -> str:
    """
    Scope indexed files to the project they came from.

    `file_path` is repository-relative, so it was globally unique across every
    project at once: indexing a second repository containing `src/main.py`
    replaced the first repository's row and cascaded away its symbols and
    embeddings.

    Rows that already exist have no project and none can be inferred — the
    column records where a repository sits on disk, and nothing in an existing
    row says where it came from. They are deleted rather than guessed at. That
    is safe because they are derived data: re-indexing rebuilds them exactly.
    Deleting cascades to symbols and embeddings, which are equally derived.

    Written to be re-runnable: every statement is guarded, so applying this
    twice neither fails nor duplicates.
    """
    return """
        DELETE FROM "files";

        ALTER TABLE "files"
            ADD COLUMN IF NOT EXISTS "project" VARCHAR(1024) NOT NULL DEFAULT '';
        COMMENT ON COLUMN "files"."project" IS
            'Absolute, link-resolved path of the repository this file was indexed from.';

        ALTER TABLE "files" ALTER COLUMN "project" DROP DEFAULT;

        ALTER TABLE "files" DROP CONSTRAINT IF EXISTS "files_file_path_key";
        DROP INDEX IF EXISTS "files_file_path_key";

        CREATE INDEX IF NOT EXISTS "idx_files_project" ON "files" ("project");
        CREATE UNIQUE INDEX IF NOT EXISTS "uid_files_project_file_path"
            ON "files" ("project", "file_path");
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    """
    Return to a globally unique path.

    The rows are dropped again on the way back: with two projects indexed, a
    shared relative path cannot satisfy a global uniqueness constraint, so
    there is no correct subset to keep.
    """
    return """
        DELETE FROM "files";

        DROP INDEX IF EXISTS "uid_files_project_file_path";
        DROP INDEX IF EXISTS "idx_files_project";

        ALTER TABLE "files" DROP COLUMN IF EXISTS "project";

        CREATE UNIQUE INDEX IF NOT EXISTS "files_file_path_key" ON "files" ("file_path");
    """
