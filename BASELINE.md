# Phase 0 Baseline (recorded 2026-08-30)

Environment and test baselines for PLAN.md ("Incremental re-index on file writes plus a delete_file tool"). Later phases compare their results against these numbers.

## Environment

- `backend/.venv/bin/python --version` → **Python 3.14.7**
- Editable install: `loom-backend 0.0.1` (`pip install -e ".[dev]"` re-ran cleanly)
- `pip check` → no broken requirements
- Dev tools: pytest 9.1.1, pytest-asyncio, mypy, basedpyright all present
- External services (loaded from `Loom/.env`): `TEST_DATABASE_URL=postgres://loom:loom_password@localhost:5432/loom_db`, ast-grep 0.44.1 on PATH

## Full suite

- Command: `cd backend && .venv/bin/pytest`
- Result: **960 passed, 4 skipped**
- The 4 skips are in `tests/test_environment.py` (environment-conditional tests).
- Note: the first full run after reinstalling the package reported `866 passed, 4 skipped, 94 errors` (OSErrors in the Postgres suites: `test_db_isolation`, `test_index_deletion`, `test_indexing`, `test_pgvector`). Two immediately subsequent full runs passed clean with identical counts, and those DB suites pass in isolation and grouped. Recorded as a transient environment hiccup around the editable reinstall, not a code failure; flagged for tracking.

## Focused suites (this work's touchpoints)

- Command: `.venv/bin/pytest tests/test_agent_tools.py tests/test_file_editor.py tests/test_execution_loop.py tests/test_indexing.py tests/test_pgvector.py`
- Result: **244 passed**

## Static analysis

- `.venv/bin/mypy .` → Success: no issues found in 119 source files
- `.venv/bin/basedpyright .` → 0 errors, 0 warnings, 0 notes

## Post-plan verification

- Focused command: `.venv/bin/pytest tests/test_agent_tools.py tests/test_file_editor.py tests/test_execution_loop.py tests/test_indexing.py tests/test_pgvector.py` → **254 passed**
- Full command: `.venv/bin/pytest` → **972 passed, 4 skipped**
- Static analysis remains clean: mypy reports no issues in 119 source files; basedpyright reports 0 errors, 0 warnings, and 0 notes.
- The full suite gained 12 passing regression tests compared with the Phase 0 baseline. No baseline failures regressed.
