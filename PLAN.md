# PLAN: Loom Maintainability Refactor

## Summary
A behavior-preserving refactor of the Loom codebase driven by a strict (`nuke`-style) maintainability audit — decomposing god objects, deduplicating triplicated code, and fixing a fragile string-matching boundary — followed by a project-wide "one class per file" convention applied to every backend Python file that currently groups multiple classes together.

## Type
Project (multi-phase)

## Global Rules
These apply across the entire plan and to every phase.

- **Execution mode:** Iterative. After completing each phase, stop, summarize what changed, and wait for explicit user authorization before starting the next phase.
- **Commit cadence:** After every phase. Per "If iterative" below, the commit for a phase happens only after the user has explicitly authorized moving on from it — never immediately upon finishing the phase's work, and this applies to the final phase too. Phase 0 is never committed on its own; its setup work is folded into the commit that follows Phase 1.
- **Branch strategy:** Create a new local branch for this work before Phase 1 begins: `refactor/maintainability-audit`. Do not commit any of this work to `main`.
- **PLAN.md handling:** Committed, kept. `PLAN.md` is included in the project's commits and remains in the repository once the plan is complete.
- **Baseline test run:** Before starting any implementation work (during Phase 0), run the full test suite for both the backend and the frontend and record the results (pass/fail/skip counts). This is the baseline every later phase's test run is compared against. Note: a prior `BASELINE.md` for a since-completed, unrelated feature existed in this repo and was deleted before this plan was written — it is not a baseline for this work.
- **Self-verification:** Both backend and frontend have real test suites — use them.
  - Backend: `cd backend && .venv/bin/pytest` (see Phase 0 for the exact venv/env details).
  - Frontend: `cd frontend && npm run test` (Vitest).
  - Treat any new failures beyond the Phase 0 baseline as a blocking problem (see "Failed verification blocks completion").
  - Writing new tests is a recommendation, not an obligation, per the standard rule — this plan is a refactor, not a coverage initiative. If a phase makes something meaningfully easier to test than before (that's the point of several phases here), note that as an opportunity in the phase summary rather than silently expanding scope to add new tests.
- **Record how each Acceptance Criterion was verified.** Append a short "— verified by: ..." note when checking off any Acceptance Criteria item.
- **Consult, don't assume:** Ask the user directly whenever there's a genuine choice to make, rather than silently picking one — this applies throughout execution, not just during planning.
- **Problems get recommendations:** Any bug, risk, or unclear scope noticed while working should come with a concrete recommendation, not just a flag.
- **No behavior changes, anywhere in this plan.** Every phase is a pure refactor: identical external behavior, identical API responses, identical event/message payloads, identical CLI/UI behavior. If executing a phase surfaces what looks like a genuine pre-existing bug, do not fix it as part of this plan — flag it via "Problems get recommendations" and leave it for a separate, explicit decision.
- **Failed verification blocks completion:** If a phase's Acceptance Criteria aren't met, or its changes introduce new test failures beyond the Phase 0 baseline, do not mark it Done. Attempt a fix within the same phase first; if the fix isn't obvious or needs a design decision, stop and ask rather than proceeding.
- **Keep the plan current:** `PLAN.md` is a living document. Fold any user-directed change — new requirement, scope change, unplanned fix — into the relevant phase (or a new one) as it happens, rather than tracking it separately.
- **If iterative:** After each phase, stop, write a summary of what changed, list any recommendations for problems noticed but out of that phase's scope, and wait for explicit authorization before continuing. Make any requested changes first, folding them into the plan per "Keep the plan current." This applies to the final phase too.
- **Backend code standards (from `backend/AGENTS.md`), apply to every file touched or created in Phases 1, 2, 4, 5, 7-11:**
  - Type-annotate everything (signatures, returns, locals).
  - Every function/class gets a reST-style docstring (`:param:`, `:returns:`, `:raises:`); endpoint functions get only a summary line.
  - Single quotes for strings.
  - Every endpoint keeps its own dedicated Pydantic request/response models — never reuse ORM models or dicts as contracts.
  - New/edited Pydantic models use `Annotated[T, Field()]` for every field.
  - A field/variable with 3+ named literal values gets an `Enum`, not inline string literals.
  - Guard clauses over nested conditionals.
  - No side-effect-only functions that mutate and return `None`; prefer returning new values.
  - Circular dependencies (a realistic risk when splitting one file with cross-referencing classes into many, e.g. Phases 7, 8, 10, 11): resolve using `from __future__ import annotations` + `TYPE_CHECKING` + deferred import + `model_rebuild()`, exactly as documented in `backend/AGENTS.md`.
  - PEP 585-style annotations (`list[str]`, not `List[str]`); `collections.abc` only when no builtin generic exists.
  - Direct imports for type annotations and instantiation (`from asyncio import Lock`), dotted form only for module-level function calls.
  - `is`/`is not` for null checks, not truthiness.
  - **Post-task:** after every phase that touches backend code, run `cd backend && .venv/bin/mypy . && .venv/bin/basedpyright .` and fix every error, even ones unrelated to that phase's change.
  - Do not introduce new dependencies or change the database schema without asking first.
- **Frontend code standards (from `frontend/AGENTS.md`), apply to Phases 3 and 6:**
  - `strict` TypeScript; treat type errors as build failures.
  - Every exported function/store/class gets a JSDoc comment (`@param`, `@returns`, `@throws`).
  - Single quotes outside template literals/JSX attributes.
  - Every API call has dedicated request/response types; no raw `any` or untyped `fetch`.
  - Zod validation at every trust boundary (API responses, WebSocket messages).
  - 3+ named literal values → a string-union/`const` object, not scattered literals.
  - Guard clauses over nested conditionals.
  - WebSocket connections stay encapsulated in a store factory — components never instantiate `WebSocket` directly.
  - A component file over ~200 lines gets split (this is the same standard driving Phase 6).
  - **Post-task:** after every phase that touches frontend code, run `cd frontend && npm run check && npm run lint` and fix every error/warning, even ones unrelated to that phase's change.
  - Do not introduce new dependencies or modify the SvelteKit routing structure without asking first.
- **One-class-per-file convention (project-wide, Phases 2, 7-11):** Every class defined anywhere in `backend/` (excluding files under `backend/tests/`) must live in its own file, with no exception for `Enum` or Pydantic model classes — this applies uniformly, including to the large event/message taxonomies. When a source file's classes number three or more and are clearly one cohesive family (an event taxonomy, a message taxonomy, a service plus its small support types), group the resulting files into a new same-named package folder rather than scattering them as siblings; for two resulting files, siblings in the existing directory are sufficient — no folder needed. Every such package's `__init__.py` must re-export its classes; this is the package's genuine new public surface, not a backwards-compatibility shim, since the old single-file module no longer exists. Do not leave the old flat module in place re-exporting from the new location "for compatibility" — the old module is deleted, and every import site across the codebase must be updated to import from the new package. This convention was checked against the frontend/TypeScript codebase too; no non-test `.ts`/`.svelte` file currently defines more than one class, so no frontend phase is needed for it.
- **Import style for split-out classes (Phases 2, 7-11), per `backend/AGENTS.md`'s "Preferred Type References" standard:** that standard requires direct imports for type annotations and instantiation (`from asyncio import Lock`, not `import asyncio` + `asyncio.Lock`) rather than dotted/module-qualified access. Every package created by this plan (`backend/services/events/`, `backend/schemas/agent_messages/`, `backend/api/v1/agent_session/`, and any other package resulting from Phases 9-11) must therefore re-export its classes from `__init__.py`, and every call site elsewhere in the codebase must import each class directly from the package root (e.g. `from services.events import ThoughtEvent`), never from the specific submodule the class happens to live in (e.g. not `from services.events.thought_event import ThoughtEvent`) and never via a dotted/qualified reference to the submodule. The per-class file layout inside a package is an internal organizational detail; the package root is the only import surface the rest of the codebase should ever reference.

## Background & Context

Loom is a local-first AI coding agent: a FastAPI + Dishka + TortoiseORM/PostgreSQL backend (`backend/`, Python 3.14, Pydantic AI) and a SvelteKit + TypeScript + TailwindCSS frontend (`frontend/`). It indexes a repo, plans a change, executes it via an agent with tools, runs it in a sandboxed container, and has a second model review the diff.

This plan originates from a whole-codebase maintainability audit performed using the `nuke` skill's strict standards (file-size limits, spaghetti-branch detection, thin-abstraction and duplication checks, "code judo" simplification), adapted from its usual diff-review framing to judge the codebase as it stands (there was no active feature branch to diff against — `main` was clean and in sync with `origin/main` at audit time). The audit surfaced 10 findings; this plan implements all of them (Phases 1-6), plus opportunistic fixes (Phase 4), plus a follow-on project-wide convention the user requested during planning: every class in the backend gets its own file (Phases 2, 7-11).

Both `backend/AGENTS.md` and `frontend/AGENTS.md` exist and are authoritative on style — see Global Rules above for the parts that bear directly on this work. Full audit findings (file/line references) are reproduced per-phase below rather than restated here.

Existing conventions worth calling out because Phases 7-11 interact with them directly:
- The backend already uses the `Enum`-for-literals and `Annotated[T, Field()]` Pydantic patterns extensively, especially in `backend/services/event_bus.py` and `backend/schemas/agent_messages.py` — the two files this plan's biggest phases (7, 8) split apart.
- `backend/services/compaction.py` and `backend/services/model_resolution.py` currently each define a byte-identical private function `_window_for_consumer` (finding #6) — Phase 4 dedupes this before Phase 11 reorganizes `compaction.py` into multiple files, so the fix lands once, in the simpler pre-split shape of the file.

## Goal / Desired Behavior

At the end of this plan:
- `backend/api/v1/agent_session.py`'s `AgentSession` class is decomposed into itself plus focused collaborators (context-overflow recovery, commit sweeping, a shared agent-run helper, and conditionally a planning coordinator), each in its own file, with no change in `AgentSession`'s external behavior or the API's observable responses.
- `backend/agents/coder.py`'s three file-mutation tools (`create_file`, `edit_file`, `delete_file`) share one internal helper instead of duplicating ~130 lines of control flow, and the module's ~30 tool functions are grouped by category into a `coder_tools/` package.
- `frontend/src/lib/stores/agent.ts`'s `createAgentStore()` no longer bundles cross-tab presence, WebSocket lifecycle, and the event reducer in one 700-line closure — each is its own module, composed together, with identical runtime behavior.
- The commit-failure resolution flow in `agent_session.py` no longer classifies a user's free-typed answer with a substring heuristic; it uses a typed/explicit boundary instead, with identical behavior for every currently-supported answer.
- `_window_for_consumer` exists in exactly one place (`backend/services/model_resolution.py`), imported by `compaction.py`.
- `frontend/src/routes/settings/+page.svelte` is decomposed into a layout root plus `IndexStatusPanel.svelte`, `SecretsEditor.svelte`, and `ModelSettingsPanel.svelte`, with identical rendered behavior.
- Every backend Python class outside `backend/tests/` lives in its own file. The two large taxonomies (`event_bus.py`, `agent_messages.py`) become packages (`backend/services/events/`, `backend/schemas/agent_messages/`) with one file per class. The remaining 19 currently-multi-class files are split per Phases 9-11.
- Throughout, both test suites pass with the same pass/fail/skip counts as the Phase 0 baseline, and both linters/type-checkers (`mypy`/`basedpyright`, `svelte-check`/`eslint`/`prettier`) are clean.

## Scope

**In scope:**
1. `backend/api/v1/agent_session.py` decomposition: `ContextOverflowRecovery`, `CommitSweeper`, a shared `_run_agent`-style helper, conditional `PlanningCoordinator`, plus moving `TerminalToolCompleted` and `SessionServices` to their own files (Phase 2).
2. `backend/agents/coder.py` file-mutation triplication fix (Phase 1) and later category split into `coder_tools/` (Phase 5).
3. `frontend/src/lib/stores/agent.ts` decomposition into presence/socket/reducer modules (Phase 3).
4. `_window_for_consumer` dedup (Phase 4).
5. Commit-failure answer classification fix (Phase 4).
6. `frontend/src/routes/settings/+page.svelte` decomposition (Phase 6).
7. Project-wide one-class-per-file split of all 21 non-test backend files that currently define more than one class (Phases 2, 7, 8, 9, 10, 11 — Phase 2 covers `agent_session.py`'s share of this).

**Out of scope:**
- Any functional/behavioral change — this plan is refactor-only (see Global Rules).
- Fixing bugs incidentally discovered while refactoring — flag them per "Problems get recommendations" instead.
- Adding new automated tests (recommend, don't add, unless a phase's own Acceptance Criteria requires touching an existing test file to keep it passing).
- Applying the one-class-per-file convention to test files (explicitly excluded by the user) or to the frontend (checked; not applicable — no non-test `.ts`/`.svelte` file has more than one class).
- New dependencies, database schema changes, or SvelteKit routing changes.
- Re-measuring or acting on nuke findings beyond the 10 already identified — this plan doesn't re-run the audit after the fact.

## Phases

### Phase 0: Environment Setup — Status: Done
- Backend: Python 3.14 venv already present at `backend/.venv`. No fresh install should be needed; confirm with `cd backend && .venv/bin/pip check`. If dependencies are missing or stale, run `cd backend && .venv/bin/pip install -e ".[dev]"`.
- Backend env: `backend/.env` and root `.env` already exist with `TEST_DATABASE_URL` set for the Postgres-backed test suites. Confirm Postgres is reachable (`docker compose` / `podman-compose` services, or however this developer normally runs them) before running the full suite.
- Backend test command: `cd backend && .venv/bin/pytest`.
- Backend lint/type-check command: `cd backend && .venv/bin/mypy . && .venv/bin/basedpyright .`.
- Frontend: Node 25.x, npm. `frontend/node_modules` already present; if `npm run test`/`npm run check` fail for missing-module reasons, run `cd frontend && npm install` first.
- Frontend test command: `cd frontend && npm run test`.
- Frontend check/lint commands: `cd frontend && npm run check` and `cd frontend && npm run lint`.
- Run all four commands above (backend test, backend lint, frontend test, frontend check+lint) now and record the exact results (pass/fail/skip counts, and any pre-existing lint/type errors) as the Phase 0 baseline in the Phase 0 completion summary. This baseline is what every later phase's verification is compared against.
- Create the `refactor/maintainability-audit` branch (see Global Rules — Branch strategy) before Phase 1 begins. Do not commit Phase 0 itself.
- **Completion summary (2026-08-30):** `backend/.venv` passed `pip check`; root `.env` supplied `TEST_DATABASE_URL`; Postgres was healthy and reachable on port 5432; and `refactor/maintainability-audit` was created. Node 26.8.1/npm 12.0.2 and existing `frontend/node_modules` were usable (the plan's Node 25.x expectation was exceeded). Baseline: backend `pytest` = **972 passed, 4 skipped**; `mypy .` = no issues in 119 source files; `basedpyright .` = 0 errors, 0 warnings, 0 notes; frontend `npm run test` = **536 passed, 6 skipped** across 33 files; `npm run check` = 0 errors, 0 warnings; `npm run lint` = prettier and eslint clean.

### Phase 1: Deduplicate `coder.py` file-mutation tools — Status: Done
- **Goal:** Remove the ~130-line triplication across `create_file`, `edit_file`, and `delete_file` in `backend/agents/coder.py` (lines 497-535, 538-585, 588-627) without changing any tool's observable behavior, error messages, or event/diff reporting.
- **Requirements:**
  - [x] Introduce one shared internal helper that performs the common sequence: `report_tool` context entry, path resolution/validation, reading the base version, invoking the mutate operation (catching `ValueError`/`OSError` exactly as today), `record_modification`, `_report_diff`, `note_pending_change`, best-effort reindex (catching `Exception` exactly as today), `report.succeed(...)`, and `_note_ignored(extra)`.
  - [x] `create_file`, `edit_file`, and `delete_file` each become a thin call into that helper, supplying only what differs: the mutate operation itself, the verb/summary text, and any per-tool messages.
  - [x] Preserve every existing error message, exception type, and success-report shape exactly.
- **Edge Cases & Error Handling:**
  - `ValueError`/`OSError` from the mutate operation must still surface with the same message and reporting path as today for all three tools.
  - The best-effort reindex failure path (caught `Exception`) must remain non-fatal to the tool call, exactly as today.
- **Affected Areas:** `backend/agents/coder.py` only.
- **Acceptance Criteria:**
  - [x] `create_file`, `edit_file`, `delete_file` no longer duplicate the shared control-flow steps listed above. — verified by: review of `_mutate_file` and its three thin tool callers in `backend/agents/coder.py`.
  - [x] Backend test suite passes with the same results as the Phase 0 baseline. — verified by: `cd backend && .venv/bin/pytest` → **972 passed, 4 skipped**; targeted `tests/test_agent_tools.py` → **97 passed**.
  - [x] `mypy`/`basedpyright` clean. — verified by: `cd backend && .venv/bin/mypy .` → no issues in 119 source files; `cd backend && .venv/bin/basedpyright .` → 0 errors, 0 warnings, 0 notes.
- **Testing Notes:** Existing tests covering `create_file`/`edit_file`/`delete_file` (in `backend/tests/test_agent_tools.py` or similar) should be run and should require no behavioral changes — if any assertion needs adjusting, that's a signal the refactor changed behavior and must be corrected, not the test.

### Phase 2: Decompose `AgentSession` — Status: Done
- **Goal:** Break up the 2408-line, 65+-method `AgentSession` god object (`backend/api/v1/agent_session.py`) into `AgentSession` plus focused collaborators, apply the one-class-per-file convention to this file's other two classes, and deduplicate the triplicated agent-run construction — all without changing `AgentSession`'s external behavior.
- **Requirements:**
  - [x] Create a package `backend/api/v1/agent_session/` (folder, since this now groups several related-but-distinct classes). Decide file names based on the classes below. `backend/api/v1/agent_session/__init__.py` must re-export `AgentSession`, `TerminalToolCompleted`, `SessionServices`, and every extracted collaborator (`ContextOverflowRecovery`, `CommitSweeper`, the shared run-helper, and `PlanningCoordinator` if extracted) — required, not optional, per Global Rules — Import style for split-out classes. Every call site elsewhere in the codebase imports directly from `backend.api.v1.agent_session` (the package root), never from an individual submodule. — verified by: package-root exports and import-site search; tests import collaborators directly only to monkeypatch the owning implementation module.
  - [x] Extract `ContextOverflowRecovery` from `_narrow_context_window`, `_narrow_after_overflow`, `_narrow_after_input_truncation`, `_summarize_after_overflow`, `_run_with_overflow_recovery` (current lines ~947-1132) into its own file. `AgentSession` delegates to it as a collaborator (constructor-injected or lazily constructed — whichever avoids implicit shared mutable state, per the "No Side Effects" standard). — verified by: `ContextOverflowRecovery` owns recovery logic; `AgentSession._run_with_overflow_recovery` is a thin legacy delegate, and targeted overflow-retry tests pass.
  - [x] Extract `CommitSweeper` from `_sweep_uncommitted`, `_resolve_commit_failures` (current lines ~1831-1969) into its own file, same delegation approach. (`_resolve_commit_failures`'s answer-classification logic is also touched by Phase 4 — see that phase's note on sequencing.) — verified by: `CommitSweeper` owns sweep/resolution logic; the `AgentSession` private entry points are thin delegates, and commit-resolution tests pass. The existing substring classification is intentionally preserved for Phase 4.
  - [x] Extract one shared agent-run helper (e.g. `_run_agent`) that assembles the capabilities list (compaction capabilities, history, usage monitor, provider usage monitor) once and wraps `_run_with_overflow_recovery`, replacing the near-identical logic currently duplicated across `_plan_structured` (1463-1529), `_plan_as_text` (1531-1580), and `_run_coder_streaming` (1133-1244). Parameterize by `output_type`, which agent to run, `context_window`, `run_path` (`CompactionRunPath`), and `event_stream_handler`/`message_history` as needed. — verified by: `AgentRunHelper.run_agent` is the sole production capability-list construction path for streamed coder and both planning paths; its live window reader is evaluated for every retry attempt.
  - [x] Move `TerminalToolCompleted` (currently line ~124) into its own file. — verified by: `terminal_tool_completed.py` and package-root export.
  - [x] Move `SessionServices` (the `Protocol`, currently line ~156) into its own file. — verified by: `session_services.py` and package-root export.
  - [x] `AgentSession` itself keeps its own file within the package, containing only the `AgentSession` class. — verified by: `agent_session.py` class scan.
  - [x] After the above extractions, re-measure `agent_session.py`'s (the `AgentSession`-class file's) line count. If it is still over 1000 lines, also extract a `PlanningCoordinator` from `_handle_planning`, `_plan_structured`, `_plan_as_text`, `_build_planning_prompt` (and any other planning-only methods) into its own file, with `AgentSession` delegating to it. If it is already under 1000 lines without this step, skip it and note that in the phase summary. — verified by: resulting `agent_session.py` is 1,777 lines, so `PlanningCoordinator` was extracted; the planning state handler and retained private entry points are thin delegates.
  - [x] Update every import site across the codebase (backend and any tests) that references `agent_session.AgentSession`, `agent_session.TerminalToolCompleted`, or `agent_session.SessionServices` to the new locations. — verified by: repository import search and targeted suite.
- **Edge Cases & Error Handling:**
  - Watch for circular imports between the extracted collaborators and `AgentSession` (e.g. a collaborator needing a type from `AgentSession` only for annotations) — resolve with the `TYPE_CHECKING` + deferred-import + `model_rebuild()` pattern from `backend/AGENTS.md` if it arises.
  - Preserve exact exception types and messages from the overflow-recovery and commit-sweeping paths.
- **Affected Areas:** `backend/api/v1/agent_session.py` (becomes a package), every file importing from it (search the codebase for `agent_session` imports before starting).
- **Acceptance Criteria:**
  - [x] `ContextOverflowRecovery`, `CommitSweeper`, the shared run-helper, `TerminalToolCompleted`, and `SessionServices` each live in their own file; `AgentSession`'s file contains only `AgentSession`. — verified by: package layout inspection.
  - [x] `_plan_structured`, `_plan_as_text`, and `_run_coder_streaming` no longer duplicate capability-list construction. — verified by: planning entry points are thin `PlanningCoordinator` delegates, streamed coder execution uses `AgentRunHelper`, and `AgentRunHelper.run_agent` is the only capability-list construction site.
  - [x] Backend test suite passes with the same results as the Phase 0 baseline. — verified by: `cd backend && .venv/bin/pytest` → **972 passed, 4 skipped**.
  - [x] `mypy`/`basedpyright` clean. — verified by: `cd backend && .venv/bin/mypy .` → no issues in 126 source files; `cd backend && .venv/bin/basedpyright .` → 0 errors, 0 warnings, 0 notes.
- **Testing Notes:** This is the highest-risk phase in the plan — run the full backend suite, not just session-related tests, since `AgentSession` is central to the whole request lifecycle.
- **Completion verification (2026-08-30):** Targeted `tests/test_agent_session.py tests/test_execution_loop.py` → **160 passed**; full backend `pytest` → **972 passed, 4 skipped**; `mypy .` → no issues in 126 source files; `basedpyright .` → 0 errors, 0 warnings, 0 notes. `PlanningCoordinator` was required and extracted; `backend/api/v1/agent_session/agent_session.py` is **1,777 lines** after the Phase 2 extractions.

### Phase 3: Split `createAgentStore()` — Status: Not Started
- **Goal:** Break the ~700-line `createAgentStore()` closure (`frontend/src/lib/stores/agent.ts`, lines ~300-1013) into separate modules for cross-tab presence, WebSocket lifecycle, and the event reducer, with identical runtime behavior.
- **Requirements:**
  - [ ] Extract `createSessionPresence()` — the `BroadcastChannel`-based cross-tab protocol (`who_has_session`/`i_have_session`, `announcePresence`/`siblingHoldsSession`/`stopAnnouncing`) — into its own module.
  - [ ] Extract a socket-connection module handling WebSocket open/teardown/reconnect logic into its own module, keeping `WebSocket` instantiation encapsulated per `frontend/AGENTS.md`'s "WebSocket Stores" standard.
  - [ ] `createAgentStore()` keeps only state and the event reducer, consuming both extracted pieces as collaborators.
  - [ ] Every exported function/store from the new modules gets a JSDoc comment per `frontend/AGENTS.md`.
- **Edge Cases & Error Handling:**
  - Reconnection timing/backoff behavior must be identical before and after.
  - Cross-tab presence handoff (a sibling tab already holding the session) must behave identically, including message ordering on the `BroadcastChannel`.
- **Affected Areas:** `frontend/src/lib/stores/agent.ts` and any test file exercising it directly (`frontend/src/lib/stores/agent.test.ts`) — tests should need no behavioral changes, only updated imports if they reach into internals.
- **Acceptance Criteria:**
  - [ ] `createAgentStore()` no longer directly contains presence-protocol or socket-lifecycle code.
  - [ ] Frontend test suite passes with the same results as the Phase 0 baseline.
  - [ ] `npm run check` and `npm run lint` clean.
- **Testing Notes:** Run `frontend/src/lib/stores/agent.test.ts` specifically, then the full frontend suite.

### Phase 4: Opportunistic fixes — dedup and typed classification — Status: Not Started
- **Goal:** Fix two small, independent issues the audit flagged as cheap and low-risk: a duplicated helper and a fragile string-matching boundary.
- **Requirements:**
  - [ ] Delete the duplicate `_window_for_consumer` in `backend/services/compaction.py` (line ~669); import it from `backend/services/model_resolution.py` (line ~163) instead, where it stays.
  - [ ] In `backend/api/v1/agent_session/`'s commit-sweeping collaborator (from Phase 2 — `_resolve_commit_failures`), replace the substring/heuristic classification of the user's free-typed answer (`answer == COMMIT_FAILURE_COMMITTED.lower() or ('commit' in answer and 'without' not in answer and 'not ' not in answer)`) with an explicit, typed classification: either constrain the questioner to return the chosen option's identity directly when a fixed option list (`COMMIT_FAILURE_OPTIONS`) is offered, or centralize the free-text interpretation in one clearly-named function instead of an inline boolean expression.
- **Edge Cases & Error Handling:**
  - Every currently-supported user answer (including the exact wording of `COMMIT_FAILURE_COMMITTED` and any other accepted options) must classify identically to before the change.
- **Affected Areas:** `backend/services/compaction.py`, `backend/services/model_resolution.py`, the commit-sweeping file from Phase 2.
- **Acceptance Criteria:**
  - [ ] Only one definition of `_window_for_consumer` exists in the backend.
  - [ ] Commit-failure answer classification no longer relies on ad-hoc substring matching.
  - [ ] Backend test suite passes with the same results as the Phase 0 baseline.
  - [ ] `mypy`/`basedpyright` clean.
- **Testing Notes:** Exercise the commit-failure resolution flow's existing tests (if any) for every previously-accepted answer wording; if no such test exists, this is a good candidate to *recommend* (not silently add) per the Global Rules test-writing note.

### Phase 5: Split `coder.py` into `coder_tools/` — Status: Not Started
- **Goal:** Split the (now-deduplicated, per Phase 1) ~1200-line `backend/agents/coder.py` — around 30 top-level tool functions — into a `coder_tools/` package grouped by category, re-exported for registration.
- **Requirements:**
  - [ ] Create `backend/agents/coder_tools/` with `files.py` (file-mutation tools), `search.py` (search tools), `verification.py` (test/lint/verification tools), and `meta.py` (`ask_user`, `complete_task`, `complete_step`, and similar meta tools).
  - [ ] Re-export everything needed for tool registration so `coder_agent`'s tool set is unchanged.
  - [ ] Re-measure line counts after the split; none of the four new files should be anywhere near the 1000-line threshold given ~30 functions split four ways.
- **Edge Cases & Error Handling:** None specific — this is a mechanical move, not a logic change.
- **Affected Areas:** `backend/agents/coder.py` (becomes a package or a thin re-export point — decide based on whether anything besides tool functions remains in it), any file importing individual tool functions directly.
- **Acceptance Criteria:**
  - [ ] All ~30 tool functions are grouped into the four category files.
  - [ ] `coder_agent`'s registered tool set is unchanged (verify by inspection or a test that lists registered tools, if one exists).
  - [ ] Backend test suite passes with the same results as the Phase 0 baseline.
  - [ ] `mypy`/`basedpyright` clean.
- **Testing Notes:** Full backend suite; pay attention to any test that imports a tool function directly by its old path.

### Phase 6: Decompose `settings/+page.svelte` — Status: Not Started
- **Goal:** Split the 819-line settings page (script block ~306 lines, mostly markup otherwise) into a layout root plus three focused child components.
- **Requirements:**
  - [ ] Extract `IndexStatusPanel.svelte` (index status display, `refreshIndexStatus`, `deleteIndex`).
  - [ ] Extract `SecretsEditor.svelte` (secrets editing UI).
  - [ ] Extract `ModelSettingsPanel.svelte` (`describeEmbeddingModel`, `detectContextWindow`, and related model-settings UI).
  - [ ] `+page.svelte` keeps the repo-picker (`validateRepo`) and theme controls directly, composing the three extracted components, per `frontend/AGENTS.md`'s "Keep data-fetching and store subscriptions in page-level components" guidance.
  - [ ] Pass data down via props; communicate upward via dispatched events or callback props, per `frontend/AGENTS.md`.
- **Edge Cases & Error Handling:** None specific — preserve existing error/loading states for each panel exactly.
- **Affected Areas:** `frontend/src/routes/settings/+page.svelte`, `frontend/src/routes/settings/page.test.ts` (update to test the composed page and/or the new components as appropriate — no behavior change expected).
- **Acceptance Criteria:**
  - [ ] `+page.svelte` is under ~200 lines, or close to it (per `frontend/AGENTS.md`'s component-size standard), with the three panels extracted.
  - [ ] Frontend test suite passes with the same results as the Phase 0 baseline.
  - [ ] `npm run check` and `npm run lint` clean.
- **Testing Notes:** Run `frontend/src/routes/settings/page.test.ts` and the full frontend suite; manually exercise the settings page in the browser (repo picker, secrets editor, model settings, index status/deletion, theme toggle) since this phase touches user-facing UI directly — per the general guidance to verify UI changes in a browser, not just via test suites.

### Phase 7: One-class-per-file — `backend/services/event_bus.py` — Status: Not Started
- **Goal:** Split the 22-class `event_bus.py` into a package with one file per class.
- **Requirements:**
  - [ ] Create `backend/services/events/` containing: `agent_event_type.py`, `session_status.py`, `sandbox_stage.py`, `tool_status.py`, `compaction_run_path.py` (the 5 enums), `base_agent_event.py`, and one file each for the 15 event models (`thought_event.py`, `plan_event.py`, `tool_action_event.py`, `tool_observation_event.py`, `commit_event.py`, `diff_event.py`, `question_event.py`, `review_event.py`, `done_event.py`, `error_event.py`, `context_usage_event.py`, `compaction_event.py`, `warning_event.py`, `system_event.py`, `sandbox_unavailable_event.py`), plus `event_bus.py` for the `EventBus` class itself.
  - [ ] Add `backend/services/events/__init__.py` re-exporting all 22 classes (required, not optional — see Global Rules — Import style for split-out classes).
  - [ ] Update every import site across the codebase referencing the old `backend.services.event_bus` module to import directly from `backend.services.events` (the package root), never from an individual submodule like `backend.services.events.thought_event`.
  - [ ] Delete the old `backend/services/event_bus.py` file entirely (fully replaced by the package).
- **Edge Cases & Error Handling:** All 15 event models subclass `BaseAgentEvent` — ensure `base_agent_event.py` has no circular dependency back on any subclass; none is expected, but verify.
- **Affected Areas:** `backend/services/event_bus.py` (deleted, replaced by `backend/services/events/`), every file importing from it.
- **Acceptance Criteria:**
  - [ ] Each of the 22 classes lives in its own file; no file in `backend/services/events/` defines more than one class.
  - [ ] Backend test suite passes with the same results as the Phase 0 baseline.
  - [ ] `mypy`/`basedpyright` clean.
- **Testing Notes:** Full backend suite — `EventBus` and its event types are used broadly across session/streaming code.

### Phase 8: One-class-per-file — `backend/schemas/agent_messages.py` — Status: Not Started
- **Goal:** Split the 18-class `agent_messages.py` into a package with one file per class, mirroring Phase 7's approach.
- **Requirements:**
  - [ ] Create `backend/schemas/agent_messages/` containing: `message_type.py` (the enum), `base_agent_message.py`, and one file each for the 16 message models (`thought_message.py`, `plan_message.py`, `action_message.py`, `observation_message.py`, `commit_message.py`, `diff_message.py`, `question_message.py`, `review_message.py`, `done_message.py`, `error_message.py`, `context_usage_message.py`, `compaction_message.py`, `warning_message.py`, `session_busy_message.py`, `system_message.py`, `sandbox_unavailable_message.py`).
  - [ ] Add `backend/schemas/agent_messages/__init__.py` re-exporting all 18 classes (required, not optional — see Global Rules — Import style for split-out classes).
  - [ ] Update every import site across the codebase referencing the old `backend.schemas.agent_messages` module to import directly from `backend.schemas.agent_messages` (the package root), never from an individual submodule like `backend.schemas.agent_messages.thought_message`.
  - [ ] Delete the old `backend/schemas/agent_messages.py` file entirely.
- **Edge Cases & Error Handling:** Same circular-dependency check as Phase 7, applied to `BaseAgentMessage` and its subclasses.
- **Affected Areas:** `backend/schemas/agent_messages.py` (deleted, replaced by the package), every file importing from it.
- **Acceptance Criteria:**
  - [ ] Each of the 18 classes lives in its own file.
  - [ ] Backend test suite passes with the same results as the Phase 0 baseline.
  - [ ] `mypy`/`basedpyright` clean.
- **Testing Notes:** Full backend suite.

### Phase 9: One-class-per-file — small schema/DTO files — Status: Not Started
- **Goal:** Split the remaining small multi-class schema/DTO files (2-5 classes each, no folder needed — siblings in the existing directory).
- **Requirements:**
  - [ ] `backend/database/enums.py` → `task_status.py`, `task_log_entry_type.py` (in `backend/database/`).
  - [ ] `backend/agents/planner.py` → extract `PlannedStep` to `planned_step.py` and `StructuredPlan` to `structured_plan.py` (in `backend/agents/`); any non-class planner logic remaining in `planner.py` imports them back.
  - [ ] `backend/services/chunking.py` → extract `SymbolSpan` to `symbol_span.py` and `Chunk` to `chunk.py` (in `backend/services/`); any non-class chunking logic remaining in `chunking.py` imports them back.
  - [ ] `backend/schemas/agent_commands.py` → `command_action.py`, `session_command.py`.
  - [ ] `backend/schemas/context_window.py` → `context_window_detection_status.py`, `context_window_detection_request.py`, `context_window_detection_response.py`.
  - [ ] `backend/schemas/index_maintenance.py` → `index_status_response.py`, `index_deletion_response.py`.
  - [ ] `backend/schemas/task_response.py` → keep `TaskResponse` in `task_response.py`; extract `TaskLogEntryResponse`, `TaskDetailResponse`, `TaskListResponse`, `TaskPatchRequest` into their own files (`task_log_entry_response.py`, `task_detail_response.py`, `task_list_response.py`, `task_patch_request.py`).
  - [ ] Update every import site across the codebase referencing any of the old module paths above for the classes that moved.
- **Edge Cases & Error Handling:** None specific — straightforward mechanical splits.
- **Affected Areas:** The 7 files listed above, plus every file importing from them.
- **Acceptance Criteria:**
  - [ ] No file among these 7 (or their replacements) defines more than one class.
  - [ ] Backend test suite passes with the same results as the Phase 0 baseline.
  - [ ] `mypy`/`basedpyright` clean.
- **Testing Notes:** Full backend suite.

### Phase 10: One-class-per-file — process/execution-layer services — Status: Not Started
- **Goal:** Split the process/execution-layer service files (each has one primary service/manager class plus 1-3 small support types) into one file per class, as siblings in `backend/services/` (each split produces ≤4 files — no folder needed per the convention's threshold).
- **Requirements:**
  - [ ] `backend/services/docker_sandbox.py` → `sandbox_error.py` (`SandboxError`), `command_result.py` (`CommandResult`); keep `DockerSandboxManager` in `docker_sandbox.py`.
  - [ ] `backend/services/git_manager.py` → `file_change.py`, `commit_info.py`, `commit_result.py`; keep `GitManagerService` in `git_manager.py`.
  - [ ] `backend/services/file_editor.py` → `edit_result.py`; keep `FileEditorService` in `file_editor.py`.
  - [ ] `backend/services/session_fsm.py` → `session_state.py` (`SessionState` enum), `invalid_state_transition_error.py`; rename the remaining file to `session_state_machine.py` to match `SessionStateMachine` (the file no longer matches its old "fsm" abbreviation once it's the eponymous file for just this one class).
  - [ ] `backend/services/user_questions.py` → `client_gone.py`, `stop_requested.py`; rename the remaining file to `user_questioner.py` to match `UserQuestioner`.
  - [ ] Update every import site across the codebase referencing any of the old module paths for classes/files that moved or were renamed.
- **Edge Cases & Error Handling:** None specific.
- **Affected Areas:** The 5 files listed above, plus every file importing from them.
- **Acceptance Criteria:**
  - [ ] No file among these 5 (or their replacements) defines more than one class.
  - [ ] Backend test suite passes with the same results as the Phase 0 baseline.
  - [ ] `mypy`/`basedpyright` clean.
- **Testing Notes:** Full backend suite; pay particular attention to sandbox and git-related tests given these services' role in the execution/verification loop.

### Phase 11: One-class-per-file — retrieval/context-management services — Status: Not Started
- **Goal:** Split the retrieval/context-management service files into one file per class, as siblings in `backend/services/`.
- **Requirements:**
  - [ ] `backend/services/code_search.py` → `lexical_match_mode.py`, `code_search_outcome.py`, `lexical_match.py`, `symbol_definition.py`; keep `CodeSearchService` in `code_search.py`.
  - [ ] `backend/services/embedding.py` → `embedding_error.py` (`EmbeddingError`), `model_description.py`, `embedding_input_too_long.py` (subclasses `EmbeddingError` — import it), `chunk_too_large_for_model.py` (same); keep `EmbeddingService` in `embedding.py`.
  - [ ] `backend/services/compaction.py` → `observed_compaction.py`, `request_history_capture.py`, `summarize_and_strip_tool_messages.py` (all three currently subclass `AbstractCapability[Any]`); non-class logic (including the now-deduplicated `_window_for_consumer` import from Phase 4) stays in `compaction.py`.
  - [ ] `backend/services/context_budget.py` → `provider_usage_divergence.py`; keep `ReportProviderUsage` in `context_budget.py`.
  - [ ] `backend/services/conversation_history.py` → `turn_role.py`, `conversation_turn.py`; keep `ConversationHistory` in `conversation_history.py`.
  - [ ] `backend/services/plan_parser.py` → `plan_format.py`, `plan_step.py`, `parsed_plan.py`; any non-class parsing logic remaining in `plan_parser.py` imports them back (note: this file has no primary class of its own once these three move out — that's fine, zero classes satisfies the convention trivially).
  - [ ] Update every import site across the codebase referencing any of the old module paths for classes that moved.
- **Edge Cases & Error Handling:** `EmbeddingInputTooLong` and `ChunkTooLargeForModel` subclassing `EmbeddingError` from a different file is a realistic circular-import risk only if `embedding_error.py` ever needs to reference the subclasses — it shouldn't; verify it doesn't.
- **Affected Areas:** The 6 files listed above, plus every file importing from them.
- **Acceptance Criteria:**
  - [ ] No file among these 6 (or their replacements) defines more than one class.
  - [ ] Backend test suite passes with the same results as the Phase 0 baseline.
  - [ ] `mypy`/`basedpyright` clean.
- **Testing Notes:** Full backend suite; pay particular attention to compaction and code-search tests.

## Definition of Done
- Every phase above has Status: Done, with every Requirements and Acceptance Criteria checkbox checked and verified per the Global Rules.
- The backend test suite passes with the same pass/fail/skip counts as the Phase 0 baseline; `mypy`/`basedpyright` are clean.
- The frontend test suite passes with the same pass/fail/skip counts as the Phase 0 baseline; `npm run check` and `npm run lint` are clean.
- No non-test backend Python file defines more than one class (re-verify with a repo-wide scan, not just per-phase spot checks).
- No behavioral change is observable anywhere: API responses, event/message payloads, and UI behavior are identical to before this plan started.
- All work is on the `refactor/maintainability-audit` branch, with one commit per completed phase (Phase 0 folded into Phase 1's commit).

## Open Questions
- **Resolved:** every package this plan creates (Phases 2, 7, 8, and any resulting from Phases 9-11) re-exports its classes from `__init__.py`, and every call site imports directly from the package root rather than the specific submodule — confirmed by the user as the correct reading of `backend/AGENTS.md`'s "Preferred Type References" standard. See Global Rules — Import style for split-out classes.
- **Left to the implementing agent's judgment:** exact file-naming choices beyond what's specified above (e.g. whether a given support-type file is named exactly after its class in snake_case, as assumed throughout) — deviate only if the codebase's existing naming conventions clearly suggest otherwise for a specific case.
- **Left to the implementing agent's judgment:** whether `backend/agents/coder.py` (Phase 5) becomes a thin package `__init__.py` re-exporting from `coder_tools/`, or is deleted entirely in favor of direct imports from `coder_tools/` — depends on how `coder_agent`'s tool registration currently works, which should be inspected at that phase's start.

## Non-Goals / Explicit Exclusions
- This plan does not re-run the `nuke` audit after these changes land to look for new findings — that would be a separate, later exercise if desired.
- This plan does not apply the one-class-per-file convention to `backend/tests/` files (`test_agent_session.py`, `test_indexing.py`) or to `frontend/src/lib/stores/agent.test.ts` — explicitly excluded per the user's decision during planning.
