# PLAN: Feedback trail across review rounds

## Summary
Loom's agent session runs a plan → execute → review loop that can be rejected and
re-worked many times, by the automated reviewer agent or by the user. Today, the
coder agent that actually executes each step never sees any of that feedback
directly — only the planner does, and only via a lossy, token-budgeted summary
that collapses anything older than the last few turns to a 160-character stub.
This plan adds a small, dedicated, never-summarized "feedback trail" that carries
reviewer and user feedback across every round, in full, to both the planner and
(for the two sources that concern actual code) the coder.

## Type
Feature

## Global Rules
These apply across the entire plan and to every phase.

- **Execution mode:** One-shot. All phases are executed in one continuous run; the
  agent reports everything at the end rather than pausing after each phase.
- **Commit cadence:** After every phase. Phase 0 is never committed on its own — it
  does not change the project, so its work is folded into the commit that follows
  Phase 1. Since execution mode is one-shot, "after every phase" means the agent
  commits each phase's work as it finishes that phase and moves on, in sequence,
  without pausing for review in between.
- **Branch strategy:** Create a new local branch for this work before Phase 1
  begins. Suggested name: `feature/feedback-trail` (the agent may adjust if that
  name collides with something already present).
- **PLAN.md handling:** `PLAN.md` is committed alongside the project's commits as
  work proceeds. Once the Definition of Done is met and the final summary is
  written, delete `PLAN.md` and commit that deletion as the very last action.
- **Baseline test run:** Before starting any implementation work, from `backend/`,
  activate the existing virtual environment (`source .venv/bin/activate` — it is
  already created and has the project installed in editable mode with dev
  dependencies; if it is somehow missing, `./dev.sh` recreates it) and run:
  `python -m pytest -m "not integration"`. This repo also has a `pytest -m integration`
  subset that needs a live Postgres (`TEST_DATABASE_URL`) and is not expected to be
  available — do not attempt to set that up; the non-integration run is the
  baseline. Note the result (1000 tests collected as of this writing, all expected
  to pass) so any later regression is attributable to this work specifically.
- **Self-verification:** After each phase, run `python -m pytest -m "not integration"`
  from `backend/` and treat any new failure beyond the established baseline as a
  blocking problem (see "Failed verification blocks completion"). Also run
  `mypy . && basedpyright .` from `backend/` after each phase — both must report
  no new errors. Writing new tests is not optional here: the Requirements below
  name specific tests to add, because this plan's correctness is exactly "the
  right prompts contain the right text," which is only checkable through tests.
- **Record how each Acceptance Criterion was verified.** When checking off an
  Acceptance Criteria item, append a short note on how it was confirmed — e.g.
  "— verified by: `pytest tests/test_feedback_trail.py`", "— verified by: re-read
  the diff against the requirement."
- **Consult, don't assume:** Ask the user directly about implementation details or
  design choices whenever there's a real choice to make, rather than silently
  picking one. This plan's design was already settled with the user in detail
  before writing (see Background & Context) — that settled design is not open for
  silent reinterpretation, but anything this plan doesn't cover is.
- **Problems get recommendations:** Whenever a problem is identified — a bug, a
  risk, unclear scope, tech debt — accompany it with a concrete recommendation,
  not just a flag.
- **Failed verification blocks completion:** If a phase's Acceptance Criteria
  aren't met, or self-verification surfaces a new test failure or a new
  mypy/basedpyright error, the agent must not mark that phase Done. Attempt to fix
  it within the same phase first; if the fix isn't obvious or needs a design
  decision, stop and ask the user rather than proceeding.
- **Keep the plan current:** `PLAN.md` is a living document. If the user directs a
  change not in the original plan, update `PLAN.md` itself to absorb it — add to
  the relevant phase, insert a new phase, or adjust Scope — rather than keeping a
  separate changelog.
- **AGENTS.md conventions (mandatory, from `backend/AGENTS.md`):** reST-style
  docstrings on every new function/class (summary line, `:param:`, `:returns:`,
  `:raises:` where applicable); `Annotated[T, Field()]` on every field of any new
  Pydantic model; PEP 585 types (`list[str]`, not `List[str]`); single quotes for
  strings; an `Enum` for any field/variable with 3+ named literal values; prefer
  guard clauses / early returns; prefer direct imports over dotted references for
  type annotations and instantiation; no functions whose primary purpose is a
  side effect returning `None` — prefer returning a new value. After all phases,
  run `mypy . && basedpyright .` from `backend/` and fix any error it reports,
  even one unrelated to this work, per `AGENTS.md`'s "Post-Task" rule.
- **If one-shot:** Execute all phases in sequence without pausing for feedback
  between them, committing after each as described above, then write one
  comprehensive summary at the end covering everything implemented across all
  phases, plus any recommendations for problems noticed but not addressed.

## Background & Context

### Current behavior (confirmed by reading the code, not assumed)
- The coder's step prompt is built by `AgentSession._build_step_prompt()` in
  `backend/api/v1/agent_session/agent_session.py`. It includes the original task
  (`self._task`), the current approved plan, the established test command, and
  summaries of steps already completed in the current plan — and nothing else. It
  never includes reviewer `comments` or a user's rejection `reason`, from any
  round.
- All feedback is mediated through the planner. `PlanningCoordinator.build_planning_prompt()`
  in `backend/api/v1/agent_session/planning_coordinator.py` builds the planner's
  prompt from `self._session._history.build_context(CONTEXT_TOKEN_BUDGET)`
  (budget: 4096 tokens, `CONTEXT_TOKEN_BUDGET` in that file).
- `ConversationHistory.build_context()` (`backend/services/conversation_history.py`)
  keeps only the last `DEFAULT_RECENT_TURNS_KEPT` (3) turns verbatim; everything
  older collapses to one summary line per iteration via `_summarise_iteration()`,
  which runs every reviewer comment and user rejection reason through `_shorten()`,
  hard-truncating each to 160 characters. So even the planner's view of feedback
  degrades once more than one re-working round has happened — the triggering
  round usually survives close to intact (it's typically still within the last 3
  turns when the next planning prompt is built), but earlier rounds do not.
- There are three distinct places a "send this back" moment happens in
  `backend/api/v1/agent_session/agent_session.py`:
  1. User rejects the **plan** before execution — `_handle_waiting_plan_approval`
     (around line 1103–1126), which calls `self._history.add(TurnRole.user,
     command.reason or '...', metadata={'rejection': True})` and
     `self._record_rejection('plan', command.reason)`.
  2. The automated **reviewer** rejects the diff — `_handle_waiting_review_approval`
     (around line 1488–1519), which calls `self._history.add(TurnRole.reviewer,
     review.comments, metadata={'approved': review.approved})`.
  3. User rejects the already-**reviewer-approved result** before merge —
     `_handle_completed` (around line 1530–1557), which calls
     `self._history.add(TurnRole.user, command.reason or '...', metadata={'rejection': True})`
     and `self._record_rejection('result', command.reason)`.
  - All three transition to `_handle_re_working`, which simply moves the state
    machine back to `SessionState.planning`.
- `ConversationHistory.iteration` already advances once per planning round via
  `advance_iteration()`, called from `PlanningCoordinator.handle_planning()`. Every
  `ConversationTurn` records the iteration it was added in.
- Separately, `backend/services/task_recorder.py`'s `TaskRecorder` already persists
  the full, untruncated text of both reviewer verdicts (`TaskLogEntryType.review`,
  via the event bus) and user rejection reasons (`TaskLogEntryType.rejection`, via
  `record_rejection()`) to the `TaskMemory` database table, and these are already
  retrievable in full after the fact via the task-detail API endpoint. This
  existing mechanism is **not** touched by this plan — it serves a different
  purpose (post-hoc audit), `TaskRecorder` is optional (`self._recorder` can be
  `None`), and there is no session-resume capability anywhere in this codebase
  (`ConversationHistory` and all other session state are purely in-memory and are
  discarded on reconnect or restart — confirmed by searching for
  resume/reconnect/rehydrate logic and finding none). The feedback trail this plan
  adds is therefore also purely in-memory, scoped to one `AgentSession` instance,
  and is not expected to survive a restart — consistent with everything else the
  session already holds.
- `backend/services/review_context.py` is the closest existing pattern for a new
  small prompt-building service module: pure functions, `Sequence`/`Callable`
  parameters, reST docstrings, no classes.

### Design, already settled with the user before this plan was written
1. A new module, `backend/services/feedback_trail.py`, holding:
   - `FeedbackSource`, a `str` `Enum` with three members: `reviewer_rejection`,
     `user_plan_rejection`, `user_result_rejection`.
   - `FeedbackEntry`, a Pydantic model with `source: Annotated[FeedbackSource, Field()]`,
     `iteration: Annotated[int, Field(ge=1)]`, and `text: Annotated[str, Field()]`
     (the full, untruncated reviewer comment or rejection reason).
   - `FeedbackTrail`, a small class holding an internal list of `FeedbackEntry`,
     with an `add(source, iteration, text) -> FeedbackEntry` method and an
     `entries` property returning an immutable view (mirroring
     `ConversationHistory.turns`). It never summarizes, truncates, or caps
     anything — every entry it is given persists in full for the life of the
     trail.
   - `render_for_planner(trail: FeedbackTrail) -> str`: renders every entry, from
     all three sources, in full, each labeled by source and iteration/round
     number. An empty trail renders to an empty string.
   - `render_for_coder(trail: FeedbackTrail) -> str`: renders only entries whose
     source is `reviewer_rejection` or `user_result_rejection`, in full, same
     labeling style. `user_plan_rejection` entries must never appear in this
     function's output, even when other entries are present. An empty trail, or
     one containing only `user_plan_rejection` entries, renders to an empty
     string.
2. `AgentSession` gains a `_feedback_trail: FeedbackTrail` field (declared
   alongside `_history: ConversationHistory`, constructed in `__init__` alongside
   where `_history` is assigned). It is appended to, alongside (not instead of)
   the existing `self._history.add(...)` / `self._record_rejection(...)` calls, at
   the three points listed above:
   - Plan rejection → `FeedbackSource.user_plan_rejection`.
   - Reviewer rejection (only when `review.approved` is `False`) →
     `FeedbackSource.reviewer_rejection`, using `review.comments`.
   - Result rejection → `FeedbackSource.user_result_rejection`.
   - In each case, the `iteration` recorded is `self._history.iteration` at that
     point, reusing the existing iteration counter rather than introducing a new
     one.
3. `PlanningCoordinator.build_planning_prompt()` gains an additional prompt
   section built from `render_for_planner(self._session._feedback_trail)`,
   appended after the existing `ConversationHistory`-derived `context` section,
   included only when non-empty. This is additive: the existing
   `ConversationHistory` summary continues to cover mechanical progress (files
   touched, tests run) that the feedback trail does not attempt to duplicate.
4. `AgentSession._build_step_prompt()` gains an additional section built from
   `render_for_coder(self._feedback_trail)`, included whenever non-empty — i.e.
   on every step of any plan produced after at least one reviewer or result
   rejection, with no special-casing of "first step only."
5. Why the coder sees raw feedback at all, given the planner is designed to
   produce self-contained steps: this is deliberate defense-in-depth, decided
   with the user, against the planner mistranslating or thinning out feedback
   when turning it into step text — not a redesign of the planner's role.
6. Why plan-rejection feedback is withheld from the coder specifically: it
   describes a plan the coder never executed and has nothing concrete to
   cross-check against, so it would add noise rather than something actionable;
   it remains visible to the planner, which is what acts on plan-level
   objections.
7. Why "every round, forever" rather than a bounded window: decided with the
   user — realistic volume is small (default session limits cap a task at 10 plan
   rejections and 10 review rejections before the system pauses to ask the user
   how to proceed; see `DEFAULT_MAX_PLAN_RETRIES` / `DEFAULT_MAX_REVIEW_RETRIES`
   in `backend/services/session_limits.py`), so there is no real context-budget
   pressure from never summarizing this specific, comparatively small trail.

## Goal / Desired Behavior
- Whenever the planner builds its prompt for a new planning round, and the
  session has accumulated any reviewer rejection, user plan-rejection, or user
  result-rejection feedback from any earlier round, that feedback appears in the
  planner's prompt in full (not summarized, not truncated), labeled by its source
  and which round it came from.
- Whenever the coder is given a step to execute, and the session has accumulated
  any reviewer-rejection or user-result-rejection feedback from any earlier
  round, that feedback appears in the coder's step prompt in full, labeled the
  same way. User plan-rejection feedback never appears in the coder's prompt.
- Nothing about the existing plan/execute/review state machine, its transitions,
  or its existing prompts changes in any way other than these additive sections.
  The original task continues to reach both planner and coder exactly as it does
  today (this was already confirmed correct and needs no change).

## Scope
- **In scope:**
  - The new `backend/services/feedback_trail.py` module and its unit tests.
  - Wiring `AgentSession` to populate the trail at the three existing
    rejection/review points.
  - Rendering the trail into `PlanningCoordinator.build_planning_prompt()` and
    `AgentSession._build_step_prompt()`.
  - Tests proving both the unit-level rendering behavior and the end-to-end
    wiring (planner prompt and coder step prompt actually contain the right
    text, and the coder prompt actually excludes plan-rejection text).
- **Out of scope:**
  - Any change to `ConversationHistory`, its summarization/truncation logic, or
    `review_context.py`'s existing `previous_feedback` mechanism (which serves
    the reviewer agent's own self-consistency check on re-review, a separate,
    unrelated concern).
  - Any change to `TaskRecorder`/`TaskMemory` persistence, the task-detail API,
    or anything DB-related. No database migration is involved in this plan.
  - Any session-resume or cross-restart persistence for the feedback trail —
    this codebase has no session-resume capability at all, for anything, so this
    is not a gap this plan needs to address.
  - Any frontend/UI change. This is a backend prompt-construction change only;
    nothing about it is user-visible except indirectly, through better-behaved
    re-working rounds.
  - Capping or bounding the feedback trail's size. If this becomes a real problem
    in practice, it would be a follow-up, not part of this plan.

## Phases

### Phase 0: Environment Setup — Status: Done
- From `backend/`, activate the existing virtual environment:
  `source .venv/bin/activate`. It is already created with the project installed
  in editable mode (`pip install -e .[dev]`); if it is missing for some reason,
  recreate it by sourcing `./dev.sh` instead (it creates and activates the venv
  and installs dependencies).
- Confirm the test suite runs: `python -m pytest -m "not integration"` from
  `backend/`. As of writing this collects 1000 tests with no setup needed beyond
  the venv (the `integration` marker subset needs `TEST_DATABASE_URL` pointing at
  a live Postgres with pgvector and is not expected to be available — do not
  attempt to provision it; it is excluded from this plan's baseline and
  self-verification runs).
- Confirm linting runs: `mypy . && basedpyright .` from `backend/`.
- Record the baseline pass/fail state of both commands before making any change.
- Create the work branch: `git checkout -b feature/feedback-trail` (adjust the
  name only if it collides with something already present).

### Phase 1: Feedback trail module — Status: Done
- **Goal:** introduce `backend/services/feedback_trail.py` as a standalone,
  fully-tested pure module, with no dependency on `AgentSession` yet.
- **Requirements:**
  - [ ] `FeedbackSource` enum with exactly the three members: `reviewer_rejection`,
    `user_plan_rejection`, `user_result_rejection`.
  - [ ] `FeedbackEntry` Pydantic model with `source`, `iteration` (≥ 1), and
    `text` fields, each `Annotated[T, Field()]` per `AGENTS.md`.
  - [ ] `FeedbackTrail` class with an `add(source, iteration, text) -> FeedbackEntry`
    method and an `entries` property returning an immutable sequence, in
    insertion order, with no summarization, truncation, or size cap of any kind.
  - [ ] `render_for_planner(trail: FeedbackTrail) -> str` — all entries, in full,
    labeled by source and iteration; empty string for an empty trail.
  - [ ] `render_for_coder(trail: FeedbackTrail) -> str` — only
    `reviewer_rejection` and `user_result_rejection` entries, in full, same
    labeling; empty string when there is nothing to show.
  - [ ] Every function/class has a reST docstring per `AGENTS.md`.
- **Edge Cases & Error Handling:**
  - An empty trail must render to an empty string from both render functions (not
    a section header with nothing under it), so callers can cheaply decide
    whether to include the section at all, matching how `build_planning_prompt()`
    already treats its `context` variable.
  - A trail containing only `user_plan_rejection` entries must render to an empty
    string from `render_for_coder`, even though `entries` itself is non-empty.
  - A trail with entries from all three sources must, in `render_for_coder`'s
    output, contain the reviewer and result-rejection text and must not contain
    the plan-rejection text anywhere in the string.
  - Very long feedback text (e.g. several paragraphs) must pass through
    `render_for_planner`/`render_for_coder` unmodified — no truncation of any
    kind is applied by this module.
- **Affected Areas:**
  - New file: `backend/services/feedback_trail.py`.
  - New file: `backend/tests/test_feedback_trail.py`.
- **Acceptance Criteria:**
  - [x] `FeedbackTrail.add()` followed by reading `.entries` returns the same
    entries back, in insertion order, with source/iteration/text preserved
    exactly. — verified by: `pytest tests/test_feedback_trail.py` (`test_entries_come_back_in_order_with_everything_preserved`).
  - [x] `render_for_planner` on a trail with one entry per source produces output
    containing all three entries' full text. — verified by: `pytest tests/test_feedback_trail.py` (`test_planner_sees_every_source_in_full`).
  - [x] `render_for_coder` on the same trail produces output containing the
    reviewer-rejection and result-rejection text, and provably not containing the
    plan-rejection text (an explicit assertion that the plan-rejection text is
    absent, not merely that the other two are present). — verified by: `pytest tests/test_feedback_trail.py` (`test_coder_sees_reviewer_and_result_feedback_but_not_plan_feedback`).
  - [x] `render_for_coder` on a trail containing only a `user_plan_rejection`
    entry returns an empty string. — verified by: `pytest tests/test_feedback_trail.py` (`test_coder_sees_nothing_for_a_plan_rejection_only_trail`).
  - [x] Both render functions return an empty string for an empty trail. — verified by: `pytest tests/test_feedback_trail.py` (`test_empty_trail_renders_to_empty_for_both_audiences`).
  - [x] `mypy . && basedpyright .` reports no new errors. — verified by: ran both from `backend/`, clean.
- **Testing Notes:** New file `backend/tests/test_feedback_trail.py`, mirroring
  `backend/tests/test_review_context.py`'s style — module-level `def test_...`
  functions, a short docstring on each explaining why the behavior matters, and
  black-box assertions against the returned strings. Test, at minimum: the
  add/entries round trip; `render_for_planner` with entries from all three
  sources; `render_for_coder` with entries from all three sources, asserting both
  inclusion (reviewer/result text present) and exclusion (plan-rejection text
  absent); `render_for_coder` on an all-plan-rejection trail; both render
  functions on an empty trail.

### Phase 2: Wire the trail into AgentSession and the planner prompt — Status: Done
- **Goal:** `AgentSession` populates the feedback trail at the three existing
  rejection/review points, and the planner's prompt includes it.
- **Requirements:**
  - [ ] `AgentSession` declares a `_feedback_trail: FeedbackTrail` field and
    constructs it in `__init__`, alongside `_history`.
  - [ ] In `_handle_waiting_plan_approval`, when the user rejects the plan, append
    a `user_plan_rejection` entry to the trail (same reason text already passed to
    `self._history.add(...)` / `self._record_rejection('plan', ...)`), tagged with
    `self._history.iteration`.
  - [ ] In `_handle_waiting_review_approval`, when `review.approved` is `False`,
    append a `reviewer_rejection` entry with `review.comments`, tagged with
    `self._history.iteration`.
  - [ ] In `_handle_completed`, when the user rejects the finished result, append
    a `user_result_rejection` entry (same reason text already passed to
    `self._history.add(...)` / `self._record_rejection('result', ...)`), tagged
    with `self._history.iteration`.
  - [ ] None of the existing `self._history.add(...)` or `self._record_rejection(...)`
    calls are removed or altered — the new calls are additive.
  - [ ] `PlanningCoordinator.build_planning_prompt()` appends a new section built
    from `render_for_planner(self._session._feedback_trail)` after the existing
    `context` section, only when that rendering is non-empty.
- **Edge Cases & Error Handling:**
  - The very first planning round (no rejections yet) must produce a planner
    prompt identical in shape to today's (no new empty section, no stray
    headers).
  - A reviewer approval (`review.approved is True`) must not add anything to the
    trail — only an actual rejection does.
  - Multiple rounds of the same rejection type (e.g. the reviewer rejects twice
    in a row) must each produce their own trail entry, both appearing in the next
    planning prompt — not just the latest one.
- **Affected Areas:**
  - `backend/api/v1/agent_session/agent_session.py`: new field + three call
    sites (`_handle_waiting_plan_approval`, `_handle_waiting_review_approval`,
    `_handle_completed`), plus the `__init__` construction.
  - `backend/api/v1/agent_session/planning_coordinator.py`:
    `build_planning_prompt()`.
- **Acceptance Criteria:**
  - [x] Extend `backend/tests/test_agent_session.py`, using the existing
    `build_session`/`stub_agents`/`opening` harness, with tests asserting the
    planner prompt (`seen['coder']` in that harness's existing naming, which the
    harness's own docstring explains captures planning prompts only, because
    `_handle_executing` is faked out there) contains: reviewer-rejection comment
    text after a reviewer rejection round; the plan-rejection reason after a plan
    rejection round; the result-rejection reason after a result rejection round.
    (`test_rejected_review_loops_back_through_planning` and
    `test_rejected_plan_is_replanned_with_the_reason` already assert something
    adjacent to this for the pre-existing summarized path — extend or add
    alongside them rather than duplicating unrelated setup.)
    — verified by: `pytest tests/test_agent_session.py` (`test_reviewer_rejection_reaches_the_next_plan_in_full`, `test_plan_rejection_reaches_the_next_plan_in_full`, `test_result_rejection_reaches_the_next_plan_in_full`, plus `test_earliest_reviewer_feedback_survives_later_rounds_in_full` proving the full text survives even once the history path would have stubbed it, and `test_an_approved_review_adds_nothing_to_the_trail`).
  - [x] A test confirms the first planning round's prompt is unaffected (no new
    section) when nothing has been rejected yet. — verified by: `pytest tests/test_agent_session.py` (`test_first_planning_prompt_carries_no_feedback_section`).
  - [x] `python -m pytest -m "not integration"` passes with no new failures. — verified by: full run from `backend/`, 878 passed / 4 skipped / 140 pre-existing Postgres-connection errors, identical to the Phase 0 baseline plus the 22 new tests.
  - [x] `mypy . && basedpyright .` reports no new errors. — verified by: ran both from `backend/`, clean.
- **Testing Notes:** Extend `backend/tests/test_agent_session.py` rather than
  creating a new file, matching its existing fixtures and conventions.

### Phase 3: Wire the trail into the coder's step prompt — Status: Not Started
- **Goal:** `_build_step_prompt()` includes reviewer/result-rejection feedback,
  and provably excludes plan-rejection feedback.
- **Requirements:**
  - [ ] `_build_step_prompt()` adds a section built from
    `render_for_coder(self._feedback_trail)`, included whenever non-empty,
    positioned so it reads naturally alongside the step's own description and
    the "steps already done" context already present.
  - [ ] When the trail is empty (no rejections have occurred yet), the step
    prompt's shape is unchanged from today.
- **Edge Cases & Error Handling:**
  - A step prompt built after only a plan rejection (no reviewer or result
    rejection yet) must not contain that plan-rejection's reason text anywhere.
  - A step prompt built after both a reviewer rejection and a plan rejection must
    contain the reviewer's comments and must not contain the plan-rejection
    reason.
  - A step prompt built after a second reviewer rejection must contain both
    rounds' comments, not just the latest.
- **Affected Areas:**
  - `backend/api/v1/agent_session/agent_session.py`: `_build_step_prompt()`.
- **Acceptance Criteria:**
  - [ ] Extend `backend/tests/test_execution_loop.py`, which already drives
    `_handle_executing()`/`_build_step_prompt()` directly against a real
    repository (see `test_each_step_is_told_which_step_it_is` and
    `test_a_later_step_is_told_what_earlier_steps_did` for the existing pattern
    of asserting on built step prompts), with tests asserting:
    - [ ] After a reviewer rejection, the next step's prompt contains the
      reviewer's comments.
    - [ ] After a user result rejection, the next step's prompt contains that
      rejection's reason.
    - [ ] After a user plan rejection, the next step's prompt does **not**
      contain that rejection's reason (the explicit negative case proving the
      coder/planner scope split actually holds, not merely inferred from the
      other assertions).
  - [ ] `python -m pytest -m "not integration"` passes with no new failures.
  - [ ] `mypy . && basedpyright .` reports no new errors.
- **Testing Notes:** Extend `backend/tests/test_execution_loop.py` rather than
  creating a new file. The negative case (plan-rejection text absent) is the
  most important single assertion in this phase — do not skip it in favor of
  only testing the two inclusion cases.

## Definition of Done
- Every phase's Status is Done: all Requirements and Acceptance Criteria
  checkboxes checked, each with a "— verified by: ..." note.
- `python -m pytest -m "not integration"` passes from `backend/`, with no
  regressions relative to the Phase 0 baseline.
- `mypy . && basedpyright .` reports no new errors from `backend/`.
- The Goal / Desired Behavior section is true end-to-end: a planner prompt built
  after any rejection contains the full feedback trail; a coder step prompt built
  after a reviewer or result rejection contains that feedback in full; a coder
  step prompt never contains plan-rejection text.
- `PLAN.md` is deleted, and that deletion committed, as the final action, per the
  PLAN.md handling Global Rule.

## Open Questions
None — the design, scope, execution mode, commit cadence, branch strategy, and
PLAN.md handling were all settled with the user before this plan was written.

## Non-Goals / Explicit Exclusions
- No change to `ConversationHistory`'s summarization/truncation behavior.
- No change to `TaskRecorder`/`TaskMemory`, the task-detail API, or any database
  schema or migration.
- No attempt to add session-resume capability; the feedback trail is in-memory
  and session-scoped, consistent with everything else in `AgentSession`.
- No frontend/UI change.
- No capping or bounding of the feedback trail's size.
