# PLAN: Measure the model's real context window instead of guessing it

## Summary
Loom budgets its coder and planner agents against a user-configured context-window number that may have nothing to do with what the locally-hosted model actually accepts. When a run dies because the prompt filled the real window, the session halves its working number and retries — repeatedly, per run, re-learning the same lesson from scratch every session. This plan replaces guess-and-halve with measurement: probe Ollama for the model's true context length before the first request, give every long-running agent one shared overflow-recovery path (including hard provider rejections, not just empty generations), and reconcile Loom's own token estimates against provider-reported usage so silent server-side truncation and hidden token growth surface before things break.

## Type
Project (multi-phase)

## Global Rules
These apply across the entire plan and to every phase. The implementing agent should treat this section as standing instructions, not a one-time note.

- **Execution mode:** One-shot. Execute all phases in sequence without pausing for feedback between them, then write one comprehensive summary at the end covering everything implemented across all phases, plus any recommendations for problems noticed but not addressed.
- **Commit cadence:** Commit after every phase. Phase 0 is never committed on its own — it only establishes the environment and baseline, so its work is folded into whichever commit follows Phase 1. Even in one-shot mode, each phase gets its own commit as it completes.
- **Branch strategy:** Stay on the current branch. Do not create a new branch for this work.
- **PLAN.md handling:** Not committed, kept. `PLAN.md` is never staged or included in any commit made as part of this work, but remains in the working directory throughout and after.
- **Baseline test run:** Before starting any implementation work, run the backend test suite and note the results: `cd backend && source .venv/bin/activate && pytest` (see Phase 0 if `.venv` does not exist yet). This plan touches backend code only; the frontend is unaffected, so no frontend baseline is required (running `npm run test` once anyway is harmless but optional). Record which failures, if any, pre-date this work, so new failures introduced later can be identified correctly.
- **Self-verification:** Run the backend suite after every phase and treat any new failure beyond the established baseline as a blocking problem (see "Failed verification blocks completion"). `backend/AGENTS.md` additionally requires running the linters as a post-task step: `mypy . && basedpyright .` — fix any error, even one unrelated to the phase at hand.
- **Record how each Acceptance Criterion was verified.** When checking off an Acceptance Criteria item, append a short note on how it was confirmed (e.g. "— verified by: ran backend test suite", "— verified by: traced the wiring from X to Y").
- **Consult, don't assume:** Ask the user directly about implementation details, design choices, or architectural decisions whenever there's a real choice to make, rather than silently picking one. Come to the user with a recommendation rather than an open-ended question where possible. Prefer a structured question/elicitation tool over plain prose when one is available.
- **Problems get recommendations:** Whenever a problem is identified — a bug, a risk, unclear scope, tech debt — accompany it with a concrete recommendation, not just a flag.
- **Failed verification blocks completion:** If a phase's own Acceptance Criteria aren't met, or the phase's changes introduce new test failures beyond the established baseline, do not mark that phase Done. Attempt to fix the issue as part of the same phase first. If it can't be resolved confidently, stop and ask the user rather than proceeding or silently shipping a known-broken phase. In one-shot mode this still applies per phase — do not carry a broken phase forward into the next.
- **Keep the plan current:** `PLAN.md` is a living document. Whenever the user directs a change that wasn't in the original plan, update `PLAN.md` itself to absorb it — add it to the relevant phase, insert a new phase if needed, adjust Scope if needed — integrated as though it had been there from the start.
- **No database migrations:** This plan deliberately makes no schema change (persisting the discovered window to the `Task` record was considered and declined by the user). If implementation reveals a migration seems necessary, stop and consult the user first — that would be a scope change.
- **No new dependencies:** Everything needed (`httpx`, `pydantic-ai-harness`) is already declared in `backend/pyproject.toml`.
- **AGENTS.md conventions apply throughout:** `backend/AGENTS.md` describes mandatory conventions: full type annotations (PEP 585 / `collections.abc` forms), reST docstrings on every function and class, single-quoted strings, guard clauses, no side-effecting `None`-returning functions, Enums over 3+ literal values, direct imports for annotations/instantiation, and the mypy/basedpyright post-task gate. Follow them in every file touched.

## Background & Context

Loom is a self-hosted coding-agent tool: FastAPI + Dishka + TortoiseORM/PostgreSQL backend built on Pydantic AI, driving planner/coder/reviewer agents through a WebSocket-controlled FSM (`backend/services/session_fsm.py`, `backend/api/v1/agent_session.py`). Models are typically local via Ollama's OpenAI-compatible endpoint.

What exists today, and why it produces the reported symptom ("The model had no room to reply … Retrying with 64,024", repeating with each halving):

- **The warning is Loom's own**, emitted by `AgentSession._narrow_after_overflow()` (`backend/api/v1/agent_session.py:679`). `_run_coder_streaming()` (`agent_session.py:778–812`) catches `UnexpectedModelBehavior`, checks it with `looks_like_no_room_to_answer()` (`backend/services/compaction.py:174` — matches 'token limit' + 'before any response', pydantic-ai's wrapper for an empty response with a length finish reason), halves the working window via `narrowed_window()` (`compaction.py:191`), and retries the entire `agent.run()`.
- **The narrowed window lives only on the session object** (`agent_session.py:706`). The `Task` row keeps the originally configured value, and every new session starts over at the configured number — hence halving warnings repeating periodically across runs/sessions.
- **The budget denominator is a guess.** `build_compaction()` (`compaction.py:217`) builds a five-tier `TieredCompaction` ladder (`DeduplicateFileReads` → `ClampOversizedMessages` → `ClearToolResults` → `SummarizingCompaction` → `SlidingWindowCompaction`) with `target_fraction=0.7` against an explicit `context_window`. `model_settings()` (`backend/services/model_resolution.py:11`) sends `num_ctx` to Ollama through `extra_body` — a request, not a guarantee: Ollama clamps it to what the model was trained for, and non-Ollama OpenAI-compatible servers ignore it entirely.
- **Fixed overhead is invisible to compaction.** Roughly 5,470 tokens of system prompt + sixteen tool schemas ride on every request (`estimate_fixed_overhead()` in `backend/services/context_budget.py`). On the first request of a run there is no history to trim, so when overhead + prompt alone exceed the real window, no amount of compaction can help. The prior plan's eighth field report established exactly this: five consecutive halvings failed identically, proving the overflow was never the conversation.
- **A guard already stops useless narrowing**: `context_window_complaint()` (`context_budget.py:47`) fires when window < overhead + `WORKING_HEADROOM_TOKENS` (4,096), and `_narrow_after_overflow` stops halving and surfaces it. `_warn_if_context_is_too_small()` (`agent_session.py:1223`) runs the same check proactively at session start.
- **Measurement precedent already exists**: `EmbeddingService.describe_model()` / `_model_info()` (`backend/services/embedding.py:198, 239`) POST to Ollama's native `/api/show` and read `model_info["<architecture>.context_length"]`, tolerating missing keys by returning None. `config_controller.py` queries `/api/tags` similarly. The chat-model path has no equivalent today.
- **Planner paths lack overflow recovery.** `_plan_structured()` (`agent_session.py:1008`) catches *all* exceptions from the structured attempt and falls back to `_plan_as_text()` (`agent_session.py:1052`), which has no recovery at all — an overflow there fails the session. Only the coder path narrows and retries.
- **Every long run starts fresh**: no `message_history=` is ever passed to `agent.run()`, so compaction operates within a single run; cross-run memory flows through prompts built from `ConversationHistory`.
- **Provider-reported usage is available but unused for calibration.** Per the pydantic-ai-harness compaction documentation (https://pydantic.dev/docs/ai/harness/compaction/), token estimates anchor on provider-reported usage of the most recent response when one exists; `ReportContextUsage` (already wired via `build_usage_monitor()`, `compaction.py:280`) reports against the *configured* window and cannot contradict it.
- **Tests**: backend suite at `backend/tests/`; relevant files include `test_execution_loop.py` (overflow/narrowing behaviour, including existing tests pinning the halve-and-retry semantics), `test_compaction.py` (ladder + `resolve_context_window`/`narrowed_window`), `test_context_budget.py`, `test_model_resolution.py` (patterns for monkeypatching `OLLAMA_BASE_URL`), `test_embedding.py` (`MockTransport` pattern for faking `/api/show`). Tests run against a real Postgres when `TEST_DATABASE_URL` is set; suites needing external services skip otherwise.
- Both `backend/AGENTS.md` (mandatory) exist; see Global Rules. The previous plan document (eight completed phases covering compaction introduction, step execution, etc.) has been replaced by this document at the user's direction; its findings survive in the Background sections above.

## Goal / Desired Behavior

1. Before the first request of a session, Loom knows — or soundly falls back — how large the serving model's context window actually is. For Ollama-served models it reads the model's context length from `/api/show`; the effective window used everywhere (compaction budget, usage monitor denominator, Ollama `num_ctx`) is the smaller of the user's configured number and that measurement.
2. When measurement is impossible (non-Ollama model name, unreachable server, unusable metadata), behavior is exactly today's — configured/default window plus halve-and-retry on failure — with no new warnings.
3. An overflow failure recovers identically on all three long-run paths (coder streaming, structured planning, text planning): narrow, retry the same call, stop at the existing complaint floor with the existing explanatory message. Hard provider rejections (HTTP 400-style context-length errors) are recognized as the same recoverable overflow, not as fatal crashes.
4. Divergence between what Loom estimated it sent and what the provider reports processing is detected during normal successful running: silently-truncated prompts (provider saw far less than sent) and unseen token growth (provider billed far more than estimated) each warn once per session with concrete numbers, and feed the working estimate so compaction targets reflect reality sooner.
5. No schema changes; no frontend changes; reviewer/classifier agents untouched.

## Scope
**In scope:**
- Backend: `services/model_resolution.py`, `services/compaction.py`, `services/context_budget.py` (as needed), `services/embedding.py` (only if sharing/refactoring the `/api/show` parsing), a possible new small service module for model-metadata probing, and `api/v1/agent_session.py`.
- Backend tests under `backend/tests/` for everything above.

**Out of scope:**
- Persisting the discovered/effective window to the `Task` record or anywhere else (user-declined; implies no migration).
- Frontend changes of any kind.
- Reviewer/classifier agents (single-shot structured calls, not long ReAct loops).
- Registry-based window resolution for cloud providers — the explicit `context_window=` override stays exactly as it is; this plan adds a local-measurement source beneath the same override.
- Changing the compaction ladder's tiers, ordering, or tuning constants beyond what feeding measured values into them requires.
- Supporting non-Ollama local servers' metadata APIs (vLLM/LM Studio/llama.cpp probing) — noted as future work only.

## Phases

### Phase 0: Environment Setup — Status: Not Started
- Python 3.14.x required (per `backend/AGENTS.md`). If `.venv` does not exist: `cd backend && python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`.
- Confirm `git` availability (repo already verified as a git repository).
- Run the baseline test suite per Global Rules and record results. Note: suites requiring `TEST_DATABASE_URL` (Postgres with pgvector) skip themselves when it is unset — record which mode applied, since later phases must compare like against like. `ast-grep` must be on `PATH` (or `AST_GREP_BIN` set) for indexing tests to run rather than skip.
- Run `mypy . && basedpyright .` once to record the linting baseline.

### Phase 1: Measure the model's real context window from Ollama — Status: Not Started
- **Goal:** Budget against `min(configured-or-default, measured)` instead of the configured number alone, and set Ollama's `num_ctx` to that same effective value.
- **Requirements:**
  - [ ] Add a way to ask the serving Ollama instance for a chat model's context length: POST `/api/show` with the resolved model name (the `ollama:` prefix stripped, tags preserved — mirror `resolve_model()`'s splitting rules), honoring `OLLAMA_BASE_URL` and its default. Read `model_info["<architecture>.context_length"]`, tolerating missing/unexpected shapes by yielding None, following the precedent and None-tolerant style of `EmbeddingService.describe_model()`/`_model_info()`. Prefer extracting whatever parsing can be shared with `embedding.py` over duplicating it; judge the cleanest seam (a small shared helper vs a new module) and note the choice.
  - [ ] Probe at most once per session, at session start, for the coder/planner model (they share one setting today). Cache the result for the session's lifetime; do not probe again mid-session even if the model setting were changed.
  - [ ] Compute the effective window: when a measurement exists, use the smaller of `resolve_context_window(configured)` and the measurement; when it does not, use `resolve_context_window(configured)` unchanged. Do NOT raise a sub-minimum measurement up to `MINIMUM_CONTEXT_WINDOW`: a model genuinely reporting less than 2,048 tokens cannot honor the floor, and pretending otherwise recreates the failure loop this plan removes. Let the complaint machinery speak instead (next item).
  - [ ] Feed the effective window into every consumer that currently takes the configured value: `build_compaction()`, `build_usage_monitor()`, `model_settings()` (so `num_ctx` asks Ollama to allocate exactly what the budget assumes — never more than the model supports), and `_warn_if_context_is_too_small()`'s check. All three long-run paths (`_run_coder_streaming`, `_plan_structured`, `_plan_as_text`) must use it.
  - [ ] When the effective window fails `context_window_complaint()` (window below overhead + headroom), emit that complaint once via the existing `_warn_if_context_is_too_small()` path and proceed — the run will fail informatively rather than loop. This replaces any temptation to keep halving below reality.
  - [ ] Fallback is silent and total: non-Ollama names, connection errors, non-200s, unparsable bodies, missing keys → behave byte-for-byte as today (configured/default window), emitting nothing new.
- **Edge Cases & Error Handling:**
  - Model name carries a tag (`ollama:qwen2.5-coder:14b`) — the full tagged name goes to `/api/show`, matching how the model was pulled.
  - Ollama reports a context length larger than configured — configured wins (never *raise* the user's ceiling).
  - Measurement succeeds but the model also has a Modelfile-level `num_ctx` override smaller than the trained maximum — `/api/show`'s top-level parameters may report it; prefer the tighter of what the metadata offers if straightforwardly available, else fall back to the architecture context length. If this proves murky, use the architecture length and say so in the phase summary.
  - Probe latency/failure must never delay or block session startup beyond a short timeout comparable to `EmbeddingService`'s.
  - `TestModel`/unit contexts: the probe is injectable/mockable (transport injection like `test_embedding.py`) so tests never need a live Ollama.
- **Affected Areas:** `services/model_resolution.py` or a sibling module (probe + effective-resolution); `services/embedding.py` (optional shared parsing extraction); `api/v1/agent_session.py` (session-start probe, effective-window plumbing into all three run paths and the warning); `tests/test_model_resolution.py` or a new test module; possibly `tests/test_agent_session.py`/`test_execution_loop.py` for plumbing assertions.
- **Acceptance Criteria:**
  - [ ] With a mocked `/api/show` reporting e.g. 8,192 tokens and a configured 128,048, the compaction capability, usage monitor, and `num_ctx` payload all carry 8,192. — verified by: unit tests with transport-injected responses.
  - [ ] With configured 8,192 and a measured 131,072, everything still uses 8,192 (measurement never raises the ceiling). — verified by: unit test.
  - [ ] Non-Ollama model name → no HTTP request is issued and behavior matches current code paths exactly. — verified by: unit test asserting no client call occurs.
  - [ ] Unreachable/malformed `/api/show` → silent fallback to configured/default; no new events emitted. — verified by: unit test with failing transport.
  - [ ] A measured window too small for overhead + headroom produces the existing human-facing complaint once at session start. — verified by: unit test asserting the warning event content.
  - [ ] Existing suite green apart from intentional updates to tests that pinned old plumbing. — verified by: pytest run.
- **Testing Notes:** Fake `/api/show` with `MockTransport` following `tests/test_embedding.py`'s handler pattern. Cover: happy path, tag-bearing names, missing `model_info`, wrong-typed values, non-200, connect error. Assert on the numbers reaching `build_compaction`/`model_settings` via the recording patterns already used in `test_execution_loop.py` (budget-recording stand-ins).

### Phase 2: One recoverable-overflow path shared by every long run — Status: Not Started
- **Goal:** Narrow-and-retry works identically for coder streaming, structured planning, and text planning — and additionally recognizes hard provider rejections, not just empty generations.
- **Requirements:**
  - [ ] Extract the retry loop currently inside `_run_coder_streaming()` (catch → `looks_like_no_room_to_answer` → `_narrow_after_overflow` → flush half-streamed thoughts → retry) into a reusable mechanism on the session, parameterized by which window field owns the attempt, and route all three long-run paths through it.
  - [ ] Preserve existing coder-path semantics exactly: same exception type recognized, same warning text, same narrowing arithmetic (`narrowed_window`), same complaint-floor stop, same thought-buffer flush behavior. Existing tests in `test_execution_loop.py` pinning these must pass unchanged wherever they describe behavior rather than structure.
  - [ ] Broaden recognition: a provider rejection of the request for exceeding context (surfacing through pydantic-ai as an API/model error carrying an HTTP 400-style status and context-length wording) must be classified alongside the empty-generation failure as recoverable overflow. Extend `looks_like_no_room_to_answer()` or add a sibling predicate in `services/compaction.py`; keep the naming honest about what it recognizes. Non-overflow exceptions propagate exactly as today — including `StopRequested`/client-disconnect conditions, which must never be swallowed or retried.
  - [ ] Structured planning: an overflow during the structured attempt narrows and retries the *structured* call itself; the existing fall-back-to-text-on-failure behavior remains only for non-overflow failures. Text planning (`_plan_as_text`) gains the same narrow-retry instead of failing the session.
  - [ ] Narrowing keeps today's coupling: narrowing after a coder overflow also lowers the planner's window (and vice versa) since both share one model — preserve the current `min(narrower, planner)` relationship unless implementation shows a cleaner symmetric rule, in which case propose it in the summary rather than switching silently.
- **Edge Cases & Error Handling:**
  - Overflow on the very first attempt of a fresh session whose window was just measured (Phase 1) — narrowing still applies; measurement reduces but cannot eliminate optimism (e.g., Modelfile `num_ctx` below trained max).
  - Repeated overflow until the complaint floor — the run ends with the existing explanatory complaint, not a crash-loop; verify the count of attempts matches today's behavior (halving sequence bounded by `MINIMUM_CONTEXT_WINDOW`).
  - A 400 rejection whose message does not mention context/length must remain unrecognized (avoid over-matching unrelated provider errors).
- **Affected Areas:** `api/v1/agent_session.py` (extraction + three call sites), `services/compaction.py` (predicate broadening), `tests/test_execution_loop.py` (new cases for planner paths + 400 recognition), `tests/test_compaction.py` (predicate unit tests).
- **Acceptance Criteria:**
  - [ ] Coder overflow still narrows and retries with identical warnings/arithmetic (existing pinned tests pass unmodified). — verified by: pytest run.
  - [ ] Simulated 400 context rejection triggers narrow-and-retry on the coder path. — verified by: new test with a raised provider-error stub.
  - [ ] Structured-planning overflow retries structured (does not degrade to text); non-overflow structured failure still falls back to text. — verified by: two new tests distinguishing the paths.
  - [ ] Text-planning overflow narrows and retries instead of failing the session. — verified by: new test.
  - [ ] Stop/disconnect conditions are never retried. — verified by: test asserting propagation.
- **Testing Notes:** Stub exceptions mirroring both failure shapes (empty-generation `UnexpectedModelBehavior`; provider error with status + message). Reuse the session fixtures from `test_execution_loop.py`.

### Phase 3: Reconcile Loom's estimate against provider-reported usage — Status: Not Started
- **Goal:** Use the provider's own accounting, available on every successful response, to catch the two failure precursors a success can legitimately reveal: silent input truncation (provider processed far less than Loom sent — small models clamping `num_ctx`) and hidden token growth (provider billed far more than Loom's heuristic sees — content invisible to the char-based estimator).
- **Requirements:**
  - [ ] After each successful model response on a long run, compare the provider-reported prompt/input token count against Loom's own estimate of what that request carried (fixed overhead from `estimate_fixed_overhead()` plus the estimated history at send time — reuse whatever accounting the compaction/monitor path already computes rather than recounting).
  - [ ] Classify divergence in both directions with named constants for the thresholds (chosen sensibly, e.g., provider-under ≲ half of estimate signals likely truncation; provider-over ≳ 1.5× signals unseen content — exact values are the agent's to tune and document):
    - Provider processed substantially *less* than sent → warn once per session that the model/server appears to be truncating input, naming the observed vs expected figures and pointing at `num_ctx`.
    - Provider processed substantially *more* than estimated → warn once per session that Loom's estimate understates reality, and feed the correction back so subsequent budgets account for it (e.g., adopt the larger observed figure into the working overhead/history accounting for the remainder of the session — the mechanism is the agent's choice; the requirement is that later requests budget against the corrected view).
  - [ ] Each direction warns at most once per session; subsequent divergences update the correction silently without new events, so the feed is never spammed.
  - [ ] Nothing here may change what is sent to the model mid-run other than the budget correction above; no history edits, no narrowing triggered by divergence alone (narrowing remains failure-driven from Phase 2).
  - [ ] The existing `ContextUsageEvent` contract and header gauge stay untouched.
- **Edge Cases & Error Handling:**
  - Runs shorter than two responses produce at most one comparison point — acceptable; the feature targets long ReAct loops.
  - Providers reporting zero/absent usage → skip reconciliation silently for that response.
  - Divergence caused by a Phase 2 retry boundary (fresh attempt after narrowing) must not double-warn; the once-per-session-per-direction rule covers it.
- **Affected Areas:** `api/v1/agent_session.py` (per-response hook on the streaming path and both planner paths), `services/context_budget.py` or `services/compaction.py` (threshold constants + correction helper), tests in `test_execution_loop.py` or a focused new module.
- **Acceptance Criteria:**
  - [ ] A synthetic run whose provider reports roughly half the estimated prompt tokens emits the truncation warning once with concrete numbers. — verified by: unit test with scripted usage figures.
  - [ ] A synthetic run whose provider reports ~1.5×+ the estimate emits the underestimate warning once and subsequent requests budget with the correction. — verified by: unit test asserting the adjusted budget reaches `build_compaction`.
  - [ ] Agreement within thresholds emits nothing new. — verified by: unit test.
  - [ ] Repeat divergences in the same session/direction add no further warnings. — verified by: unit test.
  - [ ] Existing suite green; `ContextUsageEvent` shape untouched. — verified by: pytest run + schema inspection.
- **Testing Notes:** Drive the comparison logic directly against synthetic (estimated, reported) pairs; wire-level coverage only needs one integration point asserted end-to-end.

## Definition of Done
- Every phase's Status is Done — all Requirements and Acceptance Criteria checkboxes checked, each with a verification note.
- The backend test suite passes with no regressions relative to the Phase 0 baseline; `mypy . && basedpyright .` clean.
- End-to-end behavior holds: with an Ollama model whose real context is smaller than configured, sessions budget and set `num_ctx` against the measured value from the first request; an overflow on any long-run path recovers by narrowing with one clear warning; hard provider context rejections recover the same way; estimate-vs-provider divergence surfaces once per direction with actionable numbers.
- Final one-shot summary written covering all phases, plus recommendations for anything noticed but left out of scope (candidates already known: persisting the discovered window, vLLM/LM Studio metadata probing, distinct planner model).

## Open Questions
- **Exact divergence thresholds and correction mechanics (Phase 3):** intentionally left to the implementing agent's judgment with the recommended starting points stated in the requirements; document the chosen values in the phase summary rather than asking mid-flight, unless real usage data suggests the defaults are badly wrong.
- **Shared `/api/show` parsing seam (Phase 1):** extract-from-`embedding.py` vs new standalone module — agent's judgment; whichever keeps `EmbeddingService` behavior byte-identical (its own tests must pass unmodified).
- **Modelfile `num_ctx` preference (Phase 1):** if reading the tighter of parameters-vs-architecture proves unreliable across Ollama versions, fall back to the architecture context length alone and note it in the summary — this is latitude, not a blocker.

## Non-Goals / Explicit Exclusions
- No persistence of the discovered or effective window anywhere (no columns, no settings writes) — explicitly declined by the user; revisit as future work if wanted.
- No frontend work: the settings UI, context gauge, and event contracts stay as they are.
- No changes to reviewer/classifier agents or to the compaction ladder's composition.
- No support for non-Ollama local servers' metadata endpoints in this pass.
