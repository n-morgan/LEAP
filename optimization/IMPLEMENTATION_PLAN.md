# LEAP Optimizer Implementation Plan

Companion to `OPTIMIZER_REDESIGN_PLAN.md`. The redesign plan says *what* to build and *why*; this document says *in what order* to build it, *how to verify each step before moving on*, and *how to keep the old loop running* until the new one demonstrably wins.

## Guiding principles

**Ship in slices that leave `main` green.** Every PR either leaves the legacy path as the default or passes a behind-the-flag smoke test. You should be able to stop after any PR and still have a working optimizer.

**Test at the boundary of every new abstraction.** New types get round-trip tests. New pure functions get hand-computed fixtures. New I/O paths get one real smoke run. No PR merges without at least the minimal test listed in its "Verify" section.

**Feature-flag the cutover, don't big-bang it.** A single flag `use_new_evaluator: bool = False` controls which code path `LEAPEvaluator.evaluate` takes. Flip it once the comparison run in PR 11 shows the new loop wins.

**Attribute failures, don't average them.** Every new metric is logged separately so that when something regresses, you know which dimension broke.

---

## Prerequisites (before PR 1)

Complete these once, before touching the code:

1. **Pin a baseline.** Run the current `prompt_optimizer.py` CLI on Seattle for 3 iterations with the existing logging and save the resulting `iteration_log.csv`, `candidate_log.csv`, `extracted_policies.csv`, and final optimized prompt to `optimization/logs/BASELINE_YYYY-MM-DD/`. This is the "legacy" reference all future comparisons point at.
2. **Fix the model strings.** `gpt-5.4` and `gpt-5.2` are not real. Pick the model your org has approved (e.g. `gpt-4.1-2025-04-14` or `gpt-4o-2024-11-20`), set it as a single module-level constant `MODEL` in both `evaluator.py` and `prompt_optimizer.py`, and re-run the baseline. If the baseline scores move significantly after this change, the "pinned baseline" is whatever the real model produces — note it.
3. **Choose the location split.** Pick at least one additional dev city and one test city beyond Seattle. List them in a `locations.yaml` at `optimization/locations.yaml` with fields `name`, `ground_truth_csv`, `source_document_md`, `split` (`dev` or `test`). If only Seattle is available in GENIUS for now, proceed with Seattle-only dev and add the multi-location expansion at PR 8 instead of PR 1.
4. **Confirm dependencies.** `pip install` check: `openai`, `scipy`, `numpy`, `pydantic>=2`, `python-dotenv`, `pytest`. No new dependencies are introduced anywhere in this plan.
5. **Create a feature branch.** `git checkout -b optimizer-redesign` off of whatever `main` currently is. All PRs below merge into this branch; the branch merges to `main` at cutover (after PR 11).

---

## PR-by-PR breakdown

Each PR has the same shape: **What**, **Files**, **Verify**, **Risk**. Verify is the minimum evidence to merge. Risk is what could go wrong and how to mitigate.

### PR 1 — New types and scaffolding

**What.** Introduce the new data model without changing any behavior. `LEAPEvaluator.evaluate` still returns the legacy `EvaluationOutput`. The new types are imported nowhere in production code yet, only in tests.

**Files.**
- New: `optimization/metrics.py` — all Pydantic models from Step 1 of the coding plan (`MatchedPair`, `MatchingResult`, `ExtractionBundle`, `HierarchyBundle`, `ClassificationBundle`, `QualityBundle`, `EvaluationBundle`).
- New: `optimization/tests/test_metrics_types.py` — instantiate each model from a hand-built dict, serialize to JSON, deserialize, assert round-trip equality.
- Modified: `optimization/evaluator.py` — add `use_new_evaluator: bool = False` flag on `LEAPEvaluator.__init__`, plumbed but unused.

**Verify.** `pytest optimization/tests/test_metrics_types.py` passes. Legacy CLI still runs end-to-end.

**Risk.** Low. Adds code, removes nothing.

---

### PR 2 — Category-agnostic matching with similarity threshold

**What.** Add `_match_globally` as a new method alongside the existing bucketed matching. Not yet called by `evaluate`. This is a pure algorithmic change that can be unit-tested in isolation.

**Files.**
- Modified: `optimization/evaluator.py` — add `_match_globally(extracted, ground_truth, threshold=0.55) -> MatchingResult`.
- New: `optimization/tests/test_matching.py` — three cases:
  - Five extracted, six GT, hand-designed to produce a known optimal matching; assert on pair identities.
  - Adversarial below-threshold case: one extracted is semantically unrelated to all GT; assert it lands in `unmatched_extracted`.
  - Cross-category misclassification: extracted has `primary_category="Adaptation"` and GT has `primary_category="Mitigation"` but their statements are near-identical; assert they match despite different categories.

**Verify.** All three tests pass.

**Risk.** Threshold τ = 0.55 may be wrong for your actual data. Mitigation: the third test validates the default by using real-ish policy statements from Seattle; if it fails, drop τ to 0.5 and note it in `OPTIMIZER_REDESIGN_PLAN.md` open questions.

---

### PR 3 — Bundle computation (pure functions)

**What.** All four bundle-computation functions plus the composite score, as pure functions in `metrics.py`. No calls into the evaluator. No LLM calls.

**Files.**
- Modified: `optimization/metrics.py` — add `compute_extraction_bundle`, `compute_hierarchy_bundle`, `compute_classification_bundle`, `compute_quality_bundle`, `compute_composite_score`.
- New: `optimization/tests/test_bundles.py` — for each function, one fixture where every input value is set by hand and every output value is computed by hand. Five small tests, no external dependencies.

**Verify.** All bundle tests pass. Hand-computed expected values match function outputs exactly.

**Risk.** Jensen-Shannon divergence implementation mistakes. Mitigation: use `scipy.spatial.distance.jensenshannon` directly, don't hand-roll; test against an obvious case (identical distributions → 0, fully disjoint distributions → 1).

---

### PR 4 — Wire the new evaluator path

**What.** `LEAPEvaluator.evaluate` checks `self.use_new_evaluator` and routes to either the legacy bucketed path (returns `EvaluationOutput`) or the new global-matching + bundle path (returns `EvaluationBundle`). Both paths grade matched pairs; legacy keeps the old grader prompt, new path keeps it too for now (grader context expansion is PR 5).

**Files.**
- Modified: `optimization/evaluator.py` — add `_evaluate_new` that calls `_match_globally`, grades pairs, computes all four bundles, assembles `EvaluationBundle`. The `evaluate` method dispatches based on the flag.
- New: `optimization/tests/test_evaluator_smoke.py` — loads Seattle CSVs, runs both paths on the same inputs, asserts that the overlap between legacy matched pairs and new matched pairs is at least 80%. (They won't overlap perfectly because the new path uses global matching, but they should mostly agree.)

**Verify.** Smoke test passes. Manual inspection of the new `EvaluationBundle` on Seattle shows sensible numbers (F1 in a plausible range, role agreement in a plausible range, confusion matrix populated).

**Risk.** Grading every pair via LLM is expensive on a smoke test. Mitigation: the smoke test uses a cached fixture or a subset of 10 policies per side; the full Seattle grading happens later in PR 11.

---

### PR 5 — Expanded grader context

**What.** Grader prompts now include the extracted policy's role, primary_category, and source_quote, plus the GT's role and primary_category. `_GraderOutput` gains `statement_match`, `role_match`, `category_match` fields. These new fields flow into the hierarchy and classification bundles as redundant confirming signals (not primary — the bundles compute role/category agreement independently from the extracted/GT fields).

**Files.**
- Modified: `optimization/evaluator.py` — update `_GRADER_SYSTEM`, `_GRADER_USER`, `_RLM_GRADER_SYSTEM`, `_RLM_GRADER_USER`, extend `_GraderOutput`.
- Modified: `optimization/tests/test_evaluator_smoke.py` — assert new fields are populated on graded pairs.

**Verify.** Smoke test passes with new fields. Spot-check 3 graded pairs manually and confirm the grader's `role_match` / `category_match` verdicts agree with direct comparison of the role/category fields.

**Risk.** The grader may now produce malformed JSON more often because the schema grew. Mitigation: Pydantic-enforced `beta.chat.completions.parse` path is already strict; tighten the RLM path's JSON parser as part of PR 10's housekeeping.

---

### PR 6 — Per-prong triggers and targeted feedback

**What.** Replace `update()` and the trigger logic inside `run_loop` with `_triggered_prongs` and per-prong feedback assembly. `propose_candidate` takes an `EvaluationBundle` and pulls failure examples scoped to the target prong.

**Files.**
- Modified: `optimization/prompt_optimizer.py` — add `PrognTarget` dataclass, `_triggered_prongs(current, previous, targets)`, `_assemble_prong_feedback(bundle, scope)`, rewrite `propose_candidate(scope, current_prompt, previous_prompt, current_bundle, previous_bundle) -> StructuredPrompt`.
- New: `optimization/tests/test_triggers.py` — eight cases covering the product of {extraction, hierarchy, classification} × {worsened, stagnant-bad, stagnant-good, improved}.

**Verify.** Trigger tests pass. `propose_candidate` called once for each prong on a Seattle bundle produces three different non-empty prompt sections (smoke).

**Risk.** The LLM resampler may respond poorly to the new per-prong feedback format. Mitigation: feedback format is validated by eyeball on the first real run in PR 7; revise the feedback templates if the resampler ignores signal.

---

### PR 7 — Sequential within-iteration acceptance

**What.** Rewrite `run_loop` so each iteration walks triggered prongs in priority order and threads the running prompt/bundle through. `accept_candidate` becomes multi-criterion (composite delta + per-metric floors). Single-location, single-seed, still using the legacy `EvaluationBundle → EvaluationOutput` adapter wherever needed to keep existing logs working.

**Files.**
- Modified: `optimization/prompt_optimizer.py` — rewrite `run_loop`, rewrite `accept_candidate(cand_bundle, curr_bundle, targets) -> tuple[bool, str]`.
- Modified: `optimization/tests/test_loop.py` (new) — mock `extracted_policies_fn` to return a fixed list, mock the grader to return fixed grades, run 3 iterations, assert that logs show sequential acceptance within an iteration.

**Verify.** 3-iteration run on Seattle completes without crashing. Log inspection shows at least one iteration where more than one prong was accepted.

**Risk.** Sequential acceptance can be slower per iteration (more candidates evaluated) if many prongs trigger. Mitigation: cap the number of accepted prongs per iteration at 2 via config; revisit after the comparison run.

---

### PR 8 — Multi-location evaluation with variance control

**What.** Add `LocationSet`, `evaluate_on_set(candidate_prompt, location_set.dev, seeds=2)`, and dev-aggregate computation. Loop acceptance decisions now read the dev-aggregate bundle, not a single-location bundle. End-of-loop runs once on the test set.

**Files.**
- New: `optimization/dev_test_split.py` — `LocationConfig`, `LocationSet`, `load_locations_from_yaml(path)`.
- Modified: `optimization/prompt_optimizer.py` — loop reads `location_set` rather than a single location; `evaluate_candidate` becomes `evaluate_candidate_on_set`; final test evaluation logged separately.
- Modified: CLI — consumes `locations.yaml`.

**Verify.** End-to-end run on a two-location dev set (Seattle + one other) completes. Test-set evaluation block runs once at loop end and emits `test_results.json`.

**Risk.** Running the RLM on two+ locations per iteration is the dominant cost now. Mitigation: PR 10 adds async grading, which is where the speed comes back. Until then, keep `max_iterations=2` on the test runs.

---

### PR 9 — New logging schema

**What.** `iteration_log.csv`, `candidate_log.csv`, and a new per-iteration `metrics_bundle.json` reflecting the full bundle. Legacy logs are dropped (not maintained in parallel — the baseline snapshot in Prerequisites is the only legacy record we keep).

**Files.**
- Modified: `optimization/prompt_optimizer.py` — updated CSV DictWriter fieldnames, new per-iteration bundle JSON dump.
- Modified: README or inline comment block describing the new log schema.

**Verify.** Smoke run produces all three logs. Manual inspection confirms all metrics are populated and the bundle JSON deserializes correctly.

**Risk.** Downstream tooling (if any) that reads the old CSV schema will break. Mitigation: check — is there any such tooling? If yes, keep the legacy columns as NaN stubs for one release.

---

### PR 10 — Housekeeping

**What.** Three unrelated reliability/speed improvements bundled together because none is big enough for its own PR.

1. Async concurrent grading with `asyncio.gather` + a `Semaphore(8)` around the grader call. The grading path becomes `async def _grade_pairs_async(pairs) -> list[_GraderOutput]`. The outer `evaluate` stays sync via `asyncio.run`.
2. Robust JSON parsing in `_grade_pair_rlm`: try `json.loads` on the full response first, fall back to locating the outermost `{...}` slice and retrying, fall back to one retry with "return only valid JSON" nudge, then fall back to grade 0 with reasoning "grader parse failure."
3. Remove `time.sleep(5)` in `_grade_pair`.

**Files.**
- Modified: `optimization/evaluator.py`.
- New: `optimization/tests/test_grader_parsing.py` — three adversarial inputs (clean JSON, JSON with preamble, JSON with braces in reasoning).

**Verify.** Wall time on a 50-pair evaluation drops by at least 5×. All parsing tests pass.

**Risk.** Concurrent API calls may trip rate limits. Mitigation: `Semaphore(8)` is conservative; if 429s appear, drop to 4.

---

### PR 11 — Comparison harness and cutover decision

**What.** A standalone script that runs both the legacy and new loops from the same baseline prompt on the same dev set for the same number of iterations, then produces a comparison report. This is the go/no-go gate for flipping `use_new_evaluator` to `True` as the default.

**Files.**
- New: `optimization/compare_legacy_vs_new.py` — runs both, writes `comparison_report.md` with composite score trajectories, per-bundle metric trajectories, number of accepted candidates per iteration, dev-vs-test gap, and wall time.

**Verify.** Report generated. New loop either wins the comparison (cutover decision → yes) or loses (cutover decision → no, diagnose and iterate).

**Risk.** The new loop may not win on the first comparison. Mitigation: the redesign is structurally sound; likely causes of a losing comparison are hyperparameter miscalibration (τ too high/low, composite weights, target thresholds). Tune those before declaring failure.

---

## Cutover plan

After PR 11 passes:

1. Flip `use_new_evaluator=True` as the default in `LEAPEvaluator.__init__`.
2. Run one full 10-iteration optimization on the dev set; snapshot the final prompt, metrics, and test-set scores.
3. Delete the legacy bucketed path, the legacy `EvaluationOutput` model, the legacy grader templates that referenced `"Not provided."`, and the legacy `update()` method. This is a separate cleanup PR (PR 12) after a week of running the new path in production.
4. Update `README.md` in `optimization/` to describe the new evaluator contract.

## Rollback plan

Any PR's behavior can be reverted by flipping `use_new_evaluator=False` (applies to PRs 2–10) or reverting the branch merge (PR 11–12). Until the legacy path is deleted in PR 12, rollback is always one boolean flag away.

## Definition of done

All of the following are true:
- New loop's composite score on held-out test location is at least as good as legacy loop's mean score (when normalized to the same scale) on the same test location.
- All metric bundles populate correctly on at least one non-Seattle dev location.
- Wall time per iteration is no worse than the legacy loop's wall time per iteration, and typically better after PR 10.
- `pytest optimization/tests/` is green.
- The legacy bucketed path has been deleted and the feature flag removed.

## Effort estimate

Rough calendar estimates assuming one engineer working focused half-days, for planning purposes only:

| PR | Est. work | Est. elapsed |
|---|---|---|
| 1 — Types | 0.5 day | 0.5 day |
| 2 — Global matching | 1 day | 1 day |
| 3 — Bundle funcs | 1 day | 1 day |
| 4 — Wire evaluator | 1 day | 1 day |
| 5 — Grader context | 0.5 day | 0.5 day |
| 6 — Triggers & feedback | 1 day | 1 day |
| 7 — Sequential loop | 1.5 days | 2 days |
| 8 — Multi-location | 1 day | 2 days (data prep) |
| 9 — Logging | 0.5 day | 0.5 day |
| 10 — Housekeeping | 1 day | 1 day |
| 11 — Comparison | 1 day | 1 day + eval run time |
| **Total** | **~10 days** | **~12 days** |

PR 8 has a real elapsed-time dependency on whether the second dev city and test city GENIUS data are ready. If not, flag that in Prerequisites so the blocker is known up front.

## Open execution questions

- Do we have a second city's GENIUS ground truth ready, or is that blocker for PR 8? If blocker: do PRs 1–7 and 9–10 first, then PR 8 and 11 once data is ready.
- Is there any external consumer of `iteration_log.csv` that needs legacy columns preserved through PR 9?
- What's the API rate-limit ceiling we should target with the `Semaphore` in PR 10?
- Who owns the cutover decision at PR 11, and what's the threshold for "new loop wins" — strict improvement on composite, or improvement on every prong's headline?
