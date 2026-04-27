# LEAP Optimizer Redesign Plan

## Motivation

"Output similar to the golden output" is a three-dimensional claim. The extracted *set* of policies should match the GT set (membership, proportions), the extracted *hierarchy* should match the GT hierarchy (role assignments, parent–child structure), and the extracted *labels* should match the GT labels (primary category, financial instrument, secondary category). Each of these maps cleanly to one prong of the RLM system prompt.

The current algorithm collapses all three into a single scalar score plus a handful of aggregate signals, which couples the prongs together and makes the optimizer oscillate. The redesign orthogonalizes the signals so each prong has a clean, monotone gradient to follow.

---

## Part 1 — Design

### Failure modes in the current architecture

**Bucketing-before-matching couples classification to extraction.** Policies are grouped by `primary_category` before Hungarian matching. A correctly-extracted but misclassified policy never meets its GT counterpart — it counts as unmatched in the wrong category (hurts FPR) and its GT counts as unmatched in the right category (hurts recall). One failure produces two symptoms in two different prongs' signals.

**Unbounded Hungarian matching produces phantom matches.** `linear_sum_assignment` returns the optimal 1:1 pairing regardless of similarity magnitude, so semantically unrelated pairs get matched and graded. This wastes grader calls and inflates the "matched" count that drives recall.

**The classification prong is driven by `mean_score`.** Mean score drops when extraction misses or over-extracts, so the classification prong is frequently rewritten in response to problems it cannot fix.

**Triggers fire only on worsening signals.** A prong stuck at a bad but stable value is never rewritten after iteration 1.

**Acceptance has no minimum improvement delta.** A candidate with identical scores is accepted, letting the prompt drift without forward progress.

**Grader sees only `policy_statement`.** Role, category, and field assignments cannot influence the grade, so the grade signal is blind to two of three prongs' outputs.

**No variance control, no dev/test split.** Single-location optimization plus single-seed evaluation means we cannot distinguish real improvements from noise, and the final prompt overfits to Seattle.

### New architecture overview

The loop retains the outer shape — evaluate, propose per-prong candidate, accept or reject, iterate — but every component is reworked.

**Phase 0 — Setup.** Migrate flat prompt to three prongs (unchanged). Split locations into a dev set (two or three cities) and a held-out test set (one city). All acceptance decisions use dev-set aggregate metrics. The test set is evaluated once, at the end.

**Phase 1 — Category-agnostic matching.** Embed all extracted and all GT policies across all categories. Run Hungarian on the full cross-category similarity matrix with a minimum-similarity threshold τ (starting value 0.55). Pairs below τ are treated as unmatched rather than force-assigned. Output: matched set **M**, unmatched extracted **U_E**, unmatched GT **U_G**.

**Phase 2 — Three orthogonal metric bundles.** Each bundle is computed from M, U_E, U_G. No bundle reads any state the others write. See the Metric Bundles section below.

**Phase 3 — Per-prong triggers and proposals.** Each prong owns one bundle. A prong fires when its own bundle's headline signal worsened vs. the last accepted prompt OR is still below its target threshold. Resamplers receive the failing bundle's decomposition plus concrete failure examples filtered for that prong.

**Phase 4 — Sequential within-iteration acceptance.** Candidates are proposed and evaluated one prong at a time against the running baseline. Each accepted candidate becomes the baseline for the next prong's proposal, within the same iteration. Acceptance is multi-criterion (composite improvement + no individual-metric cliff) and measured across dev locations with a small number of evaluation seeds for variance control.

**Phase 5 — Convergence and test.** Converged when no candidate is accepted for K consecutive iterations, OR all prong thresholds are met, OR max iterations reached. Final test-set evaluation is run once and the dev-vs-test gap is reported as the overfit estimate.

### Metric bundles

| Bundle | Headline signal | Other metrics | Owning prong |
|---|---|---|---|
| Extraction | Micro-F1 on M vs. GT | Micro-recall, micro-precision, per-category recall, Jensen–Shannon divergence of category-distributions, +1-coverage | extraction |
| Hierarchy | Role-agreement rate on M | Parent attribution accuracy on subs in M | hierarchy |
| Classification | Primary-category agreement on M | 4×4 category confusion matrix, financial-instrument agreement, secondary-category agreement | classification |
| Quality (cross-cutting) | +1-coverage `|{m∈M: grade=+1}| / |GT|` | Grade distribution (+1/0/-1 shares), mean grade on M | used in composite, not prong-owned |

**Composite score** for acceptance: weighted sum of Extraction F1, Hierarchy role-agreement, Classification primary-category agreement, and +1-coverage. Initial weights `0.35 / 0.20 / 0.20 / 0.25`, tunable.

### Prongs and signals — explicit mapping

**Extraction prong.** Governs "what counts as a policy, how to detect it, what fields to populate, overall extraction strategy and output format." Owns the Extraction bundle. Its edits move the extracted *set* toward the GT set in membership and proportion.

**Hierarchy prong.** Governs "parent / sub / individual role assignment and parent_statement linking." Owns the Hierarchy bundle. Its edits move the extracted *tree shape* toward the GT tree shape. Because policy hierarchies are depth-1 (`parent → sub`, or `individual`), structural similarity reduces to two metrics: role labels on M, and parent attributions on the subs in M. No tree-distance algorithm needed.

**Classification prong.** Governs "primary_category, financial_instrument, climate_relevance, secondary_category." Owns the Classification bundle. Its edits move the *labels on matched policies* toward GT labels.

### Why this converges

The matching stage is category-agnostic and threshold-gated, so M is approximately stable under classification and hierarchy edits — changing a label or a role on an extracted policy does not change whether it matches its GT counterpart (the policy statement text is what drives the cosine similarity). That means the extraction bundle is approximately invariant to classification and hierarchy edits, and vice versa.

Each prong's signal is therefore moved primarily by its own prong's edits. Three approximately-monotone loops on orthogonal axes converge to a prompt that is good on all three.

Contrast with the current design: fixing the classification prong perturbs bucketing, perturbs matching, perturbs extraction's F1, which triggers extraction to be rewritten, which can undo the classification fix. The signal is not monotone in the edit, which is the mechanical cause of oscillation.

---

## Part 2 — Coding Plan

The steps below are ordered by dependency. Each step is self-contained and independently testable so partial progress is always runnable.

### Step 1 — Types and scaffolding

**Files:** `optimization/evaluator.py`, new `optimization/metrics.py`.

Introduce the new type hierarchy without deleting old code. Keep the old `EvaluationOutput` temporarily as `LegacyEvaluationOutput` so existing callers keep working.

New Pydantic models in `metrics.py`:

```python
class MatchedPair(BaseModel):
    extracted: dict
    ground_truth: dict
    similarity: float
    grade: Literal[-1, 0, 1] | None = None
    reasoning: str | None = None

class MatchingResult(BaseModel):
    matched: list[MatchedPair]
    unmatched_extracted: list[dict]
    unmatched_gt: list[dict]
    similarity_threshold: float

class ExtractionBundle(BaseModel):
    f1: float
    precision: float
    recall: float
    per_category_recall: dict[str, float]
    category_distribution_jsd: float
    plus_one_coverage: float

class HierarchyBundle(BaseModel):
    role_agreement: float                # point-wise: fraction of M with matching role
    parent_attribution_accuracy: float   # for subs in M, fraction whose extracted parent
                                         # resolves to the same matched-pair as the GT parent

class ClassificationBundle(BaseModel):
    primary_category_agreement: float
    confusion_matrix: dict[str, dict[str, int]]  # gt_cat -> ext_cat -> count
    financial_instrument_agreement: float
    secondary_category_agreement: float

class QualityBundle(BaseModel):
    plus_one_coverage: float
    grade_distribution: dict[str, float]  # "+1" / "0" / "-1" -> fraction
    mean_grade: float

class EvaluationBundle(BaseModel):
    location: str
    matching: MatchingResult
    extraction: ExtractionBundle
    hierarchy: HierarchyBundle
    classification: ClassificationBundle
    quality: QualityBundle
    composite_score: float
```

**Validation:** unit test instantiates each bundle from a tiny fixture and round-trips through JSON.

### Step 2 — Category-agnostic matching with threshold

**File:** `optimization/evaluator.py`.

New method on `LEAPEvaluator`:

```python
def _match_globally(
    self,
    extracted: list[dict],
    ground_truth: list[dict],
    threshold: float = 0.55,
) -> MatchingResult:
    ...
```

Embed all extracted and all GT policies in a single call each (batched — existing `_embed` already supports this). Build the full N×M cosine similarity matrix. Run `linear_sum_assignment` on the full matrix. Reject any assigned pair whose similarity is below `threshold` and route both members to the unmatched sets.

**Validation:** construct a fixture with 5 extracted and 6 GT policies where the expected matching is known; assert on M, U_E, U_G. Add a below-threshold adversarial case to confirm the threshold gate works.

### Step 3 — Metric bundle computation

**Files:** `optimization/metrics.py` (pure functions), `optimization/evaluator.py` (wires them into `evaluate`).

Each bundle gets a standalone function that takes `MatchingResult` plus the original GT list and returns the bundle. Keeping them pure makes them easy to test.

```python
def compute_extraction_bundle(m: MatchingResult, gt: list[dict]) -> ExtractionBundle: ...
def compute_hierarchy_bundle(m: MatchingResult) -> HierarchyBundle: ...
def compute_classification_bundle(m: MatchingResult) -> ClassificationBundle: ...
def compute_quality_bundle(m: MatchingResult) -> QualityBundle: ...
def compute_composite_score(e: ExtractionBundle, h: HierarchyBundle,
                            c: ClassificationBundle, q: QualityBundle,
                            weights: dict[str, float]) -> float: ...
```

JSD over category distributions uses the empirical histogram of `primary_category` over extracted vs. GT. Parent attribution accuracy is one loop over subs in M: for each extracted sub, resolve its `parent_statement` to the referenced policy, look up that policy's matched-pair counterpart, and check whether that counterpart is the same as the GT sub's GT parent. Fraction correct over eligible subs. Subs whose GT counterpart is not itself a sub (or whose GT parent is not in M) are excluded from the denominator so that extraction failures don't leak into the hierarchy signal.

Update `LEAPEvaluator.evaluate` to call `_match_globally`, grade each matched pair, run the four bundle functions, and return an `EvaluationBundle`. Keep the old bucketed path behind a feature flag `legacy_bucketing: bool = False` for A/B comparison during rollout.

**Validation:** unit test each bundle function on a fixture with hand-computed expected values. Integration test: run the new `evaluate` on the existing Seattle CSVs and compare against the legacy path's matched-pair set (they should heavily overlap).

### Step 4 — Grader sees the full extraction

**File:** `optimization/evaluator.py`.

Extend `_GraderOutput` to carry a per-field assessment:

```python
class _GraderOutput(BaseModel):
    grade: Literal[-1, 0, 1]
    reasoning: str
    statement_match: Literal["match", "partial", "mismatch"]
    role_match: bool
    category_match: bool
```

Update `_GRADER_USER` and `_RLM_GRADER_USER` templates to include the extracted policy's role, primary_category, and source_quote, plus the GT's role and primary_category. These per-field signals feed into richer failure-example assembly for resamplers in Step 6.

**Validation:** sanity-check on a handful of pairs that the grader emits the extra fields and that they agree with manual inspection.

### Step 5 — Per-prong trigger and target-threshold logic

**File:** `optimization/prompt_optimizer.py`.

Replace the single-signal triggers with a per-prong check:

```python
@dataclass
class PrognTarget:
    extraction_f1: float = 0.70
    hierarchy_role_agreement: float = 0.85
    classification_primary_agreement: float = 0.85

def _triggered_prongs(
    current: EvaluationBundle,
    previous: EvaluationBundle | None,
    targets: PrognTarget,
) -> set[str]:
    triggered = set()
    if previous is None:
        return {"extraction", "hierarchy", "classification"}
    if current.extraction.f1 < previous.extraction.f1 or current.extraction.f1 < targets.extraction_f1:
        triggered.add("extraction")
    if current.hierarchy.role_agreement < previous.hierarchy.role_agreement or current.hierarchy.role_agreement < targets.hierarchy_role_agreement:
        triggered.add("hierarchy")
    if current.classification.primary_category_agreement < previous.classification.primary_category_agreement or current.classification.primary_category_agreement < targets.classification_primary_agreement:
        triggered.add("classification")
    return triggered
```

`propose_candidate` is updated to accept an `EvaluationBundle` and pull prong-specific failure examples from it: worst unmatched GT policies for extraction, worst role confusions for hierarchy, worst category confusions for classification. Feedback assembly becomes a small per-prong function rather than one dump of all grade reasoning.

**Validation:** unit test `_triggered_prongs` over a table of scenarios (worsened, stagnant-bad, stagnant-good, improved).

### Step 6 — Sequential within-iteration acceptance

**File:** `optimization/prompt_optimizer.py`.

Rewrite `run_loop` so each iteration walks triggered prongs in priority order (`extraction` → `hierarchy` → `classification`) and updates the running baseline as it goes:

```python
for t in range(max_iterations):
    baseline_eval = evaluate(current_prompt)          # one eval per iteration start
    triggered = _triggered_prongs(baseline_eval, previous_eval, targets)
    running_prompt = current_prompt
    running_eval = baseline_eval
    for scope in ordered_triggered(triggered):
        candidate = propose_candidate(scope, running_prompt, running_eval, ...)
        cand_eval = evaluate(candidate)               # multi-seed, dev-set aggregate
        ok, reason = accept_candidate(cand_eval, running_eval, targets)
        log_candidate(t, scope, ok, reason, cand_eval)
        if ok:
            running_prompt = candidate
            running_eval = cand_eval
    previous_eval = baseline_eval
    current_prompt = running_prompt
```

`accept_candidate` becomes multi-criterion:

```python
def accept_candidate(cand: EvaluationBundle, curr: EvaluationBundle,
                     min_delta: float = 0.02,
                     per_metric_floor: dict[str, float] = ...) -> tuple[bool, str]:
    if cand.composite_score < curr.composite_score + min_delta:
        return False, f"insufficient improvement: {cand.composite_score - curr.composite_score:+.3f}"
    for metric, floor in per_metric_floor.items():
        drop = get_metric(curr, metric) - get_metric(cand, metric)
        if drop >= floor:
            return False, f"{metric} cliff: {drop:.3f}"
    return True, f"accepted: composite {curr.composite_score:+.3f} -> {cand.composite_score:+.3f}"
```

**Validation:** run the optimizer for 3 iterations on Seattle with the new loop and confirm logs show within-iteration acceptance compounding (a single iteration accepting two prong rewrites against the running baseline).

### Step 7 — Variance control and dev/test split

**File:** `optimization/prompt_optimizer.py`, new `optimization/dev_test_split.py`.

Introduce a `LocationSet` abstraction:

```python
@dataclass
class LocationSet:
    dev: list[LocationConfig]   # each carries ground_truth_path, source_document_path
    test: list[LocationConfig]
```

`evaluate_candidate` becomes `evaluate_candidate_on_set(candidate, location_set.dev, seeds=2)`. It runs each dev location `seeds` times, computes per-location `EvaluationBundle`, and returns a dev-aggregate bundle (mean of each metric across locations and seeds). Variance estimates are logged per metric.

At loop termination, run once on `location_set.test` and log the dev-vs-test gap.

**Validation:** start with a dev set of two cities (Seattle + one other) and one held-out test city. Confirm that dev-aggregate metrics are sensible and test-set metrics are reported only at end.

### Step 8 — Logging schema update

**File:** `optimization/prompt_optimizer.py`.

Update `iteration_log.csv` and `candidate_log.csv` columns:

- `iteration_log.csv`: iteration, location_set_hash, composite_score, extraction_f1, extraction_precision, extraction_recall, extraction_jsd, plus_one_coverage, hierarchy_role_agreement, hierarchy_parent_attribution, classification_primary_agreement, classification_financial_agreement, classification_secondary_agreement, composite_score_std (across seeds), triggered_prongs, composed_prompt.
- `candidate_log.csv`: iteration, scope, accepted, reason, composite_score, composite_delta, per-bundle-headline-metrics, prompt_section_diff.

Add `metrics_bundle.json` per iteration so the full structured bundle is preserved for offline analysis.

**Validation:** smoke-test one run end to end and inspect the new logs.

### Step 9 — Housekeeping

**Files:** `optimization/evaluator.py`, `optimization/prompt_optimizer.py`.

Replace `gpt-5.4` / `gpt-5.2` with actual model strings. Suggest `gpt-4.1-2025-04-14` or whichever your org has approved; expose as a top-level config constant so it's set once.

Remove `time.sleep(5)`. Replace per-pair grading with `asyncio.gather`-backed concurrent grading with a semaphore (suggested concurrency 8). For larger evaluations, plumb the OpenAI Batch API into the grader path — one batch per evaluation call.

Replace the fragile JSON regex in `_grade_pair_rlm` with either a strict `json.loads` on the full response, or pydantic's `model_validate_json` after locating the first `{`-through-last-`}` slice; fall back to a retry with a "return only JSON" nudge.

**Validation:** measure wall-time on a 50-pair evaluation pre/post the async change; expect at least 5× speedup.

### Step 10 — Validation run and comparison

**Files:** `optimization/compare_legacy_vs_new.py`.

Run both the legacy and new loops from the same baseline prompt for the same number of iterations on the Seattle dev set. Report:

- Composite score trajectory for both.
- Per-bundle metric trajectory for the new loop.
- Number of accepted candidates per iteration for both.
- Dev-vs-test gap for the new loop.
- Wall-time per iteration.

This is the go/no-go gate for cutting over. Keep the legacy path behind a flag until the new loop demonstrably wins on this comparison.

---

## Rollout order at a glance

1. Types and scaffolding (metrics.py)
2. Global matching with threshold
3. Bundle computation
4. Expanded grader context
5. Per-prong triggers
6. Sequential acceptance + multi-criterion
7. Dev/test + multi-seed
8. Logging
9. Housekeeping (models, async, parsing)
10. Legacy-vs-new comparison run

Steps 1–3 unlock the core signal improvement. Steps 5–6 unlock the loop improvement. Steps 7 and 10 unlock confidence that the improvement generalizes. Steps 8–9 are quality-of-life but pay for themselves in iteration speed.

## Open questions to resolve before coding

- Target thresholds per prong: are `F1 ≥ 0.70`, `role_agreement ≥ 0.85`, `parent_attribution_accuracy ≥ 0.85`, `primary_category_agreement ≥ 0.85` the right initial values?
- Composite-score weights: `0.35 / 0.20 / 0.20 / 0.25` is a first guess; may need to be tuned after the first comparison run. Within the hierarchy bundle, whether the headline is `role_agreement` alone or a mean of `role_agreement` and `parent_attribution_accuracy` is also a choice — current preference is the mean.
- Number of evaluation seeds: 2 is the minimum for variance estimation but 3 is safer. Budget permitting, 3.
- Similarity threshold τ: start at 0.55 for `text-embedding-3-small`; validate on a handful of known-good and known-bad matched pairs from Seattle and adjust.
- Which cities go in dev vs. test: depends on what GENIUS data you have ready. At minimum, the test city should be from a different region than the dev cities to stress generalization.
