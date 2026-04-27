# optimization/

Prompt optimization loop for the climate-policy RLM pipeline.

## Files

| File | Description |
|---|---|
| `evaluator.py` | LEAP Evaluator — legacy bucketed path **plus** the redesigned global-matching path (gated by `use_new_evaluator`) |
| `llm_evaluator.py` | `LLMEvaluator` — same global matching + `EvaluationBundle` as the new path, but grades with a **single structured LLM call** per pair (no RLM); recommended for fast structured-vs-structured scoring |
| `metrics.py` | Pure-function metric bundles (extraction / hierarchy / classification / quality) and composite-score utilities |
| `dev_test_split.py` | `LocationConfig` / `LocationSet` + `locations.yaml` loader (dev set used for acceptance, test set held out) |
| `prompt_optimizer.py` | LEAP Prompt Optimizer — legacy `run_loop` and the redesigned `run_loop_v2` (sequential within-iteration acceptance, multi-location, multi-criterion) |
| `compare_legacy_vs_new.py` | Go/no-go harness: runs both loops on the same dev set and writes `comparison_report.md` |
| `tests/` | Pytest suite covering metric round-tripping, global matching, bundle math, per-prong triggers, evaluator smoke, end-to-end loop, and grader JSON parsing |

## Redesigned evaluator (new path)

Set `use_new_evaluator=True` on `LEAPEvaluator` to opt in. The evaluator returns
an `EvaluationBundle` carrying four orthogonal sub-bundles plus a composite
score.

**Faster grader (no RLM in evaluation):** use
[`llm_evaluator.LLMEvaluator`](llm_evaluator.py) instead. It returns the same
`EvaluationBundle` and composite metrics, but each pair is graded with one
`chat.completions.parse` call. By default the grader does not use the source
document (structured vs. structured); pass `include_source_document=True` to
inline the full markdown into the grader user prompt. `LEAPEvaluator` is
unchanged for A/B tests and for workflows that need the RLM when a document path
is provided.

Example (`LEAPEvaluator` — if `source_document_path` is set, grading uses the RLM
path and is slow):

```python
from evaluator import LEAPEvaluator

ev = LEAPEvaluator(use_new_evaluator=True, similarity_threshold=0.55)
bundle = ev.evaluate(
    location="Seattle_US",
    extracted_policies=rlm_output,
    ground_truth_policies=genius_output,
    rubric="...",
    source_document_path=...,
)
print(bundle.extraction.f1)
print(bundle.hierarchy.role_agreement)
print(bundle.classification.primary_category_agreement)
print(bundle.quality.plus_one_coverage)
print(bundle.composite_score)
```

Faster, same return type: `from llm_evaluator import LLMEvaluator` and the same
`evaluate(...)` call; omit `include_source_document` (default) for
structured-only grading, or set `include_source_document=True` to inline the doc
with one non-RLM grader call per pair.

Matching is category-agnostic Hungarian on the full N×M cosine similarity
matrix with a similarity threshold τ — pairs below τ are dropped to
`unmatched_extracted` / `unmatched_gt` rather than force-assigned. This
decouples extraction from classification so each prong's signal is
approximately independent of the others' edits.

Grading is concurrent (`asyncio.gather` with a `Semaphore(8)`) and the grader
emits per-field verdicts (`statement_match`, `role_match`, `category_match`)
that flow into the per-prong feedback assembly.

## Redesigned optimizer loop (`run_loop_v2`)

`run_loop_v2` walks triggered prongs in priority order
(`extraction → hierarchy → classification`) and threads the running prompt /
bundle through within a single iteration so multiple prong rewrites can
compound. Acceptance is multi-criterion (composite delta + per-metric floors)
and aggregated across the dev `LocationSet` over `seeds` runs each.

```python
from prompt_optimizer import LEAPPromptOptimizer, PrognTarget, AcceptanceConfig
from dev_test_split import load_locations_from_yaml

location_set = load_locations_from_yaml("locations.yaml")
optimizer = LEAPPromptOptimizer()
optimized = optimizer.run_loop_v2(
    location_set=location_set,
    extracted_policies_fn=my_extract_fn,
    rubric="...",
    initial_prompt=initial_prompt,
    max_iterations=10,
    seeds=2,
    targets=PrognTarget(),
    acceptance=AcceptanceConfig(),
    log_dir="logs/",
)
```

Logs (under a timestamped `v2_*` subdirectory of `log_dir`):

| File | Description |
|---|---|
| `iteration_log.csv` | one row per main evaluation, with composite, std, all bundle headlines, triggered prongs, and the composed prompt |
| `candidate_log.csv` | one row per candidate (accepted or rejected) with scope, reason, composite delta, and per-bundle headlines |
| `metrics_bundle_iter_*.json` | full structured bundle dump per iteration (for offline analysis) |
| `test_results.json` | single test-set evaluation at loop end + dev→test gap |

Run the comparison harness against legacy:

```bash
python3 compare_legacy_vs_new.py --iterations 3 --locations locations.yaml
```

It writes `logs/comparison/comparison_report.md` with composite trajectories,
per-bundle metrics, accepted-candidate counts per iteration, and dev→test gap.

## Legacy evaluator (default, kept until cutover)

---

## evaluator.py (legacy bucketed path)

Implements **Algorithm 1: LEAP Evaluator**. Groups extracted and ground-truth policies by category and role, runs optimal 1:1 Hungarian matching on embedding similarity, grades each matched pair via a grader LLM (+1 / 0 / -1), and penalizes unmatched policies on both sides.

```python
from optimization.evaluator import LEAPEvaluator

evaluator = LEAPEvaluator(model="gpt-5", embedding_model="text-embedding-3-small")

result = evaluator.evaluate(
    location="Seattle_US",
    extracted_policies=rlm_output,       # list of policy dicts from RLM
    ground_truth_policies=genius_output, # list of policy dicts from GENIUS
    rubric="Grade on specificity, commitment, and mechanism...",
    source_document=doc_markdown,        # optional, improves grading context
)

print(result.scores)   # {"Mitigation": 0.4, "Adaptation": -0.2, ...}
print(result.recall)   # {"Mitigation": 0.8, ...}
print(result.fpr)      # {"Mitigation": 0.1, ...}
```

### Output types

**`EvaluationOutput`**
| Field | Type | Description |
|---|---|---|
| `location` | `str` | Location key passed in |
| `scores` | `dict[str, float]` | Mean grade per category (includes -1 penalties) |
| `recall` | `dict[str, float]` | Matched GT count / total GT count per category |
| `fpr` | `dict[str, float]` | Unmatched extracted count / total extracted count per category |
| `grades` | `dict[str, PolicyGrade]` | Per-policy grades keyed by `'{category}::{policy_id}'` |

**`PolicyGrade`**
| Field | Type | Description |
|---|---|---|
| `policy_id` | `str` | Truncated policy statement used as identifier |
| `grade` | `Literal[-1, 0, 1]` | +1 match / 0 vague / -1 no match |
| `reasoning` | `str` | Step-by-step justification |

---

## prompt_optimizer.py

Implements **Algorithm 2: LEAP Prompt Optimizer** and **Algorithm 3: LEAP Prompt Optimization Loop**.

The RLM system prompt is decomposed into five independent prongs:

```
rho_t = ( rho_t^gen, rho_t^mit, rho_t^ada, rho_t^eff, rho_t^nbs )
```

Each prong is updated independently based on the per-location, per-category score delta. A prong reverts to its previous version when a negative signal is detected for that category, preventing a degraded prompt from being used as the base for further rewrites.

```python
from optimization.prompt_optimizer import LEAPPromptOptimizer, StructuredPrompt
from base_rlm_pipeline_v3 import CLIMATE_RLM_SYSTEM_PROMPT

# Bootstrap from existing flat prompt
initial = StructuredPrompt.from_flat(CLIMATE_RLM_SYSTEM_PROMPT)

optimizer = LEAPPromptOptimizer(model="gpt-5")

optimized = optimizer.run_loop(
    location="Seattle_US",
    extracted_policies_fn=lambda prompt: run_rlm(prompt, doc_md),  # wraps your RLM call
    ground_truth_policies=genius_policies,
    rubric="Grade on specificity, commitment, and mechanism...",
    initial_prompt=initial,
    source_document=doc_md,   # optional
    max_iterations=10,
    epsilon=0.01,
)

print(optimized.compose())   # full prompt string ready for the next RLM run
```

### StructuredPrompt

| Field | Prong | Description |
|---|---|---|
| `gen` | General | Policy definition, hierarchy rules, output format, what not to extract |
| `mit` | Mitigation | Mitigation-specific extraction and classification tips |
| `ada` | Adaptation | Adaptation-specific tips |
| `eff` | Resource Efficiency | Resource Efficiency-specific tips |
| `nbs` | Nature-Based Solutions | NBS-specific tips |

| Method | Description |
|---|---|
| `compose()` | Assemble all prongs into a single prompt string |
| `decompose(prompt)` | Parse a composed prompt string back into prongs |
| `from_flat(prompt)` | Bootstrap from a legacy flat prompt string (places content in `gen`) |

### Update rule (Algorithm 2)

For each category `c` at location `l`, iteration `t`:

```
Delta_{l,c} = S_t[l, c] - S_{t-1}[l, c]

if Delta_{l,c} > 0:  keep rho_t^c,     resample with F_{l,c}
else:                 revert rho_{t-1}^c, resample with F_{l,c}

Delta_l = mean_c( Delta_{l,c} )

if Delta_l > 0:  keep rho_t^gen,     resample with F_l
else:            revert rho_{t-1}^gen, resample with F_l
```

### Convergence (Algorithm 3)

The loop terminates when `max_c( |Delta_{l,c}| ) < epsilon` or `t >= T`.

---

## Environment

Requires `OPENAI_API_KEY` in `.env` or the environment.

```bash
pip install openai pydantic scipy numpy python-dotenv
```
