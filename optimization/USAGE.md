# LEAP Optimizer — Usage Guide

A practical, end-to-end walkthrough of the redesigned optimizer: data prep,
running the loop, reading the logs, debugging a regression. For the
algorithmic *why*, see `OPTIMIZER_REDESIGN_PLAN.md`. For API reference, see
`README.md`.

---

## 0. Prerequisites

You need:

- Python 3.11+ with `openai`, `scipy`, `numpy`, `pydantic>=2`, `python-dotenv`,
  `pytest` installed (no other dependencies — `pyyaml` is optional).
- An `OPENAI_API_KEY` in `.env` at the repo root or exported in your shell.
- At least one ground-truth CSV from GENIUS and the corresponding markdown
  source document.
- The RLM environment your `rlm_pipeline.py` already depends on
  (`rlm`, `dspy`, `docling`).

Run the test suite once to confirm the install is healthy:

```bash
cd optimization
python3 -m pytest tests/
```

You should see `46 passed`.

---

## 1. Prepare your ground-truth CSV

Each location's GT rows must live in a CSV with at minimum these columns:

| Column | Required | Notes |
|---|---|---|
| `location` | yes | The verbatim string the YAML `name` will match against. |
| `policy_statement` | yes | Free-text policy commitment. |
| `primary_category` | yes | One of: `Mitigation`, `Adaptation`, `Resource Efficiency`, `Nature-Based Solutions`. |
| `role` | recommended | `parent` / `sub` / `individual`. Defaults to `individual` when absent. |
| `parent_statement` | for subs | The `policy_statement` text of the row this row is a child of. |
| `financial_instrument` | optional | Drives the classification bundle's secondary signals. |
| `secondary_category` (or `secondary_categories`) | optional | Same. |

**Single shared CSV is fine.** You can put every city in one file and let the
loader filter by `location` per-entry. The column name defaults to `location`
but is overridable (`location_column:` in the YAML, top-level or per-entry).

Today's bundled CSV (`organized_outputs/structured_policies.csv`) is
Seattle-only and has **no** `location` column — add one before using the new
loop. A one-liner:

```python
import pandas as pd
df = pd.read_csv("organized_outputs/structured_policies.csv")
df["location"] = "Seattle_US"
df.to_csv("organized_outputs/structured_policies.csv", index=False)
```

---

## 2. Configure `locations.yaml`

This is the single source of truth for which cities the optimizer trains and
evaluates on. It lives at `optimization/locations.yaml`.

```yaml
# Optional. Default: "location". Override per-entry by adding the same key
# under a single locations[*] item.
location_column: location

locations:
  # Dev cities drive every acceptance decision. The loop evaluates the
  # current prompt across ALL dev cities (× seeds) and aggregates before
  # deciding whether to accept a candidate.
  - name: Seattle_US
    ground_truth_csv: organized_outputs/structured_policies.csv
    source_document_md: ../../GENIUS/docs/cities/seattle_markdown.md
    split: dev

  - name: Portland_US
    ground_truth_csv: organized_outputs/structured_policies.csv
    source_document_md: ../../GENIUS/docs/cities/portland_markdown.md
    split: dev

  # Test cities are evaluated EXACTLY ONCE at the end of the loop.
  # The dev→test composite gap is the overfit estimate.
  - name: Boston_US
    ground_truth_csv: organized_outputs/structured_policies.csv
    source_document_md: ../../GENIUS/docs/cities/boston_markdown.md
    split: test
```

Rules:

- `name` is matched **case-sensitive, verbatim** against the
  `location_column` value of every CSV row.
- All paths are resolved relative to `locations.yaml`'s own directory.
- Misconfiguration is loud: a missing `location` column or zero matching rows
  raises `ValueError` immediately with a hint about the actual column names.
- The plan recommends the test city be from a different region than the dev
  cities so generalization is actually being tested.

---

## 3. Train (run the optimizer)

### 3a. From the command line — apples-to-apples comparison

`compare_legacy_vs_new.py` runs both loops from the same starting prompt on
the same dev set, then writes a markdown report. Use this as your standing
go/no-go gate before flipping the default to the new evaluator.

```bash
cd optimization

python3 compare_legacy_vs_new.py \
    --iterations 3 \
    --seeds 2 \
    --locations locations.yaml
```

CLI flags:

| Flag | Default | Meaning |
|---|---|---|
| `--iterations` | `3` | Outer loop iterations (max). |
| `--seeds` | `1` | RLM extraction runs per (location, iteration) for variance estimation. |
| `--model` | `gpt-4.1-2025-04-14` | OpenAI model used by both grader and resampler. |
| `--rlm-max-iterations` | `30` | Inner RLM recursion depth per extraction call. |
| `--locations` | `optimization/locations.yaml` | Path to the YAML described above. |
| `--out` | `optimization/logs/comparison/` | Where the report and raw logs land. |

Output:

```
logs/comparison/
├── comparison_report.md      # the headline doc
├── legacy/<timestamp>/...    # legacy loop's own logs
└── new/v2_<timestamp>/...    # new loop's logs (see §4)
```

### 3b. From Python — new loop only

If you don't need the legacy comparison and just want to run the new loop:

```python
from pathlib import Path

from dev_test_split import load_locations_from_yaml
from evaluator import LEAPEvaluator
from prompt_optimizer import (
    AcceptanceConfig,
    LEAPPromptOptimizer,
    PrognTarget,
    load_migrated_baseline,
)
from rlm_pipeline import run_rlm_for_optimizer


def extract_fn(prompt, location, src_doc, trace_path):
    return run_rlm_for_optimizer(
        prompt_string=prompt,
        document_path=str(src_doc),
        trace_dir=trace_path,
        model_name="gpt-4.1-2025-04-14",
        sub_model_name="gpt-4.1-2025-04-14",
        max_iterations=30,
    )


location_set   = load_locations_from_yaml("locations.yaml")
initial_prompt = load_migrated_baseline("migrated_baseline.json")

optimizer = LEAPPromptOptimizer(model="gpt-4.1-2025-04-14")
evaluator = LEAPEvaluator(
    model="gpt-4.1-2025-04-14",
    use_new_evaluator=True,
    similarity_threshold=0.55,
)

optimized = optimizer.run_loop_v2(
    location_set=location_set,
    extracted_policies_fn=extract_fn,
    rubric="Grade on specificity, commitment, and accuracy.",
    initial_prompt=initial_prompt,
    max_iterations=10,
    seeds=2,
    targets=PrognTarget(),                 # tweak per-prong target floors
    acceptance=AcceptanceConfig(           # tweak acceptance criteria
        min_delta=0.02,
        per_metric_floor={
            "extraction.f1":                              0.10,
            "hierarchy.role_agreement":                   0.10,
            "classification.primary_category_agreement":  0.10,
            "quality.plus_one_coverage":                  0.10,
        },
    ),
    log_dir=Path("logs/"),
    evaluator=evaluator,
)

print(optimized.compose())   # final RLM system prompt
```

**LLM-only grader:** pass [`LLMEvaluator`](llm_evaluator.py) instead of
`LEAPEvaluator(use_new_evaluator=True)` when you want the same
`EvaluationBundle` and `run_loop_v2` contract without the RLM in the
evaluator. Per matched pair, grading is a single structured OpenAI call. By
default (`include_source_document=False`) the grader compares extracted rows to
ground truth only; set `include_source_document=True` if you want the full CAP
pasted into the grader user message (still one call per pair, not RLM).

```python
from llm_evaluator import LLMEvaluator

evaluator = LLMEvaluator(
    model="gpt-4.1-2025-04-14",
    similarity_threshold=0.55,
)
# use as `evaluator=` in `run_loop_v2` like the `LEAPEvaluator` example above
```

The first time you run this, the script will lazily migrate the legacy flat
`CLIMATE_RLM_SYSTEM_PROMPT` into three task prongs and cache the result at
`optimization/migrated_baseline.json`. Subsequent runs reuse the cache; delete
the file to regenerate.

### 3c. Knobs worth knowing

| Knob | Where | Effect |
|---|---|---|
| `similarity_threshold` | `LEAPEvaluator(...)` | Pairs below this cosine similarity are dropped from M to U_E/U_G. Lower → more matches but noisier; higher → tighter matches but more spurious-extraction signal. Plan default: `0.55`. |
| `seeds` | `run_loop_v2(...)` | RLM runs per (location, iteration). 1 is fastest, 2 gives a usable std, 3 is safest. |
| `max_accepted_per_iteration` | `run_loop_v2(...)` | Cap on prong rewrites accepted per iteration. Default 2 — prevents one bad iteration from compounding. Ignored when `composite_candidate=True`. |
| `composite_candidate` | `run_loop_v2(...)` | When `True`, all triggered prongs are chain-rewritten into a single candidate prompt that is evaluated **once** per iteration. Trades per-prong accept/reject attribution for ~N× fewer RLM extractions per iteration. Default `False`. |
| `min_delta` | `AcceptanceConfig(...)` | Composite-score improvement required to accept. Default `0.02`. |
| `per_metric_floor` | `AcceptanceConfig(...)` | Reject if any listed metric drops by ≥ floor. Cliff guard. |
| `targets` | `PrognTarget(...)` | Per-prong headline targets. A prong is triggered when its headline is below target *or* worsened since last iteration. |

---

## 4. Read the logs

Each `run_loop_v2` invocation writes to a fresh timestamped directory at
`<log_dir>/v2_<YYYY-MM-DDTHH-MM-SS>/`:

```
v2_2026-04-22T20-15-37/
├── iteration_log.csv               # one row per iteration baseline
├── candidate_log.csv               # one row per proposed candidate
├── metrics_bundle_iter_1.json      # full bundle dump for offline analysis
├── metrics_bundle_iter_2.json
├── ...
├── test_results.json               # held-out test eval (if any test cities)
└── iter1_<location>_seed0_trace/   # RLM trace dirs per (iter, location, seed)
```

### `iteration_log.csv`

| Column | What to look at |
|---|---|
| `composite_score` | The single number to track over iterations. |
| `composite_score_std` | Std across (locations × seeds). Big std → trust trends less. |
| `extraction_f1` / `extraction_precision` / `extraction_recall` | Headline + decomposition for the extraction prong. |
| `extraction_jsd` | Jensen-Shannon divergence between extracted and GT category distributions. Closer to 0 = better category mix. |
| `plus_one_coverage` | Quality bundle headline: matched +1 grades / |GT|. |
| `hierarchy_role_agreement` / `hierarchy_parent_attribution` | Hierarchy bundle. |
| `classification_primary_agreement` / `_financial_agreement` / `_secondary_agreement` | Classification bundle. |
| `triggered_prongs` | Pipe-separated list of which prongs fired this iteration. |
| `composed_prompt` | The full RLM system prompt at the END of the iteration (after any accepted rewrites). |

### `candidate_log.csv`

| Column | What to look at |
|---|---|
| `scope` | `extraction` / `hierarchy` / `classification` — which prong was rewritten. |
| `accepted` | True / False. |
| `reason` | Why it was rejected (composite delta below `min_delta` or which metric cliffed). |
| `composite_delta` | Signed change vs. the running baseline at the moment this candidate was evaluated. |

### `metrics_bundle_iter_*.json`

Full structured bundle (matched pairs, per-category recall, confusion matrix,
grade reasoning, the lot). Use these for ad-hoc analysis, e.g.:

```python
import json
data = json.load(open("logs/v2_.../metrics_bundle_iter_3.json"))
matched = data["bundle"]["matching"]["matched"]
worst = sorted(matched, key=lambda p: p["grade"])[:10]
for p in worst:
    print(p["ground_truth"]["policy_statement"][:80], "→", p["reasoning"])
```

### `test_results.json`

Written after the loop terminates iff your YAML has `split: test` entries.

```json
{
  "aggregate":      { ...EvaluationBundle for the test set... },
  "std_per_metric": { ... },
  "per_location":   [ ... per-test-city bundles ... ],
  "dev_vs_test_composite_gap": 0.07
}
```

A small positive gap (≤ ~0.05) is healthy; anything large means the prompt
overfit to the dev set.

---

## 5. Workflow recipes

### Recipe A: First time on a new city

1. Add the city's GT rows to your shared CSV with `location = "<city>"`.
2. Add a `locations.yaml` entry pointing to that CSV with the same `name`.
3. `python3 -m pytest tests/test_location_filter.py` — smoke-checks the
   loader path.
4. Start cheap: `--iterations 1 --seeds 1`. Confirm `iteration_log.csv`
   composite_score and headlines are populated and within `[0, 1]`.
5. Scale up: `--iterations 5 --seeds 2` once cheap run looks sane.

### Recipe B: A prong is regressing

1. Open `iteration_log.csv` — find the iteration where the relevant headline
   dropped (e.g. `extraction_f1` fell from 0.62 → 0.55).
2. Open `candidate_log.csv` filtered to that iteration. The `extraction`
   row's `accepted` will tell you whether the regression was an accepted
   candidate (loop's fault — tighten `per_metric_floor["extraction.f1"]`)
   or a baseline drift (RLM noise — bump `seeds`).
3. Crack open the matching `metrics_bundle_iter_N.json` — the
   `matching.unmatched_gt` list is the concrete set of policies the prompt
   stopped catching.
4. Compare `composed_prompt` between iteration N and N-1 in
   `iteration_log.csv` to see what the resampler changed.

### Recipe C: Loop won't converge

- Check `triggered_prongs` per iteration. If the same prong keeps firing,
  its target is too high relative to what's actually achievable — lower it
  in `PrognTarget(...)` or accept that the headline target is aspirational.
- Check `composite_score_std`. If it's ≥ ~`min_delta`, you can't tell signal
  from noise. Bump `seeds`.
- Check `candidate_log.csv` `reason` column — if every candidate is rejected
  for "insufficient improvement", consider lowering `min_delta` or
  inspecting the resampler's outputs in `composed_prompt`.

### Recipe D: Test gap is large

- The composite-weight defaults
  (`extraction_f1=0.35, hierarchy=0.20, classification=0.20, plus_one_coverage=0.25`)
  may not match your real-world priorities. Override by passing `weights=...`
  to `compute_composite_score` (currently called inside the evaluator —
  expose the weights in your wrapper if you need to tune them).
- More dev cities is the structural fix. Two dev + one test is the minimum
  the plan recommends; three dev is safer if you have the GENIUS data.

---

## 6. When to use which path

| Situation | Use |
|---|---|
| Production today (cutover not yet complete) | `LEAPEvaluator(use_new_evaluator=False)` + `optimizer.run_loop(...)` |
| Evaluating the new system before flipping the default | `compare_legacy_vs_new.py` |
| Real training on a multi-city dev set | `LEAPEvaluator(use_new_evaluator=True)` + `optimizer.run_loop_v2(...)` |
| Ad-hoc one-off scoring of a single (extracted, GT) pair set | `evaluator.evaluate(...)` directly with `use_new_evaluator=True` |

Once you've run `compare_legacy_vs_new.py`, looked at
`comparison_report.md`, and decided the new loop wins, flip the default in
`LEAPEvaluator.__init__` (or just pass `use_new_evaluator=True` everywhere
in your call sites). The legacy code stays in place until the cleanup PR
described in `IMPLEMENTATION_PLAN.md` step 12.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ValueError: ...missing the 'location' column...` | GT CSV doesn't have the column the loader expects. | Add the column, or set `location_column:` in the YAML to your existing column name. |
| `ValueError: No ground-truth rows found...` | YAML `name` doesn't match any value in the CSV verbatim. | Check case, underscores, suffixes. |
| `ValueError: run_loop_v2 requires evaluator.use_new_evaluator=True` | Passed a legacy evaluator into the new loop. | Construct `LEAPEvaluator(use_new_evaluator=True)` or `LLMEvaluator` (always new path). |
| 429 rate-limit errors during grading | Concurrency too high. | Lower `evaluator.GRADER_CONCURRENCY` from 8 to 4 (module-level constant). |
| Resampler returns garbage | Model mismatch. | Confirm `DEFAULT_MODEL` in `evaluator.py` is one your org actually has access to. |
| Loop stops after one iteration with "All targets met" but headlines look low | Targets too lenient. | Raise the relevant field on `PrognTarget(...)`. |
| `metrics_bundle_iter_*.json` files are empty / missing | `log_dir=None` was passed. | Pass a real `Path` to `run_loop_v2`. |

---

## 8. What's where (file map)

| File | Purpose |
|---|---|
| `evaluator.py` | Both evaluator paths + grader prompts + concurrent grading. |
| `metrics.py` | Bundle Pydantic models, pure-function bundle computation, composite + aggregation. |
| `dev_test_split.py` | `LocationConfig` / `LocationSet` / YAML loader with the verbatim location filter. |
| `prompt_optimizer.py` | Both optimizer loops + per-prong triggers + multi-criterion acceptance + new logging. |
| `compare_legacy_vs_new.py` | Go/no-go harness. |
| `locations.yaml` | The file you actually edit to choose what to train on. |
| `migrated_baseline.json` | Cached three-prong rewrite of the legacy flat prompt. Safe to delete to regenerate. |
| `tests/` | 46 tests across the whole stack. Run before every change. |
| `OPTIMIZER_REDESIGN_PLAN.md` | Architectural rationale (the *why*). |
| `IMPLEMENTATION_PLAN.md` | PR-by-PR rollout plan. |
| `README.md` | API reference. |
| `USAGE.md` | This file. |
