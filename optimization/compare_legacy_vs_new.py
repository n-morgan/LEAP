"""compare_legacy_vs_new.py — go/no-go comparison harness.

Runs the legacy bucketed loop and the redesigned global-matching loop from the
same starting prompt on the same dev set for the same number of iterations,
then writes ``comparison_report.md`` covering:

  * composite-score (legacy: mean of category scores; new: composite_score)
    trajectory for both;
  * per-bundle metric trajectory for the new loop;
  * number of accepted candidates per iteration for both;
  * dev-vs-test gap for the new loop;
  * wall time per iteration.

This is the gate for flipping ``use_new_evaluator`` to ``True`` in production.

Usage:
    python3 compare_legacy_vs_new.py --iterations 3 --locations locations.yaml
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json as _json
import pathlib
import time
from typing import Any, Optional

from dev_test_split import LocationSet, load_locations_from_yaml
from evaluator import DEFAULT_MODEL, LEAPEvaluator
from prompt_optimizer import (
    AcceptanceConfig,
    LEAPPromptOptimizer,
    PrognTarget,
    StructuredPrompt,
    load_migrated_baseline,
    migrate_to_task_prongs_with_notes,
)
from rlm_pipeline import CLIMATE_RLM_SYSTEM_PROMPT, run_rlm_for_optimizer


_HERE = pathlib.Path(__file__).resolve().parent


def _load_or_migrate_baseline(model: str) -> StructuredPrompt:
    baseline_path = _HERE / "migrated_baseline.json"
    if baseline_path.exists():
        return load_migrated_baseline(baseline_path)
    prompt, _, _ = migrate_to_task_prongs_with_notes(
        CLIMATE_RLM_SYSTEM_PROMPT, model=model, output_path=baseline_path,
    )
    return prompt


def _make_extract_fn(model: str, rlm_max_iterations: int):
    """Return a closure suitable for both legacy and new loops.

    Legacy ``run_loop`` calls ``extracted_policies_fn(prompt, trace_path)``.
    New ``run_loop_v2`` calls ``extracted_policies_fn(prompt, location, src, trace)``.
    We provide a single callable that detects the call shape from arity.
    """

    def fn(prompt: str, *args, **kwargs) -> list[dict]:
        if len(args) == 1:
            trace = args[0]
            doc_path = kwargs.get("source_document_path")
        elif len(args) >= 3:
            _location, doc_path, trace = args[0], args[1], args[2]
        else:
            doc_path = kwargs.get("source_document_md") or kwargs.get("source_document_path")
            trace = kwargs.get("trace_path")
        return run_rlm_for_optimizer(
            prompt_string=prompt,
            document_path=str(doc_path),
            trace_dir=trace,
            model_name=model,
            sub_model_name=model,
            max_iterations=rlm_max_iterations,
        )

    return fn


def _legacy_composite(scores: dict[str, float]) -> float:
    """Map the legacy per-category mean score onto a [0, 1]-ish composite."""
    if not scores:
        return 0.0
    mean = sum(scores.values()) / len(scores)
    # Legacy scores live in [-1, 1]; rescale to [0, 1] for like-for-like reading.
    return (mean + 1.0) / 2.0


def _read_csv_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--rlm-max-iterations", type=int, default=30)
    parser.add_argument(
        "--locations", type=pathlib.Path, default=_HERE / "locations.yaml",
    )
    parser.add_argument("--out", type=pathlib.Path, default=_HERE / "logs" / "comparison")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    location_set = load_locations_from_yaml(args.locations)
    if not location_set.dev:
        raise SystemExit("No dev locations defined in locations.yaml")

    rubric = (
        "Grade on specificity (quantified targets, deadlines, mechanisms), "
        "commitment strength (binding vs aspirational language), "
        "and accuracy relative to the source document."
    )

    extract_fn = _make_extract_fn(args.model, args.rlm_max_iterations)
    initial_prompt = _load_or_migrate_baseline(args.model)

    # ------------------------------------------------------------------
    # Run legacy loop on the first dev location only (legacy is single-location).
    # ------------------------------------------------------------------
    legacy_loc = location_set.dev[0]
    legacy_dir = args.out / "legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_evaluator = LEAPEvaluator(model=args.model, use_new_evaluator=False)
    legacy_optimizer = LEAPPromptOptimizer(model=args.model)

    print(f"\n{'#' * 60}\n# LEGACY LOOP\n{'#' * 60}")
    legacy_t0 = time.time()
    legacy_optimizer.run_loop(
        location=legacy_loc.name,
        extracted_policies_fn=extract_fn,
        ground_truth_policies=legacy_loc.load_ground_truth(),
        rubric=rubric,
        initial_prompt=initial_prompt,
        source_document_path=legacy_loc.source_document_md,
        max_iterations=args.iterations,
        log_dir=legacy_dir,
    )
    legacy_wall = time.time() - legacy_t0

    # Locate the most recent legacy run subdir
    legacy_runs = sorted([p for p in legacy_dir.iterdir() if p.is_dir()])
    legacy_run = legacy_runs[-1] if legacy_runs else None
    legacy_iter_rows = _read_csv_rows(legacy_run / "iteration_log.csv") if legacy_run else []
    legacy_cand_rows = _read_csv_rows(legacy_run / "candidate_log.csv") if legacy_run else []

    # ------------------------------------------------------------------
    # Run new loop on the full dev set.
    # ------------------------------------------------------------------
    new_dir = args.out / "new"
    new_dir.mkdir(parents=True, exist_ok=True)
    new_evaluator = LEAPEvaluator(model=args.model, use_new_evaluator=True)
    new_optimizer = LEAPPromptOptimizer(model=args.model)

    print(f"\n{'#' * 60}\n# NEW LOOP\n{'#' * 60}")
    new_t0 = time.time()
    new_optimizer.run_loop_v2(
        location_set=location_set,
        extracted_policies_fn=extract_fn,
        rubric=rubric,
        initial_prompt=initial_prompt,
        max_iterations=args.iterations,
        seeds=args.seeds,
        targets=PrognTarget(),
        acceptance=AcceptanceConfig(),
        log_dir=new_dir,
        evaluator=new_evaluator,
    )
    new_wall = time.time() - new_t0

    new_runs = sorted([p for p in new_dir.iterdir() if p.is_dir()])
    new_run = new_runs[-1] if new_runs else None
    new_iter_rows = _read_csv_rows(new_run / "iteration_log.csv") if new_run else []
    new_cand_rows = _read_csv_rows(new_run / "candidate_log.csv") if new_run else []
    test_results: Optional[dict] = None
    if new_run is not None and (new_run / "test_results.json").exists():
        with open(new_run / "test_results.json", encoding="utf-8") as fh:
            test_results = _json.load(fh)

    # ------------------------------------------------------------------
    # Build the comparison report
    # ------------------------------------------------------------------
    lines: list[str] = []
    lines.append(f"# LEAP Optimizer Comparison Report")
    lines.append(f"_Generated {datetime.datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append(f"- iterations: {args.iterations}")
    lines.append(f"- model: `{args.model}`")
    lines.append(f"- legacy log dir: `{legacy_run}`")
    lines.append(f"- new log dir:    `{new_run}`")
    lines.append(f"- legacy wall time: {legacy_wall:.1f}s")
    lines.append(f"- new wall time:    {new_wall:.1f}s")
    lines.append("")

    lines.append("## Composite-score trajectory")
    lines.append("| iter | legacy (rescaled mean→[0,1]) | new (composite) | new std |")
    lines.append("|---|---|---|---|")
    max_iters = max(len(legacy_iter_rows), len(new_iter_rows))
    for i in range(max_iters):
        leg = legacy_iter_rows[i] if i < len(legacy_iter_rows) else None
        new = new_iter_rows[i] if i < len(new_iter_rows) else None
        leg_score = (
            f"{(float(leg['mean_score']) + 1.0) / 2.0:.3f}" if leg else "-"
        )
        new_score = f"{float(new['composite_score']):.3f}" if new else "-"
        new_std = f"{float(new['composite_score_std']):.3f}" if new else "-"
        lines.append(f"| {i + 1} | {leg_score} | {new_score} | {new_std} |")
    lines.append("")

    lines.append("## New-loop per-bundle headlines per iteration")
    lines.append(
        "| iter | extraction.f1 | hierarchy.role | classification.primary | +1 coverage |"
    )
    lines.append("|---|---|---|---|---|")
    for r in new_iter_rows:
        lines.append(
            f"| {r['iteration']} | {r['extraction_f1']} | "
            f"{r['hierarchy_role_agreement']} | "
            f"{r['classification_primary_agreement']} | "
            f"{r['plus_one_coverage']} |"
        )
    lines.append("")

    lines.append("## Accepted candidates per iteration")
    lines.append("| iter | legacy accepts | new accepts |")
    lines.append("|---|---|---|")
    for i in range(max_iters):
        leg_acc = sum(
            1 for r in legacy_cand_rows
            if r.get("iteration") == str(i + 1) and r.get("accepted") == "True"
        )
        new_acc = sum(
            1 for r in new_cand_rows
            if r.get("iteration") == str(i + 1) and r.get("accepted") == "True"
        )
        lines.append(f"| {i + 1} | {leg_acc} | {new_acc} |")
    lines.append("")

    if test_results is not None:
        gap = test_results.get("dev_vs_test_composite_gap")
        lines.append("## Dev → test gap (new loop)")
        lines.append("")
        lines.append(
            f"- dev composite (final accepted): "
            f"{new_iter_rows[-1]['composite_score'] if new_iter_rows else 'n/a'}"
        )
        lines.append(
            f"- test composite: "
            f"{test_results['aggregate']['composite_score']:.3f}"
        )
        lines.append(f"- gap (dev − test): {gap:+.3f}" if gap is not None else "")
        lines.append("")

    lines.append("## Cutover gate")
    lines.append("")
    lines.append("Inspect the trajectory and gap above. Flip "
                 "``LEAPEvaluator.use_new_evaluator`` to ``True`` as the default "
                 "iff the new loop matches or beats legacy on its final "
                 "composite *and* the dev→test gap is within tolerance.")

    report_path = args.out / "comparison_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
