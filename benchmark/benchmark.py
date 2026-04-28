"""
benchmark.py — Full cross-system benchmark

Runs every config in configs/ against every corpus city and prints a
combined results table. Per-city results are written to results/ by eval.py.

Usage
-----
    # Full benchmark (all configs, all cities)
    python benchmark.py

    # Subset of cities
    python benchmark.py --cities Hiroshima Seattle

    # Subset of configs
    python benchmark.py --configs configs/openai_*.yaml

    # Skip RLM (faster/cheaper smoke test)
    python benchmark.py --skip-rlm
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import yaml
from dotenv import load_dotenv

load_dotenv()

_HERE = pathlib.Path(__file__).resolve().parent
_OPT  = _HERE.parent / "optimization"
if str(_OPT) not in sys.path:
    sys.path.insert(0, str(_OPT))

from data    import CORPUS, load_ground_truth
from metrics import LEAPEvaluator, DEFAULT_RUBRIC, CATEGORIES, EvaluationOutput
from systems import build_runner
from eval    import run_eval
from rlm_pipeline import CLIMATE_RLM_SYSTEM_PROMPT, parse_document

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

SUMMARY_METRICS: list[tuple[str, str]] = [
    ("composite_score",              "composite"),
    ("extraction_f1",                "f1"),
    ("extraction_precision",         "precision"),
    ("extraction_recall",            "recall"),
    ("plus_one_coverage",            "+1_cov"),
    ("role_agreement",               "role_agr"),
    ("primary_category_agreement",   "cat_agr"),
    ("parent_attribution_accuracy",  "parent_attr"),
    ("matched_count",                "matched"),
    ("unmatched_extracted_count",    "unm_ext"),
    ("unmatched_ground_truth_count", "unm_gt"),
]


def _print_table(
    results: dict[str, dict[str, EvaluationOutput | None]],
    city_names: list[str],
    config_labels: list[str],
) -> None:
    col_w   = 11
    label_w = 16

    for city in city_names:
        print(f"\n{'=' * 72}")
        print(f"  {city}")
        print(f"{'=' * 72}")
        header = f"  {'Metric':<{label_w}}" + "".join(
            f"{lbl[:col_w]:>{col_w}}" for lbl in config_labels
        )
        print(header)
        print("-" * len(header))

        for attr, display in SUMMARY_METRICS:
            row = f"  {display:<{label_w}}"
            for lbl in config_labels:
                res = results[city].get(lbl)
                if res is None:
                    row += f"{'ERR':>{col_w}}"
                else:
                    val = getattr(res, attr, None)
                    if isinstance(val, float):
                        row += f"{val:>{col_w}.4f}"
                    else:
                        row += f"{val:>{col_w}}"
            print(row)

        print(f"{'=' * 72}")

    print(f"\nPer-system summary.csv saved in each run folder under {_HERE / 'results'}.")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    config_paths: list[pathlib.Path],
    cities: list[str],
    grade_with_doc: bool = False,
) -> None:
    config_labels = [p.stem for p in config_paths]
    results: dict[str, dict[str, EvaluationOutput | None]] = {
        city: {} for city in cities
    }

    import datetime
    from eval import _append_results

    total       = len(config_paths) * len(cities)
    run_counter = 0

    from eval import completed_locations

    for config_path in config_paths:
        with open(config_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        runner    = build_runner(config)
        evaluator = LEAPEvaluator(
            model=config.get("evaluator_model", "gpt-5.4"),
            similarity_threshold=config.get("similarity_threshold", 0.55),
        )

        # One run_dir per config, shared across all cities.
        ts      = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = _HERE / "results" / config_path.stem / ts
        run_dir.mkdir(parents=True, exist_ok=True)

        # Resume: read any cities already completed in a previous attempt.
        done_locs = completed_locations(run_dir)

        print(f"\n{'#' * 60}")
        print(f"# Config: {config_path.stem}")
        print(f"{'#' * 60}")

        for city in cities:
            run_counter += 1
            cfg          = CORPUS[city]
            ground_truth = load_ground_truth(city)

            if cfg["location_key"] in done_locs:
                print(f"\n[{run_counter}/{total}] {config_path.stem} — {city} (already completed, skipping)")
                continue

            print(f"\n[{run_counter}/{total}] {config_path.stem} — {city} "
                  f"({len(ground_truth)} GT policies)")

            try:
                doc_markdown = parse_document(str(cfg["document"]))
                extracted    = runner.run(doc_markdown, CLIMATE_RLM_SYSTEM_PROMPT)
                result       = evaluator.evaluate(
                    location=cfg["location_key"],
                    extracted_policies=extracted,
                    ground_truth_policies=ground_truth,
                    rubric=DEFAULT_RUBRIC,
                    source_document_path=cfg["document"] if grade_with_doc else None,
                )
                results[city][config_path.stem] = result
                _append_results(run_dir, result, extracted)

                print(f"  composite={result.composite_score:.4f}  "
                      f"f1={result.extraction_f1:.4f}  "
                      f"+1_cov={result.plus_one_coverage:.4f}  "
                      f"matched={result.matched_count}")

            except Exception as e:
                print(f"  [ERROR] {e}")
                results[city][config_path.stem] = None

    _print_table(results, cities, config_labels)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LEAP full cross-system benchmark."
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="Config file paths (default: all files in configs/).",
    )
    parser.add_argument(
        "--cities", nargs="+", choices=list(CORPUS.keys()), default=None,
        help="Cities to evaluate (default: all).",
    )
    parser.add_argument(
        "--skip-rlm", action="store_true",
        help="Exclude configs whose stem starts with 'rlm_'.",
    )
    parser.add_argument(
        "--grade-with-doc", action="store_true",
        help="Pass the full source document to the grader.",
    )
    args = parser.parse_args()

    config_dir = _HERE / "configs"
    if args.configs:
        config_paths = [pathlib.Path(p) for p in args.configs]
    else:
        # GPT configs first, RLM configs after — openai_* sorts before rlm_* alphabetically.
        config_paths = sorted(config_dir.glob("*.yaml"),
                              key=lambda p: (0 if p.stem.startswith("openai_") else 1, p.stem))

    if args.skip_rlm:
        config_paths = [p for p in config_paths if not p.stem.startswith("rlm_")]

    cities = args.cities or list(CORPUS.keys())

    run_benchmark(
        config_paths=config_paths,
        cities=cities,
        grade_with_doc=args.grade_with_doc,
    )
