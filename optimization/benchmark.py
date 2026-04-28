"""
benchmark.py — Full cross-model benchmark for LEAP

Runs all configured extraction backends against every city in CITY_CONFIG and
writes per-run results (scores.json, scores.csv, grades.csv, summary.csv) to:

    evaluation_results/{model_slug}/{timestamp}/

Then prints a combined summary table grouped by city.

Backends benchmarked
--------------------
    RLM   gpt-5.5
    RLM   gpt-5.2
    GPT   gpt-5.5
    GPT   gpt-5.4
    GPT   gpt-5.4-mini
    GPT   gpt-5.2

Usage
-----
    python benchmark.py                        # all cities, all models
    python benchmark.py --cities Hiroshima Seattle   # subset of cities
    python benchmark.py --skip-rlm             # direct completions only (faster/cheaper)
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from evaluation_harness import (
    CITY_CONFIG,
    EvaluationHarness,
    load_ground_truth_for_city,
)
from evaluator import DEFAULT_RUBRIC, CATEGORIES
from rlm_pipeline import _DEFAULT_EXPERT_KNOWLEDGE_PATH
from runners import RLMRunner, OpenAIRunner

OUTPUT_DIR = _HERE / "evaluation_results"

# ---------------------------------------------------------------------------
# Runner definitions
# ---------------------------------------------------------------------------

def _build_runners(skip_rlm: bool = False) -> list[tuple[str, Any]]:
    """Return (label, runner) pairs in benchmark order."""
    runners: list[tuple[str, Any]] = []

    if not skip_rlm:
        runners += [
            ("RLM gpt-5.5", RLMRunner(
                model_name="gpt-5.5",
                expert_knowledge_path=_DEFAULT_EXPERT_KNOWLEDGE_PATH,
                max_iterations=50,
            )),
            ("RLM gpt-5.2", RLMRunner(
                model_name="gpt-5.2",
                expert_knowledge_path=_DEFAULT_EXPERT_KNOWLEDGE_PATH,
                max_iterations=50,
            )),
        ]

    runners += [
        ("GPT gpt-5.5",      OpenAIRunner(model_name="gpt-5.5",      expert_knowledge_path=_DEFAULT_EXPERT_KNOWLEDGE_PATH)),
        ("GPT gpt-5.4",      OpenAIRunner(model_name="gpt-5.4",      expert_knowledge_path=_DEFAULT_EXPERT_KNOWLEDGE_PATH)),
        ("GPT gpt-5.4-mini", OpenAIRunner(model_name="gpt-5.4-mini", expert_knowledge_path=_DEFAULT_EXPERT_KNOWLEDGE_PATH)),
        ("GPT gpt-5.2",      OpenAIRunner(model_name="gpt-5.2",      expert_knowledge_path=_DEFAULT_EXPERT_KNOWLEDGE_PATH)),
    ]

    return runners


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    cities: list[str] | None = None,
    skip_rlm: bool = False,
    evaluator_model: str = "gpt-5.4",
    similarity_threshold: float = 0.55,
) -> None:
    city_names = cities or list(CITY_CONFIG.keys())
    runners = _build_runners(skip_rlm)

    harness = EvaluationHarness(
        output_dir=OUTPUT_DIR,
        evaluator_model=evaluator_model,
        similarity_threshold=similarity_threshold,
        grade_with_document=False,
    )

    # results[city][label] = EvaluationOutput
    results: dict[str, dict[str, Any]] = {c: {} for c in city_names}

    total_runs = len(city_names) * len(runners)
    completed = 0

    for city in city_names:
        cfg = CITY_CONFIG[city]
        ground_truth = load_ground_truth_for_city(city)
        print(f"\n{'#' * 60}")
        print(f"# City: {city}  ({len(ground_truth)} GT policies)")
        print(f"{'#' * 60}")

        for label, runner in runners:
            completed += 1
            print(f"\n[{completed}/{total_runs}] {label} — {city}")
            try:
                result = harness.run(
                    runner=runner,
                    location=cfg["location_key"],
                    document_path=cfg["markdown"],
                    ground_truth_policies=ground_truth,
                    rubric=DEFAULT_RUBRIC,
                )
                results[city][label] = result
            except Exception as e:
                print(f"  [ERROR] {label} / {city} failed: {e}")
                results[city][label] = None

    _print_summary(results, city_names, runners)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

SUMMARY_METRICS: list[tuple[str, str]] = [
    ("composite_score",       "composite"),
    ("extraction_f1",         "f1"),
    ("extraction_precision",  "precision"),
    ("extraction_recall",     "recall"),
    ("plus_one_coverage",     "+1_cov"),
    ("role_agreement",        "role_agr"),
    ("primary_category_agreement", "cat_agr"),
    ("matched_count",         "matched"),
    ("unmatched_extracted_count", "unm_ext"),
    ("unmatched_ground_truth_count", "unm_gt"),
]


def _print_summary(
    results: dict[str, dict[str, Any]],
    city_names: list[str],
    runners: list[tuple[str, Any]],
) -> None:
    labels = [label for label, _ in runners]
    col_w = 10
    label_w = 26

    for city in city_names:
        print(f"\n{'=' * 70}")
        print(f"  {city}")
        print(f"{'=' * 70}")
        header = f"  {'Metric':<{label_w}}" + "".join(f"{l[:col_w]:>{col_w}}" for l in labels)
        print(header)
        print("-" * len(header))

        for attr, display in SUMMARY_METRICS:
            row = f"  {display:<{label_w}}"
            for label in labels:
                res = results[city].get(label)
                if res is None:
                    row += f"{'ERR':>{col_w}}"
                else:
                    val = getattr(res, attr, None)
                    if val is None:
                        row += f"{'N/A':>{col_w}}"
                    elif isinstance(val, float):
                        row += f"{val:>{col_w}.4f}"
                    else:
                        row += f"{val:>{col_w}}"
            print(row)

        print(f"{'=' * 70}")

    print(f"\nAll results saved under: {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LEAP full cross-model benchmark."
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        choices=list(CITY_CONFIG.keys()),
        default=None,
        help="Subset of cities to run (default: all).",
    )
    parser.add_argument(
        "--skip-rlm",
        action="store_true",
        help="Skip RLM runners (direct completions only).",
    )
    parser.add_argument(
        "--evaluator-model",
        default="gpt-5.4",
        help="Model used by LEAPEvaluator for grading pairs.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.55,
        help="Cosine similarity floor for Hungarian pair acceptance.",
    )
    args = parser.parse_args()

    run_benchmark(
        cities=args.cities,
        skip_rlm=args.skip_rlm,
        evaluator_model=args.evaluator_model,
        similarity_threshold=args.similarity_threshold,
    )
