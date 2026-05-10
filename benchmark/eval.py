"""
eval.py — Single-config evaluation entrypoint

Runs one system (defined by a config file) against one or all corpus cities
and writes results to results/{config_stem}/{timestamp}/.

Usage
-----
    # Evaluate one city
    python eval.py --config configs/openai_gpt-5.4.yaml --city Seattle

    # Evaluate all cities
    python eval.py --config configs/openai_gpt-5.4.yaml --all-cities

    # Use RLM grading with the full source document
    python eval.py --config configs/rlm_gpt-5.5.yaml --all-cities --grade-with-doc

Output per run
--------------
    results/{config_stem}/{timestamp}/
        extracted_policies.csv   raw extraction output (one row per policy)
        scores.jsonl             full EvaluationOutput per city (one JSON per line)
        scores.csv               flat one-row-per-city summary (easy to concat)
        grades.csv               per-pair comparison: full GT policy, full extracted
                                 policy, both categories/roles, grade, similarity,
                                 match flags, and LLM reasoning
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import pathlib
import re
import sys

import yaml
from dotenv import load_dotenv

load_dotenv()

_HERE = pathlib.Path(__file__).resolve().parent
_LEAP = _HERE.parent
if str(_LEAP) not in sys.path:
    sys.path.insert(0, str(_LEAP))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data    import CORPUS, load_ground_truth
from metrics import LEAPEvaluator, EvaluationOutput, DEFAULT_RUBRIC, CATEGORIES
from systems import build_runner
from core.rlm_pipeline import CLIMATE_RLM_SYSTEM_PROMPT, parse_document

RESULTS_DIR = _HERE / "results"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _append_csv(path: pathlib.Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Append rows to a CSV, writing the header only on first write."""
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def completed_locations(run_dir: pathlib.Path) -> set[str]:
    """Return location keys already present in run_dir/scores.csv."""
    scores_path = run_dir / "scores.csv"
    if not scores_path.exists():
        return set()
    with open(scores_path, newline="", encoding="utf-8") as fh:
        return {row["location"] for row in csv.DictReader(fh)}


_KEY_RE = re.compile(r"^(.+?)::gt(\d+)_ext(\d+)_")

_GRADES_FIELDS = [
    "location", "gt_index", "ext_index",
    "gt_policy", "gt_category", "gt_role",
    "produced_policy", "produced_category", "produced_role",
    "grade", "similarity", "statement_match", "role_match", "category_match", "reasoning",
]


def _append_results(
    run_dir: pathlib.Path,
    result: EvaluationOutput,
    extracted: list[dict],
    ground_truth: list[dict],
) -> None:
    """Append one city's results into the shared run_dir CSV files."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # extracted_policies.csv — stamped with location
    if extracted:
        stamped = [{"location": result.location, **p} for p in extracted]
        _append_csv(run_dir / "extracted_policies.csv", stamped, list(stamped[0].keys()))

    # scores.jsonl — one JSON object per line
    with open(run_dir / "scores.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(result.model_dump(), ensure_ascii=False) + "\n")

    # scores.csv — one row per city
    row: dict = {
        "location":                       result.location,
        "composite_score":                round(result.composite_score, 4),
        "extraction_f1":                  round(result.extraction_f1, 4),
        "extraction_precision":           round(result.extraction_precision, 4),
        "extraction_recall":              round(result.extraction_recall, 4),
        "role_agreement":                 round(result.role_agreement, 4),
        "parent_attribution_accuracy":    round(result.parent_attribution_accuracy, 4),
        "primary_category_agreement":     round(result.primary_category_agreement, 4),
        "financial_instrument_agreement": round(result.financial_instrument_agreement, 4),
        "secondary_category_agreement":   round(result.secondary_category_agreement, 4),
        "plus_one_coverage":              round(result.plus_one_coverage, 4),
        "matched_count":                  result.matched_count,
        "unmatched_extracted_count":      result.unmatched_extracted_count,
        "unmatched_ground_truth_count":   result.unmatched_ground_truth_count,
    }
    for cat in CATEGORIES:
        slug = cat.replace(" ", "_").replace("-", "_")
        row[f"{slug}_score"]  = round(result.scores.get(cat, 0.0), 4)
        row[f"{slug}_recall"] = round(result.recall.get(cat, 0.0), 4)
        row[f"{slug}_fpr"]    = round(result.fpr.get(cat, 0.0), 4)
    _append_csv(run_dir / "scores.csv", [row], list(row.keys()))

    # grades.csv — full GT vs produced comparison, one row per matched pair
    grade_rows = []
    for key, grade in result.grades.items():
        m = _KEY_RE.match(key)
        if not m:
            continue
        gj = int(m.group(2))
        ei = int(m.group(3))
        gt_row  = ground_truth[gj] if gj < len(ground_truth) else {}
        ext_row = extracted[ei]    if ei < len(extracted)    else {}
        grade_rows.append({
            "location":          result.location,
            "gt_index":          gj,
            "ext_index":         ei,
            "gt_policy":         gt_row.get("policy_statement", ""),
            "gt_category":       gt_row.get("primary_category", ""),
            "gt_role":           gt_row.get("role", ""),
            "produced_policy":   ext_row.get("policy_statement", ""),
            "produced_category": ext_row.get("primary_category", ""),
            "produced_role":     ext_row.get("role", ""),
            "grade":             grade.grade,
            "similarity":        round(grade.similarity, 4) if grade.similarity is not None else "",
            "statement_match":   grade.statement_match if grade.statement_match is not None else "",
            "role_match":        grade.role_match if grade.role_match is not None else "",
            "category_match":    grade.category_match if grade.category_match is not None else "",
            "reasoning":         grade.reasoning,
        })
    if grade_rows:
        _append_csv(run_dir / "grades.csv", grade_rows, _GRADES_FIELDS)



# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def run_eval(
    config_path: pathlib.Path,
    cities: list[str],
    grade_with_doc: bool = False,
) -> None:
    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    runner   = build_runner(config)
    ts       = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    base_dir = RESULTS_DIR / config_path.stem / ts

    evaluator = LEAPEvaluator(
        model=config.get("evaluator_model", "gpt-5.4"),
        similarity_threshold=config.get("similarity_threshold", 0.55),
    )

    run_dir      = base_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    done_locs    = completed_locations(run_dir)

    for city in cities:
        cfg          = CORPUS[city]
        ground_truth = load_ground_truth(city)
        doc_path     = cfg["document"]

        if cfg["location_key"] in done_locs:
            print(f"\n  {city} already completed — skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Config   : {config_path.name}")
        print(f"  City     : {city}  ({len(ground_truth)} GT policies)")
        print(f"{'=' * 60}")

        doc_markdown = parse_document(str(doc_path))
        print(f"  Parsed document: {len(doc_markdown):,} chars")

        print(f"  Extracting with {runner.model_slug}...")
        extracted = runner.run(doc_markdown, CLIMATE_RLM_SYSTEM_PROMPT)
        print(f"  Extracted {len(extracted)} policies")

        print("  Evaluating...")
        result = evaluator.evaluate(
            location=cfg["location_key"],
            extracted_policies=extracted,
            ground_truth_policies=ground_truth,
            rubric=DEFAULT_RUBRIC,
            source_document_path=doc_path if grade_with_doc else None,
        )

        _append_results(run_dir, result, extracted, ground_truth)

        print(f"  composite={result.composite_score:.4f}  "
              f"f1={result.extraction_f1:.4f}  "
              f"+1_cov={result.plus_one_coverage:.4f}  "
              f"matched={result.matched_count}")
        print(f"  Saved to: {run_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate one system config against corpus cities."
    )
    parser.add_argument(
        "--config", required=True, type=pathlib.Path,
        help="Path to a YAML config file, e.g. configs/openai_gpt-5.4.yaml",
    )
    city_group = parser.add_mutually_exclusive_group(required=True)
    city_group.add_argument(
        "--city", choices=list(CORPUS.keys()),
        help="Single city to evaluate.",
    )
    city_group.add_argument(
        "--all-cities", action="store_true",
        help="Evaluate against all corpus cities.",
    )
    parser.add_argument(
        "--grade-with-doc", action="store_true",
        help="Pass the full source document to the grader (uses RLM grading).",
    )
    args = parser.parse_args()

    cities = list(CORPUS.keys()) if args.all_cities else [args.city]
    run_eval(
        config_path=args.config,
        cities=cities,
        grade_with_doc=args.grade_with_doc,
    )
