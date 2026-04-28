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
        extracted_policies.csv   raw extraction output
        scores.json              full EvaluationOutput as JSON
        scores.csv               flat one-row summary (easy to concat)
        grades.csv               per-pair grades with reasoning
        summary.csv              human-readable metric/value table
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
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
from metrics import LEAPEvaluator, EvaluationOutput, DEFAULT_RUBRIC, CATEGORIES
from systems import build_runner
from rlm_pipeline import CLIMATE_RLM_SYSTEM_PROMPT, parse_document

RESULTS_DIR = _HERE / "results"


# ---------------------------------------------------------------------------
# Output helpers (mirrors evaluation_harness._save_scores)
# ---------------------------------------------------------------------------

def _save_results(run_dir: pathlib.Path, result: EvaluationOutput, policies: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    # extracted_policies.csv
    if policies:
        with open(run_dir / "extracted_policies.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(policies[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(policies)
    else:
        (run_dir / "extracted_policies.csv").write_text("policy_statement\n", encoding="utf-8")

    # scores.json
    with open(run_dir / "scores.json", "w", encoding="utf-8") as fh:
        json.dump(result.model_dump(), fh, indent=2, ensure_ascii=False)

    # scores.csv (flat, one row — easy to concat across runs)
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
    with open(run_dir / "scores.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    # grades.csv
    grade_rows = []
    for key, grade in result.grades.items():
        cat, _ = key.split("::", 1) if "::" in key else ("", key)
        grade_rows.append({
            "location":        result.location,
            "key":             key,
            "category":        cat,
            "policy_id":       grade.policy_id,
            "grade":           grade.grade,
            "similarity":      round(grade.similarity, 4) if grade.similarity is not None else "",
            "statement_match": grade.statement_match if grade.statement_match is not None else "",
            "role_match":      grade.role_match if grade.role_match is not None else "",
            "category_match":  grade.category_match if grade.category_match is not None else "",
            "reasoning":       grade.reasoning,
        })
    with open(run_dir / "grades.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "location", "key", "category", "policy_id", "grade",
            "similarity", "statement_match", "role_match", "category_match", "reasoning",
        ])
        writer.writeheader()
        writer.writerows(grade_rows)

    # summary.csv (human-readable metric/value)
    summary = [
        {"metric": "composite_score",                "value": round(result.composite_score, 4)},
        {"metric": "extraction_f1",                  "value": round(result.extraction_f1, 4)},
        {"metric": "extraction_precision",           "value": round(result.extraction_precision, 4)},
        {"metric": "extraction_recall",              "value": round(result.extraction_recall, 4)},
        {"metric": "role_agreement",                 "value": round(result.role_agreement, 4)},
        {"metric": "parent_attribution_accuracy",    "value": round(result.parent_attribution_accuracy, 4)},
        {"metric": "primary_cat_agreement",          "value": round(result.primary_category_agreement, 4)},
        {"metric": "financial_instrument_agreement", "value": round(result.financial_instrument_agreement, 4)},
        {"metric": "secondary_category_agreement",   "value": round(result.secondary_category_agreement, 4)},
        {"metric": "plus_one_coverage",              "value": round(result.plus_one_coverage, 4)},
        {"metric": "matched",                        "value": result.matched_count},
        {"metric": "unmatched_ext",                  "value": result.unmatched_extracted_count},
        {"metric": "unmatched_gt",                   "value": result.unmatched_ground_truth_count},
    ]
    for cat in CATEGORIES:
        slug = cat.replace(" ", "_").replace("-", "_")
        summary += [
            {"metric": f"{slug}_score",  "value": round(result.scores.get(cat, 0.0), 4)},
            {"metric": f"{slug}_recall", "value": round(result.recall.get(cat, 0.0), 4)},
            {"metric": f"{slug}_fpr",    "value": round(result.fpr.get(cat, 0.0), 4)},
        ]
    with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(summary)


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

    for city in cities:
        cfg          = CORPUS[city]
        ground_truth = load_ground_truth(city)
        doc_path     = cfg["document"]

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

        run_dir = base_dir / city
        _save_results(run_dir, result, extracted)

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
