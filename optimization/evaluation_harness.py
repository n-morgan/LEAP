"""
evaluation_harness.py — Multi-model evaluation harness for LEAP

Runs any supported extraction backend against a ground-truth policy set,
calls LEAPEvaluator to score the output, and writes results to:

    {output_dir}/{model_slug}/{timestamp}/
        extracted_policies.csv   — raw extraction output (one row per policy)
        scores.json              — full EvaluationOutput as JSON

Supported backends
------------------
    rlm        — RLM recursive pipeline (run_rlm_for_optimizer)
    openai     — OpenAI Chat Completions  (gpt-5.4, gpt-4o, etc.)
    anthropic  — Anthropic Messages API   (claude-opus-4-6, etc.)
    gemini     — Google Generative AI     (gemini-2.0-flash, etc.)

Usage (CLI)
-----------
    python evaluation_harness.py \\
        --backend openai \\
        --model gpt-5.4 \\
        --document ../docs/cities/seattle_markdown.md \\
        --ground-truth organized_outputs/structured_policies.csv \\
        --location Seattle_US

Usage (Python)
--------------
    from evaluation_harness import EvaluationHarness, OpenAIRunner

    runner = OpenAIRunner(model_name="gpt-5.4")
    harness = EvaluationHarness(output_dir="evaluation_results")
    result = harness.run(
        runner=runner,
        location="Seattle_US",
        document_path="path/to/seattle_markdown.md",
        ground_truth_policies=ground_truth_list,
        rubric=DEFAULT_RUBRIC,
    )
    print(result.composite_score, result.extraction_f1)
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import pathlib
from typing import Any, Optional

from dotenv import load_dotenv

from evaluator import DEFAULT_RUBRIC, LEAPEvaluator, EvaluationOutput
from rlm_pipeline import (
    CLIMATE_RLM_SYSTEM_PROMPT,
    parse_document,
    _DEFAULT_EXPERT_KNOWLEDGE_PATH,
    _HERE as _PIPELINE_HERE,
)
from runners import (
    ModelRunner,
    RLMRunner,
    OpenAIRunner,
    AnthropicRunner,
    GeminiRunner,
    RUNNER_CLASSES,
)

load_dotenv()

# ---------------------------------------------------------------------------
# City config — maps GT CSV city name → (location_key, markdown_filename)
#
# Source of truth: GENIUS/notebooks/outputs/all_cities_kept_classified_policies_final.csv
# Markdown files:  GENIUS/docs/cities/
# Las Vegas (LV.md) is excluded — no ground truth rows in the CSV.
# ---------------------------------------------------------------------------

_CITIES_DIR = _PIPELINE_HERE / "docs" / "cities"
_GT_CSV = _PIPELINE_HERE.parent.parent / "GENIUS" / "notebooks" / "outputs" / "all_cities_kept_classified_policies_final.csv"

CITY_CONFIG: dict[str, dict] = {
    "Austin": {
        "location_key": "Austin_US",
        "markdown":     _CITIES_DIR / "austin.md",
    },
    "Chicago": {
        "location_key": "Chicago_US",
        "markdown":     _CITIES_DIR / "chicago.md",
    },
    "Dakar": {
        "location_key": "Dakar_SN",
        "markdown":     _CITIES_DIR / "dakar.md",
    },
    "Geneva": {
        "location_key": "Geneva_CH",
        "markdown":     _CITIES_DIR / "geneva.md",
    },
    "Hiroshima": {
        "location_key": "Hiroshima_JP",
        "markdown":     _CITIES_DIR / "Hiroshima.md",
    },
    "Kuwait": {
        "location_key": "Kuwait_KW",
        "markdown":     _CITIES_DIR / "kuwait.md",
    },
    "Miami_Dade": {
        "location_key": "Miami_Dade_US",
        "markdown":     _CITIES_DIR / "miami_markdown.md",
    },
    "Portugal": {
        "location_key": "Portugal_PT",
        "markdown":     _CITIES_DIR / "Portugal.md",
    },
    "Seattle": {
        "location_key": "Seattle_US",
        "markdown":     _CITIES_DIR / "seattle_markdown.md",
    },
}


def load_ground_truth_for_city(city_name: str) -> list[dict[str, Any]]:
    """
    Load ground-truth policy rows for a single city from the master GT CSV.
    Filters by the ``city`` column and drops rows with empty policy_statement.
    """
    if city_name not in CITY_CONFIG:
        raise ValueError(f"Unknown city: {city_name!r}. Valid: {list(CITY_CONFIG)}")
    rows: list[dict[str, Any]] = []
    with open(_GT_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("city") == city_name and row.get("policy_statement", "").strip():
                rows.append(row)
    return rows


def load_ground_truth_all() -> dict[str, list[dict[str, Any]]]:
    """Return ground-truth policy rows grouped by city name for all cities in CITY_CONFIG."""
    gt: dict[str, list[dict[str, Any]]] = {c: [] for c in CITY_CONFIG}
    with open(_GT_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            city = row.get("city", "")
            if city in gt and row.get("policy_statement", "").strip():
                gt[city].append(row)
    return gt



# ---------------------------------------------------------------------------
# EvaluationHarness
# ---------------------------------------------------------------------------


class EvaluationHarness:
    """
    Orchestrates extraction + evaluation for one runner / location pair.

    Output layout
    -------------
    {output_dir}/{runner.model_slug}/{timestamp}/
        extracted_policies.csv   — raw extraction output
        scores.json              — full EvaluationOutput serialized to JSON
    """

    def __init__(
        self,
        output_dir: str | pathlib.Path = "evaluation_results",
        evaluator_model: str = "gpt-5.4",
        similarity_threshold: float = 0.55,
    ) -> None:
        self.output_dir = pathlib.Path(output_dir)
        self.evaluator_model = evaluator_model
        self.similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_run_dir(self, model_slug: str) -> pathlib.Path:
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = self.output_dir / model_slug / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def _save_extracted(run_dir: pathlib.Path, policies: list[dict[str, Any]]) -> None:
        if not policies:
            (run_dir / "extracted_policies.csv").write_text(
                "policy_statement\n", encoding="utf-8"
            )
            return
        path = run_dir / "extracted_policies.csv"
        fieldnames = list(policies[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(policies)

    @staticmethod
    def _save_scores(run_dir: pathlib.Path, result: EvaluationOutput) -> None:
        # Full detail — JSON
        with open(run_dir / "scores.json", "w", encoding="utf-8") as fh:
            json.dump(result.model_dump(), fh, indent=2, ensure_ascii=False)

        # Flat summary — scores.csv (one row, easy to concat across runs)
        from evaluator import CATEGORIES
        row: dict[str, Any] = {
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

        # Per-pair grades — grades.csv (one row per matched pair, includes reasoning)
        grade_rows = []
        for key, grade in result.grades.items():
            # Key format: {gt_primary_category}::gt{j}_ext{i}_{policy_id}
            category, rest = key.split("::", 1) if "::" in key else ("", key)
            grade_rows.append({
                "location":        result.location,
                "key":             key,
                "category":        category,
                "policy_id":       grade.policy_id,
                "grade":           grade.grade,
                "similarity":      round(grade.similarity, 4) if grade.similarity is not None else "",
                "statement_match": grade.statement_match if grade.statement_match is not None else "",
                "role_match":      grade.role_match if grade.role_match is not None else "",
                "category_match":  grade.category_match if grade.category_match is not None else "",
                "reasoning":       grade.reasoning,
            })

        with open(run_dir / "grades.csv", "w", newline="", encoding="utf-8") as fh:
            fieldnames = [
                "location", "key", "category", "policy_id", "grade",
                "similarity", "statement_match", "role_match", "category_match",
                "reasoning",
            ]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(grade_rows)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        runner: ModelRunner,
        location: str,
        document_path: str | pathlib.Path,
        ground_truth_policies: list[dict[str, Any]],
        rubric: str = DEFAULT_RUBRIC,
        system_prompt: str = CLIMATE_RLM_SYSTEM_PROMPT,
        source_document_path: str | pathlib.Path | None = None,
    ) -> EvaluationOutput:
        """
        Run extraction with ``runner``, evaluate, and write results.

        Args:
            runner:                  Any ModelRunner instance.
            location:                Location key, e.g. "Seattle_US".
            document_path:           Source document (PDF or markdown).
            ground_truth_policies:   List of GT policy dicts.
            rubric:                  Grading guidelines passed to LEAPEvaluator.
            system_prompt:           Extraction system prompt. Defaults to
                                     CLIMATE_RLM_SYSTEM_PROMPT.
            source_document_path:    Passed to LEAPEvaluator for RLM-graded pairs.
                                     Defaults to ``document_path``.

        Returns:
            EvaluationOutput written to the run directory.
        """
        doc_path = pathlib.Path(document_path)
        src_path = source_document_path or doc_path

        run_dir = self._make_run_dir(runner.model_slug)
        print(f"\n{'=' * 60}")
        print(f"Model    : {runner.model_slug}")
        print(f"Location : {location}")
        print(f"Document : {doc_path.name}")
        print(f"Output   : {run_dir}")
        print(f"{'=' * 60}")

        # Step 1: Parse document
        print("\n[1/3] Parsing document...")
        document_markdown = parse_document(str(doc_path))
        print(f"  {len(document_markdown):,} characters")

        # Step 2: Extract
        print(f"\n[2/3] Running extraction ({runner.model_slug})...")
        extracted = runner.run(document_markdown, system_prompt)
        print(f"  {len(extracted)} policies extracted")
        self._save_extracted(run_dir, extracted)

        # Step 3: Evaluate
        print("\n[3/3] Evaluating...")
        evaluator = LEAPEvaluator(
            model=self.evaluator_model,
            similarity_threshold=self.similarity_threshold,
        )
        result = evaluator.evaluate(
            location=location,
            extracted_policies=extracted,
            ground_truth_policies=ground_truth_policies,
            rubric=rubric,
            source_document_path=src_path,
        )
        self._save_scores(run_dir, result)

        # Print summary
        print(f"\n  composite_score        = {result.composite_score:.4f}")
        print(f"  extraction_f1          = {result.extraction_f1:.4f}  "
              f"(P={result.extraction_precision:.3f} R={result.extraction_recall:.3f})")
        print(f"  role_agreement         = {result.role_agreement:.4f}")
        print(f"  primary_cat_agreement  = {result.primary_category_agreement:.4f}")
        print(f"  plus_one_coverage      = {result.plus_one_coverage:.4f}")
        print(f"  matched={result.matched_count}  "
              f"unmatched_ext={result.unmatched_extracted_count}  "
              f"unmatched_gt={result.unmatched_ground_truth_count}")
        print(f"\n  Results saved to: {run_dir}")

        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LEAP evaluation harness — run any extraction backend and score it."
    )
    p.add_argument(
        "--backend",
        required=True,
        choices=list(RUNNER_CLASSES.keys()),
        help="Extraction backend to use.",
    )
    p.add_argument(
        "--model",
        required=True,
        help="Model name passed to the backend (e.g. gpt-5.4, claude-opus-4-6).",
    )
    # City selection: use --city for config-driven runs, or --document + --location
    # for one-off runs against a custom document.
    city_group = p.add_mutually_exclusive_group(required=True)
    city_group.add_argument(
        "--city",
        choices=list(CITY_CONFIG.keys()),
        help="Run against a configured city (resolves document and ground truth automatically).",
    )
    city_group.add_argument(
        "--all-cities",
        action="store_true",
        help="Run against all configured cities sequentially.",
    )
    city_group.add_argument(
        "--document",
        help="Path to the source document for a one-off run (requires --location and --ground-truth).",
    )
    p.add_argument(
        "--ground-truth",
        help="Ground-truth CSV path (only used with --document).",
    )
    p.add_argument(
        "--location",
        default="Unknown_Location",
        help="Location key (only used with --document).",
    )
    p.add_argument(
        "--output-dir",
        default=str(_PIPELINE_HERE / "evaluation_results"),
        help="Root directory for evaluation results.",
    )
    p.add_argument(
        "--evaluator-model",
        default="gpt-5.4",
        help="Model used by LEAPEvaluator for grading.",
    )
    p.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.55,
        help="Cosine similarity floor for Hungarian pair acceptance.",
    )
    p.add_argument(
        "--rlm-max-iterations",
        type=int,
        default=50,
        help="max_iterations for RLM backend only.",
    )
    p.add_argument(
        "--expert-knowledge",
        default=None,
        help="Path to grounding criteria PDF/txt (RLM backend only).",
    )
    return p


if __name__ == "__main__":
    # Smoke test — runs RLM and GPT against Hiroshima (14 GT policies, small doc).
    # Swap SMOKE_CITY to any key in CITY_CONFIG to test a different city.
    SMOKE_CITY   = "Hiroshima"
    MODEL        = "gpt-5.2"
    OUTPUT_DIR   = _PIPELINE_HERE / "evaluation_results"

    cfg = CITY_CONFIG[SMOKE_CITY]
    ground_truth = load_ground_truth_for_city(SMOKE_CITY)
    print(f"Loaded {len(ground_truth)} ground-truth policies for {SMOKE_CITY}")

    harness = EvaluationHarness(
        output_dir=OUTPUT_DIR,
        evaluator_model=MODEL,
        similarity_threshold=0.55,
    )

    # --- RLM pass ---
    rlm_runner = RLMRunner(
        model_name=MODEL,
        expert_knowledge_path=_DEFAULT_EXPERT_KNOWLEDGE_PATH,
        max_iterations=50,
    )
    rlm_result = harness.run(
        runner=rlm_runner,
        location=cfg["location_key"],
        document_path=cfg["markdown"],
        ground_truth_policies=ground_truth,
        source_document_path=cfg["markdown"],
    )

    # --- GPT direct pass ---
    gpt_runner = OpenAIRunner(
        model_name=MODEL,
        expert_knowledge_path=_DEFAULT_EXPERT_KNOWLEDGE_PATH,
    )
    gpt_result = harness.run(
        runner=gpt_runner,
        location=cfg["location_key"],
        document_path=cfg["markdown"],
        ground_truth_policies=ground_truth,
        source_document_path=cfg["markdown"],
    )

    # --- Side-by-side summary ---
    print(f"\n{'=' * 60}")
    print(f"Smoke test results — {SMOKE_CITY} ({len(ground_truth)} GT policies)")
    print(f"{'=' * 60}")
    print(f"{'Metric':<30} {'RLM':>10} {'GPT':>10}")
    print("-" * 52)
    for label, rv, gv in [
        ("composite_score",         rlm_result.composite_score,          gpt_result.composite_score),
        ("extraction_f1",           rlm_result.extraction_f1,            gpt_result.extraction_f1),
        ("extraction_precision",    rlm_result.extraction_precision,     gpt_result.extraction_precision),
        ("extraction_recall",       rlm_result.extraction_recall,        gpt_result.extraction_recall),
        ("role_agreement",          rlm_result.role_agreement,           gpt_result.role_agreement),
        ("primary_cat_agreement",   rlm_result.primary_category_agreement, gpt_result.primary_category_agreement),
        ("plus_one_coverage",       rlm_result.plus_one_coverage,        gpt_result.plus_one_coverage),
        ("matched",                 rlm_result.matched_count,            gpt_result.matched_count),
        ("unmatched_ext",           rlm_result.unmatched_extracted_count, gpt_result.unmatched_extracted_count),
        ("unmatched_gt",            rlm_result.unmatched_ground_truth_count, gpt_result.unmatched_ground_truth_count),
    ]:
        print(f"  {label:<28} {rv:>10.4f} {gv:>10.4f}")
    print(f"{'=' * 60}")
