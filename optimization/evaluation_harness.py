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
import re
import tempfile
from typing import Any, Optional, Protocol, runtime_checkable

from dotenv import load_dotenv

from evaluator import LEAPEvaluator, EvaluationOutput
from rlm_pipeline import (
    CLIMATE_RLM_SYSTEM_PROMPT,
    _parse_rlm_output,
    parse_document,
    run_rlm_for_optimizer,
    _DEFAULT_EXPERT_KNOWLEDGE_PATH,
    _HERE as _PIPELINE_HERE,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Default rubric (mirrors prompt_optimizer.__main__)
# ---------------------------------------------------------------------------

DEFAULT_RUBRIC = (
    "Grade on specificity (quantified targets, deadlines, mechanisms), "
    "commitment strength (binding vs aspirational language), "
    "and accuracy relative to the source document."
)

# ---------------------------------------------------------------------------
# City config — maps GT CSV city name → (location_key, markdown_filename)
#
# Source of truth: GENIUS/notebooks/outputs/all_cities_kept_classified_policies_final.csv
# Markdown files:  GENIUS/docs/cities/
# Las Vegas (LV.md) is excluded — no ground truth rows in the CSV.
# ---------------------------------------------------------------------------

_GENIUS_CITIES_DIR = _PIPELINE_HERE.parent.parent / "GENIUS" / "docs" / "cities"
_GT_CSV = _PIPELINE_HERE.parent.parent / "GENIUS" / "notebooks" / "outputs" / "all_cities_kept_classified_policies_final.csv"

CITY_CONFIG: dict[str, dict] = {
    "Austin": {
        "location_key": "Austin_US",
        "markdown":     _GENIUS_CITIES_DIR / "austin.md",
    },
    "Chicago": {
        "location_key": "Chicago_US",
        "markdown":     _GENIUS_CITIES_DIR / "chicago.md",
    },
    "Dakar": {
        "location_key": "Dakar_SN",
        "markdown":     _GENIUS_CITIES_DIR / "dakar.md",
    },
    "Geneva": {
        "location_key": "Geneva_CH",
        "markdown":     _GENIUS_CITIES_DIR / "geneva.md",
    },
    "Hiroshima": {
        "location_key": "Hiroshima_JP",
        "markdown":     _GENIUS_CITIES_DIR / "Hiroshima.md",
    },
    "Kuwait": {
        "location_key": "Kuwait_KW",
        "markdown":     _GENIUS_CITIES_DIR / "kuwait.md",
    },
    "Miami_Dade": {
        "location_key": "Miami_Dade_US",
        "markdown":     _GENIUS_CITIES_DIR / "miami_markdown.md",
    },
    "Portugal": {
        "location_key": "Portugal_PT",
        "markdown":     _GENIUS_CITIES_DIR / "Portugal.md",
    },
    "Seattle": {
        "location_key": "Seattle_US",
        "markdown":     _GENIUS_CITIES_DIR / "seattle_markdown.md",
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
# Runner protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelRunner(Protocol):
    """Any object with a .run() method can be used as a runner."""

    @property
    def model_slug(self) -> str:
        """Filesystem-safe identifier used for the output folder name."""
        ...

    def run(
        self,
        document_markdown: str,
        system_prompt: str,
    ) -> list[dict[str, Any]]:
        """
        Run extraction on ``document_markdown`` using ``system_prompt``.
        Returns a list of raw policy dicts (no DSPy validation).
        """
        ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _slugify(s: str) -> str:
    """Replace non-alphanumeric characters with hyphens for safe folder names."""
    return re.sub(r"[^a-zA-Z0-9._-]", "-", s).strip("-")


def _flat_prompt(system_prompt: str, document_markdown: str) -> str:
    """Single-turn user message: system prompt header + document."""
    return (
        f"{system_prompt}\n\n"
        f"DOCUMENT:\n{document_markdown}\n\n"
        "Extract and classify all climate policies from the document as a JSON list."
    )


# ---------------------------------------------------------------------------
# RLM runner
# ---------------------------------------------------------------------------


class RLMRunner:
    """
    RLM recursive pipeline (run_rlm_for_optimizer from rlm_pipeline.py).

    Supports expert_knowledge_path for grounding criteria injection.
    """

    def __init__(
        self,
        model_name: str = "gpt-5.4",
        sub_model_name: str | None = None,
        expert_knowledge_path: str | None = None,
        max_iterations: int = 50,
        trace_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.sub_model_name = sub_model_name or model_name
        self.expert_knowledge_path = (
            expert_knowledge_path or _DEFAULT_EXPERT_KNOWLEDGE_PATH
        )
        self.max_iterations = max_iterations
        self.trace_dir = trace_dir

    @property
    def model_slug(self) -> str:
        return _slugify(f"rlm_{self.model_name}")

    def run(self, document_markdown: str, system_prompt: str) -> list[dict[str, Any]]:
        # run_rlm_for_optimizer takes a document path, not markdown directly.
        # Write to a temp file so the existing API is satisfied.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(document_markdown)
            tmp_path = tmp.name
        try:
            return run_rlm_for_optimizer(
                prompt_string=system_prompt,
                document_path=tmp_path,
                trace_dir=self.trace_dir,
                expert_knowledge_path=self.expert_knowledge_path,
                model_name=self.model_name,
                sub_model_name=self.sub_model_name,
                max_iterations=self.max_iterations,
            )
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# OpenAI runner
# ---------------------------------------------------------------------------


class OpenAIRunner:
    """Direct OpenAI Chat Completions extraction (no recursion)."""

    def __init__(
        self,
        model_name: str = "gpt-5.4",
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self._client = None

    @property
    def model_slug(self) -> str:
        return _slugify(f"openai_{self.model_name}")

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def run(self, document_markdown: str, system_prompt: str) -> list[dict[str, Any]]:
        response = self._get_client().chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"DOCUMENT:\n{document_markdown}\n\n"
                        "Extract and classify all climate policies from the document "
                        "as a JSON list."
                    ),
                },
            ],
        )
        raw = response.choices[0].message.content or ""
        return _parse_rlm_output(raw)


# ---------------------------------------------------------------------------
# Anthropic runner
# ---------------------------------------------------------------------------


class AnthropicRunner:
    """Anthropic Messages API extraction."""

    def __init__(
        self,
        model_name: str = "claude-opus-4-6",
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def model_slug(self) -> str:
        return _slugify(f"anthropic_{self.model_name}")

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._client

    def run(self, document_markdown: str, system_prompt: str) -> list[dict[str, Any]]:
        response = self._get_client().messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"DOCUMENT:\n{document_markdown}\n\n"
                        "Extract and classify all climate policies from the document "
                        "as a JSON list."
                    ),
                }
            ],
        )
        raw = response.content[0].text if response.content else ""
        return _parse_rlm_output(raw)


# ---------------------------------------------------------------------------
# Gemini runner
# ---------------------------------------------------------------------------


class GeminiRunner:
    """Google Generative AI extraction."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature

    @property
    def model_slug(self) -> str:
        return _slugify(f"gemini_{self.model_name}")

    def run(self, document_markdown: str, system_prompt: str) -> list[dict[str, Any]]:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(temperature=self.temperature),
        )
        prompt = (
            f"DOCUMENT:\n{document_markdown}\n\n"
            "Extract and classify all climate policies from the document "
            "as a JSON list."
        )
        response = model.generate_content(prompt)
        raw = response.text or ""
        return _parse_rlm_output(raw)


# ---------------------------------------------------------------------------
# Runner registry
# ---------------------------------------------------------------------------

RUNNER_CLASSES: dict[str, type] = {
    "rlm": RLMRunner,
    "openai": OpenAIRunner,
    "anthropic": AnthropicRunner,
    "gemini": GeminiRunner,
}


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
        path = run_dir / "scores.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result.model_dump(), fh, indent=2, ensure_ascii=False)

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
    MODEL        = "gpt-5.4"
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
    gpt_runner = OpenAIRunner(model_name=MODEL)
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
