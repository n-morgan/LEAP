import csv
import datetime
import json as _json
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI

from evaluator import CATEGORIES, EvaluationOutput, LEAPEvaluator
from rlm_pipeline import CLIMATE_RLM_SYSTEM_PROMPT, run_rlm_for_optimizer

load_dotenv()

# Section headers used to compose/decompose the structured prompt string
_SECTION_HEADERS: dict[str, str] = {
    "extraction":     "## Extraction",
    "hierarchy":      "## Hierarchy",
    "classification": "## Classification",
}


# ---------------------------------------------------------------------------
# StructuredPrompt
# ---------------------------------------------------------------------------


@dataclass
class StructuredPrompt:
    """
    Three-prong task-based decomposition of the RLM system prompt.

    Each prong targets one failure mode the evaluator can detect:
      extraction    — policy detection, field population, source quoting
      hierarchy     — parent/sub/individual role assignment
      classification — primary_category and related field assignment

    Internal template per prong: Role, Include, Exclude, Decision Rules, Edge Cases.
    """

    extraction: str = ""
    hierarchy: str = ""
    classification: str = ""

    def compose(self) -> str:
        """Assemble all three prongs into a single prompt string. Empty prongs are omitted."""
        parts: list[str] = []
        for key, header in _SECTION_HEADERS.items():
            content = getattr(self, key).strip()
            if content:
                parts.append(f"{header}\n{content}")
        return "\n\n".join(parts)

    @classmethod
    def decompose(cls, prompt: str) -> "StructuredPrompt":
        """
        Parse a composed prompt string back into three prongs by locating section
        headers. Falls back to placing the full text in extraction if no headers
        are found (e.g. legacy flat prompts).
        """
        positions: list[tuple[int, str]] = []
        for key, header in _SECTION_HEADERS.items():
            idx = prompt.find(header)
            if idx != -1:
                positions.append((idx, key))
        positions.sort()

        if not positions:
            return cls(extraction=prompt.strip())

        sections: dict[str, str] = {k: "" for k in _SECTION_HEADERS}
        for i, (pos, key) in enumerate(positions):
            start = pos + len(_SECTION_HEADERS[key])
            end = positions[i + 1][0] if i + 1 < len(positions) else len(prompt)
            sections[key] = prompt[start:end].strip()

        return cls(**sections)

    @classmethod
    def from_flat(cls, prompt: str) -> "StructuredPrompt":
        """
        Bootstrap from a flat prompt string. Places full content in extraction.
        Use migrate_flat_to_task_prongs() for a proper instruction redistribution.
        """
        return cls(extraction=prompt.strip())


# ---------------------------------------------------------------------------
# Resample prompts
# ---------------------------------------------------------------------------

_RESAMPLE_SYSTEM = """\
You are a prompt optimizer rewriting one section of a climate policy extraction
system prompt. The section you receive is one of three task-based prongs:

  extraction    — instructions for identifying policies, populating extraction
                  fields (role, parent_statement, policy_statement, source_quote,
                  section_header, extraction_rationale), deciding what counts as
                  a policy, and the overall extraction strategy.
  hierarchy     — instructions for assigning parent/sub/individual roles and
                  parent_statement, including hierarchy detection rules.
  classification — instructions for assigning primary_category, financial_instrument,
                   climate_relevance, and secondary_category, including the
                   decision table and edge cases.

SCORING OBJECTIVE:
Each extracted policy is matched 1-to-1 against ground-truth policies via
embedding similarity (Hungarian algorithm). Matched pairs are graded +1 / 0 / -1.
Unmatched ground-truth policies (missed extractions) each score -1.
Unmatched extracted policies (spurious extractions) each score -1.
The final score is the mean over all these values — so the prompt must balance
recall (extract everything real) against precision (avoid spurious policies).
Recall and FPR are provided alongside the score so you can diagnose the direction
of failure:
  - High FPR, low score  → too many spurious extractions; tighten criteria.
  - Low recall, low score → missing real policies; loosen criteria or add cues.
  - Low score, ok recall  → extractions are imprecise; improve quality guidance.

You will receive:
    CATEGORY  — the prong being rewritten (extraction / hierarchy / classification)
    METRICS   — current score, recall, and FPR
    SECTION   — the current text of the prong to rewrite
    FEEDBACK  — per-policy grade reasoning from the evaluation run

Your task: rewrite SECTION to improve the score. Fix the dominant failure mode
shown by METRICS and FEEDBACK. Preserve what works.
Do not add rules so strict that extraction is suppressed — a score of -1 from
zero extractions is no better than a score of -1 from bad ones.

Return ONLY the rewritten prong text. Do not include the section header.
No preamble, no explanation.
"""

_RESAMPLE_USER = """\
CATEGORY: {category}

METRICS:
  score={score:+.3f}  recall={recall:.3f}  fpr={fpr:.3f}

SECTION:
{section}

FEEDBACK:
{feedback}
"""


# ---------------------------------------------------------------------------
# LEAPPromptOptimizer
# ---------------------------------------------------------------------------


class LEAPPromptOptimizer:
    """
    LEAP Prompt Optimizer — task-based prong update loop.

    Decomposes the RLM system prompt into three task-based prongs
    (extraction, hierarchy, classification) and updates each independently
    based on per-task signals derived from EvaluationOutput.
    """

    def __init__(self, model: str = "gpt-5.4") -> None:
        self.model = model
        self._client: Optional[OpenAI] = None

    # ------------------------------------------------------------------
    # Client
    # ------------------------------------------------------------------

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    # ------------------------------------------------------------------
    # Resample
    # ------------------------------------------------------------------

    def _resample(
        self,
        section: str,
        feedback: str,
        category: str = "General",
        score: float = 0.0,
        recall: float = 0.0,
        fpr: float = 0.0,
    ) -> str:
        """
        Rewrite one prompt prong conditioned on grade reasoning feedback and
        current performance metrics. Returns the rewritten prong text (no header).
        """
        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _RESAMPLE_SYSTEM},
                {"role": "user", "content": _RESAMPLE_USER.format(
                    category=category,
                    score=score,
                    recall=recall,
                    fpr=fpr,
                    section=section or "(empty — write from scratch based on feedback)",
                    feedback=feedback or "No specific feedback available.",
                )},
            ],
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Feedback formatting
    # ------------------------------------------------------------------

    def _format_feedback(
        self,
        evaluation: EvaluationOutput,
        category: Optional[str] = None,
    ) -> str:
        """
        Collect grade reasoning from the evaluation output.
        If category is provided, filter to grades from that category only.
        """
        lines: list[str] = []
        for key, grade in evaluation.grades.items():
            cat, policy_id = key.split("::", 1) if "::" in key else ("", key)
            if category and cat != category:
                continue
            lines.append(
                f"[{cat}] {policy_id[:70]}\n"
                f"  grade={grade.grade:+d}  {grade.reasoning}"
            )
        return "\n".join(lines) if lines else "No grade reasoning available for this scope."

    # ------------------------------------------------------------------
    # Algorithm 2 — single location update step
    # ------------------------------------------------------------------

    def update(
        self,
        location: str,
        current_prompt: StructuredPrompt,
        previous_prompt: StructuredPrompt,
        current_eval: EvaluationOutput,
        previous_scores: dict[str, float],
    ) -> StructuredPrompt:
        """
        Simplified task-prong update using aggregate signals.

        Signals:
          extraction    — mean recall (drop below 0.5 → revert)
          hierarchy     — mean score delta (negative → revert)
          classification — mean score delta (negative → revert)

        Replaced by the full candidate-based loop in Task 4.
        """
        mean_score = (
            sum(current_eval.scores.values()) / len(current_eval.scores)
            if current_eval.scores else 0.0
        )
        mean_score_prev = (
            sum(previous_scores.values()) / len(previous_scores)
            if previous_scores else 0.0
        )
        mean_delta = mean_score - mean_score_prev
        mean_recall = (
            sum(current_eval.recall.values()) / len(current_eval.recall)
            if current_eval.recall else 0.0
        )
        mean_fpr = (
            sum(current_eval.fpr.values()) / len(current_eval.fpr)
            if current_eval.fpr else 0.0
        )
        feedback = self._format_feedback(current_eval)

        ext_base = (
            current_prompt.extraction if mean_recall >= 0.5 else previous_prompt.extraction
        )
        hier_base = (
            current_prompt.hierarchy if mean_delta >= 0 else previous_prompt.hierarchy
        )
        cls_base = (
            current_prompt.classification if mean_delta >= 0 else previous_prompt.classification
        )

        print(f"  [{location}] extraction      recall={mean_recall:.3f}   "
              f"({'keep' if mean_recall >= 0.5 else 'revert'})")
        print(f"  [{location}] hierarchy       delta={mean_delta:+.3f}    "
              f"({'keep' if mean_delta >= 0 else 'revert'})")
        print(f"  [{location}] classification  delta={mean_delta:+.3f}    "
              f"({'keep' if mean_delta >= 0 else 'revert'})")

        return StructuredPrompt(
            extraction=self._resample(
                ext_base, feedback, category="extraction",
                score=mean_score, recall=mean_recall, fpr=mean_fpr,
            ),
            hierarchy=self._resample(
                hier_base, feedback, category="hierarchy",
                score=mean_score, recall=mean_recall, fpr=mean_fpr,
            ),
            classification=self._resample(
                cls_base, feedback, category="classification",
                score=mean_score, recall=mean_recall, fpr=mean_fpr,
            ),
        )

    # ------------------------------------------------------------------
    # Algorithm 3 — optimization loop
    # ------------------------------------------------------------------

    def run_loop(
        self,
        location: str,
        extracted_policies_fn: Callable[[str, Optional[pathlib.Path]], list[dict[str, Any]]],
        ground_truth_policies: list[dict[str, Any]],
        rubric: str,
        initial_prompt: StructuredPrompt,
        source_document_path: Optional[pathlib.Path | str] = None,
        max_iterations: int = 10,
        epsilon: float = 0.01,
        log_dir: Optional[pathlib.Path | str] = None,
    ) -> StructuredPrompt:
        """
        Algorithm 3: LEAP Prompt Optimization Loop for a single location.

        Repeat:
            S_t, R_t, F_t, G_t  <- Evaluator(rho_t, D_l)
            rho_{t+1}            <- PromptOptimizer(rho_t, rho_{t-1}, S_t, S_{t-1}, ...)
        Until max(|S_t - S_{t-1}|) < epsilon or t >= T.

        Args:
            location:               location key (e.g. "Seattle_US")
            extracted_policies_fn:  callable that accepts a prompt string and returns
                                    extracted policy dicts (wraps the RLM run)
            ground_truth_policies:  GENIUS ground-truth policy dicts for this location
            rubric:                 freeform grading guidelines string
            initial_prompt:         starting StructuredPrompt (rho_0)
            source_document_path:   path to the markdown source document. When
                                    provided, grading is done via RLM so the full
                                    document can be traversed.
            max_iterations:         max optimization iterations T in [5, 10]
            epsilon:                convergence threshold on max per-category delta
            log_dir:                optional base directory for run logs. Each run
                                    creates a timestamped subdirectory containing:
                                      iteration_log.csv    — one row per iteration with
                                        per-category scores, deltas, recall, FPR, mean
                                        score, extracted count, and composed prompt.
                                      extracted_policies.csv — one row per extracted
                                        policy with iteration and location prepended.

        Returns:
            Optimized StructuredPrompt rho*.
        """
        evaluator = LEAPEvaluator(model=self.model)

        current_prompt = initial_prompt
        previous_prompt = initial_prompt
        previous_scores: dict[str, float] = {c: 0.0 for c in CATEGORIES}

        # CSV logging setup — create a timestamped subdirectory for this run
        _run_dir = None
        _iter_file = _iter_writer = None
        _ext_file = _ext_writer = None
        if log_dir is not None:
            ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            _run_dir = pathlib.Path(log_dir) / ts
            _run_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Logging to: {_run_dir}")

            # iteration_log.csv
            _cat_cols: list[str] = []
            for c in CATEGORIES:
                slug = c.replace(" ", "_").replace("-", "_")
                _cat_cols += [f"{slug}_score", f"{slug}_delta",
                              f"{slug}_recall", f"{slug}_fpr"]
            _iter_file = open(_run_dir / "iteration_log.csv", "w", newline="", encoding="utf-8")
            _iter_writer = csv.DictWriter(
                _iter_file,
                fieldnames=["iteration", "location", "n_extracted", "mean_score",
                            *_cat_cols, "converged", "composed_prompt"],
            )
            _iter_writer.writeheader()

            # extracted_policies.csv — headers derived from first extraction
            # writer is created lazily on first non-empty extraction

        def _close_logs() -> None:
            if _iter_file is not None:
                _iter_file.close()
            if _ext_file is not None:
                _ext_file.close()

        for t in range(max_iterations):
            print(f"\n{'=' * 60}")
            print(f"Iteration {t + 1}/{max_iterations}  |  Location: {location}")
            print(f"{'=' * 60}")

            # Run RLM extraction with the current composed prompt
            trace_path = (
                str(_run_dir / f"iteration_{t + 1}_trace")
                if _run_dir is not None else None
            )
            extracted = extracted_policies_fn(current_prompt.compose(), trace_path)
            print(f"  Extracted {len(extracted)} policies")

            # Write extracted policies to CSV (lazy header init on first batch)
            if _run_dir is not None and extracted:
                if _ext_writer is None:
                    _ext_file = open(
                        _run_dir / "extracted_policies.csv", "w",
                        newline="", encoding="utf-8",
                    )
                    _ext_writer = csv.DictWriter(
                        _ext_file,
                        fieldnames=["iteration", "location", *extracted[0].keys()],
                        extrasaction="ignore",
                    )
                    _ext_writer.writeheader()
                for policy in extracted:
                    _ext_writer.writerow({"iteration": t + 1, "location": location, **policy})
                _ext_file.flush()

            # Evaluate against ground truth
            eval_result = evaluator.evaluate(
                location=location,
                extracted_policies=extracted,
                ground_truth_policies=ground_truth_policies,
                rubric=rubric,
                source_document_path=source_document_path,
            )

            # Report scores
            print(f"\n  Scores (S_t[l, c]):")
            for cat, score in eval_result.scores.items():
                delta = score - previous_scores.get(cat, 0.0)
                recall = eval_result.recall.get(cat, 0.0)
                fpr = eval_result.fpr.get(cat, 0.0)
                print(
                    f"    {cat:<30s}  score={score:+.3f}  "
                    f"delta={delta:+.3f}  recall={recall:.2f}  fpr={fpr:.2f}"
                )

            # Write iteration log row
            if _iter_writer is not None:
                iter_row: dict[str, Any] = {
                    "iteration": t + 1,
                    "location": location,
                    "n_extracted": len(extracted),
                    "mean_score": round(
                        sum(eval_result.scores.values()) / len(eval_result.scores), 4
                    ) if eval_result.scores else 0.0,
                    "converged": False,
                    "composed_prompt": current_prompt.compose(),
                }
                for c in CATEGORIES:
                    slug = c.replace(" ", "_").replace("-", "_")
                    iter_row[f"{slug}_score"]  = round(eval_result.scores.get(c, 0.0), 4)
                    iter_row[f"{slug}_delta"]  = round(
                        eval_result.scores.get(c, 0.0) - previous_scores.get(c, 0.0), 4
                    )
                    iter_row[f"{slug}_recall"] = round(eval_result.recall.get(c, 0.0), 4)
                    iter_row[f"{slug}_fpr"]    = round(eval_result.fpr.get(c, 0.0), 4)

            # Convergence check (skip on first iteration — no previous scores yet)
            converged = False
            if t > 0:
                max_delta = max(
                    abs(eval_result.scores.get(c, 0.0) - previous_scores.get(c, 0.0))
                    for c in CATEGORIES
                )
                if max_delta < epsilon:
                    converged = True
                    print(f"\n  Converged: max_delta={max_delta:.4f} < epsilon={epsilon}")

            if _iter_writer is not None:
                iter_row["converged"] = converged
                _iter_writer.writerow(iter_row)
                _iter_file.flush()

            if converged:
                _close_logs()
                return current_prompt

            # Update prompt prongs
            print(f"\n  Updating prompt prongs:")
            next_prompt = self.update(
                location=location,
                current_prompt=current_prompt,
                previous_prompt=previous_prompt,
                current_eval=eval_result,
                previous_scores=previous_scores,
            )

            previous_scores = dict(eval_result.scores)
            previous_prompt = current_prompt
            current_prompt = next_prompt

        print(f"\n  Reached max iterations ({max_iterations}). Returning current prompt.")
        _close_logs()
        return current_prompt


# ---------------------------------------------------------------------------
# Legacy prompt migration utility
# ---------------------------------------------------------------------------

# Ordered list of section markers in CLIMATE_RLM_SYSTEM_PROMPT → target prong.
# Preamble (before first marker) is routed to extraction.
_FLAT_SECTION_TO_PRONG: list[tuple[str, str]] = [
    ("EXTRACTION FIELDS:", "extraction"),
    ("CLASSIFICATION FIELDS", "classification"),
    ("WHAT COUNTS AS A POLICY:", "extraction"),
    ("HIERARCHY RULES:", "hierarchy"),
    ("STRATEGY:", "extraction"),
    ("OUTPUT FORMAT:", "extraction"),
]


def migrate_flat_to_task_prongs(flat_prompt: str) -> StructuredPrompt:
    """
    Rule-based redistribution of the legacy flat CLIMATE_RLM_SYSTEM_PROMPT
    into three task-based prongs.

    Mapping:
      extraction    — preamble + EXTRACTION FIELDS + WHAT COUNTS AS A POLICY
                      + STRATEGY + OUTPUT FORMAT
      hierarchy     — HIERARCHY RULES
      classification — CLASSIFICATION FIELDS
    """
    found: list[tuple[int, str, str]] = []
    for marker, prong in _FLAT_SECTION_TO_PRONG:
        idx = flat_prompt.find(marker)
        if idx != -1:
            found.append((idx, marker, prong))
    found.sort()

    if not found:
        return StructuredPrompt(extraction=flat_prompt.strip())

    chunks: dict[str, list[str]] = {
        "extraction": [], "hierarchy": [], "classification": []
    }

    preamble = flat_prompt[: found[0][0]].strip()
    if preamble:
        chunks["extraction"].append(preamble)

    for i, (pos, _marker, prong) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(flat_prompt)
        chunks[prong].append(flat_prompt[pos:end].strip())

    return StructuredPrompt(
        extraction="\n\n".join(chunks["extraction"]),
        hierarchy="\n\n".join(chunks["hierarchy"]),
        classification="\n\n".join(chunks["classification"]),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import csv
    import pathlib

    _HERE = pathlib.Path(__file__).resolve().parent       # LEAP/optimization/
    _LEAP_ROOT = _HERE.parent                              # LEAP/
    _GENIUS_DOCS = _LEAP_ROOT.parent / "GENIUS" / "docs" / "cities"

    OUTPUTS_DIR = _HERE / "organized_outputs"
    SEATTLE_DOC = str(_GENIUS_DOCS / "seattle_markdown.md")

    # Single source of truth for model and RLM settings — used by both the
    # optimizer (grader/resampler) and the extraction closure below.
    MODEL = "gpt-5.2"
    RLM_MAX_ITERATIONS = 30

    def load_policies(path: pathlib.Path) -> list[dict]:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [row for row in reader if row.get("policy_statement", "").strip()]

    ground_truth_policies = load_policies(OUTPUTS_DIR / "structured_policies.csv")
    print(f"Loaded {len(ground_truth_policies)} ground-truth policies.")

    # Live RLM extraction: the optimizer calls this with the current composed
    # prompt at each iteration, re-runs the RLM, and scores the new output.
    # This is what closes the loop — each iteration uses a fresh RLM run under
    # the candidate prompt rather than a stale cached result.
    def extracted_policies_fn(prompt: str, trace_dir: str | None = None) -> list[dict]:
        return run_rlm_for_optimizer(
            prompt_string=prompt,
            document_path=SEATTLE_DOC,
            trace_dir=trace_dir,
            model_name=MODEL,
            sub_model_name=MODEL,
            max_iterations=RLM_MAX_ITERATIONS,
        )

    DEFAULT_RUBRIC = (
        "Grade on specificity (quantified targets, deadlines, mechanisms), "
        "commitment strength (binding vs aspirational language), "
        "and accuracy relative to the source document."
    )

    # Redistribute the flat system prompt into the three task-based prongs so
    # optimization starts from a structured baseline rather than a blank slate.
    initial_prompt = migrate_flat_to_task_prongs(CLIMATE_RLM_SYSTEM_PROMPT)

    optimizer = LEAPPromptOptimizer(model=MODEL)
    optimized = optimizer.run_loop(
        location="Seattle_US",
        extracted_policies_fn=extracted_policies_fn,
        ground_truth_policies=ground_truth_policies,
        rubric=DEFAULT_RUBRIC,
        initial_prompt=initial_prompt,
        source_document_path=SEATTLE_DOC,
        max_iterations=3,
        log_dir=_HERE / "logs",
    )

    print("\n=== Optimized Prompt ===")
    print(optimized.compose())
