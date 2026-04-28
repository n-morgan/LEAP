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

FIXED GROUNDING CRITERIA:
The RLM always receives a fixed expert extraction criteria document alongside the
source policy document. This grounding criteria is provided to you under
GROUNDING CRITERIA below. You cannot change it and must not duplicate it.
Your rewrite should complement the grounding criteria — add specificity where it
is silent, resolve ambiguities it leaves open, and avoid restating rules it
already covers clearly.

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

GROUNDING CRITERIA (fixed, read-only — do not restate or contradict):
{grounding_criteria}

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
        grounding_criteria: str = "",
    ) -> str:
        """
        Rewrite one prompt prong conditioned on grade reasoning feedback,
        current performance metrics, and the fixed grounding criteria document.
        Returns the rewritten prong text (no header).
        grounding_criteria is shown to the resampler as a read-only fixed input
        so it does not duplicate or contradict rules already covered by that doc.
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
                    grounding_criteria=grounding_criteria or "(none provided)",
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
        grounding_criteria: str = "",
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
                grounding_criteria=grounding_criteria,
            ),
            hierarchy=self._resample(
                hier_base, feedback, category="hierarchy",
                score=mean_score, recall=mean_recall, fpr=mean_fpr,
                grounding_criteria=grounding_criteria,
            ),
            classification=self._resample(
                cls_base, feedback, category="classification",
                score=mean_score, recall=mean_recall, fpr=mean_fpr,
                grounding_criteria=grounding_criteria,
            ),
        )

    # ------------------------------------------------------------------
    # Candidate generation, evaluation, and acceptance (Task 3)
    # ------------------------------------------------------------------

    def propose_candidate(
        self,
        scope: Literal["extraction", "hierarchy", "classification"],
        current_prompt: StructuredPrompt,
        previous_prompt: StructuredPrompt,
        current_eval: EvaluationOutput,
        previous_eval: EvaluationOutput,
        grounding_criteria: str = "",
    ) -> StructuredPrompt:
        """
        Rewrite only the specified prong and return a new StructuredPrompt.
        The other two prongs are copied from current_prompt unchanged.

        Signal per prong:
          extraction    — mean recall (drop → revert to previous prong as base)
          hierarchy     — hierarchy_accuracy (drop → revert)
          classification — mean score (drop → revert)

        Negative signal guard: if the signal worsened since the previous
        iteration, use the previous prong as the rewrite base before resampling.
        """
        mean_score = (
            sum(current_eval.scores.values()) / len(current_eval.scores)
            if current_eval.scores else 0.0
        )
        mean_recall = (
            sum(current_eval.recall.values()) / len(current_eval.recall)
            if current_eval.recall else 0.0
        )
        mean_fpr = (
            sum(current_eval.fpr.values()) / len(current_eval.fpr)
            if current_eval.fpr else 0.0
        )
        feedback = self._format_feedback(current_eval)

        if scope == "extraction":
            prev_recall = (
                sum(previous_eval.recall.values()) / len(previous_eval.recall)
                if previous_eval.recall else 0.0
            )
            base = (
                previous_prompt.extraction if mean_recall < prev_recall
                else current_prompt.extraction
            )
            resampled = self._resample(
                base, feedback, category="extraction",
                score=mean_score, recall=mean_recall, fpr=mean_fpr,
                grounding_criteria=grounding_criteria,
            )
            return StructuredPrompt(
                extraction=resampled,
                hierarchy=current_prompt.hierarchy,
                classification=current_prompt.classification,
            )

        if scope == "hierarchy":
            base = (
                previous_prompt.hierarchy
                if current_eval.hierarchy_accuracy < previous_eval.hierarchy_accuracy
                else current_prompt.hierarchy
            )
            resampled = self._resample(
                base, feedback, category="hierarchy",
                score=mean_score, recall=mean_recall, fpr=mean_fpr,
                grounding_criteria=grounding_criteria,
            )
            return StructuredPrompt(
                extraction=current_prompt.extraction,
                hierarchy=resampled,
                classification=current_prompt.classification,
            )

        # scope == "classification"
        prev_score = (
            sum(previous_eval.scores.values()) / len(previous_eval.scores)
            if previous_eval.scores else 0.0
        )
        base = (
            previous_prompt.classification if mean_score < prev_score
            else current_prompt.classification
        )
        resampled = self._resample(
            base, feedback, category="classification",
            score=mean_score, recall=mean_recall, fpr=mean_fpr,
            grounding_criteria=grounding_criteria,
        )
        return StructuredPrompt(
            extraction=current_prompt.extraction,
            hierarchy=current_prompt.hierarchy,
            classification=resampled,
        )

    def evaluate_candidate(
        self,
        candidate_prompt: StructuredPrompt,
        location: str,
        extracted_policies_fn: Callable[[str, Optional[pathlib.Path]], list[dict[str, Any]]],
        ground_truth_policies: list[dict[str, Any]],
        rubric: str,
        evaluator: "LEAPEvaluator",
        source_document_path: Optional[pathlib.Path | str] = None,
        trace_path: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], EvaluationOutput]:
        """
        Run a full extraction and evaluation pass using the candidate prompt.
        Returns (extracted_policies, evaluation_output).
        """
        extracted = extracted_policies_fn(candidate_prompt.compose(), trace_path)
        eval_result = evaluator.evaluate(
            location=location,
            extracted_policies=extracted,
            ground_truth_policies=ground_truth_policies,
            rubric=rubric,
            source_document_path=source_document_path,
        )
        return extracted, eval_result

    def accept_candidate(
        self,
        candidate_eval: EvaluationOutput,
        current_eval: EvaluationOutput,
        score_floor: float = 0.15,
    ) -> tuple[bool, str]:
        """
        Apply guardrail thresholds to decide whether to accept the candidate.

        Rejects if any per-category score drops by more than score_floor
        compared to the current accepted evaluation.

        Returns (accepted, reason_string).
        """
        for cat in CATEGORIES:
            curr_score = current_eval.scores.get(cat, 0.0)
            cand_score = candidate_eval.scores.get(cat, 0.0)
            drop = curr_score - cand_score
            if drop >= score_floor:
                return (
                    False,
                    f"rejected: {cat} score dropped by {drop:.3f} "
                    f"(threshold {score_floor})",
                )

        mean_curr = (
            sum(current_eval.scores.values()) / len(current_eval.scores)
            if current_eval.scores else 0.0
        )
        mean_cand = (
            sum(candidate_eval.scores.values()) / len(candidate_eval.scores)
            if candidate_eval.scores else 0.0
        )
        return True, f"accepted: mean score {mean_curr:+.3f} → {mean_cand:+.3f}"

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
        grounding_criteria_path: Optional[pathlib.Path | str] = None,
        max_iterations: int = 10,
        epsilon: float = 0.01,
        log_dir: Optional[pathlib.Path | str] = None,
    ) -> StructuredPrompt:
        """
        LEAP optimization loop — new per-prong candidate contract.

        Each iteration:
          1. Evaluate the current accepted prompt.
          2. Compute three task signals from EvaluationOutput.
          3. For each prong whose signal worsened (or on iteration 1, all prongs),
             propose one candidate by rewriting only that prong.
          4. Reevaluate each candidate independently.
          5. Apply guardrails; accept at most the single best passing candidate.
          6. If no candidate passes, keep the current prompt unchanged.
          7. Write iteration and candidate rows to the log.

        Logs (under a timestamped subdirectory of log_dir):
          iteration_log.csv  — one row per main evaluation
          candidate_log.csv  — one row per candidate with scope/accepted/reason
          extracted_policies.csv — one row per extracted policy

        Args:
            location:               location key (e.g. "Seattle_US")
            extracted_policies_fn:  callable(prompt_string, trace_path) → list[dict]
            ground_truth_policies:  GENIUS ground-truth policy dicts for this location
            rubric:                 freeform grading guidelines string
            initial_prompt:         starting StructuredPrompt (rho_0)
            source_document_path:   path to the markdown source document
            max_iterations:         maximum optimization iterations
            epsilon:                convergence threshold on max per-category delta
            log_dir:                base directory for run logs

        Returns:
            Optimized StructuredPrompt rho*.
        """
        evaluator = LEAPEvaluator(model=self.model)

        # Load grounding criteria once — shown to the resampler every iteration
        # as a read-only fixed input it must complement but cannot change.
        grounding_criteria: str = ""
        if grounding_criteria_path is not None:
            gc_path = pathlib.Path(grounding_criteria_path)
            if gc_path.exists():
                grounding_criteria = gc_path.read_text(encoding="utf-8")
                print(f"  Grounding criteria loaded: {gc_path} ({len(grounding_criteria):,} chars)")
            else:
                print(f"  [WARN] grounding_criteria_path not found: {gc_path} — proceeding without it")

        current_prompt = initial_prompt
        previous_prompt = initial_prompt
        previous_eval: Optional[EvaluationOutput] = None

        # ------------------------------------------------------------------
        # CSV logging setup
        # ------------------------------------------------------------------
        _run_dir = None
        _iter_file = _iter_writer = None
        _cand_file = _cand_writer = None
        _ext_file = _ext_writer = None

        if log_dir is not None:
            ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            _run_dir = pathlib.Path(log_dir) / ts
            _run_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Logging to: {_run_dir}")

            _cat_cols: list[str] = []
            for c in CATEGORIES:
                slug = c.replace(" ", "_").replace("-", "_")
                _cat_cols += [f"{slug}_score", f"{slug}_delta",
                              f"{slug}_recall", f"{slug}_fpr"]

            _iter_file = open(_run_dir / "iteration_log.csv", "w", newline="", encoding="utf-8")
            _iter_writer = csv.DictWriter(
                _iter_file,
                fieldnames=["iteration", "location", "n_extracted", "mean_score",
                            "hierarchy_accuracy", *_cat_cols, "converged",
                            "composed_prompt"],
            )
            _iter_writer.writeheader()

            _cand_file = open(_run_dir / "candidate_log.csv", "w", newline="", encoding="utf-8")
            _cand_writer = csv.DictWriter(
                _cand_file,
                fieldnames=["iteration", "location", "scope", "accepted", "reason",
                            "mean_score", "hierarchy_accuracy"],
            )
            _cand_writer.writeheader()

        def _close_logs() -> None:
            for f in (_iter_file, _cand_file, _ext_file):
                if f is not None:
                    f.close()

        # ------------------------------------------------------------------
        # Iteration loop
        # ------------------------------------------------------------------
        for t in range(max_iterations):
            print(f"\n{'=' * 60}")
            print(f"Iteration {t + 1}/{max_iterations}  |  Location: {location}")
            print(f"{'=' * 60}")

            # Step 1: Extract with current prompt
            trace_path = (
                str(_run_dir / f"iteration_{t + 1}_trace")
                if _run_dir is not None else None
            )
            extracted = extracted_policies_fn(current_prompt.compose(), trace_path)
            print(f"  Extracted {len(extracted)} policies")

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

            # Step 2: Evaluate current prompt
            eval_result = evaluator.evaluate(
                location=location,
                extracted_policies=extracted,
                ground_truth_policies=ground_truth_policies,
                rubric=rubric,
                source_document_path=source_document_path,
            )

            mean_score = (
                sum(eval_result.scores.values()) / len(eval_result.scores)
                if eval_result.scores else 0.0
            )
            mean_recall = (
                sum(eval_result.recall.values()) / len(eval_result.recall)
                if eval_result.recall else 0.0
            )
            hier_acc = eval_result.hierarchy_accuracy

            print(f"\n  Scores:")
            prev_scores = previous_eval.scores if previous_eval else {c: 0.0 for c in CATEGORIES}
            for cat, score in eval_result.scores.items():
                delta = score - prev_scores.get(cat, 0.0)
                print(
                    f"    {cat:<30s}  score={score:+.3f}  delta={delta:+.3f}"
                    f"  recall={eval_result.recall.get(cat, 0.0):.2f}"
                    f"  fpr={eval_result.fpr.get(cat, 0.0):.2f}"
                )
            print(f"    hierarchy_accuracy={hier_acc:.3f}")

            # Write iteration log row
            if _iter_writer is not None:
                iter_row: dict[str, Any] = {
                    "iteration": t + 1,
                    "location": location,
                    "n_extracted": len(extracted),
                    "mean_score": round(mean_score, 4),
                    "hierarchy_accuracy": round(hier_acc, 4),
                    "converged": False,
                    "composed_prompt": current_prompt.compose(),
                }
                for c in CATEGORIES:
                    slug = c.replace(" ", "_").replace("-", "_")
                    iter_row[f"{slug}_score"]  = round(eval_result.scores.get(c, 0.0), 4)
                    iter_row[f"{slug}_delta"]  = round(
                        eval_result.scores.get(c, 0.0) - prev_scores.get(c, 0.0), 4
                    )
                    iter_row[f"{slug}_recall"] = round(eval_result.recall.get(c, 0.0), 4)
                    iter_row[f"{slug}_fpr"]    = round(eval_result.fpr.get(c, 0.0), 4)

            # Convergence check (skip iteration 1 — no previous eval yet)
            converged = False
            if previous_eval is not None:
                max_delta = max(
                    abs(eval_result.scores.get(c, 0.0) - previous_eval.scores.get(c, 0.0))
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

            # Step 3: Determine triggered prongs
            if previous_eval is None:
                # First iteration — bootstrap all prongs
                triggered: set[str] = {"extraction", "hierarchy", "classification"}
            else:
                prev_recall = (
                    sum(previous_eval.recall.values()) / len(previous_eval.recall)
                    if previous_eval.recall else 0.0
                )
                prev_mean_score = (
                    sum(previous_eval.scores.values()) / len(previous_eval.scores)
                    if previous_eval.scores else 0.0
                )
                triggered = set()
                if mean_recall < prev_recall:
                    triggered.add("extraction")
                if hier_acc < previous_eval.hierarchy_accuracy:
                    triggered.add("hierarchy")
                if mean_score < prev_mean_score:
                    triggered.add("classification")

            # Step 4: Generate and evaluate one candidate per triggered prong
            passing: list[tuple[str, StructuredPrompt, EvaluationOutput]] = []
            print(f"\n  Triggered prongs: {triggered or 'none'}")

            for scope in ("extraction", "hierarchy", "classification"):
                if scope not in triggered:
                    continue

                print(f"\n  Proposing candidate for prong: {scope}")
                candidate_prompt = self.propose_candidate(
                    scope=scope,
                    current_prompt=current_prompt,
                    previous_prompt=previous_prompt,
                    current_eval=eval_result,
                    previous_eval=previous_eval or eval_result,
                    grounding_criteria=grounding_criteria,
                )

                cand_trace = (
                    str(_run_dir / f"iteration_{t + 1}_candidate_{scope}_trace")
                    if _run_dir is not None else None
                )
                _, cand_eval = self.evaluate_candidate(
                    candidate_prompt, location, extracted_policies_fn,
                    ground_truth_policies, rubric, evaluator,
                    source_document_path, cand_trace,
                )

                accepted, reason = self.accept_candidate(cand_eval, eval_result)
                cand_mean = (
                    sum(cand_eval.scores.values()) / len(cand_eval.scores)
                    if cand_eval.scores else 0.0
                )
                print(f"    {reason}")

                if _cand_writer is not None:
                    _cand_writer.writerow({
                        "iteration": t + 1,
                        "location": location,
                        "scope": scope,
                        "accepted": accepted,
                        "reason": reason,
                        "mean_score": round(cand_mean, 4),
                        "hierarchy_accuracy": round(cand_eval.hierarchy_accuracy, 4),
                    })
                    _cand_file.flush()

                if accepted:
                    passing.append((scope, candidate_prompt, cand_eval))

            # Step 5: Accept the single best passing candidate (highest mean score)
            if passing:
                best_scope, best_prompt, best_eval = max(
                    passing,
                    key=lambda x: sum(x[2].scores.values()) / len(x[2].scores),
                )
                print(f"\n  Accepted candidate: {best_scope}")
                previous_prompt = current_prompt
                current_prompt = best_prompt
                eval_result = best_eval
            else:
                print(f"\n  No candidates accepted — keeping current prompt")
                previous_prompt = current_prompt

            previous_eval = eval_result

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
# Full LLM-based migration utility (Task 5)
# ---------------------------------------------------------------------------

_MIGRATION_SYSTEM = """\
You are an expert prompt engineer. Your task is to redistribute a flat climate
policy extraction system prompt into three task-based prongs:

  extraction    — instructions for identifying policies, populating extraction
                  fields (role, parent_statement, policy_statement, source_quote,
                  section_header, extraction_rationale), deciding what counts as
                  a policy, and the overall extraction strategy and output format.
  hierarchy     — instructions for assigning parent/sub/individual roles and
                  parent_statement, including hierarchy detection rules and
                  edge cases for role assignment.
  classification — instructions for assigning primary_category, financial_instrument,
                   climate_relevance, and secondary_category, including the full
                   decision table, triggers, and common errors to avoid.

Rules:
1. Every instruction in the flat prompt must appear in exactly one prong.
2. Do NOT duplicate instructions across prongs.
3. Instructions that are genuinely ambiguous (could fit two prongs equally well)
   must be placed in one chosen prong with an inline note formatted as:
   <!-- NOTE: also relevant to <other_prong> because ... -->
4. Cross-cutting preamble (e.g. "You are a climate policy analyst...") goes in
   extraction.
5. Return a JSON object with these keys:
     "extraction"    — string, full text of the extraction prong
     "hierarchy"     — string, full text of the hierarchy prong
     "classification" — string, full text of the classification prong
     "mapping_notes" — list of strings, one per legacy section describing
                       which prong it went to and why
     "warnings"      — list of strings, one per ambiguous instruction with the
                       placement decision and the reason

Return ONLY the JSON object. No preamble. No trailing text.
"""

_MIGRATION_USER = """\
FLAT PROMPT:
{flat_prompt}
"""


def migrate_to_task_prongs_with_notes(
    flat_prompt: str,
    model: str = "gpt-5.4",
    output_path: Optional[pathlib.Path | str] = None,
) -> tuple[StructuredPrompt, list[str], list[str]]:
    """
    LLM-based redistribution of the legacy flat prompt into three task-based
    prongs, with per-section mapping notes and a warning list for ambiguous
    instructions.

    Args:
        flat_prompt:  the flat system prompt to migrate (e.g. CLIMATE_RLM_SYSTEM_PROMPT)
        model:        OpenAI model to use for redistribution
        output_path:  if provided, saves the result as a JSON file at this path;
                      the file can be loaded later with load_migrated_baseline()

    Returns:
        (StructuredPrompt, mapping_notes, warnings)
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _MIGRATION_SYSTEM},
            {"role": "user", "content": _MIGRATION_USER.format(flat_prompt=flat_prompt)},
        ],
        response_format={"type": "json_object"},
    )
    data = _json.loads(response.choices[0].message.content)

    prompt = StructuredPrompt(
        extraction=data.get("extraction", ""),
        hierarchy=data.get("hierarchy", ""),
        classification=data.get("classification", ""),
    )
    mapping_notes: list[str] = data.get("mapping_notes", [])
    warnings: list[str] = data.get("warnings", [])

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            _json.dump(
                {
                    "extraction": prompt.extraction,
                    "hierarchy": prompt.hierarchy,
                    "classification": prompt.classification,
                    "mapping_notes": mapping_notes,
                    "warnings": warnings,
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Migrated baseline saved to: {output_path}")
        if warnings:
            print(f"  {len(warnings)} ambiguous instruction(s):")
            for w in warnings:
                print(f"    - {w}")

    return prompt, mapping_notes, warnings


def load_migrated_baseline(path: pathlib.Path | str) -> StructuredPrompt:
    """Load a StructuredPrompt previously saved by migrate_to_task_prongs_with_notes."""
    with open(path, encoding="utf-8") as fh:
        data = _json.load(fh)
    return StructuredPrompt(
        extraction=data["extraction"],
        hierarchy=data["hierarchy"],
        classification=data["classification"],
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import csv
    import pathlib

    _HERE = pathlib.Path(__file__).resolve().parent       # LEAP/optimization/
    _LEAP_ROOT = _HERE.parent                              # LEAP/

    OUTPUTS_DIR = _HERE / "organized_outputs"
    SEATTLE_DOC = str(_HERE / "docs" / "cities" / "seattle_markdown.md")

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

    # Load or generate the three-prong baseline.
    # If a previously migrated baseline exists on disk, load it so the LLM
    # redistribution is not repeated on every run. Otherwise generate it once
    # and save it for future runs.
    _BASELINE_PATH = _HERE / "migrated_baseline.json"
    if _BASELINE_PATH.exists():
        print(f"Loading migrated baseline from {_BASELINE_PATH}")
        initial_prompt = load_migrated_baseline(_BASELINE_PATH)
    else:
        print("Generating migrated baseline (one-time LLM redistribution)...")
        initial_prompt, mapping_notes, warnings = migrate_to_task_prongs_with_notes(
            CLIMATE_RLM_SYSTEM_PROMPT,
            model=MODEL,
            output_path=_BASELINE_PATH,
        )
        print(f"  Mapping notes ({len(mapping_notes)}):")
        for note in mapping_notes:
            print(f"    {note}")

    GROUNDING_CRITERIA_PATH = _DEFAULT_EXPERT_KNOWLEDGE_PATH

    optimizer = LEAPPromptOptimizer(model=MODEL)
    optimized = optimizer.run_loop(
        location="Seattle_US",
        extracted_policies_fn=extracted_policies_fn,
        ground_truth_policies=ground_truth_policies,
        rubric=DEFAULT_RUBRIC,
        initial_prompt=initial_prompt,
        source_document_path=SEATTLE_DOC,
        grounding_criteria_path=GROUNDING_CRITERIA_PATH,
        max_iterations=3,
        log_dir=_HERE / "logs",
    )

    print("\n=== Optimized Prompt ===")
    print(optimized.compose())
