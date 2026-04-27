import csv
import datetime
import json as _json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI

from evaluator import (
    CATEGORIES,
    DEFAULT_MODEL,
    EvaluationOutput,
    LEAPEvaluator,
)
from metrics import (
    DEFAULT_COMPOSITE_WEIGHTS,
    EvaluationBundle,
    aggregate_bundles,
    compute_composite_score,
    compute_extraction_bundle,
    compute_classification_bundle,
    compute_hierarchy_bundle,
    compute_quality_bundle,
    ExtractionBundle,
    HierarchyBundle,
    ClassificationBundle,
    QualityBundle,
    MatchingResult,
)
from dev_test_split import LocationConfig, LocationSet, load_locations_from_yaml
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
# Per-prong resampler templates (new evaluator path)
# ---------------------------------------------------------------------------

def _build_resample_user_v2(
    scope: str,
    headline_name: str,
    headline_value: float,
    target: float,
    delta: float,
    supporting: str,
    section: str,
    failure_examples: str,
) -> str:
    """Build the v2 resampler user message without ``str.format`` on a template.

    Prompt prongs and failure-example blocks can contain ``{...}`` (JSON, TeX, etc.).
    A single ``.format()`` on a string that includes those as *values* is safe, but
    a mistaken ``{n}`` or similar in the *template* or double-formatting bugs cause
    ``KeyError``. F-string interpolation only evaluates the given expressions.
    """
    section_display = section or (
        "(empty — write from scratch based on feedback)"
    )
    return (
        f"PRONG: {scope}\n\n"
        f"HEADLINE METRIC ({headline_name}): {headline_value:.3f}\n"
        f"TARGET: {target:.3f}\n"
        f"DELTA SINCE LAST ITERATION: {delta:+.3f}\n\n"
        f"SUPPORTING METRICS:\n{supporting}\n\n"
        f"CURRENT SECTION:\n{section_display}\n\n"
        f"FAILURE EXAMPLES (worst cases for this prong):\n{failure_examples}\n"
    )


# ---------------------------------------------------------------------------
# Per-prong targets and trigger logic
# ---------------------------------------------------------------------------


@dataclass
class PrognTarget:
    """Target headline values per prong. A prong is triggered when its current
    headline is below target OR has worsened since the last iteration."""

    extraction_f1: float = 0.70
    hierarchy_role_agreement: float = 0.85
    hierarchy_parent_attribution: float = 0.85
    classification_primary_agreement: float = 0.85


@dataclass
class AcceptanceConfig:
    """Multi-criterion acceptance configuration for the new evaluator path.

    A candidate is accepted if EITHER:
      (i)  composite_score improves by at least ``min_delta`` AND no metric in
           ``per_metric_floor`` drops by more than its floor, OR
      (ii) the **targeted prong**'s headline metric improves by at least
           ``prong_min_delta`` AND composite does not regress by more than
           ``composite_regression_tolerance``.

    The second clause exists because prong rewrites inside a single RLM pass
    are entangled: a large win on one prong is commonly paired with a small
    loss on another. Without (ii) the optimizer rejects real wins whenever the
    composite is momentarily flat or slightly negative.
    """

    min_delta: float = 0.005
    prong_min_delta: float = 0.05
    composite_regression_tolerance: float = 0.05
    per_metric_floor: dict[str, float] = field(default_factory=lambda: {
        "extraction.f1": 0.10,
        "hierarchy.role_agreement": 0.15,
        "classification.primary_category_agreement": 0.15,
        "quality.plus_one_coverage": 0.20,
    })


# Maps scope → dotted metric path for the per-prong acceptance clause.
_PRONG_HEADLINE: dict[str, str] = {
    "extraction":     "extraction.f1",
    "hierarchy":      "hierarchy.role_agreement",
    "classification": "classification.primary_category_agreement",
}


def _triggered_prongs(
    current: EvaluationBundle,
    previous: Optional[EvaluationBundle],
    targets: PrognTarget,
) -> set[str]:
    """Determine which prongs to rewrite this iteration.

    A prong fires when its headline metric worsened vs. the last accepted
    bundle OR is still below its target.

    On iteration 1 (no previous bundle) all three prongs are bootstrapped.
    """
    if previous is None:
        return {"extraction", "hierarchy", "classification"}

    triggered: set[str] = set()
    if (
        current.extraction.f1 < previous.extraction.f1
        or current.extraction.f1 < targets.extraction_f1
    ):
        triggered.add("extraction")

    cur_h = (current.hierarchy.role_agreement + current.hierarchy.parent_attribution_accuracy) / 2
    prev_h = (previous.hierarchy.role_agreement + previous.hierarchy.parent_attribution_accuracy) / 2
    h_target = (targets.hierarchy_role_agreement + targets.hierarchy_parent_attribution) / 2
    if cur_h < prev_h or cur_h < h_target:
        triggered.add("hierarchy")

    if (
        current.classification.primary_category_agreement
        < previous.classification.primary_category_agreement
        or current.classification.primary_category_agreement
        < targets.classification_primary_agreement
    ):
        triggered.add("classification")

    return triggered


def ordered_triggered(scopes: set[str]) -> list[str]:
    """Canonical priority order: extraction first, then hierarchy, then classification."""
    return [s for s in ("extraction", "hierarchy", "classification") if s in scopes]


def _get_metric(bundle: EvaluationBundle, dotted: str) -> float:
    """Resolve a dotted metric path on an EvaluationBundle."""
    obj: Any = bundle
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return float(obj)


# ---------------------------------------------------------------------------
# Per-prong feedback assembly
# ---------------------------------------------------------------------------


def _assemble_prong_feedback(
    bundle: EvaluationBundle,
    scope: Literal["extraction", "hierarchy", "classification"],
    n_examples: int = 6,
) -> tuple[str, str, float, float, str]:
    """Return (failure_examples, supporting, headline_value, target_default, headline_name)
    for the given prong scope.

    The failure examples are filtered so each prong only sees the failures it
    can actually move:
      extraction    — worst unmatched GT items (missed extractions),
                      capped to ``n_examples``.
      hierarchy     — matched pairs where extracted role != GT role.
      classification — matched pairs where extracted primary_category !=
                       GT primary_category.
    """
    if scope == "extraction":
        headline_name = "extraction.f1"
        headline = bundle.extraction.f1
        target = 0.70

        per_cat = ", ".join(
            f"{c}={r:.2f}" for c, r in sorted(bundle.extraction.per_category_recall.items())
        ) or "(none)"
        supporting = (
            f"  precision = {bundle.extraction.precision:.3f}\n"
            f"  recall    = {bundle.extraction.recall:.3f}\n"
            f"  per_category_recall: {per_cat}\n"
            f"  category_distribution_jsd = {bundle.extraction.category_distribution_jsd:.3f}\n"
            f"  +1 coverage = {bundle.extraction.plus_one_coverage:.3f}"
        )

        missed = bundle.matching.unmatched_gt[:n_examples]
        spurious = bundle.matching.unmatched_extracted[:n_examples]
        examples_lines: list[str] = []
        if missed:
            examples_lines.append("MISSED GT POLICIES (extractor failed to surface):")
            for p in missed:
                examples_lines.append(
                    f"  - [{p.get('primary_category', '?')}/{p.get('role', '?')}] "
                    f"{(p.get('policy_statement') or '')[:160]}"
                )
        if spurious:
            examples_lines.append("\nSPURIOUS EXTRACTIONS (no GT counterpart above threshold):")
            for p in spurious:
                examples_lines.append(
                    f"  - [{p.get('primary_category', '?')}/{p.get('role', '?')}] "
                    f"{(p.get('policy_statement') or '')[:160]}"
                )
        examples = "\n".join(examples_lines) or "(no extraction failures to report)"
        return examples, supporting, headline, target, headline_name

    if scope == "hierarchy":
        headline_name = "hierarchy.role_agreement"
        headline = bundle.hierarchy.role_agreement
        target = 0.85

        supporting = (
            f"  role_agreement              = {bundle.hierarchy.role_agreement:.3f}\n"
            f"  parent_attribution_accuracy = {bundle.hierarchy.parent_attribution_accuracy:.3f}"
        )

        # (gt_role, ext_role, statement, gt_parent, verbatim_excerpt)
        confusions: list[tuple[str, str, str, str, str]] = []
        for pair in bundle.matching.matched:
            gt_role = pair.ground_truth.get("role", "individual")
            ext_role = pair.extracted.get("role", "individual")
            if gt_role != ext_role:
                confusions.append((
                    gt_role, ext_role,
                    (pair.ground_truth.get("policy_statement") or "")[:140],
                    (pair.ground_truth.get("parent_statement") or "").strip()[:140],
                    (pair.ground_truth.get("verbatim_text") or "").strip()[:200],
                ))
            if len(confusions) >= n_examples:
                break

        if confusions:
            lines = ["ROLE CONFUSIONS (gt → ext on matched pairs):"]
            for gt_r, ext_r, stmt, gt_parent, verbatim in confusions:
                lines.append(f"  - {gt_r} → {ext_r} | {stmt}")
                if gt_parent:
                    lines.append(f"      gt.parent_statement: {gt_parent}")
                if verbatim:
                    lines.append(f"      source_excerpt:      {verbatim}")
            examples = "\n".join(lines)
        else:
            examples = "(no role confusions on the matched set)"
        return examples, supporting, headline, target, headline_name

    # classification
    headline_name = "classification.primary_category_agreement"
    headline = bundle.classification.primary_category_agreement
    target = 0.85

    cm = bundle.classification.confusion_matrix
    cm_lines = []
    for gt_cat, row in sorted(cm.items()):
        for ext_cat, count in sorted(row.items()):
            if gt_cat != ext_cat and count > 0:
                cm_lines.append(f"  - {gt_cat} → {ext_cat}: {count}")
    cm_str = "\n".join(cm_lines) or "(no off-diagonal confusions)"

    supporting = (
        f"  primary_category_agreement      = {bundle.classification.primary_category_agreement:.3f}\n"
        f"  financial_instrument_agreement  = {bundle.classification.financial_instrument_agreement:.3f}\n"
        f"  secondary_category_agreement    = {bundle.classification.secondary_category_agreement:.3f}\n"
        f"OFF-DIAGONAL CONFUSION (gt → ext: count):\n{cm_str}"
    )

    # (gt_cat, ext_cat, statement, gt_mechanism, grader_reasoning)
    confusions: list[tuple[str, str, str, str, str]] = []
    for pair in bundle.matching.matched:
        gt_cat = pair.ground_truth.get("primary_category", "Unknown")
        ext_cat = pair.extracted.get("primary_category", "Unknown")
        if gt_cat != ext_cat:
            confusions.append((
                gt_cat, ext_cat,
                (pair.ground_truth.get("policy_statement") or "")[:140],
                (pair.ground_truth.get("canonical_mechanism") or "").strip()[:140],
                (pair.reasoning or "").strip()[:320],
            ))
        if len(confusions) >= n_examples:
            break

    if confusions:
        lines = ["CATEGORY CONFUSIONS (gt → ext on matched pairs):"]
        for gt_c, ext_c, stmt, mech, reasoning in confusions:
            lines.append(f"  - {gt_c} → {ext_c} | {stmt}")
            if mech:
                lines.append(f"      gt.canonical_mechanism: {mech}")
            if reasoning:
                lines.append(f"      grader:                 {reasoning}")
        examples = "\n".join(lines)
    else:
        examples = "(no category confusions on the matched set)"
    return examples, supporting, headline, target, headline_name


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

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
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
    # Candidate generation, evaluation, and acceptance (Task 3)
    # ------------------------------------------------------------------

    def propose_candidate(
        self,
        scope: Literal["extraction", "hierarchy", "classification"],
        current_prompt: StructuredPrompt,
        previous_prompt: StructuredPrompt,
        current_eval: EvaluationOutput,
        previous_eval: EvaluationOutput,
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
    # Per-prong proposer (new evaluator path)
    # ------------------------------------------------------------------

    def propose_candidate_v2(
        self,
        scope: Literal["extraction", "hierarchy", "classification"],
        current_prompt: StructuredPrompt,
        current_bundle: EvaluationBundle,
        previous_bundle: Optional[EvaluationBundle] = None,
        targets: Optional[PrognTarget] = None,
    ) -> StructuredPrompt:
        """Rewrite the named prong on top of the current prompt using bundle-based feedback.

        Other prongs are copied through unchanged so the candidate prompt only
        differs in one prong.
        """
        targets = targets or PrognTarget()

        examples, supporting, headline, target, headline_name = _assemble_prong_feedback(
            current_bundle, scope
        )
        prev_headline = (
            _get_metric(previous_bundle, headline_name) if previous_bundle is not None else headline
        )
        delta = headline - prev_headline

        section = getattr(current_prompt, scope)
        user_msg = _build_resample_user_v2(
            scope=scope,
            headline_name=headline_name,
            headline_value=headline,
            target=target,
            delta=delta,
            supporting=supporting,
            section=section,
            failure_examples=examples,
        )

        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _RESAMPLE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        )
        rewritten = (response.choices[0].message.content or "").strip()

        return StructuredPrompt(
            extraction=rewritten if scope == "extraction" else current_prompt.extraction,
            hierarchy=rewritten if scope == "hierarchy" else current_prompt.hierarchy,
            classification=rewritten if scope == "classification" else current_prompt.classification,
        )

    def accept_candidate_v2(
        self,
        candidate: EvaluationBundle,
        current: EvaluationBundle,
        config: Optional[AcceptanceConfig] = None,
        scope: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Multi-criterion acceptance against the running baseline.

        Accept if EITHER composite improves by at least ``min_delta`` (with no
        per-metric cliff) OR the scope's headline metric improves by at least
        ``prong_min_delta`` while composite does not regress by more than
        ``composite_regression_tolerance`` (and still no per-metric cliff).
        """
        cfg = config or AcceptanceConfig()
        composite_delta = candidate.composite_score - current.composite_score

        def _check_cliff() -> Optional[str]:
            for metric, floor in cfg.per_metric_floor.items():
                drop = _get_metric(current, metric) - _get_metric(candidate, metric)
                if drop >= floor:
                    return f"{metric} cliff: dropped {drop:.3f} (floor {floor:.3f})"
            return None

        # Clause (i): composite improvement path
        if composite_delta >= cfg.min_delta:
            cliff = _check_cliff()
            if cliff is not None:
                return False, cliff
            return (
                True,
                f"accepted (composite): {current.composite_score:+.3f} → "
                f"{candidate.composite_score:+.3f}",
            )

        # Clause (ii): targeted-prong win path
        if scope in _PRONG_HEADLINE:
            headline = _PRONG_HEADLINE[scope]
            prong_delta = _get_metric(candidate, headline) - _get_metric(current, headline)
            if (
                prong_delta >= cfg.prong_min_delta
                and composite_delta >= -cfg.composite_regression_tolerance
            ):
                cliff = _check_cliff()
                if cliff is not None:
                    return False, cliff
                return (
                    True,
                    f"accepted ({scope}): {headline} "
                    f"{_get_metric(current, headline):.3f} → "
                    f"{_get_metric(candidate, headline):.3f} "
                    f"(Δ{prong_delta:+.3f}); composite Δ{composite_delta:+.3f}",
                )

        return (
            False,
            f"insufficient improvement: composite {current.composite_score:+.3f} → "
            f"{candidate.composite_score:+.3f}  (delta {composite_delta:+.3f} "
            f"< {cfg.min_delta})",
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

    # ------------------------------------------------------------------
    # New evaluator path — multi-location, sequential, multi-criterion
    # ------------------------------------------------------------------

    def evaluate_candidate_on_set(
        self,
        candidate_prompt: StructuredPrompt,
        location_set_dev: list[LocationConfig],
        extracted_policies_fn: Callable[[str, str, Optional[pathlib.Path | str], Optional[str]], list[dict]],
        rubric: str,
        evaluator: LEAPEvaluator,
        seeds: int = 1,
        trace_dir: Optional[pathlib.Path] = None,
        iteration_label: str = "",
    ) -> tuple[EvaluationBundle, dict[str, float], list[EvaluationBundle]]:
        """Evaluate one candidate prompt across all dev locations and seeds.

        ``extracted_policies_fn`` is called as ``fn(prompt, location_name,
        source_doc_path, trace_path)`` for each (location, seed). Each call
        produces a list of extracted policy dicts which the evaluator scores
        into an ``EvaluationBundle``.

        Returns:
            (aggregate_bundle, std_per_metric, per_run_bundles)
        """
        per_run: list[EvaluationBundle] = []
        for loc in location_set_dev:
            ground_truth = loc.load_ground_truth()
            for seed in range(seeds):
                trace_path = (
                    str(trace_dir / f"{iteration_label}_{loc.name}_seed{seed}_trace")
                    if trace_dir is not None else None
                )
                extracted = extracted_policies_fn(
                    candidate_prompt.compose(),
                    loc.name,
                    loc.source_document_md,
                    trace_path,
                )
                bundle = evaluator.evaluate(
                    location=loc.name,
                    extracted_policies=extracted,
                    ground_truth_policies=ground_truth,
                    rubric=rubric,
                    source_document_path=loc.source_document_md,
                )
                if not isinstance(bundle, EvaluationBundle):
                    raise RuntimeError(
                        "evaluate_candidate_on_set requires use_new_evaluator=True"
                    )
                per_run.append(bundle)

        aggregate, stds = aggregate_bundles(per_run, location_label="dev_aggregate")
        return aggregate, stds, per_run

    def run_loop_v2(
        self,
        location_set: LocationSet,
        extracted_policies_fn: Callable[[str, str, Optional[pathlib.Path | str], Optional[str]], list[dict]],
        rubric: str,
        initial_prompt: StructuredPrompt,
        max_iterations: int = 10,
        seeds: int = 1,
        targets: Optional[PrognTarget] = None,
        acceptance: Optional[AcceptanceConfig] = None,
        log_dir: Optional[pathlib.Path | str] = None,
        max_accepted_per_iteration: int = 2,
        evaluator: Optional[LEAPEvaluator] = None,
        composite_candidate: bool = True,
        k_no_accept: int = 2,
    ) -> StructuredPrompt:
        """Sequential, multi-location, multi-criterion optimization loop.

        The baseline dev-set evaluation is computed **once** before the loop
        starts and then carried forward as ``running_eval``; each accepted
        candidate's bundle becomes the next iteration's baseline for free. This
        avoids re-running the RLM on a prompt we have already evaluated.

        Each iteration:
          1. Determine triggered prongs from running vs. previous bundles.
          2. Propose candidate(s) for the triggered prongs on top of the
             running prompt. With ``composite_candidate=False`` (default) each
             triggered prong is proposed and evaluated in its own dev-set
             pass; prongs that pass ``accept_candidate_v2`` are merged into
             ``running_prompt`` in priority order, and any rejected prong is
             discarded. Later prongs are then proposed on top of the updated
             prompt. The ``max_accepted_per_iteration`` cap still applies. With
             ``composite_candidate=True`` all triggered prongs are chain-
             rewritten in a single pass and evaluated once — accept/reject
             the whole — which trades per-prong cherry-picking for ~N× fewer
             RLM extractions per iteration.
          3. Accept/reject using ``accept_candidate_v2`` against the running
             bundle. Accepted candidates advance the running pair.
          4. Convergence: no candidate accepted for K consecutive iterations
             OR all targets met. The plan defaults K=2.
          5. After the loop ends, evaluate once on the test set (if any).

        Logs (under timestamped subdir of ``log_dir``):
          iteration_log.csv      — one row per main evaluation
          candidate_log.csv      — one row per candidate (accepted or rejected)
          metrics_bundle.json    — full structured bundle per iteration
          test_results.json      — single test-set evaluation at loop end
        """
        targets = targets or PrognTarget()
        acceptance = acceptance or AcceptanceConfig()
        evaluator = evaluator or LEAPEvaluator(model=self.model, use_new_evaluator=True)

        if not evaluator.use_new_evaluator:
            raise ValueError("run_loop_v2 requires evaluator.use_new_evaluator=True")
        if not location_set.dev:
            raise ValueError("run_loop_v2 requires at least one dev location")

        # Logging setup
        run_dir, iter_writer, cand_writer, iter_file, cand_file = _open_v2_logs(log_dir)
        if run_dir is not None:
            print(f"  Logging to: {run_dir}")

        # One-shot baseline evaluation of the initial prompt. Every iteration
        # after this reuses the previously accepted candidate's bundle as its
        # baseline, so the initial prompt is the only prompt whose dev-set
        # bundle we compute explicitly.
        print(f"\n  Evaluating initial prompt on dev set (one-time baseline)...")
        running_eval, running_stds, _ = self.evaluate_candidate_on_set(
            initial_prompt, location_set.dev, extracted_policies_fn, rubric,
            evaluator, seeds=seeds, trace_dir=run_dir,
            iteration_label="iter0_baseline",
        )
        print(f"  Baseline composite={running_eval.composite_score:+.3f}  "
              f"std={running_stds['composite_score']:.3f}")
        _print_bundle_headlines(running_eval)

        current_prompt = initial_prompt
        running_prompt = initial_prompt
        previous_eval: Optional[EvaluationBundle] = None
        consecutive_no_accept = 0
        K_NO_ACCEPT = k_no_accept

        for t in range(max_iterations):
            print(f"\n{'=' * 60}\nIteration {t + 1}/{max_iterations}\n{'=' * 60}")

            # Baseline for this iteration is the running bundle from either
            # the one-shot initial evaluation (iter 1) or the last accepted
            # candidate (iter > 1). No new RLM extraction is needed here.
            baseline_eval = running_eval
            baseline_stds = running_stds
            print(f"  Running composite={baseline_eval.composite_score:+.3f}  "
                  f"std={baseline_stds['composite_score']:.3f}")
            _print_bundle_headlines(baseline_eval)

            # Step 1: determine triggered prongs
            triggered = _triggered_prongs(baseline_eval, previous_eval, targets)
            ordered = ordered_triggered(triggered)
            print(f"  Triggered prongs: {ordered or 'none'}")

            # Convergence: all targets met
            if not triggered:
                print("  All targets met — converging.")
                _write_iter_row(
                    iter_writer, t + 1, baseline_eval, baseline_stds,
                    triggered, current_prompt, converged=True, run_dir=run_dir,
                )
                break

            accepted_this_iter = 0

            if composite_candidate and len(ordered) > 1:
                # Single combined candidate: chain-rewrite every triggered
                # prong on top of the running prompt, evaluate once.
                print(f"\n  Proposing composite candidate: {'|'.join(ordered)}")
                candidate_prompt = running_prompt
                for scope in ordered:
                    candidate_prompt = self.propose_candidate_v2(
                        scope=scope,
                        current_prompt=candidate_prompt,
                        current_bundle=running_eval,
                        previous_bundle=previous_eval,
                        targets=targets,
                    )
                cand_eval, cand_stds, _ = self.evaluate_candidate_on_set(
                    candidate_prompt, location_set.dev, extracted_policies_fn, rubric,
                    evaluator, seeds=seeds, trace_dir=run_dir,
                    iteration_label=f"iter{t + 1}_cand_composite",
                )
                # For composite candidates the targeted prong is the worst-
                # performing triggered prong (largest gap to target).
                worst_scope = min(
                    ordered,
                    key=lambda s: _get_metric(running_eval, _PRONG_HEADLINE[s]),
                )
                ok, reason = self.accept_candidate_v2(
                    cand_eval, running_eval, acceptance, scope=worst_scope,
                )
                print(f"    composite (prong={worst_scope}): {reason}")

                if cand_writer is not None:
                    cand_writer.writerow({
                        "iteration": t + 1,
                        "scope": "|".join(ordered),
                        "accepted": ok,
                        "reason": reason,
                        "composite_score": round(cand_eval.composite_score, 4),
                        "composite_delta": round(
                            cand_eval.composite_score - running_eval.composite_score, 4
                        ),
                        "extraction_f1": round(cand_eval.extraction.f1, 4),
                        "hierarchy_role_agreement": round(cand_eval.hierarchy.role_agreement, 4),
                        "classification_primary_agreement": round(
                            cand_eval.classification.primary_category_agreement, 4
                        ),
                        "plus_one_coverage": round(cand_eval.quality.plus_one_coverage, 4),
                    })
                    cand_file.flush()

                if ok:
                    running_prompt = candidate_prompt
                    running_eval = cand_eval
                    running_stds = cand_stds
                    accepted_this_iter = 1

            else:
                for scope in ordered:
                    if accepted_this_iter >= max_accepted_per_iteration:
                        print(f"  Reached cap of {max_accepted_per_iteration} acceptances "
                              f"this iteration — skipping {scope}.")
                        break

                    print(f"\n  Proposing candidate for prong: {scope}")
                    candidate_prompt = self.propose_candidate_v2(
                        scope=scope,
                        current_prompt=running_prompt,
                        current_bundle=running_eval,
                        previous_bundle=previous_eval,
                        targets=targets,
                    )
                    cand_eval, cand_stds, _ = self.evaluate_candidate_on_set(
                        candidate_prompt, location_set.dev, extracted_policies_fn, rubric,
                        evaluator, seeds=seeds, trace_dir=run_dir,
                        iteration_label=f"iter{t + 1}_cand_{scope}",
                    )
                    ok, reason = self.accept_candidate_v2(
                        cand_eval, running_eval, acceptance, scope=scope,
                    )
                    print(f"    {reason}")

                    if cand_writer is not None:
                        cand_writer.writerow({
                            "iteration": t + 1,
                            "scope": scope,
                            "accepted": ok,
                            "reason": reason,
                            "composite_score": round(cand_eval.composite_score, 4),
                            "composite_delta": round(
                                cand_eval.composite_score - running_eval.composite_score, 4
                            ),
                            "extraction_f1": round(cand_eval.extraction.f1, 4),
                            "hierarchy_role_agreement": round(cand_eval.hierarchy.role_agreement, 4),
                            "classification_primary_agreement": round(
                                cand_eval.classification.primary_category_agreement, 4
                            ),
                            "plus_one_coverage": round(cand_eval.quality.plus_one_coverage, 4),
                        })
                        cand_file.flush()

                    if ok:
                        running_prompt = candidate_prompt
                        running_eval = cand_eval
                        running_stds = cand_stds
                        accepted_this_iter += 1

            # End-of-iteration logging
            _write_iter_row(
                iter_writer, t + 1, running_eval, running_stds,
                triggered, running_prompt,
                converged=False, run_dir=run_dir,
            )

            if accepted_this_iter == 0:
                consecutive_no_accept += 1
                print(f"  No candidates accepted ({consecutive_no_accept}/{K_NO_ACCEPT}).")
            else:
                consecutive_no_accept = 0

            previous_eval = baseline_eval
            current_prompt = running_prompt

            if consecutive_no_accept >= K_NO_ACCEPT:
                print(f"\n  No acceptances for {K_NO_ACCEPT} consecutive iterations — stopping.")
                break

        # Test-set evaluation (held-out, run once)
        if location_set.test and run_dir is not None:
            print(f"\n  Running held-out test set ({len(location_set.test)} location(s))...")
            test_eval, test_stds, test_per_run = self.evaluate_candidate_on_set(
                current_prompt, location_set.test, extracted_policies_fn, rubric,
                evaluator, seeds=seeds, trace_dir=run_dir,
                iteration_label="test",
            )
            test_path = run_dir / "test_results.json"
            with open(test_path, "w", encoding="utf-8") as fh:
                _json.dump({
                    "aggregate": _json.loads(test_eval.model_dump_json()),
                    "std_per_metric": test_stds,
                    "per_location": [
                        _json.loads(b.model_dump_json()) for b in test_per_run
                    ],
                    "dev_vs_test_composite_gap": (
                        (running_eval.composite_score - test_eval.composite_score)
                        if running_eval is not None else None
                    ),
                }, fh, indent=2)
            print(f"  test composite={test_eval.composite_score:+.3f}  "
                  f"dev composite={running_eval.composite_score:+.3f}  "
                  f"gap={(running_eval.composite_score - test_eval.composite_score):+.3f}")

        for fh in (iter_file, cand_file):
            if fh is not None:
                fh.close()

        return current_prompt


# ---------------------------------------------------------------------------
# Logging helpers (new evaluator path)
# ---------------------------------------------------------------------------


def _open_v2_logs(log_dir: Optional[pathlib.Path | str]):
    if log_dir is None:
        return None, None, None, None, None
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = pathlib.Path(log_dir) / f"v2_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    iter_file = open(run_dir / "iteration_log.csv", "w", newline="", encoding="utf-8")
    iter_writer = csv.DictWriter(
        iter_file,
        fieldnames=[
            "iteration",
            "composite_score",
            "composite_score_std",
            "extraction_f1",
            "extraction_precision",
            "extraction_recall",
            "extraction_jsd",
            "plus_one_coverage",
            "hierarchy_role_agreement",
            "hierarchy_parent_attribution",
            "classification_primary_agreement",
            "classification_financial_agreement",
            "classification_secondary_agreement",
            "triggered_prongs",
            "converged",
            "composed_prompt",
        ],
    )
    iter_writer.writeheader()

    cand_file = open(run_dir / "candidate_log.csv", "w", newline="", encoding="utf-8")
    cand_writer = csv.DictWriter(
        cand_file,
        fieldnames=[
            "iteration",
            "scope",
            "accepted",
            "reason",
            "composite_score",
            "composite_delta",
            "extraction_f1",
            "hierarchy_role_agreement",
            "classification_primary_agreement",
            "plus_one_coverage",
        ],
    )
    cand_writer.writeheader()

    return run_dir, iter_writer, cand_writer, iter_file, cand_file


def _write_iter_row(
    writer,
    iteration: int,
    bundle: EvaluationBundle,
    stds: dict[str, float],
    triggered: set[str],
    prompt: StructuredPrompt,
    converged: bool,
    run_dir: Optional[pathlib.Path],
) -> None:
    if writer is None:
        return
    writer.writerow({
        "iteration": iteration,
        "composite_score": round(bundle.composite_score, 4),
        "composite_score_std": round(stds.get("composite_score", 0.0), 4),
        "extraction_f1": round(bundle.extraction.f1, 4),
        "extraction_precision": round(bundle.extraction.precision, 4),
        "extraction_recall": round(bundle.extraction.recall, 4),
        "extraction_jsd": round(bundle.extraction.category_distribution_jsd, 4),
        "plus_one_coverage": round(bundle.quality.plus_one_coverage, 4),
        "hierarchy_role_agreement": round(bundle.hierarchy.role_agreement, 4),
        "hierarchy_parent_attribution": round(bundle.hierarchy.parent_attribution_accuracy, 4),
        "classification_primary_agreement": round(
            bundle.classification.primary_category_agreement, 4
        ),
        "classification_financial_agreement": round(
            bundle.classification.financial_instrument_agreement, 4
        ),
        "classification_secondary_agreement": round(
            bundle.classification.secondary_category_agreement, 4
        ),
        "triggered_prongs": "|".join(sorted(triggered)),
        "converged": converged,
        "composed_prompt": prompt.compose(),
    })

    if run_dir is not None:
        bundle_path = run_dir / f"metrics_bundle_iter_{iteration}.json"
        with open(bundle_path, "w", encoding="utf-8") as fh:
            _json.dump({
                "iteration": iteration,
                "bundle": _json.loads(bundle.model_dump_json()),
                "std_per_metric": stds,
            }, fh, indent=2)


def _print_bundle_headlines(b: EvaluationBundle) -> None:
    print(
        f"    extraction.f1={b.extraction.f1:.3f}  "
        f"hierarchy.role={b.hierarchy.role_agreement:.3f}  "
        f"classification.primary={b.classification.primary_category_agreement:.3f}  "
        f"+1 coverage={b.quality.plus_one_coverage:.3f}"
    )


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
