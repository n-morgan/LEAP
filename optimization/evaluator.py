"""
evaluator.py — LEAP Evaluator (Algorithm 1)

Grades RLM-extracted policies against ground-truth (GENIUS) policies for a given
location. Produces per-category scores, recall, and FPR, plus per-policy grade
reasoning keyed by (category, policy_id) for downstream prompt optimization.

Algorithm:
    For each category c in C:
        1. Group extracted and ground-truth policies by role (parent / sub / individual)
        2. Embed policy statements and compute pairwise cosine similarity
        3. Run Hungarian matching for optimal 1:1 alignment within each role group
        4. Grade each matched pair via GraderLLM (+1 / 0 / -1)
        5. Penalize unmatched GT policies (-1, hurts recall)
        6. Penalize unmatched extracted policies (-1, hurts FPR)
        7. Aggregate into S[l, c], R[l, c], F[l, c]

C = { Mitigation, Adaptation, Resource Efficiency, Nature-Based Solutions }

Usage:
    evaluator = LEAPEvaluator()
    result = evaluator.evaluate(
        location="Seattle_US",
        extracted_policies=rlm_output,
        ground_truth_policies=genius_output,
        rubric="Grade on specificity, commitment, and mechanism...",
        source_document=markdown_text,
    )
    print(result.scores)   # {"Mitigation": 0.4, "Adaptation": -0.2, ...}
    print(result.recall)   # {"Mitigation": 0.8, ...}
"""

import asyncio
import json
import os
import pathlib
import re
import tempfile
from typing import Any, Literal, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from scipy.optimize import linear_sum_assignment

from metrics import (
    DEFAULT_COMPOSITE_WEIGHTS,
    EvaluationBundle,
    MatchedPair,
    MatchingResult,
    compute_classification_bundle,
    compute_composite_score,
    compute_extraction_bundle,
    compute_hierarchy_bundle,
    compute_quality_bundle,
)

load_dotenv()

# Single source of truth for the OpenAI model used by both grader paths and
# the resampler (set in prompt_optimizer.py via this constant).
DEFAULT_MODEL: str = "gpt-4.1-2025-04-14"

# Default similarity threshold for category-agnostic Hungarian matching.
DEFAULT_SIMILARITY_THRESHOLD: float = 0.55

# Concurrency cap for async grading. The implementation plan's PR 10 calls for
# Semaphore(8); drop to 4 if 429s appear in production.
GRADER_CONCURRENCY: int = 8

CATEGORIES: list[str] = [
    "Mitigation",
    "Adaptation",
    "Resource Efficiency",
    "Nature-Based Solutions",
]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PolicyGrade(BaseModel):
    """Grade for a single extracted policy relative to its matched ground-truth."""

    policy_id: str = Field(
        description="Truncated policy_statement used as a stable identifier."
    )
    grade: Literal[-1, 0, 1] = Field(
        description=(
            "+1: matches ground truth in scope, commitment, and specificity. "
            " 0: directionally correct but vague or imprecise. "
            "-1: no match, hallucinated, or contradicts document intent."
        )
    )
    reasoning: str = Field(
        description="Step-by-step justification grounded in the source text."
    )


class _GraderOutput(BaseModel):
    """Structured output enforced on the grader LLM call.

    Extended in the redesign to surface per-field signals so the grader's
    verdict on statement / role / category alignment can flow into the
    targeted-feedback assembly used by the per-prong resampler.
    """

    grade: Literal[-1, 0, 1]
    reasoning: str
    statement_match: Literal["match", "partial", "mismatch"] = "partial"
    role_match: bool = True
    category_match: bool = True


class EvaluationOutput(BaseModel):
    """Full evaluation result for one location."""

    location: str
    scores: dict[str, float] = Field(
        description="Mean grade per category (includes -1 penalties)."
    )
    recall: dict[str, float] = Field(
        description="Matched GT count / total GT count per category."
    )
    fpr: dict[str, float] = Field(
        description="Unmatched extracted count / total extracted count per category."
    )
    grades: dict[str, PolicyGrade] = Field(
        description="Key format: '{category}::{policy_id}'. One entry per matched pair."
    )
    hierarchy_accuracy: float = Field(
        default=1.0,
        description=(
            "Fraction of matched pairs where extracted role equals ground-truth role. "
            "E.g. 7 correct out of 10 matched pairs gives 0.7. "
            "Defaults to 1.0 when no matched pairs exist."
        ),
    )


# ---------------------------------------------------------------------------
# Grader prompts
# ---------------------------------------------------------------------------

_GRADER_SYSTEM = """\
You are an expert unbiased evaluator grading an extracted climate policy against a
ground-truth policy produced by a reference expert system.

SCORING GUIDE (overall grade):
  +1  The extracted policy matches the ground-truth in scope, commitment, and specificity.
      The core commitment, target, and delivery mechanism are all captured correctly.
   0  Directionally correct but vague or imprecise. The general intent is right but
      key details such as targets, deadlines, or mechanisms are missing or softened.
  -1  No meaningful match, hallucinated content, or the extraction contradicts the
      document intent.

You also emit three per-field verdicts that drive the targeted-feedback loop:
  statement_match : "match" | "partial" | "mismatch"
                    — does the extracted policy_statement convey the same commitment?
  role_match      : true | false
                    — does the extracted role match the GT role
                      (parent / sub / individual)?
  category_match  : true | false
                    — does the extracted primary_category match the GT primary_category?

Ground your overall grade in the source document text when provided.
Return only the structured output — no preamble.
"""

_GRADER_USER = """\
RUBRIC:
{rubric}

SOURCE DOCUMENT (excerpt):
{source_document}

GROUND-TRUTH POLICY:
  policy_statement:   {gt_statement}
  role:               {gt_role}
  primary_category:   {gt_category}

EXTRACTED POLICY:
  policy_statement:   {ext_statement}
  role:               {ext_role}
  primary_category:   {ext_category}
  source_quote:       {ext_source_quote}
"""

# RLM grader templates — used when a full source document is provided.
# The RLM traverses the entire markdown document to locate relevant passages
# before producing a structured grade.
_RLM_GRADER_SYSTEM = """\
You are an expert unbiased evaluator grading an extracted climate policy against a
ground-truth policy. You have access to the full source document — read through it
to find the passage(s) that ground your evaluation before deciding on a grade.

SCORING GUIDE (overall grade):
  +1  The extracted policy matches the ground-truth in scope, commitment, and
      specificity. Core commitment, target, and delivery mechanism all correct.
   0  Directionally correct but vague or imprecise. Intent is right but key
      details (targets, deadlines, mechanisms) are missing or softened.
  -1  No meaningful match, hallucinated content, or contradicts document intent.

You also emit three per-field verdicts that drive the targeted-feedback loop:
  statement_match : "match" | "partial" | "mismatch"
                    — does the extracted policy_statement convey the same commitment?
  role_match      : true | false
                    — does the extracted role match the GT role
                      (parent / sub / individual)?
  category_match  : true | false
                    — does the extracted primary_category match the GT primary_category?

After reading the document, return ONLY a JSON object with exactly five keys:
  "grade":           -1, 0, or 1
  "reasoning":       step-by-step justification grounded in the source text
  "statement_match": "match" | "partial" | "mismatch"
  "role_match":      true | false
  "category_match":  true | false

No preamble. No trailing text.
"""

_RLM_GRADER_USER = """\
RUBRIC:
{rubric}

GROUND-TRUTH POLICY:
  policy_statement:   {gt_statement}
  role:               {gt_role}
  primary_category:   {gt_category}

EXTRACTED POLICY:
  policy_statement:   {ext_statement}
  role:               {ext_role}
  primary_category:   {ext_category}
  source_quote:       {ext_source_quote}

SOURCE DOCUMENT:
{document}
"""


# ---------------------------------------------------------------------------
# Robust JSON parsing for the RLM grader path
# ---------------------------------------------------------------------------


def _parse_grader_json(raw: str) -> Optional[dict]:
    """Best-effort parser for the RLM grader response.

    Strategy:
      1. Try ``json.loads`` on the full response.
      2. Fall back to slicing from the first ``{`` to the last ``}``.
      3. Return None to signal a parse failure (caller falls back to grade=0).
    """
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# LEAPEvaluator
# ---------------------------------------------------------------------------


class LEAPEvaluator:
    """
    Implements Algorithm 1: LEAP Evaluator.

    Groups policies by category and role, runs optimal 1:1 Hungarian matching
    on embedding similarity, grades each matched pair via an LLM using the
    expert rubric, and penalizes unmatched policies on both sides.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        embedding_model: str = "text-embedding-3-small",
        use_new_evaluator: bool = False,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self.use_new_evaluator = use_new_evaluator
        self.similarity_threshold = similarity_threshold
        self._client: Optional[OpenAI] = None

    # ------------------------------------------------------------------
    # Client
    # ------------------------------------------------------------------

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    # ------------------------------------------------------------------
    # Embedding and matching
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Return an (N, D) float32 embedding matrix."""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        response = self._get_client().embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return np.array([d.embedding for d in response.data], dtype=np.float32)

    def _cosine_sim(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Return (N, M) cosine similarity matrix between rows of A and B."""
        norm_a = np.linalg.norm(A, axis=1, keepdims=True) + 1e-10
        norm_b = np.linalg.norm(B, axis=1, keepdims=True) + 1e-10
        return (A / norm_a) @ (B / norm_b).T

    def _hungarian_match(self, sim: np.ndarray) -> list[tuple[int, int]]:
        """
        Optimal 1:1 matching maximising total cosine similarity.
        Returns a list of (extracted_idx, ground_truth_idx) pairs.
        """
        if sim.size == 0:
            return []
        row_ind, col_ind = linear_sum_assignment(-sim)
        return list(zip(row_ind.tolist(), col_ind.tolist()))

    def _match_globally(
        self,
        extracted: list[dict[str, Any]],
        ground_truth: list[dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> MatchingResult:
        """Category-agnostic Hungarian matching with a similarity threshold.

        All extracted and all GT policies are embedded in a single batch each.
        We build the full N x M cosine similarity matrix, run Hungarian on it,
        then drop any assigned pair whose similarity is below ``threshold``.
        Below-threshold rows / columns flow into ``unmatched_extracted`` /
        ``unmatched_gt`` respectively.

        Decoupling matching from primary_category is the structural fix in the
        redesign — a misclassified extraction can still be matched to its GT
        counterpart because the cosine signal is on policy_statement text.
        """
        if threshold is None:
            threshold = self.similarity_threshold

        if not extracted and not ground_truth:
            return MatchingResult(
                matched=[],
                unmatched_extracted=[],
                unmatched_gt=[],
                similarity_threshold=threshold,
            )

        if not extracted:
            return MatchingResult(
                matched=[],
                unmatched_extracted=[],
                unmatched_gt=list(ground_truth),
                similarity_threshold=threshold,
            )

        if not ground_truth:
            return MatchingResult(
                matched=[],
                unmatched_extracted=list(extracted),
                unmatched_gt=[],
                similarity_threshold=threshold,
            )

        ext_emb = self._embed([p.get("policy_statement", "") for p in extracted])
        gt_emb = self._embed([p.get("policy_statement", "") for p in ground_truth])
        sim = self._cosine_sim(ext_emb, gt_emb)

        pairs = self._hungarian_match(sim)
        matched_ext: set[int] = set()
        matched_gt: set[int] = set()
        matched_pairs: list[MatchedPair] = []

        for ei, gi in pairs:
            s = float(sim[ei, gi])
            if s < threshold:
                continue
            matched_ext.add(ei)
            matched_gt.add(gi)
            matched_pairs.append(
                MatchedPair(
                    extracted=extracted[ei],
                    ground_truth=ground_truth[gi],
                    similarity=s,
                )
            )

        unmatched_extracted = [p for i, p in enumerate(extracted) if i not in matched_ext]
        unmatched_gt = [p for i, p in enumerate(ground_truth) if i not in matched_gt]

        return MatchingResult(
            matched=matched_pairs,
            unmatched_extracted=unmatched_extracted,
            unmatched_gt=unmatched_gt,
            similarity_threshold=threshold,
        )

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------

    def _grade_pair_rlm(
        self,
        extracted: dict[str, Any],
        ground_truth: dict[str, Any],
        rubric: str,
        document_text: str,
    ) -> _GraderOutput:
        """Grade one pair via RLM so the full source document can be traversed."""
        from rlm import RLM
        from rlm.logger import RLMLogger

        log_dir = tempfile.mkdtemp(prefix="rlm_grade_")
        logger = RLMLogger(log_dir=log_dir)
        rlm = RLM(
            backend="openai",
            backend_kwargs={
                "model_name": self.model,
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            other_backends=["openai"],
            other_backend_kwargs=[
                {"model_name": self.model, "api_key": os.getenv("OPENAI_API_KEY")},
            ],
            environment="local",
            environment_kwargs={},
            max_depth=1,
            max_iterations=20,
            custom_system_prompt=_RLM_GRADER_SYSTEM,
            logger=logger,
            verbose=False,
        )
        prompt = _RLM_GRADER_USER.format(
            rubric=rubric,
            gt_statement=ground_truth.get("policy_statement", ""),
            gt_role=ground_truth.get("role", "individual"),
            gt_category=ground_truth.get("primary_category", "Unknown"),
            ext_statement=extracted.get("policy_statement", ""),
            ext_role=extracted.get("role", "individual"),
            ext_category=extracted.get("primary_category", "Unknown"),
            ext_source_quote=extracted.get("source_quote", ""),
            document=document_text,
        )
        result = rlm.completion(
            prompt=prompt,
            root_prompt="Grade the extracted policy against the ground truth using the source document.",
        )
        raw = result.response.strip()
        data = _parse_grader_json(raw)
        if data is None:
            return _GraderOutput(
                grade=0,
                reasoning="grader parse failure",
                statement_match="partial",
                role_match=(extracted.get("role") == ground_truth.get("role")),
                category_match=(
                    extracted.get("primary_category") == ground_truth.get("primary_category")
                ),
            )

        grade = max(-1, min(1, int(data.get("grade", 0))))
        return _GraderOutput(
            grade=grade,
            reasoning=data.get("reasoning", ""),
            statement_match=data.get("statement_match", "partial"),
            role_match=bool(data.get(
                "role_match",
                extracted.get("role") == ground_truth.get("role"),
            )),
            category_match=bool(data.get(
                "category_match",
                extracted.get("primary_category") == ground_truth.get("primary_category"),
            )),
        )

    def _grade_pair(
        self,
        extracted: dict[str, Any],
        ground_truth: dict[str, Any],
        rubric: str,
        source_document: str,
    ) -> _GraderOutput:
        """Grade one matched (extracted, ground_truth) pair.

        If source_document text is provided, delegates to _grade_pair_rlm so the
        full document is traversed by the RLM. Otherwise calls the grader LLM
        directly (no document context).
        """
        if source_document:
            return self._grade_pair_rlm(extracted, ground_truth, rubric, source_document)

        user_msg = _GRADER_USER.format(
            rubric=rubric,
            source_document="Not provided.",
            gt_statement=ground_truth.get("policy_statement", ""),
            gt_role=ground_truth.get("role", "individual"),
            gt_category=ground_truth.get("primary_category", "Unknown"),
            ext_statement=extracted.get("policy_statement", ""),
            ext_role=extracted.get("role", "individual"),
            ext_category=extracted.get("primary_category", "Unknown"),
            ext_source_quote=extracted.get("source_quote", ""),
        )
        response = self._get_client().beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": _GRADER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format=_GraderOutput,
        )
        return response.choices[0].message.parsed

    def _grade_pairs_concurrent(
        self,
        pairs: list,
        rubric: str,
        document_text: str,
    ) -> list[_GraderOutput]:
        """Grade a batch of matched pairs concurrently.

        Falls back to sequential grading when called from inside an existing
        event loop (e.g. when nested in async code). This keeps the public
        API sync while still benefiting from concurrency in the common case.
        """
        if not pairs:
            return []

        async def _bound(sem: asyncio.Semaphore, pair) -> _GraderOutput:
            async with sem:
                return await asyncio.to_thread(
                    self._grade_pair,
                    pair.extracted,
                    pair.ground_truth,
                    rubric,
                    document_text,
                )

        async def _runner() -> list[_GraderOutput]:
            sem = asyncio.Semaphore(GRADER_CONCURRENCY)
            return await asyncio.gather(*[_bound(sem, p) for p in pairs])

        try:
            asyncio.get_running_loop()
            # We are already inside an event loop — fall back to sync.
            return [
                self._grade_pair(p.extracted, p.ground_truth, rubric, document_text)
                for p in pairs
            ]
        except RuntimeError:
            return asyncio.run(_runner())

    # ------------------------------------------------------------------
    # Cell evaluation
    # ------------------------------------------------------------------

    def _evaluate_cell(
        self,
        category: str,
        extracted: list[dict[str, Any]],
        ground_truth: list[dict[str, Any]],
        rubric: str,
        source_document: str,
    ) -> tuple[float, float, float, dict[str, PolicyGrade], int, int]:
        """
        Evaluate one (location, category) cell.

        Separates policies by role (parent / sub / individual) to enforce
        hierarchy-aware matching (parent to parent, child to child), then
        runs Hungarian matching within each role group independently.

        Returns:
            score         — mean grade across all scores including -1 penalties
            recall        — matched GT / total GT
            fpr           — unmatched extracted / total extracted
            grades        — dict of PolicyGrade keyed by '{category}::{policy_id}'
            role_correct  — matched pairs where extracted role == ground-truth role
            total_matched — total matched pairs
        """
        grades: dict[str, PolicyGrade] = {}
        all_scores: list[int] = []
        role_correct = 0
        total_matched = 0

        if not extracted and not ground_truth:
            return 0.0, 1.0, 0.0, grades, 0, 0

        # Group by role for hierarchy-aware matching
        def by_role(policies: list[dict]) -> dict[str, list[dict]]:
            groups: dict[str, list[dict]] = {
                "parent": [], "sub": [], "individual": []
            }
            for p in policies:
                role = p.get("role", "individual")
                groups.setdefault(role, []).append(p)
            return groups

        ext_by_role = by_role(extracted)
        gt_by_role = by_role(ground_truth)

        # Track global indices into ext_flat / gt_flat for unmatched detection
        ext_flat: list[dict] = []
        gt_flat: list[dict] = []
        matched_ext: set[int] = set()
        matched_gt: set[int] = set()

        for role in ("parent", "sub", "individual"):
            ext_group = ext_by_role.get(role, [])
            gt_group = gt_by_role.get(role, [])

            ext_offset = len(ext_flat)
            gt_offset = len(gt_flat)
            ext_flat.extend(ext_group)
            gt_flat.extend(gt_group)

            if not ext_group or not gt_group:
                continue

            # Embed and match
            ext_emb = self._embed([p.get("policy_statement", "") for p in ext_group])
            gt_emb = self._embed([p.get("policy_statement", "") for p in gt_group])
            sim = self._cosine_sim(ext_emb, gt_emb)
            pairs = self._hungarian_match(sim)

            for local_ei, local_gi in pairs:
                global_ei = ext_offset + local_ei
                global_gi = gt_offset + local_gi
                matched_ext.add(global_ei)
                matched_gt.add(global_gi)

                graded = self._grade_pair(
                    ext_group[local_ei],
                    gt_group[local_gi],
                    rubric,
                    source_document,
                )
                policy_id = ext_group[local_ei].get("policy_statement", f"{role}_{local_ei}")[:80]
                grades[f"{category}::{policy_id}"] = PolicyGrade(
                    policy_id=policy_id,
                    grade=graded.grade,
                    reasoning=graded.reasoning,
                )
                all_scores.append(graded.grade)

                # Role accuracy: compare extracted role to ground-truth role
                ext_role = ext_group[local_ei].get("role", "individual")
                gt_role = gt_group[local_gi].get("role", "individual")
                if ext_role == gt_role:
                    role_correct += 1
                total_matched += 1

        # Unmatched GT: RLM missed them — penalize recall
        for gi in range(len(gt_flat)):
            if gi not in matched_gt:
                all_scores.append(-1)

        # Unmatched extracted: no GT counterpart — penalize FPR
        for ei in range(len(ext_flat)):
            if ei not in matched_ext:
                all_scores.append(-1)

        score = float(np.mean(all_scores)) if all_scores else 0.0
        recall = len(matched_gt) / len(gt_flat) if gt_flat else 1.0
        fpr = (len(ext_flat) - len(matched_ext)) / len(ext_flat) if ext_flat else 0.0

        return score, recall, fpr, grades, role_correct, total_matched

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        location: str,
        extracted_policies: list[dict[str, Any]],
        ground_truth_policies: list[dict[str, Any]],
        rubric: str,
        source_document_path: Optional[pathlib.Path | str] = None,
    ) -> "EvaluationOutput | EvaluationBundle":
        """
        Dispatch to either the legacy bucketed path (returns ``EvaluationOutput``)
        or the redesigned global-matching path (returns ``EvaluationBundle``).

        The dispatch is controlled by ``self.use_new_evaluator``. The legacy path
        is preserved verbatim so we can A/B-compare during the cutover.
        """
        if self.use_new_evaluator:
            return self._evaluate_new(
                location=location,
                extracted_policies=extracted_policies,
                ground_truth_policies=ground_truth_policies,
                rubric=rubric,
                source_document_path=source_document_path,
            )

        return self._evaluate_legacy(
            location=location,
            extracted_policies=extracted_policies,
            ground_truth_policies=ground_truth_policies,
            rubric=rubric,
            source_document_path=source_document_path,
        )

    def _evaluate_legacy(
        self,
        location: str,
        extracted_policies: list[dict[str, Any]],
        ground_truth_policies: list[dict[str, Any]],
        rubric: str,
        source_document_path: Optional[pathlib.Path | str] = None,
    ) -> EvaluationOutput:
        """Original bucketed-then-Hungarian path. Kept verbatim for legacy A/B."""
        doc_text = (
            pathlib.Path(source_document_path).read_text(encoding="utf-8")
            if source_document_path is not None
            else ""
        )

        scores: dict[str, float] = {}
        recall: dict[str, float] = {}
        fpr: dict[str, float] = {}
        all_grades: dict[str, PolicyGrade] = {}
        total_role_correct = 0
        total_matched_pairs = 0

        for category in CATEGORIES:
            ext_cat = [
                p for p in extracted_policies
                if p.get("primary_category") == category
            ]
            gt_cat = [
                p for p in ground_truth_policies
                if p.get("primary_category") == category
            ]

            s, r, f, cell_grades, rc, tm = self._evaluate_cell(
                category, ext_cat, gt_cat, rubric, doc_text
            )
            scores[category] = s
            recall[category] = r
            fpr[category] = f
            all_grades.update(cell_grades)
            total_role_correct += rc
            total_matched_pairs += tm

        hierarchy_accuracy = (
            total_role_correct / total_matched_pairs
            if total_matched_pairs > 0 else 1.0
        )

        return EvaluationOutput(
            location=location,
            scores=scores,
            recall=recall,
            fpr=fpr,
            grades=all_grades,
            hierarchy_accuracy=hierarchy_accuracy,
        )

    # ------------------------------------------------------------------
    # New evaluator path — global matching + orthogonal bundles
    # ------------------------------------------------------------------

    def _evaluate_new(
        self,
        location: str,
        extracted_policies: list[dict[str, Any]],
        ground_truth_policies: list[dict[str, Any]],
        rubric: str,
        source_document_path: Optional[pathlib.Path | str] = None,
    ) -> EvaluationBundle:
        """Redesigned evaluator path.

        1. Run category-agnostic Hungarian matching with similarity threshold.
        2. Grade every matched pair via the (extended) grader prompt.
        3. Compute extraction / hierarchy / classification / quality bundles.
        4. Compose the four into an ``EvaluationBundle``.
        """
        doc_text = (
            pathlib.Path(source_document_path).read_text(encoding="utf-8")
            if source_document_path is not None
            else ""
        )

        matching = self._match_globally(
            extracted_policies, ground_truth_policies, threshold=self.similarity_threshold
        )

        # Grade every matched pair concurrently. ``_grade_pair`` is sync and
        # blocking — we offload it via ``asyncio.to_thread`` and gate the
        # concurrent count with a semaphore to stay under provider rate
        # limits.
        graded_outputs = self._grade_pairs_concurrent(matching.matched, rubric, doc_text)
        for pair, graded in zip(matching.matched, graded_outputs):
            pair.grade = graded.grade
            pair.reasoning = graded.reasoning
            pair.statement_match = graded.statement_match
            pair.role_match = graded.role_match
            pair.category_match = graded.category_match

        extraction = compute_extraction_bundle(matching, ground_truth_policies)
        hierarchy = compute_hierarchy_bundle(matching)
        classification = compute_classification_bundle(matching)
        quality = compute_quality_bundle(matching, ground_truth_policies)
        composite = compute_composite_score(extraction, hierarchy, classification, quality)

        return EvaluationBundle(
            location=location,
            matching=matching,
            extraction=extraction,
            hierarchy=hierarchy,
            classification=classification,
            quality=quality,
            composite_score=composite,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import csv
    import pathlib

    OUTPUTS_DIR = pathlib.Path(__file__).parent / "organized_outputs"

    # ------------------------------------------------------------------
    # Load and reconcile both CSVs into the evaluator's expected format:
    #   list[dict] with at minimum:
    #     "policy_statement"  — text of the policy
    #     "primary_category"  — one of the four LEAP categories
    #     "role"              — "parent" | "sub" | "individual"
    #
    # RLM output columns (rlm_seattle_policies.csv):
    #   role, parent_statement, policy_statement, primary_category, ...extras
    # Structured output columns (structured_policies.csv):
    #   policy_id, role, parent_statement, policy_statement, primary_category, ...extras
    #
    # Both already use the same column names for the three required fields,
    # so no renaming is needed — we just strip rows with empty policy_statement.
    # ------------------------------------------------------------------

    def load_policies(path: pathlib.Path) -> list[dict]:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [row for row in reader if row.get("policy_statement", "").strip()]

    rlm_policies = load_policies(OUTPUTS_DIR / "rlm_seattle_policies.csv")
    structured_policies = load_policies(OUTPUTS_DIR / "structured_policies.csv")

    print(f"Loaded {len(rlm_policies)} RLM policies and "
          f"{len(structured_policies)} structured (ground-truth) policies.")

    DEFAULT_RUBRIC = (
        "Grade on specificity (quantified targets, deadlines, mechanisms), "
        "commitment strength (binding vs aspirational language), "
        "and accuracy relative to the source document."
    )

    evaluator = LEAPEvaluator()
    result = evaluator.evaluate(
        location="Seattle_US",
        extracted_policies=rlm_policies,
        ground_truth_policies=structured_policies,
        rubric=DEFAULT_RUBRIC,
        # source_document_path=pathlib.Path("path/to/seattle.md"),
    )

    # ------------------------------------------------------------------
    # Print per-category scores and the overall mean score
    # ------------------------------------------------------------------
    print("\n=== RLM vs Structured System — Seattle_US ===")
    print(f"{'Category':<25} {'Score':>7}  {'Recall':>7}  {'FPR':>7}")
    print("-" * 52)
    for cat in CATEGORIES:
        print(
            f"{cat:<25} {result.scores.get(cat, 0.0):>7.3f}"
            f"  {result.recall.get(cat, 0.0):>7.3f}"
            f"  {result.fpr.get(cat, 0.0):>7.3f}"
        )
    overall = sum(result.scores.values()) / len(result.scores) if result.scores else 0.0
    print("-" * 52)
    print(f"{'Overall mean score':<25} {overall:>7.3f}")
