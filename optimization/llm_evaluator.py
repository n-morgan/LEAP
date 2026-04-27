"""
llm_evaluator.py — RLM-free LEAP Evaluator (LLM grader only)

``LLMEvaluator`` subclasses :class:`LEAPEvaluator` and overrides the grader so each
matched pair is scored with a single structured OpenAI call
(``beta.chat.completions.parse``).  The recursive RLM path is never used.

Always returns :class:`metrics.EvaluationBundle` (``use_new_evaluator=True``).
Default behavior compares extracted policies to ground-truth only; any
``source_document_path`` is ignored unless ``include_source_document=True``.

Usage
-----
>>> from llm_evaluator import LLMEvaluator
>>> ev = LLMEvaluator()
>>> bundle = ev.evaluate(
...     location="Seattle_US",
...     extracted_policies=extracted,
...     ground_truth_policies=ground_truth,
...     rubric="...",
... )
"""

from __future__ import annotations

from typing import Any

from evaluator import (
    DEFAULT_MODEL,
    DEFAULT_SIMILARITY_THRESHOLD,
    LEAPEvaluator,
    _GRADER_SYSTEM,
    _GRADER_USER,
    _GraderOutput,
)


class LLMEvaluator(LEAPEvaluator):
    """RLM-free evaluator.

    Grades each matched pair with one structured LLM call and produces the same
    :class:`metrics.EvaluationBundle` as ``LEAPEvaluator(use_new_evaluator=True)``,
    without recursive document traversal.

    By default (``include_source_document=False``), the grader does not use the
    source document; comparison is structured-vs-structured only.  Set
    ``include_source_document=True`` to inline the full document into the grader
    user prompt (single call, not RLM).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        embedding_model: str = "text-embedding-3-small",
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        include_source_document: bool = False,
        use_new_evaluator: bool = True,
    ) -> None:
        if not use_new_evaluator:
            raise ValueError(
                "LLMEvaluator always uses the new bundle output (set use_new_evaluator=True or omit it)."
            )
        super().__init__(
            model=model,
            embedding_model=embedding_model,
            use_new_evaluator=True,
            similarity_threshold=similarity_threshold,
        )
        self.include_source_document = include_source_document

    def _grade_pair_rlm(
        self,
        extracted: dict[str, Any],
        ground_truth: dict[str, Any],
        rubric: str,
        document_text: str,
    ) -> _GraderOutput:  # pragma: no cover — never selected by LLM path
        """Not used. ``LLMEvaluator`` does not use the RLM grader."""
        raise NotImplementedError("LLMEvaluator does not use the RLM grader; use LEAPEvaluator for the RLM path.")

    def _grade_pair(
        self,
        extracted: dict[str, Any],
        ground_truth: dict[str, Any],
        rubric: str,
        source_document: str,
    ) -> _GraderOutput:
        """Grade one pair via structured output; never calls the RLM."""
        if self.include_source_document and (source_document or "").strip():
            doc_slot: str = source_document
        else:
            doc_slot = "Not provided."

        user_msg = _GRADER_USER.format(
            rubric=rubric,
            source_document=doc_slot,
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
