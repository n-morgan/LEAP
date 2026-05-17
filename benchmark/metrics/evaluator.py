"""
metrics/evaluator.py — LEAP Evaluation Protocol

Re-exports the evaluation protocol from core.evaluator.

Evaluation protocol summary
----------------------------
1. Embed every extracted and every ground-truth ``policy_statement``.
2. Build a cosine-similarity matrix and run Hungarian assignment for
   optimal 1:1 matching.
3. Drop pairs below ``similarity_threshold`` (default 0.55).
4. Grade each accepted pair (+1 / 0 / -1) via an LLM judge.
5. Compute headline metrics (precision, recall, F1) counting only
   grade +1 pairs as true positives (plus-one coverage).
6. Compute structural metrics: role agreement, parent attribution
   accuracy (LLM judge on parent_statement pairs), primary category
   agreement, financial instrument agreement, secondary category
   agreement.
7. Aggregate a weighted composite score.

See core/evaluator.py for full implementation details.
"""

from __future__ import annotations

from core.evaluator import (
    LEAPEvaluator,
    EvaluationOutput,
    DEFAULT_RUBRIC,
    CATEGORIES,
)

__all__ = ["LEAPEvaluator", "EvaluationOutput", "DEFAULT_RUBRIC", "CATEGORIES"]
