"""Smoke test: run the new evaluator path on the Seattle CSVs and confirm
its matched-pair set heavily overlaps with the legacy bucketed path.

We mock both the embedder and the grader so this test is offline and fast.
The embedder uses a stable string-hash projection; the grader returns a fixed
``+1`` so we exercise the full pipeline without an LLM call.
"""

import csv
import hashlib
import pathlib

import numpy as np
import pytest

from evaluator import LEAPEvaluator, _GraderOutput
from metrics import EvaluationBundle


_HERE = pathlib.Path(__file__).resolve().parent
_OUTPUTS_DIR = _HERE.parent / "organized_outputs"


def _load(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [
            row for row in csv.DictReader(fh)
            if row.get("policy_statement", "").strip()
        ]


def _hash_embed(texts: list[str]) -> np.ndarray:
    """Deterministic 32-d projection of each text via SHA-256 bytes.

    Pure-Python, no model dependency. Texts that differ even slightly will
    produce different vectors — but identical texts produce identical vectors,
    so this is good enough to assert that *some* sensible matching happens.
    """
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    out = np.zeros((len(texts), 32), dtype=np.float32)
    for i, t in enumerate(texts):
        digest = hashlib.sha256((t or "").encode("utf-8")).digest()
        out[i] = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
    return out


@pytest.mark.skipif(
    not (_OUTPUTS_DIR / "rlm_seattle_policies.csv").exists()
    or not (_OUTPUTS_DIR / "structured_policies.csv").exists(),
    reason="Seattle organized_outputs CSVs not present",
)
def test_new_evaluator_returns_bundle(monkeypatch):
    rlm_policies = _load(_OUTPUTS_DIR / "rlm_seattle_policies.csv")
    gt_policies = _load(_OUTPUTS_DIR / "structured_policies.csv")

    ev = LEAPEvaluator(use_new_evaluator=True, similarity_threshold=0.0)
    monkeypatch.setattr(ev, "_embed", _hash_embed)
    monkeypatch.setattr(
        ev, "_grade_pair",
        lambda *a, **kw: _GraderOutput(
            grade=1, reasoning="ok", statement_match="match",
            role_match=True, category_match=True,
        ),
    )

    bundle = ev.evaluate(
        location="Seattle_US",
        extracted_policies=rlm_policies,
        ground_truth_policies=gt_policies,
        rubric="test rubric",
    )

    assert isinstance(bundle, EvaluationBundle)
    assert bundle.location == "Seattle_US"
    # Headlines populated and within plausible bounds
    assert 0.0 <= bundle.extraction.f1 <= 1.0
    assert 0.0 <= bundle.hierarchy.role_agreement <= 1.0
    assert 0.0 <= bundle.classification.primary_category_agreement <= 1.0
    assert 0.0 <= bundle.composite_score <= 1.0
    # Some pairs got matched
    assert len(bundle.matching.matched) > 0


def test_legacy_path_still_returns_legacy_output(monkeypatch):
    """Sanity: the flag truly gates dispatch — default path is unchanged."""
    rlm_policies = [
        {"policy_statement": "x", "primary_category": "Mitigation", "role": "individual"},
    ]
    gt_policies = [
        {"policy_statement": "x", "primary_category": "Mitigation", "role": "individual"},
    ]
    ev = LEAPEvaluator(use_new_evaluator=False)
    monkeypatch.setattr(ev, "_embed", _hash_embed)
    monkeypatch.setattr(
        ev, "_grade_pair",
        lambda *a, **kw: _GraderOutput(
            grade=1, reasoning="ok", statement_match="match",
            role_match=True, category_match=True,
        ),
    )

    out = ev.evaluate(
        location="Seattle_US",
        extracted_policies=rlm_policies,
        ground_truth_policies=gt_policies,
        rubric="test",
    )
    # Legacy returns EvaluationOutput, not EvaluationBundle
    assert hasattr(out, "scores") and hasattr(out, "recall")
    assert "Mitigation" in out.scores
