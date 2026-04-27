"""Unit tests for category-agnostic global matching with similarity threshold.

These tests bypass real embeddings by monkeypatching ``LEAPEvaluator._embed`` to
return pre-cooked vectors keyed off each policy's ``policy_statement``. That
keeps the tests fast and pure — no network, no model-name fragility.
"""

import numpy as np
import pytest

from evaluator import LEAPEvaluator


def _make_embedder(vectors_by_text: dict[str, list[float]]):
    """Return a function suitable for monkeypatching ``_embed``."""

    def _embed(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.array(
            [vectors_by_text[t] for t in texts], dtype=np.float32
        )

    return _embed


def test_known_optimal_pairing(monkeypatch):
    """5 extracted x 6 GT with one obvious 1:1 pairing for 5 of them."""
    extracted = [
        {"policy_statement": f"E{i}", "primary_category": "Mitigation"} for i in range(5)
    ]
    ground_truth = [
        {"policy_statement": f"G{i}", "primary_category": "Mitigation"} for i in range(6)
    ]

    # E0..E4 each align perfectly with G0..G4. G5 has no extracted counterpart.
    vectors = {}
    for i in range(5):
        v = [0.0] * 6
        v[i] = 1.0
        vectors[f"E{i}"] = v
    for i in range(6):
        v = [0.0] * 6
        v[i] = 1.0
        vectors[f"G{i}"] = v

    ev = LEAPEvaluator(use_new_evaluator=True, similarity_threshold=0.5)
    monkeypatch.setattr(ev, "_embed", _make_embedder(vectors))

    result = ev._match_globally(extracted, ground_truth)

    assert len(result.matched) == 5
    assert len(result.unmatched_extracted) == 0
    assert len(result.unmatched_gt) == 1
    assert result.unmatched_gt[0]["policy_statement"] == "G5"

    for pair in result.matched:
        assert (
            pair.extracted["policy_statement"][1:]
            == pair.ground_truth["policy_statement"][1:]
        )
        assert pair.similarity == pytest.approx(1.0)


def test_below_threshold_pair_dropped(monkeypatch):
    """A unique extracted with no semantically similar GT must land in U_E."""
    extracted = [
        {"policy_statement": "ALIGNED"},
        {"policy_statement": "RANDOM"},
    ]
    ground_truth = [
        {"policy_statement": "ALIGNED_GT"},
        {"policy_statement": "OTHER_GT"},
    ]

    vectors = {
        "ALIGNED":     [1.0, 0.0, 0.0, 0.0],
        "ALIGNED_GT":  [1.0, 0.0, 0.0, 0.0],
        "RANDOM":      [0.0, 0.0, 1.0, 0.0],
        # OTHER_GT is nearly orthogonal to everything except ALIGNED, so
        # Hungarian will be forced to assign RANDOM <-> OTHER_GT but the
        # resulting similarity is below threshold.
        "OTHER_GT":    [0.0, 1.0, 0.0, 0.0],
    }

    ev = LEAPEvaluator(use_new_evaluator=True, similarity_threshold=0.5)
    monkeypatch.setattr(ev, "_embed", _make_embedder(vectors))

    result = ev._match_globally(extracted, ground_truth)

    assert len(result.matched) == 1
    assert result.matched[0].extracted["policy_statement"] == "ALIGNED"
    assert result.matched[0].ground_truth["policy_statement"] == "ALIGNED_GT"

    assert {p["policy_statement"] for p in result.unmatched_extracted} == {"RANDOM"}
    assert {p["policy_statement"] for p in result.unmatched_gt} == {"OTHER_GT"}


def test_cross_category_match_allowed(monkeypatch):
    """Identical policy_statement texts must match even with different
    primary_category labels — that's the whole point of decoupling matching
    from classification.
    """
    extracted = [
        {"policy_statement": "X", "primary_category": "Adaptation"},
    ]
    ground_truth = [
        {"policy_statement": "X_GT", "primary_category": "Mitigation"},
    ]

    vectors = {
        "X":    [1.0, 0.0],
        "X_GT": [1.0, 0.0],
    }

    ev = LEAPEvaluator(use_new_evaluator=True, similarity_threshold=0.5)
    monkeypatch.setattr(ev, "_embed", _make_embedder(vectors))

    result = ev._match_globally(extracted, ground_truth)

    assert len(result.matched) == 1
    pair = result.matched[0]
    assert pair.extracted["primary_category"] == "Adaptation"
    assert pair.ground_truth["primary_category"] == "Mitigation"
    assert pair.similarity == pytest.approx(1.0)


def test_empty_inputs():
    ev = LEAPEvaluator(use_new_evaluator=True)
    res = ev._match_globally([], [])
    assert res.matched == []
    assert res.unmatched_extracted == []
    assert res.unmatched_gt == []
