"""The exact paired test uses a two-sided binomial tail."""

import pytest

from evals.paired_significance import analyze, exact_mcnemar_p


def test_exact_mcnemar_known_small_case():
    assert exact_mcnemar_p(3, 0) == 0.25
    assert exact_mcnemar_p(2, 2) == 1.0
    assert exact_mcnemar_p(0, 0) == 1.0


def test_pairs_raw_answers_by_question_id_and_rejects_missing_arm():
    report = {
        "phase": "frozen_test_answers",
        "question_count": 2,
        "per_question": [
            {"question_id": "a", "system": "file", "k": 3, "correct": False},
            {"question_id": "b", "system": "lancedb_fts", "k": 3, "correct": False},
            {"question_id": "a", "system": "lancedb_fts", "k": 3, "correct": True},
            {"question_id": "b", "system": "file", "k": 3, "correct": True},
        ],
    }
    result = analyze(report, "source-hash")
    assert (result["fts_only_correct"], result["file_only_correct"]) == (1, 1)
    assert result["paired_accuracy_difference_points"] == 0.0
    report["per_question"].pop()
    with pytest.raises(ValueError, match="missing questions"):
        analyze(report, "source-hash")
