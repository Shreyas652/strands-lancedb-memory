"""Split guards use only IDs and types; gold answers stay in scoring code."""

import json

import pytest

from evals.retrieval_dev import (
    _session_id_from_file,
    _session_ref_from_file,
    _session_text,
)
from evals.split import make_split


def test_stratified_split_is_stable_and_sealed(tmp_path):
    dataset = tmp_path / "cases.json"
    cases = [
        {
            "question_id": str(number),
            "question_type": "a" if number < 15 else "b",
            "answer": "sealed",
        }
        for number in range(30)
    ]
    dataset.write_text(json.dumps(cases), encoding="utf-8")
    first = make_split(dataset, seed=7)
    assert first == make_split(dataset, seed=7)
    assert first["dev_count"] == 6
    assert first["test_count"] == 24
    assert first["dev_by_type"] == {"a": 3, "b": 3}
    assert "answer" not in json.dumps(first)
    assert "test_question_ids" not in first
    assert set(first["dev_question_ids"]).issubset(
        {case["question_id"] for case in cases}
    )


def test_split_rejects_duplicate_ids(tmp_path):
    dataset = tmp_path / "cases.json"
    dataset.write_text(
        json.dumps(
            [
                {"question_id": "same", "question_type": "a"},
                {"question_id": "same", "question_type": "b"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        make_split(dataset, seed=1)


def test_duplicate_session_ids_keep_distinct_write_headings():
    turns = [{"role": "user", "content": "same ID, separate session"}]
    first = _session_text("repeated", "2025/01/01", turns, 0)
    second = _session_text("repeated", "2025/01/02", turns, 1)
    assert first != second
    assert _session_id_from_file(first) == _session_id_from_file(second) == "repeated"
    assert _session_ref_from_file(first) == (0, "repeated")
    assert _session_ref_from_file(second) == (1, "repeated")
