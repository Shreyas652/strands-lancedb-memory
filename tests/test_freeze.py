"""The freeze rule uses complete dev arms and cannot hide an unfavorable row."""

import pytest

from evals.freeze import choose


def _answers(arms):
    rows = []
    summary = {}
    for arm, correct in arms.items():
        system, number = arm.rsplit(":", 1)
        k = int(number)
        summary[arm] = {"correct": correct, "total": 2, "accuracy": correct / 2}
        rows.extend(
            {
                "question_id": question_id,
                "system": system,
                "k": k,
                "correct": index < correct,
            }
            for index, question_id in enumerate(("a", "b"))
        )
    return {
        "phase": "dev_answers",
        "dataset_sha256": "dataset",
        "question_count": 2,
        "reader_model": "gpt-4o-mini-2024-07-18",
        "judge_model": "gpt-4o-2024-08-06",
        "temperature": 0,
        "official_prompt_sha256": "judge-prompt",
        "reader_instructions_sha256": "reader-prompt",
        "max_context_characters": 30000,
        "summary": summary,
        "per_question": rows,
    }


def _retrieval(systems):
    return {
        "phase": "dev_retrieval_only",
        "dataset_sha256": "dataset",
        "question_count": 2,
        "embedding_model_id": "local-model",
        "summary": {name: {f"hit_at_{k}": 0.5 for k in (3, 5, 10)} for name in systems},
    }


def test_freeze_prefers_more_correct_then_simpler_tie():
    manifest = {
        "dataset_sha256": "dataset",
        "dev_count": 2,
        "dev_question_ids": ["a", "b"],
        "sealed_test_ids_sha256": "sealed",
    }
    baseline = _answers({f"lancedb_fts:{k}": 1 for k in (3, 5, 10)})
    vector = _answers({f"hybrid_rrf:0.30:{k}": 2 if k == 5 else 1 for k in (3, 5, 10)})
    first = choose(
        manifest,
        baseline,
        vector,
        _retrieval(["lancedb_fts"]),
        _retrieval(["hybrid_rrf:0.30"]),
    )
    assert first["system"] == "hybrid_rrf:0.30" and first["k"] == 5
    baseline["summary"]["lancedb_fts:3"]["correct"] = 2
    baseline["summary"]["lancedb_fts:3"]["accuracy"] = 1.0
    baseline["per_question"][1]["correct"] = True
    tied = choose(
        manifest,
        baseline,
        vector,
        _retrieval(["lancedb_fts"]),
        _retrieval(["hybrid_rrf:0.30"]),
    )
    assert tied["system"] == "lancedb_fts" and tied["k"] == 3


def test_freeze_rejects_missing_dev_arm():
    manifest = {
        "dataset_sha256": "dataset",
        "dev_count": 2,
        "dev_question_ids": ["a", "b"],
        "sealed_test_ids_sha256": "sealed",
    }
    baseline = _answers({f"lancedb_fts:{k}": 1 for k in (3, 5, 10)})
    vector = _answers({"hybrid_rrf:0.30:5": 2})
    with pytest.raises(ValueError, match="incomplete"):
        choose(
            manifest,
            baseline,
            vector,
            _retrieval(["lancedb_fts"]),
            _retrieval(["hybrid_rrf:0.30"]),
        )
