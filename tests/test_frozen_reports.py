"""Sealed test reports reject changed freezes and invalid arm membership."""

import hashlib
import json
from pathlib import Path

import pytest

from evals.all_500 import combine
from evals.frozen_retrieval import sealed_ids
from evals.split import make_split, sha256_file


def test_test_ids_require_matching_frozen_config(tmp_path):
    dataset = tmp_path / "source.json"
    dataset.write_text(
        json.dumps(
            [
                {"question_id": str(number), "question_type": "user"}
                for number in range(4)
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "split.json"
    split = make_split(dataset, seed=1, dev_fraction=0.5)
    manifest.write_text(json.dumps(split), encoding="utf-8")
    frozen = tmp_path / "freeze.json"
    config = {
        "status": "FROZEN",
        "dataset_sha256": split["dataset_sha256"],
        "manifest_sha256": sha256_file(manifest),
        "sealed_test_ids_sha256": split["sealed_test_ids_sha256"],
        "runner_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "evals" / "freeze.py").read_bytes()
        ).hexdigest(),
    }
    frozen.write_text(json.dumps(config), encoding="utf-8")
    ids, _ = sealed_ids(dataset, manifest, frozen)
    assert len(ids) == 2
    config["status"] = "DRAFT"
    frozen.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen configuration"):
        sealed_ids(dataset, manifest, frozen)


def test_all_500_combiner_rejects_dev_test_overlap():
    manifest = {
        "dataset_sha256": "data",
        "case_count": 4,
        "dev_count": 2,
        "test_count": 2,
        "dev_question_ids": ["d1", "d2"],
    }
    frozen = {
        "status": "FROZEN",
        "dataset_sha256": "data",
        "system": "lancedb_fts",
        "k": 5,
    }

    def rows(ids):
        return [
            {
                "question_id": question_id,
                "system": system,
                "k": k,
                "question_type": "user",
                "correct": True,
            }
            for question_id in ids
            for system, k in (
                ("no_memory", 0),
                ("file", 5),
                ("test", 5),
                ("lancedb_fts", 5),
            )
        ]

    base = {
        "dataset_sha256": "data",
        "official_prompt_sha256": "prompt",
        "reader_model": "gpt-4o-mini-2024-07-18",
        "judge_model": "gpt-4o-2024-08-06",
    }
    dev = dict(
        base, phase="dev_answers", question_count=2, per_question=rows(["d1", "d2"])
    )
    vector = dict(dev)
    test = dict(
        base,
        phase="frozen_test_answers",
        question_count=2,
        frozen_system="lancedb_fts",
        frozen_k=5,
        per_question=rows(["t1", "t2"]),
    )
    report = combine(manifest, frozen, dev, vector, test)
    assert report["summary"]["lancedb_fts:5"]["total"] == 4
    test["per_question"] = rows(["d1", "t2"])
    with pytest.raises(ValueError, match="overlaps development"):
        combine(manifest, frozen, dev, vector, test)
