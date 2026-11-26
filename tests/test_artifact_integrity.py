"""Raw rows, not a hand-edited summary, determine reported benchmark metrics."""

import hashlib

import pytest

from scripts.check_artifact_integrity import _check_freeze_hash, _check_rows


def test_frozen_config_hash_accepts_lf_or_crlf_only():
    lf = b'{\n  "status": "FROZEN"\n}\n'
    crlf = lf.replace(b"\n", b"\r\n")
    _check_freeze_hash(hashlib.sha256(lf).hexdigest(), lf)
    _check_freeze_hash(hashlib.sha256(crlf).hexdigest(), lf)
    _check_freeze_hash(hashlib.sha256(crlf).hexdigest(), crlf)
    with pytest.raises(ValueError, match="both LF and CRLF"):
        _check_freeze_hash(hashlib.sha256(b"changed").hexdigest(), lf)


def test_answer_summary_cannot_hide_an_incorrect_row():
    report = {
        "phase": "dev_answers",
        "question_count": 2,
        "summary": {
            "lancedb_fts:5": {
                "correct": 1,
                "total": 2,
                "accuracy": 0.5,
                "by_type": {"user": {"correct": 1, "total": 2}},
            }
        },
        "per_question": [
            {
                "question_id": "a",
                "system": "lancedb_fts",
                "k": 5,
                "correct": True,
                "question_type": "user",
            },
            {
                "question_id": "b",
                "system": "lancedb_fts",
                "k": 5,
                "correct": False,
                "question_type": "user",
            },
        ],
    }
    _check_rows(report)
    report["summary"]["lancedb_fts:5"]["correct"] = 2
    with pytest.raises(ValueError, match="differ from raw rows"):
        _check_rows(report)
