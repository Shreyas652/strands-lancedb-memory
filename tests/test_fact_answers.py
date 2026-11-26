"""Fact answer prompts use validated extracted text, not whole sessions."""

import pytest

from evals.answer_facts_dev import _plans


def test_fact_plan_uses_extracted_fact_and_detects_tampering():
    cases = [
        {
            "question_id": "q",
            "question_type": "single-session-user",
            "question": "What drink?",
            "question_date": "2025/01/01",
            "haystack_sessions": [[{"role": "user", "content": "RAW_SESSION_ONLY"}]],
        }
    ]
    retrieval = {
        ("q", "fact_key_fts"): {
            "retrieved_fact_indices": [0],
            "retrieved_contents": ["The user likes tea."],
            "retrieved_session_ordinals": [0],
            "first_evidence_rank": 1,
        }
    }
    extraction = {
        "facts_by_question": {
            "q": [{"session_ordinal": 0, "key": "drink", "fact": "The user likes tea."}]
        }
    }
    plans = _plans(cases, retrieval, extraction, ["fact_key_fts"], [3], 1000)
    assert len(plans) == 2
    assert "The user likes tea" in plans[1]["reader_prompt"]
    assert "RAW_SESSION_ONLY" not in plans[1]["reader_prompt"]
    assert "The user likes tea" not in plans[0]["reader_prompt"]
    retrieval[("q", "fact_key_fts")]["retrieved_contents"] = ["TAMPERED"]
    with pytest.raises(ValueError, match="differs from extraction"):
        _plans(cases, retrieval, extraction, ["fact_key_fts"], [3], 1000)
