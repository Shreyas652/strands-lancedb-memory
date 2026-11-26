"""Offline answer-evaluation tests; no API calls or gold access by the reader."""

from evals.answer_dev import _build_context, _reader_prompt, estimate_cost, evaluate
from evals.model_api import ModelClient


def test_context_uses_exact_duplicate_session_ordinal():
    case = {
        "haystack_session_ids": ["same", "same"],
        "haystack_dates": ["2024/01/01", "2024/01/02"],
        "haystack_sessions": [
            [{"role": "user", "content": "old value"}],
            [{"role": "user", "content": "new value"}],
        ],
    }
    retrieval = {"retrieved_session_ordinals": [1], "retrieved_session_ids": ["same"]}
    context, ordinals = _build_context(case, retrieval, 1, 1000)
    assert "new value" in context and "old value" not in context
    assert ordinals == [1]


def test_reference_answer_reaches_only_judge(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.json"
    manifest = tmp_path / "manifest.json"
    retrieval = tmp_path / "retrieval.json"
    for path in (dataset, manifest, retrieval):
        path.write_text("{}", encoding="utf-8")
    prompt = _reader_prompt("question", "2024/01/01", "past conversation")
    plan = {
        "question_id": "q1",
        "question_type": "single-session-user",
        "system": "lancedb_fts",
        "k": 3,
        "reader_prompt": prompt,
        "context_characters": len("past conversation"),
        "included_session_ordinals": [0],
        "question": "question",
        "evidence_rank": 1,
    }
    references = {"q1": "gold secret only for judge"}
    assert estimate_cost([plan], references) > 0
    client = ModelClient(key="unused", ledger=tmp_path / "calls.jsonl")
    seen = []

    def fake_fetch(model, submitted_prompt, max_tokens):
        seen.append((model, submitted_prompt))
        if model.startswith("gpt-4o-mini"):
            assert "gold secret" not in submitted_prompt
            text = "candidate answer"
        else:
            assert "gold secret" in submitted_prompt
            text = "yes"
        return {
            "model": model,
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 4},
        }

    monkeypatch.setattr(client, "_fetch", fake_fetch)
    result = evaluate(
        [plan],
        references,
        client,
        dataset=dataset,
        manifest=manifest,
        retrieval_report=retrieval,
        max_chars=1000,
    )
    assert len(seen) == 2
    assert result["summary"]["lancedb_fts:3"]["accuracy"] == 1.0
    assert "gold secret" not in str(result)
