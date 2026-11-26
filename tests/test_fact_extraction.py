"""Fact extraction keeps the benchmark question and gold answer out of prompts."""

import json

import pytest

from evals.extract_dev import _load_cases, _parse_facts, _prompts, evaluate
from evals.model_api import CallResult
from evals.split import make_split


def test_extraction_prompt_contains_only_conversation_fields(tmp_path):
    source = tmp_path / "dataset.json"
    case = {
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "SECRET_QUESTION",
        "answer": "SECRET_GOLD",
        "answer_session_ids": ["s1"],
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2025/01/01 (Wed) 00:00"],
        "haystack_sessions": [[{"role": "user", "content": "I like green tea."}]],
    }
    source.write_text(json.dumps([case]), encoding="utf-8")
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps(make_split(source, seed=1, dev_fraction=0.9)))
    selected = _load_cases(source, manifest, None)
    batches = _prompts(selected, max_sessions=4, max_batch_chars=1000)
    assert len(batches) == 1
    assert "green tea" in batches[0]["prompt"]
    assert "SECRET_QUESTION" not in batches[0]["prompt"]
    assert "SECRET_GOLD" not in batches[0]["prompt"]


def test_extractor_rejects_unseen_session_reference():
    with pytest.raises(ValueError, match="outside its input"):
        _parse_facts(
            '{"facts":[{"session_ordinal":99,"key":"drink","fact":"Likes tea"}]}',
            {0},
        )


def test_extractor_accepts_strict_json_records():
    text = '{"facts":[{"session_ordinal":0,"key":"drink","fact":"Likes tea"}]}'
    assert _parse_facts(text, {0}) == [
        {"session_ordinal": 0, "key": "drink", "fact": "Likes tea"}
    ]
    assert _parse_facts(f"```json\n{text}\n```", {0}) == _parse_facts(text, {0})


def test_truncated_structured_response_retries_with_larger_limit(tmp_path):
    class Client:
        def __init__(self):
            self.limits = []
            self.records = {
                "first": {"cost_usd": 0.001},
                "retry": {"cost_usd": 0.001},
            }
            self.spent_usd = 0.002

        def call(self, model, prompt, *, max_tokens, response_format):
            self.limits.append(max_tokens)
            text = (
                '{"facts":[{"session_ordinal":0,"key":"x","fact":"cut'
                if max_tokens == 4096
                else '{"facts":[{"session_ordinal":0,"key":"x","fact":"complete"}]}'
            )
            return CallResult(
                text,
                model,
                100,
                max_tokens if max_tokens == 4096 else 50,
                0.001,
                "prompt",
                "first" if max_tokens == 4096 else "retry",
                False,
            )

    dataset = tmp_path / "dataset.json"
    manifest = tmp_path / "split.json"
    dataset.write_text("[]", encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")
    client = Client()
    result = evaluate(
        [
            {
                "question_id": "q1",
                "session_ordinals": [0],
                "sessions": [{"session_ordinal": 0, "text": "data"}],
                "prompt": "data",
            }
        ],
        client,
        dataset,
        manifest,
    )
    assert client.limits == [4096, 8192]
    assert result["fact_count"] == 1
    assert result["model_records"][-1]["max_output_tokens"] == 8192
    assert result["api_calls_attributable"] == 2


def test_truncated_multi_session_batch_splits_before_larger_retry(tmp_path):
    class Client:
        def __init__(self):
            self.limits = []
            self.records = {str(i): {"cost_usd": 0.001} for i in range(3)}
            self.spent_usd = 0.003

        def call(self, model, prompt, *, max_tokens, response_format):
            self.limits.append(max_tokens)
            index = len(self.limits) - 1
            if index == 0:
                text, output_tokens = '{"facts":[{"fact":"cut', 4096
            else:
                text, output_tokens = '{"facts":[]}', 20
            return CallResult(
                text, model, 100, output_tokens, 0.001, "prompt", str(index), False
            )

    dataset = tmp_path / "dataset.json"
    manifest = tmp_path / "split.json"
    dataset.write_text("[]", encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")
    from evals.extract_dev import _batch

    batch = _batch(
        "q1", [{"session_ordinal": 0, "text": "a"}, {"session_ordinal": 1, "text": "b"}]
    )
    client = Client()
    result = evaluate([batch], client, dataset, manifest)
    assert client.limits == [4096, 4096, 4096]
    assert [row["status"] for row in result["model_records"]] == [
        "truncated_split",
        "complete",
        "complete",
    ]
    assert result["api_calls_attributable"] == 3
