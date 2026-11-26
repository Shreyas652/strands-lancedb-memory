"""No network calls: verify ledger, replay, and cost ceiling."""

import json

import pytest

from evals.model_api import BudgetExceededError, ModelClient, load_key


def test_key_file_overrides_stale_process_value(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "stale")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=current\n", encoding="utf-8")
    assert load_key(env_file) == "current"
    env_file.write_text("OPENAI_API_KEY=\n", encoding="utf-8")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        load_key(env_file)
    env_file.write_text("OPENAI_API_KEY missing-equals\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed assignment"):
        load_key(env_file)


def test_durable_call_and_resume_without_duplicate_charge(tmp_path, monkeypatch):
    ledger = tmp_path / "calls.jsonl"
    client = ModelClient(key="unused", ledger=ledger)
    calls = []

    def fake_fetch(model, prompt, max_tokens):
        calls.append((model, prompt, max_tokens))
        return {
            "model": model,
            "choices": [{"message": {"content": "  answer  "}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 4},
        }

    monkeypatch.setattr(client, "_fetch", fake_fetch)
    first = client.call("gpt-4o-mini-2024-07-18", "question", max_tokens=16)
    assert first.text == "answer" and not first.cached
    assert first.cost_usd > 0
    resumed = ModelClient(key="unused", ledger=ledger)
    monkeypatch.setattr(resumed, "_fetch", fake_fetch)
    second = resumed.call("gpt-4o-mini-2024-07-18", "question", max_tokens=16)
    assert second.cached and second.prompt_sha256 == first.prompt_sha256
    assert first.prompt_sha256 != first.request_sha256
    assert second.request_sha256 == first.request_sha256
    assert calls == [("gpt-4o-mini-2024-07-18", "question", 16)]
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
    record = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    assert record["model"] == "gpt-4o-mini-2024-07-18"
    assert record["prompt_sha256"] == first.prompt_sha256


def test_budget_blocks_before_network(tmp_path, monkeypatch):
    client = ModelClient(
        key="unused", ledger=tmp_path / "calls.jsonl", budget_usd=0.000001
    )
    monkeypatch.setattr(
        client, "_fetch", lambda *args: pytest.fail("network was called")
    )
    with pytest.raises(BudgetExceededError):
        client.call("gpt-4o-2024-08-06", "large prompt" * 50, max_tokens=100)


def test_structured_response_format_has_distinct_cache_identity(tmp_path, monkeypatch):
    client = ModelClient(key="unused", ledger=tmp_path / "calls.jsonl")
    observed = []

    def fake_fetch(model, prompt, max_tokens, response_format=None):
        observed.append(response_format)
        return {
            "model": model,
            "choices": [{"message": {"content": '{"facts":[]}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }

    monkeypatch.setattr(client, "_fetch", fake_fetch)
    format_spec = {"type": "json_object"}
    first = client.call(
        "gpt-4o-mini-2024-07-18",
        "Return JSON",
        max_tokens=32,
        response_format=format_spec,
    )
    second = client.call(
        "gpt-4o-mini-2024-07-18",
        "Return JSON",
        max_tokens=32,
        response_format=format_spec,
    )
    plain = client.call("gpt-4o-mini-2024-07-18", "Return JSON", max_tokens=32)
    assert second.cached and not first.cached and not plain.cached
    assert first.request_sha256 != plain.request_sha256
    assert observed == [format_spec, None]
