"""Offline contract and persistence tests for the embedded memory store."""

import asyncio
import copy
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import lancedb
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from strands import Agent
from strands.memory import AddMessagesContext, MemoryEntry, MemoryManager, MemoryStore
from strands.memory.extraction.triggers import InvocationTrigger
from strands.models import Model

from strands_lancedb_memory import LanceDBMemoryStore
from strands_lancedb_memory.store import (
    EmbeddingError,
    IndexUnavailableError,
    MemoryStorageError,
    ReplayConflictError,
    SchemaMismatchError,
    StoreNotInitializedError,
)


def run(coro):
    return asyncio.run(coro)


def make_store(tmp_path, *, tenant="alice", **kwargs):
    return LanceDBMemoryStore(uri=tmp_path / "db", tenant_id=tenant, **kwargs)


def test_protocol_manager_and_fts_reopen(tmp_path):
    store = make_store(tmp_path)
    assert MemoryStore in type(store).__mro__  # Protocol is not runtime_checkable.
    manager = MemoryManager(stores=[store], add_tool_config=True)
    run(store.initialize())
    run(store.add("blue bicycle"))
    run(store.add("green bicycle"))
    assert run(store.search("bicycle", {"max_search_results": 1}))
    results = run(manager.search("bicycle", {"max_search_results": 1}))
    assert len(results) == 1
    assert isinstance(results[0], MemoryEntry)
    assert results[0].store_name == "lancedb"
    assert len(run(store.search("bicycle", {"max_search_results": 2}))) == 2
    reopened = make_store(tmp_path)
    assert len(run(reopened.search("bicycle"))) == 2
    run(store.optimize())
    assert len(run(reopened.search("bicycle"))) == 2


def test_new_process_reopen_reads_existing_memory(tmp_path):
    store = make_store(tmp_path)
    run(store.add("persistent midnight bicycle"))
    program = (
        "import asyncio,sys; "
        "from strands_lancedb_memory import LanceDBMemoryStore; "
        "store=LanceDBMemoryStore(uri=sys.argv[1],tenant_id='alice',writable=False); "
        "rows=asyncio.run(store.search('midnight')); "
        "assert [row.content for row in rows]==['persistent midnight bicycle']"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program, str(store.uri)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_two_process_writers_keep_all_rows():
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "phase2_cross_process_probe.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["observed_count"] == report["expected_count"] == 20
    assert [writer["exit_code"] for writer in report["writers"]] == [0, 0]


class StubModel(Model):
    def __init__(self):
        self.seen = []

    def update_config(self, **model_config):
        del model_config

    def get_config(self):
        return {"model_id": "offline-store-test"}

    async def structured_output(self, *args, **kwargs):
        del args, kwargs
        if False:
            yield {}

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        del tool_specs, system_prompt, kwargs
        self.seen.append(copy.deepcopy(messages))
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": "offline answer"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}


def test_real_agent_injection_tools_and_raw_extraction(tmp_path):
    store = make_store(tmp_path, ingestion="raw", agent_run_id="run-agent")
    store.extraction = {"trigger": [InvocationTrigger()]}
    run(store.add("the user's durable fact is a blue bicycle"))
    manager = MemoryManager(stores=[store], add_tool_config=True)
    model = StubModel()
    agent = Agent(model=model, callback_handler=None, memory_manager=manager)
    agent("Tell me about my blue bicycle")
    assert any(
        "<memory>" in block.get("text", "") for block in model.seen[0][-1]["content"]
    )
    assert run(store.search("bicycle"))
    assert run(store.search("Tell"))  # Invocation trigger persisted the user turn.
    search_tool = next(
        tool for tool in manager.tools if tool.tool_name == "search_memory"
    )
    add_tool = next(tool for tool in manager.tools if tool.tool_name == "add_memory")
    run(search_tool(query="bicycle"))
    run(add_tool(entries=["explicit tool memory about tea"]))
    assert [entry.content for entry in run(store.search("tea"))] == [
        "explicit tool memory about tea"
    ]


def test_raw_replay_and_distinct_identical_turns(tmp_path):
    store = make_store(tmp_path, ingestion="raw", agent_run_id="run-1")
    messages = [
        {"role": "user", "content": [{"text": "ok"}]},
        {"role": "assistant", "content": [{"text": "ok"}]},
    ]
    context = AddMessagesContext(sequence_numbers=[0, 1])
    first = run(store.add_messages(messages, context))
    second = run(
        make_store(tmp_path, ingestion="raw", agent_run_id="run-1").add_messages(
            messages, context
        )
    )
    assert first == second
    rows = run(store.search("ok", {"max_search_results": 10}))
    assert len(rows) == 2
    assert {entry.metadata["role"] for entry in rows} == {"user", "assistant"}
    with pytest.raises(ReplayConflictError):
        run(
            store.add_messages(
                [{"role": "user", "content": [{"text": "changed"}]}],
                AddMessagesContext([0]),
            )
        )


def test_tenant_filter_and_fact_history(tmp_path):
    alice = make_store(tmp_path, tenant="alice", update_strategy="fact_key")
    bob = make_store(tmp_path, tenant="bob", update_strategy="fact_key")
    run(
        alice.add(
            "favorite color blue",
            {"fact_key": "color", "valid_from": "2020-01-01T00:00:00Z"},
        )
    )
    run(
        alice.add(
            "favorite color green",
            {"fact_key": "color", "valid_from": "2021-01-01T00:00:00Z"},
        )
    )
    run(bob.add("favorite color red", {"fact_key": "color"}))
    current = run(alice.search("favorite color"))
    assert current[0].content == "favorite color green"
    assert {entry.content for entry in current} == {
        "favorite color blue",
        "favorite color green",
    }
    assert [entry.content for entry in run(alice.history("color"))] == [
        "favorite color green",
        "favorite color blue",
    ]
    old = run(alice.search_as_of("favorite color", datetime(2020, 6, 1, tzinfo=UTC)))
    assert [entry.content for entry in old] == ["favorite color blue"]
    assert [entry.content for entry in run(bob.search("favorite color"))] == [
        "favorite color red"
    ]


def test_backfill_does_not_override_current_fact(tmp_path):
    store = make_store(tmp_path, update_strategy="fact_key")
    run(
        store.add(
            "home city Phoenix",
            {"fact_key": "city", "valid_from": "2022-01-01T00:00:00Z"},
        )
    )
    run(
        store.add(
            "home city Boston",
            {"fact_key": "city", "valid_from": "2018-01-01T00:00:00Z"},
        )
    )
    assert run(store.search("home city"))[0].content == "home city Phoenix"
    assert [entry.content for entry in run(store.history("city"))] == [
        "home city Phoenix",
        "home city Boston",
    ]


def test_old_wording_resolves_to_current_fact(tmp_path):
    store = make_store(tmp_path, update_strategy="fact_key")
    run(
        store.add(
            "favorite drink coffee",
            {"fact_key": "drink", "valid_from": "2020-01-01T00:00:00Z"},
        )
    )
    run(
        store.add(
            "favorite drink tea",
            {"fact_key": "drink", "valid_from": "2022-01-01T00:00:00Z"},
        )
    )
    current = run(store.search("coffee", {"max_search_results": 1}))
    assert [entry.content for entry in current] == ["favorite drink tea"]
    historical = run(
        store.search_as_of("coffee", datetime(2021, 1, 1, tzinfo=UTC), limit=1)
    )
    assert [entry.content for entry in historical] == ["favorite drink coffee"]


def test_fact_idempotency_and_conflict(tmp_path):
    store = make_store(tmp_path)
    first = run(store.add("works in Arizona"))
    assert run(store.add("works in Arizona")) == first
    assert len(run(store.search("Arizona"))) == 1
    run(store.add("likes tea", {"idempotency_key": "event-1"}))
    with pytest.raises(ReplayConflictError):
        run(store.add("likes coffee", {"idempotency_key": "event-1"}))
    with pytest.raises(ReplayConflictError):
        run(
            store.add(
                "likes tea", {"idempotency_key": "event-1", "category": "changed"}
            )
        )
    with pytest.raises(ValueError, match="JSON serializable"):
        run(store.add("invalid metadata", {"value": object()}))


def test_failure_modes(tmp_path):
    with pytest.raises(StoreNotInitializedError):
        run(make_store(tmp_path, writable=False).initialize())
    with pytest.raises(ValueError, match="embedder"):
        make_store(tmp_path, retrieval="hybrid_rrf")
    with pytest.raises(MemoryStorageError, match="remote"):
        LanceDBMemoryStore(uri="s3://bucket/path", tenant_id="alice")
    store = make_store(tmp_path)
    run(store.initialize())
    db = lancedb.connect(str(store.uri))
    db.open_table("memories").drop_index("content_idx")
    with pytest.raises(IndexUnavailableError):
        run(make_store(tmp_path, writable=False).initialize())
    run(store.initialize())
    with pytest.raises(ValueError, match="max_search_results"):
        run(store.search("word", {"max_search_results": 0}))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"tenant": " "}, "tenant_id"),
        ({"name": " "}, "name"),
        ({"max_search_results": 0}, "max_search_results"),
        ({"retrieval": "unknown"}, "retrieval"),
        ({"ingestion": "unknown"}, "ingestion"),
        ({"update_strategy": "unknown"}, "update strategy"),
        ({"vector_weight": -1}, "vector_weight"),
        ({"recency_weight": -1}, "recency_weight"),
        ({"ingestion": "raw"}, "agent_run_id"),
        ({"ingestion": "facts", "writable": False}, "writable"),
    ],
)
def test_invalid_configuration(tmp_path, kwargs, error):
    with pytest.raises(ValueError, match=error):
        make_store(tmp_path, **kwargs)


def test_write_and_timestamp_validation(tmp_path):
    store = make_store(tmp_path, update_strategy="fact_key")
    with pytest.raises(ValueError, match="content"):
        run(store.add("  "))
    with pytest.raises(ValueError, match="fact_key"):
        run(store.add("content"))
    with pytest.raises(ValueError, match="fact_key"):
        run(store.add("content", {"fact_key": 123}))
    with pytest.raises(ValueError, match="idempotency_key"):
        run(store.add("content", {"fact_key": "a", "idempotency_key": ""}))
    with pytest.raises(ValueError, match="timezone"):
        run(store.add("content", {"fact_key": "a", "valid_from": datetime(2020, 1, 1)}))
    with pytest.raises(ValueError, match="ISO-8601"):
        run(store.add("content", {"fact_key": "a", "valid_from": "not-a-date"}))
    point = "2020-01-01T00:00:00.123456+00:00"
    run(store.add("precise instant", {"fact_key": "a", "valid_from": point}))
    assert run(store.search("precise"))[0].metadata["valid_from"] == point
    assert run(store.search_as_of(" ", datetime.now(UTC))) == []
    with pytest.raises(ValueError, match="limit"):
        run(store.search_as_of("precise", datetime.now(UTC), limit=0))
    with pytest.raises(ValueError, match="fact_key"):
        run(store.history(""))


def test_raw_input_validation_and_read_only(tmp_path):
    store = make_store(tmp_path, ingestion="raw", agent_run_id="run")
    message = {"role": "user", "content": [{"text": "hello"}]}
    with pytest.raises(ValueError, match="sequence_numbers"):
        run(store.add_messages([message]))
    with pytest.raises(ValueError, match="sequence_numbers"):
        run(store.add_messages([message, message], AddMessagesContext([0, 0])))
    with pytest.raises(ValueError, match="nonnegative"):
        run(store.add_messages([message], AddMessagesContext([-1])))
    assert (
        run(
            store.add_messages(
                [{"role": "system", "content": [{"text": "ignored"}]}],
                AddMessagesContext([1]),
            )
        )
        == []
    )
    assert (
        run(
            store.add_messages(
                [{"role": "user", "content": [{"image": "opaque"}]}],
                AddMessagesContext([2]),
            )
        )
        == []
    )
    no_run_id = make_store(tmp_path)
    with pytest.raises(ValueError, match="agent_run_id"):
        run(no_run_id.add_messages([message], AddMessagesContext([0])))
    run(store.initialize())
    read_only = make_store(tmp_path, writable=False)
    with pytest.raises(ValueError, match="read-only"):
        run(read_only.add("content"))
    with pytest.raises(ValueError, match="read-only"):
        run(read_only.add_messages([message], AddMessagesContext([0])))
    with pytest.raises(ValueError, match="read-only"):
        run(read_only.optimize())


class FailingEmbedder:
    dimension = 3
    model_id = "failing-v1"

    def embed_documents(self, texts):
        raise RuntimeError("model unavailable")

    def embed_query(self, text):
        raise RuntimeError("model unavailable")


def test_embedding_failures(tmp_path):
    store = make_store(tmp_path, retrieval="vector", embedder=FailingEmbedder())
    with pytest.raises(EmbeddingError, match="document"):
        run(store.add("anything"))
    run(store.initialize())
    with pytest.raises(EmbeddingError, match="query"):
        run(store.search("anything"))


def test_async_writer_reader_concurrency(tmp_path):
    store = make_store(tmp_path)

    async def exercise():
        writes = [store.add(f"concurrent record {i}") for i in range(25)]
        await asyncio.gather(*writes)
        reads = await asyncio.gather(
            *(store.search("concurrent", {"max_search_results": 30}) for _ in range(3))
        )
        return reads

    results = run(exercise())
    assert all(len(batch) == 25 for batch in results)


class FakeEmbedder:
    dimension = 3
    model_id = "deterministic-test-v1"

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float("blue" in text), float("green" in text), 1.0]


@pytest.mark.parametrize("mode", ["vector", "hybrid_rrf"])
def test_vector_modes(tmp_path, mode):
    store = make_store(tmp_path, retrieval=mode, embedder=FakeEmbedder())
    run(store.add("blue bicycle"))
    run(store.add("green bicycle"))
    results = run(store.search("blue bicycle", {"max_search_results": 2}))
    assert len(results) == 2
    assert results[0].content == "blue bicycle"
    with pytest.raises(SchemaMismatchError):
        run(make_store(tmp_path, retrieval="fts", embedder=None).initialize())


@settings(max_examples=12, deadline=None)
@given(
    st.text(min_size=1, max_size=35).filter(
        lambda value: value.strip() and value != "other"
    )
)
def test_adversarial_tenant_ids(tenant):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        own = make_store(root, tenant=tenant)
        other = make_store(root, tenant="other")
        run(own.add("private purple bicycle"))
        run(other.add("public purple bicycle"))
        assert [entry.content for entry in run(own.search("purple bicycle"))] == [
            "private purple bicycle"
        ]
