"""Verify bulk eval seeding preserves the store's search-visible session metadata."""

import asyncio

from evals.vector_dev import _seed_case
from strands_lancedb_memory import LanceDBMemoryStore


class TinyEmbedder:
    dimension = 3
    model_id = "tiny-test-v1"

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float("blue" in text), float("green" in text), 1.0]


def test_batch_seed_keeps_duplicate_session_occurrences(tmp_path):
    embedder = TinyEmbedder()
    store = LanceDBMemoryStore(
        uri=tmp_path / "db", tenant_id="case", retrieval="hybrid_rrf", embedder=embedder
    )
    case = {
        "question_id": "case",
        "haystack_session_ids": ["same", "same"],
        "haystack_dates": ["2020/01/01 (Wed) 00:00", "2020/01/02 (Thu) 00:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "blue bicycle"}],
            [{"role": "user", "content": "green bicycle"}],
        ],
    }
    assert _seed_case(store, case, embedder) == 2
    entries = asyncio.run(store.search("bicycle", {"max_search_results": 2}))
    assert len(entries) == 2
    assert {entry.metadata["session_id"] for entry in entries} == {"same"}
    assert {entry.metadata["session_ordinal"] for entry in entries} == {0, 1}
