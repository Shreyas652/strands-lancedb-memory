"""Fact retrieval ablation preserves source and version information."""

import asyncio

from evals.facts_retrieval_dev import _rows, _seed
from strands_lancedb_memory import LanceDBMemoryStore


def test_keyed_fact_ablation_keeps_newer_observation(tmp_path):
    case = {
        "question_id": "case",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2024/01/01 (Mon) 00:00", "2025/01/01 (Wed) 00:00"],
    }
    facts = [
        {
            "session_ordinal": 0,
            "key": "favorite-color",
            "fact": "Favorite color was blue.",
        },
        {
            "session_ordinal": 1,
            "key": "favorite-color",
            "fact": "Favorite color is red.",
        },
    ]
    store = LanceDBMemoryStore(
        uri=tmp_path / "db", tenant_id="case", update_strategy="fact_key"
    )
    rows = _rows(case, facts, keyed=True)
    assert {row["fact_key"] for row in rows} == {"favorite-color"}
    _seed(store, rows)
    result = asyncio.run(store.search("Favorite color", {"max_search_results": 2}))
    assert len(result) == 2
    assert result[0].content == "Favorite color is red."
    assert [entry.metadata["session_ordinal"] for entry in result] == [1, 0]
