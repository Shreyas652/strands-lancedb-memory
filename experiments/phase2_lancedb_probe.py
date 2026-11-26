"""Probe empty-table and FTS lifecycle behavior in installed LanceDB 0.41.0."""

import json
import tempfile
from pathlib import Path

import lancedb
import pyarrow as pa
from lancedb.expr import col
from lancedb.index import FTS


def main() -> None:
    observations: dict[str, object] = {}
    schema = pa.schema(
        [
            pa.field("entry_id", pa.string(), nullable=False),
            pa.field("tenant_id", pa.string(), nullable=False),
            pa.field("content", pa.string(), nullable=False),
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        db = lancedb.connect(tmp)
        table = db.create_table("memories", schema=schema)
        observations["empty_count"] = table.count_rows()
        observations["table_names"] = db.list_tables().tables
        try:
            table.create_index("content", config=FTS())
            observations["empty_index"] = "created"
        except Exception as exc:  # noqa: BLE001 - probe records exact installed behavior
            observations["empty_index"] = f"{type(exc).__name__}: {exc}"
        table.add([{"entry_id": "a", "tenant_id": "alice", "content": "blue bicycle"}])
        if observations["empty_index"] != "created":
            table.create_index("content", config=FTS())
        observations["index_count_after_add"] = len(table.list_indices())
        find = lambda query: [
            row["entry_id"]
            for row in table.search(query, query_type="fts", fts_columns="content")
            .where(col("tenant_id") == "alice", prefilter=True)
            .limit(5)
            .to_list()
        ]
        observations["first_fts"] = find("bicycle")
        table.add([{"entry_id": "b", "tenant_id": "alice", "content": "green bicycle"}])
        observations["post_index_fts"] = find("green")
        observations["reopen_fts"] = [
            row["entry_id"]
            for row in lancedb.connect(tmp)
            .open_table("memories")
            .search("green", query_type="fts", fts_columns="content")
            .where(col("tenant_id") == "alice", prefilter=True)
            .limit(5)
            .to_list()
        ]
        assert observations["empty_count"] == 0
        assert observations["empty_index"] == "created"
        assert observations["first_fts"] == ["a"]
        assert observations["post_index_fts"] == ["b"]
        assert observations["reopen_fts"] == ["b"]
    Path("results/phase2_lancedb_probe.json").write_text(
        json.dumps(observations, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(observations, indent=2))


if __name__ == "__main__":
    main()
