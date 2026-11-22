"""Exercise the pinned LanceDB 0.41.0 APIs used in the Phase 1 design."""

import json
import tempfile
from pathlib import Path

import lancedb
import pyarrow as pa
from lancedb.expr import col
from lancedb.index import FTS
from lancedb.rerankers import RRFReranker


def main() -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("tenant", pa.string(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), 3), nullable=False),
        ]
    )
    rows = [
        {"id": "a", "tenant": "alice", "text": "prefers blue", "vector": [1.0, 0.0, 0.0]},
        {"id": "b", "tenant": "bob", "text": "prefers blue", "vector": [1.0, 0.0, 0.0]},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        db = lancedb.connect(tmp)
        table = db.create_table("items", data=pa.Table.from_pylist(rows, schema=schema))
        table.create_index("text", config=FTS())
        query = (
            table.search(query_type="hybrid")
            .vector([1.0, 0.0, 0.0])
            .text("blue")
            .where(col("tenant") == "alice", prefilter=True)
            .rerank(RRFReranker())
        )
        hybrid = query.limit(5).to_list()
        table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(
            pa.Table.from_pylist(
                [{"id": "a", "tenant": "alice", "text": "prefers green", "vector": [0.0, 1.0, 0.0]}],
                schema=schema,
            )
        )
        table.optimize()
        reopened = lancedb.connect(tmp).open_table("items")
        rows_after = reopened.search().where(col("tenant") == "alice").to_list()
        report = {
            "lancedb_version": lancedb.__version__,
            "hybrid_ids": [row["id"] for row in hybrid],
            "filtered_ids_after_reopen": [row["id"] for row in rows_after],
            "text_after_upsert": rows_after[0]["text"],
            "count": reopened.count_rows(),
        }
        assert report["hybrid_ids"] == ["a"], report
        assert report["filtered_ids_after_reopen"] == ["a"], report
        assert report["text_after_upsert"] == "prefers green", report
        assert report["count"] == 2, report
    output = Path("results/phase0_lancedb_api.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
