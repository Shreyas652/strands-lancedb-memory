"""Measure local LanceDB memory search at 1k, 10k, and 100k synthetic rows.

Rows are bulk seeded through LanceDB so this measures bulk load throughput,
not the public store's ``add`` throughput. Search uses the public store API.
Synthetic vectors isolate index behavior from embedding-model quality.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import statistics
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from lancedb.index import IvfFlat

from strands_lancedb_memory import LanceDBMemoryStore

DIMENSION = 32


class QueryVectorEmbedder:
    """Maps synthetic probe IDs to vectors that were actually inserted."""

    dimension = DIMENSION
    model_id = "synthetic-normalized-gaussian-32-v1"

    def __init__(self) -> None:
        self.probes: dict[str, list[float]] = {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("bulk seeding supplies vectors directly")

    def embed_query(self, text: str) -> list[float]:
        return self.probes[text]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    at = (len(ordered) - 1) * fraction
    lower = math.floor(at)
    return ordered[lower] + (
        ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]
    ) * (at - lower)


def _seed(
    store: LanceDBMemoryStore, rows: int, batch_size: int, queries: int, seed: int
) -> tuple[float, list[str]]:
    rng = np.random.default_rng(seed)
    selected = rng.choice(rows, size=queries, replace=False)
    selected_set = {int(value) for value in selected}
    embedder = store.embedder
    assert isinstance(embedder, QueryVectorEmbedder)
    table = store._open_table()
    if table.count_rows() != 0:
        raise ValueError("scale run needs a fresh database")
    timestamp = int(datetime.now(UTC).timestamp() * 1_000_000)
    started = time.perf_counter()
    for offset in range(0, rows, batch_size):
        count = min(batch_size, rows - offset)
        vectors = rng.normal(size=(count, DIMENSION)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        batch = []
        for local, vector in enumerate(vectors):
            ordinal = offset + local
            probe = f"probe:{ordinal}"
            if ordinal in selected_set:
                embedder.probes[probe] = vector.tolist()
            batch.append(
                {
                    "entry_id": f"synthetic-{ordinal:09d}",
                    "tenant_id": store.tenant_id,
                    "content": f"Synthetic memory {ordinal}",
                    "role": None,
                    "run_id": None,
                    "sequence_number": None,
                    "fact_key": None,
                    "valid_from": timestamp,
                    "recorded_at": timestamp,
                    "source": "fact",
                    "metadata_json": "{}",
                    "vector": vector.tolist(),
                }
            )
        table.add(pa.Table.from_pylist(batch, schema=store._schema()))
    elapsed = time.perf_counter() - started
    if table.count_rows() != rows or len(embedder.probes) != queries:
        raise AssertionError("bulk seed did not preserve all rows or probes")
    return elapsed, [f"probe:{int(value)}" for value in selected]


async def _search(
    store: LanceDBMemoryStore, probes: list[str]
) -> tuple[list[float], list[list[str]]]:
    durations = []
    rankings = []
    await store.search(probes[0], {"max_search_results": 10})
    for probe in probes:
        started = time.perf_counter()
        found = await store.search(probe, {"max_search_results": 10})
        durations.append((time.perf_counter() - started) * 1000)
        rankings.append([str(entry.metadata["entry_id"]) for entry in found])
    return durations, rankings


def run(
    *, rows: int, queries: int, batch_size: int, partitions: int, seed: int
) -> dict[str, Any]:
    if (
        min(rows, queries, batch_size, partitions) < 1
        or queries > rows
        or partitions > rows
    ):
        raise ValueError(
            "require 1 <= queries, partitions <= rows and positive batch size"
        )
    embedder = QueryVectorEmbedder()
    with tempfile.TemporaryDirectory(prefix="strands-lmem-scale-") as temporary:
        location = Path(temporary)
        store = LanceDBMemoryStore(
            uri=location / "db",
            tenant_id="scale-user",
            retrieval="vector",
            embedder=embedder,
        )
        asyncio.run(store.initialize())
        ingest_seconds, probes = _seed(store, rows, batch_size, queries, seed)
        exact_ms, exact_ids = asyncio.run(_search(store, probes))
        table = store._open_table()
        started = time.perf_counter()
        table.create_index("vector", config=IvfFlat(num_partitions=partitions))
        index_seconds = time.perf_counter() - started
        if not any("vector" in item.columns for item in table.list_indices()):
            raise AssertionError("IVF_FLAT index was not built")
        indexed_ms, indexed_ids = asyncio.run(_search(store, probes))
        bytes_on_disk = sum(
            path.stat().st_size for path in location.rglob("*") if path.is_file()
        )
    return {
        "status": "COMPLETE",
        "rows": rows,
        "queries": queries,
        "seed": seed,
        "dimension": DIMENSION,
        "index": "IVF_FLAT",
        "partitions": partitions,
        "batch_size": batch_size,
        "versions": {
            "lancedb": version("lancedb"),
            "strands_agents": version("strands-agents"),
            "numpy": version("numpy"),
        },
        "hardware": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bulk_seed_seconds": ingest_seconds,
        "bulk_seed_rows_per_second": rows / ingest_seconds,
        "index_build_seconds": index_seconds,
        "bytes_on_disk": bytes_on_disk,
        "exact_p50_search_ms": statistics.median(exact_ms),
        "exact_p95_search_ms": _percentile(exact_ms, 0.95),
        "indexed_p50_search_ms": statistics.median(indexed_ms),
        "indexed_p95_search_ms": _percentile(indexed_ms, 0.95),
        "indexed_recall_at_10": sum(
            len(set(expected) & set(actual)) / len(expected)
            for expected, actual in zip(exact_ids, indexed_ids, strict=True)
        )
        / queries,
        "indexed_top_1_agreement": sum(
            expected[0] == actual[0]
            for expected, actual in zip(exact_ids, indexed_ids, strict=True)
        )
        / queries,
        "per_query": [
            {
                "probe": probe,
                "exact_search_ms": before,
                "indexed_search_ms": after,
                "exact_top_10": expected,
                "indexed_top_10": actual,
            }
            for probe, before, after, expected, actual in zip(
                probes, exact_ms, indexed_ms, exact_ids, indexed_ids, strict=True
            )
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--partitions", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        rows=args.rows,
        queries=args.queries,
        batch_size=args.batch_size,
        partitions=args.partitions,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps({key: value for key, value in report.items() if key != "per_query"})
    )


if __name__ == "__main__":
    main()
