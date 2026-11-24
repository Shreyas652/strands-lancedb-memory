# Phase 2 implementation report

## What I did

Implemented a local, tenant-scoped `LanceDBMemoryStore` for Strands 1.65.0 and LanceDB 0.41.0. It supports explicit FTS, vector, and weighted reciprocal rank fusion retrieval; direct facts and raw message ingestion; deterministic replay IDs; keyed fact history and as-of search; explicit optimization; and schema/embedder validation. It never downloads an embedding model or calls a paid API on its own.

The Arrow schema uses integer UTC microseconds because a timezone timestamp scalar failed to decode on the pinned Windows PyArrow runtime without an additional timezone package. A newer keyed fact is included and prioritized when a query matches only an older wording. Local writable operations use an OS file lock after the initial two-process probe found a collision.

## Verification

On Windows 11, Python 3.13.14, the following passed:

```text
uv sync --locked
uv run pytest tests --cov=strands_lancedb_memory --cov-fail-under=90 -q
  27 passed; 94.49% line coverage
uv run ruff check src tests experiments
  All checks passed
uv run ruff format --check src tests experiments
  8 files already formatted
uv run mypy --strict src/strands_lancedb_memory
  Success: no issues found in 2 source files
```

The initial process probe recorded 10/20 rows and one failed writer (`results/phase2_cross_process_before_lock.json`). The post-lock probe recorded 20/20 rows, both writers successful (`results/phase2_cross_process_probe.json`). The index lifecycle probe recorded empty-table index creation, later writes remaining searchable, and reopen searchability (`results/phase2_lancedb_probe.json`). Tests also cover a real Strands Agent using a stub model, tools, tenant isolation, replay, fact versioning, persistence across a new process, and concurrent async writes.

## Decisions and limits

The core package requires an explicit caller embedder for vector/hybrid modes. FTS is a zero-model local path. `s3://` is refused until tested. The local file lock serializes writes and writable queries across processes on one machine; it has not been tested on network filesystems. See `docs/LIMITATIONS.md` for the full current boundary.

I did not run a LongMemEval split, measure answer accuracy, test 1k/10k/100k scale, verify S3, or run Python 3.11/3.12/Linux CI locally. The matrix workflow is included for GitHub to execute after push. No paid model calls were made; cost so far is **$0, 0 calls**.
