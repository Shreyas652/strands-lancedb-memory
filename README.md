# strands-lancedb-memory

**Status: Phase 2 implementation under review.** The local LanceDB `MemoryStore` is implemented and tested offline. No LongMemEval accuracy or scale claims have been produced in this repository yet.

The package connects [Strands Agents](https://github.com/strands-agents/harness-sdk) memory to embedded [LanceDB](https://github.com/lancedb/lancedb), with tenant-scoped local storage, explicit ingestion modes, and configurable FTS, vector, or weighted reciprocal rank fusion search. Vector modes require a caller-supplied embedder. Remote `s3://` storage is currently rejected because it has not been validated.

## Local quickstart

```python
import asyncio
from strands_lancedb_memory import LanceDBMemoryStore

async def main():
    store = LanceDBMemoryStore(uri="./memory-db", tenant_id="example-user")
    await store.initialize()
    await store.add("My favorite color is blue")
    for entry in await store.search("favorite color"):
        print(entry.content)

asyncio.run(main())
```

Pass this store to `strands.memory.MemoryManager(stores=[store])` to enable Agent search and injection. Set `ingestion="raw"` with a stable `agent_run_id` to persist raw turns, or `ingestion="facts"` to use Strands' model extractor. Fact extraction may call the configured model and is not exercised by the offline test suite.

## Verify Phase 2

```powershell
uv sync --locked
uv run pytest tests --cov=strands_lancedb_memory --cov-fail-under=90
uv run ruff check src tests experiments
uv run mypy --strict src/strands_lancedb_memory
uv run python experiments/phase2_cross_process_probe.py
```

The two-process probe saves its recorded result in `results/phase2_cross_process_probe.json` when invoked with `--save results/phase2_cross_process_probe.json`. Its initial run without a process lock stored 10 of 20 rows and one writer failed. After adding a local file lock, both writers succeeded and all 20 rows were present; this is a bounded correctness probe, not a throughput or arbitrary multi-host guarantee. See [current limitations](docs/LIMITATIONS.md).

## Review the evidence

- [Installed API notes](docs/API_NOTES.md): exact Strands 1.65.0 and LanceDB 0.41.0 signatures and source lines.
- [Prior art](docs/PRIOR_ART.md): MongoDB, Pixeltable, Neo4j and Strands file-store design patterns.
- [Agent lifecycle](docs/LIFECYCLE.md): real Agent with a stub model and logged memory calls.

[Design proposal](docs/DESIGN.md): alternatives and the original Phase 1 rationale. The implementation evidence and remaining limits are recorded separately.

## Reproduce Phase 0

```powershell
uv sync --locked
uv run python experiments/api_inventory.py
uv run python experiments/phase0_lifecycle.py
uv run python experiments/lancedb_api_smoke.py
```

The generated JSON artifacts are in `results/`. These commands make no paid model calls. The LanceDB smoke uses a temporary local directory and does not claim S3 or concurrency support.
