# strands-lancedb-memory

**Status: implemented and benchmarked.** On the 400 held-out LongMemEval-S questions, the frozen LanceDB FTS configuration answered 201 correctly with a pinned GPT-4o judge, versus 186 for Strands' FileMemoryStore and 27 without memory. The 100-question development split selected this configuration before the held-out evaluation. The 255/500 combined total below is descriptive because it includes those 100 development questions.

| Metric | Value | JSON source |
| --- | ---: | --- |
| LanceDB FTS answer accuracy, held-out 400 | 50.25% | `results/frozen_answers.json#/summary/lancedb_fts:3/accuracy` |
| FileMemoryStore answer accuracy, held-out 400 | 46.5% | `results/frozen_answers.json#/summary/file:3/accuracy` |
| No-memory answer accuracy, held-out 400 | 6.75% | `results/frozen_answers.json#/summary/no_memory:0/accuracy` |
| LanceDB FTS answer accuracy, descriptive all 500 | 51.0% | `results/all_500_summary.json#/summary/lancedb_fts:3/accuracy` |
| FileMemoryStore answer accuracy, descriptive all 500 | 48.0% | `results/all_500_summary.json#/summary/file:3/accuracy` |
| No-memory answer accuracy, descriptive all 500 | 6.0% | `results/all_500_summary.json#/summary/no_memory:0/accuracy` |

The 201/400 versus 186/400 difference is not statistically significant at the 5% level (exact two-sided McNemar p = 0.063). This is not a general superiority claim.

| Paired held-out metric | Value | JSON source |
| --- | ---: | --- |
| FTS-only correct | 36 | `results/paired_significance.json#/fts_only_correct` |
| File-only correct | 21 | `results/paired_significance.json#/file_only_correct` |
| Exact two-sided McNemar p | 0.063 | `results/paired_significance.json#/exact_mcnemar_two_sided_p` |
| Paired accuracy difference, points | 3.75 | `results/paired_significance.json#/paired_accuracy_difference_points` |

| Question type, all 500 | LanceDB FTS | FileMemoryStore | Questions |
| --- | ---: | ---: | ---: |
| Knowledge update | 44 | 48 | 78 |
| Multi-session | 42 | 40 | 133 |
| Assistant statement | 54 | 52 | 56 |
| Preference | 2 | 1 | 30 |
| User statement | 64 | 56 | 70 |
| Temporal reasoning | 49 | 43 | 133 |

These are results from this repository's full-session memory and `gpt-4o-mini-2024-07-18` reader, graded with the official LongMemEval prompt using pinned `gpt-4o-2024-08-06`. They are **not an official leaderboard score**. The 15-question held-out advantage over FileMemoryStore is modest; the benchmark does not establish a general performance ranking. The main result does not measure Strands' own fact extraction. The 49/50 second-judge agreement used `gpt-4o-mini-2024-07-18`, the same model as the reader, so it is a weak independence check. Future work: re-judge the same 50-question sample with a different model family. See [the protocol and raw reports](docs/EXPERIMENTS.md).

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

Pass this store to `strands.memory.MemoryManager(stores=[store])` to enable Agent search and injection. Set `ingestion="raw"` with a stable `agent_run_id` to persist raw turns, or `ingestion="facts"` to use Strands' model extractor. In raw mode, create **one store instance and a distinct run ID per Agent**; Strands resets its sequence numbers for each Agent. Reuse an ID only to replay that same Agent run. For retries interleaved with later fact updates, pass a stable `metadata.idempotency_key`. Fact extraction may call the configured model and is not exercised by the offline test suite.

The runnable [offline Agent example](examples/local_agent.py) uses a stub model and a temporary local directory. It tests integration without a cloud account; replace the stub with your configured Strands model for real answers.

## Configuration

| Argument | Default | Meaning |
| --- | --- | --- |
| `uri` | required | Local LanceDB directory. Remote URIs are rejected. |
| `tenant_id` | required | Required filter on every read and write. Use a stable ID per user or isolation boundary. |
| `writable` | `True` | Read-only mode requires an existing table and index. |
| `max_search_results` | `5` | Default result limit; Strands may override it per search. |
| `retrieval` | `"fts"` | `"fts"`, `"vector"`, or `"hybrid_rrf"`. Vector modes require `embedder`. |
| `embedder` | `None` | Caller-supplied synchronous `Embedder` with `dimension`, `model_id`, `embed_documents`, and `embed_query`. |
| `vector_weight` | `0.5` | Vector-side weight in weighted RRF; used only for hybrid search. |
| `ingestion` | `"off"` | `"off"` for explicit writes, `"facts"` for Strands extraction, or `"raw"` for raw Agent messages. |
| `agent_run_id` | `None` | Required stable replay identity when `ingestion="raw"`. Use a new value for a new run. |
| `update_strategy` | `"append"` | `"fact_key"` gives keyed facts current-value ranking and history. |
| `recency_weight` | `0.0` | Optional nonnegative recency boost. |

Writes and searches fail explicitly if storage, indexing, or embedding fails. They never switch to another backend automatically. See [design](docs/DESIGN.md) and [limitations](docs/LIMITATIONS.md) for concurrency and update semantics.

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

## Synthetic local scale results

The script `experiments/scale_memory.py` bulk seeds normalized 32-dimensional synthetic vectors, then searches through the public store API before and after building an IVF_FLAT index. Each size uses 30 fixed-seed probes on a 12-logical-CPU Windows machine. Insertion throughput here is for direct LanceDB bulk seeding, **not** the store's public `add` method. Indexed recall compares top-10 results with exact search. The index was slower at 1k and 10k; at 100k it was slightly faster but returned only 78% of exact top-10 neighbors, so it is not enabled by default.

| Metric | Value | JSON source |
| --- | ---: | --- |
| 1k exact p50 search, ms | 16.7 | `results/scale_1k.json#/exact_p50_search_ms` |
| 1k indexed p50 search, ms | 18.9 | `results/scale_1k.json#/indexed_p50_search_ms` |
| 1k indexed recall@10 | 95.3% | `results/scale_1k.json#/indexed_recall_at_10` |
| 10k exact p50 search, ms | 23.8 | `results/scale_10k.json#/exact_p50_search_ms` |
| 10k indexed p50 search, ms | 24.8 | `results/scale_10k.json#/indexed_p50_search_ms` |
| 10k indexed recall@10 | 85.3% | `results/scale_10k.json#/indexed_recall_at_10` |
| 100k exact p50 search, ms | 66.9 | `results/scale_100k.json#/exact_p50_search_ms` |
| 100k indexed p50 search, ms | 62.9 | `results/scale_100k.json#/indexed_p50_search_ms` |
| 100k indexed recall@10 | 78.0% | `results/scale_100k.json#/indexed_recall_at_10` |

The JSON files include hardware, pinned library versions, script hash, raw per-query timings and rankings, and the index build time. These are synthetic search measurements, not LongMemEval recall or answer accuracy.

## LongMemEval-S development retrieval

The public [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) dataset was split with a fixed, question-type-stratified seed into 100 development and 400 sealed test questions. On the development split, each store indexed the same full sessions. A hit means at least one labeled evidence session appears in the first five results. The reader and judge were not involved in these retrieval numbers; their answer-quality result is below.

| Metric | Value | JSON source |
| --- | ---: | --- |
| LanceDB FTS dev evidence hit@5 | 97.0% | `results/dev_retrieval.json#/summary/lancedb_fts/hit_at_5` |
| FileMemoryStore dev evidence hit@5 | 94.0% | `results/dev_retrieval.json#/summary/file/hit_at_5` |
| TestMemoryStore dev evidence hit@5 | 93.0% | `results/dev_retrieval.json#/summary/test/hit_at_5` |
| LanceDB FTS dev p50 search, ms | 608.7 | `results/dev_retrieval.json#/summary/lancedb_fts/p50_search_ms` |
| FileMemoryStore dev p50 search, ms | 1025.9 | `results/dev_retrieval.json#/summary/file/p50_search_ms` |
| TestMemoryStore dev p50 search, ms | 54.2 | `results/dev_retrieval.json#/summary/test/p50_search_ms` |

The optional local `BAAI/bge-small-en-v1.5` model was also tested on the same 100 dev questions with a fingerprinted cached ONNX file. Weighted hybrid at vector weight 0.30 tied FTS on hit@5; equal weighting and vector-only were lower. These are retrieval comparisons only, and run-time conditions differed across the two jobs.

| Metric | Value | JSON source |
| --- | ---: | --- |
| Weighted hybrid (vector weight 0.30) dev hit@5 | 97.0% | `results/vector_dev.json#/summary/hybrid_rrf:0.30/hit_at_5` |
| Equal hybrid (vector weight 0.50) dev hit@5 | 96.0% | `results/vector_dev.json#/summary/hybrid_rrf:0.50/hit_at_5` |
| Vector-only dev hit@5 | 93.0% | `results/vector_dev.json#/summary/vector:1.00/hit_at_5` |

The split manifest (`results/longmemeval_dev_split.json`) records the dataset fingerprint and development IDs without publishing test IDs. Every retrieval row has the question ID, selected session IDs and ordinals, evidence rank, injected context size, and measured latency. The configuration was frozen before the held-out evaluation.

## LongMemEval-S development answers

The same `gpt-4o-mini-2024-07-18` reader answered every question at temperature 0. The official LongMemEval judge prompt, verified against the source at commit `9e0b455f4ef0e2ab8f2e582289761153549043fc`, was run with pinned `gpt-4o-2024-08-06`. These are **development results under our reader and judge setup**, not an official leaderboard submission. Each memory store used its first five results, and every answer and judgment is in `results/dev_answers.json`.

| Metric | Value | JSON source |
| --- | ---: | --- |
| No memory dev answer accuracy | 3.0% | `results/dev_answers.json#/summary/no_memory:0/accuracy` |
| FileMemoryStore dev answer accuracy, k=5 | 55.0% | `results/dev_answers.json#/summary/file:5/accuracy` |
| LanceDB FTS dev answer accuracy, k=5 | 53.0% | `results/dev_answers.json#/summary/lancedb_fts:5/accuracy` |
| TestMemoryStore dev answer accuracy, k=5 | 45.0% | `results/dev_answers.json#/summary/test:5/accuracy` |

LanceDB FTS had a higher evidence hit@5 than FileMemoryStore on this split, but two fewer correct answers at top 5. Retrieval hit rate alone did not predict answer quality. Both memory stores strongly improved on the no-memory arm. The dev set is only 100 questions; the frozen top-3 configuration and its held-out result are reported above.

## Reproduce Phase 0

```powershell
uv sync --locked
uv run python experiments/api_inventory.py
uv run python experiments/phase0_lifecycle.py
uv run python experiments/lancedb_api_smoke.py
```

The generated JSON artifacts are in `results/`. These commands make no paid model calls. The LanceDB smoke uses a temporary local directory and does not claim S3 or concurrency support.
