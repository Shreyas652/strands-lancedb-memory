# Design proposal — Phase 1 snapshot

Phase 2 implementation has begun at the user's direction without a phase approval pause. This document preserves the original proposal and alternatives; the implemented schema, tests, and current limits are in `src/strands_lancedb_memory/store.py`, `tests/test_store.py`, and `docs/LIMITATIONS.md`. The implementation uses `valid_from` and `recorded_at` as integer UTC microseconds, because the pinned PyArrow Windows runtime could not decode timestamp scalars without an additional timezone package. It does not store `active` or `valid_to`; current keyed versions are selected at query time. Writable local operations use an OS file lock after a two-process collision probe found lost writes without it. These changes do not alter the Phase 1 public constructor signature.

Status: **historical Phase 1 proposal**. This document records candidate decisions and test criteria. Phase 2 now has a local `MemoryStore` implementation and offline tests. Search quality at benchmark scale and S3 support remain unverified.

## Scope and invariants

The package will expose one typed `LanceDBMemoryStore` implementing the pinned Strands `MemoryStore` Protocol. All paths must bind a nonempty tenant identifier and never run an unfiltered read. The store is only responsible for retrieval, persistence, and optional raw-message ingestion; it never reads benchmark labels or scoring modules. Gold answers enter only the scoring process. Every benchmark run records dataset fingerprint, seed, configuration, model ID, temperature, prompt hash, outputs per question, latencies, and estimated billed cost. The test split is sealed before tuning.

`add` and `add_messages` are both proposed to support the manager's add tool and raw-message ablation. Strands chooses `add_messages` by default when both exist. The constructor must set an explicit extractor in its `extraction` config for fact mode; this is verified against `resolve_extraction_config.py` and covered by integration tests. Automatic extraction is off unless a caller selects a mode. A fact-mode extractor uses the caller's model via Strands `ModelExtractor`, so its API use and cost must be counted. Raw mode indexes user/assistant text, preserving role and timestamp; it does not pretend to extract facts without a model.

## Decisions and experiments

Each row states genuine alternatives, the failure/benefit tradeoff, the proposed starting choice, and the experiment that can overturn it.

| Decision | Alternatives and tradeoffs | Provisional choice | Deciding evidence |
| --- | --- | --- | --- |
| What is stored | Raw turns retain exact evidence but add noise and large indexes; session summaries reduce volume but can discard details and require a model; extracted facts are concise but extraction can miss or distort evidence; both increase cost/storage but permit ablation. | Implement raw turns and extracted facts as selectable ingestion modes, plus `add` for explicit facts; do not synthesize summaries initially. | Dev split answer accuracy and evidence-session hit rate by mode, tokens injected, storage size and ingestion cost. Keep the losing mode in results. |
| Retrieval | Vector only handles paraphrases but can miss exact names; FTS only favors terms and is the FileMemoryStore-like control; equal RRF hybrid avoids score calibration; weighted RRF may improve a dev slice but can overfit; linear score fusion requires score calibration; cross-encoder rerank costs latency/model assets. | Expose `vector`, `fts`, `hybrid_rrf`; start hybrid at equal weights. Allow a configurable weight; add a reranker only if dev gains justify it. No cross-encoder as a required dependency. | Stratified dev grid: vector, FTS, 0.5/0.5 RRF, at least two other RRF weights, with p50/p95 latency and answer accuracy. Do not pick based on test split. |
| Embeddings | Local FastEmbed can work offline but downloads a sizable model and must pin assets; caller supplied embedder avoids bundled model weight but makes setup harder; LanceDB registry simplifies wiring but hides model lifecycle/version and may download at runtime. | Core accepts a typed user supplied `Embedder`; an optional, explicitly pinned local FastEmbed adapter is the intended zero-service quickstart after dependency/model-size approval. FTS mode needs no embedder. Hybrid/vector mode fails clearly when no embedder is configured. | Verify deterministic dimensions/version, fresh install offline after prefetch, 1k latency, restart, and model-unavailable errors. Measure local asset size before requesting the additional heavy-dependency approval. |
| Fact updates | Exact caller supplied `fact_key` is deterministic but requires the caller to know a stable subject; embedding similarity is flexible but can merge unrelated facts; an LLM decision can disambiguate but adds latency/cost and nondeterminism; append-only with recency retains history but does not guarantee the current fact wins. | Version rows by explicit `fact_key` when provided, retaining history and active version. For unkeyed facts, append and apply a bounded recency tie-break, with no claim of contradiction resolution. Compare keyed versus append-only; similarity/LLM remain experiment candidates, not silent behavior. | Tests for newer/backfilled timestamps, duplicate writes, history access and mixed subjects; dev answer accuracy on temporal questions. A stable key source must be demonstrated before claiming automatic updates. |
| Time semantics | Store timestamps and recency boost; strict as-of filter allows historical questions but needs a query-time interface; leaving time to the model preserves source context but may retrieve stale facts. | Store UTC `valid_from`, `recorded_at`, optional `valid_to`; search current and historical versions with current versions ranked first, and expose a direct `search_as_of` method for strict historical filters. A modest recency boost is configurable and tuned only on dev. | Historical before/after cases, backfilled timestamps, retrieval hit rate and accuracy. Confirm default retrieval does not hide relevant history. |
| Tenant isolation | One table per tenant reduces predicate risk but creates many indexes/tables; one table with a typed tenant column is simpler and can share indexes but requires every path to filter; one database per tenant maximizes separation with heavy lifecycle overhead. | One table with a nonnullable `tenant_id` field, typed `col("tenant_id") == tenant_id` prefilter on every search, and tenant in upsert identity. No raw filter-string interpolation or implicit global tenant. | Property-based random/adversarial IDs, two-tenant same text/vectors, read/write/update/history paths; add a reopen test and query-plan checks. Reconsider per-tenant DBs if typed-filter isolation cannot be proven. |
| At-least-once idempotency | `(run_id, sequence_number)` is precise for raw replay but Strands supplies only the sequence; content hashing dedups exact facts but collapses separate `ok` turns; random UUID preserves distinct turns but duplicates replays; `merge_insert` upserts by key but does not itself create a correct key. | For raw mode require a caller supplied stable `agent_run_id` unique to an Agent lifecycle; deterministic ID = hash of tenant, run ID and sequence number, then `merge_insert`. Same run ID must be reused on replay. Distinct runs may both contain `ok`. For `add` facts, use caller `idempotency_key` when supplied, otherwise hash `(tenant, normalized exact content)` to deduplicate retries; the same exact fact is then treated as one logical fact. This policy is **not** used for raw turns. | Replay identical batch across a new process with same run ID; two `ok` turns with different sequence/run IDs; conflicting payload under same ID must fail, not overwrite; simulate partial batch write/retry. Test repeated extracted `add` separately, and document that exact duplicate explicit facts collapse. |
| Index maintenance | Rebuild on each write keeps freshness but likely ruins ingest throughput; build once plus periodic optimize amortizes work but may leave stale/fragmented indexes; background maintenance lowers foreground latency but creates shutdown/concurrency complexity. | Create FTS index when first needed on a writable table; append without per-write rebuild; offer explicit `optimize()` as a maintenance operation, with conservative cleanup settings. Do not start a background thread by default. | Search before/after index creation and for newly appended rows, 1k/10k/100k ingestion and latency, reopen, and concurrent-reader tests. If new rows are absent from FTS, rebuild or a documented dual path is required before release. |
| Concurrency | One process with an async lock is simple but does not coordinate other processes and locks can bind to a closed event loop; a process-wide thread lock plus `to_thread` protects concurrent tasks/loops but not processes; a cross-process file lock can serialize local writers but not S3 writers; relying on LanceDB's native conflict handling requires evidence. | Use `asyncio.to_thread` for blocking I/O and a path-scoped process-wide lock for local writes. Test LanceDB's cross-process behavior before choosing a file lock or declaring a single-writer limit. Never describe untested S3 multi-writer operation as safe. | N async writers/readers and two-process collision test with row counts and no lost IDs; repeat across fresh Agent event loops. Measure lock contention. |
| Storage target | Local directory is embedded and testable in CI; `s3://` supports shared persistence but brings auth, object-store consistency and writer coordination; an S3 emulator covers API shape but not AWS behavior. | Require local path support. Accept S3 only after a tested emulator round trip and a clearly documented single-writer/consistency policy; if evidence fails, ship local first and mark S3 unsupported rather than silently substitute local storage. | Temporary local directory CI, emulator create/write/reopen/read/failure modes, optional real S3 credentialed smoke outside CI. Record backend/version and never embed credentials. |

## Proposed public API

This signature is for review, not a committed API. No `MemoryStore` implementation has been written.

```python
LanceDBMemoryStore(
    *,
    uri: str | Path,
    tenant_id: str,
    name: str = "lancedb",
    description: str | None = None,
    writable: bool = True,
    max_search_results: int = 5,
    retrieval: Literal["fts", "vector", "hybrid_rrf"] = "fts",
    embedder: Embedder | None = None,
    vector_weight: float = 0.5,
    ingestion: Literal["off", "facts", "raw"] = "off",
    agent_run_id: str | None = None,
    update_strategy: Literal["append", "fact_key"] = "append",
    recency_weight: float = 0.0,
)
```

`Embedder` will expose `embed_documents(texts)` and `embed_query(text)` with declared dimension and model/version identity. Input validation rejects blank tenant/name, invalid result limit/weights, vector modes without an embedder, raw ingestion without `agent_run_id`, and an S3 URI if that backend is not verified. `extraction` will be an actual `MemoryStore` attribute derived from `ingestion` (and an optional explicit trigger configuration if needed), rather than a second ambiguous constructor switch. We will define the exact `metadata` keys (`fact_key`, timestamps, idempotency key) in typed form in Phase 2. The installed manager forwards only the base search limit, so strict historical filtering needs a separate direct `search_as_of(query, as_of, limit)` API; it cannot be smuggled through manager options. Constructor defaults prioritize a credential-free FTS path; hybrid is an explicit choice with a pinned local or user supplied embedder.

The package will implement `search`, `add`, `add_messages`, and `initialize`. `initialize` creates/opens a writable table and indexes; a read-only instance only opens existing state. It will not implement `get_tools` initially because manager-owned `search_memory`/`add_memory` cover the public operations. Additional history/maintenance methods are proposed only after their typing and tests are designed. `add` is needed for direct and tool writes and for fact extraction; `add_messages` is needed for raw ingestion and the replay test. Implementing both changes the manager's default extraction route as stated above.

## Data model and write policy

One schema versioned table will hold `entry_id`, `tenant_id`, `run_id`, `sequence_number`, `role`, `content`, `vector`, `fact_key`, `valid_from`, `valid_to`, `recorded_at`, `active`, `source`, and metadata. Tenant is part of every derived ID. Records with no vector still need an explicit schema/query policy; vector dimensions are fixed for a table and a mismatched embedder must fail on open. Store a schema and embedder identity/version marker so an upgraded model cannot silently mix vectors. Timestamps are UTC and recorded separately from asserted validity. A keyed update is a versioned write; older backfills must not replace the current active version. A raw replay with the same ID and different content is a conflict error. The exact Arrow schema and transaction sequence need Phase 2 proof: `merge_insert` alone does not establish atomic multi-row history updates.

## Error policy

| Condition | Proposed behavior |
| --- | --- |
| Missing local table | Writable `initialize` creates an empty schema/table; read-only raises `StoreNotInitializedError`. An empty writable table returns `[]` for search. |
| FTS index absent | Writable FTS/hybrid `initialize` or first query builds it under the write lock and then searches; read-only raises `IndexUnavailableError`. Missing index never causes silent empty results or an unannounced switch to vector. |
| Embedder missing, unavailable, or dimension changed | Constructor rejects a missing embedder for vector/hybrid; runtime model failure raises `EmbeddingError` with original cause; dimension mismatch raises `SchemaMismatchError`. No invisible keyword fallback. |
| S3 URI not writable / credentials absent | Initialization raises `StorageError` with the underlying cause and URI scheme, without printing credentials. No local fallback. |
| Corrupt schema, invalid query limit, invalid tenant | Typed exception or `ValueError` before a query/write; no partial cross-tenant operation. |
| Failed extraction write | Raise from the store so Strands can record/retry it. Explicit direct writes surface errors to caller; never silently swallow. |

Search before the FTS index exists is supported for a **writable** store via on-demand index creation, with a dedicated test. Exact exception names and transactional details are provisional until the installed LanceDB behavior is exercised in Phase 2. A direct `store.search` raises backend failures. The installed `MemoryManager.search` instead catches each store's search failure, logs it, and may return `[]`; this package cannot change that upstream behavior, so production users need logging/metrics on failed manager searches. Injection also fails open on formatter errors. Tests will assert both direct and manager behavior.

## Compatibility, tests, and gates

Minimum Python is 3.11; CI will cover 3.11, 3.12 and 3.13. The initial compatibility claim is **Strands Agents exactly 1.65.0**, not an unverified `>=1.65.0` range; LanceDB is pinned to 0.41.0 for development. Relax version constraints only after a compatibility matrix. Packaging will later use an Apache-2.0 license and a typed wheel; nothing is published during this phase.

Phase 2 starts with offline Protocol, Agent, replay, tenant property, update, persistence, index, failure and concurrency tests. Phase 3 builds a fixed-seed stratified LongMemEval-S dev split and compares the modes and baselines in one harness. Before any test labels are opened, the chosen config and prompt/model IDs will be frozen and reviewed. The frozen test split runs once in Phase 4, then all 500 are separately reported. No number from the prior LangGraph project is reused. The $20 model spend gate, heavy-dependency gate, PyPI/TestPyPI gate and external-PR gate remain in force.

## Approval questions

1. Is requiring a caller supplied stable `agent_run_id` for raw-message mode acceptable? The installed `AddMessagesContext` has only sequence numbers, so this is necessary for durable replay identity without merging independent identical messages.
2. Is a default local FTS quickstart plus an optional local embedding adapter acceptable until its model size and offline behavior are verified? Hybrid mode would require explicit embedder configuration.
3. Should S3 remain a documented candidate gated on emulator/real-backend evidence, with local storage as the first release target if S3 does not pass?

Approval of this document authorizes Phase 2 implementation within these bounds; any later public API change will be brought back for approval as requested.
