# Prior art read in Phase 0

I compared the pinned Strands built-in stores in `.venv/Lib/site-packages/strands/vended_memory_stores/` and three published community integrations. These are design inputs, not claims that their behavior is defective.

| Store | Useful convention | Difference for this package |
| --- | --- | --- |
| [MongoDB Strands store](https://github.com/mongodb/strands-mongodb/blob/main/python/src/strands_mongodb/memory_store.py) | Explicit scope/tenant filtering on every read and write; read-only default; content-hash idempotency for extracted facts; well-defined search readiness policy. | Use typed tenant predicates and no implicit unscoped access. Do not content-hash raw turns: two independent short messages can have identical text. Embedded local path is the required first target. |
| [Pixeltable Strands store](https://github.com/pixeltable/strands-pixeltable/blob/main/strands_pixeltable/store.py) | Implements both `add` and `add_messages`; runs blocking work in a thread and serializes local operations. | Keep both routes only with the deliberate extraction semantics documented. Its `tracking_id` or UUID approach does not alone prove durable deduplication of a replayed Strands batch, because the Strands filter may drop `tracking_id`. |
| [Neo4j Agent Memory Strands integration](https://github.com/neo4j-labs/agent-memory/blob/main/src/neo4j_agent_memory/integrations/strands/memory_store.py) | User/scope settings and sequence-aware ingestion. | Its `(run_id, seq)` seen set is process-local. Persist replay identity if available and test restart behavior; avoid promising an exactly-once guarantee that Strands' callback does not supply enough information to establish. |
| Pinned Strands [FileMemoryStore](https://github.com/strands-agents/harness-sdk/tree/main/strands-py/src/strands/vended_memory_stores/file_memory_store) | Simple local setup and keyword baseline; practical file persistence. | Keep it as an equal harness baseline. LanceDB's vector/FTS/hybrid recall is a hypothesis to measure on the dev and frozen splits, not an assumed win. |

The pinned `TestMemoryStore` is also a useful offline contract baseline. It deduplicates identical content and serializes a local read-modify-write cycle, but it is not a substitute for multi-process or S3 tests. `BedrockKnowledgeBaseStore` shows the manager can work with a remote backend, but it is not an embedded-storage baseline.

The exact package interface and line references are in `docs/API_NOTES.md`. The source repositories above may advance after this note; Phase 2 should pin a commit if code is copied or a comparison depends on a precise upstream implementation. No prior-art code has been copied into this package.
