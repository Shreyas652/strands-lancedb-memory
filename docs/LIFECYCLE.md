# Strands memory lifecycle observed with a real Agent

Phase 0 experiment: `uv run python experiments/phase0_lifecycle.py`. It instantiates `strands.Agent` with a local stub `Model` and a `MemoryManager` containing two logged `TestMemoryStore` variants. The first implements `add` only and uses a fixed extractor; the second also implements `add_messages` and uses the default raw-message route. Both use `InvocationTrigger` so each of two Agent calls fires extraction. No network model or API key is involved. The complete call log and model inputs are saved in `results/phase0_lifecycle.json`.

| Event | Observed call |
| --- | --- |
| First Agent user turn | Both stores receive `search("What color do I like?", {"max_search_results": 5})` before the model. The first model input includes the seeded `<memory>` block. |
| First extraction | Raw store receives user and assistant messages with sequence numbers `[0, 1]`. The add-only store's extractor sees the same roles and `context.default_model`, then the store receives `add("extracted durable fact", None)`. |
| Second Agent user turn | Both stores receive `search("Tell me again", {"max_search_results": 5})`. |
| Second extraction | Raw store receives user and assistant messages with sequence numbers `[2, 3]`; add-only path calls extractor and `add` again. |
| Explicit `search_memory` tool | Both stores receive `search("blue", {"max_search_results": 3})`. |
| Explicit `add_memory` tool | Both stores receive `add("Explicit tool fact", None)`. The tool's input shape is `entries=[...]`. |

The script asserts that the first stub model call contains `<memory>`. In the installed manager, injection is a per-call message transformation and extraction receives the durable role/message stream, not the injected block. `Agent` is constructed with `memory_manager=manager` so the synchronous call flushes pending extraction. The log records 14 store/extractor calls and 2 stub model calls. This test demonstrates routing and the Agent lifecycle, not durable LanceDB correctness, concurrency, or real-model answer accuracy.

Implications for the store design:

1. Supporting both `add` and `add_messages` gives explicit tool writes plus raw default extraction. The latter will **not** invoke the client's `ModelExtractor` unless configured explicitly.
2. `AddMessagesContext.sequence_numbers` distinguishes entries within a live Agent, but a new Agent can reuse the numbers. It provides no durable run identity. Stable replay IDs need another source; silently deriving them from message text would merge distinct identical turns.
3. Search latency occurs before every injected model call. Retrieval must honor the caller limit and keep tenant filtering on every query path.
4. The manager may retry extraction writes. A batch must either be idempotent under an explicit identity contract or clearly document the scope of replay protection.
