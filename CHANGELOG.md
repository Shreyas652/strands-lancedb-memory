# Changelog

## 0.1.0 — 2026-11-25

- Implement a tenant-scoped Strands `MemoryStore` on local LanceDB storage.
- Add idempotent writes, typed metadata filters, configurable FTS/vector/hybrid retrieval, and explicit fact history.
- Add an offline Strands Agent integration suite and reproducible benchmark harnesses.
- On the 400-question held-out LongMemEval-S set, FTS answered 201 correctly versus 186 for FileMemoryStore; the exact McNemar p-value is 0.063, so the difference is not statistically significant at the 5% level. This is not an official leaderboard score.
- Document local-storage, embedding, fact-extraction, concurrency, and benchmark limitations.
