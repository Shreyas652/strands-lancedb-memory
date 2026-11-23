# strands-lancedb-memory

**Status: research and design.** This repository does not yet ship a `MemoryStore` implementation. The design awaits review before implementation, and no benchmark accuracy or scale claims have been produced here.

The proposed package will connect [Strands Agents](https://github.com/strands-agents/harness-sdk) memory to embedded [LanceDB](https://github.com/lancedb/lancedb), with tenant isolation, durable writes and measured retrieval. The current evidence is limited to pinned-package API inspection and local offline probes.

## Review the evidence

- [Installed API notes](docs/API_NOTES.md): exact Strands 1.65.0 and LanceDB 0.41.0 signatures and source lines.
- [Prior art](docs/PRIOR_ART.md): MongoDB, Pixeltable, Neo4j and Strands file-store design patterns.
- [Agent lifecycle](docs/LIFECYCLE.md): real Agent with a stub model and logged memory calls.

The design proposal is prepared in Phase 1 and will be reviewed separately.

## Reproduce Phase 0

```powershell
uv sync --locked
uv run python experiments/api_inventory.py
uv run python experiments/phase0_lifecycle.py
uv run python experiments/lancedb_api_smoke.py
```

The generated JSON artifacts are in `results/`. These commands make no paid model calls. The LanceDB smoke uses a temporary local directory and does not claim S3 or concurrency support.
