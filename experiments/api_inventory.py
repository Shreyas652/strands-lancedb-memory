"""Record installed-package signatures and exact source lines for Phase 0."""

import inspect
import json
import re
from pathlib import Path

import lancedb
from lancedb.db import DBConnection
from lancedb.expr import col
from lancedb.index import FTS, BTree
from lancedb.merge import LanceMergeInsertBuilder
from lancedb.query import LanceHybridQueryBuilder, LanceQueryBuilder
from lancedb.rerankers import RRFReranker
from lancedb.table import Table
from strands import Agent
from strands.memory import AddMessagesContext, MemoryEntry, MemoryManager, MemoryStore
from strands.memory.extraction.model_extractor import ModelExtractor
from strands.memory.extraction.triggers import IntervalTrigger, InvocationTrigger
from strands.memory.extraction.types import ExtractionConfig
from strands.models import Model
from strands.vended_memory_stores.bedrock_knowledge_base import (
    BedrockKnowledgeBaseStore,
)
from strands.vended_memory_stores.file_memory_store import FileMemoryStore
from strands.vended_memory_stores.test_memory_store import TestMemoryStore

APIS = {
    "strands.Agent": Agent,
    "strands.models.Model": Model,
    "strands.memory.MemoryStore": MemoryStore,
    "strands.memory.MemoryStore.search": MemoryStore.search,
    "strands.memory.MemoryStore.add": MemoryStore.add,
    "strands.memory.MemoryStore.add_messages": MemoryStore.add_messages,
    "strands.memory.MemoryStore.initialize": MemoryStore.initialize,
    "strands.memory.MemoryStore.get_tools": MemoryStore.get_tools,
    "strands.memory.MemoryEntry": MemoryEntry,
    "strands.memory.AddMessagesContext": AddMessagesContext,
    "strands.memory.MemoryManager": MemoryManager,
    "strands.memory.MemoryManager.search": MemoryManager.search,
    "strands.memory.MemoryManager.add": MemoryManager.add,
    "strands.memory.MemoryManager.flush": MemoryManager.flush,
    "strands.memory.ExtractionConfig": ExtractionConfig,
    "strands.memory.ModelExtractor": ModelExtractor,
    "strands.memory.InvocationTrigger": InvocationTrigger,
    "strands.memory.IntervalTrigger": IntervalTrigger,
    "strands.vended.TestMemoryStore": TestMemoryStore,
    "strands.vended.FileMemoryStore": FileMemoryStore,
    "strands.vended.BedrockKnowledgeBaseStore": BedrockKnowledgeBaseStore,
    "lancedb.connect": lancedb.connect,
    "lancedb.DBConnection.create_table": DBConnection.create_table,
    "lancedb.DBConnection.open_table": DBConnection.open_table,
    "lancedb.DBConnection.list_tables": DBConnection.list_tables,
    "lancedb.Table.add": Table.add,
    "lancedb.Table.create_index": Table.create_index,
    "lancedb.Table.create_fts_index": Table.create_fts_index,
    "lancedb.Table.merge_insert": Table.merge_insert,
    "lancedb.Table.search": Table.search,
    "lancedb.Table.optimize": Table.optimize,
    "lancedb.Table.count_rows": Table.count_rows,
    "lancedb.Table.delete": Table.delete,
    "lancedb.LanceQueryBuilder.where": LanceQueryBuilder.where,
    "lancedb.LanceQueryBuilder.limit": LanceQueryBuilder.limit,
    "lancedb.LanceQueryBuilder.to_list": LanceQueryBuilder.to_list,
    "lancedb.LanceHybridQueryBuilder.vector": LanceHybridQueryBuilder.vector,
    "lancedb.LanceHybridQueryBuilder.text": LanceHybridQueryBuilder.text,
    "lancedb.LanceHybridQueryBuilder.rerank": LanceHybridQueryBuilder.rerank,
    "lancedb.LanceMergeInsertBuilder.when_matched_update_all": LanceMergeInsertBuilder.when_matched_update_all,
    "lancedb.LanceMergeInsertBuilder.when_not_matched_insert_all": LanceMergeInsertBuilder.when_not_matched_insert_all,
    "lancedb.LanceMergeInsertBuilder.execute": LanceMergeInsertBuilder.execute,
    "lancedb.FTS": FTS,
    "lancedb.BTree": BTree,
    "lancedb.RRFReranker": RRFReranker,
    "lancedb.expr.col": col,
}


def main() -> None:
    site_packages = Path(__file__).resolve().parents[1] / ".venv/Lib/site-packages"
    records = []
    for name, obj in APIS.items():
        source_file = inspect.getsourcefile(obj)
        try:
            line = inspect.getsourcelines(obj)[1]
        except (OSError, TypeError):
            line = None
        try:
            signature = str(inspect.signature(obj))
        except (TypeError, ValueError):
            if not hasattr(obj, "__annotations__"):
                raise
            signature = f"TypedDict fields: {obj.__annotations__}"
        signature = re.sub(r"0x[0-9A-Fa-f]+", "0xADDR", signature)
        records.append(
            {
                "name": name,
                "signature": signature,
                "source": f"{Path(source_file).relative_to(site_packages).as_posix()}:{line}",
            }
        )
    output = Path("results/phase0_api_inventory.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    notes = Path("docs/API_NOTES.md")
    existing = notes.read_text(encoding="utf-8")
    marker = "<!-- GENERATED API INVENTORY -->"
    if marker not in existing:
        raise ValueError(f"{notes} must contain {marker}")
    rows = ["| Installed API | Signature | Verified source |", "| --- | --- | --- |"]
    for record in records:
        signature = record["signature"].replace("|", "\\|").replace("`", "\\`")
        rows.append(f"| `{record['name']}` | `{signature}` | `{record['source']}` |")
    notes.write_text(
        existing.split(marker)[0] + marker + "\n\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    print(f"saved {output}: {len(records)} installed API signatures")


if __name__ == "__main__":
    main()
