"""Embedded LanceDB implementation of the Strands memory-store contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import threading
from collections.abc import Sequence
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import lancedb
import pyarrow as pa  # type: ignore[import-untyped]
from filelock import FileLock
from lancedb.expr import col
from lancedb.index import FTS
from lancedb.query import ColumnOrdering
from strands.memory import AddMessagesContext, MemoryEntry, MemoryStore
from strands.memory.extraction.model_extractor import ModelExtractor
from strands.memory.extraction.types import ExtractionConfig
from strands.memory.types import Metadata, SearchOptions
from strands.types.content import Message

TABLE_NAME = "memories"
SCHEMA_VERSION = "1"
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


class MemoryStorageError(RuntimeError):
    """A storage operation failed without an automatic backend substitution."""


class StoreNotInitializedError(MemoryStorageError):
    """The read-only store's table does not exist."""


class IndexUnavailableError(MemoryStorageError):
    """The required FTS index is absent in a read-only store."""


class SchemaMismatchError(MemoryStorageError):
    """An existing table was created for a different schema or embedder."""


class EmbeddingError(MemoryStorageError):
    """The configured embedder failed or returned an invalid vector."""


class ReplayConflictError(MemoryStorageError):
    """A replay ID was reused for different message content."""


class Embedder(Protocol):
    """Synchronous embedding interface; the store runs it in a worker thread."""

    dimension: int
    model_id: str

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    def embed_query(self, text: str) -> Sequence[float]: ...


def _path_lock(uri: Path) -> threading.RLock:
    key = str(uri.resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


def _utc_time(value: object | None, *, field: str) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field} must have a timezone")
    return value.astimezone(UTC)


def _micros(value: datetime) -> int:
    delta = value - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _from_micros(value: int) -> datetime:
    return _EPOCH + timedelta(microseconds=value)


def _identity(*parts: object) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _validated_vector(values: Sequence[float], dimension: int) -> list[float]:
    if len(values) != dimension:
        raise EmbeddingError(
            f"expected {dimension} embedding dimensions, got {len(values)}"
        )
    vector = [float(value) for value in values]
    if not all(math.isfinite(value) for value in vector):
        raise EmbeddingError("embedding contains a non-finite value")
    return vector


class LanceDBMemoryStore(MemoryStore):
    """A tenant-scoped Strands memory store backed by a local LanceDB folder.

    Raw extraction requires a stable, caller-supplied ``agent_run_id``. Strands
    supplies sequence numbers but does not supply that ID. Reuse the ID only
    when replaying the same Agent run; use a new ID for a new run.
    """

    def __init__(
        self,
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
    ) -> None:
        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id must be nonempty")
        if not name or not name.strip():
            raise ValueError("name must be nonempty")
        if max_search_results < 1:
            raise ValueError("max_search_results must be at least 1")
        if retrieval not in ("fts", "vector", "hybrid_rrf"):
            raise ValueError(f"unknown retrieval mode: {retrieval}")
        if ingestion not in ("off", "facts", "raw"):
            raise ValueError(f"unknown ingestion mode: {ingestion}")
        if update_strategy not in ("append", "fact_key"):
            raise ValueError(f"unknown update strategy: {update_strategy}")
        if not 0 <= vector_weight <= 1 or not math.isfinite(vector_weight):
            raise ValueError("vector_weight must be finite and between 0 and 1")
        if recency_weight < 0 or not math.isfinite(recency_weight):
            raise ValueError("recency_weight must be finite and nonnegative")
        if retrieval != "fts" and embedder is None:
            raise ValueError("vector and hybrid retrieval require an embedder")
        if embedder is not None and (embedder.dimension < 1 or not embedder.model_id):
            raise ValueError("embedder needs a positive dimension and model_id")
        if ingestion == "raw" and not agent_run_id:
            raise ValueError("raw ingestion requires a stable agent_run_id")
        if ingestion != "off" and not writable:
            raise ValueError("automatic ingestion requires a writable store")
        raw_uri = str(uri)
        if not raw_uri:
            raise ValueError("uri must be nonempty")
        if "://" in raw_uri:
            raise MemoryStorageError(
                "remote LanceDB URIs are unverified in this release; use a local path"
            )

        self.uri = Path(raw_uri).expanduser().resolve()
        self.tenant_id = tenant_id
        self.name = name
        self.description = description
        self.writable = writable
        self.max_search_results = max_search_results
        self.retrieval = retrieval
        self.embedder = embedder
        self.vector_weight = vector_weight
        self.ingestion = ingestion
        self.agent_run_id = agent_run_id
        self.update_strategy = update_strategy
        self.recency_weight = recency_weight
        self.extraction: bool | ExtractionConfig
        if ingestion == "facts":
            self.extraction = {"extractor": ModelExtractor()}
        else:
            self.extraction = ingestion == "raw"
        self._lock = _path_lock(self.uri)
        self._file_lock_path = self.uri.parent / f".{self.uri.name}.lock"

    def _process_guard(self) -> FileLock | nullcontext[None]:
        if not self.writable:
            return nullcontext()
        self.uri.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(str(self._file_lock_path), timeout=60)

    def _schema(self) -> pa.Schema:
        fields = [
            pa.field("entry_id", pa.string(), nullable=False),
            pa.field("tenant_id", pa.string(), nullable=False),
            pa.field("content", pa.string(), nullable=False),
            pa.field("role", pa.string()),
            pa.field("run_id", pa.string()),
            pa.field("sequence_number", pa.int64()),
            pa.field("fact_key", pa.string()),
            pa.field("valid_from", pa.int64(), nullable=False),
            pa.field("recorded_at", pa.int64(), nullable=False),
            pa.field("source", pa.string(), nullable=False),
            pa.field("metadata_json", pa.string(), nullable=False),
        ]
        metadata = {b"strands_lancedb_schema": SCHEMA_VERSION.encode()}
        if self.embedder is not None:
            fields.append(
                pa.field("vector", pa.list_(pa.float32(), self.embedder.dimension))
            )
            metadata[b"embedding_model_id"] = self.embedder.model_id.encode()
        return pa.schema(fields, metadata=metadata)

    def _open_table(self) -> Any:
        db = lancedb.connect(str(self.uri))
        if TABLE_NAME not in db.list_tables().tables:
            if not self.writable:
                raise StoreNotInitializedError(
                    f"table {TABLE_NAME!r} is absent at {self.uri}"
                )
            table = db.create_table(TABLE_NAME, schema=self._schema())
        else:
            table = db.open_table(TABLE_NAME)
        actual = table.schema
        expected = self._schema()
        if not actual.equals(expected, check_metadata=True):
            raise SchemaMismatchError(
                "existing table schema or embedding model differs; use a new directory or migrate explicitly"
            )
        return table

    def _ensure_index(self, table: Any) -> None:
        if self.retrieval == "vector":
            return
        has_fts = any(
            index.index_type == "FTS" and "content" in index.columns
            for index in table.list_indices()
        )
        if has_fts:
            return
        if not self.writable:
            raise IndexUnavailableError("the content FTS index is absent")
        table.create_index("content", config=FTS())

    def _initialize_sync(self) -> None:
        with self._lock, self._process_guard():
            try:
                table = self._open_table()
                self._ensure_index(table)
            except MemoryStorageError:
                raise
            except Exception as exc:
                raise MemoryStorageError(
                    f"cannot initialize LanceDB at {self.uri}"
                ) from exc

    async def initialize(self) -> None:
        """Create or validate the table and required FTS index."""
        await asyncio.to_thread(self._initialize_sync)

    def _embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        assert self.embedder is not None
        try:
            vectors = self.embedder.embed_documents(texts)
            if len(vectors) != len(texts):
                raise EmbeddingError("embedder returned the wrong number of vectors")
            return [
                _validated_vector(vector, self.embedder.dimension) for vector in vectors
            ]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("document embedding failed") from exc

    def _embed_query(self, query: str) -> list[float]:
        assert self.embedder is not None
        try:
            return _validated_vector(
                self.embedder.embed_query(query), self.embedder.dimension
            )
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("query embedding failed") from exc

    def _write_rows(self, rows: list[dict[str, Any]]) -> None:
        with self._lock, self._process_guard():
            try:
                table = self._open_table()
                self._ensure_index(table)
                for row in rows:
                    existing = (
                        table.search()
                        .where(col("entry_id") == row["entry_id"], prefilter=True)
                        .limit(2)
                        .to_list()
                    )
                    if existing and any(
                        old["tenant_id"] != row["tenant_id"]
                        or old["content"] != row["content"]
                        or old["source"] != row["source"]
                        or old["role"] != row["role"]
                        or old["run_id"] != row["run_id"]
                        or old["sequence_number"] != row["sequence_number"]
                        or old["fact_key"] != row["fact_key"]
                        or old["metadata_json"] != row["metadata_json"]
                        for old in existing
                    ):
                        raise ReplayConflictError(
                            "an entry ID was reused with different content"
                        )
                data = pa.Table.from_pylist(rows, schema=self._schema())
                table.merge_insert("entry_id").when_not_matched_insert_all().execute(
                    data
                )
            except MemoryStorageError:
                raise
            except Exception as exc:
                raise MemoryStorageError("LanceDB write failed") from exc

    async def add(self, content: str, metadata: Metadata | None = None) -> str:
        """Add an idempotent fact; identical content in one tenant is one fact."""
        if not self.writable:
            raise ValueError("store is read-only")
        if not content or not content.strip():
            raise ValueError("content must be nonempty")
        meta = dict(metadata or {})
        fact_key = meta.get("fact_key")
        if fact_key is not None and (not isinstance(fact_key, str) or not fact_key):
            raise ValueError("fact_key must be a nonempty string")
        if self.update_strategy == "fact_key" and fact_key is None:
            raise ValueError("fact_key update strategy requires metadata.fact_key")
        explicit_key = meta.get("idempotency_key")
        if explicit_key is not None and (
            not isinstance(explicit_key, str) or not explicit_key
        ):
            raise ValueError("idempotency_key must be a nonempty string")
        valid_from = _utc_time(meta.get("valid_from"), field="valid_from")
        recorded_at = datetime.now(UTC)
        serializable_meta = dict(meta)
        if isinstance(serializable_meta.get("valid_from"), datetime):
            serializable_meta["valid_from"] = valid_from.isoformat()
        try:
            metadata_json = json.dumps(
                serializable_meta, sort_keys=True, ensure_ascii=False
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON serializable") from exc
        entry_id = (
            _identity("fact", self.tenant_id, "key", explicit_key)
            if explicit_key is not None
            else _identity("fact", self.tenant_id, "content", fact_key, content.strip())
        )
        row: dict[str, Any] = {
            "entry_id": entry_id,
            "tenant_id": self.tenant_id,
            "content": content,
            "role": None,
            "run_id": None,
            "sequence_number": None,
            "fact_key": fact_key,
            "valid_from": _micros(valid_from),
            "recorded_at": _micros(recorded_at),
            "source": "fact",
            "metadata_json": metadata_json,
        }
        if self.embedder is not None:
            row["vector"] = (await asyncio.to_thread(self._embed_documents, [content]))[
                0
            ]
        await asyncio.to_thread(self._write_rows, [row])
        return entry_id

    async def add_messages(
        self, messages: list[Message], context: AddMessagesContext | None = None
    ) -> list[str]:
        """Add raw text turns with durable IDs from run ID and sequence number."""
        if not self.writable:
            raise ValueError("store is read-only")
        if self.agent_run_id is None:
            raise ValueError("add_messages requires agent_run_id")
        numbers = context.sequence_numbers if context is not None else None
        if (
            numbers is None
            or len(numbers) != len(messages)
            or len(set(numbers)) != len(numbers)
        ):
            raise ValueError("unique aligned sequence_numbers are required")
        now = datetime.now(UTC)
        rows: list[dict[str, Any]] = []
        for message, number in zip(messages, numbers, strict=True):
            if not isinstance(number, int) or number < 0:
                raise ValueError("sequence_numbers must be nonnegative integers")
            role = message.get("role")
            if role not in ("user", "assistant"):
                continue
            text = "\n".join(
                str(block["text"])
                for block in message.get("content", [])
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            ).strip()
            if not text:
                continue
            row: dict[str, Any] = {
                "entry_id": _identity("raw", self.tenant_id, self.agent_run_id, number),
                "tenant_id": self.tenant_id,
                "content": text,
                "role": role,
                "run_id": self.agent_run_id,
                "sequence_number": number,
                "fact_key": None,
                "valid_from": _micros(now),
                "recorded_at": _micros(now),
                "source": "raw",
                "metadata_json": "{}",
            }
            rows.append(row)
        if not rows:
            return []
        if self.embedder is not None:
            vectors = await asyncio.to_thread(
                self._embed_documents, [row["content"] for row in rows]
            )
            for row, vector in zip(rows, vectors, strict=True):
                row["vector"] = vector
        await asyncio.to_thread(self._write_rows, rows)
        return [row["entry_id"] for row in rows]

    def _query_rows(
        self, query: str, limit: int, as_of: datetime | None
    ) -> list[dict[str, Any]]:
        with self._lock, self._process_guard():
            try:
                table = self._open_table()
                self._ensure_index(table)
                predicate = col("tenant_id") == self.tenant_id
                if as_of is not None:
                    predicate = predicate & (col("valid_from") <= _micros(as_of))
                pool = max(limit * 4, 20)
                if self.retrieval == "fts":
                    rows = cast(
                        list[dict[str, Any]],
                        (
                            table.search(query, query_type="fts", fts_columns="content")
                            .where(predicate, prefilter=True)
                            .limit(pool)
                            .to_list()
                        ),
                    )
                elif self.retrieval == "vector":
                    vector = self._embed_query(query)
                    rows = cast(
                        list[dict[str, Any]],
                        (
                            table.search(vector, query_type="vector")
                            .where(predicate, prefilter=True)
                            .limit(pool)
                            .to_list()
                        ),
                    )
                else:
                    vector = self._embed_query(query)
                    fts = (
                        table.search(query, query_type="fts", fts_columns="content")
                        .where(predicate, prefilter=True)
                        .limit(pool)
                        .to_list()
                    )
                    semantic = (
                        table.search(vector, query_type="vector")
                        .where(predicate, prefilter=True)
                        .limit(pool)
                        .to_list()
                    )
                    scores: dict[str, float] = {}
                    fused_rows: dict[str, dict[str, Any]] = {}
                    for ranked, weight in (
                        (fts, 1 - self.vector_weight),
                        (semantic, self.vector_weight),
                    ):
                        for rank, row in enumerate(ranked, start=1):
                            key = row["entry_id"]
                            scores[key] = scores.get(key, 0.0) + weight / (60 + rank)
                            fused_rows[key] = row
                    rows = [
                        dict(fused_rows[key], _fusion_score=score)
                        for key, score in sorted(
                            scores.items(), key=lambda pair: (-pair[1], pair[0])
                        )
                    ]
                if self.update_strategy == "fact_key":
                    known_ids = {row["entry_id"] for row in rows}
                    for fact_key in {
                        row["fact_key"] for row in rows if row["fact_key"]
                    }:
                        newest = (
                            table.search()
                            .where(predicate & (col("fact_key") == fact_key))
                            .order_by(
                                [
                                    ColumnOrdering(
                                        column_name="valid_from", ascending=False
                                    ),
                                    ColumnOrdering(
                                        column_name="recorded_at", ascending=False
                                    ),
                                ]
                            )
                            .limit(1)
                            .to_list()
                        )
                        if newest and newest[0]["entry_id"] not in known_ids:
                            rows.append(newest[0])
                            known_ids.add(newest[0]["entry_id"])
                return rows
            except MemoryStorageError:
                raise
            except Exception as exc:
                raise MemoryStorageError("LanceDB search failed") from exc

    def _rank_rows(
        self, rows: list[dict[str, Any]], limit: int, as_of: datetime | None
    ) -> list[MemoryEntry]:
        latest: dict[str, int] = {}
        for row in rows:
            key = row["fact_key"]
            if key is not None:
                latest[key] = max(latest.get(key, row["valid_from"]), row["valid_from"])
        scored: list[tuple[float, dict[str, Any]]] = []
        for index, row in enumerate(rows):
            score = float(row.get("_fusion_score", 1 / (60 + index + 1)))
            if (
                row["fact_key"] is not None
                and row["valid_from"] == latest[row["fact_key"]]
            ):
                score += 1.0
            age_days = max(
                0.0,
                (_micros(as_of or datetime.now(UTC)) - row["valid_from"])
                / 86_400_000_000,
            )
            score += self.recency_weight / (1.0 + age_days)
            scored.append((score, row))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["entry_id"]))
        result: list[MemoryEntry] = []
        for score, row in scored[:limit]:
            metadata = json.loads(row["metadata_json"])
            metadata.update(
                {
                    "entry_id": row["entry_id"],
                    "role": row["role"],
                    "source": row["source"],
                    "fact_key": row["fact_key"],
                    "valid_from": _from_micros(row["valid_from"]).isoformat(),
                    "recorded_at": _from_micros(row["recorded_at"]).isoformat(),
                    "_relevanceScore": score,
                }
            )
            result.append(
                MemoryEntry(
                    content=row["content"], store_name=self.name, metadata=metadata
                )
            )
        return result

    async def search(
        self, query: str, options: SearchOptions | None = None
    ) -> list[MemoryEntry]:
        """Return tenant-scoped entries ordered by relevance."""
        limit = (
            options.get("max_search_results", self.max_search_results)
            if options
            else self.max_search_results
        )
        if limit < 1:
            raise ValueError("max_search_results must be at least 1")
        if not query or not query.strip():
            return []
        rows = await asyncio.to_thread(self._query_rows, query, limit, None)
        return self._rank_rows(rows, limit, None)

    async def search_as_of(
        self, query: str, as_of: datetime, limit: int = 5
    ) -> list[MemoryEntry]:
        """Search entries whose validity starts no later than an explicit UTC time."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        point = _utc_time(as_of, field="as_of")
        if not query or not query.strip():
            return []
        rows = await asyncio.to_thread(self._query_rows, query, limit, point)
        return self._rank_rows(rows, limit, point)

    async def history(self, fact_key: str) -> list[MemoryEntry]:
        """Return all versions of one fact key in this tenant, newest first."""
        if not fact_key:
            raise ValueError("fact_key must be nonempty")

        def read() -> list[dict[str, Any]]:
            with self._lock, self._process_guard():
                table = self._open_table()
                return cast(
                    list[dict[str, Any]],
                    (
                        table.search()
                        .where(
                            (col("tenant_id") == self.tenant_id)
                            & (col("fact_key") == fact_key)
                        )
                        .to_list()
                    ),
                )

        rows = await asyncio.to_thread(read)
        rows.sort(key=lambda row: (row["valid_from"], row["entry_id"]), reverse=True)
        return self._rank_rows(rows, len(rows), None)

    async def optimize(self) -> None:
        """Explicitly compact the table; never prune snapshots aggressively."""
        if not self.writable:
            raise ValueError("store is read-only")

        def run() -> None:
            with self._lock, self._process_guard():
                self._open_table().optimize()

        await asyncio.to_thread(run)
