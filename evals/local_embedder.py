"""Optional offline FastEmbed adapter for reproducible retrieval ablations."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EXPECTED_FASTEMBED_VERSION = "0.9.1"
DIMENSION = 384


def model_weights_sha256(cache_dir: Path) -> str:
    weights = sorted(cache_dir.rglob("*.onnx"))
    if not weights:
        raise FileNotFoundError("cached FastEmbed ONNX weights are missing")
    digest = hashlib.sha256()
    for path in weights:
        digest.update(path.relative_to(cache_dir).as_posix().encode())
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class FastEmbedAdapter:
    """Use pinned local BGE weights; initialization never downloads a model."""

    dimension = DIMENSION

    def __init__(self, cache_dir: Path) -> None:
        try:
            installed = version("fastembed")
        except PackageNotFoundError as exc:
            raise RuntimeError(
                "install the optional fastembed==0.9.1 benchmark dependency"
            ) from exc
        if installed != EXPECTED_FASTEMBED_VERSION:
            raise RuntimeError(
                f"expected fastembed {EXPECTED_FASTEMBED_VERSION}, found {installed}"
            )
        from fastembed import TextEmbedding

        self.model_weights_sha256 = model_weights_sha256(cache_dir)
        self.model_id = (
            f"{MODEL_NAME}@fastembed-{installed}:{self.model_weights_sha256}"
        )
        self._model: Any = TextEmbedding(
            model_name=MODEL_NAME,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = [
            vector.astype("float32").tolist()
            for vector in self._model.embed(list(texts))
        ]
        if len(vectors) != len(texts) or any(
            len(vector) != DIMENSION for vector in vectors
        ):
            raise RuntimeError("FastEmbed returned an unexpected document vector shape")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vector = next(self._model.query_embed(text)).astype("float32").tolist()
        if len(vector) != DIMENSION:
            raise RuntimeError("FastEmbed returned an unexpected query vector shape")
        return vector
