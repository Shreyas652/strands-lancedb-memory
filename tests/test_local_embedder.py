"""Offline checks for the optional model's reproducibility guard."""

from pathlib import Path

import pytest

from evals.local_embedder import model_weights_sha256


def test_weight_fingerprint_detects_changes(tmp_path: Path):
    model = tmp_path / "snapshot" / "model.onnx"
    model.parent.mkdir()
    model.write_bytes(b"first version")
    before = model_weights_sha256(tmp_path)
    model.write_bytes(b"second version")
    assert model_weights_sha256(tmp_path) != before


def test_missing_cached_weights_fail_explicitly(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="ONNX weights"):
        model_weights_sha256(tmp_path)
