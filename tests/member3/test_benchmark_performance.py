"""
Unit tests for EMB-008 embedding performance benchmark logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from benchmark import embedding_performance


# ---------------------------------------------------------------------------
# Tests for calculate_metrics
# ---------------------------------------------------------------------------

def test_calculate_metrics_correctness() -> None:
    """
    Verify that throughput and latency are calculated correctly.

    With 10 records:
    - Run 1: 2.0 seconds -> throughput = 5.0 records/s, latency = 200 ms/record
    - Run 2: 1.0 seconds -> throughput = 10.0 records/s, latency = 100 ms/record
    - Run 3: 4.0 seconds -> throughput = 2.5 records/s, latency = 400 ms/record

    Expected statistics:
    - mean_time: (2.0 + 1.0 + 4.0) / 3 = 2.3333333333333335
    - min_time: 1.0
    - max_time: 4.0
    - mean_throughput: (5.0 + 10.0 + 2.5) / 3 = 5.833333333333333
    - mean_latency: (200.0 + 100.0 + 400.0) / 3 = 233.33333333333334
    """
    times = [2.0, 1.0, 4.0]
    num_records = 10

    metrics = embedding_performance.calculate_metrics(times, num_records)

    assert pytest.approx(metrics["mean_time_seconds"]) == 7.0 / 3
    assert metrics["min_time_seconds"] == 1.0
    assert metrics["max_time_seconds"] == 4.0
    assert pytest.approx(metrics["mean_throughput_records_per_second"]) == 17.5 / 3
    assert pytest.approx(metrics["mean_latency_ms_per_record"]) == 700.0 / 3


def test_calculate_metrics_empty_times_raises_value_error() -> None:
    """An empty sequence of times should raise ValueError."""
    with pytest.raises(ValueError, match="Times sequence cannot be empty"):
        embedding_performance.calculate_metrics([], 10)


def test_calculate_metrics_invalid_records_raises_value_error() -> None:
    """Non-positive number of records should raise ValueError."""
    with pytest.raises(ValueError, match="Number of records must be a positive integer"):
        embedding_performance.calculate_metrics([1.0, 2.0], 0)

    with pytest.raises(ValueError, match="Number of records must be a positive integer"):
        embedding_performance.calculate_metrics([1.0, 2.0], -5)


# ---------------------------------------------------------------------------
# Mocks & Fixtures for run_benchmark integration testing
# ---------------------------------------------------------------------------

class FakeModel:
    """Minimal fake model representing SentenceTransformer."""
    device = "cpu"


@pytest.fixture
def mock_benchmark_dependencies(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """
    Mock file I/O, model loading, and embedding functions to test run_benchmark
    deterministically without loading the real SentenceTransformer model or
    reading files from disk.
    """
    mock_data = pd.DataFrame({
        "prompt_text": [f"Test prompt {i}" for i in range(120)]
    })

    # Mock pd.read_csv to return our dummy dataframe
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: mock_data)

    # Mock Path.exists to always say the dataset file exists
    monkeypatch.setattr(Path, "exists", lambda self: True)

    # Mock get_embedder and get_model_name
    monkeypatch.setattr(embedding_performance, "get_embedder", lambda: FakeModel())
    monkeypatch.setattr(embedding_performance, "get_model_name", lambda: "fake-minilm")

    # Mock embed_texts to return dummy vectors of correct shape
    calls = []

    def mock_embed_texts(texts: Any, batch_size: int = 32) -> np.ndarray:
        calls.append((list(texts), batch_size))
        return np.ones((len(texts), 384), dtype=np.float32)

    monkeypatch.setattr(embedding_performance, "embed_texts", mock_embed_texts)

    # Prevent saving output files directly to workspace during unit tests
    saved_files = {}

    def mock_json_dump(obj: Any, fp: Any, **kwargs: Any) -> None:
        saved_files["json"] = obj

    monkeypatch.setattr(json, "dump", mock_json_dump)

    def mock_write_report(data: dict[str, Any], data_path: Path, markdown_path: Path) -> None:
        saved_files["markdown"] = data

    monkeypatch.setattr(
        embedding_performance,
        "generate_markdown_report",
        mock_write_report,
    )

    return {
        "calls": calls,
        "saved_files": saved_files,
    }


# ---------------------------------------------------------------------------
# Tests for run_benchmark logic and schema validation
# ---------------------------------------------------------------------------

def test_run_benchmark_output_schema_and_fields(
    mock_benchmark_dependencies: dict[str, Any],
    tmp_path: Path,
) -> None:
    """
    Verify that the run_benchmark function returns a dictionary with all
    required fields and that the saved outputs conform to the schema.
    """
    results = embedding_performance.run_benchmark(
        data_path=Path("dummy_path.csv"),
        output_dir=tmp_path,
        batch_sizes=[1, 8, 16],
        repetitions=2,
        num_records=50,
    )

    # Required top-level fields
    assert results["task_id"] == "EMB-008"
    assert results["model"] == "fake-minilm"
    assert results["device"] == "cpu"
    assert results["embedding_dimension"] == 384
    assert results["records"] == 50
    assert isinstance(results["results"], list)

    # Validate individual results structure
    results_list = results["results"]
    assert len(results_list) == 3  # for [1, 8, 16]

    for idx, bs in enumerate([1, 8, 16]):
        res = results_list[idx]
        assert res["batch_size"] == bs
        assert res["repetitions"] == 2
        assert "mean_time_seconds" in res
        assert "min_time_seconds" in res
        assert "max_time_seconds" in res
        assert "mean_throughput_records_per_second" in res
        assert "mean_latency_ms_per_record" in res

    # Verify mock json and markdown were generated/saved
    assert mock_benchmark_dependencies["saved_files"]["json"] == results
    assert mock_benchmark_dependencies["saved_files"]["markdown"] == results


def test_run_benchmark_skips_larger_batch_sizes(
    mock_benchmark_dependencies: dict[str, Any],
    tmp_path: Path,
) -> None:
    """
    If the requested batch size is larger than the number of records,
    it must be skipped gracefully.
    """
    # 10 records, batch sizes [8, 16]
    results = embedding_performance.run_benchmark(
        data_path=Path("dummy_path.csv"),
        output_dir=tmp_path,
        batch_sizes=[8, 16],
        repetitions=1,
        num_records=10,
    )

    # Batch size 8 should be benchmarked, batch size 16 should be skipped (16 > 10)
    batch_sizes_tested = [r["batch_size"] for r in results["results"]]
    assert batch_sizes_tested == [8]
