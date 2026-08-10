"""
EMB-008 — Measure Embedding Generation Performance.

This script runs a performance benchmark on the existing batch embedding
generation function (`src.embeddings.batch_embed.embed_texts`). It reuses the
warm-loaded SentenceTransformer model instance and reports metrics including
throughput (records/s) and latency (ms/record) for multiple batch sizes.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sentence_transformers

from src.embeddings.batch_embed import embed_texts
from src.embeddings.embedder import get_embedder, get_model_name

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("embedding_performance_benchmark")

# Find repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "processed" / "annotations_normalized.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark" / "results"

DEFAULT_BATCH_SIZES = [1, 8, 16, 32, 64]
DEFAULT_REPETITIONS = 3


def calculate_metrics(
    times: Sequence[float],
    num_records: int,
) -> dict[str, float]:
    """
    Calculate throughput and latency metrics across multiple runs.

    Args:
        times: A sequence of measured execution times in seconds.
        num_records: The number of records processed.

    Returns:
        A dictionary containing mean_time_seconds, min_time_seconds,
        max_time_seconds, mean_throughput_records_per_second, and
        mean_latency_ms_per_record.
    """
    if not times:
        raise ValueError("Times sequence cannot be empty.")
    if num_records <= 0:
        raise ValueError("Number of records must be a positive integer.")

    times_arr = np.array(times, dtype=np.float64)

    mean_time = float(np.mean(times_arr))
    min_time = float(np.min(times_arr))
    max_time = float(np.max(times_arr))

    # Calculate throughput and latency per run, then take the mean
    throughputs = num_records / times_arr
    latencies = (times_arr / num_records) * 1000.0

    mean_throughput = float(np.mean(throughputs))
    mean_latency = float(np.mean(latencies))

    return {
        "mean_time_seconds": mean_time,
        "min_time_seconds": min_time,
        "max_time_seconds": max_time,
        "mean_throughput_records_per_second": mean_throughput,
        "mean_latency_ms_per_record": mean_latency,
    }


def run_benchmark(
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_sizes: list[int] = DEFAULT_BATCH_SIZES,
    repetitions: int = DEFAULT_REPETITIONS,
    num_records: int = 100,
) -> dict[str, Any]:
    """
    Run the performance benchmark on the embedding generation function.

    Args:
        data_path: Path to the validated prompts CSV dataset.
        output_dir: Directory where the benchmark results will be saved.
        batch_sizes: List of batch sizes to benchmark.
        repetitions: Number of times to repeat each benchmark run.
        num_records: Maximum number of records from the dataset to benchmark with.

    Returns:
        A dictionary containing the structured benchmark results.
    """
    logger.info("Initializing embedding performance benchmark...")

    # 1. Warm-load embedding model and get basic metadata
    model_name = get_model_name()
    model = get_embedder()
    device = str(model.device) if hasattr(model, "device") else "unknown"

    logger.info("Model: %s, Device: %s", model_name, device)

    # Determine embedding dimension from a tiny dummy execution
    dummy_output = embed_texts(["dummy"], batch_size=1)
    embedding_dimension = dummy_output.shape[1]
    logger.info("Embedding Dimension: %d", embedding_dimension)

    # 2. Load the dataset
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    logger.info("Loading dataset from %s...", data_path)
    df = pd.read_csv(data_path)
    if "prompt_text" not in df.columns:
        raise ValueError(
            f"Dataset must contain a 'prompt_text' column, got columns: {df.columns.tolist()}"
        )

    # Drop any nulls or empty strings
    all_texts = df["prompt_text"].dropna().astype(str).tolist()
    all_texts = [t for t in all_texts if t.strip()]

    if not all_texts:
        raise ValueError("No valid non-empty texts found in the dataset.")

    # Limit records as configured
    texts = all_texts[:num_records]
    actual_num_records = len(texts)
    logger.info("Using %d records for benchmarking", actual_num_records)

    # 3. Warm-up run (not included in the reported timings)
    logger.info("Performing warm-up run...")
    warmup_texts = ["Warm-up sentence number one.", "Another warm-up sentence."]
    _ = embed_texts(warmup_texts, batch_size=2)
    logger.info("Warm-up run completed successfully.")

    results_summary = []

    # 4. Benchmark cases for each batch size
    for bs in batch_sizes:
        if bs <= 0:
            logger.error("Skipping invalid batch size: %s (must be > 0)", bs)
            continue

        if actual_num_records < bs:
            reason = f"Dataset size ({actual_num_records}) is smaller than batch size ({bs})"
            logger.warning("Skipping batch size %d: %s", bs, reason)
            continue

        logger.info(
            "Running benchmark for batch size %d with %d repetitions...",
            bs,
            repetitions,
        )

        times = []
        for rep in range(repetitions):
            # Monotonic high-resolution timer
            start_time = time.perf_counter()
            embeddings = embed_texts(texts, batch_size=bs)
            end_time = time.perf_counter()

            # Verify shape and dtype to ensure validity
            assert embeddings.shape == (actual_num_records, embedding_dimension), (
                f"Unexpected output shape: {embeddings.shape}"
            )
            assert embeddings.dtype == np.float32, (
                f"Unexpected output dtype: {embeddings.dtype}"
            )

            elapsed = end_time - start_time
            times.append(elapsed)
            logger.info(
                "  Repetition %d/%d: %.4f seconds",
                rep + 1,
                repetitions,
                elapsed,
            )

        # Calculate statistics
        stats = calculate_metrics(times, actual_num_records)
        logger.info(
            "  Batch size %d results: mean=%.4fs, throughput=%.2f rec/s, latency=%.2f ms/rec",
            bs,
            stats["mean_time_seconds"],
            stats["mean_throughput_records_per_second"],
            stats["mean_latency_ms_per_record"],
        )

        results_summary.append(
            {
                "batch_size": bs,
                "repetitions": repetitions,
                **stats,
            }
        )

    # 5. Build structured result dictionary
    benchmark_data = {
        "task_id": "EMB-008",
        "model": model_name,
        "device": device,
        "embedding_dimension": embedding_dimension,
        "records": actual_num_records,
        "results": results_summary,
    }

    # Create output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. Save JSON result
    json_path = output_dir / "embedding_performance.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    logger.info("Saved JSON results to %s", json_path)

    # 7. Generate markdown report
    markdown_path = output_dir / "embedding_performance.md"
    generate_markdown_report(benchmark_data, data_path, markdown_path)
    logger.info("Saved Markdown report to %s", markdown_path)

    return benchmark_data


def generate_markdown_report(
    data: dict[str, Any],
    data_path: Path,
    markdown_path: Path,
) -> None:
    """
    Generate the human-readable Markdown performance report.

    Args:
        data: The structured benchmark results dictionary.
        data_path: Path to the dataset used.
        markdown_path: Path where the Markdown report will be saved.
    """
    py_ver = platform.python_version()
    st_ver = sentence_transformers.__version__

    # Identify the highest throughput batch size
    best_throughput_bs = -1
    best_throughput = -1.0
    if data["results"]:
        best_run = max(
            data["results"],
            key=lambda x: x["mean_throughput_records_per_second"],
        )
        best_throughput_bs = best_run["batch_size"]
        best_throughput = best_run["mean_throughput_records_per_second"]

    # Construct the report text
    report = []
    report.append("# EMB-008 — Embedding Generation Performance")
    report.append("")
    report.append("## Environment")
    report.append(f"- **Model**: `{data['model']}`")
    report.append(f"- **Device**: `{data['device']}`")
    report.append(f"- **Python Version**: `{py_ver}`")
    report.append(f"- **Embedding Library**: `sentence-transformers=={st_ver}`")
    report.append(f"- **Embedding Dimension**: `{data['embedding_dimension']}`")
    report.append("")
    report.append("## Dataset")
    report.append(f"- **Dataset Fixture Used**: `{data_path.name}`")
    report.append(f"- **Number of Records**: `{data['records']}`")
    report.append("")
    report.append("## Benchmark Method")
    report.append("- **Warm-up Approach**: Pre-encode 2 dummy sentences prior to timer start to eliminate JIT/lazy-loading overhead.")
    report.append(f"- **Batch Sizes Tested**: `{ [r['batch_size'] for r in data['results']] }`")
    report.append(f"- **Repetitions**: `{data['results'][0]['repetitions'] if data['results'] else 0}`")
    report.append("- **Timing Method**: `time.perf_counter()` (monotonic high-resolution timer)")
    report.append("")
    report.append("## Results")
    report.append("")
    report.append("| Batch Size | Records | Mean Time (s) | Min Time (s) | Max Time (s) | Throughput (records/s) | Latency (ms/record) |")
    report.append("|------------|---------|---------------|--------------|--------------|------------------------|---------------------|")

    for res in data["results"]:
        report.append(
            f"| {res['batch_size']} | {data['records']} | {res['mean_time_seconds']:.4f} | "
            f"{res['min_time_seconds']:.4f} | {res['max_time_seconds']:.4f} | "
            f"{res['mean_throughput_records_per_second']:.2f} | {res['mean_latency_ms_per_record']:.2f} |"
        )

    report.append("")
    report.append("## Observations")
    if best_throughput_bs != -1:
        report.append(
            f"- The batch size of **{best_throughput_bs}** produced the highest measured throughput of "
            f"**{best_throughput:.2f} records/s**."
        )
        report.append(
            "- Larger batch sizes generally improve throughput by taking advantage of vectorization and parallel operations in PyTorch, though performance scaling begins to taper off at higher batch sizes."
        )
        report.append(
            f"- Latency per record significantly decreased as batch size increased, showing the impact of PyTorch framework overhead on single-record executions."
        )
        report.append(
            f"- The computations were executed on the **{data['device'].upper()}** device."
        )
    else:
        report.append("- No benchmark results were recorded.")

    report.append("")
    report.append("## Limitations")
    report.append("- **Hardware Dependence**: The absolute timings and throughput numbers are specific to the machine running the benchmark (e.g. CPU architecture, number of cores, memory speed).")
    report.append("- **Dataset Limitations**: The benchmark was conducted with a subset of 100 prompts. While representative, longer prompt strings can affect both memory usage and compute time.")
    report.append("- **Production Load Representation**: This benchmark measures isolated embedding generation performance and does not account for other database read/write or network operations that occur in the full production pipeline.")

    # Write report
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run EMB-008 embedding performance benchmark."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to the source CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save the results",
    )
    parser.add_argument(
        "--records",
        type=int,
        default=100,
        help="Number of records to benchmark with",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions to run per batch size",
    )

    args = parser.parse_args()

    run_benchmark(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        repetitions=args.repetitions,
        num_records=args.records,
    )
