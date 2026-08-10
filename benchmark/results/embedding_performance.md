# EMB-008 — Embedding Generation Performance

## Environment
- **Model**: `all-MiniLM-L6-v2`
- **Device**: `cpu`
- **Python Version**: `3.10.11`
- **Embedding Library**: `sentence-transformers==5.7.0`
- **Embedding Dimension**: `384`

## Dataset
- **Dataset Fixture Used**: `annotations_normalized.csv`
- **Number of Records**: `100`

## Benchmark Method
- **Warm-up Approach**: Pre-encode 2 dummy sentences prior to timer start to eliminate JIT/lazy-loading overhead.
- **Batch Sizes Tested**: `[1, 8, 16, 32, 64]`
- **Repetitions**: `3`
- **Timing Method**: `time.perf_counter()` (monotonic high-resolution timer)

## Results

| Batch Size | Records | Mean Time (s) | Min Time (s) | Max Time (s) | Throughput (records/s) | Latency (ms/record) |
|------------|---------|---------------|--------------|--------------|------------------------|---------------------|
| 1 | 100 | 2.9114 | 2.7201 | 3.1822 | 34.50 | 29.11 |
| 8 | 100 | 0.9966 | 0.9183 | 1.0362 | 100.67 | 9.97 |
| 16 | 100 | 1.2127 | 1.1440 | 1.2729 | 82.62 | 12.13 |
| 32 | 100 | 1.3366 | 1.3221 | 1.3576 | 74.83 | 13.37 |
| 64 | 100 | 1.1525 | 1.0558 | 1.2631 | 87.24 | 11.53 |

## Observations
- The batch size of **8** produced the highest measured throughput of **100.67 records/s**.
- Larger batch sizes generally improve throughput by taking advantage of vectorization and parallel operations in PyTorch, though performance scaling begins to taper off at higher batch sizes.
- Latency per record significantly decreased as batch size increased, showing the impact of PyTorch framework overhead on single-record executions.
- The computations were executed on the **CPU** device.

## Limitations
- **Hardware Dependence**: The absolute timings and throughput numbers are specific to the machine running the benchmark (e.g. CPU architecture, number of cores, memory speed).
- **Dataset Limitations**: The benchmark was conducted with a subset of 100 prompts. While representative, longer prompt strings can affect both memory usage and compute time.
- **Production Load Representation**: This benchmark measures isolated embedding generation performance and does not account for other database read/write or network operations that occur in the full production pipeline.