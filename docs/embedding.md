# PECS Embedding Pipeline

This document describes the design, implementation, and operational procedures for the PECS embedding pipeline.

## 1. Overview

The embedding layer is a core component of the **PECS Team 2** RAG (Retrieval-Augmented Generation) pipeline. It is responsible for converting preprocessed text prompts into dense, high-dimensional vector representations that capture semantic meaning.

The data flow is structured as follows:

```
[Input Dataset]
      │
      ▼
[Preprocessing & Normalization (EMB-002)]
      │
      ▼
[Error Handling & Validation (EMB-007)]
      │
      ▼
[Batch Embedding Generation (EMB-003)]
      │
      ▼
[Embedding Serialization (EMB-006)]
      │
      ▼
[ChromaDB Ingestion (VDB-xxx)]
      │
      ▼
[Vector Retrieval (RAG-xxx)]
```

### Workstream Ownership
**Member 3** owns the following components in the PECS pipeline:
1. **Embedding Generation**: Model loading, preprocessing integration, batch inference, validation, and error handling.
2. **Embedding Persistence**: Serialization of generated vectors and metadata into an ingestion-ready format.
3. **ChromaDB Vector Store**: Schema definition, ingestion, indexing, and collection management.
4. **Retrieval**: Semantic query search and passing relevant contexts to downstream tasks.

*Note: Downstream consumers like the Question Engine and Assessment Engine use the retrieved context but do not own embedding generation or vector storage.*

---

## 2. Current Embedding Model

The canonical model used across the entire PECS workspace is:

- **Model Name**: `all-MiniLM-L6-v2` (loaded via the Hugging Face hub / sentence-transformers library)
- **Embedding Dimension**: `384`
- **Output Data Type**: `float32` (stored as NumPy arrays in memory and converted to plain Python float lists on disk)
- **Execution Device**: Running on the device returned by PyTorch/SentenceTransformers (defaults to `cpu` in our standard benchmark and evaluation environments).

### Warm-Load Behavior
To prevent cold-start overhead during API calls, the model is warm-loaded exactly once when the [embedder.py](file:///d:/PROJECT/PECS-Evaluation-API-RAG-Platform-GIT/src/embeddings/embedder.py) module is imported.
The module creates a single, global private instance of the `SentenceTransformer` model:
```python
_model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
```
All downstream callers access this pre-cached instance via the `get_embedder()` function, ensuring zero initialization overhead for subsequent calls.

---

## 3. Input Dataset

The embedding pipeline consumes the normalized dataset created and handed off by the Member 2 annotation workflow.

- **Actual File Path**: [data/processed/annotations_normalized.csv](file:///d:/PROJECT/PECS-Evaluation-API-RAG-Platform-GIT/data/processed/annotations_normalized.csv)
- **Expected Format**: Standard CSV format with a header row.
- **Required Columns**:
  - `record_id`: A unique string identifier for each prompt. This MUST be used as the vector store key (not `prompt_id`, as `prompt_id` can be duplicated across distinct text variations).
  - `prompt_text`: The raw prompt string that will be encoded.
- **Optional Metadata Columns**:
  - `domain`, `skill_level`, `c1_score` through `c6_score`, `final_score`, etc.
- **Embedded Field**: The `prompt_text` column.
- **Normalization**:
  Prior to embedding, every text is passed to `preprocess_for_embedding()` (in [embed_preprocess.py](file:///d:/PROJECT/PECS-Evaluation-API-RAG-Platform-GIT/src/embeddings/embed_preprocess.py)), which applies the canonical DS-005 text normalization rules from `normalize_text()` (defined in [normalization.py](file:///d:/PROJECT/PECS-Evaluation-API-RAG-Platform-GIT/src/common/normalization.py)).
- **Error Handling**:
  Inputs are validated using `validate_embedding_input()` (in [embed_error_handling.py](file:///d:/PROJECT/PECS-Evaluation-API-RAG-Platform-GIT/src/embeddings/embed_error_handling.py)). Empty strings (after normalization) or oversized strings exceeding `max_chars` (default `10,000`) are rejected, returning structured `EMPTY_INPUT` or `INPUT_TOO_LARGE` codes.

---

## 4. Environment Setup

To set up the workspace for embedding operations, run the following commands:

1. **Clone the Repository**
   ```bash
   git clone <repo-url>
   cd PECS-Evaluation-API-RAG-Platform-GIT
   ```

2. **Set Up Python Virtual Environment**
   ```bash
   python -m venv .venv
   # On Windows (cmd/powershell):
   .venv\Scripts\activate
   # On Unix/macOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Environment Setup**
   Run the embedding test suite to confirm that the package resolves and the SentenceTransformer model downloads/loads successfully:
   ```bash
   $env:PYTHONPATH="."; .venv\Scripts\pytest tests/member3/test_embedder.py
   ```

---

## 5. Embedding Generation

Embedding generation is integrated programmatically inside the Python codebase.

### Code Interface
Generate embeddings in batches using `embed_texts()` in [batch_embed.py](file:///d:/PROJECT/PECS-Evaluation-API-RAG-Platform-GIT/src/embeddings/batch_embed.py):
```python
from src.embeddings.batch_embed import embed_texts

texts = ["Explain methods to handle outliers in financial datasets.", "Suggest ways to improve model performance."]
embeddings = embed_texts(texts, batch_size=32)

# returns np.ndarray of shape (2, 384) with dtype float32
```

### Configuration Parameters
- `batch_size`: Max number of texts sent to `model.encode()` in one batch. Defaults to `32`. Must be a positive integer.
- `texts`: A sequence of strings. If empty, the function returns an empty `(0, 384)` float32 array without calling the model.

---

## 6. Regeneration Procedure

Follow these steps to regenerate embeddings when the source corpus changes:

1. **Update the Dataset**
   Replace the source file at [data/processed/annotations_normalized.csv](file:///d:/PROJECT/PECS-Evaluation-API-RAG-Platform-GIT/data/processed/annotations_normalized.csv) with the updated version.

2. **Verify Dataset Structure**
   Verify that the new file contains non-empty `record_id` and `prompt_text` columns.

3. **Validate and Filter Inputs**
   Run all texts through `validate_embedding_input()` (from [embed_error_handling.py](file:///d:/PROJECT/PECS-Evaluation-API-RAG-Platform-GIT/src/embeddings/embed_error_handling.py)) to identify empty inputs or oversized text values.

4. **Execute Generation**
   Load the CSV file using `pandas`, extract `prompt_text` and `record_id`, and run `embed_texts()` with the desired batch size.
   
5. **Verify Embedding Properties**
   - Assert the output shape is exactly `(N, 384)`, where `N` is the number of inputs.
   - Assert the array dtype is `np.float32`.
   - Assert all elements are finite numbers (no `NaN` or `Inf`).

6. **Serialize and Write Output**
   Convert each record into an `EmbeddingRecord` object and pass them to `serialize_embeddings()`. Save the serialized output to disk using `save_serialized()` (from [embedding_serializer.py](file:///d:/PROJECT/PECS-Evaluation-API-RAG-Platform-GIT/src/embeddings/embedding_serializer.py)).

7. **Downstream ChromaDB Rebuild**
   Load the serialized file and write to ChromaDB.
   > [!IMPORTANT]
   > Changing the underlying embeddings does not automatically update ChromaDB. When regenerating the corpus, you must delete the existing ChromaDB collection or reset the database, and run the ingestion script using `collection.upsert(**load_serialized(path))` to rebuild the index.

---

## 7. Embedding Persistence / EMB-006

To decoupled embedding generation from database operations, generated vectors are serialized into a standard, machine-readable JSON structure.

### `EmbeddingRecord` Dataclass
```python
@dataclass
class EmbeddingRecord:
    id: str                 # Unique record ID (record_id)
    embedding: Any          # 1D vector of 384 dimensions (np.ndarray or list)
    metadata: Dict[str, Any] # Flat dictionary of scalar values
    document: Optional[str] # Original raw text string (prompt_text)
```

### JSON Serialization Schema
The serialized output produced by `serialize_embeddings()` and written to disk by `save_serialized()` follows this format:
```json
{
  "ids": ["rec_001", "rec_002"],
  "embeddings": [
    [0.012, -0.045, ... 384 floats ...],
    [-0.089, 0.052, ... 384 floats ...]
  ],
  "metadatas": [
    {"domain": "CODE", "skill_level": "developing"},
    {"domain": "EDU", "skill_level": "strong"}
  ],
  "documents": [
    "Write a SQL query...",
    "Design a fraud detection system..."
  ]
}
```
This dictionary structure exactly matches ChromaDB's `collection.upsert()` keyword arguments one-to-one, allowing direct unpacking: `collection.upsert(**load_serialized(path))`.

---

## 8. Validation Checklist

Before handing off generated embeddings to ChromaDB ingestion, verify each item:

- [ ] Input file `data/processed/annotations_normalized.csv` exists and is non-empty.
- [ ] Every record contains a unique, non-empty `record_id`.
- [ ] No duplicate `record_id` values exist in the input dataset.
- [ ] Normalization succeeds using the canonical ds-005 preprocessor.
- [ ] Text inputs pass validation (no empty/whitespace strings or text exceeding 10,000 characters).
- [ ] Generated embedding vector array matches input count `N`.
- [ ] Every vector has exactly `384` dimensions.
- [ ] The output array dtype is `float32`.
- [ ] All vector values are finite (no `NaN`/`Inf` floats).
- [ ] Serialized JSON contains `ids`, `embeddings`, `metadatas`, and `documents` keys.
- [ ] Unique record IDs and prompt texts match the source CSV row-for-row.
- [ ] Local determinism unit tests pass successfully.
- [ ] All existing test files in `tests/member3` run and pass.

---

## 9. Performance Baseline — EMB-008

The batch embedding generation function `embed_texts()` was benchmarked using [benchmark/embedding_performance.py](file:///d:/PROJECT/PECS-Evaluation-API-RAG-Platform-GIT/benchmark/embedding_performance.py) on a 100-prompt subset from `annotations_normalized.csv`.

### Baseline Performance Metrics (CPU)
*These measurements are specific to the machine running the benchmark and should be used as a relative baseline only.*

- **Model**: `all-MiniLM-L6-v2`
- **Device**: `CPU`
- **Number of Records**: `100`
- **Embedding Dimension**: `384`

| Batch Size | Mean Time (s) | Min Time (s) | Max Time (s) | Throughput (records/s) | Latency (ms/record) |
|------------|---------------|--------------|--------------|------------------------|---------------------|
| 1          | 2.9114        | 2.7201       | 3.1822       | 34.50                  | 29.11               |
| 8          | 0.9966        | 0.9183       | 1.0362       | 100.67                 | 9.97                |
| 16         | 1.2127        | 1.1440       | 1.2729       | 82.62                  | 12.13               |
| 32         | 1.3366        | 1.3221       | 1.3576       | 74.83                  | 13.37               |
| 64         | 1.1525        | 1.0558       | 1.2631       | 87.24                  | 11.53               |

### Key Baseline Takeaways
1. **Optimal Batch Size**: On CPU, a batch size of **8** offered the best combination of execution time and throughput (reaching **100.67 records/s**).
2. **Framework Overhead**: Batch size 1 incurred massive PyTorch/sentence-transformers framework overhead per record (29.11 ms/record), which is resolved by batching.
3. **Future Benchmarking**: Future modifications to the embedding implementation should run the same benchmark script to verify there is no regression:
   ```bash
   $env:PYTHONPATH="."; .venv\Scripts\python.exe -m benchmark.embedding_performance
   ```

---

## 10. Troubleshooting

| Symptom / Error | Cause | Diagnostic / Resolution |
|-----------------|-------|-------------------------|
| `FileNotFoundError: ... annotations_normalized.csv` | The input dataset has not been generated. | Run the Member 2 preprocessing script to normalize the raw annotations. |
| `ModuleNotFoundError: No module named 'sentence_transformers'` | Dependencies are not installed. | Activate your virtual environment and run `pip install -r requirements.txt`. |
| `huggingface_hub.utils.HFValidationError` or network timeout | Could not fetch weights from HF. | Ensure internet access is enabled. If offline, make sure cache is populated under `~/.cache/huggingface/`. |
| `ValueError: max_chars must be a positive integer` | Invalid validation configuration. | Double check argument passed to `validate_embedding_input()`. |
| Error Code: `EMPTY_INPUT` | Input string is whitespace or empty after normalization. | Check the source record. Empty prompts cannot be embedded. |
| Error Code: `INPUT_TOO_LARGE` | Normalized prompt length exceeds max character limit. | Check source text. Oversized inputs must be truncated or split. |
| `ValueError: expected embedding dimension 384` | Wrong vector size passed to serializer. | Ensure the model has not been switched to a different model (e.g. 512/768-dim models) without updating `EXPECTED_DIMENSION`. |
| `ValueError: duplicate id found` | Multiple records share the same ID. | Ensure `record_id` is used, and it is unique. Do not use `prompt_id`. |

---

## 11. Reproducibility

To ensure identical embedding outputs across environments and runs, the pipeline maintains strict determinism:

1. **Pinned Dependency**: `sentence-transformers` is installed as configured in `requirements.txt`.
2. **Model Version**: The model is locked to `all-MiniLM-L6-v2` (which loads revision-tracked pytorch files).
3. **Deterministic Normalization**: Preprocessing is handled by DS-005 rules, ensuring identical character transforms.
4. **Deterministic Inference**: Inference is run in inference mode (`torch.no_grad()`) on CPU/GPU.
5. **Validation of Reproducibility**: Run the cross-process determinism check to confirm bitwise similarity:
   ```bash
   $env:PYTHONPATH="."; .venv\Scripts\pytest tests/member3/test_embedding_determinism.py
   ```

---

## 12. ChromaDB Handoff

The output produced by the serialization layer matches ChromaDB's API.

The handoff workflow operates as follows:
1. `load_serialized("data/processed/embeddings.json")` reads the persisted dictionary.
2. The returned dictionary contains the exact keys: `ids`, `embeddings`, `metadatas`, and `documents`.
3. In the future ChromaDB integration module, the data is unpacked directly:
   ```python
   import chromadb
   from src.embeddings.embedding_serializer import load_serialized

   client = chromadb.Client()
   collection = client.get_or_create_collection("pecs_prompts")

   data = load_serialized("data/processed/embeddings.json")
   collection.upsert(**data)
   ```
4. This ensures that metadata splits (train/validation/test) and `record_id` keys are ingested cleanly without any code modifications or reshaping.

---

## 13. Scope and Ownership

Member 3 owns the core semantic storage and retrieval tier of PECS. The boundary is defined as:

### Member 3 In-Scope Components
- Warm loading sentence-transformers model instance (`src/embeddings/embedder.py`).
- Preprocessing and normalization wrapping (`src/embeddings/embed_preprocess.py`).
- Batch embedding generation (`src/embeddings/batch_embed.py`).
- Input validation and length safety limits (`src/embeddings/embed_error_handling.py`).
- Vector serialization and deserialization (`src/embeddings/embedding_serializer.py`).
- ChromaDB collection lifecycle and ingestion scripts.
- Dense query retrieval.

### Out-of-Scope Components
- Question Generation, Prompt Scoring, Rubric Evaluations (owned by Member 2 / Assessment Engine).
- Frontend or external API endpoints consuming evaluation scores (owned by Downstream APIs).

---

## 14. Verification Procedure

Use this tutorial to verify the embedding pipeline end-to-end:

### End-to-End Verification Tutorial

1. **Perform Setup**
   Activate virtual environment and verify dependencies are installed.
   ```bash
   .venv\Scripts\activate
   ```

2. **Run Tests**
   Execute the full test suite to verify the code:
   ```bash
   $env:PYTHONPATH="."; .venv\Scripts\pytest
   ```
   All tests must pass.

3. **Run Performance Benchmark**
   Run the benchmark to verify throughput:
   ```bash
   $env:PYTHONPATH="."; .venv\Scripts\python.exe -m benchmark.embedding_performance
   ```
   Confirm files `benchmark/results/embedding_performance.json` and `embedding_performance.md` are generated and updated.

4. **Verify Persistence Flow**
   Open a python shell and verify serialization works:
   ```python
   import numpy as np
   from src.embeddings.embedding_serializer import EmbeddingRecord, serialize_embeddings

   records = [
       EmbeddingRecord(
           id="test_1",
           embedding=np.random.rand(384).tolist(),
           metadata={"domain": "CODE"},
           document="Test text content"
       )
   ]
   serialized = serialize_embeddings(records)
   assert "embeddings" in serialized
   assert len(serialized["embeddings"][0]) == 384
   print("Persistence verified successfully!")
   ```
