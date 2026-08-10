# EMB-010 — Semantic Near-Duplicate Audit

## 1. Purpose
This audit acts as a semantic second-pass checking mechanism on the PECS prompt corpus. It builds upon the lexical near-duplicate detection implemented in DS-007 by computing semantic cosine similarities using text embedding vectors generated using the PECS embedding model. By checking both passes, we can identify semantically duplicate prompt definitions that were missed by simple lexical string matching.

## 2. Audit Configuration
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Embedding Dimension**: `384`
- **Similarity Metric**: Cosine Similarity
- **Configured Semantic Threshold**: `0.92` (The semantic threshold of 0.92 is adopted from the PECS Master Blueprint's documented dataset deduplication threshold ('Deduplicated at cosine 0.92'). The threshold is used here as the semantic-audit threshold for EMB-010 and is not presented as an independently specified value in the EMB-010 task tracker.)
- **Configured Lexical Threshold (DS-007)**: `0.9`
- **Corpus Source Path**: `data/processed/annotations_normalized.csv`
- **Number of Audited Records**: `331`
- **Comparison Scope**: Full pairwise cross-comparison (O(N^2)). Due to the small size of the corpus (331 prompts), a complete pairwise check is computationally efficient and guarantees global coverage across all domains and splits.
- **Audit Run Date/Time**: `2026-08-10 16:32:05 UTC`

## 3. Method
1. **Preprocessing**: Prompt texts are preprocessed and normalized using the canonical DS-005 rules from `src/embeddings/embed_preprocess.py`.
2. **Embedding Loading**: Embeddings are loaded from the persisted serialization format at `data/processed/embeddings.json`. If missing, embeddings are generated using the warm-loaded model instance (`batch_embed.py`) and saved.
3. **Similarity Computation**: Pairwise cosine similarity is computed between all normalized embedding vectors using NumPy matrix operations. Self-pairs and reversed orderings are excluded.
4. **Lexical Comparison**: For each pair exceeding either the lexical or semantic threshold, the SequenceMatcher-based lexical similarity ratio is calculated using Python's `difflib.SequenceMatcher`.
5. **Categorization**: Pairs are classified into `lexical + semantic` (overlapping detection), `semantic-only` (detected only by embeddings), or `lexical-only` (detected only by string similarity).

## 4. Summary

| Metric | Measured Count |
|--------|----------------|
| Total Records Audited | 331 |
| Lexical Near-Duplicate Pairs (>= 0.9) | 0 |
| Semantic Near-Duplicate Pairs (>= 0.92) | 4 |
| Overlapping Pairs (Lexical + Semantic) | 0 |
| **Semantic-Only Pairs (Missed by DS-007)** | **4** |
| Lexical-Only Pairs (Missed by Cosine Similarity) | 0 |

## 5. Semantic-Only Findings

The following table lists high-similarity semantic near-duplicate prompt pairs that were **missed by the lexical DS-007 pass** (lexical similarity < 0.90):

| Prompt ID 1 | Prompt ID 2 | Domain | Level | Cosine Similarity | Prompt 1 | Prompt 2 |
|-------------|-------------|--------|-------|-------------------|----------|----------|
| `PELS-L1-CODE-018` | `PELS-L1-CODE-024` | CODE | developing | 0.9824 | "Explain linear regression with a simple example." | "Explain linear regression in simple terms with one example." |
| `PELS-L1-CODE-019` | `PELS-L1-CODE-028` | CODE | weak/developing | 0.9674 | "Explain K-means clustering and where it is used." | "Explain K-means clustering and where it is used in real life." |
| `PELS-L1-CODE-DSA-004` | `PELS-L1-CODE-DSA-053` | CODE | weak | 0.9543 | "how does recursion work" | "explain recursion" |
| `PELS-L1-CODE-DSA-005` | `PELS-L1-CODE-DSA-017` | CODE | weak | 0.9211 | "explain dynamic programming" | "Explain dynamic programming with a simple example for a beginner." |

## 6. Overlap With DS-007
A total of **0** semantic near-duplicate findings were also detected by the lexical DS-007 detector. Therefore, all **4** semantic findings in this audit were semantic-only findings missed by the lexical pass. This demonstrates that lexical matching alone can miss prompts with high semantic similarity, such as those that utilize paraphrasing or grammatical restructuring.

## 7. Interpretation
- **Semantic Similarity**: High cosine similarity indicates that the embedding vectors occupy close coordinates in vector space, meaning the two prompts have a high degree of semantic overlap.
- **Near-Duplicate Candidate**: Pairs flagged in this audit are candidates for review. They represent situations where the prompts likely assess the same skill, task, or conceptual domain with minimal variation.
- **Confirmed Duplicate**: A high cosine similarity does NOT guarantee that the two prompts are identical in intent or duplicate in practice. For instance, prompts requesting code in different languages or slightly varying constraints may yield high cosine similarities while still representing distinct evaluation test cases. Manual verification is always required.

## 8. Recommended Review Action
1. **Manual Inspection**: The annotation and scoring teams should manually review the flagged semantic-only pairs.
2. **De-duplication Policy**: If the prompts are confirmed duplicates and evaluate the same skill without adding unique test value, one of the records should be flagged for removal in a future corpus revision.
3. **Do Not Automatically Delete**: To preserve dataset integrity and prevent accidental data loss, no records should be deleted automatically based on similarity scores alone.

## 9. Limitations
- **Embedding-Model Dependence**: Semantic measurements are tied to the representation capabilities of `all-MiniLM-L6-v2`. Changing the model will alter the vector space coordinates and resulting similarities.
- **Threshold Dependence**: The audit findings are sensitive to the configured threshold of `0.92`. A lower threshold will flag more false positives, while a higher threshold may miss real duplicates.
- **Corpus Size**: This pairwise comparison scales as O(N^2) and is suitable for our small corpus of 331 records. Larger datasets would require indexing or approximate nearest neighbor search (e.g., via ChromaDB query methods).
- **Intent Resolution**: Cosine similarity measures phrase-level representation proximity, not absolute logical equivalence.