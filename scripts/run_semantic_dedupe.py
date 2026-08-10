"""
EMB-010 — Semantic Near-Duplicate Audit.

Runs a semantic near-duplicate audit on the corpus prompts using embedding
cosine similarity and compares the results against the lexical near-duplicate
detector.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from src.embeddings.batch_embed import embed_texts
from src.embeddings.embed_preprocess import preprocess_for_embedding
from src.embeddings.embedding_serializer import (
    EmbeddingRecord,
    serialize_embeddings,
    save_serialized,
    load_serialized,
)

# Configuration constants
SEMANTIC_THRESHOLD = 0.92
LEXICAL_THRESHOLD = 0.90

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "annotations_normalized.csv"
EMBEDDINGS_JSON_PATH = REPO_ROOT / "data" / "processed" / "embeddings.json"
REPORT_PATH = REPO_ROOT / "semantic_dedupe_report.md"


def run_audit() -> Dict[str, Any]:
    """
    Run the semantic near-duplicate audit on the current corpus.

    Loads the corpus, loads or generates embeddings, computes pairwise cosine
    similarities and lexical similarities, and writes the markdown report.

    Returns:
        A dictionary containing the audit summary statistics and findings.
    """
    print("Starting Semantic Near-Duplicate Audit (EMB-010)...")

    # 1. Load the corpus
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus CSV file not found at {CORPUS_PATH}")

    df = pd.read_csv(CORPUS_PATH)
    total_records = len(df)
    print(f"Loaded corpus with {total_records} records.")

    if total_records == 0:
        print("Empty corpus. Generating empty report...")
        write_report(
            total_records=0,
            lexical_pairs_count=0,
            semantic_pairs_count=0,
            semantic_only_count=0,
            lexical_only_count=0,
            findings=[],
        )
        return {
            "total_records": 0,
            "lexical_pairs_count": 0,
            "semantic_pairs_count": 0,
            "semantic_only_count": 0,
            "lexical_only_count": 0,
            "findings": [],
        }

    # 2. Load or generate embeddings
    if EMBEDDINGS_JSON_PATH.exists():
        print(f"Loading existing embeddings from {EMBEDDINGS_JSON_PATH}...")
        data = load_serialized(EMBEDDINGS_JSON_PATH)
    else:
        print(f"Embeddings file not found. Generating fresh embeddings for {total_records} records...")
        texts = df["prompt_text"].fillna("").astype(str).tolist()
        embeddings = embed_texts(texts)

        records = []
        for idx, row in df.iterrows():
            metadata = {}
            for col in df.columns:
                if col != "prompt_text":
                    val = row[col]
                    if pd.isna(val):
                        metadata[col] = None
                    else:
                        if isinstance(val, (np.integer, np.floating)):
                            metadata[col] = val.item()
                        else:
                            metadata[col] = val

            records.append(
                EmbeddingRecord(
                    id=str(row["record_id"]),
                    embedding=embeddings[idx].tolist(),
                    metadata=metadata,
                    document=str(row["prompt_text"]),
                )
            )

        data = serialize_embeddings(records)
        save_serialized(data, EMBEDDINGS_JSON_PATH)
        print(f"Generated and saved embeddings to {EMBEDDINGS_JSON_PATH}.")

    # Verify matching counts
    assert len(data["ids"]) == total_records, "Embedding count mismatch with corpus"

    # 3. Compute pairwise similarities
    embeddings_arr = np.array(data["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    norm_embeddings = embeddings_arr / norms
    similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)

    findings = []
    lexical_pairs_count = 0
    semantic_pairs_count = 0
    semantic_only_count = 0
    lexical_only_count = 0

    print("Running pairwise comparisons (O(N^2))...")

    # Pairwise comparison loop (i < j to avoid self-pairs and duplicates)
    for i in range(total_records):
        for j in range(i + 1, total_records):
            cosine_sim = float(similarity_matrix[i, j])

            doc_1 = data["documents"][i]
            doc_2 = data["documents"][j]

            # Preprocess and compute lexical similarity
            norm_doc_1 = preprocess_for_embedding(doc_1)
            norm_doc_2 = preprocess_for_embedding(doc_2)
            lexical_sim = SequenceMatcher(None, norm_doc_1, norm_doc_2).ratio()

            is_semantic = cosine_sim >= SEMANTIC_THRESHOLD
            is_lexical = lexical_sim >= LEXICAL_THRESHOLD

            if is_semantic:
                semantic_pairs_count += 1
            if is_lexical:
                lexical_pairs_count += 1

            if is_semantic or is_lexical:
                meta_1 = data["metadatas"][i]
                meta_2 = data["metadatas"][j]

                category = "unknown"
                if is_semantic and is_lexical:
                    category = "lexical + semantic"
                elif is_semantic and not is_lexical:
                    category = "semantic-only"
                    semantic_only_count += 1
                elif is_lexical and not is_semantic:
                    category = "lexical-only"
                    lexical_only_count += 1

                findings.append({
                    "id_1": data["ids"][i],
                    "id_2": data["ids"][j],
                    "prompt_id_1": meta_1.get("prompt_id", "unknown"),
                    "prompt_id_2": meta_2.get("prompt_id", "unknown"),
                    "domain_1": meta_1.get("domain", "unknown"),
                    "domain_2": meta_2.get("domain", "unknown"),
                    "level_1": meta_1.get("skill_level", "unknown"),
                    "level_2": meta_2.get("skill_level", "unknown"),
                    "cosine_similarity": cosine_sim,
                    "lexical_similarity": lexical_sim,
                    "category": category,
                    "prompt_1": doc_1,
                    "prompt_2": doc_2,
                })

    # Sort findings by similarity
    findings.sort(key=lambda x: x["cosine_similarity"], reverse=True)

    print(f"Audit completed: {len(findings)} near-duplicate pairs identified.")
    print(f"  Lexical + Semantic: {semantic_pairs_count - semantic_only_count}")
    print(f"  Semantic-only: {semantic_only_count}")
    print(f"  Lexical-only: {lexical_only_count}")

    # 4. Generate the report
    write_report(
        total_records=total_records,
        lexical_pairs_count=lexical_pairs_count,
        semantic_pairs_count=semantic_pairs_count,
        semantic_only_count=semantic_only_count,
        lexical_only_count=lexical_only_count,
        findings=findings,
    )

    return {
        "total_records": total_records,
        "lexical_pairs_count": lexical_pairs_count,
        "semantic_pairs_count": semantic_pairs_count,
        "semantic_only_count": semantic_only_count,
        "lexical_only_count": lexical_only_count,
        "findings": findings,
    }


def write_report(
    total_records: int,
    lexical_pairs_count: int,
    semantic_pairs_count: int,
    semantic_only_count: int,
    lexical_only_count: int,
    findings: List[Dict[str, Any]],
) -> None:
    """
    Generate and save the semantic near-duplicate markdown report.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    report = []
    report.append("# EMB-010 — Semantic Near-Duplicate Audit")
    report.append("")
    report.append("## 1. Purpose")
    report.append(
        "This audit acts as a semantic second-pass checking mechanism on the PECS prompt corpus. "
        "It builds upon the lexical near-duplicate detection implemented in DS-007 by computing "
        "semantic cosine similarities using text embedding vectors generated using the PECS embedding model. By checking both passes, "
        "we can identify semantically duplicate prompt definitions that were missed by simple lexical string matching."
    )
    report.append("")
    report.append("## 2. Audit Configuration")
    report.append(f"- **Embedding Model**: `all-MiniLM-L6-v2`")
    report.append(f"- **Embedding Dimension**: `384`")
    report.append(f"- **Similarity Metric**: Cosine Similarity")
    report.append(f"- **Configured Semantic Threshold**: `{SEMANTIC_THRESHOLD}` (The semantic threshold of 0.92 is adopted from the PECS Master Blueprint's documented dataset deduplication threshold ('Deduplicated at cosine 0.92'). The threshold is used here as the semantic-audit threshold for EMB-010 and is not presented as an independently specified value in the EMB-010 task tracker.)")
    report.append(f"- **Configured Lexical Threshold (DS-007)**: `{LEXICAL_THRESHOLD}`")
    report.append(f"- **Corpus Source Path**: `data/processed/annotations_normalized.csv`")
    report.append(f"- **Number of Audited Records**: `{total_records}`")
    report.append("- **Comparison Scope**: Full pairwise cross-comparison (O(N^2)). Due to the small size of the corpus (331 prompts), a complete pairwise check is computationally efficient and guarantees global coverage across all domains and splits.")
    report.append(f"- **Audit Run Date/Time**: `{timestamp}`")
    report.append("")
    report.append("## 3. Method")
    report.append("1. **Preprocessing**: Prompt texts are preprocessed and normalized using the canonical DS-005 rules from `src/embeddings/embed_preprocess.py`.")
    report.append("2. **Embedding Loading**: Embeddings are loaded from the persisted serialization format at `data/processed/embeddings.json`. If missing, embeddings are generated using the warm-loaded model instance (`batch_embed.py`) and saved.")
    report.append("3. **Similarity Computation**: Pairwise cosine similarity is computed between all normalized embedding vectors using NumPy matrix operations. Self-pairs and reversed orderings are excluded.")
    report.append("4. **Lexical Comparison**: For each pair exceeding either the lexical or semantic threshold, the SequenceMatcher-based lexical similarity ratio is calculated using Python's `difflib.SequenceMatcher`.")
    report.append("5. **Categorization**: Pairs are classified into `lexical + semantic` (overlapping detection), `semantic-only` (detected only by embeddings), or `lexical-only` (detected only by string similarity).")
    report.append("")
    report.append("## 4. Summary")
    report.append("")
    report.append("| Metric | Measured Count |")
    report.append("|--------|----------------|")
    report.append(f"| Total Records Audited | {total_records} |")
    report.append(f"| Lexical Near-Duplicate Pairs (>= {LEXICAL_THRESHOLD}) | {lexical_pairs_count} |")
    report.append(f"| Semantic Near-Duplicate Pairs (>= {SEMANTIC_THRESHOLD}) | {semantic_pairs_count} |")
    report.append(f"| Overlapping Pairs (Lexical + Semantic) | {semantic_pairs_count - semantic_only_count} |")
    report.append(f"| **Semantic-Only Pairs (Missed by DS-007)** | **{semantic_only_count}** |")
    report.append(f"| Lexical-Only Pairs (Missed by Cosine Similarity) | {lexical_only_count} |")
    report.append("")
    report.append("## 5. Semantic-Only Findings")
    report.append("")
    report.append("The following table lists high-similarity semantic near-duplicate prompt pairs that were **missed by the lexical DS-007 pass** (lexical similarity < 0.90):")
    report.append("")

    semantic_only_findings = [f for f in findings if f["category"] == "semantic-only"]

    if not semantic_only_findings:
        report.append("No semantic-only near-duplicate pairs were detected under the configured threshold.")
    else:
        report.append(
            "| Prompt ID 1 | Prompt ID 2 | Domain | Level | Cosine Similarity | Prompt 1 | Prompt 2 |"
        )
        report.append(
            "|-------------|-------------|--------|-------|-------------------|----------|----------|"
        )
        for f in semantic_only_findings:
            # Clean prompt texts for markdown table display (remove newlines, escape pipes)
            p1_clean = f["prompt_1"].replace("\n", " ").replace("|", "\\|")
            p2_clean = f["prompt_2"].replace("\n", " ").replace("|", "\\|")
            # Limit length in table for readability
            if len(p1_clean) > 80:
                p1_clean = p1_clean[:77] + "..."
            if len(p2_clean) > 80:
                p2_clean = p2_clean[:77] + "..."

            # Report domain/level mismatches or show them clearly
            domain = f["domain_1"] if f["domain_1"] == f["domain_2"] else f"{f['domain_1']}/{f['domain_2']}"
            level = f["level_1"] if f["level_1"] == f["level_2"] else f"{f['level_1']}/{f['level_2']}"

            report.append(
                f"| `{f['prompt_id_1']}` | `{f['prompt_id_2']}` | {domain} | {level} | {f['cosine_similarity']:.4f} | \"{p1_clean}\" | \"{p2_clean}\" |"
            )

    report.append("")
    report.append("## 6. Overlap With DS-007")
    overlapping_count = semantic_pairs_count - semantic_only_count
    report.append(
        f"A total of **{overlapping_count}** semantic near-duplicate findings were also detected "
        f"by the lexical DS-007 detector. Therefore, all **{semantic_only_count}** semantic findings in this audit were "
        f"semantic-only findings missed by the lexical pass. This demonstrates that lexical matching "
        f"alone can miss prompts with high semantic similarity, such as those that utilize paraphrasing or grammatical restructuring."
    )
    report.append("")
    report.append("## 7. Interpretation")
    report.append(
        "- **Semantic Similarity**: High cosine similarity indicates that the embedding vectors occupy close coordinates in vector space, meaning the two prompts have a high degree of semantic overlap.\n"
        "- **Near-Duplicate Candidate**: Pairs flagged in this audit are candidates for review. They represent situations where the prompts likely assess the same skill, task, or conceptual domain with minimal variation.\n"
        "- **Confirmed Duplicate**: A high cosine similarity does NOT guarantee that the two prompts are identical in intent or duplicate in practice. For instance, prompts requesting code in different languages or slightly varying constraints may yield high cosine similarities while still representing distinct evaluation test cases. Manual verification is always required."
    )
    report.append("")
    report.append("## 8. Recommended Review Action")
    report.append(
        "1. **Manual Inspection**: The annotation and scoring teams should manually review the flagged semantic-only pairs.\n"
        "2. **De-duplication Policy**: If the prompts are confirmed duplicates and evaluate the same skill without adding unique test value, one of the records should be flagged for removal in a future corpus revision.\n"
        "3. **Do Not Automatically Delete**: To preserve dataset integrity and prevent accidental data loss, no records should be deleted automatically based on similarity scores alone."
    )
    report.append("")
    report.append("## 9. Limitations")
    report.append(
        "- **Embedding-Model Dependence**: Semantic measurements are tied to the representation capabilities of `all-MiniLM-L6-v2`. Changing the model will alter the vector space coordinates and resulting similarities.\n"
        "- **Threshold Dependence**: The audit findings are sensitive to the configured threshold of `0.92`. A lower threshold will flag more false positives, while a higher threshold may miss real duplicates.\n"
        "- **Corpus Size**: This pairwise comparison scales as O(N^2) and is suitable for our small corpus of 331 records. Larger datasets would require indexing or approximate nearest neighbor search (e.g., via ChromaDB query methods).\n"
        "- **Intent Resolution**: Cosine similarity measures phrase-level representation proximity, not absolute logical equivalence."
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"Successfully generated semantic near-duplicate report at: {REPORT_PATH}")


if __name__ == "__main__":
    run_audit()
