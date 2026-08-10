"""
EMB-005 — Verify deterministic/reproducible embedding generation.

Purpose:
    Confirm that the existing embedding pipeline (EMB-001/002/003) produces
    the same embedding vector for the same input text, both:
      (a) across repeated calls within a single Python process, and
      (b) across separate Python process restarts.

    This protects against silent nondeterminism creeping in later --
    e.g. from a model update, a change in tokenization/preprocessing,
    or an accidental dependency on process-local state -- which would
    quietly break RAG retrieval consistency (VDB-xxx / RAG-xxx) since
    the same calibration text must always map to the same vector.

Contract under test:
    embed_texts(texts, batch_size=32) -> np.ndarray
        Identical input text -> cosine similarity of the resulting
        vectors, run-to-run, is 1.0 within floating-point tolerance.

Scope:
    This file exercises the REAL embedding pipeline (no fake embedder),
    since determinism is a property of the actual model + preprocessing,
    not something a fake can meaningfully verify. It does not duplicate
    EMB-004's shape/dtype checks beyond the minimum needed to sanity
    check the vectors under test here.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.embeddings.batch_embed import embed_texts

# Fixed input text used for all determinism checks in this file.
FIXED_TEXT = "deterministic PECS embedding test"

# Cosine similarity between two embeddings of identical text should be
# essentially exactly 1.0. all-MiniLM-L6-v2 on CPU is a deterministic
# forward pass (no dropout/sampling at inference time), so the only
# expected deviation is floating-point summation order noise, which is
# far smaller than 1e-5. We use 1e-5 as a safety margin rather than
# exact equality, since bitwise-identical floats are not guaranteed
# across separate process invocations / BLAS thread scheduling.
COSINE_SIMILARITY_TOLERANCE = 1e-5

# Repository root, derived relative to this test file so the test works
# on any developer's machine after cloning (no hardcoded paths).
# tests/member3/test_embedding_determinism.py -> parents[2] == repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1D vectors.

    Fails loudly (rather than silently dividing by zero) if either
    vector has zero norm, since a real sentence embedding should never
    be the zero vector -- that itself would indicate a broken model.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    assert norm_a > 0, "Embedding vector 'a' has zero norm (unexpected zero vector)"
    assert norm_b > 0, "Embedding vector 'b' has zero norm (unexpected zero vector)"

    return float(np.dot(a, b) / (norm_a * norm_b))


def _run_embedding_in_subprocess(text: str) -> np.ndarray:
    """
    Run the real embedding pipeline in a fresh Python subprocess and
    return the resulting embedding vector for `text`.

    A fresh interpreter is used (via sys.executable) to genuinely
    simulate a process restart, as required by the EMB-005 acceptance
    criterion. The subprocess:
      - inherits the current environment (so any existing config is
        preserved),
      - has the repository root explicitly added to PYTHONPATH (so
        `from src.embeddings...` imports resolve regardless of how the
        parent pytest process was launched),
      - imports and calls the existing embed_texts() function directly
        (no new embedding implementation is introduced),
      - prints the resulting vector as JSON on stdout, which the parent
        process parses back into a NumPy array.
    """
    script = (
        "import json\n"
        "from src.embeddings.batch_embed import embed_texts\n"
        f"vector = embed_texts([{text!r}])[0]\n"
        "print(json.dumps(vector.tolist()))\n"
    )

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_root_str = str(REPO_ROOT)
    env["PYTHONPATH"] = (
        repo_root_str
        if not existing_pythonpath
        else repo_root_str + os.pathsep + existing_pythonpath
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root_str,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"Subprocess embedding run failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    vector_list = json.loads(result.stdout.strip())
    return np.array(vector_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Test 1 — Same-process determinism
# ---------------------------------------------------------------------------


def test_same_process_repeated_calls_are_deterministic():
    """
    Calling embed_texts() twice, in the same process, on identical input
    must produce (near-)identical vectors.

    We compare via cosine similarity rather than exact float equality,
    because floating-point summation order can introduce tiny numerical
    noise even when the underlying computation is logically deterministic
    (e.g. differing internal batching/threading). Cosine similarity close
    to 1.0 within a small tolerance is the right acceptance test for
    "same embedding," matching the tracker's explicit acceptance
    criterion.
    """
    texts = [FIXED_TEXT]

    run_1 = embed_texts(texts)
    run_2 = embed_texts(texts)

    # Sanity: both runs produce a single 384-dim float32 vector.
    assert run_1.shape == (1, 384)
    assert run_2.shape == (1, 384)
    assert run_1.dtype == np.float32
    assert run_2.dtype == np.float32
    assert np.all(np.isfinite(run_1)), "run_1 embedding contains non-finite values"
    assert np.all(np.isfinite(run_2)), "run_2 embedding contains non-finite values"

    similarity = cosine_similarity(run_1[0], run_2[0])

    assert np.isclose(similarity, 1.0, atol=COSINE_SIMILARITY_TOLERANCE), (
        f"Expected cosine similarity ~1.0 for repeated calls on identical "
        f"input, got {similarity}. This suggests the embedding pipeline "
        f"has become nondeterministic within a single process."
    )


# ---------------------------------------------------------------------------
# Test 2 — Cross-process (restart) determinism
# ---------------------------------------------------------------------------


def test_cross_process_restart_is_deterministic():
    """
    Embedding the same fixed text in two separate, freshly-started Python
    processes must produce (near-)identical vectors.

    This specifically covers the tracker's "across process restarts"
    requirement, which a same-process test cannot verify -- a bug tied to
    module-level state (e.g. an uninitialized cache, a non-deterministic
    warm-load path) could pass a same-process check while still failing
    across restarts.

    Only two subprocesses are used (the minimum needed for a pairwise
    comparison), since each subprocess must warm-load the full
    SentenceTransformer model.
    """
    vector_from_process_1 = _run_embedding_in_subprocess(FIXED_TEXT)
    vector_from_process_2 = _run_embedding_in_subprocess(FIXED_TEXT)

    assert vector_from_process_1.shape == (384,)
    assert vector_from_process_2.shape == (384,)
    assert vector_from_process_1.dtype == np.float32
    assert vector_from_process_2.dtype == np.float32
    assert np.all(np.isfinite(vector_from_process_1)), (
        "Vector from subprocess 1 contains non-finite values"
    )
    assert np.all(np.isfinite(vector_from_process_2)), (
        "Vector from subprocess 2 contains non-finite values"
    )

    similarity = cosine_similarity(vector_from_process_1, vector_from_process_2)

    assert np.isclose(similarity, 1.0, atol=COSINE_SIMILARITY_TOLERANCE), (
        f"Expected cosine similarity ~1.0 across two fresh process runs on "
        f"identical input, got {similarity}. This suggests the embedding "
        f"pipeline is not reproducible across process restarts."
    )