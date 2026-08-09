from __future__ import annotations

import math
from collections import Counter
import pandas as pd


def cohens_kappa(a, b) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if pd.notna(x) and pd.notna(y)]
    if not pairs:
        return float("nan")
    n = len(pairs)
    categories = sorted(set([x for x, _ in pairs] + [y for _, y in pairs]))
    po = sum(x == y for x, y in pairs) / n
    counts_a, counts_b = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum((counts_a[c] / n) * (counts_b[c] / n) for c in categories)
    if math.isclose(1 - pe, 0):
        return 1.0 if math.isclose(po, 1) else 0.0
    return (po - pe) / (1 - pe)


def agreement_by_criterion(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(1, 7):
        k = cohens_kappa(df[f"gpt_c{i}"], df[f"gem_c{i}"])
        k2 = cohens_kappa(df[f"gpt_c{i}"], df[f"cla_c{i}"])
        k3 = cohens_kappa(df[f"gem_c{i}"], df[f"cla_c{i}"])
        rows.append({"criterion": f"C{i}", "gpt_gem_kappa": k, "gpt_claude_kappa": k2, "gem_claude_kappa": k3})
    return pd.DataFrame(rows)
