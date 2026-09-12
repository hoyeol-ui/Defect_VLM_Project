from __future__ import annotations
import numpy as np
from typing import Any

def trapezoid_aulc(rows: list[dict[str, Any]], metric: str) -> float:
    ordered = sorted(rows, key=lambda row: int(row['round']))
    if [int(row['round']) for row in ordered] != list(range(6)):
        raise RuntimeError('AULC requires exactly rounds 0..5')
    x = np.asarray([float(row['budget_fraction']) for row in ordered], dtype=np.float64)
    y = np.asarray([float(row[metric]) for row in ordered], dtype=np.float64)
    if not np.all(np.isfinite(y)) or not np.all(np.diff(x) > 0):
        raise RuntimeError('AULC received invalid values')
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))

def percentile_ci(values: list[float], seed: int, replicates: int=100000) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    boot = array[indices].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return (float(low), float(high))
