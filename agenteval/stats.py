from __future__ import annotations

import math


def wilson_interval(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a pass rate.

    Better than the normal approximation at the extremes that matter here: with
    10/10 passes the honest statement is "somewhere in 72%-100%", not "100%".
    """
    if n <= 0:
        return (0.0, 1.0)
    p = passes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
