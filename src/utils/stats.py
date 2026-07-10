"""Statistics utilities: dataset profiling and rigorous metric aggregation.

Two responsibilities:

1. **Dataset statistics** -- profile the interaction data (sparsity, popularity
   skew, cold-item / cold-user counts, temporal span). These numbers go in the
   README and justify design choices (e.g. why a popularity baseline is strong).

2. **Metric aggregation** -- because the project lives or dies on honesty, every
   ablation is run over multiple seeds and reported as ``mean +/- std`` with a
   normal-approx confidence interval. We also provide a paired bootstrap test
   over per-user metrics so a claimed improvement can be checked against noise
   rather than asserted. The rule enforced downstream: *a difference smaller than
   the seed std, or whose CI straddles zero, is reported as "within noise".*
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Dataset statistics                                                          #
# --------------------------------------------------------------------------- #
def dataset_statistics(
    interactions: pd.DataFrame,
    user_col: str = "user_idx",
    item_col: str = "item_idx",
    time_col: str = "timestamp",
) -> dict:
    """Compute descriptive statistics for an interaction table.

    Args:
        interactions: Rows of (user, item, ..., timestamp) interactions.
        user_col: User id column.
        item_col: Item id column.
        time_col: Unix-timestamp column.

    Returns:
        A JSON-serialisable dict of statistics (counts, sparsity, popularity
        Gini, per-user/per-item interaction quantiles, temporal span).
    """
    n_int = int(len(interactions))
    n_users = int(interactions[user_col].nunique())
    n_items = int(interactions[item_col].nunique())
    density = n_int / (n_users * n_items) if n_users and n_items else 0.0

    items_per_user = interactions.groupby(user_col).size().to_numpy()
    users_per_item = interactions.groupby(item_col).size().to_numpy()

    stats = {
        "n_interactions": n_int,
        "n_users": n_users,
        "n_items": n_items,
        "density": density,
        "sparsity": 1.0 - density,
        "items_per_user": _quantile_summary(items_per_user),
        "users_per_item": _quantile_summary(users_per_item),
        "popularity_gini": float(_gini(users_per_item)),
    }
    if time_col in interactions.columns:
        ts = interactions[time_col].to_numpy()
        stats["time_span_days"] = float((ts.max() - ts.min()) / 86400.0)
    return stats


def _quantile_summary(values: np.ndarray) -> dict:
    """Return min/median/mean/p90/p99/max for a 1-D array."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array (popularity concentration).

    0 = perfectly uniform popularity, 1 = one item takes everything. This
    quantifies the head/tail skew that drives the cold-start and coverage story.
    """
    values = np.sort(np.asarray(values, dtype=np.float64))
    n = values.size
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * values) / (n * values.sum())) - (n + 1.0) / n)


# --------------------------------------------------------------------------- #
# Metric aggregation across seeds                                             #
# --------------------------------------------------------------------------- #
@dataclass
class AggregatedMetric:
    """A metric summarised across seeds.

    Attributes:
        mean: Sample mean across seeds.
        std: Sample standard deviation (ddof=1 when n > 1, else 0).
        n: Number of seeds.
        ci95_halfwidth: Half-width of a 95% normal-approx confidence interval.
        values: The per-seed values (kept for provenance / plotting).
    """

    mean: float
    std: float
    n: int
    ci95_halfwidth: float
    values: list[float]

    def as_dict(self) -> dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "n": self.n,
            "ci95_halfwidth": self.ci95_halfwidth,
            "values": self.values,
        }

    def pretty(self) -> str:
        """Human-readable ``mean +/- std`` string for logs and the README."""
        return f"{self.mean:.4f} +/- {self.std:.4f} (n={self.n})"


def aggregate_over_seeds(values: Sequence[float]) -> AggregatedMetric:
    """Aggregate a metric measured once per seed."""
    arr = np.asarray(list(values), dtype=np.float64)
    n = int(arr.size)
    mean = float(arr.mean()) if n else float("nan")
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    # 1.96 * SEM for a normal-approx 95% interval; degenerate when n <= 1.
    ci = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return AggregatedMetric(mean=mean, std=std, n=n, ci95_halfwidth=ci, values=arr.tolist())


def is_within_noise(
    delta: float,
    reference_std: float,
    ci95_halfwidth: Optional[float] = None,
) -> bool:
    """Decide whether an observed improvement is indistinguishable from noise.

    We call a delta "within noise" if it is smaller than the seed std, or if a
    provided 95% CI half-width exceeds the delta magnitude. This is the guardrail
    that stops us from writing a fragile directional claim (see the proposal's
    de-risking rule).
    """
    if abs(delta) <= reference_std:
        return True
    if ci95_halfwidth is not None and abs(delta) <= ci95_halfwidth:
        return True
    return False


def paired_bootstrap_pvalue(
    per_user_a: Sequence[float],
    per_user_b: Sequence[float],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap over per-user metrics to test A vs B.

    Given per-user metric values for two systems evaluated on the *same* users,
    resample users with replacement ``n_boot`` times and measure how often the
    mean difference flips sign. Returns the observed mean delta, a 95% bootstrap
    interval, and a two-sided p-value proxy.

    This complements seed variance: seed std captures training noise, the
    bootstrap captures evaluation-set (user-sampling) noise.
    """
    a = np.asarray(per_user_a, dtype=np.float64)
    b = np.asarray(per_user_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Per-user arrays must align (same users, same order).")
    rng = np.random.default_rng(seed)
    diff = a - b
    observed = float(diff.mean())
    n = diff.size
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = diff[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Two-sided p-value proxy: fraction of bootstrap means on the opposite side
    # of zero from the observed effect, doubled.
    if observed >= 0:
        p = 2.0 * float((boot <= 0).mean())
    else:
        p = 2.0 * float((boot >= 0).mean())
    return {
        "observed_delta": observed,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_value": min(1.0, p),
        "significant_at_0.05": bool(min(1.0, p) < 0.05),
    }


def head_tail_split(
    item_popularity: dict[int, int],
    head_fraction: float = 0.2,
) -> tuple[set[int], set[int]]:
    """Partition items into a popular *head* and a long *tail*.

    The head is the top ``head_fraction`` of items by interaction count. Used to
    report metrics per-segment so we can see *where* a change acts (the core of
    the direction-agnostic negative-sampling analysis).
    """
    if not item_popularity:
        return set(), set()
    ranked = sorted(item_popularity.items(), key=lambda kv: kv[1], reverse=True)
    n_head = max(1, int(len(ranked) * head_fraction))
    head = {item for item, _ in ranked[:n_head]}
    tail = {item for item, _ in ranked[n_head:]}
    return head, tail


def catalog_coverage(recommended_items: Iterable[int], n_catalog: int) -> float:
    """Fraction of the catalog that appears in any recommendation list."""
    unique = set(recommended_items)
    return len(unique) / n_catalog if n_catalog else 0.0
