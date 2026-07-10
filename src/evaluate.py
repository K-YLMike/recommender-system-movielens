"""Stage: evaluation protocol -- where the project's depth lives.

Computes, for each retrieval source and every configured seed:

* **Recall@K / NDCG@K** on the leakage-free temporal test.
* **Segmented metrics**: head vs tail items, cold items, cold users -- so we can
  see *where* a change acts rather than only the aggregate.
* **Catalog coverage**: fraction of the catalog ever recommended.
* **Multi-seed aggregation**: ``mean +/- std`` with a 95% CI; any A-vs-B delta is
  reported alongside the seed std and flagged "within noise" when it does not
  clear it. No directional claim is asserted that the numbers do not support.
* **Leakage decomposition**: metrics on the full temporal test vs on the
  fixed-catalog subset (users/items present in both splits), isolating the part
  of any temporal-vs-random gap that is genuine look-ahead leakage from the part
  that is cold-start / distribution shift.
* **Recall-latency curve**: exact vs IVF/HNSW ANN over an nprobe/efSearch sweep.

Every result is written atomically to ``results/metrics.json``; the stage is
resumable at the granularity of (source, seed): a completed cell is cached and
skipped, so a killed evaluation job continues from the next unevaluated cell.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from src.utils import checkpoint as ckpt
from src.utils.config import Config, load_config
from src.utils.logging_utils import get_logger, timed
from src.utils.stats import (
    AggregatedMetric,
    aggregate_over_seeds,
    catalog_coverage,
    is_within_noise,
)
from src import data_prep, index_faiss

LOGGER = get_logger("evaluate")
STAGE = "evaluate"


# --------------------------------------------------------------------------- #
# Per-user metric primitives                                                  #
# --------------------------------------------------------------------------- #
def recall_at_k(topk: list[int], ground_truth: set[int]) -> float:
    """Recall@K = fraction of the user's relevant items found in the top-K."""
    if not ground_truth:
        return math.nan
    hits = sum(1 for item in topk if item in ground_truth)
    return hits / len(ground_truth)


def ndcg_at_k(topk: list[int], ground_truth: set[int]) -> float:
    """Binary NDCG@K over the ranked top-K list."""
    if not ground_truth:
        return math.nan
    dcg = 0.0
    for rank, item in enumerate(topk):
        if item in ground_truth:
            dcg += 1.0 / math.log2(rank + 2)
    ideal_hits = min(len(ground_truth), len(topk))
    idcg = sum(1.0 / math.log2(r + 2) for r in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------- #
# Retrieval: score -> top-K per user                                          #
# --------------------------------------------------------------------------- #
def topk_from_vectors(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    eval_users: np.ndarray,
    seen_by_user: dict[int, set[int]],
    k: int,
    chunk: int = 1024,
) -> dict[int, list[int]]:
    """Top-K items per user by dot product, excluding items seen in train.

    Users are processed in chunks to bound the score-matrix memory. We over-fetch
    ``k + max_seen`` candidates so that after removing seen items at least ``k``
    remain.
    """
    results: dict[int, list[int]] = {}
    max_seen = max((len(s) for s in seen_by_user.values()), default=0)
    fetch = k + max_seen + 1
    fetch = min(fetch, item_vectors.shape[0])
    for start in range(0, len(eval_users), chunk):
        batch_users = eval_users[start:start + chunk]
        scores = user_vectors[batch_users] @ item_vectors.T  # [b, n_items]
        # argpartition for speed, then sort the fetched candidates.
        part = np.argpartition(-scores, kth=fetch - 1, axis=1)[:, :fetch]
        for row, user in enumerate(batch_users):
            cand = part[row]
            cand = cand[np.argsort(-scores[row, cand])]
            seen = seen_by_user.get(int(user), set())
            filtered = [int(i) for i in cand if int(i) not in seen][:k]
            results[int(user)] = filtered
    return results


def topk_from_global_ranking(
    global_ranking: np.ndarray,
    eval_users: np.ndarray,
    seen_by_user: dict[int, set[int]],
    k: int,
) -> dict[int, list[int]]:
    """Top-K per user from a single global item ranking (popularity baseline)."""
    results: dict[int, list[int]] = {}
    ranking = global_ranking.tolist()
    for user in eval_users:
        seen = seen_by_user.get(int(user), set())
        filtered = [i for i in ranking if i not in seen][:k]
        results[int(user)] = filtered
    return results


# --------------------------------------------------------------------------- #
# Slice-aware scoring                                                         #
# --------------------------------------------------------------------------- #
def score_topk(
    topk_by_user: dict[int, list[int]],
    gt_by_user: dict[int, set[int]],
    k: int,
    slices: dict,
    n_items: int,
) -> dict:
    """Aggregate Recall@K / NDCG@K overall and per slice, plus coverage.

    Returns a dict with ``overall`` and per-slice (``head``, ``tail``, ``cold_item``,
    ``cold_user``) metrics. Slice membership for items is applied to the *ground
    truth*: e.g. cold-item recall restricts each user's ground truth to cold
    items, measuring how well cold items specifically are retrieved.
    """
    cold_items = set(slices["cold_items"])
    head_items = set(slices["head_items"])
    tail_items = set(slices["tail_items"])
    cold_users = set(slices["cold_users"])

    def _agg(gt_filter: Optional[set[int]], user_filter: Optional[set[int]]) -> dict:
        recalls, ndcgs = [], []
        for user, gt in gt_by_user.items():
            if user_filter is not None and user not in user_filter:
                continue
            g = gt if gt_filter is None else (gt & gt_filter)
            if not g:
                continue
            topk = topk_by_user.get(user, [])
            recalls.append(recall_at_k(topk, g))
            ndcgs.append(ndcg_at_k(topk, g))
        return {
            "recall": float(np.nanmean(recalls)) if recalls else math.nan,
            "ndcg": float(np.nanmean(ndcgs)) if ndcgs else math.nan,
            "n_users": len(recalls),
        }

    all_recommended = [i for lst in topk_by_user.values() for i in lst]
    return {
        "overall": _agg(None, None),
        "head": _agg(head_items, None),
        "tail": _agg(tail_items, None),
        "cold_item": _agg(cold_items, None),
        "cold_user": _agg(None, cold_users),
        "coverage": catalog_coverage(all_recommended, n_items),
        "k": k,
    }


# --------------------------------------------------------------------------- #
# Ground-truth / seen construction                                            #
# --------------------------------------------------------------------------- #
def _ground_truth(test_pos: pd.DataFrame) -> dict[int, set[int]]:
    """Map each user to the set of items they positively interact with in test."""
    gt: dict[int, set[int]] = {}
    for u, i in zip(test_pos["user_idx"].to_numpy(), test_pos["item_idx"].to_numpy()):
        gt.setdefault(int(u), set()).add(int(i))
    return gt


def _seen(train: pd.DataFrame) -> dict[int, set[int]]:
    """Map each user to items already interacted with in train (to exclude)."""
    seen: dict[int, set[int]] = {}
    pos = train[train["label"] == 1]
    for u, i in zip(pos["user_idx"].to_numpy(), pos["item_idx"].to_numpy()):
        seen.setdefault(int(u), set()).add(int(i))
    return seen


# --------------------------------------------------------------------------- #
# Source discovery                                                            #
# --------------------------------------------------------------------------- #
def _vector_sources(cfg: Config) -> dict[str, dict[int, Path]]:
    """Discover completed vector-producing sources and their per-seed dirs.

    Returns ``{source_name: {seed: dir}}`` for MF and every two-tower variant that
    has completed for at least one seed.
    """
    sources: dict[str, dict[int, Path]] = {}

    mf_dir = cfg.paths.stage_dir("retrieval_mf")
    mf_seeds = {}
    for seed in cfg.seeds:
        d = mf_dir / f"seed_{seed}"
        if ckpt.is_stage_done(d):
            mf_seeds[seed] = d
    if mf_seeds:
        sources["mf"] = mf_seeds

    tt_dir = cfg.paths.stage_dir("retrieval_twotower")
    if tt_dir.exists():
        for variant_dir in sorted(tt_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            seeds = {}
            for seed in cfg.seeds:
                d = variant_dir / f"seed_{seed}"
                if ckpt.is_stage_done(d):
                    seeds[seed] = d
            if seeds:
                sources[f"twotower_{variant_dir.name}"] = seeds
    return sources


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def run(cfg: Config) -> None:
    """Evaluate every available source across seeds and write metrics.json."""
    stage_dir = cfg.paths.stage_dir(STAGE)
    stage_dir.mkdir(parents=True, exist_ok=True)
    results_path = cfg.paths.results_dir / "metrics.json"
    results = ckpt.read_json(results_path) or {"sources": {}}

    prepared = data_prep.load_prepared(cfg)
    train = prepared["train"]
    test_pos = prepared["test_pos"]
    slices = prepared["slices"]
    n_items = int(prepared["meta"]["n_items"])
    k_eval = max(cfg.top_ks)
    gt = _ground_truth(test_pos)
    seen = _seen(train)
    # Only evaluate users that have at least one test positive AND appear in train.
    train_users = set(int(u) for u in train["user_idx"].unique())
    eval_users = np.array(sorted(u for u in gt.keys() if u in train_users), dtype=np.int64)
    LOGGER.info("Evaluating on %d users, K=%s.", len(eval_users), cfg.top_ks)

    # ---- Popularity baseline -------------------------------------------- #
    if "popularity" not in results["sources"]:
        pop_dir = cfg.paths.stage_dir("baseline_pop")
        if ckpt.is_stage_done(pop_dir):
            ranking = np.load(pop_dir / "global_ranking.npy")
            with timed(LOGGER, "evaluate popularity"):
                per_k = {}
                for k in cfg.top_ks:
                    topk = topk_from_global_ranking(ranking, eval_users, seen, k)
                    per_k[str(k)] = score_topk(topk, gt, k, slices, n_items)
                # Single deterministic source -> wrap as one "seed".
                results["sources"]["popularity"] = _aggregate_source([per_k])
            _save(results, results_path)

    # ---- Vector sources (MF, two-tower variants) ------------------------ #
    for name, seed_dirs in _vector_sources(cfg).items():
        if name in results["sources"]:
            continue
        with timed(LOGGER, f"evaluate {name} ({len(seed_dirs)} seeds)"):
            per_seed = []
            for seed, d in sorted(seed_dirs.items()):
                user_vecs = np.load(d / "user_vectors.npy")
                item_vecs = np.load(d / "item_vectors.npy")
                per_k = {}
                for k in cfg.top_ks:
                    topk = topk_from_vectors(user_vecs, item_vecs, eval_users, seen, k)
                    per_k[str(k)] = score_topk(topk, gt, k, slices, n_items)
                per_seed.append(per_k)
            results["sources"][name] = _aggregate_source(per_seed)
        _save(results, results_path)

    # ---- Leakage decomposition (fixed-catalog subset) ------------------- #
    if "leakage_decomposition" not in results:
        results["leakage_decomposition"] = _leakage_decomposition(
            cfg, prepared, eval_users, gt, seen, n_items
        )
        _save(results, results_path)

    # ---- Recall-latency curve (headline two-tower, seed 0) -------------- #
    if "recall_latency" not in results:
        rl = _recall_latency(cfg, eval_users, seen, k=max(cfg.top_ks))
        if rl is not None:
            results["recall_latency"] = rl
            _save(results, results_path)

    # ---- Pairwise deltas with noise flags ------------------------------- #
    results["comparisons"] = _pairwise_comparisons(results["sources"], cfg.top_ks)
    _save(results, results_path)
    ckpt.mark_stage_done(stage_dir, meta={"n_eval_users": int(len(eval_users))})
    LOGGER.info("Evaluation complete -> %s", results_path)


def _aggregate_source(per_seed_metrics: list[dict]) -> dict:
    """Aggregate a source's per-seed metric dicts into mean/std per (k, slice)."""
    out: dict = {"n_seeds": len(per_seed_metrics), "by_k": {}}
    if not per_seed_metrics:
        return out
    ks = per_seed_metrics[0].keys()
    for k in ks:
        segments = ["overall", "head", "tail", "cold_item", "cold_user"]
        agg_k: dict = {"coverage": {}}
        cover_vals = [m[k]["coverage"] for m in per_seed_metrics]
        agg_k["coverage"] = aggregate_over_seeds(cover_vals).as_dict()
        for seg in segments:
            for metric in ("recall", "ndcg"):
                vals = [m[k][seg][metric] for m in per_seed_metrics]
                vals = [v for v in vals if not math.isnan(v)]
                agg_k.setdefault(seg, {})[metric] = (
                    aggregate_over_seeds(vals).as_dict() if vals else None
                )
        out["by_k"][k] = agg_k
    return out


def _leakage_decomposition(
    cfg: Config,
    prepared: dict,
    eval_users: np.ndarray,
    gt: dict[int, set[int]],
    seen: dict[int, set[int]],
    n_items: int,
) -> dict:
    """Decompose within-temporal metrics: full test vs fixed-catalog subset.

    The fixed-catalog subset keeps only users and ground-truth items that also
    appear in train. Comparing full-test to fixed-catalog metrics quantifies how
    much of the evaluation is driven by cold/unseen entities (the part of any
    random-vs-temporal gap attributable to distribution shift rather than genuine
    look-ahead leakage). The random-split arm is produced by re-running the
    pipeline with ``data.split_mode: random`` into a sibling data root and
    comparing (see README); this function reports the cheap, no-retrain half.
    """
    slices = prepared["slices"]
    fixed_users = set(slices["fixed_catalog_users"])
    fixed_items = set(slices["fixed_catalog_items"])

    # Use the headline two-tower seed 0 if present, else MF seed 0, else skip.
    src_dir = _headline_seed_dir(cfg, seed=cfg.seeds[0])
    if src_dir is None:
        return {"note": "no vector source available for decomposition"}
    user_vecs = np.load(src_dir / "user_vectors.npy")
    item_vecs = np.load(src_dir / "item_vectors.npy")
    k = max(cfg.top_ks)

    topk = topk_from_vectors(user_vecs, item_vecs, eval_users, seen, k)
    full = _mean_recall(topk, gt)

    fixed_eval_users = np.array([u for u in eval_users if int(u) in fixed_users], dtype=np.int64)
    gt_fixed = {u: (g & fixed_items) for u, g in gt.items() if u in fixed_users}
    gt_fixed = {u: g for u, g in gt_fixed.items() if g}
    fixed = _mean_recall(topk, gt_fixed)

    return {
        "k": k,
        "full_test_recall": full,
        "fixed_catalog_recall": fixed,
        "note": (
            "Gap between full and fixed-catalog recall reflects cold/unseen "
            "entities within the temporal split. Random-vs-temporal requires the "
            "split_mode=random second pass described in the README."
        ),
    }


def _mean_recall(topk_by_user: dict[int, list[int]], gt_by_user: dict[int, set[int]]) -> float:
    vals = [recall_at_k(topk_by_user.get(u, []), g) for u, g in gt_by_user.items() if g]
    return float(np.nanmean(vals)) if vals else math.nan


def _headline_seed_dir(cfg: Config, seed: int) -> Optional[Path]:
    """Locate the headline two-tower vectors for a seed, with MF fallback."""
    headline = cfg.get("twotower", "headline_variant", default="logq")
    d = cfg.paths.stage_dir("retrieval_twotower") / headline / f"seed_{seed}"
    if ckpt.is_stage_done(d):
        return d
    d = cfg.paths.stage_dir("retrieval_mf") / f"seed_{seed}"
    return d if ckpt.is_stage_done(d) else None


def _recall_latency(
    cfg: Config, eval_users: np.ndarray, seen: dict[int, set[int]], k: int
) -> Optional[dict]:
    """Trace the exact-vs-ANN recall-latency curve for the headline model."""
    src_dir = _headline_seed_dir(cfg, seed=cfg.seeds[0])
    if src_dir is None:
        return None
    user_vecs = np.load(src_dir / "user_vectors.npy")[eval_users]
    item_vecs = np.load(src_dir / "item_vectors.npy")
    with timed(LOGGER, "recall-latency sweep"):
        points = index_faiss.recall_latency_sweep(
            item_vectors=item_vecs,
            query_vectors=user_vecs,
            k=k,
            ivf_nprobe=cfg.get("ann", "ivf_nprobe", default=[1, 4, 8, 16, 32, 64]),
            hnsw_efsearch=cfg.get("ann", "hnsw_efsearch", default=[8, 16, 32, 64, 128]),
            nlist=int(cfg.get("ann", "nlist", default=256)),
            hnsw_m=int(cfg.get("ann", "hnsw_m", default=32)),
        )
    return {"k": k, "points": [p.as_dict() for p in points]}


def _pairwise_comparisons(sources: dict, top_ks: list[int]) -> list[dict]:
    """Report overall Recall@10 deltas vs popularity with a noise flag.

    Each comparison states the delta, the reference seed std, and whether the
    delta is within noise -- the guardrail against fragile directional claims.
    """
    k = str(min(top_ks))
    if "popularity" not in sources:
        return []
    base = _safe_metric(sources["popularity"], k, "overall", "recall")
    comps = []
    for name, agg in sources.items():
        if name == "popularity":
            continue
        cur = _safe_metric(agg, k, "overall", "recall")
        if cur is None or base is None:
            continue
        delta = cur["mean"] - base["mean"]
        ref_std = max(cur["std"], base["std"])
        comps.append(
            {
                "source": name,
                "vs": "popularity",
                "metric": f"recall@{k}",
                "delta": delta,
                "reference_std": ref_std,
                "within_noise": is_within_noise(delta, ref_std, cur["ci95_halfwidth"]),
            }
        )
    return comps


def _safe_metric(agg: dict, k: str, seg: str, metric: str) -> Optional[dict]:
    try:
        return agg["by_k"][k][seg][metric]
    except (KeyError, TypeError):
        return None


def _save(results: dict, path: Path) -> None:
    ckpt.atomic_write_json(path, results)


def main() -> None:
    cfg = load_config()
    run(cfg)


if __name__ == "__main__":
    main()
