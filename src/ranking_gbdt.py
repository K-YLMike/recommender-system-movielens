"""Stage: GBDT reranker (LightGBM LambdaRank).

The second stage of the two-stage pipeline. It takes a modest candidate set per
user from stage-1 retrieval and reorders it with a gradient-boosted ranker that
sees richer per-(user, item) features than a single dot product:

* stage-1 retrieval score (dot product of the two-tower vectors),
* item popularity and mean rating (from train only, no leakage),
* user activity level,
* content similarity between the item and the user's history profile,
* genre affinity between the item and the user's genre profile.

Labels are the graded train ratings for positives and 0 for sampled negatives,
grouped per user (each user is one LambdaRank query). Features use train
information exclusively, so the reranker introduces no look-ahead leakage.

The "honest negative result" the project is prepared to report lives here: on
clean MovieLens, a learned reranker does not always beat the retrieval ordering
by more than seed noise. Whatever the outcome, the delta is reported against the
seed std and flagged if it is within noise.

Resumability: LightGBM training is minutes, so the stage is made idempotent via a
``_DONE`` marker (skip if present) and the model is saved atomically; a killed
job simply retrains cleanly.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import checkpoint as ckpt
from src.utils.config import Config, load_config
from src.utils.logging_utils import get_logger, timed
from src.utils.seed import set_global_seed
from src.utils.stats import aggregate_over_seeds, is_within_noise
from src import data_prep, content_features, evaluate as ev

LOGGER = get_logger("ranking_gbdt")
STAGE = "ranking_gbdt"

FEATURE_NAMES = [
    "s1_score",
    "item_pop",
    "item_mean_rating",
    "user_activity",
    "content_sim",
    "genre_affinity",
]


def _headline_vectors(cfg: Config, seed: int):
    """Load the headline two-tower user/item vectors for a seed (MF fallback)."""
    d = ev._headline_seed_dir(cfg, seed)
    if d is None:
        raise RuntimeError("No retrieval vectors available for ranking features.")
    return np.load(d / "user_vectors.npy"), np.load(d / "item_vectors.npy"), d


def _user_profiles(
    history_flat, history_offsets, content_emb, genre_multihot, n_users
):
    """Precompute per-user content and genre profiles from train history.

    Returns ``(user_content[n_users, c], user_genre[n_users, g])`` mean profiles.
    Cold users (no history) get zero profiles.
    """
    c_dim = content_emb.shape[1]
    g_dim = genre_multihot.shape[1]
    user_content = np.zeros((n_users, c_dim), dtype=np.float32)
    user_genre = np.zeros((n_users, g_dim), dtype=np.float32)
    for u in range(n_users):
        start, end = int(history_offsets[u]), int(history_offsets[u + 1])
        items = history_flat[start:end]
        if len(items) == 0:
            continue
        user_content[u] = content_emb[items].mean(axis=0)
        user_genre[u] = genre_multihot[items].mean(axis=0)
    return user_content, user_genre


def _features(
    users: np.ndarray,
    items: np.ndarray,
    user_vecs,
    item_vecs,
    item_pop,
    item_mean_rating,
    user_activity,
    user_content,
    user_genre,
    content_emb,
    genre_multihot,
) -> np.ndarray:
    """Vectorised per-(user, item) feature matrix in ``FEATURE_NAMES`` order."""
    s1 = np.sum(user_vecs[users] * item_vecs[items], axis=1)
    csim = np.sum(user_content[users] * content_emb[items], axis=1)
    gaff = np.sum(user_genre[users] * genre_multihot[items], axis=1)
    return np.column_stack(
        [
            s1,
            item_pop[items],
            item_mean_rating[items],
            user_activity[users],
            csim,
            gaff,
        ]
    ).astype(np.float32)


def _build_training_set(cfg, prepared, user_vecs, item_vecs, seed):
    """Assemble grouped LambdaRank training data from train interactions.

    For each user: positives are their train positives (graded by rating);
    negatives are uniformly sampled unseen items (label 0). Users are the query
    groups. Returns ``(X, y, group_sizes)``.
    """
    rng = np.random.default_rng(seed)
    train = prepared["train"]
    pos = train[train["label"] == 1]
    n_items = int(prepared["meta"]["n_items"])
    n_users = int(prepared["meta"]["n_users"])
    content_emb = content_features.load_content_embeddings(cfg)
    genre_multihot = prepared["genre_multihot"]

    item_pop = np.bincount(pos["item_idx"].to_numpy(), minlength=n_items).astype(np.float32)
    item_pop = np.log1p(item_pop)
    mean_r = train.groupby("item_idx")["rating"].mean()
    item_mean_rating = np.full(n_items, float(train["rating"].mean()), dtype=np.float32)
    item_mean_rating[mean_r.index.to_numpy()] = mean_r.to_numpy(dtype=np.float32)
    user_activity = np.log1p(
        np.bincount(pos["user_idx"].to_numpy(), minlength=n_users).astype(np.float32)
    )
    user_content, user_genre = _user_profiles(
        prepared["history_flat"], prepared["history_offsets"], content_emb, genre_multihot, n_users
    )

    neg_per_pos = int(cfg.get("ranking_gbdt", "neg_per_pos", default=4))
    max_users = int(cfg.get("ranking_gbdt", "max_train_users", default=20000))
    seen = ev._seen(train)

    pos_by_user: dict[int, list[tuple[int, float]]] = {}
    for u, i, r in zip(pos["user_idx"].to_numpy(), pos["item_idx"].to_numpy(), pos["rating"].to_numpy()):
        pos_by_user.setdefault(int(u), []).append((int(i), float(r)))

    users_all = list(pos_by_user.keys())
    if len(users_all) > max_users:
        users_all = list(rng.choice(users_all, size=max_users, replace=False))

    rows_u, rows_i, rows_y, groups = [], [], [], []
    for u in users_all:
        entries = pos_by_user[u]
        seen_u = seen.get(u, set())
        group_len = 0
        for item, rating in entries:
            rows_u.append(u)
            rows_i.append(item)
            rows_y.append(_graded_label(rating))
            group_len += 1
            for _ in range(neg_per_pos):
                neg = int(rng.integers(0, n_items))
                if neg in seen_u:
                    continue
                rows_u.append(u)
                rows_i.append(neg)
                rows_y.append(0)
                group_len += 1
        groups.append(group_len)

    X = _features(
        np.array(rows_u), np.array(rows_i), user_vecs, item_vecs,
        item_pop, item_mean_rating, user_activity, user_content, user_genre,
        content_emb, genre_multihot,
    )
    y = np.array(rows_y, dtype=np.int32)
    aux = {
        "item_pop": item_pop,
        "item_mean_rating": item_mean_rating,
        "user_activity": user_activity,
        "user_content": user_content,
        "user_genre": user_genre,
        "content_emb": content_emb,
        "genre_multihot": genre_multihot,
    }
    return X, y, np.array(groups, dtype=np.int32), aux


def _graded_label(rating: float) -> int:
    """Map a 0.5-5.0 rating to an integer LambdaRank gain (0..4)."""
    return int(max(0, min(4, round(rating - 1))))


def run(cfg: Config, seed: int) -> None:
    """Train the reranker for a seed and evaluate reranked NDCG."""
    import lightgbm as lgb

    seed_dir = cfg.paths.stage_dir(STAGE) / f"seed_{seed}"
    if ckpt.is_stage_done(seed_dir):
        LOGGER.info("GBDT seed %d already complete -- skipping.", seed)
        return
    seed_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)

    prepared = data_prep.load_prepared(cfg)
    user_vecs, item_vecs, _ = _headline_vectors(cfg, seed)

    with timed(LOGGER, f"build training set seed {seed}"):
        X, y, groups, aux = _build_training_set(cfg, prepared, user_vecs, item_vecs, seed)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": cfg.top_ks,
        "learning_rate": float(cfg.get("ranking_gbdt", "learning_rate", default=0.05)),
        "num_leaves": int(cfg.get("ranking_gbdt", "num_leaves", default=31)),
        "min_data_in_leaf": int(cfg.get("ranking_gbdt", "min_data_in_leaf", default=50)),
        "seed": seed,
        "verbosity": -1,
    }
    num_round = int(cfg.get("ranking_gbdt", "num_boost_round", default=200))

    with timed(LOGGER, f"train lambdarank seed {seed}"):
        dtrain = lgb.Dataset(X, label=y, group=groups, feature_name=FEATURE_NAMES)
        model = lgb.train(params, dtrain, num_boost_round=num_round)

    # Save model atomically (write to temp path, then rename).
    tmp = seed_dir / "model.txt.tmp"
    model.save_model(str(tmp))
    tmp.replace(seed_dir / "model.txt")

    with timed(LOGGER, f"evaluate rerank seed {seed}"):
        metrics = _evaluate_rerank(cfg, prepared, model, user_vecs, item_vecs, aux)

    ckpt.atomic_write_json(seed_dir / "metrics.json", metrics)
    ckpt.mark_stage_done(seed_dir, meta={"seed": seed})
    LOGGER.info("GBDT seed %d complete -> %s", seed, seed_dir)


def _evaluate_rerank(cfg, prepared, model, user_vecs, item_vecs, aux) -> dict:
    """Rerank stage-1 candidates and compute NDCG before vs after reranking.

    Retrieves ``candidate_pool`` items per user from stage-1, scores them with the
    GBDT, and compares NDCG@K of the stage-1 order to the reranked order. The
    difference is the reranker's contribution (which may be within noise).
    """
    train = prepared["train"]
    test_pos = prepared["test_pos"]
    n_items = int(prepared["meta"]["n_items"])
    gt = ev._ground_truth(test_pos)
    seen = ev._seen(train)
    train_users = set(int(u) for u in train["user_idx"].unique())
    eval_users = np.array(sorted(u for u in gt if u in train_users), dtype=np.int64)

    pool = int(cfg.get("ranking_gbdt", "candidate_pool", default=200))
    k = max(cfg.top_ks)

    s1_topk = ev.topk_from_vectors(user_vecs, item_vecs, eval_users, seen, pool)
    ndcg_s1, ndcg_rr = [], []
    for user in eval_users:
        cands = s1_topk.get(int(user), [])
        if not cands:
            continue
        g = gt.get(int(user), set())
        if not g:
            continue
        items = np.array(cands, dtype=np.int64)
        users = np.full(len(items), int(user), dtype=np.int64)
        feats = _features(
            users, items, user_vecs, item_vecs,
            aux["item_pop"], aux["item_mean_rating"], aux["user_activity"],
            aux["user_content"], aux["user_genre"], aux["content_emb"], aux["genre_multihot"],
        )
        scores = model.predict(feats)
        reranked = [int(items[j]) for j in np.argsort(-scores)][:k]
        ndcg_s1.append(ev.ndcg_at_k(cands[:k], g))
        ndcg_rr.append(ev.ndcg_at_k(reranked, g))
    return {
        "k": k,
        "candidate_pool": pool,
        "ndcg_stage1": float(np.nanmean(ndcg_s1)) if ndcg_s1 else math.nan,
        "ndcg_reranked": float(np.nanmean(ndcg_rr)) if ndcg_rr else math.nan,
        "n_users": len(ndcg_s1),
    }


def aggregate_seeds(cfg: Config) -> None:
    """Aggregate per-seed rerank metrics into results/ranking.json with noise flag."""
    s1_vals, rr_vals = [], []
    for seed in cfg.seeds:
        m = ckpt.read_json(cfg.paths.stage_dir(STAGE) / f"seed_{seed}" / "metrics.json")
        if m:
            s1_vals.append(m["ndcg_stage1"])
            rr_vals.append(m["ndcg_reranked"])
    if not s1_vals:
        return
    s1_agg = aggregate_over_seeds(s1_vals)
    rr_agg = aggregate_over_seeds(rr_vals)
    delta = rr_agg.mean - s1_agg.mean
    ref_std = max(s1_agg.std, rr_agg.std)
    out = {
        "ndcg_stage1": s1_agg.as_dict(),
        "ndcg_reranked": rr_agg.as_dict(),
        "delta": delta,
        "reference_std": ref_std,
        "within_noise": is_within_noise(delta, ref_std, rr_agg.ci95_halfwidth),
    }
    cfg.paths.results_dir.mkdir(parents=True, exist_ok=True)
    ckpt.atomic_write_json(cfg.paths.results_dir / "ranking.json", out)
    LOGGER.info("Ranking aggregate: stage1=%s reranked=%s within_noise=%s",
                s1_agg.pretty(), rr_agg.pretty(), out["within_noise"])


def main() -> None:
    cfg = load_config()
    for seed in cfg.seeds:
        run(cfg, seed)
    aggregate_seeds(cfg)


if __name__ == "__main__":
    main()
