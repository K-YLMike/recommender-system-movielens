"""Stage: data preparation.

Loads raw MovieLens (``ml-1m`` ``.dat`` or ``ml-25m`` ``.csv``), converts ratings
to implicit feedback, reindexes ids to contiguous integers, performs a
leakage-free **global temporal split**, and precomputes every artifact the later
stages and the evaluation protocol need:

* ``train.parquet`` / ``test.parquet`` -- interactions with ``label`` and rating.
* ``movies.parquet``                   -- title + genre multi-hot for the content tower.
* ``user_history.npz``                 -- per-user chronological train history (user tower).
* ``slices.json``                      -- cold-item / cold-user / head / tail / fixed-catalog sets.
* ``stats.json``                       -- dataset statistics for the README.

Resumability: the stage is *idempotent*. If its ``_DONE.json`` marker exists it
returns immediately; otherwise it recomputes from scratch (data prep is minutes,
not worth mid-stage checkpointing) and writes every output atomically before
finally writing the marker. A crash therefore leaves no half-written outputs and
simply reruns cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import checkpoint as ckpt
from src.utils.config import Config, load_config
from src.utils.logging_utils import get_logger, timed
from src.utils.stats import dataset_statistics, head_tail_split

LOGGER = get_logger("data_prep")

STAGE = "data_prep"


# --------------------------------------------------------------------------- #
# Raw loading (handles both MovieLens formats)                                #
# --------------------------------------------------------------------------- #
def _load_raw(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw ratings and movies for the configured dataset variant.

    Returns:
        ``(ratings, movies)`` with normalised column names
        ``user_id, item_id, rating, timestamp`` and ``item_id, title, genres``.
    """
    raw_dir = cfg.paths.raw_dir
    if cfg.dataset == "ml-1m":
        base = raw_dir / "ml-1m"
        ratings = pd.read_csv(
            base / "ratings.dat",
            sep="::",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"],
            encoding="latin-1",
        )
        movies = pd.read_csv(
            base / "movies.dat",
            sep="::",
            engine="python",
            names=["item_id", "title", "genres"],
            encoding="latin-1",
        )
    elif cfg.dataset == "ml-25m":
        base = raw_dir / "ml-25m"
        ratings = pd.read_csv(base / "ratings.csv")
        ratings = ratings.rename(
            columns={"userId": "user_id", "movieId": "item_id"}
        )
        movies = pd.read_csv(base / "movies.csv").rename(
            columns={"movieId": "item_id"}
        )
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset!r}")
    return ratings, movies


# --------------------------------------------------------------------------- #
# Reindexing and implicit labels                                              #
# --------------------------------------------------------------------------- #
def _reindex(
    ratings: pd.DataFrame, movies: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Map raw user/item ids to contiguous 0..N-1 indices.

    Items are indexed over the *movie catalog* (from ``movies``), not only rated
    movies, so the content tower can represent items that have no interactions --
    exactly the cold-start case the content tower is meant to help.
    """
    user_ids = np.sort(ratings["user_id"].unique())
    item_ids = np.sort(movies["item_id"].unique())
    user_map = {int(u): i for i, u in enumerate(user_ids)}
    item_map = {int(m): i for i, m in enumerate(item_ids)}

    ratings = ratings[ratings["item_id"].isin(item_map)].copy()
    ratings["user_idx"] = ratings["user_id"].map(user_map).astype(np.int64)
    ratings["item_idx"] = ratings["item_id"].map(item_map).astype(np.int64)

    movies = movies.copy()
    movies["item_idx"] = movies["item_id"].map(item_map).astype(np.int64)
    movies = movies.sort_values("item_idx").reset_index(drop=True)
    return ratings, movies, user_map, item_map


def _apply_implicit_label(ratings: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add an implicit ``label`` column: 1 if rating >= threshold else 0.

    Retrieval uses positives only; ranking keeps the graded ``rating``.
    """
    ratings = ratings.copy()
    ratings["label"] = (ratings["rating"] >= threshold).astype(np.int8)
    return ratings


# --------------------------------------------------------------------------- #
# Temporal split + slices                                                     #
# --------------------------------------------------------------------------- #
def _temporal_split(
    ratings: pd.DataFrame, test_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Global temporal split: the latest ``test_frac`` of interactions is test.

    A single global timestamp threshold (rather than per-user leave-last-out)
    prevents look-ahead leakage: nothing in train post-dates anything used to
    train against future test interactions.
    """
    ratings = ratings.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    cut = int(len(ratings) * (1.0 - test_frac))
    threshold_ts = ratings.iloc[cut]["timestamp"]
    train = ratings[ratings["timestamp"] < threshold_ts].reset_index(drop=True)
    test = ratings[ratings["timestamp"] >= threshold_ts].reset_index(drop=True)
    return train, test


def _build_genre_multihot(movies: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Turn the pipe-separated ``genres`` field into a multi-hot matrix.

    Returns:
        ``(multihot[n_items, n_genres], genre_vocab)`` aligned to ``item_idx``.
    """
    genre_lists = movies["genres"].fillna("").apply(
        lambda g: [x for x in g.split("|") if x and x != "(no genres listed)"]
    )
    vocab = sorted({g for lst in genre_lists for g in lst})
    gidx = {g: i for i, g in enumerate(vocab)}
    multihot = np.zeros((len(movies), len(vocab)), dtype=np.float32)
    for row, lst in enumerate(genre_lists):
        for g in lst:
            multihot[row, gidx[g]] = 1.0
    return multihot, vocab


def _build_user_history(train: pd.DataFrame, n_users: int) -> dict[int, list[int]]:
    """Per-user chronological list of positively-interacted train items.

    Used by the user tower (mean-pool of recent items) and by ranking features.
    Only positives are kept, matching the retrieval signal.
    """
    pos = train[train["label"] == 1].sort_values("timestamp", kind="mergesort")
    history: dict[int, list[int]] = {u: [] for u in range(n_users)}
    for user_idx, item_idx in zip(pos["user_idx"].to_numpy(), pos["item_idx"].to_numpy()):
        history[int(user_idx)].append(int(item_idx))
    return history


def _compute_slices(
    train: pd.DataFrame, test: pd.DataFrame, n_items: int, cold_threshold: int
) -> dict:
    """Compute cold / head / tail / fixed-catalog slices for evaluation.

    * ``cold_items`` / ``cold_users``: train interaction count <= ``cold_threshold``.
    * ``head`` / ``tail``: top-20% vs rest of items by train popularity.
    * ``fixed_catalog_*``: users/items appearing in *both* train and test -- the
      subset on which we recompute metrics to isolate pure temporal leakage from
      cold-start / distribution shift in the leakage decomposition.
    """
    item_train_counts = train.groupby("item_idx").size()
    user_train_counts = train.groupby("user_idx").size()

    item_pop = {int(i): int(c) for i, c in item_train_counts.items()}
    # Items never seen in train are maximally cold (count 0).
    cold_items = {i for i in range(n_items) if item_pop.get(i, 0) <= cold_threshold}
    cold_users = {
        int(u) for u, c in user_train_counts.items() if int(c) <= cold_threshold
    }

    head, tail = head_tail_split(item_pop, head_fraction=0.2)

    train_users = set(train["user_idx"].unique().tolist())
    train_items = set(train["item_idx"].unique().tolist())
    test_users = set(test["user_idx"].unique().tolist())
    test_items = set(test["item_idx"].unique().tolist())

    return {
        "cold_threshold": cold_threshold,
        "cold_items": sorted(int(x) for x in cold_items),
        "cold_users": sorted(int(x) for x in cold_users),
        "head_items": sorted(int(x) for x in head),
        "tail_items": sorted(int(x) for x in tail),
        "fixed_catalog_users": sorted(int(x) for x in (train_users & test_users)),
        "fixed_catalog_items": sorted(int(x) for x in (train_items & test_items)),
        "item_train_popularity": {str(k): v for k, v in item_pop.items()},
    }


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def run(cfg: Config) -> None:
    """Execute the data-prep stage (idempotent)."""
    stage_dir = cfg.paths.stage_dir(STAGE)
    if ckpt.is_stage_done(stage_dir):
        LOGGER.info("Stage '%s' already complete at %s -- skipping.", STAGE, stage_dir)
        return
    stage_dir.mkdir(parents=True, exist_ok=True)

    with timed(LOGGER, "load raw"):
        ratings, movies = _load_raw(cfg)
        ratings, movies, user_map, item_map = _reindex(ratings, movies)
        ratings = _apply_implicit_label(ratings, cfg.positive_rating_threshold)

    n_users = len(user_map)
    n_items = len(item_map)
    LOGGER.info("Users=%d Items=%d Interactions=%d", n_users, n_items, len(ratings))

    with timed(LOGGER, "temporal split"):
        test_frac = float(cfg.get("data", "test_fraction", default=0.1))
        train, test = _temporal_split(ratings, test_frac)
        # Retrieval evaluates against positives only; keep the label for ranking.
        test_pos = test[test["label"] == 1].reset_index(drop=True)
        LOGGER.info("Train=%d Test=%d (test positives=%d)", len(train), len(test), len(test_pos))

    with timed(LOGGER, "content + history + slices"):
        genre_multihot, genre_vocab = _build_genre_multihot(movies)
        history = _build_user_history(train, n_users)
        cold_threshold = int(cfg.get("data", "cold_threshold", default=5))
        slices = _compute_slices(train, test_pos, n_items, cold_threshold)

    with timed(LOGGER, "dataset statistics"):
        stats = dataset_statistics(train, time_col="timestamp")
        stats["n_test_positive_interactions"] = int(len(test_pos))
        stats["n_cold_items"] = len(slices["cold_items"])
        stats["n_cold_users"] = len(slices["cold_users"])
        stats["dataset"] = cfg.dataset

    # ---- write every artifact atomically, marker last -------------------- #
    with timed(LOGGER, "write artifacts"):
        train.to_parquet(stage_dir / "train.parquet", index=False)
        test.to_parquet(stage_dir / "test.parquet", index=False)
        test_pos.to_parquet(stage_dir / "test_pos.parquet", index=False)
        movies[["item_idx", "title", "genres"]].to_parquet(
            stage_dir / "movies.parquet", index=False
        )
        ckpt.atomic_write_npy(stage_dir / "genre_multihot.npy", genre_multihot)

        # user_history saved as a ragged structure via npz (offsets + flat).
        flat, offsets = _ragged_to_flat(history, n_users)
        np.savez(
            stage_dir / "user_history.npz.tmp", flat=flat, offsets=offsets
        )  # np.savez appends .npz
        # np.savez wrote user_history.npz.tmp.npz; rename atomically.
        (stage_dir / "user_history.npz.tmp.npz").replace(stage_dir / "user_history.npz")

        ckpt.atomic_write_json(
            stage_dir / "meta.json",
            {
                "n_users": n_users,
                "n_items": n_items,
                "genre_vocab": genre_vocab,
                "positive_rating_threshold": cfg.positive_rating_threshold,
                "test_fraction": test_frac,
            },
        )
        ckpt.atomic_write_json(stage_dir / "slices.json", slices)
        ckpt.atomic_write_json(stage_dir / "stats.json", stats)
        # Mirror stats into results for the README.
        cfg.paths.results_dir.mkdir(parents=True, exist_ok=True)
        ckpt.atomic_write_json(cfg.paths.results_dir / "dataset_stats.json", stats)

    ckpt.mark_stage_done(stage_dir, meta={"n_users": n_users, "n_items": n_items})
    LOGGER.info("Data prep complete -> %s", stage_dir)


def _ragged_to_flat(history: dict[int, list[int]], n_users: int) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a per-user list-of-items into (flat, offsets) CSR-style arrays.

    ``offsets`` has length ``n_users + 1``; user ``u``'s history is
    ``flat[offsets[u]:offsets[u + 1]]``. Compact and fast to load.
    """
    lengths = np.array([len(history.get(u, [])) for u in range(n_users)], dtype=np.int64)
    offsets = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    flat = np.empty(int(offsets[-1]), dtype=np.int64)
    for u in range(n_users):
        seq = history.get(u, [])
        flat[offsets[u]:offsets[u + 1]] = seq
    return flat, offsets


def load_prepared(cfg: Config) -> dict:
    """Convenience loader used by downstream stages.

    Returns a dict of the prepared artifacts (dataframes + arrays + metadata).
    """
    stage_dir = cfg.paths.stage_dir(STAGE)
    if not ckpt.is_stage_done(stage_dir):
        raise RuntimeError("data_prep has not completed; run it first.")
    meta = ckpt.read_json(stage_dir / "meta.json")
    slices = ckpt.read_json(stage_dir / "slices.json")
    hist = np.load(stage_dir / "user_history.npz")
    return {
        "train": pd.read_parquet(stage_dir / "train.parquet"),
        "test": pd.read_parquet(stage_dir / "test.parquet"),
        "test_pos": pd.read_parquet(stage_dir / "test_pos.parquet"),
        "movies": pd.read_parquet(stage_dir / "movies.parquet"),
        "genre_multihot": np.load(stage_dir / "genre_multihot.npy"),
        "history_flat": hist["flat"],
        "history_offsets": hist["offsets"],
        "meta": meta,
        "slices": slices,
    }


def main() -> None:
    """CLI entry: ``python -m src.data_prep``."""
    cfg = load_config()
    run(cfg)


if __name__ == "__main__":
    main()
