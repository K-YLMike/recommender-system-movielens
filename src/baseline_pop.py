"""Stage: popularity baseline recall.

The simplest possible recommender: rank every item by a smoothed popularity
score and return the same top-K to everyone (minus items the user already
consumed in train). Every later stage must beat this; on rating data with heavy
head skew it is a *strong* baseline, which is exactly why it is worth reporting.

Smoothing: we use a count-prior-adjusted mean rating (a Bayesian shrink toward
the global mean) so a movie with one 5-star rating does not outrank a broadly
loved one. Popularity ranking then falls out of the adjusted score.

Resumability: trivially idempotent -- the whole computation is a groupby, so we
just skip if the ``_DONE`` marker exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import checkpoint as ckpt
from src.utils.config import Config, load_config
from src.utils.logging_utils import get_logger, timed
from src import data_prep

LOGGER = get_logger("baseline_pop")
STAGE = "baseline_pop"


def _popularity_scores(train: pd.DataFrame, n_items: int, prior_strength: float) -> np.ndarray:
    """Bayesian-shrunk popularity score per item index.

    score(i) = (v_i * mean_i + m * C) / (v_i + m)
    where v_i is the rating count, mean_i the mean rating, C the global mean, and
    m the prior strength (pseudo-count). Items with no ratings get the prior C.
    """
    grouped = train.groupby("item_idx")["rating"].agg(["count", "mean"])
    global_mean = float(train["rating"].mean())
    scores = np.full(n_items, global_mean, dtype=np.float32)
    counts = grouped["count"].to_numpy(dtype=np.float64)
    means = grouped["mean"].to_numpy(dtype=np.float64)
    idx = grouped.index.to_numpy()
    shrunk = (counts * means + prior_strength * global_mean) / (counts + prior_strength)
    scores[idx] = shrunk.astype(np.float32)
    return scores


def run(cfg: Config) -> None:
    """Compute and persist the global popularity ranking."""
    stage_dir = cfg.paths.stage_dir(STAGE)
    if ckpt.is_stage_done(stage_dir):
        LOGGER.info("Stage '%s' already complete -- skipping.", STAGE)
        return
    stage_dir.mkdir(parents=True, exist_ok=True)

    prepared = data_prep.load_prepared(cfg)
    train = prepared["train"]
    n_items = int(prepared["meta"]["n_items"])
    prior = float(cfg.get("baseline_pop", "prior_strength", default=20.0))

    with timed(LOGGER, "popularity scores"):
        scores = _popularity_scores(train, n_items, prior)
        ranking = np.argsort(-scores).astype(np.int64)  # descending

    ckpt.atomic_write_npy(stage_dir / "item_scores.npy", scores)
    ckpt.atomic_write_npy(stage_dir / "global_ranking.npy", ranking)
    ckpt.mark_stage_done(stage_dir, meta={"prior_strength": prior})
    LOGGER.info("Popularity baseline complete -> %s", stage_dir)


def main() -> None:
    cfg = load_config()
    run(cfg)


if __name__ == "__main__":
    main()
