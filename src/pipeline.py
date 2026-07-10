"""End-to-end pipeline orchestrator.

Runs the stages in dependency order, each of which is independently resumable, so
the whole thing can be driven by a chain of 1-hour Slurm jobs. A single stage can
also be run in isolation (``--stage twotower``) which is how the sbatch scripts
invoke individual GPU-bound steps.

Stage order and dependencies::

    data_prep -> baseline_pop
              -> content_features -> retrieval_twotower -> index/evaluate
              -> retrieval_mf ----/                        \\-> ranking_gbdt

Because every stage checks its own ``_DONE`` marker first, re-running the full
pipeline after a partial completion only executes what is missing.
"""

from __future__ import annotations

import argparse

from src.utils.config import Config, load_config
from src.utils.logging_utils import get_logger
from src import (
    data_prep,
    baseline_pop,
    content_features,
    retrieval_mf,
    retrieval_twotower,
    ranking_gbdt,
    evaluate,
)

LOGGER = get_logger("pipeline")

# Stage name -> callable taking the config and running that stage (all seeds).
STAGES: dict[str, object] = {
    "data_prep": lambda cfg: data_prep.run(cfg),
    "baseline_pop": lambda cfg: baseline_pop.run(cfg),
    "content_features": lambda cfg: content_features.run(cfg),
    "retrieval_mf": lambda cfg: _run_mf(cfg),
    "retrieval_twotower": lambda cfg: _run_twotower(cfg),
    "ranking_gbdt": lambda cfg: _run_ranking(cfg),
    "evaluate": lambda cfg: evaluate.run(cfg),
}

# Default execution order for a full run.
ORDER = [
    "data_prep",
    "baseline_pop",
    "content_features",
    "retrieval_mf",
    "retrieval_twotower",
    "evaluate",
    "ranking_gbdt",
]


def _run_mf(cfg: Config) -> None:
    for seed in cfg.seeds:
        retrieval_mf.run(cfg, seed)


def _run_twotower(cfg: Config) -> None:
    variants = cfg.get("twotower", "variants", default=["in_batch", "hard", "logq"])
    headline = cfg.get("twotower", "headline_variant", default="logq")
    for seed in cfg.seeds:
        for variant in variants:
            retrieval_twotower.run(cfg, variant, seed, use_content=True)
        retrieval_twotower.run(cfg, headline, seed, use_content=False)


def _run_ranking(cfg: Config) -> None:
    for seed in cfg.seeds:
        ranking_gbdt.run(cfg, seed)
    ranking_gbdt.aggregate_seeds(cfg)


def main() -> None:
    """CLI: run one stage (``--stage``) or the full ordered pipeline."""
    parser = argparse.ArgumentParser(description="MovieLens two-stage recommender pipeline.")
    parser.add_argument(
        "--stage",
        choices=list(STAGES) + ["all"],
        default="all",
        help="Which stage to run (default: all, in dependency order).",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    LOGGER.info("Dataset=%s data_root=%s seeds=%s", cfg.dataset, cfg.paths.data_root, cfg.seeds)

    stages = ORDER if args.stage == "all" else [args.stage]
    for stage in stages:
        LOGGER.info("=== stage: %s ===", stage)
        STAGES[stage](cfg)
    LOGGER.info("Pipeline done (stage=%s).", args.stage)


if __name__ == "__main__":
    main()
