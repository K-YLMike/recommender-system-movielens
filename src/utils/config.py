"""Configuration loading and path resolution.

A single YAML file (``configs/config.yaml``) drives the whole pipeline. The most
important switches:

* ``dataset``            -- ``ml-1m`` (fast dev) or ``ml-25m`` (final numbers).
* ``seeds``              -- list of RNG seeds for multi-seed ablations.
* ``negative_sampling``  -- which negative-sampling variants to ablate.
* ``paths.data_root``    -- root for everything large and regenerable (raw
  downloads, checkpoints, model cache, outputs). Defaults to the **project
  directory itself** so models and datasets live alongside the code under
  ``/home/<user>/recommender_system_movielens``; override with ``RSM_DATA_ROOT``
  (e.g. point at scratch) if the home-directory quota is tight.

Keeping every path in one resolved object means no stage ever hard-codes a
location, which is what makes the code portable between the dev box and the
cluster.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Paths:
    """Resolved filesystem locations for one dataset variant.

    ``data_root`` holds everything large and regenerable (raw downloads,
    processed splits, embeddings, checkpoints, model cache, results) and defaults
    to the project directory itself. Each stage gets its own subdirectory so
    completion markers never collide.
    """

    data_root: Path
    dataset: str

    @property
    def raw_dir(self) -> Path:
        """Where the raw MovieLens download is extracted (shared across variants)."""
        return self.data_root / "raw"

    @property
    def variant_dir(self) -> Path:
        """Per-dataset working directory (ml-1m vs ml-25m kept separate)."""
        return self.data_root / self.dataset

    def stage_dir(self, stage: str) -> Path:
        """Working directory for a named stage under the current variant."""
        return self.variant_dir / stage

    @property
    def results_dir(self) -> Path:
        return self.variant_dir / "results"


@dataclass
class Config:
    """Top-level configuration object handed to every stage."""

    raw: dict[str, Any]
    paths: Paths

    # --- convenience accessors -------------------------------------------- #
    @property
    def dataset(self) -> str:
        return self.raw["dataset"]

    @property
    def seeds(self) -> list[int]:
        return list(self.raw["seeds"])

    @property
    def top_ks(self) -> list[int]:
        return list(self.raw["eval"]["top_ks"])

    @property
    def positive_rating_threshold(self) -> float:
        return float(self.raw["data"]["positive_rating_threshold"])

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested lookup: ``cfg.get('twotower', 'dim', default=64)``."""
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def load_config(config_path: str | Path = "configs/config.yaml") -> Config:
    """Load and resolve the pipeline configuration.

    The data root is resolved with this precedence:
    1. ``RSM_DATA_ROOT`` environment variable (set in the sbatch header), else
    2. ``paths.data_root`` from the YAML, else
    3. the project directory itself (the repo root containing ``configs/``), so
       data and models sit alongside the code by default.

    Args:
        config_path: Path to the YAML config.

    Returns:
        A fully resolved :class:`Config`.
    """
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    env_root = os.environ.get("RSM_DATA_ROOT")
    yaml_root = raw.get("paths", {}).get("data_root")
    # Default: keep everything (raw data, checkpoints, model cache, outputs) under
    # the project directory itself -- i.e. the repo root that contains configs/.
    # This resolves to /home/<user>/recommender_system_movielens on the cluster.
    # Override with RSM_DATA_ROOT (e.g. point at scratch if home quota is tight).
    repo_root = Path(config_path).resolve().parent.parent
    data_root = Path(env_root or yaml_root or repo_root).expanduser()

    paths = Paths(data_root=data_root, dataset=raw["dataset"])
    return Config(raw=raw, paths=paths)
