#!/usr/bin/env python
"""Runner for the 'retrieval_twotower' stage -- point chain.sh / run_stage.sbatch here.

Train the headline two-tower over all negative-sampling variants x seeds (checkpointed).

This script runs the whole 'retrieval_twotower' stage (all seeds / variants). Every unit
checkpoints and skips when already complete, so a chain of 1-hour jobs advances
the stage roughly one hour per job and later jobs become fast no-ops once done.

Usage (from the project root):
    python scripts/run_retrieval_twotower.py
    ./chain.sh run_stage.sbatch scripts/run_retrieval_twotower.py <num_jobs> <job_name>
"""

import os
import sys

# Make 'src' importable and locate the config regardless of the current dir.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import pipeline  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config(os.path.join(REPO_ROOT, "configs", "config.yaml"))
    pipeline.STAGES["retrieval_twotower"](cfg)


if __name__ == "__main__":
    main()
