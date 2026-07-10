"""Stage: matrix-factorization (BPR) retrieval baseline.

A classic Bayesian Personalized Ranking model (Rendle et al., 2009): learn a
user and an item embedding table so that observed (positive) pairs score higher
than sampled negatives under a dot product. On MovieLens this is a *strong*
retrieval baseline that the two-tower does not automatically beat on aggregate
Recall -- which is exactly the honest framing we want in the results table.

Training is checkpointed with :class:`TrainingCheckpointer`: model + optimizer +
step + RNG state are saved every ``ckpt_every_steps`` atomically, and a resumed
job restores the exact stream. This is what lets a multi-epoch run span several
1-hour Slurm jobs. When the target number of steps is reached we export the
final user/item vectors and write the stage marker.

Multi-seed: this stage is invoked once per seed; outputs live under
``<stage>/seed_<k>/`` so seeds never clobber each other.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import checkpoint as ckpt
from src.utils.checkpoint import TrainingCheckpointer
from src.utils.config import Config, load_config
from src.utils.logging_utils import get_logger, timed
from src.utils.seed import capture_rng_state, restore_rng_state, set_global_seed
from src import data_prep

LOGGER = get_logger("retrieval_mf")
STAGE = "retrieval_mf"


def _seed_dir(cfg: Config, seed: int) -> Path:
    return cfg.paths.stage_dir(STAGE) / f"seed_{seed}"


def _build_bpr_model(n_users: int, n_items: int, dim: int):
    """Construct the BPR embedding tables as a small torch module."""
    import torch
    import torch.nn as nn

    class BPRModel(nn.Module):
        """User/item embedding tables scored by dot product."""

        def __init__(self) -> None:
            super().__init__()
            self.user_emb = nn.Embedding(n_users, dim)
            self.item_emb = nn.Embedding(n_items, dim)
            nn.init.normal_(self.user_emb.weight, std=0.01)
            nn.init.normal_(self.item_emb.weight, std=0.01)

        def forward(
            self, users: "torch.Tensor", pos: "torch.Tensor", neg: "torch.Tensor"
        ) -> "torch.Tensor":
            """Return BPR loss for a batch of (user, pos, neg) triples."""
            u = self.user_emb(users)
            i = self.item_emb(pos)
            j = self.item_emb(neg)
            pos_score = (u * i).sum(dim=-1)
            neg_score = (u * j).sum(dim=-1)
            return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

    return BPRModel()


def _sample_batch(
    pos_users: np.ndarray,
    pos_items: np.ndarray,
    n_items: int,
    batch_size: int,
    rng: np.random.Generator,
):
    """Sample a batch of BPR triples with uniform random negatives.

    Negatives are drawn uniformly from the catalog; a rare false-negative (a
    sampled item the user actually likes) is acceptable noise for BPR.
    """
    idx = rng.integers(0, len(pos_users), size=batch_size)
    users = pos_users[idx]
    pos = pos_items[idx]
    neg = rng.integers(0, n_items, size=batch_size)
    return users, pos, neg


def run(cfg: Config, seed: int) -> None:
    """Train (or resume) the BPR model for a single seed."""
    import torch

    seed_dir = _seed_dir(cfg, seed)
    if ckpt.is_stage_done(seed_dir):
        LOGGER.info("MF seed %d already complete -- skipping.", seed)
        return
    seed_dir.mkdir(parents=True, exist_ok=True)

    prepared = data_prep.load_prepared(cfg)
    train = prepared["train"]
    pos = train[train["label"] == 1]
    n_users = int(prepared["meta"]["n_users"])
    n_items = int(prepared["meta"]["n_items"])
    pos_users = pos["user_idx"].to_numpy()
    pos_items = pos["item_idx"].to_numpy()

    dim = int(cfg.get("retrieval_mf", "dim", default=64))
    lr = float(cfg.get("retrieval_mf", "lr", default=0.01))
    weight_decay = float(cfg.get("retrieval_mf", "weight_decay", default=1e-6))
    batch_size = int(cfg.get("retrieval_mf", "batch_size", default=4096))
    total_steps = int(cfg.get("retrieval_mf", "total_steps", default=20000))
    ckpt_every = int(cfg.get("retrieval_mf", "ckpt_every_steps", default=2000))
    log_every = int(cfg.get("retrieval_mf", "log_every_steps", default=500))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_seed(seed)
    model = _build_bpr_model(n_users, n_items, dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    checkpointer = TrainingCheckpointer(seed_dir / "ckpts", keep_last=2)
    start_step = 0
    state = checkpointer.load_latest()
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng_state(state.get("rng"))
        start_step = int(state["step"])
        LOGGER.info("Resumed MF seed %d from step %d.", seed, start_step)
    else:
        LOGGER.info("Starting MF seed %d fresh.", seed)

    rng = np.random.default_rng(seed + start_step)  # advance stream on resume
    model.train()
    with timed(LOGGER, f"MF training seed {seed} [{start_step}->{total_steps}]"):
        for step in range(start_step, total_steps):
            users, p, n = _sample_batch(pos_users, pos_items, n_items, batch_size, rng)
            loss = model(
                torch.as_tensor(users, device=device),
                torch.as_tensor(p, device=device),
                torch.as_tensor(n, device=device),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % log_every == 0:
                LOGGER.info("MF seed %d step %d/%d loss=%.4f", seed, step + 1, total_steps, float(loss))
            if (step + 1) % ckpt_every == 0:
                checkpointer.save(
                    step + 1,
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step + 1,
                        "rng": capture_rng_state(),
                    },
                )

    # Export final vectors and mark done.
    with torch.no_grad():
        user_vecs = model.user_emb.weight.detach().cpu().numpy().astype(np.float32)
        item_vecs = model.item_emb.weight.detach().cpu().numpy().astype(np.float32)
    ckpt.atomic_write_npy(seed_dir / "user_vectors.npy", user_vecs)
    ckpt.atomic_write_npy(seed_dir / "item_vectors.npy", item_vecs)
    ckpt.mark_stage_done(seed_dir, meta={"dim": dim, "total_steps": total_steps, "seed": seed})
    LOGGER.info("MF seed %d complete -> %s", seed, seed_dir)


def main() -> None:
    """Train MF for every configured seed (each seed resumes independently)."""
    cfg = load_config()
    for seed in cfg.seeds:
        run(cfg, seed)


if __name__ == "__main__":
    main()
