"""Stage: two-tower retrieval (headline model) with a content tower.

Architecture
------------
* **User tower**: ``user_id`` embedding concatenated with the mean-pooled
  embeddings of the user's most recent ``history_len`` train items (using the
  *shared* item-id embedding table), passed through an MLP -> ``u in R^d``.
  Cold users with no history fall back to the id embedding alone.
* **Item tower (content)**: ``item_id`` embedding + a projection of the genre
  multi-hot + (optional) a projection of the pretrained title/content embedding,
  passed through an MLP -> ``v in R^d``. The content path is what lets brand-new
  items be represented without any collaborative signal (cold-start).

Training: in-batch sampled softmax (InfoNCE) with temperature ``tau``. Each row's
positive is the item at the same batch position; all other in-batch items are
negatives.

Negative-sampling ablation (the central, *direction-agnostic* study)
--------------------------------------------------------------------
The ``variant`` switch selects one of:

* ``in_batch``  -- in-batch negatives only (uncorrected).
* ``hard``      -- in-batch negatives plus mined hard extra negatives.
* ``logq``      -- in-batch negatives with a ``log Q`` popularity correction that
  subtracts each candidate's log sampling frequency from its logit (Yi et al.,
  2019). We do **not** assume which segment (head/tail/coverage) this helps; the
  evaluation stage measures it and the README reports whatever the data shows.

Resumability: identical checkpointing scheme to the MF stage -- model/optimizer/
step/RNG saved atomically every ``ckpt_every_steps`` and restored on resume, so a
run survives arbitrarily many 1-hour job kills. Outputs are namespaced by
``<stage>/<variant>/seed_<k>/`` so every ablation cell is independent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import checkpoint as ckpt
from src.utils.checkpoint import TrainingCheckpointer
from src.utils.config import Config, load_config
from src.utils.logging_utils import get_logger, timed
from src.utils.seed import capture_rng_state, restore_rng_state, set_global_seed
from src import data_prep, content_features

LOGGER = get_logger("retrieval_twotower")
STAGE = "retrieval_twotower"


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
def _build_model(
    n_users: int,
    n_items: int,
    dim: int,
    n_genres: int,
    content_dim: int,
    use_content: bool,
):
    """Construct the two-tower module.

    ``content_dim`` is ignored when ``use_content`` is False (the no-content
    ablation), which lets us reuse the exact same architecture minus the content
    path for a fair comparison.
    """
    import torch
    import torch.nn as nn

    class TwoTower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.item_id_emb = nn.Embedding(n_items + 1, dim, padding_idx=n_items)
            self.user_id_emb = nn.Embedding(n_users, dim)
            self.genre_proj = nn.Linear(n_genres, dim)
            self.use_content = use_content
            if use_content:
                self.content_proj = nn.Linear(content_dim, dim)
            item_in = dim * (3 if use_content else 2)
            self.item_mlp = nn.Sequential(
                nn.Linear(item_in, dim), nn.ReLU(), nn.Linear(dim, dim)
            )
            self.user_mlp = nn.Sequential(
                nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim)
            )
            nn.init.normal_(self.item_id_emb.weight, std=0.01)
            nn.init.normal_(self.user_id_emb.weight, std=0.01)

        def item_tower(
            self,
            item_idx: "torch.Tensor",
            genre_feat: "torch.Tensor",
            content_feat: Optional["torch.Tensor"],
        ) -> "torch.Tensor":
            """Encode items into normalised vectors ``v``."""
            parts = [self.item_id_emb(item_idx), self.genre_proj(genre_feat)]
            if self.use_content:
                parts.append(self.content_proj(content_feat))
            v = self.item_mlp(torch.cat(parts, dim=-1))
            return torch.nn.functional.normalize(v, dim=-1)

        def user_tower(
            self, user_idx: "torch.Tensor", hist_idx: "torch.Tensor", hist_mask: "torch.Tensor"
        ) -> "torch.Tensor":
            """Encode users via id embedding + masked mean-pool of history items."""
            hist_emb = self.item_id_emb(hist_idx)  # [B, L, d]
            mask = hist_mask.unsqueeze(-1).float()  # [B, L, 1]
            summed = (hist_emb * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / counts  # zero vector for cold users
            u = self.user_mlp(torch.cat([self.user_id_emb(user_idx), pooled], dim=-1))
            return torch.nn.functional.normalize(u, dim=-1)

    return TwoTower()


# --------------------------------------------------------------------------- #
# Feature tensors                                                             #
# --------------------------------------------------------------------------- #
def _padded_histories(
    history_flat: np.ndarray, history_offsets: np.ndarray, history_len: int, pad_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build a ``[n_users, history_len]`` matrix of the most recent items.

    Users with fewer than ``history_len`` items are right-padded with ``pad_idx``;
    the mask marks valid positions. Recency is preserved (history is stored in
    chronological order, we take the tail).
    """
    n_users = len(history_offsets) - 1
    hist = np.full((n_users, history_len), pad_idx, dtype=np.int64)
    mask = np.zeros((n_users, history_len), dtype=np.bool_)
    for u in range(n_users):
        start, end = int(history_offsets[u]), int(history_offsets[u + 1])
        seq = history_flat[start:end]
        if len(seq) == 0:
            continue
        recent = seq[-history_len:]
        hist[u, : len(recent)] = recent
        mask[u, : len(recent)] = True
    return hist, mask


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def _variant_dir(cfg: Config, variant: str, seed: int) -> Path:
    return cfg.paths.stage_dir(STAGE) / variant / f"seed_{seed}"


def run(cfg: Config, variant: str, seed: int, use_content: bool = True) -> None:
    """Train (or resume) the two-tower for one (variant, seed) cell."""
    import torch
    import torch.nn.functional as F

    tag = variant if use_content else f"{variant}_nocontent"
    out_dir = cfg.paths.stage_dir(STAGE) / tag / f"seed_{seed}"
    if ckpt.is_stage_done(out_dir):
        LOGGER.info("Two-tower [%s seed %d] already complete -- skipping.", tag, seed)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared = data_prep.load_prepared(cfg)
    train = prepared["train"]
    pos = train[train["label"] == 1]
    n_users = int(prepared["meta"]["n_users"])
    n_items = int(prepared["meta"]["n_items"])
    genre_multihot = prepared["genre_multihot"]
    n_genres = genre_multihot.shape[1]

    content = None
    content_dim = 0
    if use_content:
        content = content_features.load_content_embeddings(cfg)
        content_dim = content.shape[1]

    # Hyper-parameters.
    dim = int(cfg.get("twotower", "dim", default=64))
    tau = float(cfg.get("twotower", "temperature", default=0.05))
    lr = float(cfg.get("twotower", "lr", default=0.005))
    batch_size = int(cfg.get("twotower", "batch_size", default=2048))
    history_len = int(cfg.get("twotower", "history_len", default=30))
    total_steps = int(cfg.get("twotower", "total_steps", default=30000))
    ckpt_every = int(cfg.get("twotower", "ckpt_every_steps", default=2000))
    log_every = int(cfg.get("twotower", "log_every_steps", default=500))
    n_extra_neg = int(cfg.get("twotower", "extra_negatives", default=2048))
    hard_pool = int(cfg.get("twotower", "hard_pool", default=8192))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_seed(seed)

    # Static feature tensors (moved to device once).
    genre_t = torch.as_tensor(genre_multihot, device=device)
    content_t = torch.as_tensor(content, device=device) if use_content else None
    hist_np, mask_np = _padded_histories(
        prepared["history_flat"], prepared["history_offsets"], history_len, pad_idx=n_items
    )
    hist_t = torch.as_tensor(hist_np, device=device)
    mask_t = torch.as_tensor(mask_np, device=device)

    # log Q popularity correction term (per item), used only by the logq variant.
    item_counts = np.bincount(pos["item_idx"].to_numpy(), minlength=n_items).astype(np.float64)
    item_prob = item_counts / max(item_counts.sum(), 1.0)
    log_q = torch.as_tensor(np.log(item_prob + 1e-12), device=device, dtype=torch.float32)

    pos_users = pos["user_idx"].to_numpy()
    pos_items = pos["item_idx"].to_numpy()

    model = _build_model(n_users, n_items, dim, n_genres, content_dim, use_content).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    checkpointer = TrainingCheckpointer(out_dir / "ckpts", keep_last=2)
    start_step = 0
    state = checkpointer.load_latest()
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng_state(state.get("rng"))
        start_step = int(state["step"])
        LOGGER.info("Resumed two-tower [%s seed %d] from step %d.", tag, seed, start_step)
    else:
        LOGGER.info("Starting two-tower [%s seed %d] fresh.", tag, seed)

    def item_vectors(item_idx: "torch.Tensor") -> "torch.Tensor":
        cfeat = content_t[item_idx] if use_content else None
        return model.item_tower(item_idx, genre_t[item_idx], cfeat)

    rng = np.random.default_rng(seed + start_step)
    model.train()
    with timed(LOGGER, f"two-tower [{tag} seed {seed}] [{start_step}->{total_steps}]"):
        for step in range(start_step, total_steps):
            batch_idx = rng.integers(0, len(pos_users), size=batch_size)
            u_idx = torch.as_tensor(pos_users[batch_idx], device=device)
            i_idx = torch.as_tensor(pos_items[batch_idx], device=device)

            u = model.user_tower(u_idx, hist_t[u_idx], mask_t[u_idx])  # [B, d]
            v = item_vectors(i_idx)  # [B, d] -- in-batch candidates

            cand_idx = i_idx
            cand_v = v
            # Optionally append extra (uniform or hard) negatives as shared columns.
            if variant in ("hard",) and n_extra_neg > 0:
                pool = torch.as_tensor(
                    rng.integers(0, n_items, size=hard_pool), device=device
                )
                pool_v = item_vectors(pool)
                # Batch-level hard mining: pick items with the highest mean score
                # against the current users (a shared, defensible approximation of
                # hard negatives; documented as such in the README).
                mean_u = F.normalize(u.mean(dim=0, keepdim=True), dim=-1)
                pool_scores = (mean_u @ pool_v.t()).squeeze(0)
                hard_sel = torch.topk(pool_scores, k=min(n_extra_neg, hard_pool)).indices
                cand_idx = torch.cat([i_idx, pool[hard_sel]], dim=0)
                cand_v = torch.cat([v, pool_v[hard_sel]], dim=0)

            logits = (u @ cand_v.t()) / tau  # [B, C]
            if variant == "logq":
                logits = logits - log_q[cand_idx].unsqueeze(0)  # popularity correction
            labels = torch.arange(u.shape[0], device=device)  # diagonal positives
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % log_every == 0:
                LOGGER.info(
                    "TT [%s seed %d] step %d/%d loss=%.4f",
                    tag, seed, step + 1, total_steps, float(loss),
                )
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

    # Export final user/item vectors for indexing + evaluation.
    with timed(LOGGER, f"export vectors [{tag} seed {seed}]"):
        model.eval()
        with torch.no_grad():
            all_items = torch.arange(n_items, device=device)
            item_vecs = _batched_item_vectors(
                model, all_items, genre_t, content_t, use_content, batch=8192
            ).cpu().numpy().astype(np.float32)
            all_users = torch.arange(n_users, device=device)
            user_vecs = _batched_user_vectors(
                model, all_users, hist_t, mask_t, batch=8192
            ).cpu().numpy().astype(np.float32)

    ckpt.atomic_write_npy(out_dir / "item_vectors.npy", item_vecs)
    ckpt.atomic_write_npy(out_dir / "user_vectors.npy", user_vecs)
    ckpt.mark_stage_done(
        out_dir,
        meta={
            "variant": variant,
            "use_content": use_content,
            "dim": dim,
            "temperature": tau,
            "total_steps": total_steps,
            "seed": seed,
        },
    )
    LOGGER.info("Two-tower [%s seed %d] complete -> %s", tag, seed, out_dir)


def _batched_item_vectors(model, item_idx, genre_t, content_t, use_content, batch):
    """Encode all items in batches to bound memory during export."""
    import torch

    out = []
    for start in range(0, len(item_idx), batch):
        chunk = item_idx[start:start + batch]
        cfeat = content_t[chunk] if use_content else None
        out.append(model.item_tower(chunk, genre_t[chunk], cfeat))
    return torch.cat(out, dim=0)


def _batched_user_vectors(model, user_idx, hist_t, mask_t, batch):
    """Encode all users in batches to bound memory during export."""
    import torch

    out = []
    for start in range(0, len(user_idx), batch):
        chunk = user_idx[start:start + batch]
        out.append(model.user_tower(chunk, hist_t[chunk], mask_t[chunk]))
    return torch.cat(out, dim=0)


def main() -> None:
    """Train every (variant, seed) cell, plus the no-content ablation.

    Variants and seeds come from config. We also train the headline variant
    without the content tower to power the fair cold-start comparison.
    """
    cfg = load_config()
    variants = cfg.get("twotower", "variants", default=["in_batch", "hard", "logq"])
    headline = cfg.get("twotower", "headline_variant", default="logq")
    for seed in cfg.seeds:
        for variant in variants:
            run(cfg, variant, seed, use_content=True)
        # No-content counterpart of the headline variant (fair cold-start baseline
        # is a popularity backoff, handled in evaluate; this gives the model-side
        # no-content comparison for the main table).
        run(cfg, headline, seed, use_content=False)


if __name__ == "__main__":
    main()
