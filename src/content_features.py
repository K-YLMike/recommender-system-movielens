"""Stage: content features (resumable text-embedding inference).

Encodes each movie's title (optionally title + genres as a short text) into a
dense vector with a pretrained SentenceTransformer. These embeddings feed the
item content tower and directly enable cold-start recall for items with no
collaborative signal.

This is the canonical *resumable inference* job: encoding tens of thousands of
titles can exceed the 1-hour wall clock when the model is large or the GPU is
shared. We therefore process items in fixed-size shards and commit each shard to
disk atomically, tracking progress in a :class:`ShardManifest`. On resume we
recompute only the shards that are missing or were interrupted mid-write; every
already-finished shard is reused untouched. When all shards are present we
assemble them into a single ``content_embeddings.npy`` and write the stage
marker.

The embedding model is downloaded once into ``HF_HOME`` (set on scratch by the
sbatch header) and thereafter loaded offline, so subsequent jobs never re-fetch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import checkpoint as ckpt
from src.utils.checkpoint import ShardManifest
from src.utils.config import Config, load_config
from src.utils.logging_utils import get_logger, timed
from src import data_prep

LOGGER = get_logger("content_features")
STAGE = "content_features"


def _build_texts(movies: pd.DataFrame, use_genres: bool) -> list[str]:
    """Compose the text encoded per item.

    Combining the title with genres gives the encoder a little structured
    context, which tends to help cold-item matching. Titles are ordered by
    ``item_idx`` so the embedding row order matches the item index.
    """
    movies = movies.sort_values("item_idx")
    if use_genres:
        genres = movies["genres"].fillna("").str.replace("|", ", ", regex=False)
        texts = (movies["title"].fillna("") + " | genres: " + genres).tolist()
    else:
        texts = movies["title"].fillna("").tolist()
    return texts


def _load_encoder(model_name: str):
    """Load the SentenceTransformer encoder (torch/GPU aware).

    Imported lazily so stages that do not need embeddings never import the heavy
    sentence-transformers stack.
    """
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Loading encoder %s on %s", model_name, device)
    return SentenceTransformer(model_name, device=device)


def run(cfg: Config) -> None:
    """Encode item content, resuming shard-by-shard."""
    stage_dir = cfg.paths.stage_dir(STAGE)
    if ckpt.is_stage_done(stage_dir):
        LOGGER.info("Stage '%s' already complete -- skipping.", STAGE)
        return
    stage_dir.mkdir(parents=True, exist_ok=True)

    prepared = data_prep.load_prepared(cfg)
    movies = prepared["movies"]
    use_genres = bool(cfg.get("content", "use_genres_in_text", default=True))
    model_name = cfg.get(
        "content", "model_name", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    shard_size = int(cfg.get("content", "shard_size", default=4096))
    batch_size = int(cfg.get("content", "batch_size", default=256))

    texts = _build_texts(movies, use_genres)
    total = len(texts)
    manifest = ShardManifest.load_or_create(
        shard_dir=stage_dir / "shards", total_units=total, shard_size=shard_size
    )
    pending = manifest.pending_shards()
    LOGGER.info(
        "Content encoding: %d/%d shards already done, %d pending.",
        len(manifest.completed),
        manifest.num_shards(),
        len(pending),
    )

    if pending:
        encoder = _load_encoder(model_name)
        for shard_idx in pending:
            start, end = manifest.shard_range(shard_idx)
            with timed(LOGGER, f"encode shard {shard_idx} [{start}:{end}]"):
                emb = encoder.encode(
                    texts[start:end],
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype(np.float32)
            manifest.commit_shard(shard_idx, emb)

    if not manifest.is_complete():
        # Ran out of wall clock before finishing; the marker is intentionally
        # NOT written, so the next job resumes with the remaining shards.
        LOGGER.warning(
            "Content encoding incomplete (%d/%d shards). Re-submit to continue.",
            len(manifest.completed),
            manifest.num_shards(),
        )
        return

    with timed(LOGGER, "assemble embeddings"):
        embeddings = manifest.assemble()
        assert embeddings.shape[0] == total, "assembled rows != item count"
        ckpt.atomic_write_npy(stage_dir / "content_embeddings.npy", embeddings)

    ckpt.mark_stage_done(
        stage_dir,
        meta={"model_name": model_name, "dim": int(embeddings.shape[1]), "n_items": total},
    )
    LOGGER.info("Content features complete -> %s", stage_dir)


def load_content_embeddings(cfg: Config) -> np.ndarray:
    """Load the assembled content embeddings (requires the stage to be done)."""
    stage_dir = cfg.paths.stage_dir(STAGE)
    if not ckpt.is_stage_done(stage_dir):
        raise RuntimeError("content_features not complete.")
    return np.load(stage_dir / "content_embeddings.npy")


def main() -> None:
    cfg = load_config()
    run(cfg)


if __name__ == "__main__":
    main()
