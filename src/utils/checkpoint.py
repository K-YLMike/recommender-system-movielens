"""Crash-safe checkpointing and resume primitives.

This module is the backbone that lets every stage of the pipeline survive being
killed by the Slurm wall-clock limit (H100 jobs on Explorer are capped at ~1h).

Design goals
------------
1. **Atomic writes.** A file is written to a temporary path, ``fsync``-ed, then
   ``os.replace``-d into its final location. ``os.replace`` is atomic on POSIX,
   so a reader never observes a half-written file: the final path is either the
   previous complete version or the new complete version, never a torn one.

2. **Explicit completion markers.** A multi-file artifact (e.g. sharded
   embeddings, a training run with many checkpoints) is only considered
   *finished* once a ``_DONE.json`` marker is written -- again atomically, and
   only *after* every output has been safely flushed to disk. Presence of the
   marker is the single source of truth for "this stage is complete, skip it".

3. **Resume from the last *complete* unit.** If a job dies mid-way, the next job
   inspects the manifest / latest-pointer to find the last unit that was fully
   written, discards anything partial, and continues from there. This directly
   implements the requested behaviour: "if the last run didn't finish recording,
   start from the last unit that *did*."

Nothing here is specific to recommenders; it is deliberately generic so all
stages share exactly one resume implementation.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# Name of the marker file that flags a directory/stage as fully complete.
DONE_MARKER = "_DONE.json"


# --------------------------------------------------------------------------- #
# Atomic primitives                                                           #
# --------------------------------------------------------------------------- #
def _atomic_write_bytes(path: Path, write_fn: Callable[[Any], None]) -> None:
    """Write to ``path`` atomically.

    ``write_fn`` receives an open binary file handle and is responsible for
    writing the payload. We flush + fsync the handle, then ``os.replace`` the
    temp file onto the destination so readers only ever see a complete file.

    Args:
        path: Final destination path.
        write_fn: Callback that writes the payload to the given file handle.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temp file in the same directory so os.replace stays on one
    # filesystem (rename across filesystems is not atomic).
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            write_fn(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)  # atomic on POSIX
    except BaseException:
        # Clean up the partial temp file on any failure (including SIGTERM
        # surfacing as KeyboardInterrupt/SystemExit) so we never leave litter.
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def atomic_write_json(path: str | Path, obj: Any) -> None:
    """Serialise ``obj`` to JSON and write it atomically."""
    def _write(handle: Any) -> None:
        handle.write(json.dumps(obj, indent=2, sort_keys=True).encode("utf-8"))

    _atomic_write_bytes(Path(path), _write)


def read_json(path: str | Path) -> Optional[dict]:
    """Read a JSON file, returning ``None`` if it is missing or unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        # A truncated/corrupt JSON is treated as "not there" so callers fall
        # back to recomputing the unit rather than crashing.
        return None


def atomic_write_npy(path: str | Path, array: np.ndarray) -> None:
    """Write a NumPy array to ``.npy`` atomically."""
    def _write(handle: Any) -> None:
        np.save(handle, array, allow_pickle=False)

    _atomic_write_bytes(Path(path), _write)


def atomic_save_torch(path: str | Path, state: Any) -> None:
    """Save a torch object (state dict, etc.) atomically.

    Imported lazily so non-torch stages (data prep, FAISS, GBDT) never pay the
    torch import cost.
    """
    import torch  # local import on purpose

    def _write(handle: Any) -> None:
        torch.save(state, handle)

    _atomic_write_bytes(Path(path), _write)


# --------------------------------------------------------------------------- #
# Stage-level completion marker                                               #
# --------------------------------------------------------------------------- #
def mark_stage_done(stage_dir: str | Path, meta: Optional[dict] = None) -> None:
    """Write the ``_DONE.json`` marker for a stage directory.

    Call this only once every output for the stage is safely on disk.

    Args:
        stage_dir: Directory whose completion we are flagging.
        meta: Optional metadata (metrics, row counts, timing) stored in the
            marker for debugging and downstream sanity checks.
    """
    stage_dir = Path(stage_dir)
    payload = {"done": True, "meta": meta or {}}
    atomic_write_json(stage_dir / DONE_MARKER, payload)


def is_stage_done(stage_dir: str | Path) -> bool:
    """Return ``True`` iff the stage's ``_DONE.json`` marker exists and is valid."""
    marker = read_json(Path(stage_dir) / DONE_MARKER)
    return bool(marker and marker.get("done") is True)


# --------------------------------------------------------------------------- #
# Sharded-artifact manifest (resumable inference, e.g. text embeddings)       #
# --------------------------------------------------------------------------- #
@dataclass
class ShardManifest:
    """Tracks which shards of a sharded artifact have been fully written.

    Used by resumable *inference* jobs (e.g. encoding 62K movie titles into
    embeddings): the work is split into shards, each shard is written atomically,
    and the manifest -- itself written atomically -- records the shards that are
    known-good. On resume, any shard not listed in the manifest (or whose file is
    missing / the wrong size) is simply recomputed.

    Attributes:
        path: Location of the manifest JSON.
        shard_dir: Directory holding the shard files.
        total_units: Total number of items to process (e.g. number of movies).
        shard_size: Number of items per shard.
        completed: Mapping ``shard_index -> expected_row_count`` for done shards.
    """

    path: Path
    shard_dir: Path
    total_units: int
    shard_size: int
    completed: dict[int, int] = field(default_factory=dict)

    @classmethod
    def load_or_create(
        cls,
        shard_dir: str | Path,
        total_units: int,
        shard_size: int,
    ) -> "ShardManifest":
        """Load an existing manifest, validating each recorded shard on disk."""
        shard_dir = Path(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)
        path = shard_dir / "manifest.json"
        obj = read_json(path)
        completed: dict[int, int] = {}
        if obj is not None:
            for k, v in obj.get("completed", {}).items():
                idx = int(k)
                shard_file = shard_dir / f"shard_{idx:06d}.npy"
                # Re-validate: file must exist and have the recorded row count.
                if shard_file.exists():
                    try:
                        arr = np.load(shard_file, mmap_mode="r")
                        if arr.shape[0] == int(v):
                            completed[idx] = int(v)
                    except (ValueError, OSError):
                        # Corrupt/truncated shard -> drop it, it'll be recomputed.
                        pass
        return cls(
            path=path,
            shard_dir=shard_dir,
            total_units=total_units,
            shard_size=shard_size,
            completed=completed,
        )

    def num_shards(self) -> int:
        """Total number of shards for ``total_units`` items."""
        return (self.total_units + self.shard_size - 1) // self.shard_size

    def pending_shards(self) -> list[int]:
        """Indices of shards still needing computation, in order."""
        return [i for i in range(self.num_shards()) if i not in self.completed]

    def shard_range(self, idx: int) -> tuple[int, int]:
        """Half-open ``[start, end)`` item range covered by shard ``idx``."""
        start = idx * self.shard_size
        end = min(start + self.shard_size, self.total_units)
        return start, end

    def shard_file(self, idx: int) -> Path:
        """Path of the ``.npy`` file for shard ``idx``."""
        return self.shard_dir / f"shard_{idx:06d}.npy"

    def commit_shard(self, idx: int, array: np.ndarray) -> None:
        """Write one shard atomically, then update the manifest atomically.

        Ordering matters: the shard file is fully written *before* the manifest
        records it. If the job dies between those two steps, the (untracked)
        shard is simply recomputed next time -- correct, just slightly wasteful.
        """
        atomic_write_npy(self.shard_file(idx), array)
        self.completed[idx] = int(array.shape[0])
        atomic_write_json(
            self.path,
            {
                "total_units": self.total_units,
                "shard_size": self.shard_size,
                "num_shards": self.num_shards(),
                "completed": {str(k): v for k, v in sorted(self.completed.items())},
            },
        )

    def is_complete(self) -> bool:
        """Return ``True`` iff every shard is present and validated."""
        return len(self.completed) == self.num_shards()

    def assemble(self) -> np.ndarray:
        """Concatenate all shards into one array (call only when complete)."""
        if not self.is_complete():
            raise RuntimeError(
                f"Cannot assemble incomplete manifest: "
                f"{len(self.completed)}/{self.num_shards()} shards present."
            )
        parts = [np.load(self.shard_file(i)) for i in range(self.num_shards())]
        return np.concatenate(parts, axis=0)


# --------------------------------------------------------------------------- #
# Training checkpoints (torch): step-based, keep-last-K, corruption-safe       #
# --------------------------------------------------------------------------- #
class TrainingCheckpointer:
    """Manages resumable training checkpoints for a single run.

    A "run" is one (model, seed, config) combination -- e.g. the two-tower with
    ``logQ`` negatives and seed 0. Checkpoints are keyed by global step. A
    ``latest.json`` pointer is updated atomically *after* a checkpoint file is
    fully written, so on resume we always load a checkpoint we know is complete;
    a checkpoint that was interrupted mid-write is never pointed to and is
    ignored, and we fall back to the previous good one.

    Args:
        run_dir: Directory for this run's checkpoints.
        keep_last: Number of recent checkpoints to retain on disk.
    """

    def __init__(self, run_dir: str | Path, keep_last: int = 3) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self._pointer = self.run_dir / "latest.json"

    def _ckpt_path(self, step: int) -> Path:
        return self.run_dir / f"ckpt_step_{step:09d}.pt"

    def save(self, step: int, state: dict) -> None:
        """Persist a checkpoint at ``step`` and advance the latest-pointer.

        ``state`` should contain everything needed to resume exactly: model and
        optimizer state dicts, scheduler state, ``step``/``epoch``, RNG states,
        and the best-metric-so-far. See ``retrieval_twotower`` for the payload.
        """
        path = self._ckpt_path(step)
        atomic_save_torch(path, state)  # file is complete before we point at it
        atomic_write_json(self._pointer, {"step": int(step), "file": path.name})
        self._prune()

    def _prune(self) -> None:
        """Delete old checkpoints beyond ``keep_last`` (never the pointed one)."""
        ckpts = sorted(self.run_dir.glob("ckpt_step_*.pt"))
        pointer = read_json(self._pointer) or {}
        keep_name = pointer.get("file")
        stale = ckpts[: max(0, len(ckpts) - self.keep_last)]
        for path in stale:
            if path.name != keep_name:
                try:
                    path.unlink()
                except OSError:
                    pass

    def load_latest(self) -> Optional[dict]:
        """Load the newest *complete* checkpoint, or ``None`` if none exist.

        Robustness: we try the pointed checkpoint first; if it fails to load
        (corrupt), we walk backwards through the remaining checkpoint files until
        one loads. This tolerates the rare case where the pointer and the file
        disagree.
        """
        import torch  # local import

        pointer = read_json(self._pointer)
        candidates: list[Path] = []
        if pointer and pointer.get("file"):
            pointed = self.run_dir / pointer["file"]
            if pointed.exists():
                candidates.append(pointed)
        # Fallback candidates: all checkpoints, newest first, minus the pointed.
        for path in sorted(self.run_dir.glob("ckpt_step_*.pt"), reverse=True):
            if path not in candidates:
                candidates.append(path)

        for path in candidates:
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except Exception:  # noqa: BLE001 -- any load failure => treat as corrupt
                # Truncated/corrupt checkpoint (UnpicklingError, BadZipFile,
                # EOFError, ...) -> fall back to the next older checkpoint. We
                # deliberately catch broadly: the whole point is to survive a
                # checkpoint that was interrupted mid-write.
                continue
        return None
