"""Stage/utility: FAISS approximate nearest-neighbour retrieval.

Provides the machinery for the recall-latency study:

* An **exact** inner-product index (``IndexFlatIP``) as the ground-truth top-K.
* Approximate indexes (**IVF**, **HNSW**) whose speed/accuracy tradeoff is swept
  over ``nprobe`` / ``efSearch``.

The exact index defines "true" recall; each approximate configuration is scored
by how much recall it retains versus how fast it answers, producing the
recall-latency *curve* (not two isolated points) that the evaluation reports.

FAISS index construction is fast, so this is used as a library by ``evaluate``
rather than run as its own long checkpointed stage; a cached exact-search result
can be reused across ANN configs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


def build_flat_ip(item_vectors: np.ndarray):
    """Build an exact inner-product index over item vectors."""
    import faiss

    dim = item_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(item_vectors, dtype=np.float32))
    return index


def build_ivf(item_vectors: np.ndarray, nlist: int = 256):
    """Build an IVF (inverted-file) index; ``nlist`` = number of coarse cells."""
    import faiss

    dim = item_vectors.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    vecs = np.ascontiguousarray(item_vectors, dtype=np.float32)
    index.train(vecs)
    index.add(vecs)
    return index


def build_hnsw(item_vectors: np.ndarray, m: int = 32):
    """Build an HNSW graph index; ``m`` = graph connectivity."""
    import faiss

    dim = item_vectors.shape[1]
    index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    index.add(np.ascontiguousarray(item_vectors, dtype=np.float32))
    return index


def search(index, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """Run a top-K search and measure per-query latency.

    Returns:
        ``(neighbour_indices[n_queries, k], mean_ms_per_query)``.
    """
    q = np.ascontiguousarray(query_vectors, dtype=np.float32)
    start = time.perf_counter()
    _, idx = index.search(q, k)
    elapsed = time.perf_counter() - start
    ms_per_query = 1000.0 * elapsed / max(len(q), 1)
    return idx, ms_per_query


@dataclass
class SweepPoint:
    """One point on the recall-latency curve."""

    index_type: str
    param_name: str
    param_value: int
    recall_at_k_vs_exact: float
    ms_per_query: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _recall_vs_exact(approx_idx: np.ndarray, exact_idx: np.ndarray) -> float:
    """Fraction of the exact top-K retrieved by the approximate search.

    Averaged over queries. This isolates *index* recall loss from *model*
    quality: it asks "given the model's vectors, how much of the true top-K did
    the ANN structure return?"
    """
    n = exact_idx.shape[0]
    k = exact_idx.shape[1]
    hits = 0
    for row in range(n):
        hits += len(set(approx_idx[row]).intersection(exact_idx[row].tolist()))
    return hits / (n * k)


def recall_latency_sweep(
    item_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
    ivf_nprobe: list[int],
    hnsw_efsearch: list[int],
    nlist: int = 256,
    hnsw_m: int = 32,
) -> list[SweepPoint]:
    """Sweep IVF ``nprobe`` and HNSW ``efSearch`` to trace recall vs latency.

    The exact FlatIP search is computed once as the ground truth. Returns a list
    of :class:`SweepPoint`, one per configuration, ready to serialise and plot.
    """
    import faiss

    exact = build_flat_ip(item_vectors)
    exact_idx, exact_ms = search(exact, query_vectors, k)

    points = [SweepPoint("flat_ip", "exact", 1, 1.0, exact_ms)]

    ivf = build_ivf(item_vectors, nlist=nlist)
    for nprobe in ivf_nprobe:
        ivf.nprobe = nprobe
        idx, ms = search(ivf, query_vectors, k)
        points.append(
            SweepPoint("ivf", "nprobe", nprobe, _recall_vs_exact(idx, exact_idx), ms)
        )

    hnsw = build_hnsw(item_vectors, m=hnsw_m)
    for ef in hnsw_efsearch:
        hnsw.hnsw.efSearch = ef
        idx, ms = search(hnsw, query_vectors, k)
        points.append(
            SweepPoint("hnsw", "efSearch", ef, _recall_vs_exact(idx, exact_idx), ms)
        )
    return points
