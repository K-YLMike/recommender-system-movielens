#!/usr/bin/env python
"""Generate result figures for the MovieLens two-stage recommender.

Reads the three JSON files written by the pipeline
(``results/metrics.json`` and ``results/ranking.json``) and produces four
publication-ready figures for the README / interview:

  1. overall Recall@10 & NDCG@10 per model, with +/- std error bars
  2. segmented Recall@10 (head / tail / cold-item) + catalog coverage
  3. ANN recall-latency curve (exact vs IVF vs HNSW)
  4. stage-1 vs reranked NDCG@10

Usage
-----
    python plots.py --results-dir ml-1m/results --out-dir ml-1m/results/figures

If --results-dir is omitted it defaults to "<dataset>/results" for the dataset
in configs/config.yaml, falling back to ml-1m/results.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless: no display needed on the cluster
import matplotlib.pyplot as plt
import numpy as np

# Consistent, readable order + friendly labels for the models.
MODEL_ORDER = [
    "popularity",
    "mf",
    "twotower_in_batch",
    "twotower_hard",
    "twotower_logq",
    "twotower_logq_nocontent",
]
MODEL_LABELS = {
    "popularity": "Popularity",
    "mf": "BPR-MF",
    "twotower_in_batch": "2T in-batch",
    "twotower_hard": "2T hard",
    "twotower_logq": "2T logQ",
    "twotower_logq_nocontent": "2T logQ\n(no content)",
}


def _load(results_dir: str) -> tuple[dict, dict]:
    with open(os.path.join(results_dir, "metrics.json")) as f:
        metrics = json.load(f)
    ranking_path = os.path.join(results_dir, "ranking.json")
    ranking = None
    if os.path.exists(ranking_path):
        with open(ranking_path) as f:
            ranking = json.load(f)
    return metrics, ranking


def _get(metrics: dict, model: str, k: str, seg: str, metric: str):
    """Return (mean, std) for one cell, or (nan, 0) if absent."""
    try:
        node = metrics["sources"][model]["by_k"][k][seg][metric]
        return node["mean"], node["std"]
    except (KeyError, TypeError):
        return float("nan"), 0.0


def _present_models(metrics: dict) -> list[str]:
    return [m for m in MODEL_ORDER if m in metrics.get("sources", {})]


# --------------------------------------------------------------------------- #
# Figure 1: overall Recall@10 & NDCG@10                                       #
# --------------------------------------------------------------------------- #
def fig_overall(metrics: dict, k: str, out_path: str) -> None:
    models = _present_models(metrics)
    labels = [MODEL_LABELS[m] for m in models]
    rec = [_get(metrics, m, k, "overall", "recall") for m in models]
    ndcg = [_get(metrics, m, k, "overall", "ndcg") for m in models]
    rec_mean, rec_std = zip(*rec)
    ndcg_mean, ndcg_std = zip(*ndcg)

    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, rec_mean, w, yerr=rec_std, capsize=4,
           label=f"Recall@{k}", color="#4C72B0")
    ax.bar(x + w / 2, ndcg_mean, w, yerr=ndcg_std, capsize=4,
           label=f"NDCG@{k}", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("score")
    ax.set_title(f"Overall retrieval quality @ {k} (mean +/- std over seeds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2: segmented recall + coverage                                       #
# --------------------------------------------------------------------------- #
def fig_segments(metrics: dict, k: str, out_path: str) -> None:
    models = _present_models(metrics)
    labels = [MODEL_LABELS[m] for m in models]
    segs = ["head", "tail", "cold_item"]
    seg_colors = {"head": "#55A868", "tail": "#C44E52", "cold_item": "#8172B3"}

    x = np.arange(len(models))
    w = 0.25
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for i, seg in enumerate(segs):
        means = [_get(metrics, m, k, seg, "recall")[0] for m in models]
        stds = [_get(metrics, m, k, seg, "recall")[1] for m in models]
        ax1.bar(x + (i - 1) * w, means, w, yerr=stds, capsize=3,
                label=seg, color=seg_colors[seg])
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8, rotation=15)
    ax1.set_ylabel(f"Recall@{k}")
    ax1.set_title(f"Segmented Recall@{k}: head vs tail vs cold-item")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    cov_mean, cov_std = [], []
    for m in models:
        try:
            node = metrics["sources"][m]["by_k"][k]["coverage"]
            cov_mean.append(node["mean"])
            cov_std.append(node["std"])
        except (KeyError, TypeError):
            cov_mean.append(float("nan"))
            cov_std.append(0.0)
    ax2.bar(x, cov_mean, 0.6, yerr=cov_std, capsize=4, color="#4C72B0")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8, rotation=15)
    ax2.set_ylabel("catalog coverage")
    ax2.set_title(f"Catalog coverage @ {k}")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3: recall-latency curve                                              #
# --------------------------------------------------------------------------- #
def fig_recall_latency(metrics: dict, out_path: str) -> None:
    rl = metrics.get("recall_latency")
    if not rl:
        return
    points = rl["points"]
    k = rl.get("k", "")
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for index_type, marker, color in [
        ("ivf", "o", "#4C72B0"),
        ("hnsw", "s", "#DD8452"),
    ]:
        pts = [p for p in points if p["index_type"] == index_type]
        pts.sort(key=lambda p: p["ms_per_query"])
        xs = [p["ms_per_query"] * 1000 for p in pts]  # ms -> microseconds
        ys = [p["recall_at_k_vs_exact"] for p in pts]
        ax.plot(xs, ys, marker=marker, color=color, label=index_type.upper())
        for p in pts:
            ax.annotate(str(p["param_value"]),
                        (p["ms_per_query"] * 1000, p["recall_at_k_vs_exact"]),
                        fontsize=7, textcoords="offset points", xytext=(4, -8))

    exact = [p for p in points if p["index_type"] == "flat_ip"]
    if exact:
        ax.axhline(1.0, ls="--", color="gray", alpha=0.6,
                   label=f"exact ({exact[0]['ms_per_query']*1000:.1f} us)")
    ax.set_xlabel("latency (microseconds / query)")
    ax.set_ylabel(f"recall@{k} vs exact")
    ax.set_title(f"ANN recall-latency tradeoff (labels = nprobe / efSearch)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 4: rerank                                                            #
# --------------------------------------------------------------------------- #
def fig_rerank(ranking: dict, out_path: str) -> None:
    if not ranking:
        return
    s1 = ranking["ndcg_stage1"]
    rr = ranking["ndcg_reranked"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    bars = ax.bar(
        ["Stage-1\n(retrieval)", "Stage-2\n(GBDT rerank)"],
        [s1["mean"], rr["mean"]],
        yerr=[s1["std"], rr["std"]],
        capsize=5,
        color=["#4C72B0", "#55A868"],
    )
    ax.set_ylabel("NDCG@10")
    verdict = "significant" if not ranking.get("within_noise", True) else "within noise"
    ax.set_title(f"Reranking effect: +{ranking['delta']:.3f} ({verdict})")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, [s1["mean"], rr["mean"]]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _default_results_dir() -> str:
    try:
        import yaml
        with open("configs/config.yaml") as f:
            ds = yaml.safe_load(f).get("dataset", "ml-1m")
        return os.path.join(ds, "results")
    except Exception:
        return "ml-1m/results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--k", default="10", help="top-K to plot for bar charts")
    args = ap.parse_args()

    results_dir = args.results_dir or _default_results_dir()
    out_dir = args.out_dir or os.path.join(results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    metrics, ranking = _load(results_dir)

    fig_overall(metrics, args.k, os.path.join(out_dir, "fig1_overall.png"))
    fig_segments(metrics, args.k, os.path.join(out_dir, "fig2_segments.png"))
    fig_recall_latency(metrics, os.path.join(out_dir, "fig3_recall_latency.png"))
    fig_rerank(ranking, os.path.join(out_dir, "fig4_rerank.png"))

    print(f"Figures written to {out_dir}/")
    for f in sorted(os.listdir(out_dir)):
        print("  ", f)


if __name__ == "__main__":
    main()
