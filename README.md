# MovieLens Two-Stage Recommender

A retrieval-then-ranking recommender on MovieLens, built to demonstrate
**judgment about a ranking system under real constraints** rather than a single
headline number. The interesting content is in the *evaluation*: leakage-aware
temporal splitting, a direction-agnostic negative-sampling study, a fair
cold-start comparison, multi-seed variance reporting, and an honest account of
where a more complex model does **not** beat a simpler one.

> Every result below is measured over 3 seeds and reported as `mean ± std`. A
> difference smaller than the seed std is labelled *within noise*, never an
> improvement — the code enforces this (`src/utils/stats.py::is_within_noise`).

---

## TL;DR — what the results actually say

- **Negative sampling is the decisive knob.** Same two-tower architecture:
  `logQ` correction reaches **Recall@10 0.047**, plain `in-batch` only **0.009** —
  a **5×** gap. This is the project's central finding, and its direction came out
  of the data, not an assumption.
- **The two-tower only *edges* a strong MF baseline** (0.047 vs 0.043). Reported
  honestly: no crushing win.
- **The win is in coverage and the tail, not the head.** vs the popularity
  baseline, `logQ` has **26× catalog coverage** (0.39 vs 0.015) and **20× tail
  recall** — which is exactly why looking only at Recall@10 is misleading.
- **Hard negatives made it *worse*** here — an honest negative result worth
  discussing, not hiding.
- **Two-stage reranking works:** GBDT reranking lifts NDCG@10 from
  **0.147 → 0.187 (+27%, above noise)**.
- **ANN is cheap:** HNSW retains **~97% recall at ~1/10 the latency** of exact search.

---

## Architecture

```
user
  |
  |-- Stage 1 (retrieval): from ~3.7k movies, fetch a few hundred candidates
  |     |-- popularity baseline
  |     |-- BPR matrix factorization (baseline)
  |     |-- two-tower + content tower   <-- headline model
  |           (content tower = pretrained title/genre text embeddings; enables cold-start)
  |
  |-- Stage 2 (ranking): rerank the candidates with a GBDT LambdaRank model
```

- **User tower**: user-id embedding + mean-pooled recent item history.
- **Item (content) tower**: item-id embedding + genre projection + a projection
  of a pretrained sentence embedding of the title. The content path lets brand-new
  items be represented with no interaction history.
- **Negative sampling ablation** (`in_batch` / `hard` / `logq`): the central,
  direction-agnostic study — we measure the head/tail/coverage cross-section and
  report whatever the data shows.

---

## Results (MovieLens-1M, 3 seeds)

### Retrieval quality @ 10

| Retriever | Recall@10 | NDCG@10 | Coverage@10 |
|---|---|---|---|
| Popularity | 0.0271 | 0.108 | 0.015 |
| BPR-MF | 0.0433 ± .0005 | 0.136 ± .002 | 0.346 |
| Two-tower (in-batch) | 0.0094 ± .0010 | 0.024 ± .003 | 0.804 |
| Two-tower (hard) | 0.0090 ± .0013 | 0.030 ± .003 | 0.728 |
| **Two-tower (logQ)** | **0.0470 ± .0020** | **0.143 ± .006** | 0.394 |
| Two-tower (logQ, no content) | 0.0465 ± .0002 | 0.140 ± .003 | 0.382 |

![Overall retrieval quality](docs/figures/fig1_overall.png)

### Where the models act — head / tail / cold-item + coverage

`logQ` doesn't beat popularity on the head; it wins by covering the catalog and
reaching the tail. `in-batch` spreads coverage widest but at a large accuracy cost.

![Segmented recall and coverage](docs/figures/fig2_segments.png)

### Cold-start (honest)

On ML-1M every model scores ~0 on cold items, and the content tower shows **no
visible cold-start gain** (logQ 0.047 vs logQ-no-content 0.0465). ML-1M simply
has too few cold items (544, most with no test positives) to show the effect —
this is exactly the hypothesis to test on **ML-25M**, where cold items are plentiful.

### Reranking (Stage 2)

GBDT reranking of the top-200 candidates lifts NDCG@10 by **+0.039 (above noise)**.

![Reranking effect](docs/figures/fig4_rerank.png)

### ANN recall vs latency

Exact inner-product search vs IVF (`nprobe`) and HNSW (`efSearch`). HNSW at
`efSearch=32` keeps ~97% recall at ~1/10 the exact latency.

![Recall-latency tradeoff](docs/figures/fig3_recall_latency.png)

Dataset profile (ML-1M): 6,011 users, 3,678 items, 900k train interactions,
sparsity 0.96, popularity Gini 0.638 (`docs/results/dataset_stats.json`).

---

## Quick start (local / Colab — no HPC needed)

```bash
git clone https://github.com/K-YLMike/recommender-system-movielens.git
cd recommender-system-movielens

python -m venv .venv && source .venv/bin/activate     # or use conda
pip install -r requirements.txt

# 1. download MovieLens once (into ./raw)
bash data/download.sh ml-1m        # or: both  /  ml-25m

# 2. run the pipeline, one stage at a time (each is idempotent + resumable)
python scripts/run_data_prep.py
python scripts/run_baseline_pop.py
python scripts/run_content_features.py
python scripts/run_retrieval_mf.py
python scripts/run_retrieval_twotower.py
python scripts/run_evaluate.py
python scripts/run_ranking_gbdt.py

# 3. figures -> ml-1m/results/figures/
python plots.py --results-dir ml-1m/results
```

Results land in `ml-1m/results/` (`metrics.json`, `ranking.json`,
`dataset_stats.json`). On CPU, ML-1M runs end-to-end in a matter of minutes; only
the two-tower benefits from a GPU. With a CUDA GPU, swap `faiss-cpu` for
`faiss-gpu-cu12` and install a CUDA torch build (see `requirements.txt`).

Everything (data, checkpoints, model cache, outputs) is written under the project
directory by default; override with the `RSM_DATA_ROOT` environment variable.

---

## Configuration

All knobs live in `configs/config.yaml`. The ones you are most likely to touch:

| key | meaning |
|---|---|
| `dataset` | `ml-1m` (dev) or `ml-25m` (final numbers) |
| `seeds` | seeds for multi-seed ablations (report mean ± std) |
| `data.positive_rating_threshold` | rating ≥ this becomes an implicit positive (default 4.0) |
| `data.test_fraction` | most-recent fraction held out as the temporal test set |
| `twotower.variants` | negative-sampling ablation: `in_batch` / `hard` / `logq` |
| `twotower.total_steps`, `ckpt_every_steps` | training length / checkpoint frequency |
| `ann.ivf_nprobe`, `ann.hnsw_efsearch` | ANN sweep points for the recall-latency curve |

---

## Repository layout

```
├── src/                  # pipeline stages + utilities
│   ├── data_prep.py          # load, temporal split, slices, dataset stats
│   ├── baseline_pop.py       # popularity baseline
│   ├── content_features.py   # resumable title-embedding inference
│   ├── retrieval_mf.py       # BPR matrix factorization
│   ├── retrieval_twotower.py # two-tower + content tower (headline)
│   ├── index_faiss.py        # exact + IVF/HNSW ANN, recall-latency sweep
│   ├── ranking_gbdt.py       # LightGBM LambdaRank reranker
│   ├── evaluate.py           # metrics, slices, multi-seed, leakage, latency
│   ├── pipeline.py           # stage registry
│   └── utils/                # checkpoint, config, seed, stats, logging
├── scripts/run_<stage>.py    # one runnable entry point per stage
├── configs/config.yaml       # single source of truth
├── data/download.sh          # one-time MovieLens download
├── plots.py                  # generate the figures above from results JSON
├── docs/                     # published results + figures
└── requirements.txt
```

The pipeline is built to be **crash-safe and resumable**: every stage writes
outputs atomically and marks completion with a `_DONE.json` file; training
checkpoints model+optimizer+RNG state and resumes exactly. (This also lets it run
as a chain of time-limited jobs on an HPC cluster.)

---

## How a production system would differ

This is the offline skeleton, not the scale. In production you would add:
streaming/near-real-time features and embeddings; sharded ANN with quantization
over billions of items; multi-objective ranking (engagement, diversity,
freshness, business rules); position/selection-bias correction and delayed
feedback; online A/B testing instead of a static temporal split; and continuous
retraining.

---

## License

MIT — see [LICENSE](LICENSE).
