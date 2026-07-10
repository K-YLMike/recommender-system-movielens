# MovieLens Two-Stage Recommender

A retrieval then ranking recommender on MovieLens, built in the same shape a real
industrial system takes: a cheap first stage that narrows a large catalog down to
a few hundred candidates, and an expensive second stage that carefully reorders
them. The point of the project is not to chase one big accuracy number. It is to
show clear judgment about how a ranking system behaves under realistic
constraints, and to stay honest about what works and what does not.

Instead of training one model and reporting one good-looking metric, the project
is organized around a few questions: does the two-stage structure earn its
complexity, what actually decides whether the two-tower works, where does each
model help once you look past the aggregate number, and can the evaluation even be
trusted. Baselines come first (popularity, then BPR matrix factorization), then a
two-tower retriever with a content tower, then a gradient-boosted reranker, all
compared under one protocol: a temporal split to avoid leakage, three seeds, and a
rule that any difference smaller than the seed noise is reported as "within noise"
rather than sold as a win.

## Getting Started

```bash
git clone https://github.com/K-YLMike/recommender-system-movielens.git
cd recommender-system-movielens

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash data/download.sh ml-1m          # download MovieLens once, into ./raw

python scripts/run_data_prep.py
python scripts/run_baseline_pop.py
python scripts/run_content_features.py
python scripts/run_retrieval_mf.py
python scripts/run_retrieval_twotower.py
python scripts/run_evaluate.py
python scripts/run_ranking_gbdt.py

python plots.py --results-dir ml-1m/results
```

Results are written to `ml-1m/results/`. On a CPU the MovieLens-1M run finishes in
a few minutes; only the two-tower really benefits from a GPU (with one, swap
`faiss-cpu` for `faiss-gpu-cu12` and install a CUDA torch build). Switch to the
larger dataset by setting `dataset: ml-25m` in `configs/config.yaml`.

## Results

MovieLens-1M, averaged over 3 seeds. The noise rule is enforced in code
(`src/utils/stats.py`).

**Negative sampling is the decisive knob, not the architecture.** The same
two-tower reaches Recall@10 of 0.047 with a logQ popularity correction but only
0.009 with plain in-batch negatives, a 5x swing from one design choice.

**The two-tower only edges a strong baseline.** logQ two-tower gets 0.047 versus
0.043 for BPR matrix factorization. On clean MovieLens a well-tuned classic
baseline is hard to beat, and that is reported plainly.

**The real advantage is coverage and the tail, not the head.** Against the
popularity baseline, the logQ two-tower has about 26x the catalog coverage (0.39
versus 0.015) and roughly 20x the tail recall, while the head is similar. This is
why looking only at aggregate Recall would mislead.

**Hard negatives made it worse** here, likely because the batch-level mining
introduces bias on a dataset this small. It is kept as an honest negative result.

**Reranking works.** The GBDT reranker lifts NDCG@10 from 0.147 to 0.187, a 27
percent relative gain that clears the noise.

**Approximate search is nearly free.** HNSW keeps about 97 percent of exact recall
at roughly one tenth of the latency.

**Cold-start needs bigger data to show up.** On MovieLens-1M the content tower
shows no measurable cold-start gain, because the data has only 544 cold items, most
without test interactions. That is a limit of the data, and the thing to re-test on
MovieLens-25M.

### Retrieval quality @ 10

| Retriever | Recall@10 | NDCG@10 | Coverage@10 |
|---|---|---|---|
| Popularity | 0.0271 | 0.108 | 0.015 |
| BPR-MF | 0.0433 | 0.136 | 0.346 |
| Two-tower (in-batch) | 0.0094 | 0.024 | 0.804 |
| Two-tower (hard) | 0.0090 | 0.030 | 0.728 |
| **Two-tower (logQ)** | **0.0470** | **0.143** | 0.394 |
| Two-tower (logQ, no content) | 0.0465 | 0.140 | 0.382 |

![Overall retrieval quality](docs/figures/fig1_overall.png)

![Segmented recall and coverage](docs/figures/fig2_segments.png)

![Reranking effect](docs/figures/fig4_rerank.png)

![Recall-latency tradeoff](docs/figures/fig3_recall_latency.png)

Dataset profile: 6,011 users, 3,678 items, 900k training interactions, sparsity
0.96, popularity Gini 0.64 (`docs/results/dataset_stats.json`).

## How it works

Stage one retrieves a few hundred candidates from about 3.7k items using three
retrievers (popularity, BPR matrix factorization, and the headline two-tower).
Stage two reranks those candidates with a LightGBM LambdaRank model.

The two-tower has a user tower (user embedding plus a mean pool of recent item
history) and an item tower (item embedding, a genre projection, and a projection of
a pretrained sentence embedding of the title). That content path lets a brand new
item with no interactions still be represented, which is the basis for cold-start
handling.

The pipeline is crash-safe and resumable: each stage writes outputs atomically and
marks completion with a `_DONE.json` file, and training checkpoints the model,
optimizer, and RNG state so a run resumes exactly where it stopped.

## Future Work

This is the offline skeleton of a recommender, and there is a clear path to grow
it. The immediate next step is MovieLens-25M, where the content tower's
cold-start value should actually show up. Beyond that, the natural extensions are
streaming and near-real-time features, approximate search over much larger
catalogs, multi-objective ranking that balances relevance with diversity and
freshness, correction for position and selection bias, and moving from a static
offline split to online A/B testing with continuous retraining.

## License

Released under the MIT License: free to use, modify, and distribute, with no
warranty. Full text in [LICENSE](LICENSE).
