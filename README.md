# MovieLens Two-Stage Recommender

A retrieval then ranking recommender built on MovieLens, in the same shape a real
industrial recommender takes: a cheap first stage that narrows a large catalog
down to a few hundred candidates, and an expensive second stage that carefully
reorders them. The goal of this project is not to chase one big accuracy number.
It is to show sound judgment about how a ranking system behaves under realistic
constraints, and to be honest about what does and does not work.

## The story

Most recommender demos train one model, report one metric that looks good, and
stop there. That skips the parts that actually matter in practice: whether the
evaluation is even measuring the right thing, whether a fancier model beats a
simple baseline once you account for noise, and where a system helps versus where
it quietly fails.

So this project is organized around questions rather than models:

1. **Does the two-stage structure earn its complexity?** Build a strong, honest
   baseline first (popularity, then BPR matrix factorization), then a two-tower
   retriever with a content tower, then a gradient-boosted reranker. Compare them
   under one protocol.
2. **What actually decides whether the two-tower works?** The suspicion going in
   was that negative sampling matters more than architecture. So the two-tower is
   trained under three negative-sampling schemes (in-batch, hard, and a logQ
   popularity correction) and compared head to head.
3. **Where does each model help?** Aggregate Recall hides a lot. Every metric is
   also broken down by popular head items, long-tail items, cold items, cold
   users, and catalog coverage, so you can see what a change is really doing.
4. **Is the evaluation trustworthy?** The split is temporal (train on the past,
   test on the future) to avoid leakage, results are averaged over three seeds
   with standard deviation, and any difference smaller than that seed noise is
   reported as "within noise" rather than dressed up as a win.

The interesting output of the project is not a leaderboard row. It is a set of
findings, several of which cut against the naive expectation.

## What the results say

All numbers are MovieLens-1M, averaged over 3 seeds, reported as mean plus or
minus standard deviation. The pipeline enforces the noise rule in code
(`src/utils/stats.py`).

**1. Negative sampling is the decisive knob, not the architecture.** The same
two-tower reaches Recall@10 of 0.047 with the logQ correction but only 0.009 with
plain in-batch negatives. That is a 5x swing from one design choice, and the
direction came out of the experiment rather than being assumed.

**2. The two-tower only edges a strong baseline.** logQ two-tower lands at 0.047
Recall@10 versus 0.043 for BPR matrix factorization. On clean MovieLens, a
well-tuned classic baseline is hard to beat, and this is reported plainly instead
of being hidden.

**3. The real advantage is coverage and the tail, not the head.** Compared with
the popularity baseline, the logQ two-tower reaches about 26x the catalog coverage
(0.39 versus 0.015) and roughly 20x the tail recall, while the head is similar.
This is exactly why looking only at aggregate Recall@10 would be misleading.

**4. Hard negatives made it worse.** The hard-negative variant underperformed
plain in-batch here, likely because the batch-level mining used introduces bias on
a dataset this small. It is kept in the results as an honest negative result.

**5. Reranking works.** The GBDT reranker lifts NDCG@10 from 0.147 to 0.187, a
27 percent relative gain that clears the seed noise. The two-stage structure pays
for itself.

**6. Approximate search is nearly free.** HNSW retains about 97 percent of exact
recall at roughly one tenth of the exact search latency, which is the practical
reason production systems index with ANN instead of brute force.

**7. Cold-start needs bigger data to show up.** On MovieLens-1M the content tower
shows no measurable cold-start gain, because the dataset has only 544 cold items,
most without test interactions. This is a limitation of the data, not the idea,
and it is the specific thing to re-test on MovieLens-25M.

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

Aggregate scores hide where the models act. Broken out by segment, the popularity
baseline barely covers the catalog, in-batch spreads widest but at a heavy
accuracy cost, and logQ balances the two.

![Segmented recall and coverage](docs/figures/fig2_segments.png)

Stage two reranking then reorders the top candidates and improves NDCG well beyond
seed noise.

![Reranking effect](docs/figures/fig4_rerank.png)

And approximate nearest-neighbor search recovers almost all of the exact recall at
a fraction of the latency.

![Recall-latency tradeoff](docs/figures/fig3_recall_latency.png)

Dataset profile (MovieLens-1M): 6,011 users, 3,678 items, 900k training
interactions, sparsity 0.96, popularity Gini 0.64. Full statistics are in
`docs/results/dataset_stats.json`.

## How it is built

```
user request
   |
   Stage 1  retrieval  (fetch a few hundred candidates from ~3.7k items)
     popularity baseline
     BPR matrix factorization
     two-tower with a content tower   (headline model)
   |
   Stage 2  ranking  (rerank the candidates)
     LightGBM LambdaRank
```

The two-tower has a user tower (user embedding plus a mean pool of the user's
recent item history) and an item tower (item embedding, a genre projection, and a
projection of a pretrained sentence embedding of the movie title). That content
path is what allows a brand new item with no interactions to still be represented,
which is the mechanism behind cold-start handling.

The pipeline is crash-safe and resumable by design. Every stage writes its outputs
atomically and marks completion with a `_DONE.json` file, and training checkpoints
the model, optimizer, and RNG state so a run resumes exactly where it stopped. That
property also let the whole thing run as a chain of time-limited jobs on an HPC
cluster, though none of that is needed to run it locally.

## Run it (local or Colab, no cluster needed)

```bash
git clone https://github.com/K-YLMike/recommender-system-movielens.git
cd recommender-system-movielens

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# download MovieLens once (into ./raw)
bash data/download.sh ml-1m          # or: both / ml-25m

# run the pipeline one stage at a time (each is idempotent and resumable)
python scripts/run_data_prep.py
python scripts/run_baseline_pop.py
python scripts/run_content_features.py
python scripts/run_retrieval_mf.py
python scripts/run_retrieval_twotower.py
python scripts/run_evaluate.py
python scripts/run_ranking_gbdt.py

# regenerate the figures from the results
python plots.py --results-dir ml-1m/results
```

Results are written to `ml-1m/results/`. On a CPU the MovieLens-1M run finishes in
a few minutes; only the two-tower really benefits from a GPU. With a CUDA GPU,
swap `faiss-cpu` for `faiss-gpu-cu12` and install a CUDA build of torch (see the
notes in `requirements.txt`). Everything the pipeline produces is written under the
project directory by default, which you can redirect with the `RSM_DATA_ROOT`
environment variable.

## Configuration

Every knob lives in `configs/config.yaml`. The ones worth knowing:

| key | meaning |
|---|---|
| `dataset` | `ml-1m` for development, `ml-25m` for final numbers |
| `seeds` | seeds for multi-seed averaging |
| `data.positive_rating_threshold` | rating at or above this becomes a positive (default 4.0) |
| `data.test_fraction` | most recent fraction of interactions held out for testing |
| `twotower.variants` | the negative-sampling ablation: in_batch, hard, logq |
| `twotower.total_steps`, `ckpt_every_steps` | training length and checkpoint frequency |
| `ann.ivf_nprobe`, `ann.hnsw_efsearch` | sweep points for the recall-latency curve |

## Layout

```
src/            pipeline stages and utilities
  data_prep.py          load, temporal split, slices, dataset statistics
  baseline_pop.py       popularity baseline
  content_features.py   resumable title-embedding inference
  retrieval_mf.py       BPR matrix factorization
  retrieval_twotower.py two-tower with content tower (headline)
  index_faiss.py        exact plus IVF and HNSW search, recall-latency sweep
  ranking_gbdt.py       LightGBM LambdaRank reranker
  evaluate.py           metrics, segments, multi-seed aggregation, latency
  pipeline.py           stage registry
  utils/                checkpointing, config, seeding, stats, logging
scripts/        one runnable entry point per stage
configs/        config.yaml
data/           download.sh
plots.py        build the figures from the results JSON
docs/           published results and figures
```

## What a production system would add

This is the offline skeleton, not the scale. A production version would add
streaming and near-real-time features, sharded and quantized ANN over billions of
items, multi-objective ranking that weighs engagement against diversity, freshness,
and business rules, correction for position and selection bias and delayed
feedback, online A/B testing instead of a static temporal split, and continuous
retraining.

## License

MIT. See [LICENSE](LICENSE).
