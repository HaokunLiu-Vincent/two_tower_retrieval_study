# Two-Tower Recommendation: A Statistical Machine Learning Case Study

[Notebook](two_tower_movielens.ipynb) · [Dataset](#dataset) · [Results](#experimental-results) · [How to run](#how-to-run) · [Limitations](#limitations-and-future-work)

## Project Overview

A Two-Tower recommender independently learns user and item representations, then scores their compatibility with a dot product. This project uses PyTorch and MovieLens Latest Small to investigate the relationship between pointwise binary cross-entropy (BCE), generalization, and Top-K retrieval performance.

This is an educational statistical machine learning case study, rather than a production recommendation system. The notebook contains data preparation, training, saved metrics and plots, and example movie recommendations.

## Research Question

**Does continuously minimizing BCE loss necessarily improve Top-K retrieval performance?**

The experiment tracks training BCE, validation BCE, Recall@20, and NDCG@20 across ten epochs. Its saved results show that the best validation BCE and the best retrieval performance occur at different epochs.

## Dataset

Explicit ratings from **MovieLens Latest Small** are converted into implicit feedback: ratings **≥ 4.0** are positives; lower ratings are not directly labeled as negatives. Counts below come from the notebook's saved outputs.

| Quantity | Count |
| --- | ---: |
| Raw ratings / users | 100,836 / 610 |
| Movies in `movies.csv` / movies appearing in ratings | 9,742 / 9,724 |
| Positive interactions | 48,580 |
| Users / movies with at least one positive | 609 / 6,298 |
| Encoded user IDs / movie IDs | 609 / 9,724 |
| Users retained for splitting | 608 |
| Train / validation / test positives | 47,363 / 608 / 608 |
| Training examples: positive + sampled negative | 236,815 = 47,363 + 189,452 |
| Validation BCE examples: positive + sampled negative | 3,040 = 608 + 2,432 |

**Split:** Sort each user's positives by timestamp; reserve the latest for test, the second latest for validation, and all earlier positives for training. Users with fewer than three positives are skipped. One encoded user is excluded, accounting for the difference between 48,580 positives before filtering and 48,579 across the splits. This is a per-user chronological split, not a shared global time cutoff.

**Negative sampling:** Each training positive receives four movies sampled uniformly from movies the user has never rated, giving a **1:4 positive-to-negative ratio**. Sampling excludes all of that user's ratings, including held-out interactions and ratings below 4.0. Four negatives are distinct within each draw, but can repeat across draws. Training negatives are generated once and reused across epochs. Validation BCE uses a separate fixed sample of four negatives per positive, with sampling seed `123`.

## Model Architecture

The User Tower and Item Tower have independent parameters:

```text
User ID  → trainable 32-D embedding → Linear(32, 64) → ReLU → Linear(64, 32) → z_u
Movie ID → trainable 32-D embedding → Linear(32, 64) → ReLU → Linear(64, 32) → z_i
                                      Compatibility logit: s(u, i) = z_uᵀ z_i
```

Each tower therefore uses a **32 → 64 → 32 MLP**, with ReLU between the linear layers. The model returns raw logits; `BCEWithLogitsLoss` incorporates the sigmoid transformation internally. Retrieval ranks by the raw dot product.

Training uses **Adam**, learning rate **0.001**, batch size **512**, and **10 epochs**. The main random seed is **42**. Inputs contain IDs only; movie titles are used for displaying recommendations, not as model features.

## Statistical Machine Learning Framework

- **Hypothesis family $H$:** Two-Tower scoring functions $h_\theta(u,i)=f_{\theta_u}(u)^\top g_{\theta_i}(i)$, with learnable ID embeddings and the MLPs above.
- **Loss function $L$:** Pointwise BCE on a logit $s$ and label $y\in\{0,1\}$, $L(s,y)=\log(1+e^s)-ys$, implemented numerically stably by `BCEWithLogitsLoss`.
- **Learning algorithm $A$:** Shuffled minibatch training with Adam, followed by checkpoint selection using validation Recall@20.
- **Empirical training risk $E_{\mathrm{in}}$** and **estimated out-of-sample risk** on the fixed validation sample:

$$
E_{\mathrm{in}}(h)=\frac{1}{N_{\mathrm{train}}}\sum_{(u,i,y)\in D_{\mathrm{train}}}L(h(u,i),y),
\qquad
\widehat E_{\mathrm{out}}^{\mathrm{val}}(h)=\frac{1}{N_{\mathrm{val}}}\sum_{(u,i,y)\in D_{\mathrm{val}}}L(h(u,i),y).
$$

The logged training BCE averages minibatch losses **while parameters are being updated**, so it approximates training risk rather than measuring the final epoch checkpoint on the entire training set. Validation BCE is measured after each epoch on fixed parameters. It estimates loss under this sampled validation protocol, not necessarily deployment risk; repeated checkpoint selection also uses this validation set.

BCE measures prediction quality on individual labeled pairs. Retrieval metrics measure whether a held-out positive ranks highly among thousands of candidates. Lower pointwise loss need not improve that ranking.

## Evaluation Methodology

1. Measure **validation BCE** on the fixed 3,040-example sample after every epoch.
2. For retrieval, score all **9,724 encoded movies** per user and mask that user's **training positives only**. This is not evaluation against just four sampled negatives.
3. Compute **Recall@20** as the primary metric and **NDCG@20** as the secondary metric, averaged over 608 users. With one target per user, Recall@20 equals the hit rate. NDCG@20 is $1/\log_2(r+1)$ if the target appears at rank $r\leq20$, and zero otherwise.
4. Retain a deep copy of the weights with the highest validation Recall@20, then restore it for final held-out test retrieval evaluation. Test metrics do not determine the checkpoint.

Test evaluation also masks only training positives; validation positives remain candidates. The later example recommendation function instead excludes **all** movies the user has rated. The selected checkpoint is held in memory; the notebook does not export a weights file.

## Experimental Results

These are the **existing saved results**, rounded to six decimal places. No additional experiments or notebook execution were performed while preparing this README.

| Checkpoint | Epoch | Train BCE | Validation BCE | Validation Recall@20 | Validation NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial epoch | 1 | 0.462547 | 0.423010 | 0.034539 | 0.013370 |
| Best validation Recall@20; selected | **3** | 0.300536 | 0.357224 | **0.059211** | **0.024523** |
| Lowest validation BCE | **4** | 0.284734 | **0.344411** | 0.047697 | 0.018594 |
| Final training epoch | 10 | 0.256813 | 0.363910 | 0.042763 | 0.021003 |

The restored **epoch 3** model achieves held-out **test Recall@20 = 0.054276** and **test NDCG@20 = 0.023506**. Test BCE is not computed or reported by the notebook.

**Observed:** Training BCE decreases every epoch. From epoch 3 to 4, validation BCE improves while both retrieval metrics worsen. After epoch 4, validation BCE stays above its minimum despite continued training-loss reduction. By epoch 10, validation BCE exceeds logged training BCE by approximately **0.107097**.

**Interpretation:** The widening loss gap is consistent with overfitting, subject to the different training and validation measurement procedures. Different optimal epochs suggest a potential mismatch between the sampled pointwise objective and full-catalog retrieval. This single run illustrates that lower BCE does not guarantee better Top-K performance; it does not establish a general causal conclusion or a benchmark advantage.

## How to Run

1. Open [Google Colab](https://colab.research.google.com/) and upload [two_tower_movielens.ipynb](two_tower_movielens.ipynb) using **File → Upload notebook**, as described in the [Colab FAQ](https://research.google.com/colaboratory/faq.html). Work in a copy to retain the published outputs.
2. Use a Python 3 runtime. Optionally select a GPU through **Runtime → Change runtime type**; the code automatically uses CUDA when available and otherwise uses CPU. The saved run used CUDA.
3. Required third-party libraries, based on the actual imports, are **NumPy, pandas, PyTorch, tqdm, and Matplotlib**. If any are missing, install them in a temporary setup cell in your copy: `%pip install numpy pandas torch tqdm matplotlib`. Other imports (`os`, `random`, `zipfile`, `urllib.request`, `copy`) are from Python's standard library; `display()` is supplied by the notebook environment.
4. Run the cells in order, starting with environment setup. The notebook downloads the [GroupLens ZIP archive](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip) to `/content/ml-latest-small.zip` if absent, extracts it under `/content`, and reads `ratings.csv` and `movies.csv`. It then encodes IDs, splits positives, samples negatives, trains, plots metrics, and evaluates the selected model.

The `/content` paths target Colab. Seeds are set, but library versions and deterministic GPU execution are not pinned, so a new run may differ from the saved results. The dataset download is not checksum-pinned; GroupLens notes that the [Latest datasets can change](https://grouplens.org/datasets/movielens/latest/). No repository-specific Colab badge is included because no Git remote is configured in the supplied project.

## Limitations and Future Work

- **Single training run:** Main seed `42` and fixed validation-sampling seed `123`; no repeated runs, uncertainty estimates, or seed-sensitivity study.
- **Unobserved is not disliked:** Sampled negatives are never-rated movies, not verified dislikes. Fixed samples can repeat across training draws and overlap validation negatives, limiting independence and coverage.
- **Cold start:** ID-only embeddings cannot represent previously unseen users or items; no content or contextual features are modeled.
- **Candidate filtering:** Retrieval excludes training positives only, leaving previously rated lower-score movies and other held-out positives as competitors. Its filtering differs from the example recommendation function.
- **Temporal limitations:** Splitting is per-user, with no global cutoff or explicit timestamp tie-breaker. Movie vocabulary and negative exclusion use full-dataset information, including future ratings; candidates are not restricted by movie availability at the prediction time. This is not a strictly historical deployment simulation.
- **Scope:** One held-out positive per user and a sampled BCE distribution provide a limited view of recommendation quality. No comparative baselines or production serving system are evaluated.

Future experiments could repeat multiple seeds, audit temporal and sampling protocols, compare ranking-oriented losses such as BPR or contrastive objectives, and add popularity, matrix-factorization, and stronger retrieval baselines. These are proposed experiments, not demonstrated improvements.

## Acknowledgements

Credit to **GroupLens Research** and **MovieLens** for the [MovieLens Latest Small dataset](https://grouplens.org/datasets/movielens/latest/). This project serves academic and educational study of statistical learning and recommendation evaluation.

AI assistance was used to prepare this README, with technical descriptions and numbers checked against the existing notebook. This acknowledgement describes documentation preparation and makes no claim about how the original implementation was authored.
