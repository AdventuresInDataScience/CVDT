# Roadmap

## Current status (v0.8.0)

The 0.8.0 changes were written and logic-reviewed but have **not yet been
compiled, tested or benchmarked**. They were developed without a Rust
toolchain, and the maths was checked with a Python port of the scoring
functions. See `CHANGELOG.md` for the full list of 0.8.0 changes.

A literature search (summarised under "Prior art and novelty" below) found
that the headline idea, cross-validated split selection at every node shown
to beat ctree, **is already published**. The next steps are ordered so that
the remaining, narrower novelty claim is tested cheaply before any slow
benchmarking is done.

## Next steps, in order

### Step 1: compile and test

- `cargo build` and `cargo test`, fixing any compile errors first.
- `cargo clippy` for warnings.
- Build the Python extension and run `pytest tests/test_sklearn.py`.
- Spot-check determinism: refit with the same seed, and compare serial and
  parallel fits.

### Step 2: pilot study (go / no-go for publication)

The pilot answers one question: **does consistency-aware split selection beat
plain mean cross-validated selection on low-SNR data?** Plain mean selection
is the part covered by prior art (ALOOF, RobustCART), so this is the novelty
claim that has to hold.

CVDT can run both sides of the comparison:

- **Prior-art baseline:** `scoring="honest"` with `aggregator="mean"`. This is
  K-fold, ALOOF-style held-out selection, with no reward for consistency.
- **Novel variants:**
  - `aggregator="quantile"` (try `agg_q` of 0.1 and 0.25)
  - `aggregator="moderated_signal_to_noise"`
  - `aggregator="moderated_mean_minus_lambda_std"`
  - each with `cv_repeats` of 1 and 5

Use a handful of synthetic settings: a few informative features plus many
noise features, two or three SNR levels and two sample sizes. Hold
`null_permutations`, `shrinkage` and `min_samples_leaf` fixed across arms so
that only the aggregator differs.

**Decision rule.** Fix this before running the pilot:

- **Go:** at least one consistency-aware variant clearly beats the mean
  baseline on test error at equal or smaller tree size, in the low-SNR
  settings, consistently across seeds. Proceed to Step 3.
- **No-go:** no clear gain. The remaining methods novelty is weak; consider a
  software paper only (see "Publication options"), or stop.

### Step 3: full benchmark (only after a "go")

**Baselines, all required:**

- CART with `ccp_alpha` tuned by CV.
- ctree (R partykit). Export the same train/test splits to CSV, fit ctree in a
  short R script, write predictions back to CSV, and score them in the Python
  benchmark. `rpy2` is not needed.
- **ALOOF** (Painsky and Rosset), the closest prior art. No implementation has
  been located yet. Options: find the authors' code, implement it (variable
  chosen by leave-one-out selection, then an ordinary CART split on that
  variable), or, as a weaker substitute, use CVDT's honest scoring with the
  mean aggregator and say so explicitly.
- CVDT with `scoring="in_sample"` (the 0.7 behaviour), as an ablation.

**CVDT configurations.** Add one feature at a time, so each one's contribution
is visible:

1. Honest scoring with the mean aggregator.
2. Swap in the best consistency-aware aggregator from the pilot.
3. Add `null_permutations=19`.
4. Add `shrinkage`, tuned by CV.
5. Vary `min_samples_leaf`. The default is now 30; compare 1, 10 and 30.

**Data.**

- **Synthetic low-SNR problems:** a range of SNRs and sample sizes, plus a
  variant with informative missing values.
- **Planned divergence cases**, constructed so each method's selection rule
  matters:
  - *Large n, tiny real effect.* The effect becomes highly significant, so
    ctree splits on it, but it barely improves held-out loss, so CVDT should
    not.
  - *Unstable winner.* One split has a slightly higher mean gain but a high
    variance across folds; another is nearly as good on average and
    consistent. Plain mean selection (ALOOF-style) picks the first, a
    consistency-aware aggregator the second. This is the case that tests
    CVDT's own contribution.
- **Real data:** a few low-SNR datasets.

**Metrics.**

- Test accuracy or MSE.
- Tree size: leaves and depth.
- Recovery of the true informative features.
- Structural stability across resamples, for example the Jaccard overlap of
  split sets, as in the RobustCART post.
- Behaviour as `max_depth` grows: the original failure mode.
- Fit time, especially for `null_permutations` and `cv_repeats`.

Report the cases where a baseline wins as well as those where CVDT wins.

## What exactly to target

**Do not claim** that cross-validated split selection is new, or that beating
ctree with it is new. Both are published (ALOOF).

**Target claim**, to be supported or rejected by Steps 2 and 3:

> On low signal-to-noise problems, choosing each split by the *consistency* of
> its cross-validated held-out gain (mean penalised by spread, a lower
> quantile, or a variance-moderated signal-to-noise ratio), combined with a
> permutation null on that same score as the stopping rule, gives lower test
> error at equal or smaller tree size, better recovery of true features and
> more stable trees than mean cross-validated selection (ALOOF-style), ctree
> and tuned CART.

**Claimed contributions**, in order of strength. None turned up in the search,
but that is not proof; re-check before writing.

1. **Consistency-aware aggregation of per-fold held-out gains** as the split
   criterion. ALOOF sums held-out loss and RobustCART averages it; neither
   rewards consistency.
2. **Empirical-Bayes moderation of fold-score variance across the candidate
   splits at a node** (limma-style). No precedent was found in tree
   induction.
3. **A permutation null built from the cross-validated selection statistic
   itself**, maximised over all candidates. Jensen and Cohen proposed
   randomization testing as a remedy in general, so present this as a new
   combination, not a new idea.
4. **Fast honest K-fold scoring**, with training-fold counts recovered by
   subtraction so there is no extra pass over the data. An engineering
   contribution.

## Publication options

- **Methods paper**, only if Step 2 is a "go" and Step 3 confirms it. Frame it
  as an incremental improvement on ALOOF-style selection for low-SNR data, with
  ALOOF as the primary baseline. A strong result fits a statistics or applied
  machine-learning journal; a top venue is unlikely unless the gains are
  large.
- **Software paper** (for example JOSS) for the CVDT package itself. This does
  not depend on methods novelty, but it does need a working, tested,
  documented release, which means Step 1.

## Prior art and novelty

Found in a literature search in October 2026. Links were not resolved through
the institutional LibKey connector, which was not enabled at the time.

### Cross-validated split selection (covers the core idea)

- **Painsky, A. and Rosset, S. "Cross-Validated Variable Selection in
  Tree-Based Methods Improves Predictive Performance." IEEE TPAMI; arXiv
  1512.03444 (2015).** "ALOOF".
  - Chooses the splitting variable at each node by leave-one-out
    cross-validation, then makes an ordinary CART split on that variable.
  - Uses the same estimates as the stopping rule instead of pruning.
  - Argues ctree's significance criterion is the wrong target for prediction,
    and reports beating ctree.
  - Notes that taking the best of many nearly unbiased estimates can still be
    over-optimistic, which is the gap CVDT's permutation null addresses.
  - Follow-up work by the same group applies cross-validated selection to
    multi-target trees and gradient-boosting feature importance.
- **Kohavi, R. "Scaling Up the Accuracy of Naive-Bayes Classifiers: A
  Decision-Tree Hybrid." KDD 1996.** NBTree chooses the split attribute at each
  node by cross-validated accuracy, and makes the node a leaf if no split
  beats it.
- **Frank, E. and Witten, I. H. (1996), and Sabato, S. and Shalev-Shwartz, S.
  (JMLR 2008).** Cross-validation and generalization-based selection of
  multiway splits.
- **"RobustCART"** (G. Sood, blog post with code, July 2025;
  github.com/finite-sample/robust-cart). At each node, draws many subsamples,
  fits a one-split model on each, scores it by held-out loss, and keeps the
  split with the best average loss.
- **ZTree** (Cheng et al., arXiv 2509.12688, 2025). Uses a cross-validated test
  statistic to decide whether a split is warranted.

### Multiple comparisons and permutation tests

- **Jensen, D. D. and Cohen, P. R. "Multiple Comparisons in Induction
  Algorithms." Machine Learning 38, 309–338 (2000);
  doi:10.1023/A:1007631014630.** Taking the maximum score over many candidates
  causes overfitting and attribute-selection errors; names randomization
  testing and cross-validation as remedies.
- **Frank, E. and Witten, I. H. "Using a Permutation Test for Attribute
  Selection in Decision Trees." ICML 1998.**
- **Hothorn, T., Hornik, K. and Zeileis, A. "Unbiased Recursive Partitioning:
  A Conditional Inference Framework." JCGS 15(3), 2006.** ctree.
- **Strobl, C. et al. "Unbiased split selection for classification trees based
  on the Gini Index." Computational Statistics & Data Analysis (2007).** Exact
  distribution of the maximally selected Gini gain, used as a p-value
  criterion.

### Selection frequency and stability

- **Dannegger (2000), bootstrap node-level stabilisation (survival trees).** At
  each node, picks the split that wins most often across bootstrap samples.
- **Lee, J., Sim, M. K. and Hong, J.-S. "Assessing Decision Tree Stability."
  IEEE Access (2024).** Adaptive node-level stabilisation by the most
  frequently selected predictor across bootstraps.
- **Bertsimas, D. and Digalakis, V. "Improving Stability in Decision Tree
  Models" (arXiv 2305.17299, 2023).**

### Honest estimation, held-out impurity, shrinkage

- **Athey and Imbens (2016); Wager and Athey (2018).** Honest trees, where one
  sample chooses the splits and another estimates the leaf values.
- **Li et al. (2019); Zhou and Hooker (2021).** Out-of-bag and debiased
  impurity-based feature importance.
- **Agarwal et al. (2022).** Hierarchical shrinkage, used as is in CVDT.

## Further update path, only if the results call for it

These were discussed during the 0.8 design and deliberately deferred. Several
now double as baselines, because they have prior art.

### Bootstrap-calibrated split uncertainty (supports contribution 1)

Repeated K-fold (`cv_repeats`) gives a spread of fold scores. Because the
folds overlap, that spread understates the uncertainty of the mean gain. A
bootstrap estimates the gain's sampling variability directly, which would
make the consistency-aware criteria better founded. Worth doing if the pilot
shows consistency-aware selection helps.

- **Table-level bootstrap (cheap).** For a fixed candidate on a fold, the
  honest classification score depends only on class counts: training and
  validation, left and right child. Resampling those count tables with
  multinomial draws (or Dirichlet draws, for a Bayesian bootstrap) gives a
  bootstrap distribution of the honest gain at O(B·C) cost per candidate, with
  no extra pass over the data. For regression, use a normal or delta-method
  approximation of the child moments.
  - Caveat: each candidate is resampled independently, so this captures that
    candidate's uncertainty but not the selection effect across candidates.
- **Joint Poisson bootstrap with out-of-bag scoring.** In each replicate, give
  every sample a Poisson(1) weight, fit on the weighted in-bag samples and
  score on the out-of-bag samples. Every candidate is scored on the same
  replicate, which keeps honest scoring and adds calibrated spread. In fast
  mode this costs one weighted histogram pass per replicate.

### Selection frequency (prior art; implement as a baseline)

Choose the split that wins most often across replicates. This is Dannegger's
method and Lee et al.'s, so it is not a contribution, but it is a natural
competitor for the consistency claim. It requires the joint bootstrap,
because candidates must share resamples.

### Nested selection (prior art; implement as a baseline)

In each fold, let the training part choose its best split and the held-out
part score that choice. This is essentially ALOOF's approach (with K folds
instead of leave-one-out), so implementing it inside CVDT would give a
faithful ALOOF-style baseline for Step 3.

### Other

The existing items in `README.md` ("Status and future work") still apply:
parent histogram subtraction, sample weighting, feature importance, pruning,
and forest or boosting variants.
