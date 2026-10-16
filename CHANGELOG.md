# Changelog

All notable changes to CVDT are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - Unreleased

### Fixed
- **Per-fold scores were never negative.** The per-fold gain was impurity
  decrease computed on the validation labels alone, which concavity makes
  `>= 0` for any split (including pure noise); the training fold only supplied
  bin edges. The `score <= min_impurity_decrease` stop (default 0) therefore
  almost never fired and trees grew to `max_depth`, with an upward bias that
  grows as nodes shrink. Fixed by honest scoring (below), now the default.
- **Single-fold candidates looked perfectly consistent.** `FoldStats.std` was
  the population std over successful folds only, so a candidate valid on one
  fold had `std = 0` and an effectively infinite `signal_to_noise`. The std is
  now Bessel-corrected (`n − 1`) and `NaN` below two folds (such candidates are
  excluded by variance-penalising aggregators), and `min_valid_folds` (default
  2) discards candidates scoring on too few folds.
- **Fast mode ignored missing values when scoring.** The histogram scorers
  (`score_classif`, `score_regr`, `score_classif_objective`) dropped samples
  with a missing feature value from that feature's fold totals, yet routed
  them right when partitioning. Scores ignored where missing samples ended up,
  and gain was normalised per *present* sample, favouring mostly-missing
  features. Missing values now count toward fold totals (hence the right
  child), matching routing and the strict path.
- **Strict mode stopped splitting when missingness was informative** (since
  0.7). With threshold splits, the strict path scored the top-bin cut (every
  present value left, missing right) but could not build a rule for it, so
  when that cut won, the node silently became a leaf; the fast path skipped
  the cut entirely, so the two paths disagreed. Both paths now offer it as the
  "x is present" split (`SplitRule::ContinuousBin` with both bounds open) when
  the node has missing values, and never otherwise.
- **Fast-mode regression lost precision on large-offset targets.** Moments
  were accumulated from raw `Σy²`, which cancels catastrophically when the
  targets' offset dwarfs their spread (y ≈ 1e9 ± 10 left no significant
  digits). `score_regr` now accumulates `y − c` with `c` a cheap node-mean
  estimate; gains are unchanged in exact arithmetic.
- **Honest scoring could pick a split that does not split.** A cut that
  sends every node sample the same way (common in fast mode below an earlier
  cut on the same feature, or a lone category) scored exactly 0 under honest
  scoring. With `min_impurity_decrease < 0` it outranked real splits scoring
  between the threshold and 0, then failed the partition check, so the node
  became a leaf. Such cuts are no longer offered on either path
  (`split_states`, `score_classif`, `score_regr`).
- **`export_graphviz` printed a literal `\n` in internal-node labels.** The
  DOT line break was escaped a second time; it now renders as a line break.
- **Unpickling an older estimator broke `get_params`.** Parameters added since
  the pickle was written were missing, so `repr`, `get_params` and `clone`
  raised `AttributeError`. They are now filled with the values that reproduce
  the fitted behaviour (`split_style="bin"` before 0.7, `scoring="in_sample"`
  and `min_valid_folds=1` before 0.8), else their defaults.

### Added
- `Scoring::{Honest, InSample}` / `scoring="honest" | "in_sample"`. Honest
  scoring fits parent/child models on the training fold and measures held-out
  Brier / log-loss / squared / absolute error. On the fast path the training
  counts come from `node total − fold`, so it costs no extra data pass.
- `Criterion::supports_heldout` / `Criterion::heldout_loss`, plus
  `brier_loss_counts`, `log_loss_counts`, `sq_loss_moments`.
- `split_states`: the candidate states the strict path scores (adds the
  presence cut only when the node has missing values, and drops a lone state
  that every node sample matches). Tree exports show the
  presence split as `lower = -inf, upper = +inf`; `export_text` /
  `export_graphviz` render it as `x is present`.
- Repeated K-fold: `cv_repeats`.
- Aggregators: `Quantile`, `ModeratedSignalToNoise`,
  `ModeratedMeanMinusLambdaStd`, and the node-level
  `Aggregator::aggregate_node` hook (default: per-candidate `aggregate`).
- Permutation-null stopping test: `null_permutations`, `null_quantile`.
- Hierarchical shrinkage of predictions: `shrinkage` (`Task::shrink`).
- Python: all of the above as estimator parameters (`scoring`, `cv_repeats`,
  `min_valid_folds`, `null_permutations`, `null_quantile`, `shrinkage`,
  `agg_q`, `agg_d0`), passed to the native core through an optional `extra`
  dict so the positional constructor signatures are unchanged.

### Changed
- **Default behaviour changed:** honest scoring is the default, so split
  scores and tree shapes differ from 0.7. Use `Scoring::InSample` /
  `scoring="in_sample"` to reproduce 0.7 split scores. Honest scoring removes
  the per-candidate upward bias but not the selection effect of keeping the
  best of many candidates, so on pure noise it does **not** reliably give
  smaller trees: the effect is strongest at tiny nodes, where in-sample
  scoring had an implicit floor (a fold with an empty validation child did
  not count) and honest scoring has none, so strict-mode noise trees can even
  be larger. Use `null_permutations` (the stopping rule that accounts for the
  search) and/or `min_samples_leaf` to stop noise splits.
- Documentation: `PYTHON.md` and `API_REFERENCE.md` cover all 0.8 parameters
  and signatures; the `Quantile` aggregator and `cv_repeats` are no longer
  described as giving confidence bounds / independent scores (K-fold scores
  are correlated); the held-out losses are documented as equal to
  `n · impurity` in-sample exactly for Gini, squared and absolute error, and
  only approximately for entropy (log-loss smoothing).
- **Default `min_samples_leaf` is now 30** (was 1), so every leaf estimate
  rests on at least 30 samples and nodes under 60 samples are not split. This
  is what stops honest scoring from chasing noise at tiny nodes (on pure noise
  it cut strict-mode trees from ~80 leaves to under 10). Set
  `min_samples_leaf=1` to recover 0.7 growth, e.g. on small datasets. With
  `split_style="bin"` a child is a single quantile bin, so the default needs
  `n / n_bins > 30` at a node before any bin-membership split is possible.
- **`min_samples_leaf` now restricts which split is chosen.** Candidates whose
  node-level partition would leave a smaller child are excluded before
  selection. Previously only the winning split was checked, so a node became a
  leaf whenever its best candidate was too lopsided, even if an admissible
  split existed. This changes trees for any `min_samples_leaf > 1`.
- `FoldStats.std` is the sample (not population) std; it is `NaN` with fewer
  than two folds.
- `score_classif`, `score_regr`, `eval_feature`, and
  `Task::score_feature_{fast,strict}` take a trailing `scoring: Scoring`
  argument.

## [0.7.0] - 2026-07-04

### Added
- **CART-style ordered threshold splits for continuous features**, selected via
  the new `split_style` parameter (`SplitStyle` in the Rust core). A continuous
  split is now a single ordered cut `x < edge` — where `edge` is one of the
  quantile bin boundaries — instead of a one-bin-vs-rest membership test. The
  cut is still scored by K-fold cross-validation on held-out folds; only the
  partition geometry changes. This respects the natural ordering of continuous
  variables and generally generalises better. Available on both the `strict`
  and `fast` (histogram) paths and for impurity- and objective-driven splits.
  - Python: `CVDTClassifier(split_style="threshold")` /
    `CVDTRegressor(split_style="threshold")`. Accepts `"threshold"` (default),
    `"bin"`, or the alias `"bin_membership"`.
  - Rust: `TreeParams.split_style: SplitStyle::{Threshold, BinMembership}`.
  - `n_bins` now doubles as the threshold resolution: it sets the number of
    candidate cut points (`n_bins - 1`) for a continuous feature.

### Changed
- **Default behaviour changed:** `split_style` defaults to `"threshold"`
  (`SplitStyle::Threshold`), so continuous features are now cut as `x < edge`
  out of the box. To reproduce pre-0.7.0 (v1/v2) behaviour exactly, pass
  `split_style="bin"` (Python) / `SplitStyle::BinMembership` (Rust).
- Tree exports render continuous threshold branches as `x[f] < edge` (open
  lower bound / `lower = -inf`); single-bin branches still render as the
  half-open interval `lo <= x[f] < hi`.
- Documentation (`README.md`, `PYTHON.md`, `API_REFERENCE.md`, crate-level docs)
  updated to describe both split styles.

### Fixed
- `python/cvdt/__init__.py` `__version__` was stale at `0.5.0`; it is now kept
  in step with the packaged version.

[0.8.0]: https://github.com/AdventuresInDataScience/CVDT/releases/tag/v0.8.0
[0.7.0]: https://github.com/AdventuresInDataScience/CVDT/releases/tag/v0.7.0
