"""scikit-learn-compatible estimators wrapping the native CVDT core.

These classes implement the estimator contract so that CVDT drops into the
sklearn ecosystem: ``Pipeline``, ``cross_val_score``, ``GridSearchCV``,
``clone``, ``get_params``/``set_params`` all work. The heavy lifting is done
by the Rust extension ``cvdt._cvdt``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import validation as _skl_validation
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._cvdt import RawClassifier, RawObjectiveClassifier, RawRegressor

# ---------------------------------------------------------------------------
# Version-compatibility shims
#
# sklearn renamed ``force_all_finite`` -> ``ensure_all_finite`` in 1.6. CVDT
# routes NaN/non-finite continuous values to the right child on purpose, so we
# must allow non-finite input. Try the new keyword first, fall back to the old.
#
# sklearn also moved the ``_check_feature_names`` / ``_check_n_features`` helpers
# off ``BaseEstimator`` (instance methods, <= ~1.5) to module-level functions in
# ``sklearn.utils.validation`` (>= ~1.6, and the instance methods are gone in
# 1.9). Prefer the instance method when present, else the module function.
# ---------------------------------------------------------------------------

def _record_feature_names(est, X, reset):
    method = getattr(est, "_check_feature_names", None)
    if method is not None:
        method(X, reset=reset)
    else:
        _skl_validation._check_feature_names(est, X, reset=reset)


def _record_n_features(est, X, reset):
    method = getattr(est, "_check_n_features", None)
    if method is not None:
        method(X, reset=reset)
    else:
        _skl_validation._check_n_features(est, X, reset=reset)


# Pick the "allow non-finite" keyword once, by inspecting the signature, rather
# than catching TypeError from the call itself — a data-driven TypeError raised
# *inside* check_array (e.g. non-numeric input) must propagate unmasked.
def _finite_kwarg(func):
    import inspect

    if "ensure_all_finite" in inspect.signature(func).parameters:
        return {"ensure_all_finite": False}
    return {"force_all_finite": False}


_XY_FINITE = _finite_kwarg(check_X_y)
_ARR_FINITE = _finite_kwarg(check_array)


def _check_X_y(X, y):
    return check_X_y(X, y, dtype=np.float64, **_XY_FINITE)


def _check_array(X):
    return check_array(X, dtype=np.float64, **_ARR_FINITE)


def _normalise_categorical(categorical_features, n_features):
    """Validate + sort the categorical column indices."""
    if categorical_features is None:
        return []
    idx = sorted(int(c) for c in categorical_features)
    for c in idx:
        if c < 0 or c >= n_features:
            raise ValueError(
                f"categorical feature index {c} out of range for {n_features} features"
            )
    return idx


_AGGREGATORS = {
    "mean",
    "median",
    "trimmed_mean",
    "signal_to_noise",
    "mean_minus_lambda_std",
    "quantile",
    "moderated_signal_to_noise",
    "moderated_mean_minus_lambda_std",
}

_SCORINGS = {"honest", "in_sample"}

# Values that reproduce older behaviour, used when unpickling an estimator saved
# before the parameter existed. Parameters not listed here did not change
# behaviour when they were introduced, so their current default is used.
_LEGACY_PARAMS = {
    "split_style": "bin",  # added in 0.7; earlier trees used single-bin splits
    "scoring": "in_sample",  # added in 0.8; earlier fold scores were in-sample
    "min_valid_folds": 1,  # added in 0.8; earlier, one successful fold sufficed
}

_OBJECTIVES = {"precision", "recall", "f1", "fbeta", "accuracy"}
_AVERAGES = {"binary", "micro", "macro", "weighted"}


class _CVDTBase(BaseEstimator):
    """Shared plumbing for the CVDT estimators.

    Note: this base intentionally defines **no** ``__init__`` with parameters.
    scikit-learn discovers hyper-parameters by introspecting the *concrete*
    class's ``__init__`` and ignores ``**kwargs``, so each estimator lists every
    parameter explicitly (and stores it verbatim, unmodified) — that is what
    makes ``get_params``/``set_params``/``clone``/``GridSearchCV`` work.

    split_style : {"threshold", "bin"}, default="threshold"
        How a continuous feature is partitioned at a split (categorical features
        are always tested by category equality).

        - ``"threshold"`` (default) — a CART-style *ordered* cut: the left child
          is ``x < edge`` and the right child is everything else, where ``edge``
          is one of the ``n_bins - 1`` quantile boundaries. This respects the
          natural ordering of a continuous variable, so the tree can recursively
          narrow a range; increase ``n_bins`` for finer thresholds. When a node
          has missing values, one more cut is offered: "x is present" (every
          present value left, missing right).
        - ``"bin"`` — the original CVDT behaviour: a single quantile bin versus
          all others (``x in [lo, hi)``). Kept for comparison; ``"bin_membership"``
          is an accepted alias.

        Both styles are scored by cross-validation on held-out folds, and
        missing / non-finite values always route to the right child.

    scoring : {"honest", "in_sample"}, default="honest"
        ``"honest"`` fits the parent/child models (class proportions, means,
        medians) on each training fold and measures their loss (Brier,
        log-loss, squared / absolute error) on the held-out fold, so a split
        with no signal scores <= 0 in expectation. The tree keeps the best of
        many candidates, though, and that maximum is usually positive even on
        pure noise, so on its own this does not stop noise splits (least of
        all at small nodes); use ``null_permutations`` and/or
        ``min_samples_leaf`` for that. ``"in_sample"`` reproduces
        CVDT <= 0.7, where the per-fold gain is impurity decrease on the
        validation labels alone (never negative, biased upward on small
        nodes). Ignored by objective mode, which is already honest.
    cv_repeats : int, default=1
        Repeated K-fold: independently shuffled K-fold rounds per node, pooled
        into ``cv_folds * cv_repeats`` scores per candidate. Stabilises the
        spread estimate used by variance-aware aggregators (the pooled scores
        reuse the same samples, so they are correlated). Cost is linear.
    min_valid_folds : int, default=2
        Candidates scoring on fewer folds than this are discarded.
    null_permutations : int, default=0
        If > 0, a permutation-null stopping test at every node: targets are
        permuted this many times, all candidates rescored, and a split is
        accepted only if its score beats the ``null_quantile`` quantile of the
        permuted best scores. Adapts to node size and the number of candidates
        searched, so unlike honest scoring alone it accounts for the search;
        19 is a reasonable start. Cost: ``null_permutations`` extra scorings at
        each node whose best score passes ``min_impurity_decrease``.
    null_quantile : float, default=0.95
        Quantile of the permutation null the best split must beat.
    shrinkage : float, default=0.0
        Hierarchical-shrinkage strength (Agarwal et al., 2022): each node's
        deviation from its parent is weighted by ``1 / (1 + shrinkage / n_parent)``.
        Leaves the tree structure unchanged; 0 disables it.
    agg_q : float, default=0.25
        Quantile for ``aggregator="quantile"``: a quantile of the per-fold
        scores (risk-averse), not a confidence bound on the mean gain.
    min_samples_leaf : int, default=30
        Minimum samples in each child. Candidate splits that would leave a
        smaller child are excluded before the best one is chosen. The default
        follows the usual n >= 30 rule of thumb for a stable mean or
        proportion and stops honest scoring from chasing noise at tiny nodes;
        it also means a node needs at least 60 samples to split, so lower it
        (e.g. to 5 or 1) for small datasets. With ``split_style="bin"`` a
        child is a single quantile bin, so a node of ``n`` samples can only
        split if ``n / n_bins`` exceeds ``min_samples_leaf``.
    agg_d0 : float, default=4.0
        Prior weight (in fold-equivalents) for the ``moderated_*`` aggregators,
        which shrink each candidate's fold-score variance toward the median
        variance of all candidates at the node.
    """

    # -- shared param validation ------------------------------------------
    def _validate_common(self):
        if self.aggregator not in _AGGREGATORS:
            raise ValueError(
                f"aggregator must be one of {sorted(_AGGREGATORS)}, got {self.aggregator!r}"
            )
        if self.mode not in ("strict", "fast"):
            raise ValueError(f"mode must be 'strict' or 'fast', got {self.mode!r}")
        if self.split_style not in ("threshold", "bin", "bin_membership"):
            raise ValueError(
                "split_style must be 'threshold' (CART-style ordered cut) or "
                f"'bin' (single quantile bin vs rest), got {self.split_style!r}"
            )
        if self.n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")
        if self.scoring not in _SCORINGS:
            raise ValueError(
                f"scoring must be one of {sorted(_SCORINGS)}, got {self.scoring!r}"
            )
        if self.cv_repeats < 1:
            raise ValueError("cv_repeats must be >= 1")
        if self.min_valid_folds < 1:
            raise ValueError("min_valid_folds must be >= 1")
        if self.null_permutations < 0:
            raise ValueError("null_permutations must be >= 0")
        if not (0.0 <= self.null_quantile <= 1.0):
            raise ValueError("null_quantile must be in [0, 1]")
        if self.shrinkage < 0:
            raise ValueError("shrinkage must be >= 0")
        if not (0.0 <= self.agg_q <= 1.0):
            raise ValueError("agg_q must be in [0, 1]")
        if self.agg_d0 < 0:
            raise ValueError("agg_d0 must be >= 0")

    def _common_kwargs(self, categorical):
        return dict(
            max_depth=self.max_depth,
            min_samples_split=int(self.min_samples_split),
            min_samples_leaf=int(self.min_samples_leaf),
            min_impurity_decrease=float(self.min_impurity_decrease),
            n_bins=int(self.n_bins),
            cv_folds=int(self.cv_folds),
            cv_seed=int(self.cv_seed),
            cv_shuffle=bool(self.cv_shuffle),
            mode=str(self.mode),
            split_style=str(self.split_style),
            aggregator=str(self.aggregator),
            agg_frac=float(self.agg_frac),
            agg_eps=float(self.agg_eps),
            agg_lambda=float(self.agg_lambda),
            parallel=bool(self.parallel),
            n_threads=int(self.n_threads),
            parallel_min_samples=int(self.parallel_min_samples),
            categorical=categorical,
            extra=dict(
                scoring=str(self.scoring),
                cv_repeats=int(self.cv_repeats),
                min_valid_folds=int(self.min_valid_folds),
                null_permutations=int(self.null_permutations),
                null_quantile=float(self.null_quantile),
                shrinkage=float(self.shrinkage),
                agg_q=float(self.agg_q),
                agg_d0=float(self.agg_d0),
            ),
        )

    # -- tags (new + legacy APIs) -----------------------------------------
    def __sklearn_tags__(self):
        # sklearn >= 1.6
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _more_tags(self):
        # sklearn < 1.6
        return {"allow_nan": True, "requires_y": True}

    # -- pickling ----------------------------------------------------------
    # The fitted model lives in the Rust extension object ``self._model``, which
    # is not directly picklable. We instead persist its tree as the plain-array
    # dict returned by ``export_tree`` (fully picklable) and rebuild the native
    # model on load. Everything else (params, ``classes_``, ``n_features_in_``,
    # ...) pickles normally via the estimator's ``__dict__``.
    def __getstate__(self):
        state = super().__getstate__()
        state = dict(state) if state is not None else self.__dict__.copy()
        model = state.pop("_model", None)
        if model is not None:
            state["__cvdt_tree__"] = model.export_tree()
        return state

    def __setstate__(self, state):
        state = dict(state)
        tree = state.pop("__cvdt_tree__", None)
        super().__setstate__(state)
        self._fill_missing_params()
        if tree is not None:
            self._model = self._rebuild_model(tree)

    def _fill_missing_params(self):
        # A pickle from an older CVDT lacks parameters added since, which would
        # make ``get_params`` / ``repr`` / ``clone`` raise AttributeError. A
        # missing parameter predates its introduction, so it gets the value
        # that reproduces the behaviour the model was fitted with (see
        # ``_LEGACY_PARAMS``), else its current default. The fitted tree itself
        # is restored from the pickle and predicts exactly as before.
        import inspect

        params = inspect.signature(type(self).__init__).parameters
        for name, p in params.items():
            if p.default is inspect.Parameter.empty or name in self.__dict__:
                continue
            setattr(self, name, _LEGACY_PARAMS.get(name, p.default))

    def _rebuild_model(self, tree):  # pragma: no cover - overridden
        raise NotImplementedError


class CVDTClassifier(ClassifierMixin, _CVDTBase):
    """Cross-Validated Decision Tree classifier.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
        Impurity measure. Ignored when ``objective`` is set.
    objective : {None, "precision", "recall", "f1", "fbeta", "accuracy"}, default=None
        If given, splits are chosen to greedily improve this metric on the
        held-out folds instead of reducing impurity — an explicit optimisation
        of the target metric rather than a proxy. ``None`` uses ``criterion``.
    average : {"binary", "micro", "macro", "weighted"}, default="binary"
        Averaging for the objective on multiclass problems.
    pos_label : int, default=1
        Positive class when ``average="binary"``.
    beta : float, default=1.0
        β for ``objective="fbeta"``.

    (plus the common CVDT hyper-parameters; see ``_CVDTBase``.)

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    n_features_in_ : int
    feature_names_in_ : ndarray, present only if X had string column names.

    Notes
    -----
    Objective mode accepts a split only when it *improves* the metric over
    making the node a leaf, so trees are typically shallower and tuned to the
    metric. Set ``min_impurity_decrease`` below 0 to allow non-improving splits.
    ``pos_label`` is interpreted as an index into the sorted ``classes_``.
    """

    def __init__(
        self,
        *,
        criterion="gini",
        objective=None,
        average="binary",
        pos_label=1,
        beta=1.0,
        max_depth=8,
        min_samples_split=2,
        min_samples_leaf=30,
        min_impurity_decrease=0.0,
        n_bins=8,
        cv_folds=5,
        cv_seed=42,
        cv_shuffle=True,
        mode="strict",
        split_style="threshold",
        aggregator="mean",
        agg_frac=0.1,
        agg_eps=1e-12,
        agg_lambda=1.0,
        parallel=False,
        n_threads=1,
        parallel_min_samples=512,
        categorical_features=None,
        scoring="honest",
        cv_repeats=1,
        min_valid_folds=2,
        null_permutations=0,
        null_quantile=0.95,
        shrinkage=0.0,
        agg_q=0.25,
        agg_d0=4.0,
    ):
        self.criterion = criterion
        self.objective = objective
        self.average = average
        self.pos_label = pos_label
        self.beta = beta
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.n_bins = n_bins
        self.cv_folds = cv_folds
        self.cv_seed = cv_seed
        self.cv_shuffle = cv_shuffle
        self.mode = mode
        self.split_style = split_style
        self.aggregator = aggregator
        self.agg_frac = agg_frac
        self.agg_eps = agg_eps
        self.agg_lambda = agg_lambda
        self.parallel = parallel
        self.n_threads = n_threads
        self.parallel_min_samples = parallel_min_samples
        self.categorical_features = categorical_features
        self.scoring = scoring
        self.cv_repeats = cv_repeats
        self.min_valid_folds = min_valid_folds
        self.null_permutations = null_permutations
        self.null_quantile = null_quantile
        self.shrinkage = shrinkage
        self.agg_q = agg_q
        self.agg_d0 = agg_d0

    def fit(self, X, y):
        self._validate_common()
        if self.criterion not in ("gini", "entropy"):
            raise ValueError(
                f"criterion must be 'gini' or 'entropy', got {self.criterion!r}"
            )
        if self.objective is not None:
            if self.objective not in _OBJECTIVES:
                raise ValueError(
                    f"objective must be one of {sorted(_OBJECTIVES)} or None, "
                    f"got {self.objective!r}"
                )
            if self.average not in _AVERAGES:
                raise ValueError(
                    f"average must be one of {sorted(_AVERAGES)}, got {self.average!r}"
                )
        # Record feature names from the original X before it is coerced.
        _record_feature_names(self, X, reset=True)
        X, y = _check_X_y(X, y)
        check_classification_targets(y)
        _record_n_features(self, X, reset=True)

        self._encoder = LabelEncoder().fit(y)
        self.classes_ = self._encoder.classes_
        y_enc = self._encoder.transform(y).astype(np.int64)

        cat = _normalise_categorical(self.categorical_features, X.shape[1])
        n_classes = int(len(self.classes_))
        if self.objective is None:
            model = RawClassifier(
                n_classes=n_classes,
                criterion=str(self.criterion),
                **self._common_kwargs(cat),
            )
        else:
            pos = int(self.pos_label)
            if not (0 <= pos < n_classes):
                pos = 0
            model = RawObjectiveClassifier(
                n_classes=n_classes,
                objective=str(self.objective),
                average=str(self.average),
                pos_label=pos,
                beta=float(self.beta),
                **self._common_kwargs(cat),
            )
        model.fit(np.ascontiguousarray(X, dtype=np.float64), y_enc)
        self._model = model
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        _record_feature_names(self, X, reset=False)
        X = _check_array(X)
        _record_n_features(self, X, reset=False)
        idx = np.asarray(self._model.predict(np.ascontiguousarray(X, dtype=np.float64)))
        return self.classes_[idx]

    def predict_proba(self, X):
        check_is_fitted(self)
        _record_feature_names(self, X, reset=False)
        X = _check_array(X)
        _record_n_features(self, X, reset=False)
        proba = np.asarray(
            self._model.predict_proba(np.ascontiguousarray(X, dtype=np.float64))
        )
        return proba

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def get_depth(self):
        """Depth of the fitted tree (root-only tree has depth 0)."""
        check_is_fitted(self)
        return int(self._model.depth())

    def get_n_leaves(self):
        """Number of leaves in the fitted tree."""
        check_is_fitted(self)
        return int(self._model.n_leaves())

    def get_tree(self):
        """Return the fitted tree as a dict of parallel arrays (for custom plots)."""
        check_is_fitted(self)
        return self._model.export_tree()

    def export_text(self, feature_names=None, class_names=None):
        """Return a readable, rule-style text rendering of the fitted tree."""
        from ._treeviz import export_text as _t

        return _t(self.get_tree(), feature_names=feature_names, class_names=class_names)

    def export_graphviz(self, feature_names=None, class_names=None):
        """Return a Graphviz DOT string for the fitted tree."""
        from ._treeviz import export_graphviz as _g

        return _g(self.get_tree(), feature_names=feature_names, class_names=class_names)

    def _rebuild_model(self, tree):
        n_classes = int(len(self.classes_))
        cat = _normalise_categorical(self.categorical_features, self.n_features_in_)
        if self.objective is None:
            return RawClassifier.from_tree(tree, n_classes, cat)
        return RawObjectiveClassifier.from_tree(tree, n_classes, cat)


class CVDTRegressor(RegressorMixin, _CVDTBase):
    """Cross-Validated Decision Tree regressor.

    Parameters
    ----------
    criterion : {"mse", "variance", "mae"}, default="mse"
        Impurity measure. Note ``"mae"`` is not available with ``mode="fast"``
        (the median has no additive sufficient statistic for the histogram
        path); fitting will raise in that combination.
    """

    def __init__(
        self,
        *,
        criterion="mse",
        max_depth=8,
        min_samples_split=2,
        min_samples_leaf=30,
        min_impurity_decrease=0.0,
        n_bins=8,
        cv_folds=5,
        cv_seed=42,
        cv_shuffle=True,
        mode="strict",
        split_style="threshold",
        aggregator="mean",
        agg_frac=0.1,
        agg_eps=1e-12,
        agg_lambda=1.0,
        parallel=False,
        n_threads=1,
        parallel_min_samples=512,
        categorical_features=None,
        scoring="honest",
        cv_repeats=1,
        min_valid_folds=2,
        null_permutations=0,
        null_quantile=0.95,
        shrinkage=0.0,
        agg_q=0.25,
        agg_d0=4.0,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.n_bins = n_bins
        self.cv_folds = cv_folds
        self.cv_seed = cv_seed
        self.cv_shuffle = cv_shuffle
        self.mode = mode
        self.split_style = split_style
        self.aggregator = aggregator
        self.agg_frac = agg_frac
        self.agg_eps = agg_eps
        self.agg_lambda = agg_lambda
        self.parallel = parallel
        self.n_threads = n_threads
        self.parallel_min_samples = parallel_min_samples
        self.categorical_features = categorical_features
        self.scoring = scoring
        self.cv_repeats = cv_repeats
        self.min_valid_folds = min_valid_folds
        self.null_permutations = null_permutations
        self.null_quantile = null_quantile
        self.shrinkage = shrinkage
        self.agg_q = agg_q
        self.agg_d0 = agg_d0

    def fit(self, X, y):
        self._validate_common()
        if self.criterion not in ("mse", "variance", "mae"):
            raise ValueError(
                f"criterion must be 'mse', 'variance' or 'mae', got {self.criterion!r}"
            )
        _record_feature_names(self, X, reset=True)
        X, y = _check_X_y(X, y)
        _record_n_features(self, X, reset=True)
        y = np.asarray(y, dtype=np.float64)

        cat = _normalise_categorical(self.categorical_features, X.shape[1])
        model = RawRegressor(
            criterion=str(self.criterion),
            **self._common_kwargs(cat),
        )
        model.fit(np.ascontiguousarray(X, dtype=np.float64), y)
        self._model = model
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        _record_feature_names(self, X, reset=False)
        X = _check_array(X)
        _record_n_features(self, X, reset=False)
        return np.asarray(self._model.predict(np.ascontiguousarray(X, dtype=np.float64)))

    def get_depth(self):
        """Depth of the fitted tree (root-only tree has depth 0)."""
        check_is_fitted(self)
        return int(self._model.depth())

    def get_n_leaves(self):
        """Number of leaves in the fitted tree."""
        check_is_fitted(self)
        return int(self._model.n_leaves())

    def get_tree(self):
        """Return the fitted tree as a dict of parallel arrays (for custom plots)."""
        check_is_fitted(self)
        return self._model.export_tree()

    def export_text(self, feature_names=None):
        """Return a readable, rule-style text rendering of the fitted tree."""
        from ._treeviz import export_text as _t

        return _t(self.get_tree(), feature_names=feature_names)

    def export_graphviz(self, feature_names=None):
        """Return a Graphviz DOT string for the fitted tree."""
        from ._treeviz import export_graphviz as _g

        return _g(self.get_tree(), feature_names=feature_names)

    def _rebuild_model(self, tree):
        cat = _normalise_categorical(self.categorical_features, self.n_features_in_)
        return RawRegressor.from_tree(tree, cat)
