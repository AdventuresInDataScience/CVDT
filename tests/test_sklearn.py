"""Python-side tests for the sklearn-compatible estimators.

Requires the extension to be built first:
    maturin develop --features python
Then:
    pytest tests/test_sklearn.py
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score

from cvdt import CVDTClassifier, CVDTRegressor


def test_classifier_learns():
    X, y = make_classification(
        n_samples=400, n_features=8, n_informative=5, n_classes=2, random_state=0
    )
    clf = CVDTClassifier(cv_folds=4, max_depth=6).fit(X, y)
    assert clf.score(X, y) > 0.85
    assert set(clf.classes_) == set(np.unique(y))
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_regressor_learns():
    X, y = make_regression(n_samples=400, n_features=6, noise=5.0, random_state=0)
    reg = CVDTRegressor(cv_folds=4, max_depth=8).fit(X, y)
    assert reg.score(X, y) > 0.5


def test_get_set_params_and_clone():
    clf = CVDTClassifier(criterion="entropy", max_depth=3, aggregator="median")
    params = clf.get_params()
    assert params["criterion"] == "entropy"
    assert params["max_depth"] == 3
    clf2 = clone(clf)
    assert clf2.get_params() == params
    clf.set_params(max_depth=7)
    assert clf.max_depth == 7


def test_fast_mode_matches_strict_on_separable():
    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4, n_classes=2,
        class_sep=3.0, random_state=1,
    )
    strict = CVDTClassifier(mode="strict", cv_folds=4).fit(X, y)
    # Fast mode bins globally *once* and never re-subdivides, and each split is a
    # one-vs-rest `feature == bin` test that isolates a single bin. With too many
    # bins each split peels only a thin sliver, so with a bounded depth most of
    # the data lands in one impure "rest" leaf. The bin count must therefore be
    # matched to the data's granularity (unlike strict, which re-bins per node);
    # 16 bins separate these clusters cleanly, whereas 32+ would strand the bulk
    # of the samples. See the "Strict vs Fast mode" note in README.md.
    fast = CVDTClassifier(mode="fast", cv_folds=4, n_bins=16).fit(X, y)
    assert strict.score(X, y) > 0.9
    assert fast.score(X, y) > 0.9


def test_mae_fast_raises():
    X, y = make_regression(n_samples=100, n_features=4, random_state=0)
    with pytest.raises(Exception):
        CVDTRegressor(criterion="mae", mode="fast").fit(X, y)


def test_pipeline_cross_val():
    X, y = make_classification(n_samples=300, n_features=6, random_state=0)
    scores = cross_val_score(CVDTClassifier(cv_folds=3), X, y, cv=4)
    assert scores.mean() > 0.6


def test_nan_is_accepted():
    X, y = make_classification(n_samples=200, n_features=5, random_state=0)
    X = X.copy()
    X[0, 0] = np.nan
    clf = CVDTClassifier(cv_folds=3).fit(X, y)  # should not raise
    assert clf.predict(X[:5]).shape == (5,)


def test_categorical_features():
    rng = np.random.default_rng(0)
    n = 300
    cat = rng.integers(0, 3, size=n).astype(float)
    cont = rng.normal(size=n)
    X = np.column_stack([cat, cont])
    y = (cat.astype(int) == 1).astype(int)  # class fully determined by the category
    clf = CVDTClassifier(cv_folds=4, categorical_features=[0]).fit(X, y)
    assert clf.score(X, y) > 0.9


@pytest.mark.parametrize("objective", ["f1", "precision", "recall", "accuracy"])
def test_objective_mode_learns(objective):
    X, y = make_classification(
        n_samples=400, n_features=8, n_informative=5, n_classes=2,
        weights=[0.8, 0.2], random_state=3,
    )
    clf = CVDTClassifier(
        objective=objective, average="binary", pos_label=1, cv_folds=4, max_depth=6
    ).fit(X, y)
    assert clf.predict(X).shape == (X.shape[0],)
    assert clf.score(X, y) > 0.5


def test_objective_recall_beats_gini_on_recall():
    # On an imbalanced problem, optimising recall should not do worse on recall
    # than the impurity default (usually better). Objective mode is self-stopping
    # (a split must *improve* the metric over making the node a leaf), and for
    # recall that threshold is reached quickly, yielding a very shallow tree. To
    # let the recall objective actually express itself we allow non-improving
    # splits via `min_impurity_decrease < 0`, exactly as the docstring documents.
    from sklearn.metrics import recall_score

    X, y = make_classification(
        n_samples=800, n_features=10, n_informative=6, n_classes=2,
        weights=[0.85, 0.15], random_state=5,
    )
    gini = CVDTClassifier(cv_folds=4, max_depth=6).fit(X, y)
    rec = CVDTClassifier(
        objective="recall", average="binary", pos_label=1, cv_folds=4,
        max_depth=6, min_impurity_decrease=-1.0,
    ).fit(X, y)
    r_gini = recall_score(y, gini.predict(X), pos_label=1, zero_division=0)
    r_rec = recall_score(y, rec.predict(X), pos_label=1, zero_division=0)
    assert r_rec >= r_gini - 1e-9


def test_objective_multiclass_macro_f1():
    X, y = make_classification(
        n_samples=450, n_features=8, n_informative=6, n_classes=3, random_state=0
    )
    clf = CVDTClassifier(objective="f1", average="macro", cv_folds=4).fit(X, y)
    assert clf.predict_proba(X).shape == (X.shape[0], 3)


def test_objective_params_roundtrip():
    from sklearn.base import clone

    clf = CVDTClassifier(objective="fbeta", beta=2.0, average="macro")
    p = clf.get_params()
    assert p["objective"] == "fbeta"
    assert p["beta"] == 2.0
    assert clone(clf).get_params() == p


# --- determinism, aggregators, criteria, errors --------------------------

def test_determinism_same_seed():
    X, y = make_classification(n_samples=300, n_features=6, random_state=0)
    a = CVDTClassifier(cv_folds=5, cv_seed=99).fit(X, y).predict(X)
    b = CVDTClassifier(cv_folds=5, cv_seed=99).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "aggregator",
    ["mean", "median", "trimmed_mean", "signal_to_noise", "mean_minus_lambda_std"],
)
def test_all_aggregators_fit(aggregator):
    X, y = make_classification(n_samples=250, n_features=6, random_state=1)
    clf = CVDTClassifier(aggregator=aggregator, cv_folds=4).fit(X, y)
    assert clf.predict(X).shape == (X.shape[0],)


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_both_criteria_fit(criterion):
    X, y = make_classification(n_samples=250, n_features=6, random_state=2)
    clf = CVDTClassifier(criterion=criterion, cv_folds=4).fit(X, y)
    assert clf.score(X, y) > 0.6


@pytest.mark.parametrize("criterion", ["mse", "variance", "mae"])
def test_regressor_criteria(criterion):
    X, y = make_regression(n_samples=250, n_features=5, noise=5.0, random_state=0)
    # mae is strict-only; default mode is strict, so all three work here.
    reg = CVDTRegressor(criterion=criterion, cv_folds=4).fit(X, y)
    assert reg.predict(X).shape == (X.shape[0],)


def test_predict_log_proba_is_finite_where_proba_positive():
    X, y = make_classification(n_samples=200, n_features=5, random_state=0)
    clf = CVDTClassifier(cv_folds=4).fit(X, y)
    lp = clf.predict_log_proba(X)
    p = clf.predict_proba(X)
    assert lp.shape == p.shape
    assert np.all(lp[p > 0] <= 0.0 + 1e-9)


def test_set_params_changes_behavior():
    X, y = make_classification(n_samples=200, n_features=5, random_state=0)
    clf = CVDTClassifier(max_depth=1)
    clf.set_params(max_depth=8, criterion="entropy")
    assert clf.max_depth == 8
    assert clf.criterion == "entropy"
    clf.fit(X, y)
    assert clf.predict(X).shape == (X.shape[0],)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"criterion": "nonsense"},
        {"aggregator": "nope"},
        {"mode": "turbo"},
        {"objective": "auc"},
        {"objective": "f1", "average": "bogus"},
        {"n_bins": 1},
        {"cv_folds": 1},
    ],
)
def test_invalid_params_raise(kwargs):
    X, y = make_classification(n_samples=120, n_features=4, random_state=0)
    with pytest.raises(Exception):
        CVDTClassifier(**kwargs).fit(X, y)


def test_predict_feature_count_mismatch_raises():
    X, y = make_classification(n_samples=150, n_features=6, random_state=0)
    clf = CVDTClassifier(cv_folds=4).fit(X, y)
    with pytest.raises(Exception):
        clf.predict(X[:, :3])  # wrong number of features


def test_regressor_fast_mae_raises():
    X, y = make_regression(n_samples=120, n_features=4, random_state=0)
    with pytest.raises(Exception):
        CVDTRegressor(criterion="mae", mode="fast").fit(X, y)


def test_dataframe_feature_names_recorded():
    pd = pytest.importorskip("pandas")
    X, y = make_classification(n_samples=150, n_features=4, random_state=0)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    clf = CVDTClassifier(cv_folds=4).fit(df, y)
    assert list(clf.feature_names_in_) == ["a", "b", "c", "d"]
    assert clf.predict(df).shape == (X.shape[0],)


def test_tree_size_accessors():
    X, y = make_classification(n_samples=200, n_features=5, random_state=0)
    clf = CVDTClassifier(cv_folds=4, max_depth=4).fit(X, y)
    assert clf.get_depth() >= 0
    assert clf.get_depth() <= 4
    assert clf.get_n_leaves() >= 1
    reg = CVDTRegressor(cv_folds=4, max_depth=4).fit(X, y.astype(float))
    assert reg.get_n_leaves() >= 1


def test_tree_export_text_and_graphviz():
    X, y = make_classification(n_samples=200, n_features=5, n_classes=2, random_state=0)
    clf = CVDTClassifier(cv_folds=4, max_depth=3).fit(X, y)
    txt = clf.export_text(feature_names=[f"f{i}" for i in range(5)], class_names=["neg", "pos"])
    assert isinstance(txt, str) and len(txt) > 0
    dot = clf.export_graphviz()
    assert dot.startswith("digraph")
    tree = clf.get_tree()
    assert tree["node_count"] == len(tree["children_left"])
    # leaves have no children
    for i in range(tree["node_count"]):
        if tree["is_leaf"][i]:
            assert tree["children_left"][i] == -1

    reg = CVDTRegressor(cv_folds=4, max_depth=3).fit(X, y.astype(float))
    assert "value=" in reg.export_text() or "empty" in reg.export_text()


def test_get_params_lists_all_hyperparameters():
    keys = set(CVDTClassifier().get_params().keys())
    expected = {
        "criterion", "objective", "average", "pos_label", "beta",
        "max_depth", "min_samples_split", "min_samples_leaf",
        "min_impurity_decrease", "n_bins", "cv_folds", "cv_seed", "cv_shuffle",
        "mode", "aggregator", "agg_frac", "agg_eps", "agg_lambda",
        "parallel", "n_threads", "parallel_min_samples", "categorical_features",
    }
    assert expected <= keys


# Full estimator conformance. Kept last; if a specific sub-check is too strict
# for a from-scratch estimator, prefer xfail over deleting the whole sweep.
def test_sklearn_check_estimator():
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(CVDTClassifier())
    check_estimator(CVDTRegressor())
