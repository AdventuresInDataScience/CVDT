"""CVDT in the scikit-learn ecosystem.

Run after building the extension (see PYTHON.md):
    maturin develop --release --features python
    python examples/sklearn_usage.py
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cvdt import CVDTClassifier, CVDTRegressor


def classification_demo():
    X, y = make_classification(
        n_samples=600, n_features=10, n_informative=5, n_classes=3, random_state=0
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

    clf = CVDTClassifier(criterion="gini", cv_folds=5, max_depth=6)
    clf.fit(Xtr, ytr)
    print("classifier test accuracy:", clf.score(Xte, yte))
    print("proba row 0:", clf.predict_proba(Xte[:1]))

    # Works inside a Pipeline.
    pipe = make_pipeline(StandardScaler(), CVDTClassifier(cv_folds=4))
    scores = cross_val_score(pipe, X, y, cv=5)
    print("pipeline 5-fold CV accuracy: %.3f +/- %.3f" % (scores.mean(), scores.std()))

    # Works with GridSearchCV — including the CVDT-specific aggregator knob.
    grid = GridSearchCV(
        CVDTClassifier(),
        {
            "max_depth": [4, 8],
            "aggregator": ["mean", "mean_minus_lambda_std"],
            "mode": ["strict", "fast"],
        },
        cv=3,
    )
    grid.fit(X, y)
    print("best params:", grid.best_params_)
    print("best CV score: %.3f" % grid.best_score_)


def regression_demo():
    X, y = make_regression(n_samples=500, n_features=8, noise=8.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

    reg = CVDTRegressor(criterion="mse", cv_folds=5, max_depth=8)
    reg.fit(Xtr, ytr)
    print("regressor test R^2:", reg.score(Xte, yte))

    # Fast (histogram) mode for speed.
    fast = CVDTRegressor(criterion="mse", mode="fast", n_bins=32).fit(Xtr, ytr)
    print("fast-mode R^2:", fast.score(Xte, yte))


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("== classification ==")
    classification_demo()
    print("\n== regression ==")
    regression_demo()
