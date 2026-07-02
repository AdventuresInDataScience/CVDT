"""Type stubs for the native ``cvdt._cvdt`` extension (Rust/PyO3)."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

class RawClassifier:
    def __init__(
        self,
        n_classes: int,
        criterion: str,
        max_depth: Optional[int],
        min_samples_split: int,
        min_samples_leaf: int,
        min_impurity_decrease: float,
        n_bins: int,
        cv_folds: int,
        cv_seed: int,
        cv_shuffle: bool,
        mode: str,
        aggregator: str,
        agg_frac: float,
        agg_eps: float,
        agg_lambda: float,
        parallel: bool,
        n_threads: int,
        parallel_min_samples: int,
        categorical: list[int],
    ) -> None: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.int64]) -> None: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.int64]: ...
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def depth(self) -> int: ...
    def n_leaves(self) -> int: ...
    def export_tree(self) -> dict: ...

class RawObjectiveClassifier:
    def __init__(
        self,
        n_classes: int,
        objective: str,
        average: str,
        pos_label: int,
        beta: float,
        max_depth: Optional[int],
        min_samples_split: int,
        min_samples_leaf: int,
        min_impurity_decrease: float,
        n_bins: int,
        cv_folds: int,
        cv_seed: int,
        cv_shuffle: bool,
        mode: str,
        aggregator: str,
        agg_frac: float,
        agg_eps: float,
        agg_lambda: float,
        parallel: bool,
        n_threads: int,
        parallel_min_samples: int,
        categorical: list[int],
    ) -> None: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.int64]) -> None: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.int64]: ...
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def depth(self) -> int: ...
    def n_leaves(self) -> int: ...
    def export_tree(self) -> dict: ...

class RawRegressor:
    def __init__(
        self,
        criterion: str,
        max_depth: Optional[int],
        min_samples_split: int,
        min_samples_leaf: int,
        min_impurity_decrease: float,
        n_bins: int,
        cv_folds: int,
        cv_seed: int,
        cv_shuffle: bool,
        mode: str,
        aggregator: str,
        agg_frac: float,
        agg_eps: float,
        agg_lambda: float,
        parallel: bool,
        n_threads: int,
        parallel_min_samples: int,
        categorical: list[int],
    ) -> None: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def depth(self) -> int: ...
    def n_leaves(self) -> int: ...
    def export_tree(self) -> dict: ...
