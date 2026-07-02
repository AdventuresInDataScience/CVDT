//! Global feature binning for the histogram ("fast") build path.
//!
//! Unlike the exact path — which refits continuous bin edges on every training
//! fold — the fast path bins the whole dataset **once** at fit time and reuses
//! those compact codes at every node. This is the LightGBM-style trade-off: it
//! gives up per-fold edge refitting (a weak, feature-only leak, since impurity
//! is still measured on held-out *labels*) in exchange for turning the dominant
//! per-node work into cheap histogram accumulation over `u16` codes.
//!
//! Continuous features are quantile-binned; categorical features are densely
//! re-indexed to `0..n_categories`. Missing / unknown values map to
//! [`MISSING_BIN_CODE`] and are excluded from candidate states (they route to
//! the "not-in-state" child at prediction time).

use crate::data::{Bin, CatId, Column, FeatureValue, MISSING_BIN_CODE, UNKNOWN_CAT};
use crate::encoder::{bin_bounds, bin_of, quantile_edges, MISSING_BIN};

/// Per-feature binning metadata produced once at fit time.
pub struct FeatureBins {
    /// Whether the source column is continuous.
    pub is_continuous: bool,
    /// Continuous cut points (empty for categorical).
    pub edges: Vec<f64>,
    /// Categorical: dense id → original category id, sorted ascending (empty for
    /// continuous).
    pub categories: Vec<CatId>,
    /// Number of valid bin slots; valid codes are `0..max_bin`.
    pub max_bin: usize,
}

impl FeatureBins {
    /// Code a raw feature value into a bin id, or [`MISSING_BIN_CODE`].
    pub fn code(&self, value: FeatureValue) -> Bin {
        match (self.is_continuous, value) {
            (true, FeatureValue::Continuous(x)) => {
                let b = bin_of(&self.edges, x);
                if b == MISSING_BIN {
                    MISSING_BIN_CODE
                } else {
                    b as Bin
                }
            }
            (false, FeatureValue::Categorical(c)) => {
                if c == UNKNOWN_CAT {
                    return MISSING_BIN_CODE;
                }
                match self.categories.binary_search(&c) {
                    Ok(pos) => pos as Bin,
                    Err(_) => MISSING_BIN_CODE,
                }
            }
            // Query value type disagrees with the column type → treat as missing.
            _ => MISSING_BIN_CODE,
        }
    }

    /// Half-open bounds for continuous bin `b` (for building a routing rule).
    pub fn cont_bounds(&self, b: usize) -> Option<(Option<f64>, Option<f64>)> {
        bin_bounds(&self.edges, b)
    }

    /// Original category id for dense id `b`.
    pub fn category(&self, b: usize) -> Option<CatId> {
        self.categories.get(b).copied()
    }
}

/// The whole dataset binned once: compact per-feature codes plus metadata.
pub struct BinnedData {
    /// `codes[feature][sample]` in `0..max_bin`, or [`MISSING_BIN_CODE`].
    pub codes: Vec<Vec<Bin>>,
    /// Per-feature metadata.
    pub bins: Vec<FeatureBins>,
    /// Number of samples.
    pub n_samples: usize,
}

impl BinnedData {
    /// Bin every column once. `n_bins` controls continuous quantile binning.
    ///
    /// Errors if a feature needs more bin slots than the compact [`Bin`] type
    /// can represent.
    pub fn fit(columns: &[Column], n_bins: usize) -> Result<BinnedData, String> {
        let cap = MISSING_BIN_CODE as usize; // codes must stay strictly below this
        let mut codes = Vec::with_capacity(columns.len());
        let mut bins = Vec::with_capacity(columns.len());
        let n_samples = columns.first().map_or(0, |c| c.len());

        for (f, col) in columns.iter().enumerate() {
            match col {
                Column::Continuous(v) => {
                    let edges = quantile_edges(v, n_bins);
                    let max_bin = edges.len() + 1;
                    if max_bin > cap {
                        return Err(format!(
                            "feature {f}: {max_bin} continuous bins exceed fast-mode limit"
                        ));
                    }
                    let col_codes: Vec<Bin> = v
                        .iter()
                        .map(|&x| {
                            let b = bin_of(&edges, x);
                            if b == MISSING_BIN {
                                MISSING_BIN_CODE
                            } else {
                                b as Bin
                            }
                        })
                        .collect();
                    bins.push(FeatureBins {
                        is_continuous: true,
                        edges,
                        categories: Vec::new(),
                        max_bin,
                    });
                    codes.push(col_codes);
                }
                Column::Categorical(c) => {
                    let mut distinct: Vec<CatId> =
                        c.iter().copied().filter(|&x| x != UNKNOWN_CAT).collect();
                    distinct.sort_unstable();
                    distinct.dedup();
                    let max_bin = distinct.len();
                    if max_bin > cap {
                        return Err(format!(
                            "feature {f}: {max_bin} categories exceed fast-mode limit"
                        ));
                    }
                    let col_codes: Vec<Bin> = c
                        .iter()
                        .map(|&x| {
                            if x == UNKNOWN_CAT {
                                MISSING_BIN_CODE
                            } else {
                                match distinct.binary_search(&x) {
                                    Ok(pos) => pos as Bin,
                                    Err(_) => MISSING_BIN_CODE,
                                }
                            }
                        })
                        .collect();
                    bins.push(FeatureBins {
                        is_continuous: false,
                        edges: Vec::new(),
                        categories: distinct,
                        max_bin,
                    });
                    codes.push(col_codes);
                }
            }
        }
        Ok(BinnedData {
            codes,
            bins,
            n_samples,
        })
    }

    /// Number of feature columns.
    pub fn n_features(&self) -> usize {
        self.codes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_codes_match_bin_of() {
        let v: Vec<f64> = (0..40).map(|x| x as f64).collect();
        let bd = BinnedData::fit(&[Column::Continuous(v.clone())], 4).unwrap();
        let fb = &bd.bins[0];
        for (i, &x) in v.iter().enumerate() {
            assert_eq!(bd.codes[0][i] as u32, bin_of(&fb.edges, x));
        }
    }

    #[test]
    fn categorical_is_densely_indexed() {
        // Categories 2 and 7 -> dense ids 0 and 1.
        let bd = BinnedData::fit(&[Column::Categorical(vec![7, 2, 7, 2])], 4).unwrap();
        let fb = &bd.bins[0];
        assert_eq!(fb.max_bin, 2);
        assert_eq!(fb.category(0), Some(2));
        assert_eq!(fb.category(1), Some(7));
        assert_eq!(bd.codes[0], vec![1, 0, 1, 0]);
    }

    #[test]
    fn unknown_category_codes_as_missing() {
        let bd = BinnedData::fit(&[Column::Categorical(vec![3, UNKNOWN_CAT, 3])], 4).unwrap();
        assert_eq!(bd.codes[0][1], MISSING_BIN_CODE);
    }

    #[test]
    fn code_roundtrips_a_query_value() {
        let bd = BinnedData::fit(&[Column::Categorical(vec![10, 20, 30])], 4).unwrap();
        let fb = &bd.bins[0];
        assert_eq!(fb.code(FeatureValue::cat(20)), 1);
        assert_eq!(fb.code(FeatureValue::cat(999)), MISSING_BIN_CODE);
    }
}
