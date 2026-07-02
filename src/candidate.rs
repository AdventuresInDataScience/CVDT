//! Candidate split generation.
//!
//! Given the samples at a node, enumerate every `feature == state` split that
//! actually occurs in the node. This module produces plain candidate objects
//! and knows nothing about impurity, cross-validation or scoring.
//!
//! * Continuous features: fit node-level quantile edges and emit one candidate
//!   per bin id that at least one sample falls into.
//! * Categorical features: emit one candidate per category id present.
//!
//! The node-level encoding is used only to enumerate *which* states exist; the
//! actual impurity scoring re-fits bin edges on each training fold, so the
//! enumeration here does not leak validation information into the score.

use std::collections::BTreeSet;

use crate::data::{Column, UNKNOWN_CAT};
use crate::encoder::{bin_of, quantile_edges, MISSING_BIN};

/// A single `feature == state` candidate split.
///
/// For continuous features `state` is a quantile bin id; for categorical
/// features it is a category id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    /// Index of the feature this split tests.
    pub feature: usize,
    /// The bin id / category id the feature is compared against.
    pub state: u32,
}

/// Distinct states present for one column over the given sample `indices`.
///
/// Returned sorted ascending for deterministic downstream behaviour. Missing
/// continuous values and unknown categories are excluded.
pub fn present_states(col: &Column, indices: &[usize], n_bins: usize) -> Vec<u32> {
    let mut set: BTreeSet<u32> = BTreeSet::new();
    match col {
        Column::Continuous(v) => {
            let vals: Vec<f64> = indices.iter().map(|&i| v[i]).collect();
            let edges = quantile_edges(&vals, n_bins);
            for &i in indices {
                let b = bin_of(&edges, v[i]);
                if b != MISSING_BIN {
                    set.insert(b);
                }
            }
        }
        Column::Categorical(c) => {
            for &i in indices {
                let x = c[i];
                if x != UNKNOWN_CAT {
                    set.insert(x);
                }
            }
        }
    }
    set.into_iter().collect()
}

/// Enumerate all candidate splits across all features for a node.
pub fn generate_candidates(columns: &[Column], indices: &[usize], n_bins: usize) -> Vec<Candidate> {
    let mut out = Vec::new();
    for (f, col) in columns.iter().enumerate() {
        for s in present_states(col, indices, n_bins) {
            out.push(Candidate {
                feature: f,
                state: s,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorical_states_are_the_present_ids() {
        let col = Column::Categorical(vec![2, 0, 2, 1, 0]);
        let idx: Vec<usize> = (0..5).collect();
        assert_eq!(present_states(&col, &idx, 4), vec![0, 1, 2]);
    }

    #[test]
    fn categorical_respects_index_subset() {
        let col = Column::Categorical(vec![2, 0, 2, 1, 0]);
        // only look at samples 0 and 2 -> only category 2 present
        assert_eq!(present_states(&col, &[0, 2], 4), vec![2]);
    }

    #[test]
    fn continuous_states_are_bin_ids() {
        let col = Column::Continuous((0..40).map(|x| x as f64).collect());
        let idx: Vec<usize> = (0..40).collect();
        let states = present_states(&col, &idx, 4);
        // 4 bins -> ids 0..=3, all populated by a uniform spread
        assert_eq!(states, vec![0, 1, 2, 3]);
    }

    #[test]
    fn generate_covers_all_features() {
        let cols = vec![
            Column::Categorical(vec![0, 1, 0, 1]),
            Column::Continuous(vec![0.0, 1.0, 2.0, 3.0]),
        ];
        let idx: Vec<usize> = (0..4).collect();
        let cands = generate_candidates(&cols, &idx, 2);
        assert!(cands.iter().any(|c| c.feature == 0));
        assert!(cands.iter().any(|c| c.feature == 1));
    }
}
