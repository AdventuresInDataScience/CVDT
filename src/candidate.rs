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

/// The states the tree actually scores for one feature at a node.
///
/// Identical to [`present_states`] except for threshold (`prefix`) splits of a
/// continuous feature, where the **top** bin's cut sends every present value
/// left and only missing values right: an "is present" test. That cut is kept
/// only when the node has missing values (otherwise its right child would be
/// empty), and the tree realises it as a [`crate::SplitRule::ContinuousBin`]
/// with both bounds open. The fast path applies the same rule to its global
/// bins, so both paths offer the same candidate set.
///
/// A state that every node sample matches (a lone category, or a constant
/// continuous feature under single-bin splits, with no missing values) would
/// send everything left, so it is not offered either. Such a candidate could
/// never score under [`crate::Scoring::InSample`] or objective mode, but
/// honest scoring would give it exactly 0.
pub fn split_states(col: &Column, indices: &[usize], n_bins: usize, prefix: bool) -> Vec<u32> {
    match col {
        Column::Continuous(v) if prefix => {
            let vals: Vec<f64> = indices.iter().map(|&i| v[i]).collect();
            let edges = quantile_edges(&vals, n_bins);
            let top = edges.len() as u32;
            let has_missing = vals.iter().any(|x| !x.is_finite());
            let mut set: BTreeSet<u32> = BTreeSet::new();
            for &x in &vals {
                let b = bin_of(&edges, x);
                if b != MISSING_BIN && (has_missing || b != top) {
                    set.insert(b);
                }
            }
            set.into_iter().collect()
        }
        _ => {
            let states = present_states(col, indices, n_bins);
            if states.len() == 1 && !any_missing(col, indices) {
                Vec::new()
            } else {
                states
            }
        }
    }
}

/// Whether any of the node samples is missing (non-finite) / unknown.
fn any_missing(col: &Column, indices: &[usize]) -> bool {
    match col {
        Column::Continuous(v) => indices.iter().any(|&i| !v[i].is_finite()),
        Column::Categorical(c) => indices.iter().any(|&i| c[i] == UNKNOWN_CAT),
    }
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
    fn split_states_offers_presence_cut_only_with_missing() {
        let idx: Vec<usize> = (0..40).collect();
        // No missing values: the top-bin threshold cut would send everything
        // left, so it is not offered.
        let full = Column::Continuous((0..40).map(|x| x as f64).collect());
        assert_eq!(split_states(&full, &idx, 4, true), vec![0, 1, 2]);
        // Bin-membership style is unaffected.
        assert_eq!(split_states(&full, &idx, 4, false), vec![0, 1, 2, 3]);
        // With missing values the top cut is "is present" and is offered.
        let gappy = Column::Continuous(
            (0..40)
                .map(|x| if x % 5 == 0 { f64::NAN } else { x as f64 })
                .collect(),
        );
        assert_eq!(split_states(&gappy, &idx, 4, true), vec![0, 1, 2, 3]);
        // Categorical columns ignore `prefix`.
        let cat = Column::Categorical(vec![2, 0, 2, 1, 0]);
        assert_eq!(split_states(&cat, &[0, 1, 2, 3, 4], 4, true), vec![0, 1, 2]);
    }

    #[test]
    fn split_states_drops_a_lone_state_that_covers_the_node() {
        let idx: Vec<usize> = (0..4).collect();
        // One category everywhere: the split would send everything left.
        assert!(split_states(&Column::Categorical(vec![3; 4]), &idx, 4, false).is_empty());
        // With an unknown value the same state is a real split.
        let gappy = Column::Categorical(vec![3, 3, UNKNOWN_CAT, 3]);
        assert_eq!(split_states(&gappy, &idx, 4, false), vec![3]);
        // Constant continuous feature, single-bin style.
        assert!(split_states(&Column::Continuous(vec![5.0; 4]), &idx, 4, false).is_empty());
        let nan = Column::Continuous(vec![5.0, f64::NAN, 5.0, 5.0]);
        assert_eq!(split_states(&nan, &idx, 4, false), vec![3]);
        // present_states is unchanged: it lists what occurs.
        assert_eq!(present_states(&Column::Categorical(vec![3; 4]), &idx, 4), vec![3]);
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
