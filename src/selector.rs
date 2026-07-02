//! Split selection.
//!
//! The selector ranks candidates that have already been scored by the
//! aggregation layer and returns the winner. It knows nothing about how the
//! score was produced — it only compares final numbers and breaks ties
//! deterministically so that a tree fit is fully reproducible (and so that the
//! parallel and serial code paths agree regardless of the order in which
//! per-feature work happens to complete).

use crate::candidate::Candidate;
use crate::cross_validation::FoldStats;

/// A candidate paired with its aggregated score and the statistics behind it.
#[derive(Clone, Debug)]
pub struct ScoredCandidate {
    /// The split being scored.
    pub candidate: Candidate,
    /// Final aggregated ranking score (higher is better).
    pub score: f64,
    /// The per-fold statistics the score was derived from.
    pub stats: FoldStats,
}

/// Pick the best candidate.
///
/// Rules:
/// * candidates whose score is not finite (e.g. `-inf` from a fold set with no
///   successful folds, or `NaN`) are ignored;
/// * the highest score wins;
/// * ties are broken deterministically by `(feature, state)` ascending.
///
/// Returns `None` when there is no candidate with a finite score.
pub fn select_best(scored: &[ScoredCandidate]) -> Option<&ScoredCandidate> {
    let mut best: Option<&ScoredCandidate> = None;
    for cand in scored {
        if !cand.score.is_finite() {
            continue;
        }
        best = Some(match best {
            None => cand,
            Some(b) => {
                if cand.score > b.score {
                    cand
                } else if cand.score < b.score {
                    b
                } else {
                    // Exact tie: prefer the lexicographically smaller split.
                    let key_c = (cand.candidate.feature, cand.candidate.state);
                    let key_b = (b.candidate.feature, b.candidate.state);
                    if key_c < key_b {
                        cand
                    } else {
                        b
                    }
                }
            }
        });
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sc(feature: usize, state: u32, score: f64) -> ScoredCandidate {
        ScoredCandidate {
            candidate: Candidate { feature, state },
            score,
            stats: FoldStats::from_scores(vec![score], 1),
        }
    }

    #[test]
    fn picks_highest_score() {
        let v = vec![sc(0, 0, 0.1), sc(1, 2, 0.9), sc(2, 1, 0.5)];
        let best = select_best(&v).unwrap();
        assert_eq!(
            best.candidate,
            Candidate {
                feature: 1,
                state: 2
            }
        );
    }

    #[test]
    fn ties_break_by_feature_then_state() {
        let v = vec![sc(3, 5, 0.5), sc(1, 9, 0.5), sc(1, 2, 0.5)];
        let best = select_best(&v).unwrap();
        // All tied on score -> smallest (feature, state) = (1, 2).
        assert_eq!(
            best.candidate,
            Candidate {
                feature: 1,
                state: 2
            }
        );
    }

    #[test]
    fn ignores_non_finite_scores() {
        let v = vec![
            sc(0, 0, f64::NEG_INFINITY),
            sc(1, 1, f64::NAN),
            sc(2, 2, 0.3),
        ];
        let best = select_best(&v).unwrap();
        assert_eq!(
            best.candidate,
            Candidate {
                feature: 2,
                state: 2
            }
        );
    }

    #[test]
    fn none_when_all_invalid() {
        let v = vec![sc(0, 0, f64::NEG_INFINITY), sc(1, 1, f64::NAN)];
        assert!(select_best(&v).is_none());
    }

    #[test]
    fn empty_is_none() {
        assert!(select_best(&[]).is_none());
    }
}
