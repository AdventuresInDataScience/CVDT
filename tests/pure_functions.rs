//! Exact-value tests for the pure, deterministic building blocks: impurity
//! criteria, quantile encoding, fold-score aggregators and split selection.

use cvdt::criterion::{entropy_from_counts, gini_from_counts, variance_from_moments};
use cvdt::{
    bin_of, present_states, quantile_edges, select_best, Aggregator, Candidate, Column, Criterion,
    Entropy, FoldStats, Gini, Mae, Mean, MeanMinusLambdaStd, Median, Mse, ScoredCandidate,
    SignalToNoise, TrimmedMean, Variance, MISSING_BIN, UNKNOWN_CAT,
};

fn approx(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
}

// --- criteria --------------------------------------------------------------

#[test]
fn gini_known_values() {
    approx(Gini::new(2).impurity(&[0, 0, 1, 1]), 0.5);
    approx(Gini::new(2).impurity(&[0, 0, 0, 0]), 0.0); // pure
    approx(Gini::new(3).impurity(&[0, 1, 2]), 2.0 / 3.0); // uniform 3-class
    approx(Gini::new(2).impurity(&[]), 0.0); // empty
}

#[test]
fn entropy_known_values() {
    approx(Entropy::new(2).impurity(&[0, 1]), 1.0); // 1 bit
    approx(Entropy::new(2).impurity(&[1, 1, 1, 1]), 0.0); // pure
                                                          // p=(0.75, 0.25): -(0.75 log2 0.75 + 0.25 log2 0.25)
    let expect = -(0.75_f64 * 0.75_f64.log2() + 0.25_f64 * 0.25_f64.log2());
    approx(Entropy::new(2).impurity(&[0, 0, 0, 1]), expect);
}

#[test]
fn variance_mse_mae_known_values() {
    approx(Variance.impurity(&[1.0, 3.0]), 1.0);
    approx(Variance.impurity(&[5.0, 5.0, 5.0]), 0.0);
    approx(Mse.impurity(&[1.0, 3.0]), 1.0); // Mse == Variance
    approx(Mae.impurity(&[0.0, 0.0, 10.0, 10.0]), 5.0); // median 5
    approx(Mae.impurity(&[1.0, 2.0, 3.0]), 2.0 / 3.0); // median 2
    approx(Variance.impurity(&[0.0, 0.0, 10.0, 10.0]), 25.0);
}

#[test]
fn from_counts_helpers() {
    approx(gini_from_counts(&[2, 2], 4), 0.5);
    approx(gini_from_counts(&[4, 0], 4), 0.0);
    approx(entropy_from_counts(&[1, 1], 2), 1.0);
    approx(variance_from_moments(2.0, 4.0, 10.0), 1.0); // n,sum,sumsq of {1,3}
    approx(variance_from_moments(0.0, 0.0, 0.0), 0.0);
}

// --- encoder ---------------------------------------------------------------

#[test]
fn quantile_edges_equal_frequency() {
    let v: Vec<f64> = (0..10).map(|x| x as f64).collect();
    assert_eq!(quantile_edges(&v, 2), vec![5.0]);
    assert_eq!(quantile_edges(&v, 5), vec![2.0, 4.0, 6.0, 8.0]);
    assert!(quantile_edges(&v, 1).is_empty()); // n_bins <= 1
    assert!(quantile_edges(&[f64::NAN, f64::INFINITY], 4).is_empty()); // no finite
}

#[test]
fn bin_of_routing() {
    let edges = [2.0, 4.0, 6.0, 8.0];
    assert_eq!(bin_of(&edges, 1.0), 0);
    assert_eq!(bin_of(&edges, 3.0), 1);
    assert_eq!(bin_of(&edges, 4.0), 2); // edge is <=, goes up
    assert_eq!(bin_of(&edges, 9.0), 4);
    assert_eq!(bin_of(&edges, f64::NAN), MISSING_BIN);
    assert_eq!(bin_of(&edges, f64::INFINITY), MISSING_BIN);
}

#[test]
fn present_states_unique_sorted() {
    let col = Column::Continuous(vec![0.0, 1.0, 2.0, 3.0]);
    let idx: Vec<usize> = (0..4).collect();
    let states = present_states(&col, &idx, 2); // edge at 2.0 -> bins {0,1}
    assert_eq!(states, vec![0, 1]);

    let cat = Column::Categorical(vec![5, 5, 2, UNKNOWN_CAT]);
    let states = present_states(&cat, &idx, 8);
    assert_eq!(states, vec![2, 5]); // unknown excluded, sorted, unique
}

// --- aggregators -----------------------------------------------------------

#[test]
fn aggregator_known_values() {
    let s = FoldStats::from_scores(vec![1.0, 2.0, 3.0], 3);
    approx(Mean.aggregate(&s), 2.0);
    // population std of {1,2,3} = sqrt(2/3)
    let std = (2.0_f64 / 3.0).sqrt();
    approx(MeanMinusLambdaStd { lambda: 1.0 }.aggregate(&s), 2.0 - std);
    approx(SignalToNoise { eps: 0.0 }.aggregate(&s), 2.0 / std);

    approx(
        Median.aggregate(&FoldStats::from_scores(vec![1.0, 2.0, 3.0, 4.0], 4)),
        2.5,
    );
    approx(
        Median.aggregate(&FoldStats::from_scores(vec![1.0, 2.0, 3.0], 3)),
        2.0,
    );
}

#[test]
fn empty_stats_sink_to_neg_infinity() {
    let empty = FoldStats::from_scores(vec![], 5);
    assert_eq!(Mean.aggregate(&empty), f64::NEG_INFINITY);
    assert_eq!(Median.aggregate(&empty), f64::NEG_INFINITY);
    assert_eq!(
        SignalToNoise { eps: 1e-9 }.aggregate(&empty),
        f64::NEG_INFINITY
    );
}

#[test]
fn trimmed_mean_resists_outlier() {
    // {1,2,3,4,100}, frac 0.25 -> trim 1 from each end -> mean{2,3,4} = 3
    let s = FoldStats::from_scores(vec![1.0, 2.0, 3.0, 4.0, 100.0], 5);
    approx(TrimmedMean { frac: 0.25 }.aggregate(&s), 3.0);
    // frac 0 == plain mean
    approx(
        TrimmedMean { frac: 0.0 }.aggregate(&s),
        (1.0 + 2.0 + 3.0 + 4.0 + 100.0) / 5.0,
    );
}

// --- selector --------------------------------------------------------------

fn sc(feature: usize, state: u32, score: f64) -> ScoredCandidate {
    ScoredCandidate {
        candidate: Candidate { feature, state },
        score,
        stats: FoldStats::from_scores(vec![score], 1),
    }
}

#[test]
fn select_best_picks_max() {
    let v = vec![sc(0, 0, 0.1), sc(1, 2, 0.5), sc(2, 1, 0.3)];
    let best = select_best(&v).unwrap();
    assert_eq!((best.candidate.feature, best.candidate.state), (1, 2));
}

#[test]
fn select_best_tie_breaks_lexicographically() {
    let v = vec![sc(3, 5, 0.5), sc(1, 9, 0.5), sc(1, 2, 0.5)];
    let best = select_best(&v).unwrap();
    // lowest (feature, state) among ties
    assert_eq!((best.candidate.feature, best.candidate.state), (1, 2));
}

#[test]
fn select_best_skips_non_finite() {
    let v = vec![sc(0, 0, f64::NEG_INFINITY), sc(1, 1, 0.2)];
    let best = select_best(&v).unwrap();
    assert_eq!(best.candidate.feature, 1);

    let all_bad = vec![sc(0, 0, f64::NEG_INFINITY), sc(1, 1, f64::NAN)];
    assert!(select_best(&all_bad).is_none());
    assert!(select_best(&[]).is_none());
}
