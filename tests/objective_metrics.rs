//! Exact-value tests for the objective metric layer: the confusion-matrix
//! sufficient statistic and the precision/recall/F1/Fβ/accuracy objectives.

use cvdt::{Accuracy, Average, ClassObjective, Confusion, FBeta, Precision, Recall, F1};

fn approx(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
}

/// Build the running-example 2-class confusion:
///   group predicts 1 to {class0:2, class1:8}
///   group predicts 0 to {class0:5, class1:1}
/// => tp0=5 fp0=1 fn0=2 support0=7 ; tp1=8 fp1=2 fn1=1 support1=9 ; total=16
fn example() -> Confusion {
    let mut cm = Confusion::zero(2);
    cm.add_group(1, &[2, 8]);
    cm.add_group(0, &[5, 1]);
    cm
}

#[test]
fn confusion_counts_are_correct() {
    let cm = example();
    assert_eq!(cm.tp, vec![5, 8]);
    assert_eq!(cm.fp, vec![1, 2]);
    assert_eq!(cm.fn_, vec![2, 1]);
    assert_eq!(cm.support, vec![7, 9]);
    assert_eq!(cm.total, 16);
}

#[test]
fn precision_binary_and_macro_and_micro_weighted() {
    let cm = example();
    approx(
        Precision {
            average: Average::Binary { pos_label: 1 },
        }
        .score(&cm),
        0.8, // 8 / 10
    );
    approx(
        Precision {
            average: Average::Binary { pos_label: 0 },
        }
        .score(&cm),
        5.0 / 6.0,
    );
    approx(
        Precision {
            average: Average::Macro,
        }
        .score(&cm),
        (5.0 / 6.0 + 0.8) / 2.0,
    );
    // micro precision = sumTP / (sumTP + sumFP) = 13/16
    approx(
        Precision {
            average: Average::Micro,
        }
        .score(&cm),
        13.0 / 16.0,
    );
    // weighted = (7 * 5/6 + 9 * 0.8) / 16
    approx(
        Precision {
            average: Average::Weighted,
        }
        .score(&cm),
        (7.0 * (5.0 / 6.0) + 9.0 * 0.8) / 16.0,
    );
}

#[test]
fn recall_and_f1_binary() {
    let cm = example();
    approx(
        Recall {
            average: Average::Binary { pos_label: 1 },
        }
        .score(&cm),
        8.0 / 9.0,
    );
    // F1 = 2*tp / (2*tp + fp + fn) = 16 / 19
    approx(
        F1 {
            average: Average::Binary { pos_label: 1 },
        }
        .score(&cm),
        16.0 / 19.0,
    );
}

#[test]
fn fbeta_reduces_to_f1_at_beta_one_and_weights_recall_at_beta_two() {
    let cm = example();
    approx(
        FBeta {
            beta: 1.0,
            average: Average::Binary { pos_label: 1 },
        }
        .score(&cm),
        16.0 / 19.0,
    );
    // beta=2: (1+4)tp / ((1+4)tp + 4 fn + fp) = 40 / (40 + 4 + 2) = 40/46
    approx(
        FBeta {
            beta: 2.0,
            average: Average::Binary { pos_label: 1 },
        }
        .score(&cm),
        40.0 / 46.0,
    );
}

#[test]
fn accuracy_is_correct_over_total() {
    let cm = example();
    approx(Accuracy.score(&cm), 13.0 / 16.0);
}

#[test]
fn zero_division_yields_zero() {
    let empty = Confusion::zero(3);
    approx(
        Precision {
            average: Average::Macro,
        }
        .score(&empty),
        0.0,
    );
    approx(
        Recall {
            average: Average::Micro,
        }
        .score(&empty),
        0.0,
    );
    approx(Accuracy.score(&empty), 0.0);
    // out-of-range pos_label is treated as score 0, not a panic
    approx(
        Precision {
            average: Average::Binary { pos_label: 9 },
        }
        .score(&empty),
        0.0,
    );
}

#[test]
fn multiclass_macro_recall() {
    // 3 classes. One group per predicted class.
    let mut cm = Confusion::zero(3);
    cm.add_group(0, &[6, 2, 0]); // predict 0
    cm.add_group(1, &[1, 5, 1]); // predict 1
    cm.add_group(2, &[0, 0, 4]); // predict 2
                                 // recall_k = tp_k / support_k
                                 // support: 0->7, 1->7, 2->5 ; tp: 0->6,1->5,2->4
    let r = Recall {
        average: Average::Macro,
    }
    .score(&cm);
    approx(r, (6.0 / 7.0 + 5.0 / 7.0 + 4.0 / 5.0) / 3.0);
}
