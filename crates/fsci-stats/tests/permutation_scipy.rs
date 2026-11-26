//! Exact and randomized scalar permutation regressions.
//!
//! Run with FSCI_REQUIRE_SCIPY_ORACLE=1 and FSCI_PYTHON pointing to the pinned
//! SciPy 1.17.1 / NumPy 2.4.3 interpreter to require 45 live comparisons.
//! Without the required-oracle flag, an unavailable Python/SciPy is an explicit
//! skip; all other tests run independently of Python.

use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use fsci_runtime::{AuditAction, RuntimeMode};
use fsci_stats::audit::sync_audit_ledger;
use fsci_stats::{
    PermutationAlternative, PermutationMethod, PermutationTestOptions, PermutationTestResult,
    PermutationType, StatsError, StatsOptions, active_stats_mode, monte_carlo_test,
    permutation_test, permutation_test_samples, permutation_test_with_options,
    with_hardened_max_resamples, with_hardened_stats_caps,
};

fn mean(sample: &[f64]) -> f64 {
    sample.iter().sum::<f64>() / sample.len() as f64
}

fn difference(samples: &[&[f64]]) -> f64 {
    mean(samples[0]) - mean(samples[1])
}

fn options(kind: PermutationType, alternative: PermutationAlternative) -> PermutationTestOptions {
    PermutationTestOptions {
        permutation_type: kind,
        alternative,
        n_resamples: 1_000,
        rng: 1729,
        ..PermutationTestOptions::default()
    }
}

fn sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(f64::total_cmp);
    values
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance}"
    );
}

#[test]
fn exact_two_sided_test_uses_smaller_tail_and_is_translation_invariant() {
    let data: &[&[f64]] = &[&[1.0, 2.0], &[3.0, 4.0]];
    for (alternative, expected) in [
        (PermutationAlternative::Less, 1.0 / 6.0),
        (PermutationAlternative::Greater, 1.0),
        (PermutationAlternative::TwoSided, 1.0 / 3.0),
    ] {
        for offset in [0.0, 10.0, -10.0] {
            let result = permutation_test_samples(
                data,
                |samples| difference(samples) + offset,
                options(PermutationType::Independent, alternative),
            )
            .unwrap();
            assert!(result.exact);
            assert_eq!(result.statistic, -2.0 + offset);
            assert_eq!(result.pvalue, expected);
            assert_eq!(
                result.null_distribution,
                [-2.0, -1.0, 0.0, 0.0, 1.0, 2.0]
                    .map(|value| value + offset)
                    .to_vec()
            );
        }
    }
    let legacy = permutation_test(data[0], data[1], |x, y| mean(x) - mean(y) + 10.0, 99, 7);
    assert_eq!(legacy, (8.0, 1.0 / 3.0));
    let method = PermutationMethod::new(99, Some(8), 7).unwrap();
    assert_eq!(
        method
            .test(data[0], data[1], |x, y| mean(x) - mean(y) + 10.0)
            .unwrap(),
        legacy
    );
}

#[test]
fn exact_switch_occurs_at_the_number_of_distinct_partitions() {
    let data: &[&[f64]] = &[&[1.0, 2.0], &[3.0, 4.0]];
    for budget in [1, 5, 6, 7, usize::MAX] {
        let result = permutation_test_samples(
            data,
            difference,
            PermutationTestOptions {
                n_resamples: budget,
                ..PermutationTestOptions::default()
            },
        )
        .unwrap();
        assert_eq!(result.exact, budget >= 6);
        assert_eq!(result.null_distribution.len(), budget.min(6));
        if result.exact {
            assert_eq!(result.pvalue, 1.0 / 3.0);
        }
    }
    let duplicate_values: &[&[f64]] = &[&[1.0, 1.0], &[1.0, 1.0]];
    let result = permutation_test_samples(
        duplicate_values,
        difference,
        PermutationTestOptions::default(),
    )
    .unwrap();
    assert!(result.exact);
    assert_eq!(result.null_distribution, vec![0.0; 6]);
    assert_eq!(result.pvalue, 1.0);
}

#[test]
fn exact_independent_test_supports_unequal_and_multiple_samples() {
    let unequal: &[&[f64]] = &[&[1.0, 2.0], &[3.0, 5.0, 8.0]];
    let result = permutation_test_samples(
        unequal,
        difference,
        options(PermutationType::Independent, PermutationAlternative::Less),
    )
    .unwrap();
    assert_eq!(result.null_distribution.len(), 10);
    assert_eq!(result.pvalue, 0.1);
    let three: &[&[f64]] = &[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 8.0]];
    let result = permutation_test_samples(
        three,
        |samples| {
            assert_eq!(samples.len(), 3);
            assert!(samples.iter().all(|sample| sample.len() == 2));
            let all: Vec<_> = samples
                .iter()
                .flat_map(|sample| sample.iter().copied())
                .collect();
            assert_eq!(sorted(all), vec![1.0, 2.0, 3.0, 4.0, 5.0, 8.0]);
            mean(samples[0]) - 2.0 * mean(samples[1]) + 3.0 * mean(samples[2])
        },
        options(
            PermutationType::Independent,
            PermutationAlternative::TwoSided,
        ),
    )
    .unwrap();
    assert!(result.exact);
    assert_eq!(result.null_distribution.len(), 90);
}

#[test]
fn exact_paired_samples_enumerate_signs_and_sample_label_permutations() {
    let one: &[&[f64]] = &[&[1.0, 2.0, 4.0]];
    let sign_result = permutation_test_samples(
        one,
        |samples| samples[0].iter().sum(),
        options(PermutationType::Samples, PermutationAlternative::Greater),
    )
    .unwrap();
    assert!(sign_result.exact);
    assert_eq!(sign_result.pvalue, 1.0 / 8.0);
    assert_eq!(
        sorted(sign_result.null_distribution),
        vec![-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]
    );
    let paired: &[&[f64]] = &[&[1.0, 2.0, 3.0], &[2.0, 4.0, 8.0]];
    let result = permutation_test_samples(
        paired,
        difference,
        options(PermutationType::Samples, PermutationAlternative::Less),
    )
    .unwrap();
    assert_eq!(result.null_distribution.len(), 8);
    assert_eq!(result.pvalue, 1.0 / 8.0);
    let three: &[&[f64]] = &[&[1.0, 2.0], &[3.0, 5.0], &[7.0, 11.0]];
    let result = PermutationMethod::new(36, Some(3), 19)
        .unwrap()
        .test_samples(
            three,
            |samples| {
                for index in 0..2 {
                    let column: Vec<_> = samples.iter().map(|sample| sample[index]).collect();
                    let original: Vec<_> = three.iter().map(|sample| sample[index]).collect();
                    assert_eq!(sorted(column), sorted(original));
                }
                mean(samples[0]) - 2.0 * mean(samples[1]) + 3.0 * mean(samples[2])
            },
            PermutationType::Samples,
            PermutationAlternative::TwoSided,
        )
        .unwrap();
    assert!(result.exact);
    assert_eq!(result.null_distribution.len(), 36);
}

#[test]
fn exact_pairings_permute_every_sample_and_allow_one_sample() {
    let one: &[&[f64]] = &[&[1.0, 3.0, 2.0]];
    let result = permutation_test_samples(
        one,
        |samples| {
            samples[0]
                .iter()
                .enumerate()
                .map(|(i, x)| (i + 1) as f64 * x)
                .sum()
        },
        options(PermutationType::Pairings, PermutationAlternative::Greater),
    )
    .unwrap();
    assert!(result.exact);
    assert_eq!(result.null_distribution.len(), 6);
    assert_eq!(
        sorted(result.null_distribution),
        vec![10.0, 11.0, 11.0, 13.0, 13.0, 14.0]
    );
    assert_eq!(result.pvalue, 0.5);
    let two: &[&[f64]] = &[&[1.0, 2.0, 3.0], &[4.0, 8.0, 5.0]];
    let result = permutation_test_samples(
        two,
        |samples| samples[0][0] + 2.0 * samples[1][1],
        options(PermutationType::Pairings, PermutationAlternative::TwoSided),
    )
    .unwrap();
    assert_eq!(result.null_distribution.len(), 36);
    let mut expected = Vec::new();
    for &x in two[0] {
        for &y in two[1] {
            expected.extend([x + 2.0 * y; 4]);
        }
    }
    assert_eq!(sorted(result.null_distribution), sorted(expected));
}

#[test]
fn floating_point_ties_use_scipy_relative_tolerance() {
    let data: &[&[f64]] = &[&[0.0, 0.0], &[1.0, 1.0]];
    for alternative in [
        PermutationAlternative::Less,
        PermutationAlternative::Greater,
        PermutationAlternative::TwoSided,
    ] {
        let result = permutation_test_samples(
            data,
            |samples| 1.0 + f64::EPSILON * samples[0].iter().sum::<f64>(),
            options(PermutationType::Independent, alternative),
        )
        .unwrap();
        assert_eq!(result.statistic, 1.0);
        assert!(
            result
                .null_distribution
                .iter()
                .any(|&value| value > result.statistic)
        );
        assert_eq!(result.pvalue, 1.0);
    }
}

fn check_random_pvalues(data: &[&[f64]], kind: PermutationType) {
    let base = PermutationTestOptions {
        permutation_type: kind,
        n_resamples: 5,
        rng: 912_317,
        ..PermutationTestOptions::default()
    };
    let statistic = |samples: &[&[f64]]| {
        if kind == PermutationType::Independent {
            return samples
                .iter()
                .enumerate()
                .map(|(row, sample)| (row + 1) as f64 * mean(sample))
                .sum();
        }
        samples
            .iter()
            .enumerate()
            .map(|(row, sample)| {
                sample
                    .iter()
                    .enumerate()
                    .map(|(column, &value)| (row + 1) as f64 * (column + 1) as f64 * value)
                    .sum::<f64>()
            })
            .sum::<f64>()
    };
    let first = permutation_test_samples(data, statistic, base).unwrap();
    let second = permutation_test_samples(data, statistic, base).unwrap();
    assert_eq!(
        first, second,
        "same seed must reproduce the complete null distribution"
    );
    assert!(!first.exact);
    assert_eq!(first.null_distribution.len(), 5);
    let tolerance = 100.0 * f64::EPSILON * first.statistic.abs();
    let less = first
        .null_distribution
        .iter()
        .filter(|&&value| value <= first.statistic + tolerance)
        .count();
    let greater = first
        .null_distribution
        .iter()
        .filter(|&&value| value >= first.statistic - tolerance)
        .count();
    let less_p = (less + 1) as f64 / 6.0;
    let greater_p = (greater + 1) as f64 / 6.0;
    for (alternative, expected) in [
        (PermutationAlternative::Less, less_p),
        (PermutationAlternative::Greater, greater_p),
        (
            PermutationAlternative::TwoSided,
            (2.0 * less_p.min(greater_p)).min(1.0),
        ),
    ] {
        let result = permutation_test_samples(
            data,
            statistic,
            PermutationTestOptions {
                alternative,
                ..base
            },
        )
        .unwrap();
        assert_eq!(result.null_distribution, first.null_distribution);
        assert_eq!(result.pvalue, expected);
        assert!(result.pvalue > 0.0 && result.pvalue <= 1.0);
    }
}

#[test]
fn randomized_tests_are_reproducible_and_use_the_plus_one_correction() {
    check_random_pvalues(&[&[1.0, 2.0], &[3.0, 4.0]], PermutationType::Independent);
    check_random_pvalues(&[&[1.0, 2.0, 4.0]], PermutationType::Samples);
    check_random_pvalues(
        &[&[1.0, 2.0, 3.0], &[2.0, 4.0, 8.0]],
        PermutationType::Samples,
    );
    check_random_pvalues(&[&[1.0, 3.0, 2.0]], PermutationType::Pairings);
    check_random_pvalues(
        &[&[1.0, 2.0, 3.0], &[4.0, 8.0, 5.0]],
        PermutationType::Pairings,
    );
}

#[test]
fn randomized_modes_preserve_their_exchangeability_constraints() {
    let data: &[&[f64]] = &[&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]];
    for kind in [
        PermutationType::Independent,
        PermutationType::Samples,
        PermutationType::Pairings,
    ] {
        let result = permutation_test_samples(
            data,
            |samples| {
                match kind {
                    PermutationType::Independent => {
                        let pooled: Vec<_> = samples
                            .iter()
                            .flat_map(|sample| sample.iter().copied())
                            .collect();
                        assert_eq!(
                            sorted(pooled),
                            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
                        );
                    }
                    PermutationType::Samples => {
                        for index in 0..4 {
                            assert_eq!(
                                sorted(vec![samples[0][index], samples[1][index]]),
                                vec![data[0][index], data[1][index]]
                            );
                        }
                    }
                    PermutationType::Pairings => {
                        assert_eq!(sorted(samples[0].to_vec()), data[0]);
                        assert_eq!(sorted(samples[1].to_vec()), data[1]);
                    }
                }
                difference(samples)
            },
            PermutationTestOptions {
                permutation_type: kind,
                n_resamples: 7,
                rng: 57,
                ..PermutationTestOptions::default()
            },
        )
        .unwrap();
        assert!(!result.exact);
        assert_eq!(result.null_distribution.len(), 7);
    }
}

#[test]
fn invalid_inputs_fail_before_invoking_the_statistic() {
    let calls = AtomicUsize::new(0);
    let statistic = |_: &[&[f64]]| {
        calls.fetch_add(1, Ordering::Relaxed);
        0.0
    };
    let valid: &[&[f64]] = &[&[1.0, 2.0], &[3.0, 4.0]];
    assert!(matches!(
        permutation_test_samples(
            valid,
            statistic,
            PermutationTestOptions {
                n_resamples: 0,
                ..PermutationTestOptions::default()
            }
        ),
        Err(StatsError::InvalidArgument(_))
    ));
    for data in [&[][..], &[&[1.0, 2.0][..]][..]] {
        assert!(matches!(
            permutation_test_samples(data, statistic, PermutationTestOptions::default()),
            Err(StatsError::InvalidArgument(_))
        ));
    }
    assert!(matches!(
        permutation_test_samples(
            &[&[1.0], &[2.0, 3.0]],
            statistic,
            PermutationTestOptions::default()
        ),
        Err(StatsError::DataTooSmall { .. })
    ));
    for kind in [PermutationType::Samples, PermutationType::Pairings] {
        assert!(matches!(
            permutation_test_samples(
                &[],
                statistic,
                options(kind, PermutationAlternative::TwoSided)
            ),
            Err(StatsError::InvalidArgument(_))
        ));
        assert!(matches!(
            permutation_test_samples(
                &[&[1.0, 2.0], &[3.0, 4.0, 5.0]],
                statistic,
                options(kind, PermutationAlternative::TwoSided)
            ),
            Err(StatsError::ShapeMismatch {
                expected: 2,
                actual: 3
            })
        ));
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
            assert!(matches!(
                permutation_test_samples(
                    &[&[1.0, value], &[3.0, 4.0]],
                    statistic,
                    PermutationTestOptions {
                        mode,
                        ..PermutationTestOptions::default()
                    }
                ),
                Err(StatsError::NonFiniteInput { .. })
            ));
        }
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn nonfinite_statistics_return_typed_errors_including_the_method_api() {
    let data: &[&[f64]] = &[&[1.0, 2.0], &[3.0, 4.0]];
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            permutation_test_samples(data, |_| bad, PermutationTestOptions::default()),
            Err(StatsError::Arithmetic { .. })
        ));
        assert!(matches!(
            PermutationMethod::default().test(data[0], data[1], |_, _| bad),
            Err(StatsError::Arithmetic { .. })
        ));
    }
    let calls = AtomicUsize::new(0);
    let error = permutation_test_samples(
        data,
        |_| {
            let call = calls.fetch_add(1, Ordering::Relaxed);
            if call == 0 { 0.0 } else { f64::NAN }
        },
        PermutationTestOptions::default(),
    );
    assert!(matches!(error, Err(StatsError::Arithmetic { .. })));
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    let legacy = permutation_test(data[0], data[1], |_, _| 0.0, 0, 0);
    assert!(legacy.0.is_nan() && legacy.1.is_nan());
}

#[test]
fn resource_preflight_rejects_limits_and_unrepresentable_output_without_callbacks() {
    let calls = AtomicUsize::new(0);
    let statistic = |_: &[&[f64]]| {
        calls.fetch_add(1, Ordering::Relaxed);
        0.0
    };
    let data: &[&[f64]] = &[&[1.0, 2.0], &[3.0, 4.0]];
    let hardened = PermutationTestOptions {
        mode: RuntimeMode::Hardened,
        ..PermutationTestOptions::default()
    };
    let capped =
        with_hardened_max_resamples(4, || permutation_test_samples(data, statistic, hardened));
    assert!(matches!(capped, Err(StatsError::ResourceExhausted { .. })));
    let pairwise = with_hardened_stats_caps(100, 1, || {
        permutation_test_samples(data, statistic, hardened)
    });
    assert!(matches!(
        pairwise,
        Err(StatsError::ResourceExhausted { .. })
    ));
    let combined = with_hardened_stats_caps(3, 100, || {
        permutation_test_samples(data, statistic, hardened)
    });
    assert!(matches!(
        combined,
        Err(StatsError::ResourceExhausted { .. })
    ));
    let long = [1.0; 30];
    let overflow = permutation_test_samples(
        &[&long],
        statistic,
        PermutationTestOptions {
            permutation_type: PermutationType::Pairings,
            n_resamples: usize::MAX,
            ..PermutationTestOptions::default()
        },
    );
    assert!(matches!(
        overflow,
        Err(StatsError::ResourceExhausted { .. })
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
        permutation_test_samples(
            data,
            |samples| {
                assert_eq!(active_stats_mode(), mode);
                difference(samples)
            },
            PermutationTestOptions {
                mode,
                n_resamples: 6,
                ..PermutationTestOptions::default()
            },
        )
        .unwrap();
    }
}

#[test]
fn audited_wrapper_reports_failures_once_and_preserves_projection() {
    let x = [1.0, 2.0];
    let y = [3.0, 4.0];
    for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
        let ledger = sync_audit_ledger();
        let error = permutation_test_with_options(
            &x,
            &y,
            |_, _| f64::NAN,
            6,
            0,
            StatsOptions::new(mode),
            Some(&ledger),
        );
        assert!(matches!(error, Err(StatsError::Arithmetic { .. })));
        let guard = ledger.lock().unwrap();
        assert_eq!(guard.len(), 1);
        assert!(matches!(
            guard.entries()[0].action,
            AuditAction::FailClosed { .. }
        ));
    }
    let calls = AtomicUsize::new(0);
    for capped in [false, true] {
        let ledger = sync_audit_ledger();
        let statistic = |_: &[f64], _: &[f64]| {
            calls.fetch_add(1, Ordering::Relaxed);
            0.0
        };
        let error = with_hardened_max_resamples(if capped { 5 } else { 100 }, || {
            permutation_test_with_options(
                if capped { &x } else { &[1.0][..] },
                &y,
                statistic,
                6,
                0,
                StatsOptions::new(RuntimeMode::Hardened),
                Some(&ledger),
            )
        });
        assert!(error.is_err());
        let guard = ledger.lock().unwrap();
        assert_eq!(guard.len(), 1);
        assert!(matches!(
            guard.entries()[0].action,
            AuditAction::FailClosed { .. }
        ));
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let ledger = sync_audit_ledger();
    let projected_error = permutation_test_with_options(
        &[1.0, f64::NAN],
        &y,
        |_, _| {
            calls.fetch_add(1, Ordering::Relaxed);
            0.0
        },
        6,
        0,
        StatsOptions::new(RuntimeMode::Hardened).with_non_finite_projection(true),
        Some(&ledger),
    );
    assert!(matches!(
        projected_error,
        Err(StatsError::DataTooSmall {
            required: 2,
            got: 1
        })
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    {
        let guard = ledger.lock().unwrap();
        assert_eq!(guard.len(), 2);
        assert!(matches!(
            guard.entries()[0].action,
            AuditAction::BoundedRecovery { .. }
        ));
        assert!(matches!(
            guard.entries()[1].action,
            AuditAction::FailClosed { .. }
        ));
    }
    let ledger = sync_audit_ledger();
    let projected = permutation_test_with_options(
        &[1.0, f64::NAN, 2.0],
        &y,
        |x, y| mean(x) - mean(y),
        6,
        0,
        StatsOptions::new(RuntimeMode::Hardened).with_non_finite_projection(true),
        Some(&ledger),
    )
    .unwrap();
    assert_eq!(projected, (-2.0, 1.0 / 3.0));
    let guard = ledger.lock().unwrap();
    assert_eq!(guard.len(), 1);
    assert!(matches!(
        guard.entries()[0].action,
        AuditAction::BoundedRecovery { .. }
    ));
}

fn monte_carlo_draw(seed: u64) -> Vec<f64> {
    vec![
        ((seed >> 16) % 11) as f64 - 5.0,
        ((seed >> 32) % 13) as f64 - 6.0,
    ]
}

#[test]
fn monte_carlo_tails_are_translation_invariant_and_keep_the_existing_seed_stream() {
    for offset in [0.0, 10.0, -10.0] {
        for (alternative, expected) in [("less", 0.125), ("greater", 1.0), ("two-sided", 0.25)] {
            let result = monte_carlo_test(
                &[-2.0, -2.0],
                |_| vec![0.0, 0.0],
                |data| mean(data) + offset,
                7,
                0,
                alternative,
            );
            assert_eq!(result.statistic, -2.0 + offset);
            assert_eq!(result.null_distribution, vec![offset; 7]);
            assert_eq!(result.pvalue, expected);
        }
    }
    let mut seed = 37_u64;
    let expected: Vec<f64> = (0..23)
        .map(|_| {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            mean(&monte_carlo_draw(seed)) + 10.0
        })
        .collect();
    for alternative in ["less", "greater", "two-sided"] {
        let result = monte_carlo_test(
            &[-3.0, -1.0],
            monte_carlo_draw,
            |data| mean(data) + 10.0,
            23,
            37,
            alternative,
        );
        assert_eq!(result.null_distribution, expected);
    }
    for alternative in ["less", "greater", "two-sided"] {
        let result = monte_carlo_test(
            &[0.0, 0.0],
            |_| vec![1.0, 1.0],
            |data| 1.0 + f64::EPSILON * data.iter().sum::<f64>(),
            7,
            0,
            alternative,
        );
        assert!(
            result
                .null_distribution
                .iter()
                .all(|&value| value > result.statistic)
        );
        assert_eq!(result.pvalue, 1.0);
    }
}

const MONTE_CARLO_ORACLE: &str = r#"
import os
import numpy as np
import scipy
from scipy import stats

print('versions', scipy.__version__, np.__version__)
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3')
for case in range(5):
    for alternative in ('two-sided', 'less', 'greater'):
        state = 37
        def rvs(size):
            global state
            if case < 2:
                return np.zeros(size)
            if case == 4:
                return np.ones(size)
            out = np.empty(size)
            for row in range(size[0]):
                state = (state*6364136223846793005 + 1) & ((1 << 64) - 1)
                out[row] = [(state >> 16) % 11 - 5, (state >> 32) % 13 - 6]
            return out
        offset = 10.0 if case in (1, 3) else 0.0
        def statistic(data):
            if case == 4:
                return 1.0 + np.finfo(float).eps*np.sum(data)
            return np.mean(data) + offset
        data = np.array([0.0, 0.0] if case == 4 else [-2.0, -2.0] if case < 2 else [-3.0, -1.0])
        result = stats.monte_carlo_test(
            data, rvs, statistic, vectorized=False,
            n_resamples=7 if case < 2 or case == 4 else 23,
            alternative=alternative)
        print(case, alternative, format(result.statistic, '.17g'),
              format(result.pvalue, '.17g'), len(result.null_distribution))
        print(' '.join(format(value, '.17g') for value in result.null_distribution))
"#;

#[test]
fn scipy_monte_carlo_distributions_and_tails_match() {
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let output = match Command::new(&python)
        .args(["-c", MONTE_CARLO_ORACLE])
        .output()
    {
        Ok(output) => output,
        Err(error) if !required => {
            eprintln!("SKIP Monte Carlo SciPy oracle: cannot start {python}: {error}");
            return;
        }
        Err(error) => panic!("required Monte Carlo SciPy oracle could not start {python}: {error}"),
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !required
            && (stderr.contains("ModuleNotFoundError: No module named 'scipy'")
                || stderr.contains("ModuleNotFoundError: No module named 'numpy'"))
        {
            eprintln!("SKIP Monte Carlo SciPy oracle: {stderr}");
            return;
        }
        panic!("Monte Carlo SciPy oracle failed: {stderr}");
    }
    let text = String::from_utf8(output.stdout).unwrap();
    let mut lines = text.lines();
    let versions = lines.next().expect("oracle version line");
    assert!(versions.starts_with("versions "));
    eprintln!("Monte Carlo oracle {versions}");
    let mut comparisons = 0;
    while let Some(header) = lines.next() {
        let fields: Vec<_> = header.split_whitespace().collect();
        assert_eq!(fields.len(), 5, "oracle header: {header}");
        let case: usize = fields[0].parse().unwrap();
        let expected_statistic: f64 = fields[2].parse().unwrap();
        let expected_pvalue: f64 = fields[3].parse().unwrap();
        let count: usize = fields[4].parse().unwrap();
        let expected_null: Vec<f64> = lines
            .next()
            .expect("oracle null distribution")
            .split_whitespace()
            .map(|value| value.parse().unwrap())
            .collect();
        assert_eq!(expected_null.len(), count);
        let data = if case == 4 {
            [0.0, 0.0]
        } else if case < 2 {
            [-2.0, -2.0]
        } else {
            [-3.0, -1.0]
        };
        let offset = if case == 1 || case == 3 { 10.0 } else { 0.0 };
        let actual = monte_carlo_test(
            &data,
            |seed| {
                if case < 2 {
                    vec![0.0, 0.0]
                } else if case == 4 {
                    vec![1.0, 1.0]
                } else {
                    monte_carlo_draw(seed)
                }
            },
            |data| {
                if case == 4 {
                    1.0 + f64::EPSILON * data.iter().sum::<f64>()
                } else {
                    mean(data) + offset
                }
            },
            count,
            37,
            fields[1],
        );
        close(actual.statistic, expected_statistic, 2e-14);
        close(actual.pvalue, expected_pvalue, 2e-14);
        assert_eq!(actual.null_distribution, expected_null);
        comparisons += 1;
    }
    assert_eq!(comparisons, 15);
    eprintln!("matched {comparisons} SciPy Monte Carlo distribution and tail cases");
}

struct Case {
    kind: PermutationType,
    data: Vec<Vec<f64>>,
    permutations: usize,
}

fn oracle_cases() -> Vec<Case> {
    vec![
        Case {
            kind: PermutationType::Independent,
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            permutations: 6,
        },
        Case {
            kind: PermutationType::Independent,
            data: vec![vec![1.0, 2.0], vec![3.0, 5.0, 8.0]],
            permutations: 10,
        },
        Case {
            kind: PermutationType::Independent,
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 8.0]],
            permutations: 90,
        },
        Case {
            kind: PermutationType::Samples,
            data: vec![vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 8.0]],
            permutations: 8,
        },
        Case {
            kind: PermutationType::Samples,
            data: vec![vec![1.0, 2.0], vec![3.0, 5.0], vec![7.0, 11.0]],
            permutations: 36,
        },
        Case {
            kind: PermutationType::Samples,
            data: vec![vec![1.0, 2.0, 4.0]],
            permutations: 8,
        },
        Case {
            kind: PermutationType::Pairings,
            data: vec![vec![1.0, 3.0, 2.0]],
            permutations: 6,
        },
        Case {
            kind: PermutationType::Pairings,
            data: vec![vec![1.0, 2.0, 3.0], vec![4.0, 8.0, 5.0]],
            permutations: 36,
        },
        Case {
            kind: PermutationType::Pairings,
            data: vec![vec![1.0, 2.0, 3.0], vec![4.0, 8.0, 5.0]],
            permutations: 36,
        },
        Case {
            kind: PermutationType::Independent,
            data: vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            permutations: 6,
        },
    ]
}

fn oracle_statistic(case: usize, samples: &[&[f64]]) -> f64 {
    match case {
        0 => difference(samples) + 10.0,
        1 | 3 => difference(samples),
        2 | 4 => mean(samples[0]) - 2.0 * mean(samples[1]) + 3.0 * mean(samples[2]),
        5 => samples[0].iter().sum(),
        6 => samples[0]
            .iter()
            .enumerate()
            .map(|(i, &value)| (i + 1) as f64 * value)
            .sum(),
        7 => {
            let mx = mean(samples[0]);
            let my = mean(samples[1]);
            let numerator: f64 = samples[0]
                .iter()
                .zip(samples[1])
                .map(|(&x, &y)| (x - mx) * (y - my))
                .sum();
            let xx: f64 = samples[0].iter().map(|&x| (x - mx).powi(2)).sum();
            let yy: f64 = samples[1].iter().map(|&y| (y - my).powi(2)).sum();
            numerator / (xx * yy).sqrt()
        }
        8 => samples[0][0] + 2.0 * samples[1][1],
        9 => 1.0 + f64::EPSILON * samples[0].iter().sum::<f64>(),
        _ => unreachable!(),
    }
}

const ORACLE: &str = r#"
import os
import numpy as np
import scipy
from scipy import stats

print('versions', scipy.__version__, np.__version__)
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3'), \
        f'required oracle is SciPy 1.17.1 / NumPy 2.4.3, got {scipy.__version__} / {np.__version__}'
cases = [
    ('independent', [[1, 2], [3, 4]]),
    ('independent', [[1, 2], [3, 5, 8]]),
    ('independent', [[1, 2], [3, 4], [5, 8]]),
    ('samples', [[1, 2, 3], [2, 4, 8]]),
    ('samples', [[1, 2], [3, 5], [7, 11]]),
    ('samples', [[1, 2, 4]]),
    ('pairings', [[1, 3, 2]]),
    ('pairings', [[1, 2, 3], [4, 8, 5]]),
    ('pairings', [[1, 2, 3], [4, 8, 5]]),
    ('independent', [[0, 0], [1, 1]]),
]

def statistic(case, samples):
    if case == 0:
        return np.mean(samples[0]) - np.mean(samples[1]) + 10
    if case in (1, 3):
        return np.mean(samples[0]) - np.mean(samples[1])
    if case in (2, 4):
        return np.mean(samples[0]) - 2*np.mean(samples[1]) + 3*np.mean(samples[2])
    if case == 5:
        return np.sum(samples[0])
    if case == 6:
        return np.dot(np.arange(1, 4), samples[0])
    if case == 7:
        return stats.pearsonr(samples[0], samples[1]).statistic
    if case == 8:
        return samples[0][0] + 2*samples[1][1]
    return 1.0 + np.finfo(float).eps*np.sum(samples[0])

for case, (kind, raw) in enumerate(cases):
    data = tuple(np.array(row, dtype=float) for row in raw)
    for alternative in ('two-sided', 'less', 'greater'):
        result = stats.permutation_test(
            data, lambda *samples: statistic(case, samples),
            permutation_type=kind, alternative=alternative,
            n_resamples=np.inf, vectorized=False)
        values = result.null_distribution
        print(case, alternative, format(result.statistic, '.17g'),
              format(result.pvalue, '.17g'), len(values))
        print(' '.join(format(value, '.17g') for value in values))
"#;

#[test]
fn scipy_exact_permutation_distributions_and_tails_match() {
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let output = match Command::new(&python).args(["-c", ORACLE]).output() {
        Ok(output) => output,
        Err(error) if !required => {
            eprintln!("SKIP permutation SciPy oracle: cannot start {python}: {error}");
            return;
        }
        Err(error) => panic!("required permutation SciPy oracle could not start {python}: {error}"),
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !required
            && (stderr.contains("ModuleNotFoundError: No module named 'scipy'")
                || stderr.contains("ModuleNotFoundError: No module named 'numpy'"))
        {
            eprintln!("SKIP permutation SciPy oracle: {stderr}");
            return;
        }
        panic!("permutation SciPy oracle failed: {stderr}");
    }
    let text = String::from_utf8(output.stdout).unwrap();
    let mut lines = text.lines();
    let versions = lines.next().expect("oracle version line");
    assert!(versions.starts_with("versions "));
    eprintln!("permutation oracle {versions}");
    let cases = oracle_cases();
    let mut comparisons = 0;
    while let Some(header) = lines.next() {
        let fields: Vec<_> = header.split_whitespace().collect();
        assert_eq!(fields.len(), 5, "oracle header: {header}");
        let index: usize = fields[0].parse().unwrap();
        let alternative = match fields[1] {
            "two-sided" => PermutationAlternative::TwoSided,
            "less" => PermutationAlternative::Less,
            "greater" => PermutationAlternative::Greater,
            other => panic!("unexpected alternative {other}"),
        };
        let expected_statistic: f64 = fields[2].parse().unwrap();
        let expected_pvalue: f64 = fields[3].parse().unwrap();
        let count: usize = fields[4].parse().unwrap();
        let expected_null: Vec<f64> = lines
            .next()
            .expect("oracle null distribution")
            .split_whitespace()
            .map(|value| value.parse().unwrap())
            .collect();
        let case = &cases[index];
        assert_eq!(count, case.permutations);
        assert_eq!(expected_null.len(), count);
        let data: Vec<&[f64]> = case.data.iter().map(Vec::as_slice).collect();
        for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
            let actual: PermutationTestResult = permutation_test_samples(
                &data,
                |samples| oracle_statistic(index, samples),
                PermutationTestOptions {
                    mode,
                    ..options(case.kind, alternative)
                },
            )
            .unwrap();
            assert!(actual.exact, "case {index} {alternative:?}");
            assert_eq!(actual.null_distribution.len(), count);
            close(actual.statistic, expected_statistic, 2e-14);
            close(actual.pvalue, expected_pvalue, 2e-14);
            for (actual, expected) in sorted(actual.null_distribution)
                .into_iter()
                .zip(sorted(expected_null.clone()))
            {
                close(actual, expected, 2e-13);
            }
        }
        comparisons += 1;
    }
    assert_eq!(comparisons, 30);
    eprintln!("matched {comparisons} SciPy exact permutation cases in both runtime modes");
}
