#![forbid(unsafe_code)]
//! Metamorphic invariants for `fsci_signal::detrend`.
//!
//! Resolves [frankenscipy-fsa50]. Direct reference coverage already
//! checks fixed examples. This harness adds transformation properties
//! that must hold for SciPy-shaped constant and linear detrending:
//!
//!   1. Constant detrend is idempotent and leaves zero mean.
//!   2. Linear detrend removes any affine sequence to near zero.
//!   3. Adding an affine trend before linear detrend preserves the
//!      residual produced from the untrended data.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{DetrendType, detrend};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-018";
const ZERO_TOL: f64 = 1.0e-10;
const RESIDUAL_TOL: f64 = 1.0e-10;

#[derive(Debug, Clone, Serialize)]
struct CaseLog {
    case_id: String,
    invariant: String,
    detail: String,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct MetamorphicLog {
    test_id: String,
    case_count: usize,
    pass_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseLog>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("fixtures/artifacts/{PACKET_ID}/metamorphic"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create signal detrend metamorphic output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &MetamorphicLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize signal detrend log");
    fs::write(path, json).expect("write signal detrend log");
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().map(|v| v.abs()).fold(0.0_f64, nan_max)
}

fn zero_tolerance(values: &[f64]) -> f64 {
    ZERO_TOL.max(max_abs(values) * 1.0e-15)
}

fn max_pair_abs_diff(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() {
        return f64::INFINITY;
    }

    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, nan_max)
}

/// `f64::max` returns the other operand when one is NaN, so folding residuals with it reads a
/// NaN entry as agreement. This keeps the NaN, and `NaN <= tol` then fails the case.
fn nan_max(acc: f64, d: f64) -> f64 {
    if acc.is_nan() || d.is_nan() {
        f64::NAN
    } else {
        acc.max(d)
    }
}

fn affine_sequence(len: usize, slope: f64, intercept: f64) -> Vec<f64> {
    (0..len).map(|i| slope * i as f64 + intercept).collect()
}

fn add_affine(data: &[f64], slope: f64, intercept: f64) -> Vec<f64> {
    data.iter()
        .enumerate()
        .map(|(i, value)| value + slope * i as f64 + intercept)
        .collect()
}

fn push_case(cases: &mut Vec<CaseLog>, case_id: &str, invariant: &str, detail: String, pass: bool) {
    cases.push(CaseLog {
        case_id: case_id.to_string(),
        invariant: invariant.to_string(),
        detail,
        pass,
    });
}

#[test]
fn metamorphic_signal_detrend() {
    let start = Instant::now();
    let mut cases = Vec::new();

    let constant_fixtures: Vec<(&str, Vec<f64>)> = vec![
        ("mixed_offsets_n5", vec![3.0, -1.0, 2.0, 10.0, 6.0]),
        (
            "wide_dynamic_range_n7",
            vec![1.0e6, -2.5e5, 3.75e5, 9.0e5, -8.0e5, 4.5e5, 1.25e5],
        ),
        (
            "oscillating_n9",
            (0..9)
                .map(|i| 4.0 + (i as f64 * 0.7).sin() - (i as f64 * 0.3).cos())
                .collect(),
        ),
    ];

    for (case_id, data) in &constant_fixtures {
        let once = detrend(data, DetrendType::Constant).expect("constant detrend once");
        let twice = detrend(&once, DetrendType::Constant).expect("constant detrend twice");
        let mean_abs = mean(&once).abs();
        let idempotent_diff = max_pair_abs_diff(&once, &twice);

        push_case(
            &mut cases,
            case_id,
            "constant_zero_mean",
            format!("abs_mean={mean_abs}"),
            mean_abs <= ZERO_TOL,
        );
        push_case(
            &mut cases,
            case_id,
            "constant_idempotent",
            format!("max_diff={idempotent_diff}"),
            idempotent_diff <= RESIDUAL_TOL,
        );
    }

    let affine_fixtures = vec![
        ("positive_slope_n8", 8_usize, 2.5_f64, -7.0_f64),
        ("negative_slope_n13", 13, -0.75, 4.25),
        ("flat_affine_n3", 3, 0.0, 11.0),
        ("large_intercept_n21", 21, 1.0e-3, 1.0e9),
    ];

    for (case_id, len, slope, intercept) in affine_fixtures {
        let data = affine_sequence(len, slope, intercept);
        let residual = detrend(&data, DetrendType::Linear).expect("linear detrend affine data");
        let residual_max = max_abs(&residual);
        let allowed = zero_tolerance(&data);
        push_case(
            &mut cases,
            case_id,
            "linear_removes_affine_sequence",
            format!(
                "max_abs={residual_max}, tolerance={allowed}, len={len}, slope={slope}, intercept={intercept}"
            ),
            residual_max <= allowed,
        );
    }

    let residual_fixtures: Vec<(&str, Vec<f64>, f64, f64)> = vec![
        (
            "curved_residual_n8",
            vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0],
            3.5,
            -12.0,
        ),
        (
            "oscillating_residual_n16",
            (0..16)
                .map(|i| (i as f64 * 0.45).sin() + 0.2 * (i as f64 * 1.7).cos())
                .collect(),
            -1.25,
            8.0,
        ),
        (
            "signed_residual_n9",
            vec![2.0, -3.5, 1.25, 7.0, -2.0, 0.5, 6.25, -4.0, 3.75],
            0.125,
            100.0,
        ),
    ];

    for (case_id, residual_data, slope, intercept) in residual_fixtures {
        let base_residual =
            detrend(&residual_data, DetrendType::Linear).expect("linear detrend base residual");
        let trended = add_affine(&residual_data, slope, intercept);
        let trended_residual =
            detrend(&trended, DetrendType::Linear).expect("linear detrend trended residual");
        let residual_diff = max_pair_abs_diff(&base_residual, &trended_residual);

        push_case(
            &mut cases,
            case_id,
            "linear_residual_invariant_to_added_affine",
            format!("max_diff={residual_diff}, slope={slope}, intercept={intercept}"),
            residual_diff <= RESIDUAL_TOL,
        );
    }

    let pass_count = cases.iter().filter(|case| case.pass).count();
    let all_pass = pass_count == cases.len();
    let log = MetamorphicLog {
        test_id: "metamorphic_signal_detrend".into(),
        case_count: cases.len(),
        pass_count,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: cases.clone(),
    };

    emit_log(&log);

    for case in &cases {
        if !case.pass {
            eprintln!(
                "signal detrend metamorphic fail: {} {} - {}",
                case.case_id, case.invariant, case.detail
            );
        }
    }

    assert!(
        all_pass,
        "signal detrend metamorphic failed: {} of {} cases pass",
        pass_count,
        cases.len()
    );
}
