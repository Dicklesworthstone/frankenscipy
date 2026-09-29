#![forbid(unsafe_code)]
//! Property-based coverage for fsci_interpolate poly extras.
//!
//! Resolves [frankenscipy-v5j6s]. Three uncovered polynomial helpers:
//!   * polyval_der(coeffs, x, der): derivatives at x (descending-order
//!     coeffs, matches the polyval convention)
//!   * ratval(p, q, x): rational function p(x)/q(x) (ASCENDING-order
//!     coeffs; differs from polyval's descending convention — verified
//!     against fsci's source at crates/fsci-interpolate/src/lib.rs:3527)
//!   * polyroots(coeffs): real-root finder; descending-order coeffs
//!     matching the polyval convention

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_interpolate::{polyroots, polyval_der, ratval};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create poly_extras diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

#[test]
fn diff_interpolate_poly_extras() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    // Each check compares an infallible fsci value against its analytic answer, so every check
    // is one compared case of its function's arm.
    let mut ledger = CompareLedger::new(
        "diff_interpolate_poly_extras",
        &["polyval_der", "ratval", "polyroots"],
    );
    let mut check = |arm: &str, id: &str, ok: bool, note: String| {
        ledger.compared(arm, id, ok);
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    // === polyval_der ===
    // p(x) = x² - 5x + 6 (descending: [1, -5, 6])
    // p'(x) = 2x - 5;  p''(x) = 2
    {
        let coeffs = vec![1.0_f64, -5.0, 6.0];
        let derivs = polyval_der(&coeffs, 4.0, 2);
        // p(4) = 16 - 20 + 6 = 2; p'(4) = 3; p''(4) = 2
        check(
            "polyval_der",
            "polyval_der_quadratic_at_x4_returns_3_values",
            derivs.len() == 3
                && (derivs[0] - 2.0).abs() < ABS_TOL
                && (derivs[1] - 3.0).abs() < ABS_TOL
                && (derivs[2] - 2.0).abs() < ABS_TOL,
            format!("derivs={derivs:?}"),
        );
    }
    // Constant polynomial [7]: p(x) = 7, all derivs are 0
    {
        let coeffs = vec![7.0_f64];
        let derivs = polyval_der(&coeffs, 3.0, 2);
        check(
            "polyval_der",
            "polyval_der_constant_higher_order_zero",
            derivs.len() == 3
                && (derivs[0] - 7.0).abs() < ABS_TOL
                && derivs[1].abs() < ABS_TOL
                && derivs[2].abs() < ABS_TOL,
            format!("derivs={derivs:?}"),
        );
    }
    // Cubic [1, 0, 0, 0]: p(x) = x³
    // p(2) = 8, p'(2) = 12, p''(2) = 12, p'''(2) = 6
    {
        let coeffs = vec![1.0_f64, 0.0, 0.0, 0.0];
        let derivs = polyval_der(&coeffs, 2.0, 3);
        check(
            "polyval_der",
            "polyval_der_cubic_x3_at_x2",
            derivs.len() == 4
                && (derivs[0] - 8.0).abs() < ABS_TOL
                && (derivs[1] - 12.0).abs() < ABS_TOL
                && (derivs[2] - 12.0).abs() < ABS_TOL
                && (derivs[3] - 6.0).abs() < ABS_TOL,
            format!("derivs={derivs:?}"),
        );
    }
    // Empty coeffs: returns all-zero vector of length der+1
    {
        let derivs = polyval_der(&[], 1.0, 2);
        check(
            "polyval_der",
            "polyval_der_empty_returns_zeros",
            derivs.len() == 3 && derivs.iter().all(|&v| v == 0.0),
            format!("derivs={derivs:?}"),
        );
    }

    // === ratval (ASCENDING-order coefficients) ===
    // f(x) = (1 + 2x) / (1 + x)
    // f(0) = 1 / 1 = 1
    // f(1) = 3 / 2 = 1.5
    // f(2) = 5 / 3 ≈ 1.6667
    {
        let p = vec![1.0_f64, 2.0]; // 1 + 2x
        let q = vec![1.0_f64, 1.0]; // 1 + x
        check(
            "ratval",
            "ratval_simple_at_zero",
            (ratval(&p, &q, 0.0) - 1.0).abs() < ABS_TOL,
            String::new(),
        );
        check(
            "ratval",
            "ratval_simple_at_one",
            (ratval(&p, &q, 1.0) - 1.5).abs() < ABS_TOL,
            String::new(),
        );
        check(
            "ratval",
            "ratval_simple_at_two",
            (ratval(&p, &q, 2.0) - 5.0 / 3.0).abs() < ABS_TOL,
            String::new(),
        );
    }
    // Denominator zero → NaN
    {
        let p = vec![1.0_f64, 0.0];
        let q = vec![0.0_f64, 0.0]; // q(x) = 0
        let v = ratval(&p, &q, 1.0);
        check(
            "ratval",
            "ratval_zero_denom_nan",
            v.is_nan(),
            format!("v={v}"),
        );
    }
    // q has only constant term 5: f(x) = p(x) / 5
    {
        let p = vec![10.0_f64, 0.0, 1.0]; // 10 + x²
        let q = vec![5.0_f64];
        // f(3) = (10 + 9) / 5 = 19/5 = 3.8
        check(
            "ratval",
            "ratval_constant_denom",
            (ratval(&p, &q, 3.0) - 3.8).abs() < ABS_TOL,
            String::new(),
        );
    }

    // === polyroots ===
    // Linear: x - 3 = 0 (descending: [1, -3]) → {3}
    {
        let roots = polyroots(&[1.0_f64, -3.0]);
        check(
            "polyroots",
            "polyroots_linear",
            roots.len() == 1 && (roots[0] - 3.0).abs() < ABS_TOL,
            format!("roots={roots:?}"),
        );
    }
    // Quadratic: x² - 5x + 6 = 0 → {2, 3}
    {
        let roots = polyroots(&[1.0_f64, -5.0, 6.0]);
        let set: HashSet<i64> = roots.iter().map(|r| r.round() as i64).collect();
        let expected: HashSet<i64> = [2_i64, 3].iter().copied().collect();
        check(
            "polyroots",
            "polyroots_quadratic_distinct",
            roots.len() == 2 && set == expected,
            format!("roots={roots:?}"),
        );
    }
    // Quadratic with no real roots: x² + 1 = 0 → []
    {
        let roots = polyroots(&[1.0_f64, 0.0, 1.0]);
        check(
            "polyroots",
            "polyroots_no_real_roots_empty",
            roots.is_empty(),
            format!("roots={roots:?}"),
        );
    }
    // Constant returns no roots
    {
        let roots = polyroots(&[5.0_f64]);
        check(
            "polyroots",
            "polyroots_constant_no_roots",
            roots.is_empty(),
            format!("roots={roots:?}"),
        );
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_interpolate_poly_extras".into(),
        category: "fsci_interpolate::{polyval_der, ratval, polyroots} coverage".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("poly_extras mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(
        all_pass,
        "poly extras coverage failed: {} cases",
        diffs.len()
    );
    // The checks are inline, not a case vector: polyval_der and polyroots each have 4 designed
    // checks and ratval 5, so every arm must compare at least 4.
    ledger.finish(4);
}
