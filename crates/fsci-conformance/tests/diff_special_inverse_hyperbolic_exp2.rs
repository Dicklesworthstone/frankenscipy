#![forbid(unsafe_code)]
//! Cover fsci_special::{arcsinh, arccosh, arctanh, exp2_iterated}.
//!
//! Resolves [frankenscipy-tfced]. These are scalar wrappers over std
//! f64 math methods (arcsinh = asinh, arccosh = acosh, arctanh = atanh)
//! plus exp2_iterated which computes exp(exp(x)) — closed-form
//! comparison against the same f64 primitives.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_special::convenience::exp2_iterated;
use fsci_special::{arccosh, arcsinh, arctanh};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-14;
/// One ledger arm per fsci function; the reference side is the std f64 primitive or the
/// round-trip input.
const ARMS: [&str; 4] = ["arcsinh", "arccosh", "arctanh", "exp2_iterated"];

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    actual: f64,
    expected: f64,
    abs_diff: f64,
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
    fs::create_dir_all(output_dir()).expect("create inv_hyper diff dir");
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
fn diff_special_inverse_hyperbolic_exp2() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut ledger = CompareLedger::new("diff_special_inverse_hyperbolic_exp2", &ARMS);
    let mut check = |arm: &str, id: &str, actual: f64, expected: f64| {
        // Every reference is finite; a non-finite fsci value is recorded as an fsci failure.
        let Some((expected, actual)) = ledger.pair(arm, id, Some(expected), Some(actual)) else {
            return;
        };
        let abs_diff = (actual - expected).abs();
        ledger.compared(arm, id, abs_diff <= ABS_TOL);
        diffs.push(CaseDiff {
            case_id: id.into(),
            actual,
            expected,
            abs_diff,
            pass: abs_diff <= ABS_TOL,
            note: String::new(),
        });
    };

    // arcsinh: defined for all real x; identity arcsinh(sinh(x)) = x
    let arcsinh_xs = [-3.0_f64, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 100.0];
    for &x in &arcsinh_xs {
        check("arcsinh", &format!("arcsinh_{x}"), arcsinh(x), x.asinh());
    }
    // sinh-arcsinh round-trip
    let arcsinh_round_trip_xs = [-2.5_f64, 0.0, 1.5];
    for &x in &arcsinh_round_trip_xs {
        let s = x.sinh();
        let back = arcsinh(s);
        check("arcsinh", &format!("arcsinh_round_trip_{x}"), back, x);
    }

    // arccosh: defined for x >= 1
    let arccosh_xs = [1.0_f64, 1.5, 2.0, 5.0, 100.0];
    for &x in &arccosh_xs {
        check("arccosh", &format!("arccosh_{x}"), arccosh(x), x.acosh());
    }
    // cosh-arccosh round-trip for x >= 0 (arccosh always non-negative)
    let arccosh_round_trip_xs = [0.0_f64, 0.5, 2.0, 3.5];
    for &x in &arccosh_round_trip_xs {
        let c = x.cosh();
        let back = arccosh(c);
        check("arccosh", &format!("arccosh_round_trip_{x}"), back, x);
    }

    // arctanh: defined for |x| < 1
    let arctanh_xs = [-0.99_f64, -0.5, -0.1, 0.0, 0.1, 0.5, 0.99];
    for &x in &arctanh_xs {
        check("arctanh", &format!("arctanh_{x}"), arctanh(x), x.atanh());
    }
    // tanh-arctanh round-trip
    let arctanh_round_trip_xs = [-2.0_f64, -0.5, 0.0, 0.5, 2.0];
    for &x in &arctanh_round_trip_xs {
        let t = x.tanh();
        let back = arctanh(t);
        check("arctanh", &format!("arctanh_round_trip_{x}"), back, x);
    }

    // exp2_iterated(x) = exp(exp(x))
    let exp2_iterated_xs = [-2.0_f64, -1.0, 0.0, 0.5, 1.0, 2.0];
    for &x in &exp2_iterated_xs {
        check(
            "exp2_iterated",
            &format!("exp2_iterated_{x}"),
            exp2_iterated(x),
            x.exp().exp(),
        );
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_special_inverse_hyperbolic_exp2".into(),
        category: "fsci_special::{arcsinh, arccosh, arctanh, exp2_iterated} coverage".into(),
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
            eprintln!(
                "inv_hyper mismatch: {} actual={} expected={} abs={}",
                d.case_id, d.actual, d.expected, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "inv_hyper/exp2 coverage failed: {} cases",
        diffs.len()
    );
    // Arms have different case sets (exp2_iterated has the fewest); each must compare all of
    // its own.
    let min_per_arm = [
        arcsinh_xs.len() + arcsinh_round_trip_xs.len(),
        arccosh_xs.len() + arccosh_round_trip_xs.len(),
        arctanh_xs.len() + arctanh_round_trip_xs.len(),
        exp2_iterated_xs.len(),
    ]
    .into_iter()
    .min()
    .expect("four arms");
    ledger.finish(min_per_arm);
}
