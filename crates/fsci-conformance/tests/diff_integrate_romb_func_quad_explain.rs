#![forbid(unsafe_code)]
//! Cover fsci_integrate::{romb_func, quad_explain}.
//!
//! Resolves [frankenscipy-gpwig]. romb_func is the
//! scipy.integrate.romb-style adaptive Romberg with optional
//! max_order and tol. quad_explain wraps quad() and returns
//! (QuadResult, human-readable explanation string).

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_integrate::{QuadOptions, quad_explain, romb_func};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
/// One ledger arm per routine; a case's arm is its id prefix.
const ARMS: [&str; 2] = ["romb_func", "quad_explain"];

fn arm_of(case_id: &str) -> &'static str {
    ARMS.into_iter()
        .find(|arm| case_id.starts_with(&format!("{arm}_")))
        .unwrap_or_else(|| panic!("case {case_id} names no declared arm"))
}

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
    fs::create_dir_all(output_dir()).expect("create romb_func diff dir");
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
fn diff_integrate_romb_func_quad_explain() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut ledger = CompareLedger::new("diff_integrate_romb_func_quad_explain", &ARMS);
    // Every check records one CaseDiff, so `diffs` is the designed case list, and one ledger
    // verdict (a NaN integral fails its own comparison).
    let mut check = |id: &str, ok: bool, note: String| {
        ledger.compared(arm_of(id), id, ok);
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    // === romb_func: closed-form integrals ===
    // Constant: ∫_{0}^{2} 5 dx = 10
    {
        let r = romb_func(|_| 5.0, 0.0, 2.0, None, None).expect("constant");
        check(
            "romb_func_constant",
            r.converged && (r.integral - 10.0).abs() < 1e-10,
            format!("integral={} converged={}", r.integral, r.converged),
        );
    }

    // Quadratic: ∫_{0}^{1} x² dx = 1/3
    {
        let r = romb_func(|x| x * x, 0.0, 1.0, None, None).expect("quadratic");
        check(
            "romb_func_quadratic",
            r.converged && (r.integral - 1.0 / 3.0).abs() < 1e-9,
            format!("integral={} converged={}", r.integral, r.converged),
        );
    }

    // Cubic: ∫_{0}^{2} x³ dx = 4
    {
        let r = romb_func(|x| x * x * x, 0.0, 2.0, None, None).expect("cubic");
        check(
            "romb_func_cubic",
            r.converged && (r.integral - 4.0).abs() < 1e-9,
            format!("integral={} converged={}", r.integral, r.converged),
        );
    }

    // Sine: ∫_{0}^{π} sin(x) dx = 2
    {
        let r = romb_func(|x: f64| x.sin(), 0.0, PI, None, None).expect("sine");
        check(
            "romb_func_sin",
            r.converged && (r.integral - 2.0).abs() < 1e-8,
            format!("integral={} converged={}", r.integral, r.converged),
        );
    }

    // Exp: ∫_{0}^{1} exp(x) dx = e - 1
    {
        let r = romb_func(|x: f64| x.exp(), 0.0, 1.0, Some(15), Some(1e-12)).expect("exp");
        let expected = std::f64::consts::E - 1.0;
        check(
            "romb_func_exp",
            r.converged && (r.integral - expected).abs() < 1e-10,
            format!("integral={} converged={}", r.integral, r.converged),
        );
    }

    // Custom max_order=4 with tight tol — should not converge for a tough integrand
    {
        // f(x) = sqrt(x) on [0,1]: ∫ = 2/3 ≈ 0.667, but sqrt(x) has unbounded
        // higher derivatives at 0 → Romberg converges slowly
        let r = romb_func(|x: f64| x.sqrt(), 0.0, 1.0, Some(4), Some(1e-14)).expect("sqrt");
        // Even non-converged, integral should be in the ballpark
        check(
            "romb_func_sqrt_low_order_imperfect",
            (r.integral - 2.0 / 3.0).abs() < 1e-2,
            format!(
                "integral={} converged={} (allowed-not-converge)",
                r.integral, r.converged
            ),
        );
    }

    // Non-finite bounds → error
    {
        let r = romb_func(|_| 1.0, f64::NAN, 1.0, None, None);
        check(
            "romb_func_nan_bound_errors",
            r.is_err(),
            format!("res={r:?}"),
        );
    }

    // === quad_explain: success message ===
    {
        let opts = QuadOptions::default();
        let (result, msg) = quad_explain(|x| x * x, 0.0, 1.0, opts);
        check(
            "quad_explain_success_integral_correct",
            result.converged && (result.integral - 1.0 / 3.0).abs() < 1e-8,
            format!("integral={}", result.integral),
        );
        check(
            "quad_explain_success_message_contains_converged",
            msg.contains("converged") && !msg.contains("did NOT"),
            format!("msg={msg}"),
        );
    }

    // === quad_explain: failure path (invalid bounds) ===
    {
        let opts = QuadOptions::default();
        // NaN upper bound triggers quad() error
        let (result, msg) = quad_explain(|x| x * x, 0.0, f64::NAN, opts);
        check(
            "quad_explain_failure_nan_integral",
            result.integral.is_nan() && !result.converged,
            format!("result={result:?}"),
        );
        check(
            "quad_explain_failure_message_contains_failed",
            msg.contains("failed"),
            format!("msg={msg}"),
        );
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_integrate_romb_func_quad_explain".into(),
        category: "fsci_integrate::{romb_func, quad_explain} coverage".into(),
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
                "romb_func/quad_explain mismatch: {} — {}",
                d.case_id, d.note
            );
        }
    }

    assert!(
        all_pass,
        "romb_func/quad_explain coverage failed: {} cases",
        diffs.len()
    );
    // romb_func has 7 checks and quad_explain 4; each arm must compare all of its own.
    let min_per_arm = ARMS
        .iter()
        .map(|arm| diffs.iter().filter(|d| arm_of(&d.case_id) == *arm).count())
        .min()
        .expect("ARMS is non-empty");
    ledger.finish(min_per_arm);
}
