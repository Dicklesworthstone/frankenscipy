#![forbid(unsafe_code)]
//! Analytic and live-SciPy tests for constrained differential evolution.
//!
//! Cover feasible-only objective evaluation, objective-scale-independent
//! convergence, explicit absolute tolerance, and infeasible populations, in
//! addition to convex objectives with known constrained minima.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::{
    DifferentialEvolutionOptions, differential_evolution, differential_evolution_constrained,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-2;
const TOL: f64 = ABS_TOL;

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    abs_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    max_abs_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create dec diff dir");
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
fn diff_opt_de_constrained() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    // The analytic optimum is the reference side; an fsci error is `rust_failed`.
    let mut ledger = CompareLedger::new(
        "diff_opt_de_constrained",
        &["differential_evolution_constrained"],
    );

    let opts = DifferentialEvolutionOptions {
        maxiter: 500,
        popsize: 20,
        seed: Some(42),
        ..DifferentialEvolutionOptions::default()
    };

    // Problem 1: min x²+y² s.t. x+y >= 1, on [-2,2]²
    // Constraint violation: max(0, 1 - (x+y))
    // Analytical: (0.5, 0.5), f* = 0.5
    let f1 = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
    let cv1 = |x: &[f64]| (1.0 - (x[0] + x[1])).max(0.0);
    let res1 =
        differential_evolution_constrained(f1, &[(-2.0_f64, 2.0), (-2.0, 2.0)], cv1, opts.clone())
            .ok();
    if let Some((f_star1, f_at1)) = ledger.pair(
        "differential_evolution_constrained",
        "dec_linear_constraint",
        Some(0.5),
        res1.map(|r| f1(&r.x)),
    ) {
        let abs_d1 = (f_at1 - f_star1).abs();
        max_overall = max_overall.max(abs_d1);
        ledger.compared(
            "differential_evolution_constrained",
            "dec_linear_constraint",
            abs_d1 <= TOL,
        );
        diffs.push(CaseDiff {
            case_id: "dec_linear_constraint".into(),
            abs_diff: abs_d1,
            pass: abs_d1 <= TOL,
        });
    }

    // Problem 2: min (x-2)² + (y-2)² s.t. x² + y² <= 1 (disk constraint), on [-3, 3]²
    // Analytical: closest point on unit circle to (2, 2), i.e. (1/sqrt(2), 1/sqrt(2))
    // Distance: sqrt(2)*sqrt(2)·(2 - 1/sqrt(2)) = 2 - 1/sqrt(2) and squared distance ≈ 1.7157
    // Compute: ‖(2,2) − (1/√2, 1/√2)‖² = 2·(2 − 1/√2)²
    let target_f = 2.0 * (2.0 - 1.0 / 2.0_f64.sqrt()).powi(2);
    let f2 = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2);
    let cv2 = |x: &[f64]| ((x[0] * x[0] + x[1] * x[1]) - 1.0).max(0.0);
    let res2 =
        differential_evolution_constrained(f2, &[(-3.0_f64, 3.0), (-3.0, 3.0)], cv2, opts.clone())
            .ok();
    if let Some((target_f, f_at2)) = ledger.pair(
        "differential_evolution_constrained",
        "dec_disk_constraint",
        Some(target_f),
        res2.map(|r| f2(&r.x)),
    ) {
        let abs_d2 = (f_at2 - target_f).abs();
        max_overall = max_overall.max(abs_d2);
        ledger.compared(
            "differential_evolution_constrained",
            "dec_disk_constraint",
            abs_d2 <= TOL,
        );
        diffs.push(CaseDiff {
            case_id: "dec_disk_constraint".into(),
            abs_diff: abs_d2,
            pass: abs_d2 <= TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_de_constrained".into(),
        category: "fsci_opt::differential_evolution_constrained property test".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("dec mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "dec conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    // Two analytic problems, each compared once.
    ledger.finish(2);
}

#[derive(Deserialize)]
struct OracleResult {
    case_id: String,
    x: f64,
    fun: Option<f64>,
    success: bool,
    nit: usize,
    nfev: usize,
    objective_calls: usize,
}

const DE_ORACLE: &str = r#"
import json
import math
import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint

rows = []
for name, exponent, atol in [
    ('scale_unit', 0, 0.), ('scale_small', -100, 0.),
    ('scale_large', 100, 0.), ('absolute_tolerance', -100, 1e-12),
    ('feasible_domain', 0, 0.), ('infeasible_domain', 0, 0.),
]:
    calls = [0]
    def objective(x):
        calls[0] += 1
        if name == 'infeasible_domain':
            raise AssertionError('infeasible objective must never be called')
        if name == 'feasible_domain':
            assert x[0] >= 1., x
            return math.log(x[0]) + 1.
        return math.ldexp(1., exponent) * (x[0] - .25)**2

    kwargs = dict(seed=7, tol=1e-8, atol=atol, polish=False, maxiter=1000)
    bounds = [(-5., 5.)]
    if name == 'feasible_domain':
        bounds = [(-2., 2.)]
        kwargs['constraints'] = NonlinearConstraint(lambda x: x[0], 1., np.inf)
    elif name == 'infeasible_domain':
        bounds = [(0., 2.)]
        kwargs['constraints'] = NonlinearConstraint(lambda x: x[0], 3., np.inf)
        kwargs['maxiter'] = 30
    result = differential_evolution(objective, bounds, **kwargs)
    assert result.nfev == calls[0], (name, result.nfev, calls[0])
    rows.append(dict(case_id=name, x=float(result.x[0]),
                     fun=float(result.fun) if np.isfinite(result.fun) else None,
                     success=bool(result.success), nit=int(result.nit),
                     nfev=int(result.nfev), objective_calls=calls[0]))
print(json.dumps(rows, allow_nan=False))
"#;

#[test]
fn diff_opt_de_feasibility_and_stopping_match_live_scipy() {
    let required = std::env::var_os("FSCI_REQUIRE_SCIPY_ORACLE").is_some();
    let available = fsci_conformance::scipy_oracle_command()
        .args(["-c", "import scipy, numpy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(!required, "required SciPy oracle is unavailable");
        eprintln!("SKIP differential evolution live oracle: SciPy is unavailable");
        return;
    }
    let output = fsci_conformance::scipy_oracle_command()
        .args(["-c", DE_ORACLE])
        .output()
        .expect("run differential evolution oracle");
    assert!(
        output.status.success(),
        "differential evolution oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let reference: Vec<OracleResult> =
        serde_json::from_slice(&output.stdout).expect("valid differential evolution oracle JSON");
    let mut ledger = CompareLedger::new("diff_opt_de_feasibility_and_stopping", &["de_contract"]);
    for expected in &reference {
        let calls = std::cell::Cell::new(0);
        let options = DifferentialEvolutionOptions {
            seed: Some(7),
            maxiter: if expected.case_id == "infeasible_domain" {
                30
            } else {
                1000
            },
            atol: if expected.case_id == "absolute_tolerance" {
                1e-12
            } else {
                0.0
            },
            ..Default::default()
        };
        let observed = match expected.case_id.as_str() {
            "feasible_domain" => differential_evolution_constrained(
                |x| {
                    assert!(x[0] >= 1.0, "infeasible objective evaluation at {x:?}");
                    calls.set(calls.get() + 1);
                    x[0].ln() + 1.0
                },
                &[(-2.0, 2.0)],
                |x| (1.0 - x[0]).max(0.0),
                options,
            ),
            "infeasible_domain" => differential_evolution_constrained(
                |_| panic!("infeasible objective must never be called"),
                &[(0.0, 2.0)],
                |x| (3.0 - x[0]).max(0.0),
                options,
            ),
            "scale_unit" | "scale_small" | "scale_large" | "absolute_tolerance" => {
                let exponent = match expected.case_id.as_str() {
                    "scale_unit" => 0,
                    "scale_large" => 100,
                    _ => -100,
                };
                differential_evolution(
                    |x| {
                        calls.set(calls.get() + 1);
                        2.0_f64.powi(exponent) * (x[0] - 0.25).powi(2)
                    },
                    &[(-5.0, 5.0)],
                    options,
                )
            }
            other => panic!("unknown oracle case {other}"),
        };
        let observed = observed.expect("a finite oracle case must not be refused");
        assert_eq!(observed.success, expected.success, "{}", expected.case_id);
        assert_eq!(observed.nfev, calls.get(), "{}", expected.case_id);
        assert_eq!(
            expected.nfev, expected.objective_calls,
            "{}",
            expected.case_id
        );
        if expected.case_id == "absolute_tolerance" {
            // Random generators differ; both populations must stop at the first
            // generation when their whole spread is below the requested atol.
            assert_eq!(observed.nit, 1);
            assert_eq!(expected.nit, 1);
        } else if expected.case_id == "infeasible_domain" {
            assert_eq!(observed.nfev, 0);
            assert_eq!(expected.nfev, 0);
            assert_eq!(observed.fun, Some(f64::INFINITY));
            assert!(expected.fun.is_none());
            assert!(observed.maxcv.unwrap() > 0.0);
        } else {
            assert!(
                (observed.x[0] - expected.x).abs() <= 1e-6,
                "{}",
                expected.case_id
            );
            if expected.case_id.starts_with("scale_") {
                assert!((observed.x[0] - 0.25).abs() < 1e-10);
                assert!(observed.nit > 1 && expected.nit > 1);
            } else {
                assert_eq!(observed.maxcv, Some(0.0));
                assert!((observed.fun.unwrap() - expected.fun.unwrap()).abs() < 1e-6);
            }
        }
        println!(
            "DE {}: x={:?} scipy_x={:.16e} nit={} scipy_nit={} nfev={} scipy_nfev={}",
            expected.case_id,
            observed.x,
            expected.x,
            observed.nit,
            expected.nit,
            observed.nfev,
            expected.nfev
        );
        ledger.compared("de_contract", &expected.case_id, true);
    }
    assert_eq!(reference.len(), 6, "every declared case must be compared");
    ledger.finish(6);
}
