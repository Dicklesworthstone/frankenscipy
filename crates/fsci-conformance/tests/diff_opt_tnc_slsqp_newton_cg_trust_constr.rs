#![forbid(unsafe_code)]
//! Property-based test for fsci_opt::{tnc, slsqp, newton_cg, trust_constr}.
//!
//! Resolves [frankenscipy-8smu5]. All four optimizers minimize f(x)
//! and accept MinimizeOptions. Test on quadratic and Rosenbrock
//! objectives; verify converged solutions reach the known global
//! minimum at 1e-3 abs on the function value.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::minimize::{newton_cg, slsqp, tnc, trust_constr, trust_exact};
use fsci_opt::{MinimizeOptions, OptError, OptimizeResult};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const TOL: f64 = 1.0e-3;

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
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
    fs::create_dir_all(output_dir()).expect("create min_methods diff dir");
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

fn quadratic(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>()
}

fn rosen(x: &[f64]) -> f64 {
    (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
}

fn quadratic_grad(x: &[f64]) -> Vec<f64> {
    x.iter().map(|v| 2.0 * v).collect()
}

fn rosen_grad(x: &[f64]) -> Vec<f64> {
    vec![
        -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]),
        200.0 * (x[1] - x[0] * x[0]),
    ]
}

/// fsci's final function value, or `None` when the optimizer erred, reported no convergence, or
/// returned no value. The analytic minimum is always reachable, so each of these is a failure.
fn converged_fun(res: Result<OptimizeResult, OptError>) -> Option<f64> {
    res.ok().filter(|r| r.success).and_then(|r| r.fun)
}

#[test]
fn diff_opt_tnc_slsqp_newton_cg_trust_constr() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    let opts = MinimizeOptions::default();
    let mut ledger = CompareLedger::new(
        "diff_opt_tnc_slsqp_newton_cg_trust_constr",
        &["tnc", "trust_constr", "slsqp", "newton_cg", "trust_exact"],
    );

    // Both objectives have the global minimum value 0 (SciPy's side is the analytic answer).
    // tnc/trust_constr: quadratic only (defect do5nd: weak on Rosen).
    let q_x0 = vec![2.0_f64, -1.0];
    let single = [
        ("tnc", converged_fun(tnc(&quadratic, &q_x0, opts))),
        (
            "trust_constr",
            converged_fun(trust_constr(&quadratic, &q_x0, opts)),
        ),
    ];
    for (op, fsci_fun) in single {
        let case_id = format!("{op}_quad");
        let Some((_, fval)) = ledger.pair(op, &case_id, Some(0.0), fsci_fun) else {
            continue;
        };
        max_overall = max_overall.max(fval);
        ledger.compared(op, &case_id, fval <= TOL);
        diffs.push(CaseDiff {
            case_id,
            op: op.into(),
            abs_diff: fval,
            pass: fval <= TOL,
        });
    }

    // slsqp, newton_cg, trust_exact: both quadratic and Rosen. newton_cg gets the analytic
    // gradient, as SciPy's Newton-CG requires `jac`. Without one fsci differences the objective,
    // and on the quadratic from (2, -1) that stops on precision loss at x = 0, which is also
    // SciPy's verdict given the same gradient (fsci-opt's
    // newton_cg_fd_gradient_follows_scipy_on_the_sphere, frankenscipy-fd4wz).
    for (label, f, grad, x0) in [
        (
            "quad",
            quadratic as fn(&[f64]) -> f64,
            quadratic_grad as fn(&[f64]) -> Vec<f64>,
            vec![2.0_f64, -1.0],
        ),
        (
            "rosen",
            rosen as fn(&[f64]) -> f64,
            rosen_grad as fn(&[f64]) -> Vec<f64>,
            vec![0.0_f64, 0.0],
        ),
    ] {
        let with_grad = MinimizeOptions {
            gradient: Some(grad),
            ..opts
        };
        let runs = [
            ("slsqp", converged_fun(slsqp(&f, &x0, opts))),
            ("newton_cg", converged_fun(newton_cg(&f, &x0, with_grad))),
            ("trust_exact", converged_fun(trust_exact(&f, &x0, opts))),
        ];
        for (op, fsci_fun) in runs {
            let case_id = format!("{op}_{label}");
            let Some((_, fval)) = ledger.pair(op, &case_id, Some(0.0), fsci_fun) else {
                continue;
            };
            max_overall = max_overall.max(fval);
            ledger.compared(op, &case_id, fval <= TOL);
            diffs.push(CaseDiff {
                case_id,
                op: op.into(),
                abs_diff: fval,
                pass: fval <= TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_tnc_slsqp_newton_cg_trust_constr".into(),
        category: "fsci_opt::{tnc, slsqp, newton_cg, trust_constr, trust_exact} property test"
            .into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "min_methods conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    // The smallest arms: tnc and trust_constr run one case each, the other three two.
    ledger.finish(1);
}
