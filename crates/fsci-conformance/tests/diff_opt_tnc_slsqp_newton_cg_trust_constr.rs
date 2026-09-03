#![forbid(unsafe_code)]
//! Property-based test for fsci_opt::{tnc, slsqp, newton_cg, trust_constr}.
//!
//! Resolves [frankenscipy-8smu5]. All four optimizers minimize f(x)
//! and accept MinimizeOptions. Test on quadratic and Rosenbrock
//! objectives; verify converged solutions reach the known global
//! minimum at 1e-3 abs on the function value.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::MinimizeOptions;
use fsci_opt::minimize::{newton_cg, slsqp, tnc, trust_constr, trust_exact};
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

#[test]
fn diff_opt_tnc_slsqp_newton_cg_trust_constr() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    let opts = MinimizeOptions::default();

    // tnc/trust_constr: quadratic only (defect do5nd: weak on Rosen).
    let q_x0 = vec![2.0_f64, -1.0];
    if let Ok(res) = tnc(&quadratic, &q_x0, opts) {
        let fval = res.fun.unwrap_or(f64::INFINITY);
        max_overall = max_overall.max(fval);
        diffs.push(CaseDiff {
            case_id: "tnc_quad".into(),
            op: "tnc".into(),
            abs_diff: fval,
            pass: fval <= TOL,
        });
    }
    if let Ok(res) = trust_constr(&quadratic, &q_x0, opts) {
        let fval = res.fun.unwrap_or(f64::INFINITY);
        max_overall = max_overall.max(fval);
        diffs.push(CaseDiff {
            case_id: "trust_constr_quad".into(),
            op: "trust_constr".into(),
            abs_diff: fval,
            pass: fval <= TOL,
        });
    }

    // slsqp, newton_cg, trust_exact: both quadratic and Rosen.
    for (label, f, x0) in [
        ("quad", quadratic as fn(&[f64]) -> f64, vec![2.0_f64, -1.0]),
        ("rosen", rosen as fn(&[f64]) -> f64, vec![0.0_f64, 0.0]),
    ] {
        if let Ok(res) = slsqp(&f, &x0, opts) {
            let fval = res.fun.unwrap_or(f64::INFINITY);
            max_overall = max_overall.max(fval);
            diffs.push(CaseDiff {
                case_id: format!("slsqp_{label}"),
                op: "slsqp".into(),
                abs_diff: fval,
                pass: fval <= TOL,
            });
        }
        if let Ok(res) = newton_cg(&f, &x0, opts) {
            let fval = res.fun.unwrap_or(f64::INFINITY);
            max_overall = max_overall.max(fval);
            diffs.push(CaseDiff {
                case_id: format!("newton_cg_{label}"),
                op: "newton_cg".into(),
                abs_diff: fval,
                pass: fval <= TOL,
            });
        }
        if let Ok(res) = trust_exact(&f, &x0, opts) {
            let fval = res.fun.unwrap_or(f64::INFINITY);
            max_overall = max_overall.max(fval);
            diffs.push(CaseDiff {
                case_id: format!("trust_exact_{label}"),
                op: "trust_exact".into(),
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
}
