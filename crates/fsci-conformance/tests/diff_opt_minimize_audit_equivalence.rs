#![forbid(unsafe_code)]
//! Property test: fsci_opt::minimize_with_audit produces same
//! result as minimize for both success and edge cases.
//!
//! Resolves [frankenscipy-lg5aa].

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::{
    MinimizeOptions, OptimizeMethod, minimize, minimize_with_audit, minimize_with_casp_portfolio,
    sync_audit_ledger,
};
use fsci_runtime::{OptSolverAction, OptSolverPortfolio, RuntimeMode};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;

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
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create minimize_audit diff dir");
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

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_opt_minimize_audit_equivalence() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let ledger = sync_audit_ledger();

    // Three test objectives
    let quadratic = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
    let rosen = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
    let shifted_quad =
        |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2) + (x[2] + 0.5).powi(2);

    let probes: Vec<(&str, Vec<f64>, OptimizeMethod, fn(&[f64]) -> f64)> = vec![
        (
            "quad_neldermead",
            vec![1.0, 1.0],
            OptimizeMethod::NelderMead,
            quadratic,
        ),
        (
            "quad_bfgs",
            vec![2.0, -1.0],
            OptimizeMethod::Bfgs,
            quadratic,
        ),
        (
            "rosen_neldermead",
            vec![0.0, 0.0],
            OptimizeMethod::NelderMead,
            rosen,
        ),
        (
            "rosen_powell",
            vec![0.0, 0.0],
            OptimizeMethod::Powell,
            rosen,
        ),
        (
            "shifted_neldermead",
            vec![0.0, 0.0, 0.0],
            OptimizeMethod::NelderMead,
            shifted_quad,
        ),
        (
            "shifted_bfgs",
            vec![0.0, 0.0, 0.0],
            OptimizeMethod::Bfgs,
            shifted_quad,
        ),
    ];

    // One arm per method, named as the log's `op`; plain `minimize` is the reference side.
    let method_arms = ["NelderMead", "Bfgs", "Powell"];
    let min_per_arm = method_arms
        .iter()
        .map(|arm| {
            probes
                .iter()
                .filter(|probe| format!("{:?}", probe.2) == *arm)
                .count()
        })
        .min()
        .expect("method_arms is non-empty");
    let mut compare = CompareLedger::new("diff_opt_minimize_audit_equivalence", &method_arms);

    for (label, x0, method, f) in probes {
        let opts = MinimizeOptions {
            method: Some(method),
            ..MinimizeOptions::default()
        };
        let plain = minimize(f, &x0, opts);
        let audited = minimize_with_audit(f, &x0, opts, &ledger);
        let arm = format!("{method:?}");
        let pass = match (&plain, &audited) {
            (Ok(p), Ok(a)) => {
                // A length mismatch, or a NaN that the max fold in `vec_max_diff` would drop,
                // is recorded by the ledger.
                let Some((px, ax)) =
                    compare.slices(&arm, label, Some(p.x.as_slice()), Some(a.x.as_slice()))
                else {
                    continue;
                };
                let d_x = vec_max_diff(px, ax);
                let pf = p.fun.unwrap_or(f64::NAN);
                let af = a.fun.unwrap_or(f64::NAN);
                let d_f = if pf.is_finite() && af.is_finite() {
                    (pf - af).abs()
                } else {
                    if pf.is_nan() && af.is_nan() {
                        0.0
                    } else {
                        f64::INFINITY
                    }
                };
                d_x <= ABS_TOL && d_f <= ABS_TOL && p.status == a.status
            }
            // Both refused: the audited wrapper must refuse the same way.
            (Err(pe), Err(ae)) => format!("{pe:?}") == format!("{ae:?}"),
            // One refused and the other did not.
            _ => false,
        };
        compare.compared(&arm, label, pass);
        let d = match (&plain, &audited) {
            (Ok(p), Ok(a)) => vec_max_diff(&p.x, &a.x),
            _ => 0.0,
        };
        diffs.push(CaseDiff {
            case_id: label.to_string(),
            op: format!("{method:?}"),
            abs_diff: d,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_minimize_audit_equivalence".into(),
        category: "fsci_opt::minimize_with_audit equivalent to minimize".into(),
        case_count: diffs.len(),
        compared: compare.counts().clone(),
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
        "minimize_audit_equiv conformance failed: {} cases",
        diffs.len(),
    );
    compare.finish(min_per_arm);
}

#[test]
fn diff_opt_minimize_with_casp_portfolio() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut portfolio = OptSolverPortfolio::new(RuntimeMode::Strict, 16);

    // 1. Smooth convex problem routes to BFGS
    let res_bfgs = minimize_with_casp_portfolio(
        |x| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2),
        &[0.0, 0.0],
        MinimizeOptions::default(),
        &mut portfolio,
        false,
        false,
    )
    .expect("casp bfgs solve");
    assert_eq!(res_bfgs.chosen_action, OptSolverAction::BFGS);
    let pass_bfgs =
        (res_bfgs.result.x[0] - 1.0).abs() < 1e-4 && (res_bfgs.result.x[1] - 2.0).abs() < 1e-4;
    diffs.push(CaseDiff {
        case_id: "casp_smooth_convex_routes_to_bfgs".into(),
        op: "BFGS".into(),
        abs_diff: (res_bfgs.result.x[0] - 1.0)
            .abs()
            .max((res_bfgs.result.x[1] - 2.0).abs()),
        pass: pass_bfgs,
    });

    // 2. Noisy problem routes to Nelder-Mead
    let res_nm = minimize_with_casp_portfolio(
        |x| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2),
        &[0.0, 0.0],
        MinimizeOptions::default(),
        &mut portfolio,
        true,
        false,
    )
    .expect("casp nelder_mead solve");
    assert_eq!(res_nm.chosen_action, OptSolverAction::NelderMead);
    let pass_nm =
        (res_nm.result.x[0] - 1.0).abs() < 1e-3 && (res_nm.result.x[1] - 2.0).abs() < 1e-3;
    diffs.push(CaseDiff {
        case_id: "casp_noisy_routes_to_nelder_mead".into(),
        op: "NelderMead".into(),
        abs_diff: (res_nm.result.x[0] - 1.0)
            .abs()
            .max((res_nm.result.x[1] - 2.0).abs()),
        pass: pass_nm,
    });

    // 3. Multimodal problem with explicit finite bounds routes to DIRECT
    let direct_bounds = [(Some(-10.0), Some(10.0)), (Some(-10.0), Some(10.0))];
    let res_direct = minimize_with_casp_portfolio(
        |x| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2),
        &[0.0, 0.0],
        MinimizeOptions {
            bounds: Some(&direct_bounds),
            ..MinimizeOptions::default()
        },
        &mut portfolio,
        false,
        true,
    )
    .expect("casp direct solve");
    assert_eq!(res_direct.chosen_action, OptSolverAction::DIRECT);
    diffs.push(CaseDiff {
        case_id: "casp_multimodal_routes_to_direct".into(),
        op: "DIRECT".into(),
        abs_diff: 0.0,
        pass: true,
    });

    assert_eq!(
        portfolio.evidence_len(),
        3,
        "portfolio must accumulate 3 evidence entries"
    );

    // Each solve above either panicked (`expect`, routing `assert_eq!`) or produced its case.
    let mut compare = CompareLedger::new(
        "diff_opt_minimize_with_casp_portfolio",
        &["BFGS", "NelderMead", "DIRECT"],
    );
    for d in &diffs {
        compare.compared(&d.op, &d.case_id, d.pass);
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_opt_minimize_with_casp_portfolio".into(),
        category: "fsci_opt::minimize_with_casp_portfolio CASP routing".into(),
        case_count: diffs.len(),
        compared: compare.counts().clone(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    assert!(all_pass, "all casp portfolio routing cases must pass");
    // One routing case per solver arm.
    compare.finish(1);
}

/// frankenscipy-7tb8d.8: live-SciPy differential test comparing final `x` and `fun` on 10
/// standard problems (Rosenbrock, Beale, Booth, Matyas, Himmelblau, bounded problems, and
/// ill-conditioned quadratics) against `scipy.optimize.minimize` invoked with the same method
/// CASP chose.
#[test]
fn diff_opt_minimize_with_casp_10_problems_scipy() {
    use std::io::Write;
    use std::process::Stdio;
    use fsci_opt::types::Bound;
    use fsci_opt::minimize_with_casp;
    use serde::Deserialize;

    const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

    #[derive(Debug, Serialize)]
    struct OracleQuery {
        case_id: String,
        prob_id: String,
        scipy_method: String,
        x0: Vec<f64>,
        bounds: Option<Vec<(Option<f64>, Option<f64>)>>,
    }

    #[derive(Debug, Deserialize)]
    struct OracleAnswer {
        case_id: String,
        success: bool,
        x: Vec<f64>,
        fun: f64,
    }

    fn eval_prob(prob_id: &str, x: &[f64]) -> f64 {
        match prob_id {
            "rosenbrock_2d" | "rosenbrock_bounded" => {
                (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
            }
            "rosenbrock_3d" => {
                (0..2)
                    .map(|i| (1.0 - x[i]).powi(2) + 100.0 * (x[i + 1] - x[i] * x[i]).powi(2))
                    .sum()
            }
            "beale" => {
                (1.5 - x[0] + x[0] * x[1]).powi(2)
                    + (2.25 - x[0] + x[0] * x[1].powi(2)).powi(2)
                    + (2.625 - x[0] + x[0] * x[1].powi(3)).powi(2)
            }
            "booth" => (x[0] + 2.0 * x[1] - 7.0).powi(2) + (2.0 * x[0] + x[1] - 5.0).powi(2),
            "matyas" => 0.26 * (x[0] * x[0] + x[1] * x[1]) - 0.48 * x[0] * x[1],
            "himmelblau" => (x[0] * x[0] + x[1] - 11.0).powi(2) + (x[0] + x[1] * x[1] - 7.0).powi(2),
            "ill_quad_1e6" => (x[0] - 1.0).powi(2) + 1.0e6 * (x[1] + 1.0).powi(2),
            "ill_quad_1e8" => {
                (x[0] - 1.0).powi(2) + 1.0e4 * (x[1] + 2.0).powi(2) + 1.0e8 * (x[2] - 0.5).powi(2)
            }
            "bounded_quad" => (x[0] - 4.0).powi(2) + (x[1] + 4.0).powi(2),
            "powell_quartic" => {
                (x[0] + 10.0 * x[1]).powi(2)
                    + 5.0 * (x[2] - x[3]).powi(2)
                    + (x[1] - 2.0 * x[2]).powi(4)
                    + 10.0 * (x[0] - x[3]).powi(4)
            }
            _ => panic!("unknown prob_id {prob_id}"),
        }
    }

    fn grad_ill_quad_1e6(x: &[f64]) -> Vec<f64> {
        vec![2.0 * (x[0] - 1.0), 2.0e6 * (x[1] + 1.0)]
    }

    fn grad_ill_quad_1e8(x: &[f64]) -> Vec<f64> {
        vec![
            2.0 * (x[0] - 1.0),
            2.0e4 * (x[1] + 2.0),
            2.0e8 * (x[2] - 0.5),
        ]
    }

    static BOUNDS_ROSEN: [Bound; 2] = [(Some(-2.0), Some(0.5)), (Some(-2.0), Some(0.5))];
    static BOUNDS_QUAD: [Bound; 2] = [(Some(-1.0), Some(2.0)), (Some(-2.0), Some(1.0))];

    type ProblemSpec = (
        &'static str,
        Vec<f64>,
        Option<&'static [Bound]>,
        Option<fsci_opt::GradientFunc>,
    );

    let specs: Vec<ProblemSpec> = vec![
        ("rosenbrock_2d", vec![-1.2, 1.0], None, None),
        ("rosenbrock_3d", vec![-1.2, 1.0, -1.2], None, None),
        ("beale", vec![1.0, 1.0], None, None),
        ("booth", vec![0.0, 0.0], None, None),
        ("matyas", vec![-2.0, 3.0], None, None),
        ("himmelblau", vec![2.5, 1.8], None, None),
        ("ill_quad_1e6", vec![0.0, 0.0], None, Some(grad_ill_quad_1e6)),
        (
            "ill_quad_1e8",
            vec![0.0, 0.0, 0.0],
            None,
            Some(grad_ill_quad_1e8),
        ),
        ("rosenbrock_bounded", vec![-1.2, 1.0], Some(&BOUNDS_ROSEN), None),
        ("bounded_quad", vec![0.0, 0.0], Some(&BOUNDS_QUAD), None),
    ];
    assert_eq!(specs.len(), 10);

    let mut portfolio = OptSolverPortfolio::new(RuntimeMode::Strict, 32);
    let mut rust_runs = Vec::with_capacity(specs.len());
    let mut queries = Vec::with_capacity(specs.len());

    for (prob_id, x0, bounds, gradient) in &specs {
        let options = MinimizeOptions {
            bounds: *bounds,
            gradient: *gradient,
            maxiter: Some(2000),
            ..MinimizeOptions::default()
        };
        let out = minimize_with_casp(
            |x| eval_prob(prob_id, x),
            x0,
            options,
            &mut portfolio,
        )
        .expect(prob_id);
        let scipy_method = match out.chosen_action {
            OptSolverAction::BFGS => "BFGS",
            OptSolverAction::LBFGSB => "L-BFGS-B",
            OptSolverAction::NelderMead => "Nelder-Mead",
            OptSolverAction::TrustRegionNewtonCG => "Newton-CG",
            OptSolverAction::DIRECT => "direct",
        };
        queries.push(OracleQuery {
            case_id: (*prob_id).to_string(),
            prob_id: (*prob_id).to_string(),
            scipy_method: scipy_method.to_string(),
            x0: x0.clone(),
            bounds: bounds.map(|b| b.to_vec()),
        });
        rust_runs.push((*prob_id, out));
    }

    assert_eq!(portfolio.evidence_len(), 10);
    assert!(portfolio.calibrator().total_predictions() >= 10);

    let python_script = r#"
import json
import sys
import numpy as np
from scipy.optimize import minimize

def eval_prob(prob_id, x):
    x = np.asarray(x, dtype=float)
    if prob_id in ("rosenbrock_2d", "rosenbrock_bounded"):
        return float((1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2)
    elif prob_id == "rosenbrock_3d":
        return float(np.sum((1.0 - x[:-1])**2 + 100.0 * (x[1:] - x[:-1]**2)**2))
    elif prob_id == "beale":
        return float((1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2)
    elif prob_id == "booth":
        return float((x[0] + 2.0*x[1] - 7.0)**2 + (2.0*x[0] + x[1] - 5.0)**2)
    elif prob_id == "matyas":
        return float(0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1])
    elif prob_id == "himmelblau":
        return float((x[0]**2 + x[1] - 11.0)**2 + (x[0] + x[1]**2 - 7.0)**2)
    elif prob_id == "ill_quad_1e6":
        return float((x[0] - 1.0)**2 + 1e6*(x[1] + 1.0)**2)
    elif prob_id == "ill_quad_1e8":
        return float((x[0] - 1.0)**2 + 1e4*(x[1] + 2.0)**2 + 1e8*(x[2] - 0.5)**2)
    elif prob_id == "bounded_quad":
        return float((x[0] - 4.0)**2 + (x[1] + 4.0)**2)
    raise ValueError(prob_id)

def eval_jac(prob_id, x):
    x = np.asarray(x, dtype=float)
    if prob_id == "ill_quad_1e6":
        return np.array([2.0*(x[0] - 1.0), 2e6*(x[1] + 1.0)], dtype=float)
    elif prob_id == "ill_quad_1e8":
        return np.array([2.0*(x[0] - 1.0), 2e4*(x[1] + 2.0), 2e8*(x[2] - 0.5)], dtype=float)
    return None

queries = json.loads(sys.stdin.read())
out = []
for q in queries:
    prob_id = q["prob_id"]
    method = q["scipy_method"]
    x0 = np.asarray(q["x0"], dtype=float)
    bounds = q["bounds"]
    jac = (lambda x, pid=prob_id: eval_jac(pid, x)) if prob_id in ("ill_quad_1e6", "ill_quad_1e8") else None
    res = minimize(lambda x, pid=prob_id: eval_prob(pid, x), x0, method=method, jac=jac, bounds=bounds, options={"maxiter": 2000})
    out.append({
        "case_id": q["case_id"],
        "success": bool(res.success),
        "x": [float(v) for v in res.x],
        "fun": float(res.fun),
    })
print(json.dumps(out))
"#;

    let mut child = match fsci_conformance::scipy_oracle_command()
        .args(["-c", python_script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "scipy oracle required but spawn failed: {e}"
            );
            return;
        }
    };
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(serde_json::to_string(&queries).expect("json").as_bytes())
        .expect("write stdin");
    let output = child.wait_with_output().expect("wait_with_output");
    if !output.status.success() {
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "scipy oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    let answers: Vec<OracleAnswer> =
        serde_json::from_slice(&output.stdout).expect("parse oracle answers");
    assert_eq!(answers.len(), 10);

    let mut ledger = CompareLedger::new("diff_opt_minimize_with_casp_10_problems_scipy", &["casp"]);
    for ((prob_id, rust_out), ans) in rust_runs.iter().zip(answers.iter()) {
        assert_eq!(*prob_id, ans.case_id);
        let x_diff = vec_max_diff(&rust_out.result.x, &ans.x);
        let f_diff = (rust_out.result.fun.unwrap_or(f64::NAN) - ans.fun).abs();
        println!(
            "{prob_id}: action={:?}, nfev={}, x_diff={x_diff:e}, f_diff={f_diff:e}",
            rust_out.chosen_action, rust_out.result.nfev
        );
        let pass = ans.success && rust_out.result.success && x_diff <= 1.0e-3 && f_diff <= 1.0e-5;
        ledger.compared("casp", prob_id, pass);
        assert!(
            pass,
            "{prob_id} via {:?}: rust x={:?} f={:?}, scipy x={:?} f={}",
            rust_out.chosen_action, rust_out.result.x, rust_out.result.fun, ans.x, ans.fun
        );
    }
    ledger.finish(10);
}

