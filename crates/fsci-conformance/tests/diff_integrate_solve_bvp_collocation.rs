#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `fsci_integrate::solve_bvp` (frankenscipy-1ksfv.7):
//! the collocation solver against `scipy.integrate.solve_bvp` on a boundary layer
//! (y'' = 100·y, tol 1e-6), Troesch's problem, both branches of Bratu's problem, a
//! Sturm–Liouville eigenvalue with an unknown parameter, ε·y'' + y' = 0 with ε = 1e-3, the
//! Emden equation through the singular term S, a run that exceeds `max_nodes`, and a linear
//! problem.
//!
//! Per case the status must match, the final node count must be within ±50% of SciPy's,
//! unknown parameters must agree to `P_REL_TOL`, and the spline solutions `sol(x)` at 201
//! points must agree to `SOL_TOL_FACTOR`·tol. The solver is a transcription of SciPy's, so
//! most cases reproduce SciPy's mesh exactly; the log counts them.
//!
//! Every case must be compared: a SciPy failure or an fsci error is a FAILED case, not a
//! skipped one (frankenscipy-olv0j.1).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_integrate::{BvpOptions, solve_bvp};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-008";
const P_REL_TOL: f64 = 1.0e-6;
const SOL_TOL_FACTOR: f64 = 10.0;
const NODE_RATIO_TOL: f64 = 1.5;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

type Rhs = fn(f64, &[f64], &[f64]) -> Vec<f64>;
type Bc = fn(&[f64], &[f64], &[f64]) -> Vec<f64>;

struct Case {
    id: &'static str,
    fun: Rhs,
    bc: Bc,
    x: Vec<f64>,
    y: Vec<Vec<f64>>,
    p: Vec<f64>,
    singular: Option<Vec<Vec<f64>>>,
    tol: f64,
    max_nodes: usize,
}

fn linspace(a: f64, b: f64, m: usize) -> Vec<f64> {
    (0..m)
        .map(|i| a + (b - a) * i as f64 / (m - 1) as f64)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn case(
    id: &'static str,
    fun: Rhs,
    bc: Bc,
    x: Vec<f64>,
    y: Vec<Vec<f64>>,
    p: &[f64],
    tol: f64,
    max_nodes: usize,
) -> Case {
    Case {
        id,
        fun,
        bc,
        x,
        y,
        p: p.to_vec(),
        singular: None,
        tol,
        max_nodes,
    }
}

fn cases() -> Vec<Case> {
    let exp100: Rhs = |_, y, _| vec![y[1], 100.0 * y[0]];
    let exp100_bc: Bc = |ya, yb, _| vec![ya[0] - 1.0, yb[0]];
    let bratu: Rhs = |_, y, _| vec![y[1], -y[0].exp()];
    let zero_ends: Bc = |ya, yb, _| vec![ya[0], yb[0]];
    let x01 = linspace(0.0, 1.0, 11);
    let mut emden = case(
        "emden_singular",
        |_, y, _| vec![y[1], -y[0].powi(5)],
        |ya, yb, _| vec![ya[1], yb[0] - 0.75_f64.sqrt()],
        linspace(0.0, 1.0, 10),
        vec![vec![0.75_f64.sqrt(); 10], vec![1e-4; 10]],
        &[],
        1e-3,
        1000,
    );
    emden.singular = Some(vec![vec![0.0, 0.0], vec![0.0, -2.0]]);
    vec![
        case(
            "exp100",
            exp100,
            exp100_bc,
            linspace(0.0, 10.0, 11),
            vec![vec![0.0; 11], vec![0.0; 11]],
            &[],
            1e-6,
            1000,
        ),
        case(
            "troesch5",
            |_, y, _| vec![y[1], 5.0 * (5.0 * y[0]).sinh()],
            |ya, yb, _| vec![ya[0], yb[0] - 1.0],
            x01.clone(),
            vec![x01.clone(), vec![1.0; 11]],
            &[],
            1e-3,
            1000,
        ),
        case(
            "bratu_low",
            bratu,
            zero_ends,
            linspace(0.0, 1.0, 5),
            vec![vec![0.0; 5], vec![0.0; 5]],
            &[],
            1e-3,
            1000,
        ),
        case(
            "bratu_high",
            bratu,
            zero_ends,
            linspace(0.0, 1.0, 5),
            vec![vec![3.0; 5], vec![0.0; 5]],
            &[],
            1e-3,
            1000,
        ),
        case(
            "sturm_liouville",
            |_, y, p| vec![y[1], -p[0] * p[0] * y[0]],
            |ya, yb, p| vec![ya[0], yb[0], ya[1] - p[0]],
            linspace(0.0, 1.0, 5),
            vec![vec![0.0, 1.0, 0.0, -1.0, 0.0], vec![0.0; 5]],
            &[6.0],
            1e-3,
            1000,
        ),
        case(
            "boundary_layer",
            |_, y, _| vec![y[1], -y[1] / 1e-3],
            |ya, yb, _| vec![ya[0], yb[0] - 1.0],
            x01,
            vec![vec![0.0; 11], vec![0.0; 11]],
            &[],
            1e-3,
            1000,
        ),
        emden,
        case(
            "exp100_max_nodes",
            exp100,
            exp100_bc,
            linspace(0.0, 10.0, 11),
            vec![vec![0.0; 11], vec![0.0; 11]],
            &[],
            1e-6,
            40,
        ),
        case(
            "linear",
            |_, y, _| vec![y[1], 0.0],
            |ya, yb, _| vec![ya[0], yb[0] - 1.0],
            linspace(0.0, 1.0, 3),
            vec![vec![0.0; 3], vec![0.0; 3]],
            &[],
            1e-3,
            1000,
        ),
    ]
}

#[derive(Debug, Clone, Serialize)]
struct QueryCase {
    case_id: String,
    x: Vec<f64>,
    y: Vec<Vec<f64>>,
    p: Vec<f64>,
    s: Option<Vec<Vec<f64>>>,
    tol: f64,
    max_nodes: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleArm {
    case_id: String,
    status: Option<usize>,
    nodes: Option<usize>,
    x: Option<Vec<f64>>,
    p: Option<Vec<f64>>,
    sol: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    fsci_status: usize,
    scipy_status: usize,
    fsci_nodes: usize,
    scipy_nodes: usize,
    same_mesh: bool,
    max_sol_diff: f64,
    max_p_rel_diff: f64,
    pass: bool,
    reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    same_mesh_count: usize,
    pass: bool,
    timestamp_ms: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn scipy_oracle_or_skip(query: &[QueryCase]) -> Option<Vec<OracleArm>> {
    let script = r#"
import json, sys, warnings
import numpy as np
from scipy.integrate import solve_bvp

FUN = {
    "exp100": lambda x, y: np.vstack((y[1], 100 * y[0])),
    "troesch5": lambda x, y: np.vstack((y[1], 5 * np.sinh(5 * y[0]))),
    "bratu_low": lambda x, y: np.vstack((y[1], -np.exp(y[0]))),
    "bratu_high": lambda x, y: np.vstack((y[1], -np.exp(y[0]))),
    "sturm_liouville": lambda x, y, p: np.vstack((y[1], -p[0]**2 * y[0])),
    "boundary_layer": lambda x, y: np.vstack((y[1], -y[1] / 1e-3)),
    "emden_singular": lambda x, y: np.vstack((y[1], -y[0]**5)),
    "exp100_max_nodes": lambda x, y: np.vstack((y[1], 100 * y[0])),
    "linear": lambda x, y: np.vstack((y[1], np.zeros_like(x))),
}
BC = {
    "exp100": lambda ya, yb: np.array([ya[0] - 1, yb[0]]),
    "troesch5": lambda ya, yb: np.array([ya[0], yb[0] - 1]),
    "bratu_low": lambda ya, yb: np.array([ya[0], yb[0]]),
    "bratu_high": lambda ya, yb: np.array([ya[0], yb[0]]),
    "sturm_liouville": lambda ya, yb, p: np.array([ya[0], yb[0], ya[1] - p[0]]),
    "boundary_layer": lambda ya, yb: np.array([ya[0], yb[0] - 1]),
    "emden_singular": lambda ya, yb: np.array([ya[1], yb[0] - (3/4)**0.5]),
    "exp100_max_nodes": lambda ya, yb: np.array([ya[0] - 1, yb[0]]),
    "linear": lambda ya, yb: np.array([ya[0], yb[0] - 1]),
}

out = []
for case in json.load(sys.stdin):
    cid = case["case_id"]
    arm = {"case_id": cid, "status": None, "nodes": None, "x": None, "p": None, "sol": None}
    try:
        x = np.array(case["x"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = solve_bvp(FUN[cid], BC[cid], x, np.array(case["y"]),
                          p=case["p"] or None,
                          S=None if case["s"] is None else np.array(case["s"]),
                          tol=case["tol"], max_nodes=case["max_nodes"])
        xs = np.linspace(x[0], x[-1], 201)
        arm.update(status=int(r.status), nodes=int(r.x.size), x=r.x.tolist(),
                   p=[] if r.p is None else [float(v) for v in r.p],
                   sol=r.sol(xs).T.tolist())
    except Exception:
        pass
    out.append(arm)
print(json.dumps(out))
"#;
    let query_json = serde_json::to_string(query).expect("serialize bvp query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for the solve_bvp oracle: {e}"
            );
            eprintln!("skipping solve_bvp oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open solve_bvp oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "solve_bvp oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping solve_bvp oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for solve_bvp oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "solve_bvp oracle failed: {stderr}"
        );
        eprintln!("skipping solve_bvp oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse solve_bvp oracle JSON"))
}

#[test]
fn diff_integrate_solve_bvp_collocation() {
    let cases = cases();
    let query: Vec<QueryCase> = cases
        .iter()
        .map(|c| QueryCase {
            case_id: c.id.to_string(),
            x: c.x.clone(),
            y: c.y.clone(),
            p: c.p.clone(),
            s: c.singular.clone(),
            tol: c.tol,
            max_nodes: c.max_nodes,
        })
        .collect();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    let arms: HashMap<String, OracleArm> = oracle
        .into_iter()
        .map(|arm| (arm.case_id.clone(), arm))
        .collect();

    let mut diffs = Vec::new();
    for case in &cases {
        let arm = &arms[case.id];
        let options = BvpOptions {
            tol: case.tol,
            max_nodes: case.max_nodes,
            singular_term: case.singular.as_deref(),
            ..BvpOptions::default()
        };
        let fsci = solve_bvp(case.fun, case.bc, &case.x, &case.y, &case.p, options);
        let mut diff = CaseDiff {
            case_id: case.id.to_string(),
            fsci_status: usize::MAX,
            scipy_status: arm.status.unwrap_or(usize::MAX),
            fsci_nodes: 0,
            scipy_nodes: arm.nodes.unwrap_or(0),
            same_mesh: false,
            max_sol_diff: f64::NAN,
            max_p_rel_diff: f64::NAN,
            pass: false,
            reason: String::new(),
        };
        match (fsci, &arm.sol, &arm.x, &arm.p) {
            (Err(e), ..) => diff.reason = format!("fsci error {e}"),
            (Ok(_), None, ..) | (Ok(_), _, None, _) | (Ok(_), _, _, None) => {
                diff.reason = "SciPy produced no result".to_string();
            }
            (Ok(r), Some(scipy_sol), Some(scipy_x), Some(scipy_p)) => {
                diff.fsci_status = r.status;
                diff.fsci_nodes = r.x.len();
                diff.same_mesh = r.x.len() == scipy_x.len()
                    && r.x.iter().zip(scipy_x).all(|(a, b)| (a - b).abs() <= 1e-12);
                let xs = linspace(case.x[0], case.x[case.x.len() - 1], 201);
                diff.max_sol_diff = xs
                    .iter()
                    .zip(scipy_sol)
                    .flat_map(|(&t, want)| {
                        let got = r.sol(t);
                        got.into_iter()
                            .zip(want.clone())
                            .map(|(g, w)| (g - w).abs())
                            .collect::<Vec<_>>()
                    })
                    .fold(0.0, f64::max);
                diff.max_p_rel_diff =
                    r.p.iter()
                        .zip(scipy_p)
                        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
                        .fold(0.0, f64::max);
                let mut problems = Vec::new();
                if r.status != diff.scipy_status {
                    problems.push(format!("status {} vs {}", r.status, diff.scipy_status));
                }
                let (hi, lo) = (
                    diff.fsci_nodes.max(diff.scipy_nodes) as f64,
                    diff.fsci_nodes.min(diff.scipy_nodes) as f64,
                );
                if hi > NODE_RATIO_TOL * lo {
                    problems.push(format!("nodes {} vs {}", diff.fsci_nodes, diff.scipy_nodes));
                }
                if r.p.len() != scipy_p.len()
                    || diff.max_p_rel_diff.is_nan()
                    || diff.max_p_rel_diff > P_REL_TOL
                {
                    problems.push(format!("p rel diff {:e}", diff.max_p_rel_diff));
                }
                if diff.max_sol_diff.is_nan() || diff.max_sol_diff > SOL_TOL_FACTOR * case.tol {
                    problems.push(format!("sol diff {:e}", diff.max_sol_diff));
                }
                diff.pass = problems.is_empty();
                diff.reason = problems.join("; ");
            }
        }
        diffs.push(diff);
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let same_mesh_count = diffs.iter().filter(|d| d.pass && d.same_mesh).count();
    let log = DiffLog {
        test_id: "diff_integrate_solve_bvp_collocation".into(),
        category: "scipy.integrate.solve_bvp (collocation)".into(),
        case_count: diffs.len(),
        same_mesh_count,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        cases: diffs.clone(),
    };
    fs::create_dir_all(output_dir()).expect("create solve_bvp diff dir");
    fs::write(
        output_dir().join("diff_integrate_solve_bvp_collocation.json"),
        serde_json::to_string_pretty(&log).expect("serialize solve_bvp log"),
    )
    .expect("write solve_bvp log");
    for d in &diffs {
        println!(
            "{} status {}/{} nodes {}/{} same_mesh={} sol_diff={:e} p_rel={:e} {}",
            d.case_id,
            d.fsci_status,
            d.scipy_status,
            d.fsci_nodes,
            d.scipy_nodes,
            d.same_mesh,
            d.max_sol_diff,
            d.max_p_rel_diff,
            d.reason
        );
    }
    println!(
        "{} cases compared, {same_mesh_count} on SciPy's exact mesh",
        diffs.len()
    );
    assert_eq!(diffs.len(), cases.len(), "every case must be compared");
    assert!(all_pass, "solve_bvp vs scipy.integrate.solve_bvp failed");
}
