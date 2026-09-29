#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `least_squares` with SciPy's default Trust Region
//! Reflective method (frankenscipy-1ksfv.6): Rosenbrock with and without a bound, an
//! exponential fit with outliers under bounds and under each robust loss (soft_l1, huber,
//! cauchy, arctan), `x_scale='jac'` on a badly scaled problem, an underdetermined system, a
//! `max_nfev` stop, and a box with both bounds active.
//!
//! Per case the status and `active_mask` must match SciPy's, `x` must agree to `X_REL_TOL`
//! (relative to max(|x|, 1e-6)) and the cost to `COST_REL_TOL`. The port follows SciPy's
//! `trf.py` statement by statement, so it also takes SciPy's number of function evaluations;
//! the log counts the cases that do.
//!
//! Every case must be compared: a SciPy failure or an fsci error is a FAILED case (olv0j.1).

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::{LeastSquaresOptions, LossKind, least_squares, least_squares_bounded};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-003";
const X_REL_TOL: f64 = 1.0e-6;
const COST_REL_TOL: f64 = 1.0e-8;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    x0: Vec<f64>,
    lb: Option<Vec<f64>>,
    ub: Option<Vec<f64>>,
    loss: String,
    f_scale: f64,
    x_scale_jac: bool,
    max_nfev: Option<usize>,
}

fn case(id: &str, x0: &[f64]) -> Case {
    Case {
        case_id: id.to_string(),
        x0: x0.to_vec(),
        lb: None,
        ub: None,
        loss: "linear".to_string(),
        f_scale: 1.0,
        x_scale_jac: false,
        max_nfev: None,
    }
}

fn bounded(id: &str, x0: &[f64], lb: &[f64], ub: &[f64]) -> Case {
    Case {
        lb: Some(lb.to_vec()),
        ub: Some(ub.to_vec()),
        ..case(id, x0)
    }
}

fn with_loss(mut c: Case, loss: &str, f_scale: f64) -> Case {
    c.loss = loss.to_string();
    c.f_scale = f_scale;
    c
}

fn cases() -> Vec<Case> {
    let inf = f64::INFINITY;
    let mut xscale = case("xscale_jac", &[1.0, 1e-6]);
    xscale.x_scale_jac = true;
    let mut starved = case("max_nfev", &[-1.2, 1.0]);
    starved.max_nfev = Some(3);
    vec![
        case("rosen_unbounded", &[-1.2, 1.0]),
        bounded("rosen_bounded", &[2.0, 2.0], &[-inf, 1.5], &[inf, inf]),
        bounded("exp_bounds", &[1.0, 1.0], &[0.0, 0.5], &[10.0, 5.0]),
        with_loss(case("exp_soft_l1", &[1.0, 1.0]), "soft_l1", 0.1),
        with_loss(case("exp_huber", &[1.0, 1.0]), "huber", 1.0),
        with_loss(case("exp_cauchy", &[1.0, 1.0]), "cauchy", 0.5),
        with_loss(case("exp_arctan", &[1.0, 1.0]), "arctan", 0.5),
        with_loss(
            bounded("exp_soft_l1_bounds", &[1.0, 1.0], &[0.0, 0.5], &[10.0, 5.0]),
            "soft_l1",
            0.1,
        ),
        xscale,
        case("underdetermined", &[0.0, 0.0]),
        starved,
        bounded("box_both_active", &[1.0, 1.0], &[0.0, 0.0], &[2.0, 2.0]),
    ]
}

fn exp_data() -> (Vec<f64>, Vec<f64>) {
    let t: Vec<f64> = (0..40).map(|i| 4.0 * f64::from(i) / 39.0).collect();
    let mut y: Vec<f64> = t.iter().map(|ti| 2.0 * (-0.3 * ti).exp()).collect();
    y[5] += 3.0;
    y[20] -= 2.5;
    (t, y)
}

fn residuals(id: &str, x: &[f64], t: &[f64], y: &[f64]) -> Vec<f64> {
    if id.starts_with("rosen") || id == "max_nfev" {
        vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]]
    } else if id.starts_with("exp") {
        t.iter()
            .zip(y)
            .map(|(ti, yi)| x[0] * (-x[1] * ti).exp() - yi)
            .collect()
    } else if id == "xscale_jac" {
        vec![x[0] - 2.0, 1e6 * (x[1] - 3e-6), x[0] * x[1] - 6e-6]
    } else if id == "underdetermined" {
        vec![x[0] + 2.0 * x[1] - 3.0]
    } else {
        vec![x[0] - 3.0, x[1] + 1.0]
    }
}

fn loss_kind(name: &str) -> LossKind {
    match name {
        "soft_l1" => LossKind::SoftL1,
        "huber" => LossKind::Huber,
        "cauchy" => LossKind::Cauchy,
        "arctan" => LossKind::Arctan,
        _ => LossKind::Linear,
    }
}

#[derive(Debug, Clone, Deserialize)]
struct OracleArm {
    case_id: String,
    x: Option<Vec<f64>>,
    cost: Option<f64>,
    status: Option<i32>,
    nfev: Option<usize>,
    active_mask: Option<Vec<i8>>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    x_rel_diff: f64,
    cost_rel_diff: f64,
    fsci_status: i32,
    scipy_status: i32,
    fsci_nfev: usize,
    scipy_nfev: usize,
    pass: bool,
    reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    same_nfev_count: usize,
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

fn scipy_oracle_or_skip(query: &[Case]) -> Option<Vec<OracleArm>> {
    let script = r#"
import json, math, sys, warnings
import numpy as np
from scipy.optimize import least_squares

t = np.linspace(0, 4, 40)
y = 2 * np.exp(-0.3 * t)
y[5] += 3
y[20] -= 2.5

def residuals(cid):
    if cid.startswith("rosen") or cid == "max_nfev":
        return lambda x: np.array([10 * (x[1] - x[0]**2), 1 - x[0]])
    if cid.startswith("exp"):
        return lambda p: p[0] * np.exp(-p[1] * t) - y
    if cid == "xscale_jac":
        return lambda x: np.array([x[0] - 2, 1e6 * (x[1] - 3e-6), x[0] * x[1] - 6e-6])
    if cid == "underdetermined":
        return lambda x: np.array([x[0] + 2 * x[1] - 3])
    return lambda x: np.array([x[0] - 3, x[1] + 1])

out = []
for case in json.load(sys.stdin):
    arm = {"case_id": case["case_id"], "x": None, "cost": None, "status": None,
           "nfev": None, "active_mask": None}
    try:
        kw = dict(loss=case["loss"], f_scale=case["f_scale"], method="trf")
        if case["x_scale_jac"]:
            kw["x_scale"] = "jac"
        if case["max_nfev"] is not None:
            kw["max_nfev"] = case["max_nfev"]
        if case["lb"] is not None:
            # JSON has no infinity: serde writes an infinite bound as null.
            lb = [-math.inf if v is None else v for v in case["lb"]]
            ub = [math.inf if v is None else v for v in case["ub"]]
            kw["bounds"] = (lb, ub)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = least_squares(residuals(case["case_id"]), np.array(case["x0"]), **kw)
        arm.update(x=r.x.tolist(), cost=float(r.cost), status=int(r.status), nfev=int(r.nfev),
                   active_mask=[int(v) for v in r.active_mask])
    except Exception:
        pass
    out.append(arm)
print(json.dumps(out))
"#;
    let query_json = serde_json::to_string(query).expect("serialize trf query");
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
                "failed to spawn python3 for the trf oracle: {e}"
            );
            eprintln!("skipping trf oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open trf oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "trf oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping trf oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for trf oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "trf oracle failed: {stderr}"
        );
        eprintln!("skipping trf oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse trf oracle JSON"))
}

#[test]
fn diff_opt_least_squares_trf() {
    let cases = cases();
    let Some(oracle) = scipy_oracle_or_skip(&cases) else {
        return;
    };
    let arms: HashMap<String, OracleArm> = oracle
        .into_iter()
        .map(|arm| (arm.case_id.clone(), arm))
        .collect();
    let (t, y) = exp_data();

    let mut diffs = Vec::new();
    let mut ledger = CompareLedger::new("diff_opt_least_squares_trf", &["least_squares_trf"]);
    for case in &cases {
        let arm = &arms[&case.case_id];
        let options = LeastSquaresOptions {
            loss: loss_kind(&case.loss),
            f_scale: case.f_scale,
            x_scale_jac: case.x_scale_jac,
            max_nfev: case.max_nfev,
            ..LeastSquaresOptions::default()
        };
        let id = case.case_id.clone();
        let fun = |x: &[f64]| residuals(&id, x, &t, &y);
        let fsci = match (&case.lb, &case.ub) {
            (Some(lb), Some(ub)) => least_squares_bounded(fun, &case.x0, lb, ub, options),
            _ => least_squares(fun, &case.x0, options),
        };
        let mut diff = CaseDiff {
            case_id: case.case_id.clone(),
            x_rel_diff: f64::NAN,
            cost_rel_diff: f64::NAN,
            fsci_status: i32::MIN,
            scipy_status: arm.status.unwrap_or(i32::MIN),
            fsci_nfev: 0,
            scipy_nfev: arm.nfev.unwrap_or(0),
            pass: false,
            reason: String::new(),
        };
        let scipy = match (&arm.x, arm.cost, &arm.active_mask) {
            (Some(x), Some(cost), Some(mask)) => Some((x.as_slice(), cost, mask)),
            _ => None,
        };
        match (&fsci, scipy) {
            (Err(e), _) => diff.reason = format!("fsci error {e}"),
            (Ok(_), None) => diff.reason = "SciPy produced no result".to_string(),
            (Ok(_), Some(_)) => {}
        }
        // `None` is recorded by the ledger: SciPy gave no result, or fsci returned an error.
        if let Some(((x, cost, mask), r)) =
            ledger.both("least_squares_trf", &case.case_id, scipy, fsci.ok())
        {
            diff.fsci_status = r.status;
            diff.fsci_nfev = r.nfev;
            // The ledger rejects a length mismatch and a NaN coordinate (recording the case);
            // the zip below would truncate the one and the max fold swallow the other.
            let x_pair = ledger.slices(
                "least_squares_trf",
                &case.case_id,
                Some(x),
                Some(r.x.as_slice()),
            );
            diff.x_rel_diff = x_pair.map_or(f64::NAN, |(x, fsci_x)| {
                fsci_x
                    .iter()
                    .zip(x)
                    .map(|(a, b)| (a - b).abs() / b.abs().max(1e-6))
                    .fold(0.0, f64::max)
            });
            diff.cost_rel_diff = (r.cost - cost).abs() / cost.abs().max(1e-12);
            let mut problems = Vec::new();
            if r.status != diff.scipy_status {
                problems.push(format!("status {} vs {}", r.status, diff.scipy_status));
            }
            if &r.active_mask != mask {
                problems.push(format!("active_mask {:?} vs {mask:?}", r.active_mask));
            }
            if diff.x_rel_diff.is_nan() || diff.x_rel_diff > X_REL_TOL {
                problems.push(format!("x rel diff {:e}", diff.x_rel_diff));
            }
            // A zero-cost solution compares on the absolute scale (cost ≤ 1e-12 both).
            if diff.cost_rel_diff.is_nan() || diff.cost_rel_diff > COST_REL_TOL {
                problems.push(format!("cost rel diff {:e}", diff.cost_rel_diff));
            }
            diff.pass = problems.is_empty();
            diff.reason = problems.join("; ");
            if x_pair.is_some() {
                ledger.compared("least_squares_trf", &case.case_id, diff.pass);
            }
        }
        diffs.push(diff);
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let same_nfev_count = diffs
        .iter()
        .filter(|d| d.pass && d.fsci_nfev == d.scipy_nfev)
        .count();
    let log = DiffLog {
        test_id: "diff_opt_least_squares_trf".into(),
        category: "scipy.optimize.least_squares(method='trf')".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        same_nfev_count,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        cases: diffs.clone(),
    };
    fs::create_dir_all(output_dir()).expect("create trf diff dir");
    fs::write(
        output_dir().join("diff_opt_least_squares_trf.json"),
        serde_json::to_string_pretty(&log).expect("serialize trf log"),
    )
    .expect("write trf log");
    for d in &diffs {
        println!(
            "{} status {}/{} nfev {}/{} x_rel={:e} cost_rel={:e} {}",
            d.case_id,
            d.fsci_status,
            d.scipy_status,
            d.fsci_nfev,
            d.scipy_nfev,
            d.x_rel_diff,
            d.cost_rel_diff,
            d.reason
        );
    }
    println!(
        "{} cases compared, {same_nfev_count} with SciPy's evaluation count",
        diffs.len()
    );
    assert_eq!(diffs.len(), cases.len(), "every case must be compared");
    assert!(all_pass, "least_squares(trf) vs scipy failed");
    ledger.finish(cases.len());
}
