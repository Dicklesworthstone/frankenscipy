#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.odr` covariance reporting.
//!
//! The ODR covariance path must marginalize over fitted input corrections, not
//! just invert the beta-only normal matrix.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_odr::{Data, ODR, unilinear};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-018";
const REL_TOL: f64 = 1.5e-1;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

type TestResult<T> = Result<T, String>;

#[derive(Debug, Clone, Serialize)]
struct OdrCase {
    case_id: &'static str,
    x: Vec<f64>,
    y: Vec<f64>,
    we: Vec<f64>,
    wd: Vec<f64>,
    beta0: Vec<f64>,
    sclb: Option<Vec<f64>>,
    scld: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<OdrCase>,
    analytic_jacobian: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleCase {
    case_id: String,
    beta: Vec<f64>,
    cov_beta: Vec<Vec<f64>>,
    sd_beta: Vec<f64>,
    res_var: f64,
    info: i32,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OracleCase>,
}

#[derive(Debug, Clone)]
struct FsciCaseResult {
    beta: Vec<f64>,
    cov_beta: Vec<Vec<f64>>,
    sd_beta: Vec<f64>,
    res_var: f64,
    success: bool,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    beta_rel_diff: f64,
    sd_beta_rel_diff: f64,
    cov_diag_rel_diff: f64,
    res_var_rel_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    max_rel_diff: f64,
    tolerance: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() -> TestResult<()> {
    fs::create_dir_all(output_dir()).map_err(|err| format!("create odr diff output dir: {err}"))
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn emit_log(log: &DiffLog) -> TestResult<()> {
    ensure_output_dir()?;
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log)
        .map_err(|err| format!("serialize odr diff log: {err}"))?;
    fs::write(path, json).map_err(|err| format!("write odr diff log: {err}"))
}

fn generate_query() -> OracleQuery {
    let x = (0..7).map(f64::from).collect::<Vec<_>>();
    let y = vec![1.20, 2.90, 5.15, 6.80, 9.10, 10.95, 13.12];
    OracleQuery {
        analytic_jacobian: false,
        points: vec![
            OdrCase {
                case_id: "linear_equal_input_response_weights",
                x: x.clone(),
                y: y.clone(),
                we: vec![1.0; y.len()],
                wd: vec![1.0; x.len()],
                beta0: vec![1.5, 0.5],
                sclb: None,
                scld: None,
            },
            OdrCase {
                case_id: "linear_stronger_input_weight",
                x: x.clone(),
                y,
                we: vec![1.0; x.len()],
                wd: vec![4.0; x.len()],
                beta0: vec![1.5, 0.5],
                sclb: None,
                scld: None,
            },
        ],
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> TestResult<Option<OracleResult>> {
    let mut preflight = match fsci_conformance::scipy_oracle_command()
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => {
            if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
                return Err(format!(
                    "{REQUIRE_SCIPY_ENV}=1 but python3 is unavailable for scipy.odr oracle"
                ));
            }
            eprintln!("skipping odr oracle: python3 not available");
            return Ok(None);
        }
    };
    {
        let Some(stdin) = preflight.stdin.as_mut() else {
            return Err("open odr oracle preflight stdin".into());
        };
        if let Err(err) = stdin.write_all(b"import numpy\nfrom scipy import odr\n") {
            if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
                return Err(format!("write scipy.odr preflight script: {err}"));
            }
            eprintln!("skipping odr oracle: preflight write failed ({err})");
            return Ok(None);
        }
    }
    let available = preflight.wait().is_ok_and(|status| status.success());
    if !available {
        if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
            return Err(format!(
                "{REQUIRE_SCIPY_ENV}=1 but numpy/scipy.odr is unavailable"
            ));
        }
        eprintln!("skipping odr oracle: numpy/scipy.odr not available");
        return Ok(None);
    }

    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import odr

query = json.loads(sys.argv[1])
points = []

for case in query["points"]:
    x = np.array(case["x"], dtype=float)
    y = np.array(case["y"], dtype=float)
    we = np.array(case["we"], dtype=float)
    wd = np.array(case["wd"], dtype=float)
    beta0 = np.array(case["beta0"], dtype=float)
    model = (odr.unilinear if query["analytic_jacobian"]
             else odr.Model(lambda beta, x: beta[0] * x + beta[1]))
    data = odr.Data(x, y, we=we, wd=wd)
    solver = odr.ODR(data, model, beta0=beta0,
                     sclb=case["sclb"], scld=case["scld"])
    if query["analytic_jacobian"]:
        solver.set_job(deriv=3)
    out = solver.run()
    cov = np.asarray(out.cov_beta, dtype=float)
    points.append({
        "case_id": case["case_id"],
        "beta": [float(v) for v in out.beta],
        "cov_beta": [[float(v) for v in row] for row in cov.tolist()],
        "sd_beta": [float(v) for v in out.sd_beta],
        "res_var": float(out.res_var),
        "info": int(out.info),
    })

print(json.dumps({"points": points}))
"#;
    let query_json =
        serde_json::to_string(query).map_err(|err| format!("serialize odr query: {err}"))?;
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-")
        .arg(&query_json)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
                return Err(format!("failed to spawn python3 for odr oracle: {err}"));
            }
            eprintln!("skipping odr oracle: python3 not available ({err})");
            return Ok(None);
        }
    };
    {
        let Some(stdin) = child.stdin.as_mut() else {
            return Err("open odr oracle stdin".into());
        };
        if let Err(err) = stdin.write_all(script.as_bytes()) {
            let output = child
                .wait_with_output()
                .map_err(|wait_err| format!("wait for failed odr oracle: {wait_err}"))?;
            let stderr = String::from_utf8_lossy(&output.stderr);
            if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
                return Err(format!(
                    "odr oracle script write failed: {err}; stderr: {stderr}"
                ));
            }
            eprintln!("skipping odr oracle: script write failed ({err})\n{stderr}");
            return Ok(None);
        }
    }
    let output = child
        .wait_with_output()
        .map_err(|err| format!("wait for odr oracle: {err}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
            return Err(format!("odr oracle failed: {stderr}"));
        }
        eprintln!("skipping odr oracle: scipy not available\n{stderr}");
        return Ok(None);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout)
        .map(Some)
        .map_err(|err| format!("parse odr oracle JSON: {err}"))
}

fn run_fsci_case(case: &OdrCase) -> TestResult<FsciCaseResult> {
    let data = Data::new(case.x.clone(), case.y.clone())
        .map_err(|err| format!("build odr data: {err}"))?
        .with_response_weights(case.we.clone())
        .map_err(|err| format!("set odr response weights: {err}"))?
        .with_input_weights(case.wd.clone())
        .map_err(|err| format!("set odr input weights: {err}"))?;
    let mut solver = ODR::new(data, unilinear(), case.beta0.clone())
        .map_err(|err| format!("build odr solver: {err}"))?;
    if let Some(scale) = &case.sclb {
        solver = solver
            .with_beta_scale(scale.clone())
            .map_err(|err| format!("set sclb: {err}"))?;
    }
    if let Some(scale) = &case.scld {
        solver = solver
            .with_input_scale(scale.clone())
            .map_err(|err| format!("set scld: {err}"))?;
    }
    let output = solver
        .run()
        .map_err(|err| format!("run odr solver: {err}"))?;
    Ok(FsciCaseResult {
        beta: output.beta,
        cov_beta: output.cov_beta,
        sd_beta: output.sd_beta,
        res_var: output.res_var,
        success: output.success,
    })
}

fn rel_diff(left: f64, right: f64) -> f64 {
    if left == right {
        return 0.0;
    }
    (left - right).abs() / right.abs().max(1.0e-12)
}

fn max_rel_diff_vec(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() {
        return f64::INFINITY;
    }
    left.iter()
        .zip(right.iter())
        .map(|(&lhs, &rhs)| rel_diff(lhs, rhs))
        .fold(0.0_f64, f64::max)
}

fn max_rel_diff_diag(left: &[Vec<f64>], right: &[Vec<f64>]) -> f64 {
    if left.len() != right.len() {
        return f64::INFINITY;
    }
    left.iter()
        .zip(right.iter())
        .enumerate()
        .map(
            |(idx, (lhs_row, rhs_row))| match (lhs_row.get(idx), rhs_row.get(idx)) {
                (Some(&lhs), Some(&rhs)) => rel_diff(lhs, rhs),
                _ => f64::INFINITY,
            },
        )
        .fold(0.0_f64, f64::max)
}

/// The diagonal the covariance metric reads, for the ledger's presence and non-finite checks.
/// A row too short to hold its diagonal element reads as NaN.
fn diagonal(matrix: &[Vec<f64>]) -> Vec<f64> {
    matrix
        .iter()
        .enumerate()
        .map(|(idx, row)| row.get(idx).copied().unwrap_or(f64::NAN))
        .collect()
}

#[test]
fn diff_odr_covariance_coupling() -> TestResult<()> {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query)? else {
        return Ok(());
    };
    if oracle.points.len() != query.points.len() {
        return Err(format!(
            "oracle returned {} points for {} queries",
            oracle.points.len(),
            query.points.len()
        ));
    }

    let oracle_by_id = oracle
        .points
        .into_iter()
        .map(|point| (point.case_id.clone(), point))
        .collect::<HashMap<_, _>>();
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_rel = 0.0_f64;
    let mut ledger = CompareLedger::new(
        "diff_odr_covariance_coupling",
        &["beta", "sd_beta", "cov_diag", "res_var"],
    );

    for case in &query.points {
        let Some(expected) = oracle_by_id.get(case.case_id) else {
            return Err(format!("oracle omitted case {}", case.case_id));
        };
        let id = case.case_id;
        let actual = run_fsci_case(case)
            .inspect_err(|err| eprintln!("odr fsci failure: {id} {err}"))
            .ok();
        let scipy_diag = diagonal(&expected.cov_beta);
        let fsci_diag = actual.as_ref().map(|a| diagonal(&a.cov_beta));
        let beta = ledger
            .slices(
                "beta",
                id,
                Some(expected.beta.as_slice()),
                actual.as_ref().map(|a| a.beta.as_slice()),
            )
            .map(|(s, f)| max_rel_diff_vec(f, s));
        let sd_beta = ledger
            .slices(
                "sd_beta",
                id,
                Some(expected.sd_beta.as_slice()),
                actual.as_ref().map(|a| a.sd_beta.as_slice()),
            )
            .map(|(s, f)| max_rel_diff_vec(f, s));
        let cov_diag = ledger
            .slices(
                "cov_diag",
                id,
                Some(scipy_diag.as_slice()),
                fsci_diag.as_deref(),
            )
            .and(actual.as_ref())
            .map(|a| max_rel_diff_diag(&a.cov_beta, &expected.cov_beta));
        let res_var = ledger
            .pair(
                "res_var",
                id,
                Some(expected.res_var),
                actual.as_ref().map(|a| a.res_var),
            )
            .map(|(s, f)| rel_diff(f, s));
        for (arm, d) in [
            ("beta", beta),
            ("sd_beta", sd_beta),
            ("cov_diag", cov_diag),
            ("res_var", res_var),
        ] {
            if let Some(d) = d {
                ledger.compared(arm, id, d <= REL_TOL);
            }
        }
        let (
            Some(beta_rel_diff),
            Some(sd_beta_rel_diff),
            Some(cov_diag_rel_diff),
            Some(res_var_rel_diff),
        ) = (beta, sd_beta, cov_diag, res_var)
        else {
            continue; // the ledger recorded why this case was not compared
        };
        let case_max = beta_rel_diff
            .max(sd_beta_rel_diff)
            .max(cov_diag_rel_diff)
            .max(res_var_rel_diff);
        max_rel = max_rel.max(case_max);
        diffs.push(CaseDiff {
            case_id: case.case_id.into(),
            beta_rel_diff,
            sd_beta_rel_diff,
            cov_diag_rel_diff,
            res_var_rel_diff,
            pass: case_max <= REL_TOL,
        });
    }

    let all_pass = diffs.iter().all(|diff| diff.pass);
    let log = DiffLog {
        test_id: "diff_odr_covariance_coupling".into(),
        category: "scipy.odr.covariance".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_rel_diff: max_rel,
        tolerance: REL_TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log)?;

    for diff in &diffs {
        if !diff.pass {
            eprintln!(
                "odr covariance mismatch: {} beta_rel={} sd_rel={} cov_diag_rel={} res_var_rel={}",
                diff.case_id,
                diff.beta_rel_diff,
                diff.sd_beta_rel_diff,
                diff.cov_diag_rel_diff,
                diff.res_var_rel_diff
            );
        }
    }

    if !all_pass {
        return Err(format!(
            "scipy.odr covariance conformance failed: {} cases, max_rel={max_rel}",
            diffs.len()
        ));
    }
    ledger.finish(query.points.len());
    Ok(())
}

#[test]
fn diff_odr_mixed_magnitude_scaling() -> TestResult<()> {
    const SCALE_REL_TOL: f64 = 1e-6;
    let mut points = Vec::new();
    for (ids, x_scale, beta, beta0, sclb, noise) in [
        (
            ["tiny_default", "tiny_explicit"],
            1e-9,
            [2.0, 1e-9],
            [1.0, 2e-9],
            [0.5, 1e9],
            1e-12,
        ),
        (
            ["large_slope_default", "large_slope_explicit"],
            1e-3,
            [1e3, 0.25],
            [5e2, 0.5],
            [1e-3, 1.0],
            1e-3,
        ),
        (
            ["small_slope_default", "small_slope_explicit"],
            1e3,
            [1e-3, 0.25],
            [5e-4, 0.5],
            [1e3, 1.0],
            1e-3,
        ),
    ] {
        let x: Vec<f64> = (0..13).map(|i| (f64::from(i) - 6.0) * x_scale).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| beta[0] * v + beta[1] + noise * (0.7 * i as f64).sin())
            .collect();
        for (explicit, case_id) in ids.into_iter().enumerate() {
            points.push(OdrCase {
                case_id,
                x: x.clone(),
                y: y.clone(),
                we: vec![1.0; y.len()],
                wd: vec![1.0; x.len()],
                beta0: beta0.to_vec(),
                sclb: (explicit == 1).then(|| sclb.to_vec()),
                scld: (explicit == 1).then(|| vec![1.0 / x_scale]),
            });
        }
    }
    // unilinear() supplies analytic Jacobians in Rust. Give ODRPACK the same
    // derivatives so tiny-variable covariance is not limited by finite-difference
    // roundoff in the oracle. The original covariance test retains its FD oracle.
    let query = OracleQuery {
        points,
        analytic_jacobian: true,
    };
    let Some(oracle) = scipy_oracle_or_skip(&query)? else {
        return Ok(());
    };
    assert_eq!(oracle.points.len(), query.points.len());
    let mut ledger = CompareLedger::new("diff_odr_mixed_magnitude_scaling", &["beta", "cov_diag"]);
    for (case, expected) in query.points.iter().zip(&oracle.points) {
        assert_eq!(case.case_id, expected.case_id);
        assert!(
            (1..=3).contains(&(expected.info % 10)),
            "{}: SciPy info={}",
            case.case_id,
            expected.info
        );
        let actual = run_fsci_case(case)?;
        assert!(actual.success, "{}: fsci did not converge", case.case_id);
        let expected_diag = diagonal(&expected.cov_beta);
        let actual_diag = diagonal(&actual.cov_beta);
        for (arm, expected_values, actual_values) in [
            ("beta", expected.beta.as_slice(), actual.beta.as_slice()),
            ("cov_diag", expected_diag.as_slice(), actual_diag.as_slice()),
        ] {
            if let Some((s, f)) = ledger.slices(
                arm,
                case.case_id,
                Some(expected_values),
                Some(actual_values),
            ) {
                let error = max_rel_diff_vec(f, s);
                println!(
                    "ODR scaling {} {arm}: SciPy={s:?} fsci={f:?} relative_error={error:.3e}",
                    case.case_id
                );
                ledger.compared(arm, case.case_id, error <= SCALE_REL_TOL);
            }
        }
    }
    ledger.finish(query.points.len());
    Ok(())
}
