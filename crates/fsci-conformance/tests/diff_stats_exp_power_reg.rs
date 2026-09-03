#![forbid(unsafe_code)]
//! Live numerical reference checks for fsci's two nonlinear
//! regression utilities:
//!   • `exponential_regression(x, y)` — fits y = a · exp(b·x)
//!     via log-linear regression on (x, ln y).
//!   • `power_regression(x, y)` — fits y = a · x^b via
//!     log-log regression on (ln x, ln y).
//!
//! Resolves [frankenscipy-kjt1n]. The oracle reproduces both
//! transformations in numpy: linregress on transformed
//! targets, then `(exp(intercept), slope)`. Both fsci and
//! the oracle filter to positive y (and positive x for
//! power).
//!
//! 3 (x, y) fixtures × 2 funcs × 2 arms (a, b) = 12 cases
//! via subprocess. Tol 1e-9 abs (linregress precision +
//! exp(intercept) accumulator drift).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{exponential_regression, power_regression};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    x: Vec<f64>,
    y: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    a: Option<f64>,
    b: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
    arm: String,
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
    fs::create_dir_all(output_dir()).expect("create exp_power_reg diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize exp_power_reg diff log");
    fs::write(path, json).expect("write exp_power_reg diff log");
}

fn generate_query() -> OracleQuery {
    // Each fixture is a (x, y) pair where y > 0 (and x > 0 for power
    // regression). y is constructed from a known target so the
    // recovered (a, b) match the construction.
    let exp_fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // y = 2 * exp(0.5 * x)
        (
            "exp_clean",
            (1..=10).map(|i| i as f64).collect(),
            (1..=10).map(|i| 2.0 * (0.5 * i as f64).exp()).collect(),
        ),
        // y = 1 * exp(-0.3 * x) — decay
        (
            "exp_decay",
            (1..=12).map(|i| i as f64).collect(),
            (1..=12).map(|i| (-0.3 * i as f64).exp()).collect(),
        ),
        // y = 5 * exp(0.1 * x), small slope
        (
            "exp_small_b",
            (1..=15).map(|i| i as f64).collect(),
            (1..=15).map(|i| 5.0 * (0.1 * i as f64).exp()).collect(),
        ),
    ];
    let pow_fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // y = 2 * x^1.5
        (
            "pow_clean",
            (1..=10).map(|i| i as f64).collect(),
            (1..=10).map(|i| 2.0 * (i as f64).powf(1.5)).collect(),
        ),
        // y = 0.5 * x^2.0
        (
            "pow_quadratic",
            (1..=12).map(|i| i as f64).collect(),
            (1..=12).map(|i| 0.5 * (i as f64).powf(2.0)).collect(),
        ),
        // y = 3 * x^0.5
        (
            "pow_sqrt",
            (1..=15).map(|i| i as f64).collect(),
            (1..=15).map(|i| 3.0 * (i as f64).powf(0.5)).collect(),
        ),
    ];

    let mut points = Vec::new();
    for (name, x, y) in &exp_fixtures {
        points.push(PointCase {
            case_id: format!("exp_{name}"),
            func: "exponential_regression".into(),
            x: x.clone(),
            y: y.clone(),
        });
    }
    for (name, x, y) in &pow_fixtures {
        points.push(PointCase {
            case_id: format!("pow_{name}"),
            func: "power_regression".into(),
            x: x.clone(),
            y: y.clone(),
        });
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import stats

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    x = np.array(case["x"], dtype=float)
    y = np.array(case["y"], dtype=float)
    a_val = None; b_val = None
    try:
        if func == "exponential_regression":
            mask = y > 0
            xm = x[mask]
            lny = np.log(y[mask])
            res = stats.linregress(xm, lny)
            a_val = fnone(math.exp(res.intercept))
            b_val = fnone(res.slope)
        elif func == "power_regression":
            mask = (x > 0) & (y > 0)
            lnx = np.log(x[mask])
            lny = np.log(y[mask])
            res = stats.linregress(lnx, lny)
            a_val = fnone(math.exp(res.intercept))
            b_val = fnone(res.slope)
    except Exception:
        pass
    points.append({"case_id": cid, "a": a_val, "b": b_val})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize exp_power_reg query");
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
                "failed to spawn python3 for exp_power_reg oracle: {e}"
            );
            eprintln!("skipping exp_power_reg oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open exp_power_reg oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "exp_power_reg oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping exp_power_reg oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for exp_power_reg oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "exp_power_reg oracle failed: {stderr}"
        );
        eprintln!("skipping exp_power_reg oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse exp_power_reg oracle JSON"))
}

#[test]
fn diff_stats_exp_power_reg() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let (rust_a, rust_b) = match case.func.as_str() {
            "exponential_regression" => exponential_regression(&case.x, &case.y),
            "power_regression" => power_regression(&case.x, &case.y),
            _ => continue,
        };

        if let Some(scipy_a) = scipy_arm.a
            && rust_a.is_finite()
        {
            let abs_diff = (rust_a - scipy_a).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                arm: "a".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some(scipy_b) = scipy_arm.b
            && rust_b.is_finite()
        {
            let abs_diff = (rust_b - scipy_b).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                arm: "b".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_exp_power_reg".into(),
        category: "exponential_regression + power_regression".into(),
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
            eprintln!(
                "exp_power_reg {} mismatch: {} arm={} abs={}",
                d.func, d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "exp_power_reg conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
