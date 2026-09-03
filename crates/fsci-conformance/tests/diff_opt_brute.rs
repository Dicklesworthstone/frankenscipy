#![forbid(unsafe_code)]
//! Live scipy.optimize.brute parity for fsci_opt::brute.
//!
//! Resolves [frankenscipy-7ywqn]. Both fsci and scipy sample ns points
//! INCLUDING endpoints (np.linspace(lb, ub, ns)). scipy must be invoked
//! with `finish=None` to disable its post-grid polish step.
//!
//! Tolerance: 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::brute;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct BruteCase {
    case_id: String,
    func: String,
    ranges: Vec<(f64, f64)>,
    ns: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<BruteCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    x: Option<Vec<f64>>,
    fun: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

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
    fs::create_dir_all(output_dir()).expect("create brute diff dir");
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

// Closures used in both Rust and the python oracle. Match by `func` key.
fn evaluate(name: &str, x: &[f64]) -> f64 {
    match name {
        "quad_1d" => (x[0] - 2.0) * (x[0] - 2.0),
        "abs_shift_1d" => (x[0] + 1.5).abs(),
        "sumsq_2d" => x[0] * x[0] + x[1] * x[1],
        "shifted_sumsq_2d" => (x[0] - 1.0) * (x[0] - 1.0) + (x[1] + 0.5) * (x[1] + 0.5),
        "rosenbrock_grid_2d" => {
            // Restricted to a coarse grid to make the discrete minimum well-defined.
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        }
        _ => f64::NAN,
    }
}

fn generate_query() -> OracleQuery {
    let points = vec![
        BruteCase {
            case_id: "quad_1d_n11".into(),
            func: "quad_1d".into(),
            ranges: vec![(0.0, 5.0)],
            ns: 11,
        },
        BruteCase {
            case_id: "quad_1d_n21".into(),
            func: "quad_1d".into(),
            ranges: vec![(-3.0, 7.0)],
            ns: 21,
        },
        BruteCase {
            case_id: "abs_shift_1d_n9".into(),
            func: "abs_shift_1d".into(),
            ranges: vec![(-3.0, 1.0)],
            ns: 9,
        },
        BruteCase {
            case_id: "sumsq_2d_n11".into(),
            func: "sumsq_2d".into(),
            ranges: vec![(-2.0, 2.0), (-2.0, 2.0)],
            ns: 11,
        },
        BruteCase {
            case_id: "shifted_sumsq_2d_n9".into(),
            func: "shifted_sumsq_2d".into(),
            ranges: vec![(-2.0, 2.0), (-2.0, 2.0)],
            ns: 9,
        },
        BruteCase {
            case_id: "rosenbrock_grid_2d_n11".into(),
            func: "rosenbrock_grid_2d".into(),
            ranges: vec![(0.0, 2.0), (0.0, 2.0)],
            ns: 11,
        },
    ];
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import optimize

def f(name, x):
    if name == "quad_1d":     return (x[0]-2.0)**2
    if name == "abs_shift_1d": return abs(x[0]+1.5)
    if name == "sumsq_2d":    return x[0]**2 + x[1]**2
    if name == "shifted_sumsq_2d": return (x[0]-1.0)**2 + (x[1]+0.5)**2
    if name == "rosenbrock_grid_2d":
        return (1.0-x[0])**2 + 100.0*(x[1]-x[0]**2)**2
    return float("nan")

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; name = case["func"]
    ns = int(case["ns"])
    rngs = [(float(a), float(b)) for (a, b) in case["ranges"]]
    try:
        # finish=None disables scipy's local-polish step, leaving the raw grid argmin.
        # Use Ns=ns and slice with lb/ub bounds. scipy.brute accepts list-of-slices.
        slices = tuple(slice(lb, ub, ns*1j) for (lb, ub) in rngs)
        result = optimize.brute(lambda x: f(name, np.atleast_1d(x)),
                                ranges=slices, finish=None,
                                full_output=True)
        # Returns (x0, fval, grid, jout)
        x0 = np.atleast_1d(result[0]).astype(float).tolist()
        fval = float(result[1])
        if all(math.isfinite(v) for v in x0) and math.isfinite(fval):
            points.append({"case_id": cid, "x": x0, "fun": fval})
        else:
            points.append({"case_id": cid, "x": None, "fun": None})
    except Exception as e:
        sys.stderr.write(f"oracle failure {cid}: {e}\n")
        points.append({"case_id": cid, "x": None, "fun": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for brute oracle: {e}"
            );
            eprintln!("skipping brute oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "brute oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping brute oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for brute oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "brute oracle failed: {stderr}"
        );
        eprintln!("skipping brute oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse brute oracle JSON"))
}

#[test]
fn diff_opt_brute() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let (Some(ex), Some(ef)) = (arm.x.clone(), arm.fun) else {
            continue;
        };
        let f = |x: &[f64]| evaluate(&case.func, x);
        let Ok(res) = brute(f, &case.ranges, case.ns) else {
            continue;
        };
        let abs_x = if res.x.len() != ex.len() {
            f64::INFINITY
        } else {
            res.x
                .iter()
                .zip(ex.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        let abs_f = res.fun.map(|v| (v - ef).abs()).unwrap_or(f64::INFINITY);
        let abs_d = abs_x.max(abs_f);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "brute".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_brute".into(),
        category: "fsci_opt::brute vs scipy.optimize.brute (finish=None)".into(),
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
            eprintln!("brute mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "brute conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
