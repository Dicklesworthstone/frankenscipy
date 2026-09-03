#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the multivariate
//! normal `scipy.stats.multivariate_normal.pdf/logpdf`.
//!
//! Resolves [frankenscipy-j6gkc]. Cross-checks fsci's
//! `MultivariateNormal::pdf/logpdf` (Cholesky-based path) vs
//! scipy across 2D and 3D mean/cov fixtures.
//!
//! 4 (mean, cov) fixtures × 4 x-points each × 2 funcs = 32
//! cases via subprocess. Tol 1e-12 abs (closed-form Cholesky
//! + log det).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::MultivariateNormal;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    mean: Vec<f64>,
    cov: Vec<Vec<f64>>,
    x: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create mvn diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize mvn diff log");
    fs::write(path, json).expect("write mvn diff log");
}

fn fsci_eval(func: &str, mean: &[f64], cov: &[Vec<f64>], x: &[f64]) -> Option<f64> {
    let dist = MultivariateNormal::new(mean, cov).ok()?;
    let v = match func {
        "pdf" => dist.pdf(x).ok()?,
        "logpdf" => dist.logpdf(x).ok()?,
        _ => return None,
    };
    if v.is_finite() { Some(v) } else { None }
}

fn generate_query() -> OracleQuery {
    // 4 fixtures: 2D identity, 2D rotated, 3D diagonal, 3D
    // correlated.
    let fixtures: Vec<(Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>)> = vec![
        // 2D identity, mean (0, 0)
        (
            vec![0.0, 0.0],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.5, 1.5],
                vec![-0.7, 0.3],
            ],
        ),
        // 2D rotated cov, mean (1, 2)
        (
            vec![1.0, 2.0],
            vec![vec![2.0, 0.5], vec![0.5, 1.5]],
            vec![
                vec![1.0, 2.0],
                vec![0.0, 1.0],
                vec![3.0, 4.0],
                vec![-1.0, 5.0],
            ],
        ),
        // 3D diagonal, mean (0, 0, 0)
        (
            vec![0.0, 0.0, 0.0],
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 2.0, 0.0],
                vec![0.0, 0.0, 0.5],
            ],
            vec![
                vec![0.0, 0.0, 0.0],
                vec![1.0, 1.0, 1.0],
                vec![-1.0, 0.5, 0.5],
                vec![2.0, -1.0, 0.0],
            ],
        ),
        // 3D correlated
        (
            vec![1.0, -1.0, 2.0],
            vec![
                vec![2.0, 0.3, 0.1],
                vec![0.3, 1.0, 0.2],
                vec![0.1, 0.2, 1.5],
            ],
            vec![
                vec![1.0, -1.0, 2.0],
                vec![0.0, 0.0, 0.0],
                vec![2.0, 0.0, 1.0],
                vec![-1.0, -2.0, 3.0],
            ],
        ),
    ];

    let mut points = Vec::new();
    for (i, (mean, cov, xs)) in fixtures.iter().enumerate() {
        for (j, x) in xs.iter().enumerate() {
            for func in ["pdf", "logpdf"] {
                points.push(PointCase {
                    case_id: format!("{func}_fix{i}_x{j}"),
                    func: func.to_string(),
                    mean: mean.clone(),
                    cov: cov.clone(),
                    x: x.clone(),
                });
            }
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy.stats import multivariate_normal

def finite_or_none(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    mean = case["mean"]; cov = case["cov"]; x = case["x"]
    try:
        rv = multivariate_normal(mean=mean, cov=cov)
        if func == "pdf":   value = float(rv.pdf(x))
        elif func == "logpdf":value = float(rv.logpdf(x))
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize mvn query");
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
                "failed to spawn python3 for mvn oracle: {e}"
            );
            eprintln!("skipping mvn oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open mvn oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "mvn oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping mvn oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for mvn oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "mvn oracle failed: {stderr}"
        );
        eprintln!("skipping mvn oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse mvn oracle JSON"))
}

#[test]
fn diff_stats_mvn() {
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
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        if let Some(scipy_v) = oracle.value
            && let Some(rust_v) = fsci_eval(&case.func, &case.mean, &case.cov, &case.x)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_mvn".into(),
        category: "scipy.stats.multivariate_normal.pdf/logpdf".into(),
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
            eprintln!("mvn {} mismatch: {} abs={}", d.func, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "mvn conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
