#![forbid(unsafe_code)]
//! Live numerical reference checks for the matrix-valued
//! statistics functions:
//!   • `cov_matrix(data)`  — sample covariance (ddof=1),
//!     equivalent to `numpy.cov(data.T, rowvar=False)` after
//!     transposition (fsci uses rows = observations).
//!   • `corr_matrix(data)` — Pearson correlation matrix,
//!     equivalent to `numpy.corrcoef(data.T, rowvar=False)`.
//!
//! Resolves [frankenscipy-3x4jy]. The oracle uses numpy's
//! cov/corrcoef directly. Each case compares the full matrix
//! element-wise (max-abs aggregation per case).
//!
//! 3 datasets × 2 funcs = 6 cases via subprocess. Tol 1e-12
//! abs (closed-form sums + ratios).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{corr_matrix, cov_matrix};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    data: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    matrix: Option<Vec<Vec<f64>>>,
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
    fs::create_dir_all(output_dir()).expect("create cov_corr_matrix diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize cov_corr_matrix diff log");
    fs::write(path, json).expect("write cov_corr_matrix diff log");
}

fn generate_query() -> OracleQuery {
    // data is n × d (rows = observations, cols = features).
    let datasets: Vec<(&str, Vec<Vec<f64>>)> = vec![
        // 2D dataset, 8 obs
        (
            "two_d",
            (1..=8)
                .map(|i| vec![i as f64, (i as f64).powi(2) / 10.0])
                .collect(),
        ),
        // 3D dataset, 12 obs
        (
            "three_d",
            (1..=12)
                .map(|i| {
                    let x = i as f64;
                    vec![x, x.sqrt(), (x - 5.0).abs()]
                })
                .collect(),
        ),
        // 2D with strong correlation
        (
            "correlated",
            (1..=10)
                .map(|i| {
                    let x = i as f64;
                    vec![x, 2.0 * x + 1.0]
                })
                .collect(),
        ),
    ];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        for func in ["cov_matrix", "corr_matrix"] {
            points.push(PointCase {
                case_id: format!("{name}_{func}"),
                func: func.into(),
                data: data.clone(),
            });
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np

def mat_or_none(arr):
    out = []
    for row in arr:
        row_out = []
        for v in row:
            try:
                v = float(v)
            except Exception:
                return None
            if not math.isfinite(v):
                return None
            row_out.append(v)
        out.append(row_out)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    # data is rows=obs, cols=features (fsci convention).
    data = np.array(case["data"], dtype=float)
    val = None
    try:
        if func == "cov_matrix":
            # numpy.cov uses rowvar=True by default (rows = variables);
            # we have rows=obs so transpose for ddof=1 sample cov.
            val = mat_or_none(np.cov(data.T, ddof=1).tolist())
        elif func == "corr_matrix":
            val = mat_or_none(np.corrcoef(data.T).tolist())
    except Exception:
        val = None
    points.append({"case_id": cid, "matrix": val})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize cov_corr_matrix query");
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
                "failed to spawn python3 for cov_corr_matrix oracle: {e}"
            );
            eprintln!("skipping cov_corr_matrix oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open cov_corr_matrix oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "cov_corr_matrix oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping cov_corr_matrix oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for cov_corr_matrix oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "cov_corr_matrix oracle failed: {stderr}"
        );
        eprintln!("skipping cov_corr_matrix oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse cov_corr_matrix oracle JSON"))
}

#[test]
fn diff_stats_cov_corr_matrix() {
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
        let Some(scipy_mat) = &scipy_arm.matrix else {
            continue;
        };
        let rust_mat = match case.func.as_str() {
            "cov_matrix" => cov_matrix(&case.data),
            "corr_matrix" => corr_matrix(&case.data),
            _ => continue,
        };
        if rust_mat.len() != scipy_mat.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let mut max_local = 0.0_f64;
        let mut shape_ok = true;
        for (rrow, srow) in rust_mat.iter().zip(scipy_mat.iter()) {
            if rrow.len() != srow.len() {
                shape_ok = false;
                break;
            }
            for (a, b) in rrow.iter().zip(srow.iter()) {
                if a.is_finite() {
                    max_local = max_local.max((a - b).abs());
                }
            }
        }
        if !shape_ok {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        max_overall = max_overall.max(max_local);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            abs_diff: max_local,
            pass: max_local <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_cov_corr_matrix".into(),
        category: "cov_matrix + corr_matrix (numpy reference)".into(),
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
                "cov_corr_matrix {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "cov_corr_matrix conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
