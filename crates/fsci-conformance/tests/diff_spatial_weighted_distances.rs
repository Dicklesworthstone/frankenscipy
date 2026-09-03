#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_spatial::mahalanobis and
//! seuclidean (weighted/whitened distance metrics).
//!
//! Resolves [frankenscipy-uo2ln]. 1e-12 abs.
//!
//! NOTE: wminkowski is intentionally excluded — fsci preserves the
//! deprecated `(w_i * |x-y|)^p` convention while
//! scipy.spatial.distance.minkowski uses the current `w_i * |x-y|^p`
//! convention. Not a defect.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_spatial::{mahalanobis, seuclidean};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    x: Vec<f64>,
    y: Vec<f64>,
    /// For seuclidean: per-dim variance vector.
    v: Vec<f64>,
    /// For mahalanobis: VI as row-major flat (n × n).
    vi: Vec<f64>,
    n: usize,
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
    fs::create_dir_all(output_dir()).expect("create weighted_distances diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize weighted_distances diff log");
    fs::write(path, json).expect("write weighted_distances diff log");
}

fn rows_of(a_flat: &[f64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|r| (0..cols).map(|c| a_flat[r * cols + c]).collect())
        .collect()
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // seuclidean
    let seu_cases: &[(&str, Vec<f64>, Vec<f64>, Vec<f64>)] = &[
        (
            "seu_3",
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![1.0, 4.0, 9.0],
        ),
        (
            "seu_5",
            vec![0.1, 0.5, 1.0, 2.0, -1.0],
            vec![0.2, 0.4, 1.5, 1.8, 0.5],
            vec![0.5, 1.0, 2.0, 1.5, 1.0],
        ),
        (
            "seu_unit_var",
            vec![3.0, 4.0],
            vec![0.0, 0.0],
            vec![1.0, 1.0],
        ),
    ];
    for (name, x, y, v) in seu_cases {
        points.push(PointCase {
            case_id: (*name).into(),
            op: "seuclidean".into(),
            x: x.clone(),
            y: y.clone(),
            v: v.clone(),
            vi: vec![],
            n: x.len(),
        });
    }

    // mahalanobis — provide VI directly (positive-definite SPD inverse)
    // For the 3-D case, use VI = inv of a tridiagonal SPD matrix.
    let cov_3 = [[2.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 1.5]];
    let vi_3 = invert_3x3_spd(cov_3);
    let cov_2 = [[2.0, 0.6], [0.6, 1.0]];
    let vi_2 = invert_2x2_spd(cov_2);
    let cov_diag_4 = [
        [3.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 1.5, 0.0],
        [0.0, 0.0, 0.0, 4.0],
    ];
    let vi_diag_4 = invert_diag_4(cov_diag_4);

    let mah_cases: &[(&str, Vec<f64>, Vec<f64>, Vec<f64>, usize)] = &[
        (
            "mah_3_dense",
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vi_3.iter().flatten().copied().collect(),
            3,
        ),
        (
            "mah_2_dense",
            vec![1.0, -2.0],
            vec![-3.0, 4.0],
            vi_2.iter().flatten().copied().collect(),
            2,
        ),
        (
            "mah_4_diag",
            vec![0.5, 1.0, 2.0, -1.0],
            vec![1.5, 2.0, 0.0, 0.5],
            vi_diag_4.iter().flatten().copied().collect(),
            4,
        ),
    ];
    for (name, x, y, vi, n) in mah_cases {
        points.push(PointCase {
            case_id: (*name).into(),
            op: "mahalanobis".into(),
            x: x.clone(),
            y: y.clone(),
            v: vec![],
            vi: vi.clone(),
            n: *n,
        });
    }

    OracleQuery { points }
}

fn invert_2x2_spd(m: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    [
        [m[1][1] / det, -m[0][1] / det],
        [-m[1][0] / det, m[0][0] / det],
    ]
}

fn invert_3x3_spd(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let a = m;
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let cofactor = |i: usize, j: usize| -> f64 {
        let mut sub = [[0.0; 2]; 2];
        let mut sr = 0;
        for r in 0..3 {
            if r == i {
                continue;
            }
            let mut sc = 0;
            for c in 0..3 {
                if c == j {
                    continue;
                }
                sub[sr][sc] = a[r][c];
                sc += 1;
            }
            sr += 1;
        }
        let sign = if (i + j).is_multiple_of(2) { 1.0 } else { -1.0 };
        sign * (sub[0][0] * sub[1][1] - sub[0][1] * sub[1][0])
    };
    let mut inv = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            inv[j][i] = cofactor(i, j) / det;
        }
    }
    inv
}

fn invert_diag_4(m: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut inv = [[0.0_f64; 4]; 4];
    for i in 0..4 {
        inv[i][i] = 1.0 / m[i][i];
    }
    inv
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.spatial import distance

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    x = np.array(case["x"], dtype=float)
    y = np.array(case["y"], dtype=float)
    try:
        if op == "seuclidean":
            v = np.array(case["v"], dtype=float)
            d = float(distance.seuclidean(x, y, v))
        elif op == "mahalanobis":
            n = int(case["n"])
            vi = np.array(case["vi"], dtype=float).reshape(n, n)
            d = float(distance.mahalanobis(x, y, vi))
        else:
            d = None
        if d is None or not math.isfinite(d):
            points.append({"case_id": cid, "value": None})
        else:
            points.append({"case_id": cid, "value": d})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize weighted_distances query");
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
                "failed to spawn python3 for weighted_distances oracle: {e}"
            );
            eprintln!("skipping weighted_distances oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open weighted_distances oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "weighted_distances oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping weighted_distances oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for weighted_distances oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "weighted_distances oracle failed: {stderr}"
        );
        eprintln!("skipping weighted_distances oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse weighted_distances oracle JSON"))
}

#[test]
fn diff_spatial_weighted_distances() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.value else {
            continue;
        };
        let fsci_v = match case.op.as_str() {
            "seuclidean" => seuclidean(&case.x, &case.y, &case.v),
            "mahalanobis" => {
                let vi_mat = rows_of(&case.vi, case.n, case.n);
                mahalanobis(&case.x, &case.y, &vi_mat)
            }
            _ => continue,
        };
        let abs_d = (fsci_v - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_spatial_weighted_distances".into(),
        category: "scipy.spatial.distance seuclidean + mahalanobis".into(),
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
        "weighted_distances conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
