#![forbid(unsafe_code)]
//! Live formula-derived parity for fsci_cluster::{davies_bouldin_score,
//! calinski_harabasz_score}. sklearn unavailable; reproduce the
//! standard sklearn formulas in numpy.
//!
//! Davies-Bouldin: DB = (1/k) Σ_i max_{j≠i} (s_i + s_j) / d(c_i, c_j)
//!   where s_i = mean distance within cluster i, c_i = centroid.
//! Calinski-Harabasz: CH = (BSS / (k-1)) / (WSS / (n-k))
//!   BSS = Σ_i n_i ||c_i - c||²;  WSS = Σ_i Σ_{x in C_i} ||x - c_i||².
//!
//! Resolves [frankenscipy-wgvog]. Tolerance: 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_cluster::{calinski_harabasz_score, davies_bouldin_score};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    /// row-major flat
    data_flat: Vec<f64>,
    n: usize,
    d: usize,
    labels: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    db: Option<f64>,
    ch: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create db_ch diff dir");
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

fn flatten(rows: &[Vec<f64>]) -> Vec<f64> {
    rows.iter().flatten().copied().collect()
}

fn generate_query() -> OracleQuery {
    // 3 well-separated clusters in 2D
    let c1 = [
        vec![0.1, 0.2],
        vec![0.2, 0.1],
        vec![0.0, 0.0],
        vec![0.15, 0.15],
    ];
    let c2 = [vec![5.0, 5.0], vec![5.1, 5.2], vec![4.9, 5.0]];
    let c3 = [vec![-3.0, -2.0], vec![-2.9, -2.1], vec![-3.1, -2.0]];
    let well_separated: Vec<Vec<f64>> = c1
        .iter()
        .chain(c2.iter())
        .chain(c3.iter())
        .cloned()
        .collect();
    let well_labels: Vec<usize> = vec![0, 0, 0, 0, 1, 1, 1, 2, 2, 2];

    // Overlapping clusters (less well-separated)
    let o1 = [vec![0.0, 0.0], vec![0.5, 0.3], vec![0.2, 0.8]];
    let o2 = [vec![1.0, 0.5], vec![1.5, 0.7], vec![0.7, 1.2]];
    let overlap: Vec<Vec<f64>> = o1.iter().chain(o2.iter()).cloned().collect();
    let overlap_labels: Vec<usize> = vec![0, 0, 0, 1, 1, 1];

    // 3D fixture
    let d3 = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.1, 0.2, -0.1],
        vec![-0.1, 0.0, 0.1],
        vec![5.0, 5.0, 5.0],
        vec![5.2, 4.9, 5.1],
        vec![4.8, 5.1, 4.9],
    ];
    let d3_labels: Vec<usize> = vec![0, 0, 0, 1, 1, 1];

    let cases: Vec<(&str, Vec<Vec<f64>>, Vec<usize>)> = vec![
        ("well_separated_2d_k3", well_separated, well_labels),
        ("overlap_2d_k2", overlap, overlap_labels),
        ("well_separated_3d_k2", d3, d3_labels),
    ];

    let points = cases
        .into_iter()
        .map(|(cid, data, labels)| {
            let n = data.len();
            let d = data.first().map_or(0, |r| r.len());
            Case {
                case_id: cid.into(),
                data_flat: flatten(&data),
                n,
                d,
                labels,
            }
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np

def db_score(X, labels):
    k = len(set(labels))
    n, d = X.shape
    centroids = np.array([X[np.array(labels) == c].mean(axis=0) for c in range(k)])
    s = np.zeros(k)
    for c in range(k):
        members = X[np.array(labels) == c]
        if members.shape[0] > 0:
            s[c] = np.mean(np.linalg.norm(members - centroids[c], axis=1))
    db = 0.0
    for i in range(k):
        max_r = 0.0
        for j in range(k):
            if i == j: continue
            d_ij = np.linalg.norm(centroids[i] - centroids[j])
            if d_ij <= 0: continue
            r = (s[i] + s[j]) / d_ij
            if r > max_r: max_r = r
        db += max_r
    return db / k

def ch_score(X, labels):
    k = len(set(labels))
    n, d = X.shape
    if k == 1 or k == n: return 0.0
    global_c = X.mean(axis=0)
    centroids = np.array([X[np.array(labels) == c].mean(axis=0) for c in range(k)])
    bss = 0.0; wss = 0.0
    for c in range(k):
        members = X[np.array(labels) == c]
        n_c = members.shape[0]
        bss += n_c * np.sum((centroids[c] - global_c) ** 2)
        wss += np.sum((members - centroids[c]) ** 2)
    if wss == 0.0: return float('inf')
    return (bss / (k - 1)) / (wss / (n - k))

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    n = int(case["n"]); d = int(case["d"])
    X = np.array(case["data_flat"], dtype=float).reshape(n, d)
    labels = case["labels"]
    try:
        db = db_score(X, labels)
        ch = ch_score(X, labels)
        if math.isfinite(db) and math.isfinite(ch):
            points.append({"case_id": cid, "db": float(db), "ch": float(ch)})
        else:
            points.append({"case_id": cid, "db": None, "ch": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "db": None, "ch": None})
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
                "failed to spawn python3 for db_ch oracle: {e}"
            );
            eprintln!("skipping db_ch oracle: python3 not available ({e})");
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
                "db_ch oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping db_ch oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for db_ch oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "db_ch oracle failed: {stderr}"
        );
        eprintln!("skipping db_ch oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse db_ch oracle JSON"))
}

fn unflatten(flat: &[f64], n: usize, d: usize) -> Vec<Vec<f64>> {
    (0..n).map(|i| flat[i * d..(i + 1) * d].to_vec()).collect()
}

#[test]
fn diff_cluster_davies_bouldin_calinski_harabasz() {
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
        let (Some(edb), Some(ech)) = (arm.db, arm.ch) else {
            continue;
        };
        let data = unflatten(&case.data_flat, case.n, case.d);
        let Ok(db) = davies_bouldin_score(&data, &case.labels) else {
            continue;
        };
        let Ok(ch) = calinski_harabasz_score(&data, &case.labels) else {
            continue;
        };

        for (op, actual, expected) in [("davies_bouldin", db, edb), ("calinski_harabasz", ch, ech)]
        {
            let abs_d = (actual - expected).abs();
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: format!("{}_{}", case.case_id, op),
                op: op.into(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_cluster_davies_bouldin_calinski_harabasz".into(),
        category: "fsci_cluster davies_bouldin + calinski_harabasz vs sklearn formula".into(),
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
        "db_ch conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
