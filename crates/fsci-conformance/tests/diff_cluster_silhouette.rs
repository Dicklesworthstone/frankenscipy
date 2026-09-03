#![forbid(unsafe_code)]
//! Live formula-derived parity for fsci_cluster::{silhouette_score,
//! silhouette_samples}. sklearn unavailable; reproduce the standard
//! silhouette formula in numpy.
//!
//! For each sample i: s(i) = (b(i) - a(i)) / max(a(i), b(i))
//!   a(i) = mean Euclidean distance to other points in its cluster.
//!   b(i) = min over other clusters K of mean distance to points in K.
//! silhouette_score = mean over all i.
//! silhouette_samples returns the vector of s(i).
//!
//! Resolves [frankenscipy-2x5lj]. Tolerance: 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_cluster::{silhouette_samples, silhouette_score};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
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
    score: Option<f64>,
    samples: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create silhouette diff dir");
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
    let c1 = [
        vec![0.0, 0.0],
        vec![0.1, 0.2],
        vec![0.2, 0.1],
        vec![0.15, 0.05],
    ];
    let c2 = [vec![5.0, 5.0], vec![5.2, 5.1], vec![4.9, 5.0]];
    let c3 = [vec![-3.0, -2.0], vec![-2.9, -2.1], vec![-3.1, -1.9]];
    let well_sep: Vec<Vec<f64>> = c1
        .iter()
        .chain(c2.iter())
        .chain(c3.iter())
        .cloned()
        .collect();
    let well_labels: Vec<usize> = vec![0, 0, 0, 0, 1, 1, 1, 2, 2, 2];

    let o1 = [vec![0.0, 0.0], vec![0.5, 0.3], vec![0.2, 0.8]];
    let o2 = [vec![1.0, 0.5], vec![1.5, 0.7], vec![0.7, 1.2]];
    let overlap: Vec<Vec<f64>> = o1.iter().chain(o2.iter()).cloned().collect();
    let overlap_labels: Vec<usize> = vec![0, 0, 0, 1, 1, 1];

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
        ("well_sep_2d_k3", well_sep, well_labels),
        ("overlap_2d_k2", overlap, overlap_labels),
        ("well_sep_3d_k2", d3, d3_labels),
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

def silhouette_samples(X, labels):
    n = X.shape[0]
    labels = np.array(labels)
    out = np.zeros(n)
    # Pairwise distances
    dists = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    for i in range(n):
        own = labels[i]
        # a(i): mean distance to others in own cluster (exclude self)
        own_mask = (labels == own)
        own_mask[i] = False
        if own_mask.sum() == 0:
            out[i] = 0.0
            continue
        a_i = dists[i, own_mask].mean()
        # b(i): min over other clusters of mean distance
        b_i = float("inf")
        for c in set(labels.tolist()):
            if c == own: continue
            mask = (labels == c)
            if mask.sum() == 0: continue
            mean_dist = dists[i, mask].mean()
            if mean_dist < b_i: b_i = mean_dist
        if b_i == float("inf"):
            out[i] = 0.0
            continue
        max_ab = max(a_i, b_i)
        if max_ab > 0:
            out[i] = (b_i - a_i) / max_ab
        else:
            out[i] = 0.0
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    n = int(case["n"]); d = int(case["d"])
    X = np.array(case["data_flat"], dtype=float).reshape(n, d)
    labels = case["labels"]
    try:
        samples = silhouette_samples(X, labels)
        score = float(samples.mean())
        s_list = [float(v) for v in samples.tolist()]
        if all(math.isfinite(v) for v in s_list) and math.isfinite(score):
            points.append({"case_id": cid, "score": score, "samples": s_list})
        else:
            points.append({"case_id": cid, "score": None, "samples": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "score": None, "samples": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
    let mut child = match Command::new("python3")
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
                "failed to spawn python3 for silhouette oracle: {e}"
            );
            eprintln!("skipping silhouette oracle: python3 not available ({e})");
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
                "silhouette oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping silhouette oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for silhouette oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "silhouette oracle failed: {stderr}"
        );
        eprintln!("skipping silhouette oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse silhouette oracle JSON"))
}

fn unflatten(flat: &[f64], n: usize, d: usize) -> Vec<Vec<f64>> {
    (0..n).map(|i| flat[i * d..(i + 1) * d].to_vec()).collect()
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
fn diff_cluster_silhouette() {
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
        let (Some(escore), Some(esamples)) = (arm.score, arm.samples.as_ref()) else {
            continue;
        };
        let data = unflatten(&case.data_flat, case.n, case.d);
        let Ok(score) = silhouette_score(&data, &case.labels) else {
            continue;
        };
        let Ok(samples) = silhouette_samples(&data, &case.labels) else {
            continue;
        };

        let abs_score = (score - escore).abs();
        max_overall = max_overall.max(abs_score);
        diffs.push(CaseDiff {
            case_id: format!("{}_score", case.case_id),
            op: "silhouette_score".into(),
            abs_diff: abs_score,
            pass: abs_score <= ABS_TOL,
        });
        let abs_samples = vec_max_diff(&samples, esamples);
        max_overall = max_overall.max(abs_samples);
        diffs.push(CaseDiff {
            case_id: format!("{}_samples", case.case_id),
            op: "silhouette_samples".into(),
            abs_diff: abs_samples,
            pass: abs_samples <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_cluster_silhouette".into(),
        category: "fsci_cluster silhouette_score + silhouette_samples vs sklearn formula".into(),
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
        "silhouette conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
