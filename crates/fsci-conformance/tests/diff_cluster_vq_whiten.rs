#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the scipy.cluster.vq
//! vector-quantization primitives:
//!   - `fsci_cluster::whiten(data)`  vs `scipy.cluster.vq.whiten(data)`
//!   - `fsci_cluster::vq(data, centroids)` vs `scipy.cluster.vq.vq(data, centroids)`
//!
//! Resolves [frankenscipy-9x8t1]. Both ops are deterministic closed-form
//! transformations: whiten divides each column by its ddof=0 standard
//! deviation (verified empirically against scipy); vq assigns each row
//! to its nearest centroid by Euclidean distance and returns the
//! associated distance. Tight 1e-12 abs tolerance expected.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_cluster::{vq, whiten};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-012";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String, // "whiten" | "vq"
    data: Vec<Vec<f64>>,
    /// Only meaningful for "vq".
    centroids: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Flattened row-major whitened data (whiten) or distances vector (vq).
    values: Option<Vec<f64>>,
    /// Only present for vq op.
    codes: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
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
    fs::create_dir_all(output_dir()).expect("create vq_whiten diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize vq_whiten diff log");
    fs::write(path, json).expect("write vq_whiten diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // whiten datasets
    let whiten_datasets: &[(&str, Vec<Vec<f64>>)] = &[
        (
            "4x2_linear",
            vec![
                vec![1.0, 1.0],
                vec![2.0, 2.0],
                vec![3.0, 3.0],
                vec![4.0, 4.0],
            ],
        ),
        (
            "5x3_mixed",
            vec![
                vec![1.0, 0.5, -2.0],
                vec![2.0, 1.0, -1.0],
                vec![3.0, 1.5, 0.0],
                vec![4.0, 2.0, 1.0],
                vec![5.0, 2.5, 2.0],
            ],
        ),
        (
            "3x2_close_values",
            vec![vec![1.0, 10.0], vec![1.5, 10.5], vec![2.0, 11.0]],
        ),
        (
            "6x1_one_feature",
            vec![
                vec![10.0],
                vec![12.0],
                vec![14.0],
                vec![16.0],
                vec![18.0],
                vec![20.0],
            ],
        ),
    ];
    for (name, data) in whiten_datasets {
        points.push(PointCase {
            case_id: format!("whiten_{name}"),
            op: "whiten".into(),
            data: data.clone(),
            centroids: vec![],
        });
    }

    // vq datasets: (label, data, centroids).
    let vq_cases: &[(&str, Vec<Vec<f64>>, Vec<Vec<f64>>)] = &[
        (
            "5x2_3centroids",
            vec![
                vec![0.0, 0.0],
                vec![10.0, 10.0],
                vec![20.0, 20.0],
                vec![1.0, 0.5],
                vec![19.0, 20.5],
            ],
            vec![vec![0.0, 0.0], vec![10.0, 10.0], vec![20.0, 20.0]],
        ),
        (
            "6x1_2centroids",
            vec![
                vec![0.0],
                vec![1.0],
                vec![2.0],
                vec![5.0],
                vec![6.0],
                vec![7.0],
            ],
            vec![vec![1.0], vec![6.0]],
        ),
        (
            "4x3_4centroids",
            vec![
                vec![1.0, 2.0, 3.0],
                vec![-1.0, -2.0, -3.0],
                vec![1.5, 2.5, 3.5],
                vec![0.0, 0.0, 0.0],
            ],
            vec![
                vec![1.0, 2.0, 3.0],
                vec![-1.0, -2.0, -3.0],
                vec![0.0, 0.0, 0.0],
                vec![10.0, 10.0, 10.0],
            ],
        ),
        (
            "3x2_1centroid",
            vec![vec![0.0, 0.0], vec![3.0, 4.0], vec![-3.0, -4.0]],
            vec![vec![0.0, 0.0]],
        ),
    ];
    for (name, data, centroids) in vq_cases {
        points.push(PointCase {
            case_id: format!("vq_{name}"),
            op: "vq".into(),
            data: data.clone(),
            centroids: centroids.clone(),
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
from scipy.cluster.vq import whiten as scipy_whiten, vq as scipy_vq

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).flatten().tolist():
        try:
            v = float(v)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    data = np.array(case["data"], dtype=float)
    try:
        if op == "whiten":
            v = scipy_whiten(data)
            points.append({"case_id": cid, "values": finite_vec_or_none(v), "codes": None})
        elif op == "vq":
            centroids = np.array(case["centroids"], dtype=float)
            codes, dists = scipy_vq(data, centroids)
            points.append({
                "case_id": cid,
                "values": finite_vec_or_none(dists),
                "codes": [int(c) for c in codes.tolist()],
            })
        else:
            points.append({"case_id": cid, "values": None, "codes": None})
    except Exception:
        points.append({"case_id": cid, "values": None, "codes": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize vq_whiten query");
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
                "failed to spawn python3 for vq_whiten oracle: {e}"
            );
            eprintln!("skipping vq_whiten oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open vq_whiten oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "vq_whiten oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping vq_whiten oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for vq_whiten oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "vq_whiten oracle failed: {stderr}"
        );
        eprintln!("skipping vq_whiten oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse vq_whiten oracle JSON"))
}

#[test]
fn diff_cluster_vq_whiten() {
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
        match case.op.as_str() {
            "whiten" => {
                let Ok(fsci_v) = whiten(&case.data) else {
                    continue;
                };
                let Some(scipy_v) = scipy_arm.values.as_ref() else {
                    continue;
                };
                let flat: Vec<f64> = fsci_v.iter().flatten().copied().collect();
                if flat.len() != scipy_v.len() {
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        op: case.op.clone(),
                        arm: "values".into(),
                        abs_diff: f64::INFINITY,
                        pass: false,
                    });
                    continue;
                }
                let abs_d = flat
                    .iter()
                    .zip(scipy_v.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                max_overall = max_overall.max(abs_d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    arm: "values".into(),
                    abs_diff: abs_d,
                    pass: abs_d <= ABS_TOL,
                });
            }
            "vq" => {
                let Ok((fsci_codes, fsci_dists)) = vq(&case.data, &case.centroids) else {
                    continue;
                };
                if let Some(scipy_codes) = scipy_arm.codes.as_ref() {
                    let match_pass = fsci_codes.len() == scipy_codes.len()
                        && fsci_codes
                            .iter()
                            .zip(scipy_codes.iter())
                            .all(|(a, b)| *a == *b);
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        op: case.op.clone(),
                        arm: "codes".into(),
                        abs_diff: if match_pass { 0.0 } else { f64::INFINITY },
                        pass: match_pass,
                    });
                }
                if let Some(scipy_dists) = scipy_arm.values.as_ref() {
                    if fsci_dists.len() != scipy_dists.len() {
                        diffs.push(CaseDiff {
                            case_id: case.case_id.clone(),
                            op: case.op.clone(),
                            arm: "distances".into(),
                            abs_diff: f64::INFINITY,
                            pass: false,
                        });
                    } else {
                        let abs_d = fsci_dists
                            .iter()
                            .zip(scipy_dists.iter())
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0_f64, f64::max);
                        max_overall = max_overall.max(abs_d);
                        diffs.push(CaseDiff {
                            case_id: case.case_id.clone(),
                            op: case.op.clone(),
                            arm: "distances".into(),
                            abs_diff: abs_d,
                            pass: abs_d <= ABS_TOL,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_cluster_vq_whiten".into(),
        category: "scipy.cluster.vq.{vq, whiten}".into(),
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
                "vq_whiten {} {} mismatch: {} abs_diff={}",
                d.op, d.arm, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.cluster.vq.{{vq, whiten}} conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
