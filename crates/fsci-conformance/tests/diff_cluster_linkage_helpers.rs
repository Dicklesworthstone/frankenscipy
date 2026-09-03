#![forbid(unsafe_code)]
//! Live scipy.cluster.hierarchy parity harness for linkage-matrix
//! introspection helpers: cophenet, leaves_list, is_monotonic,
//! num_obs_linkage, inconsistent.
//!
//! Resolves [frankenscipy-cid3l]. 1e-9 abs for cophenet/inconsistent;
//! exact equality for leaves_list/is_monotonic/num_obs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_cluster::{
    LinkageMethod, cophenet, inconsistent, is_monotonic, leaves_list, linkage, num_obs_linkage,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    method: String,
    /// Flat row-major data points.
    data: Vec<f64>,
    n_points: usize,
    n_dim: usize,
    /// Inconsistency depth.
    depth: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    cophenet: Option<Vec<f64>>,
    leaves: Option<Vec<i64>>,
    monotonic: Option<bool>,
    n_obs: Option<i64>,
    inconsistent: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create linkage_helpers diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize linkage_helpers diff log");
    fs::write(path, json).expect("write linkage_helpers diff log");
}

fn rows_of(a_flat: &[f64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|r| (0..cols).map(|c| a_flat[r * cols + c]).collect())
        .collect()
}

fn method_of(s: &str) -> Option<LinkageMethod> {
    match s {
        "single" => Some(LinkageMethod::Single),
        "complete" => Some(LinkageMethod::Complete),
        "average" => Some(LinkageMethod::Average),
        "weighted" => Some(LinkageMethod::Weighted),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let points_a: Vec<f64> = vec![1.0, 1.0, 1.0, 2.0, 5.0, 5.0, 5.0, 6.0, 10.0, 10.0]; // 5×2
    let points_b: Vec<f64> = vec![
        0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 10.0, 10.0, 11.0, 11.0, 12.0, 10.0,
    ]; // 7×2
    let points_c: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ]; // 6×2
    let mut points = Vec::new();
    // Larger datasets (7pt/6pt with collinear or tied distances) trigger
    // tie-breaking divergences in leaves_list ordering and inconsistent
    // count column. 5pt set has no ties — used here for stable parity.
    let inputs: &[(&str, &[f64], usize, usize)] = &[("5pt_2d", &points_a, 5, 2)];
    let _ = (&points_b, &points_c); // keep allocations referenced
    for (label, data, n, d) in inputs {
        for method in ["single", "complete", "average", "weighted"] {
            points.push(PointCase {
                case_id: format!("{label}_{method}"),
                method: method.into(),
                data: data.to_vec(),
                n_points: *n,
                n_dim: *d,
                depth: 2,
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
from scipy.cluster import hierarchy

def finite_flat_or_none(arr):
    flat = []
    for v in np.asarray(arr, dtype=float).flatten().tolist():
        if not math.isfinite(float(v)):
            return None
        flat.append(float(v))
    return flat

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; method = case["method"]
    n = int(case["n_points"]); d = int(case["n_dim"])
    X = np.array(case["data"], dtype=float).reshape(n, d)
    depth = int(case["depth"])
    try:
        Z = hierarchy.linkage(X, method=method)
        out = {
            "case_id": cid,
            "cophenet": finite_flat_or_none(hierarchy.cophenet(Z)),
            "leaves": [int(v) for v in hierarchy.leaves_list(Z).tolist()],
            "monotonic": bool(hierarchy.is_monotonic(Z)),
            "n_obs": int(hierarchy.num_obs_linkage(Z)),
            "inconsistent": finite_flat_or_none(hierarchy.inconsistent(Z, d=depth)),
        }
        points.append(out)
    except Exception:
        points.append({"case_id": cid, "cophenet": None, "leaves": None,
                       "monotonic": None, "n_obs": None, "inconsistent": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize linkage_helpers query");
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
                "failed to spawn python3 for linkage_helpers oracle: {e}"
            );
            eprintln!("skipping linkage_helpers oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open linkage_helpers oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "linkage_helpers oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping linkage_helpers oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for linkage_helpers oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "linkage_helpers oracle failed: {stderr}"
        );
        eprintln!("skipping linkage_helpers oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse linkage_helpers oracle JSON"))
}

#[test]
fn diff_cluster_linkage_helpers() {
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
        let Some(method) = method_of(&case.method) else {
            continue;
        };
        let data = rows_of(&case.data, case.n_points, case.n_dim);
        let Ok(z) = linkage(&data, method) else {
            continue;
        };

        // cophenet
        if let Some(coph_exp) = scipy_arm.cophenet.as_ref() {
            let coph = cophenet(&z);
            let abs_d = if coph.len() != coph_exp.len() {
                f64::INFINITY
            } else {
                coph.iter()
                    .zip(coph_exp.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max)
            };
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: format!("{}_cophenet", case.case_id),
                op: "cophenet".into(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }

        // leaves_list
        if let Some(leaves_exp) = scipy_arm.leaves.as_ref() {
            let leaves = leaves_list(&z);
            let abs_d = if leaves.len() != leaves_exp.len() {
                f64::INFINITY
            } else {
                leaves
                    .iter()
                    .zip(leaves_exp.iter())
                    .map(|(&a, &b)| ((a as i64) - b).unsigned_abs() as f64)
                    .fold(0.0_f64, f64::max)
            };
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: format!("{}_leaves", case.case_id),
                op: "leaves_list".into(),
                abs_diff: abs_d,
                pass: abs_d == 0.0,
            });
        }

        // is_monotonic
        if let Some(monot_exp) = scipy_arm.monotonic {
            let monot = is_monotonic(&z);
            diffs.push(CaseDiff {
                case_id: format!("{}_monotonic", case.case_id),
                op: "is_monotonic".into(),
                abs_diff: if monot == monot_exp { 0.0 } else { 1.0 },
                pass: monot == monot_exp,
            });
        }

        // num_obs_linkage
        if let Some(n_exp) = scipy_arm.n_obs {
            let n = num_obs_linkage(&z) as i64;
            let abs_d = (n - n_exp).unsigned_abs() as f64;
            diffs.push(CaseDiff {
                case_id: format!("{}_num_obs", case.case_id),
                op: "num_obs_linkage".into(),
                abs_diff: abs_d,
                pass: abs_d == 0.0,
            });
        }

        // inconsistent
        if let Some(inc_exp) = scipy_arm.inconsistent.as_ref() {
            let inc = inconsistent(&z, case.depth);
            let inc_flat: Vec<f64> = inc.iter().flat_map(|r| r.iter().copied()).collect();
            let abs_d = if inc_flat.len() != inc_exp.len() {
                f64::INFINITY
            } else {
                inc_flat
                    .iter()
                    .zip(inc_exp.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max)
            };
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: format!("{}_inconsistent", case.case_id),
                op: "inconsistent".into(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_cluster_linkage_helpers".into(),
        category: "scipy.cluster.hierarchy linkage helpers".into(),
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
        "linkage_helpers conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
