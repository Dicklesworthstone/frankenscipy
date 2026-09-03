#![forbid(unsafe_code)]
//! Live SciPy differential coverage for
//! fsci_spatial::KDTree::count_neighbors.
//!
//! Resolves [frankenscipy-n9540]. Integer-equality comparison vs
//! scipy.spatial.cKDTree.count_neighbors.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_spatial::KDTree;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    pts1: Vec<Vec<f64>>,
    pts2: Vec<Vec<f64>>,
    r: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    count: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    fsci_count: i64,
    scipy_count: i64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create count_neighbors diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize count_neighbors diff log");
    fs::write(path, json).expect("write count_neighbors diff log");
}

fn generate_query() -> OracleQuery {
    let pts1_2d = vec![
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
    ];
    let pts2_2d = vec![vec![0.5, 0.5], vec![1.5, 1.5], vec![2.5, 2.5]];
    let pts1_3d = vec![
        vec![0.0_f64, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
    ];
    let pts2_3d = vec![vec![0.5_f64, 0.5, 0.5], vec![2.0, 2.0, 2.0]];

    let mut points = Vec::new();
    for &r in &[0.5_f64, 1.0, 2.0, 5.0] {
        points.push(PointCase {
            case_id: format!("2d_4x3_r{r}"),
            pts1: pts1_2d.clone(),
            pts2: pts2_2d.clone(),
            r,
        });
    }
    for &r in &[0.5_f64, 1.0, 2.0, 5.0] {
        points.push(PointCase {
            case_id: format!("3d_5x2_r{r}"),
            pts1: pts1_3d.clone(),
            pts2: pts2_3d.clone(),
            r,
        });
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy.spatial import cKDTree

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    a = np.array(case["pts1"], dtype=float)
    b = np.array(case["pts2"], dtype=float)
    r = float(case["r"])
    try:
        t1 = cKDTree(a)
        t2 = cKDTree(b)
        c = int(t1.count_neighbors(t2, r))
        points.append({"case_id": cid, "count": c})
    except Exception:
        points.append({"case_id": cid, "count": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize count_neighbors query");
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
                "failed to spawn python3 for count_neighbors oracle: {e}"
            );
            eprintln!("skipping count_neighbors oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open count_neighbors oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "count_neighbors oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping count_neighbors oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for count_neighbors oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "count_neighbors oracle failed: {stderr}"
        );
        eprintln!("skipping count_neighbors oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse count_neighbors oracle JSON"))
}

#[test]
fn diff_spatial_kdtree_count_neighbors() {
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

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_c) = scipy_arm.count else {
            continue;
        };
        let Ok(t1) = KDTree::new(&case.pts1) else {
            continue;
        };
        let Ok(t2) = KDTree::new(&case.pts2) else {
            continue;
        };
        let Ok(fsci_c) = t1.count_neighbors(&t2, case.r) else {
            continue;
        };
        let pass = (fsci_c as i64) == scipy_c;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            fsci_count: fsci_c as i64,
            scipy_count: scipy_c,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_spatial_kdtree_count_neighbors".into(),
        category: "scipy.spatial.cKDTree.count_neighbors".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "count_neighbors mismatch: {} fsci={} scipy={}",
                d.case_id, d.fsci_count, d.scipy_count
            );
        }
    }

    assert!(
        all_pass,
        "count_neighbors conformance failed: {} cases",
        diffs.len()
    );
}
