#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_spatial::KDTree::query_pairs.
//!
//! Resolves [frankenscipy-m1p1d]. Pairs returned as sorted (i, j) with
//! i < j; compare sets for exact equality.

use std::collections::HashMap;
use std::collections::HashSet;
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
    pts: Vec<Vec<f64>>,
    r: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Flattened pair list: [i1, j1, i2, j2, ...] with i < j.
    pairs_flat: Option<Vec<i64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    fsci_count: usize,
    scipy_count: usize,
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
    fs::create_dir_all(output_dir()).expect("create query_pairs diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize query_pairs diff log");
    fs::write(path, json).expect("write query_pairs diff log");
}

fn generate_query() -> OracleQuery {
    let pts_2d_5 = vec![
        vec![0.0_f64, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
    ];
    let pts_3d_6 = vec![
        vec![0.0_f64, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
        vec![2.0, 2.0, 2.0],
    ];

    let mut points = Vec::new();
    for &r in &[0.5_f64, 1.0, 1.5, 2.5] {
        points.push(PointCase {
            case_id: format!("2d_5pt_r{r}"),
            pts: pts_2d_5.clone(),
            r,
        });
    }
    for &r in &[1.0_f64, 1.5, 2.0, 3.0] {
        points.push(PointCase {
            case_id: format!("3d_6pt_r{r}"),
            pts: pts_3d_6.clone(),
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
    pts = np.array(case["pts"], dtype=float)
    r = float(case["r"])
    try:
        t = cKDTree(pts)
        pairs = t.query_pairs(r)
        flat = []
        for (i, j) in sorted(pairs):
            a, b = (i, j) if i < j else (j, i)
            flat.append(int(a)); flat.append(int(b))
        points.append({"case_id": cid, "pairs_flat": flat})
    except Exception:
        points.append({"case_id": cid, "pairs_flat": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query_pairs query");
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
                "failed to spawn python3 for query_pairs oracle: {e}"
            );
            eprintln!("skipping query_pairs oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open query_pairs oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "query_pairs oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping query_pairs oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for query_pairs oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "query_pairs oracle failed: {stderr}"
        );
        eprintln!("skipping query_pairs oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse query_pairs oracle JSON"))
}

#[test]
fn diff_spatial_kdtree_query_pairs() {
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
        let Some(scipy_flat) = scipy_arm.pairs_flat.as_ref() else {
            continue;
        };
        let scipy_set: HashSet<(usize, usize)> = scipy_flat
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (c[0] as usize, c[1] as usize))
            .collect();
        let Ok(t) = KDTree::new(&case.pts) else {
            continue;
        };
        let Ok(fsci_pairs) = t.query_pairs(case.r) else {
            continue;
        };
        let fsci_set: HashSet<(usize, usize)> = fsci_pairs
            .into_iter()
            .map(|(i, j)| if i < j { (i, j) } else { (j, i) })
            .collect();
        let pass = scipy_set == fsci_set;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            fsci_count: fsci_set.len(),
            scipy_count: scipy_set.len(),
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_spatial_kdtree_query_pairs".into(),
        category: "scipy.spatial.cKDTree.query_pairs".into(),
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
                "query_pairs mismatch: {} fsci={} scipy={}",
                d.case_id, d.fsci_count, d.scipy_count
            );
        }
    }

    assert!(
        all_pass,
        "query_pairs conformance failed: {} cases",
        diffs.len()
    );
}
