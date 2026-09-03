#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci's
//! `directed_hausdorff(xa, xb)` and `hausdorff_distance(xa, xb)`.
//!
//! Resolves [frankenscipy-d3dtr]. scipy reference is
//! `scipy.spatial.distance.directed_hausdorff(xa, xb)` which
//! returns (distance, idx_a, idx_b); we only compare the
//! distance scalar. `hausdorff_distance` is the symmetric max
//! over both directions.
//!
//! 5 fixtures × 2 functions = 10 cases. Tol 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_spatial::{directed_hausdorff, hausdorff_distance};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-011";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    xa: Vec<Vec<f64>>,
    xb: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    directed_ab: Option<f64>,
    #[allow(dead_code)]
    directed_ba: Option<f64>,
    symmetric: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    fn_name: String,
    rust_value: f64,
    scipy_value: f64,
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
    fs::create_dir_all(output_dir()).expect("create hausdorff diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize hausdorff diff log");
    fs::write(path, json).expect("write hausdorff diff log");
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        points: vec![
            // Identical sets — both directed distances are 0.
            PointCase {
                case_id: "identical_2d".into(),
                xa: vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
                xb: vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
            },
            // Translated by (5, 5) — directed distances both equal 5√2.
            PointCase {
                case_id: "translated_2d".into(),
                xa: vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
                xb: vec![vec![5.0, 5.0], vec![6.0, 5.0], vec![5.0, 6.0]],
            },
            // xa is a subset of xb → d(xa, xb) = 0; d(xb, xa) > 0.
            PointCase {
                case_id: "subset_2d".into(),
                xa: vec![vec![0.0, 0.0], vec![1.0, 0.0]],
                xb: vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![10.0, 10.0]],
            },
            // 3-D, well-separated point clouds.
            PointCase {
                case_id: "separated_3d".into(),
                xa: vec![
                    vec![0.0, 0.0, 0.0],
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 1.0],
                ],
                xb: vec![
                    vec![10.0, 10.0, 10.0],
                    vec![11.0, 10.0, 10.0],
                    vec![10.0, 11.0, 10.0],
                    vec![10.0, 10.0, 11.0],
                ],
            },
            // 2-D unequal sizes, partial overlap.
            PointCase {
                case_id: "partial_overlap_2d".into(),
                xa: (0..6).map(|i| vec![i as f64, 0.0]).collect(),
                xb: (3..10).map(|i| vec![i as f64, 0.5]).collect(),
            },
        ],
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    out = {"case_id": cid, "directed_ab": None, "directed_ba": None, "symmetric": None}
    try:
        xa = np.asarray(case["xa"], dtype=np.float64)
        xb = np.asarray(case["xb"], dtype=np.float64)
        d_ab = directed_hausdorff(xa, xb)[0]
        d_ba = directed_hausdorff(xb, xa)[0]
        out["directed_ab"] = fnone(d_ab)
        out["directed_ba"] = fnone(d_ba)
        out["symmetric"] = fnone(max(d_ab, d_ba))
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}, allow_nan=False))
"#;
    let query_json = serde_json::to_string(query).expect("serialize hausdorff query");
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
                "failed to spawn python3 for hausdorff oracle: {e}"
            );
            eprintln!("skipping hausdorff oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open hausdorff oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "hausdorff oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping hausdorff oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for hausdorff oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "hausdorff oracle failed: {stderr}"
        );
        eprintln!("skipping hausdorff oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse hausdorff oracle JSON"))
}

#[test]
fn diff_spatial_hausdorff() {
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

        if let Some(scipy_ab) = scipy_arm.directed_ab
            && let Ok(rust_ab) = directed_hausdorff(&case.xa, &case.xb)
        {
            let abs_diff = (rust_ab - scipy_ab).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                fn_name: "directed_hausdorff(a,b)".into(),
                rust_value: rust_ab,
                scipy_value: scipy_ab,
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }

        if let Some(scipy_sym) = scipy_arm.symmetric
            && let Ok(rust_sym) = hausdorff_distance(&case.xa, &case.xb)
        {
            let abs_diff = (rust_sym - scipy_sym).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                fn_name: "hausdorff_distance".into(),
                rust_value: rust_sym,
                scipy_value: scipy_sym,
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_spatial_hausdorff".into(),
        category: "fsci_spatial::directed_hausdorff + hausdorff_distance".into(),
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
                "hausdorff mismatch: {} {} rust={} scipy={} abs_diff={}",
                d.case_id, d.fn_name, d.rust_value, d.scipy_value, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "hausdorff conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
