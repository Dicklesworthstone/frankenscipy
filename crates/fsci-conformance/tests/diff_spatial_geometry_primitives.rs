#![forbid(unsafe_code)]
//! Live numpy/scipy differential coverage for fsci_spatial
//! geometry summary primitives.
//!
//! Resolves [frankenscipy-xdu90]. Each is a closed-form
//! deterministic computation:
//!   • centroid(points) ≡ numpy.mean(points, axis=0)
//!   • medoid(points)   ≡ argmin over i of Σ_j ‖p_i − p_j‖
//!   • diameter(points) ≡ max ‖p_i − p_j‖ for i < j
//!   • spread(points)   ≡ mean ‖p_i − centroid‖
//!
//! 4 fixtures × 4 functions = 16 cases. Tol 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_spatial::{centroid, diameter, medoid, spread};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-011";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    points: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    centroid: Option<Vec<f64>>,
    medoid: Option<i64>,
    diameter: Option<f64>,
    spread: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    sub_check: String,
    detail: String,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create geometry-primitives diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize geometry-primitives diff log");
    fs::write(path, json).expect("write geometry-primitives diff log");
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        points: vec![
            // Simple 2-D, easy-to-verify centroid
            PointCase {
                case_id: "simple_2d_n4".into(),
                points: vec![
                    vec![0.0, 0.0],
                    vec![2.0, 0.0],
                    vec![0.0, 2.0],
                    vec![2.0, 2.0],
                ],
            },
            // 3-D blob
            PointCase {
                case_id: "blob_3d_n10".into(),
                points: vec![
                    vec![0.5, 0.25, 0.5],
                    vec![1.5, 0.5, 1.0],
                    vec![2.0, 0.25, 1.5],
                    vec![1.0, 0.75, 0.25],
                    vec![1.25, 0.5, 0.75],
                    vec![0.75, 0.25, 1.25],
                    vec![1.75, 0.75, 0.5],
                    vec![0.25, 1.0, 1.0],
                    vec![1.5, 0.25, 0.5],
                    vec![1.0, 0.5, 1.5],
                ],
            },
            // Unit cube vertices, 8 points in 3-D
            PointCase {
                case_id: "unit_cube_n8".into(),
                points: vec![
                    vec![0.0, 0.0, 0.0],
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![1.0, 1.0, 0.0],
                    vec![0.0, 0.0, 1.0],
                    vec![1.0, 0.0, 1.0],
                    vec![0.0, 1.0, 1.0],
                    vec![1.0, 1.0, 1.0],
                ],
            },
            // 1-D long chain
            PointCase {
                case_id: "chain_1d_n20".into(),
                points: (0..20).map(|i| vec![(i as f64) * 0.5]).collect(),
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
from scipy.spatial.distance import cdist, pdist

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
    P = np.asarray(case["points"], dtype=np.float64)
    out = {"case_id": cid, "centroid": None, "medoid": None, "diameter": None, "spread": None}
    try:
        c = P.mean(axis=0)
        out["centroid"] = [fnone(v) for v in c.tolist()]
        # medoid: argmin of sum of distances to all other points
        dist_sums = cdist(P, P).sum(axis=1)
        out["medoid"] = int(np.argmin(dist_sums))
        if P.shape[0] >= 2:
            out["diameter"] = fnone(pdist(P).max())
        else:
            out["diameter"] = 0.0
        out["spread"] = fnone(np.linalg.norm(P - c, axis=1).mean())
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}, allow_nan=False))
"#;
    let query_json = serde_json::to_string(query).expect("serialize geometry-primitives query");
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
                "failed to spawn python3 for geometry-primitives oracle: {e}"
            );
            eprintln!("skipping geometry-primitives oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open geometry-primitives oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "geometry-primitives oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping geometry-primitives oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for geometry-primitives oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "geometry-primitives oracle failed: {stderr}"
        );
        eprintln!("skipping geometry-primitives oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse geometry-primitives oracle JSON"))
}

#[test]
fn diff_spatial_geometry_primitives() {
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
    let mut cases = Vec::new();

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");

        // centroid
        if let Some(scipy_c) = scipy_arm.centroid.as_ref() {
            let rust_c = centroid(&case.points);
            let pass = rust_c.len() == scipy_c.len()
                && rust_c
                    .iter()
                    .zip(scipy_c.iter())
                    .all(|(r, s)| (r - s).abs() <= ABS_TOL);
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "centroid".into(),
                detail: format!("rust={rust_c:?}, scipy={scipy_c:?}"),
                pass,
            });
        }

        // medoid
        if let Some(scipy_m) = scipy_arm.medoid
            && let Some(rust_m) = medoid(&case.points)
        {
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "medoid".into(),
                detail: format!("rust={rust_m}, scipy={scipy_m}"),
                pass: rust_m as i64 == scipy_m,
            });
        }

        // diameter
        if let Some(scipy_d) = scipy_arm.diameter {
            let rust_d = diameter(&case.points);
            let abs_diff = (rust_d - scipy_d).abs();
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "diameter".into(),
                detail: format!("rust={rust_d}, scipy={scipy_d}, abs_diff={abs_diff}"),
                pass: abs_diff <= ABS_TOL,
            });
        }

        // spread
        if let Some(scipy_s) = scipy_arm.spread {
            let rust_s = spread(&case.points);
            let abs_diff = (rust_s - scipy_s).abs();
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "spread".into(),
                detail: format!("rust={rust_s}, scipy={scipy_s}, abs_diff={abs_diff}"),
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let pass_count = cases.iter().filter(|c| c.pass).count();
    let all_pass = pass_count == cases.len();

    let log = DiffLog {
        test_id: "diff_spatial_geometry_primitives".into(),
        category: "fsci_spatial::{centroid,medoid,diameter,spread}".into(),
        case_count: cases.len(),
        pass_count,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: cases.clone(),
    };

    emit_log(&log);

    for c in &cases {
        if !c.pass {
            eprintln!(
                "geometry-primitives mismatch: {} {} — {}",
                c.case_id, c.sub_check, c.detail
            );
        }
    }

    assert!(
        all_pass,
        "geometry-primitives conformance failed: {} of {} cases pass",
        pass_count,
        cases.len()
    );
}
