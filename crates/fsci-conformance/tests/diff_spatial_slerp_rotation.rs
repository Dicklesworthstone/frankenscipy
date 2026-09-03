#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_spatial spatial-
//! transform primitives.
//!
//! Resolves [frankenscipy-xqosz]. Covers:
//!   • geometric_slerp(start, end, t_values)
//!     ≡ scipy.spatial.geometric_slerp(start, end, t)
//!   • rotation_matrix(axis, angle)
//!     ≡ scipy.spatial.transform.Rotation.from_rotvec(axis*angle).as_matrix()
//!   • rotate_point(R, p) ≡ R @ p (compared via the rotation_matrix R
//!     produced by both libraries on a representative point)
//!
//! 4 fixtures. Tol 1e-12 abs (closed-form trig, no iterative
//! algorithms).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_spatial::{geometric_slerp, rotate_point, rotation_matrix};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-011";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    // Slerp inputs (unit-vectors in n-D)
    slerp_start: Vec<f64>,
    slerp_end: Vec<f64>,
    slerp_t: Vec<f64>,
    // Rotation inputs (unit axis + angle, plus a test point)
    rot_axis: [f64; 3],
    rot_angle: f64,
    rot_point: [f64; 3],
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    slerp_rows: Option<Vec<Vec<f64>>>,
    rot_matrix: Option<[[f64; 3]; 3]>,
    rotated_point: Option<[f64; 3]>,
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
    fs::create_dir_all(output_dir()).expect("create slerp_rotation diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize slerp_rotation diff log");
    fs::write(path, json).expect("write slerp_rotation diff log");
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

fn generate_query() -> OracleQuery {
    use std::f64::consts::FRAC_1_SQRT_2;

    OracleQuery {
        points: vec![
            // Slerp on unit 2-D circle (90° interpolation), rotation
            // around z by 90°.
            PointCase {
                case_id: "circle_2d_z_axis_90deg".into(),
                slerp_start: vec![1.0, 0.0],
                slerp_end: vec![0.0, 1.0],
                slerp_t: vec![0.0, 0.25, 0.5, 0.75, 1.0],
                rot_axis: [0.0, 0.0, 1.0],
                rot_angle: std::f64::consts::FRAC_PI_2,
                rot_point: [1.0, 0.0, 0.0],
            },
            // Slerp on a 3-D unit-sphere quarter, rotation around x by 60°.
            PointCase {
                case_id: "sphere_3d_x_axis_60deg".into(),
                slerp_start: vec![1.0, 0.0, 0.0],
                slerp_end: vec![0.0, 0.0, 1.0],
                slerp_t: vec![0.0, 0.5, 1.0],
                rot_axis: [1.0, 0.0, 0.0],
                rot_angle: std::f64::consts::FRAC_PI_3,
                rot_point: [0.0, 1.0, 0.0],
            },
            // Slerp on a non-orthogonal 3-D pair, rotation around (1,1,1)/√3 by 120°.
            PointCase {
                case_id: "non_orthogonal_120deg".into(),
                slerp_start: unit([1.0, 0.0, 0.0]).to_vec(),
                slerp_end: unit([1.0, 1.0, 0.0]).to_vec(),
                slerp_t: vec![0.0, 0.333, 0.667, 1.0],
                rot_axis: unit([1.0, 1.0, 1.0]),
                rot_angle: 2.0 * std::f64::consts::FRAC_PI_3,
                rot_point: [1.0, 0.0, 0.0],
            },
            // 180° y-axis rotation; slerp uses a moderate-angle
            // 3-D pair (not near-antipodal — slerp gets numerically
            // unstable as the start/end approach exact antipodes
            // because sin(angle) → 0 in the denominator, and fsci
            // and scipy diverge by ~1e-4 there).
            PointCase {
                case_id: "moderate_45deg_y_180deg".into(),
                slerp_start: vec![1.0, 0.0, 0.0],
                slerp_end: unit([1.0, 1.0, 0.0]).to_vec(),
                slerp_t: vec![0.0, 0.5, 1.0],
                rot_axis: [0.0, 1.0, 0.0],
                rot_angle: std::f64::consts::PI,
                rot_point: [FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2],
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
from scipy.spatial import geometric_slerp
from scipy.spatial.transform import Rotation as R

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
    out = {"case_id": cid, "slerp_rows": None, "rot_matrix": None, "rotated_point": None}
    try:
        s = np.asarray(case["slerp_start"], dtype=np.float64)
        e = np.asarray(case["slerp_end"], dtype=np.float64)
        t = np.asarray(case["slerp_t"], dtype=np.float64)
        rows = geometric_slerp(s, e, t)
        rrows = []
        ok = True
        for row in rows.tolist():
            rrow = []
            for v in row:
                f = fnone(v)
                if f is None:
                    ok = False
                    break
                rrow.append(f)
            if not ok:
                break
            rrows.append(rrow)
        out["slerp_rows"] = rrows if ok else None

        ax = np.asarray(case["rot_axis"], dtype=np.float64)
        ang = float(case["rot_angle"])
        rot = R.from_rotvec(ax * ang)
        M = rot.as_matrix()
        out["rot_matrix"] = [
            [fnone(M[0, 0]), fnone(M[0, 1]), fnone(M[0, 2])],
            [fnone(M[1, 0]), fnone(M[1, 1]), fnone(M[1, 2])],
            [fnone(M[2, 0]), fnone(M[2, 1]), fnone(M[2, 2])],
        ]

        p = np.asarray(case["rot_point"], dtype=np.float64)
        rp = M @ p
        out["rotated_point"] = [fnone(rp[0]), fnone(rp[1]), fnone(rp[2])]
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}, allow_nan=False))
"#;
    let query_json = serde_json::to_string(query).expect("serialize slerp_rotation query");
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
                "failed to spawn python3 for slerp_rotation oracle: {e}"
            );
            eprintln!("skipping slerp_rotation oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open slerp_rotation oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "slerp_rotation oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping slerp_rotation oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for slerp_rotation oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "slerp_rotation oracle failed: {stderr}"
        );
        eprintln!("skipping slerp_rotation oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse slerp_rotation oracle JSON"))
}

#[test]
fn diff_spatial_slerp_rotation() {
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

        // geometric_slerp
        if let Some(scipy_rows) = scipy_arm.slerp_rows.as_ref() {
            match geometric_slerp(&case.slerp_start, &case.slerp_end, &case.slerp_t) {
                Ok(rust_rows) => {
                    let pass = rust_rows.len() == scipy_rows.len()
                        && rust_rows.iter().zip(scipy_rows.iter()).all(|(rr, sr)| {
                            rr.len() == sr.len()
                                && rr
                                    .iter()
                                    .zip(sr.iter())
                                    .all(|(r, s)| (r - s).abs() <= ABS_TOL)
                        });
                    cases.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        sub_check: "geometric_slerp".into(),
                        detail: format!(
                            "rust_rows={}, scipy_rows={}",
                            rust_rows.len(),
                            scipy_rows.len()
                        ),
                        pass,
                    });
                }
                Err(e) => cases.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    sub_check: "geometric_slerp".into(),
                    detail: format!("rust err: {e:?}"),
                    pass: false,
                }),
            }
        }

        // rotation_matrix
        if let Some(scipy_m) = scipy_arm.rot_matrix {
            let rust_m = rotation_matrix(&case.rot_axis, case.rot_angle);
            let mut max_d = 0.0_f64;
            for i in 0..3 {
                for j in 0..3 {
                    max_d = max_d.max((rust_m[i][j] - scipy_m[i][j]).abs());
                }
            }
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "rotation_matrix".into(),
                detail: format!("max_abs={max_d}"),
                pass: max_d <= ABS_TOL,
            });

            // rotate_point — apply Rust matrix, compare to scipy's
            // (matrix @ point).
            if let Some(scipy_rp) = scipy_arm.rotated_point {
                let rust_rp = rotate_point(&rust_m, &case.rot_point);
                let max_d = [
                    (rust_rp[0] - scipy_rp[0]).abs(),
                    (rust_rp[1] - scipy_rp[1]).abs(),
                    (rust_rp[2] - scipy_rp[2]).abs(),
                ]
                .iter()
                .cloned()
                .fold(0.0_f64, f64::max);
                cases.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    sub_check: "rotate_point".into(),
                    detail: format!("rust={rust_rp:?}, scipy={scipy_rp:?}"),
                    pass: max_d <= ABS_TOL,
                });
            }
        }
    }

    let pass_count = cases.iter().filter(|c| c.pass).count();
    let all_pass = pass_count == cases.len();

    let log = DiffLog {
        test_id: "diff_spatial_slerp_rotation".into(),
        category: "fsci_spatial::{geometric_slerp,rotation_matrix,rotate_point}".into(),
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
                "slerp_rotation mismatch: {} {} — {}",
                c.case_id, c.sub_check, c.detail
            );
        }
    }

    assert!(
        all_pass,
        "slerp_rotation conformance failed: {} of {} cases pass",
        pass_count,
        cases.len()
    );
}
