#![forbid(unsafe_code)]
//! Live numpy differential coverage for fsci_spatial coordinate
//! transforms.
//!
//! Resolves [frankenscipy-lyl4z]. fsci uses ISO 80000-2 (physics)
//! convention:
//!   • spherical (r, θ, φ): r ≥ 0, θ ∈ [0, π] is the polar angle
//!     from +z, φ ∈ (-π, π] is the azimuth from +x in the xy-plane.
//!     x = r·sin(θ)·cos(φ), y = r·sin(θ)·sin(φ), z = r·cos(θ).
//!   • cylindrical (ρ, θ, z): ρ ≥ 0, θ ∈ (-π, π], z arbitrary.
//!     x = ρ·cos(θ), y = ρ·sin(θ).
//!
//! scipy has no built-in coordinate transforms; the oracle
//! reproduces these formulas in numpy. Each forward direction is
//! diffed against numpy, and round-trip identity is checked
//! (cart → sph → cart ≈ original; cart → cyl → cart ≈ original).
//!
//! 5 fixtures × {forward sph, forward cyl, sph round-trip, cyl
//! round-trip} ≈ 20 cases. Tol 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_spatial::{
    cartesian_to_cylindrical, cartesian_to_spherical, cylindrical_to_cartesian,
    spherical_to_cartesian,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-011";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    sph: Option<[f64; 3]>,
    cyl: Option<[f64; 3]>,
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
    fs::create_dir_all(output_dir()).expect("create coord-transforms diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize coord-transforms diff log");
    fs::write(path, json).expect("write coord-transforms diff log");
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        points: vec![
            // 1. Generic 3-D point
            PointCase {
                case_id: "generic_xyz_111".into(),
                x: 1.0,
                y: 1.0,
                z: 1.0,
            },
            // 2. On +x axis only — ρ=1, sph polar=π/2, azim=0
            PointCase {
                case_id: "x_axis_only".into(),
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            // 3. On +z axis only — sph polar=0, ρ=0
            PointCase {
                case_id: "z_axis_only".into(),
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            // 4. Quadrant: +x, -y, +z
            PointCase {
                case_id: "mixed_quadrant".into(),
                x: 2.0,
                y: -3.0,
                z: 1.5,
            },
            // 5. Negative-octant
            PointCase {
                case_id: "neg_octant".into(),
                x: -1.5,
                y: -2.5,
                z: -0.75,
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

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

def cart_to_sph(x, y, z):
    # ISO 80000-2: θ polar from +z, φ azimuth from +x.
    r = math.sqrt(x*x + y*y + z*z)
    if r > 0:
        theta = math.acos(max(-1.0, min(1.0, z / r)))
    else:
        theta = 0.0
    phi = math.atan2(y, x)
    return r, theta, phi

def cart_to_cyl(x, y, z):
    rho = math.sqrt(x*x + y*y)
    theta = math.atan2(y, x)
    return rho, theta, z

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    x, y, z = case["x"], case["y"], case["z"]
    out = {"case_id": cid, "sph": None, "cyl": None}
    try:
        r, t, p = cart_to_sph(x, y, z)
        out["sph"] = [fnone(r), fnone(t), fnone(p)]
        rho, th, zz = cart_to_cyl(x, y, z)
        out["cyl"] = [fnone(rho), fnone(th), fnone(zz)]
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}, allow_nan=False))
"#;
    let query_json = serde_json::to_string(query).expect("serialize coord-transforms query");
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
                "failed to spawn python3 for coord-transforms oracle: {e}"
            );
            eprintln!("skipping coord-transforms oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open coord-transforms oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "coord-transforms oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping coord-transforms oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for coord-transforms oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "coord-transforms oracle failed: {stderr}"
        );
        eprintln!("skipping coord-transforms oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse coord-transforms oracle JSON"))
}

#[test]
fn diff_spatial_coord_transforms() {
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

        // Forward spherical
        if let Some(scipy_sph) = scipy_arm.sph {
            let (r, t, p) = cartesian_to_spherical(case.x, case.y, case.z);
            let abs_diff = [
                (r - scipy_sph[0]).abs(),
                (t - scipy_sph[1]).abs(),
                (p - scipy_sph[2]).abs(),
            ];
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "cartesian_to_spherical".into(),
                detail: format!(
                    "rust=({r}, {t}, {p}), numpy={scipy_sph:?}, max_abs={}",
                    abs_diff.iter().cloned().fold(0.0_f64, f64::max)
                ),
                pass: abs_diff.iter().all(|d| *d <= ABS_TOL),
            });

            // Round-trip: sph → cart, expect ≈ original (x, y, z).
            let (rx, ry, rz) = spherical_to_cartesian(r, t, p);
            let rt_diff = [
                (rx - case.x).abs(),
                (ry - case.y).abs(),
                (rz - case.z).abs(),
            ];
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "sph_round_trip_identity".into(),
                detail: format!(
                    "orig=({}, {}, {}), recovered=({rx}, {ry}, {rz}), max_abs={}",
                    case.x,
                    case.y,
                    case.z,
                    rt_diff.iter().cloned().fold(0.0_f64, f64::max)
                ),
                pass: rt_diff.iter().all(|d| *d <= ABS_TOL),
            });
        }

        // Forward cylindrical
        if let Some(scipy_cyl) = scipy_arm.cyl {
            let (rho, t, zz) = cartesian_to_cylindrical(case.x, case.y, case.z);
            let abs_diff = [
                (rho - scipy_cyl[0]).abs(),
                (t - scipy_cyl[1]).abs(),
                (zz - scipy_cyl[2]).abs(),
            ];
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "cartesian_to_cylindrical".into(),
                detail: format!(
                    "rust=({rho}, {t}, {zz}), numpy={scipy_cyl:?}, max_abs={}",
                    abs_diff.iter().cloned().fold(0.0_f64, f64::max)
                ),
                pass: abs_diff.iter().all(|d| *d <= ABS_TOL),
            });

            // Round-trip
            let (rx, ry, rz) = cylindrical_to_cartesian(rho, t, zz);
            let rt_diff = [
                (rx - case.x).abs(),
                (ry - case.y).abs(),
                (rz - case.z).abs(),
            ];
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "cyl_round_trip_identity".into(),
                detail: format!(
                    "orig=({}, {}, {}), recovered=({rx}, {ry}, {rz}), max_abs={}",
                    case.x,
                    case.y,
                    case.z,
                    rt_diff.iter().cloned().fold(0.0_f64, f64::max)
                ),
                pass: rt_diff.iter().all(|d| *d <= ABS_TOL),
            });
        }
    }

    let pass_count = cases.iter().filter(|c| c.pass).count();
    let all_pass = pass_count == cases.len();

    let log = DiffLog {
        test_id: "diff_spatial_coord_transforms".into(),
        category: "fsci_spatial coordinate transforms".into(),
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
                "coord-transforms mismatch: {} {} — {}",
                c.case_id, c.sub_check, c.detail
            );
        }
    }

    assert!(
        all_pass,
        "coord-transforms conformance failed: {} of {} cases pass",
        pass_count,
        cases.len()
    );
}
