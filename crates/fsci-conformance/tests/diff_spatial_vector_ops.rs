#![forbid(unsafe_code)]
//! Live numpy differential coverage for fsci_spatial vector
//! primitives.
//!
//! Resolves [frankenscipy-wckdw]. Covers:
//!   • dot(a, b)        ≡ numpy.dot(a, b)
//!   • angle_between(a, b) ≡ arccos(clamp(dot/(‖a‖·‖b‖), -1, 1))
//!   • cross_3d(a, b)   ≡ numpy.cross(a, b)   (3-D only)
//!   • normalize(v)     ≡ v / ‖v‖   (or v if ‖v‖=0)
//!
//! 5 fixtures (orthogonal, parallel, anti-parallel, generic 3-D,
//! 4-D for dot/angle/normalize) × per-fn applicability ≈ 16 cases.
//! Tol 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_spatial::{angle_between, cross_3d, dot, normalize};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-011";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    a: Vec<f64>,
    b: Vec<f64>,
    is_3d: bool,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    dot: Option<f64>,
    angle: Option<f64>,
    cross: Option<[f64; 3]>,
    norm_a: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create vector-ops diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize vector-ops diff log");
    fs::write(path, json).expect("write vector-ops diff log");
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        points: vec![
            // 3-D orthogonal: angle = π/2
            PointCase {
                case_id: "orthogonal_3d".into(),
                a: vec![1.0, 0.0, 0.0],
                b: vec![0.0, 1.0, 0.0],
                is_3d: true,
            },
            // 3-D parallel: angle = 0
            PointCase {
                case_id: "parallel_3d".into(),
                a: vec![1.0, 1.0, 1.0],
                b: vec![2.0, 2.0, 2.0],
                is_3d: true,
            },
            // 3-D anti-parallel: angle = π
            PointCase {
                case_id: "antiparallel_3d".into(),
                a: vec![1.0, 0.0, 0.0],
                b: vec![-1.0, 0.0, 0.0],
                is_3d: true,
            },
            // Generic 3-D
            PointCase {
                case_id: "generic_3d".into(),
                a: vec![1.0, 2.0, 3.0],
                b: vec![4.0, -1.0, 2.0],
                is_3d: true,
            },
            // 4-D (no cross_3d)
            PointCase {
                case_id: "generic_4d".into(),
                a: vec![1.0, 0.5, -0.25, 2.0],
                b: vec![-0.5, 1.0, 0.75, 0.25],
                is_3d: false,
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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    a = np.asarray(case["a"], dtype=np.float64)
    b = np.asarray(case["b"], dtype=np.float64)
    out = {"case_id": cid, "dot": None, "angle": None, "cross": None, "norm_a": None}
    try:
        out["dot"] = fnone(np.dot(a, b))
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na * nb > 0:
            cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
            out["angle"] = fnone(math.acos(cos))
        else:
            out["angle"] = 0.0
        if case["is_3d"]:
            c = np.cross(a, b)
            out["cross"] = [fnone(v) for v in c.tolist()]
        if na > 0:
            out["norm_a"] = [fnone(v) for v in (a / na).tolist()]
        else:
            out["norm_a"] = [fnone(v) for v in a.tolist()]
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}, allow_nan=False))
"#;
    let query_json = serde_json::to_string(query).expect("serialize vector-ops query");
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
                "failed to spawn python3 for vector-ops oracle: {e}"
            );
            eprintln!("skipping vector-ops oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open vector-ops oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "vector-ops oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping vector-ops oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for vector-ops oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "vector-ops oracle failed: {stderr}"
        );
        eprintln!("skipping vector-ops oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse vector-ops oracle JSON"))
}

#[test]
fn diff_spatial_vector_ops() {
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

        // dot
        if let Some(scipy_d) = scipy_arm.dot {
            let r = dot(&case.a, &case.b);
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "dot".into(),
                detail: format!("rust={r}, numpy={scipy_d}"),
                pass: (r - scipy_d).abs() <= ABS_TOL,
            });
        }

        // angle_between
        if let Some(scipy_ang) = scipy_arm.angle {
            let r = angle_between(&case.a, &case.b);
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "angle_between".into(),
                detail: format!("rust={r}, numpy={scipy_ang}"),
                pass: (r - scipy_ang).abs() <= ABS_TOL,
            });
        }

        // cross_3d (only when is_3d)
        if let Some(scipy_c) = scipy_arm.cross {
            let a3 = [case.a[0], case.a[1], case.a[2]];
            let b3 = [case.b[0], case.b[1], case.b[2]];
            let r = cross_3d(&a3, &b3);
            let abs_diff = [
                (r[0] - scipy_c[0]).abs(),
                (r[1] - scipy_c[1]).abs(),
                (r[2] - scipy_c[2]).abs(),
            ];
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "cross_3d".into(),
                detail: format!("rust={r:?}, numpy={scipy_c:?}"),
                pass: abs_diff.iter().all(|d| *d <= ABS_TOL),
            });
        }

        // normalize(a)
        if let Some(scipy_n) = scipy_arm.norm_a.as_ref() {
            let r = normalize(&case.a);
            let pass = r.len() == scipy_n.len()
                && r.iter()
                    .zip(scipy_n.iter())
                    .all(|(rv, sv)| (rv - sv).abs() <= ABS_TOL);
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                sub_check: "normalize".into(),
                detail: format!("rust={r:?}, numpy={scipy_n:?}"),
                pass,
            });
        }
    }

    let pass_count = cases.iter().filter(|c| c.pass).count();
    let all_pass = pass_count == cases.len();

    let log = DiffLog {
        test_id: "diff_spatial_vector_ops".into(),
        category: "fsci_spatial::{dot,angle_between,cross_3d,normalize}".into(),
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
                "vector-ops mismatch: {} {} — {}",
                c.case_id, c.sub_check, c.detail
            );
        }
    }

    assert!(
        all_pass,
        "vector-ops conformance failed: {} of {} cases pass",
        pass_count,
        cases.len()
    );
}
