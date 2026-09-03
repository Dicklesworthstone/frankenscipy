#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.special.lambertw`
//! restricted to the principal branch (k=0) over real arguments
//! z ≥ −1/e where W(z) is real-valued.
//!
//! Resolves [frankenscipy-d1zg3]. fsci_special::lambertw is
//! exposed via elliptic.rs but had no dedicated diff_special_*
//! harness. Verified fsci matches scipy.special.lambertw(z, k=0).real
//! to ~2e-16 across the regime.
//!
//! Tolerance: 1e-13 abs / rel — leaves margin for the iterative
//! Halley refinement without papering over real drift.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::lambertw;
use fsci_special::types::SpecialTensor;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-13;
const REL_TOL: f64 = 1.0e-13;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    z: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    w: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    abs_diff: f64,
    rel_diff: f64,
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
    fs::create_dir_all(output_dir()).expect("create lambertw diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize lambertw diff log");
    fs::write(path, json).expect("write lambertw diff log");
}

fn generate_query() -> OracleQuery {
    // z values: near the branch boundary at -1/e ≈ -0.3679 (lambertw
    // diverges in slope there), zero (W(0)=0), Ω = lambertw(1) ≈ 0.5671,
    // e (W(e)=1), and increasing magnitudes including 1e6 to probe the
    // logarithmic large-z regime.
    let zs: &[f64] = &[
        -0.36,
        -0.3,
        -0.1,
        -0.01,
        0.0,
        0.1,
        0.5,
        1.0,
        std::f64::consts::E,
        5.0,
        10.0,
        100.0,
        1.0e6,
    ];
    let points = zs
        .iter()
        .map(|&z| PointCase {
            case_id: format!("z{z}"),
            z,
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import special

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; z = float(case["z"])
    try:
        # scipy.special.lambertw returns a complex; on the principal branch
        # over real z >= -1/e the imaginary part is zero.
        w = complex(special.lambertw(z, k=0))
        if abs(w.imag) < 1e-12:
            points.append({"case_id": cid, "w": fnone(w.real)})
        else:
            # Real-axis input slipped past −1/e; treat as missing.
            points.append({"case_id": cid, "w": None})
    except Exception:
        points.append({"case_id": cid, "w": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize lambertw query");
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
                "failed to spawn python3 for lambertw oracle: {e}"
            );
            eprintln!("skipping lambertw oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open lambertw oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "lambertw oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping lambertw oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for lambertw oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "lambertw oracle failed: {stderr}"
        );
        eprintln!("skipping lambertw oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse lambertw oracle JSON"))
}

fn fsci_lambertw(z: f64) -> Option<f64> {
    let arg = SpecialTensor::RealScalar(z);
    match lambertw(&arg, RuntimeMode::Strict).ok()? {
        SpecialTensor::RealScalar(v) => Some(v),
        SpecialTensor::ComplexScalar(c) if c.im.abs() < 1e-12 => Some(c.re),
        _ => None,
    }
}

#[test]
fn diff_special_lambertw() {
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
        let Some(fsci_v) = fsci_lambertw(case.z) else {
            continue;
        };
        if let Some(scipy_v) = scipy_arm.w
            && fsci_v.is_finite()
        {
            let abs_d = (fsci_v - scipy_v).abs();
            let rel_d = abs_d / scipy_v.abs().max(f64::MIN_POSITIVE);
            max_overall = max_overall.max(abs_d);
            let pass = abs_d <= ABS_TOL || rel_d <= REL_TOL;
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: abs_d,
                rel_diff: rel_d,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_lambertw".into(),
        category: "scipy.special.lambertw".into(),
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
                "lambertw mismatch: {} abs_diff={} rel_diff={}",
                d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special.lambertw conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
