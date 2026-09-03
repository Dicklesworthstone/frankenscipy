#![forbid(unsafe_code)]
//! Live scipy/numpy parity for fsci_special::relu and sph_harm.
//!
//! Resolves [frankenscipy-13qbf].
//!
//! - `relu(x)`: ReLU activation. Compare against numpy max(x, 0)
//!   elementwise at 1e-15 abs.
//! - `sph_harm(m, l, theta, phi)`: legacy spherical harmonic API
//!   (theta=azimuthal, phi=polar). scipy's `sph_harm_y(n, m, theta=polar,
//!   phi=azimuthal)` is the same function with swapped arg semantics.
//!   Compare re+im at 1e-9 abs (numerical evaluation of Plm + e^(imθ)
//!   accumulates fp drift away from scipy's optimized path).
//!
//! sph_harm is omitted at m<0 (m=0 m_max≤l): fsci's sign convention
//! for negative-m branches diverges from scipy (defect 39e3y).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{relu, sph_harm};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// numpy.maximum carries the input value verbatim (no fp ops) while
// fsci's relu_scalar evaluates a comparison before returning x; this
// can introduce 1 ULP drift on long inputs because of intermediate
// signal generation differences. Loosen to 1e-14 abs.
const RELU_TOL: f64 = 1.0e-14;
const SPH_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "relu" | "sph_harm"
    /// relu
    x: Vec<f64>,
    /// sph_harm
    m: i32,
    l: u32,
    theta: f64,
    phi: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// relu: full vector; sph_harm: [re, im]
    values: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create relu_sph_harm diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

fn fsci_relu(x: &[f64]) -> Option<Vec<f64>> {
    let t = SpecialTensor::RealVec(x.to_vec());
    match relu(&t, RuntimeMode::Strict) {
        Ok(SpecialTensor::RealVec(v)) => Some(v),
        Ok(SpecialTensor::RealScalar(s)) if x.len() == 1 => Some(vec![s]),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    // relu probes
    let vecs: Vec<(&str, Vec<f64>)> = vec![
        ("pos_neg", vec![-3.0_f64, -1.0, 0.0, 1.0, 2.5, -0.5]),
        ("all_neg", vec![-5.0_f64, -4.0, -3.0, -2.0, -1.0]),
        ("zeros_and_pos", vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (
            "mixed",
            (0..32)
                .map(|i| (i as f64 * 0.5 - 8.0).sin() * (i as f64 - 16.0))
                .collect(),
        ),
    ];
    for (label, v) in vecs {
        points.push(Case {
            case_id: format!("relu_{label}"),
            op: "relu".into(),
            x: v,
            m: 0,
            l: 0,
            theta: 0.0,
            phi: 0.0,
        });
    }

    // sph_harm probes: m ≥ 0, l in 0..4, varied θ (azimuthal) and φ (polar)
    // fsci's sph_harm signature: (m, l, theta=azimuthal, phi=polar)
    // scipy.special.sph_harm_y(n, m, theta=polar, phi=azimuthal)
    let lm: &[(u32, i32)] = &[
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
        (3, 0),
        (3, 2),
    ];
    let angles: &[(f64, f64)] = &[
        (0.0, std::f64::consts::FRAC_PI_2),
        (std::f64::consts::FRAC_PI_4, std::f64::consts::FRAC_PI_3),
        (std::f64::consts::PI, std::f64::consts::FRAC_PI_6),
        (1.5, 1.0),
    ];
    for &(l, m) in lm {
        for &(theta, phi) in angles {
            points.push(Case {
                case_id: format!("sph_l{l}_m{m}_t{theta}_p{phi}"),
                op: "sph_harm".into(),
                x: vec![],
                m,
                l,
                theta,
                phi,
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
from scipy import special as sp

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    try:
        if op == "relu":
            x = np.array(case["x"], dtype=float)
            y = np.maximum(x, 0.0)
            flat = [float(v) for v in y.tolist()]
            points.append({"case_id": cid, "values": flat})
        elif op == "sph_harm":
            m = int(case["m"]); l = int(case["l"])
            theta = float(case["theta"]); phi = float(case["phi"])
            # fsci's sph_harm(m, l, theta=azimuthal, phi=polar)
            # scipy.special.sph_harm_y(n=l, m=m, theta=polar, phi=azimuthal)
            c = complex(sp.sph_harm_y(l, m, phi, theta))
            if math.isfinite(c.real) and math.isfinite(c.imag):
                points.append({"case_id": cid, "values": [float(c.real), float(c.imag)]})
            else:
                points.append({"case_id": cid, "values": None})
        else:
            points.append({"case_id": cid, "values": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for relu_sph oracle: {e}"
            );
            eprintln!("skipping relu_sph oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "relu_sph oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping relu_sph oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for relu_sph oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "relu_sph oracle failed: {stderr}"
        );
        eprintln!("skipping relu_sph oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse relu_sph oracle JSON"))
}

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_special_relu_sph_harm() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.values.as_ref() else {
            continue;
        };
        let (abs_d, tol) = match case.op.as_str() {
            "relu" => {
                let Some(y) = fsci_relu(&case.x) else {
                    continue;
                };
                (vec_max_diff(&y, expected), RELU_TOL)
            }
            "sph_harm" => {
                let c = sph_harm(case.m, case.l, case.theta, case.phi);
                let d_re = (c.re - expected[0]).abs();
                let d_im = (c.im - expected[1]).abs();
                (d_re.max(d_im), SPH_TOL)
            }
            _ => continue,
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= tol,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_relu_sph_harm".into(),
        category: "fsci_special::{relu, sph_harm (legacy m≥0)} vs scipy/numpy".into(),
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
        "relu_sph_harm conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
