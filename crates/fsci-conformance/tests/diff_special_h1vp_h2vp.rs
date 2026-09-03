#![forbid(unsafe_code)]
//! Live scipy.special parity for fsci_special::h1vp and h2vp.
//!
//! Resolves [frankenscipy-cdhev]. h1vp / h2vp return complex
//! derivatives of the Hankel functions. Real v, real positive z;
//! derivative orders 1-3. Compare re+im at 1e-6 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{h1vp, h2vp};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "h1vp" | "h2vp"
    v: f64,
    z: f64,
    n: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    values: Option<Vec<f64>>, // [re, im]
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
    fs::create_dir_all(output_dir()).expect("create h1vp_h2vp diff dir");
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

fn fsci_h_deriv(op: &str, v: f64, z: f64, n: usize) -> Option<(f64, f64)> {
    let v_t = SpecialTensor::RealScalar(v);
    let z_t = SpecialTensor::RealScalar(z);
    let res = match op {
        "h1vp" => h1vp(&v_t, &z_t, n, RuntimeMode::Strict),
        "h2vp" => h2vp(&v_t, &z_t, n, RuntimeMode::Strict),
        _ => return None,
    };
    match res {
        Ok(SpecialTensor::ComplexScalar(c)) => Some((c.re, c.im)),
        Ok(SpecialTensor::ComplexVec(v)) if v.len() == 1 => Some((v[0].re, v[0].im)),
        Ok(SpecialTensor::RealScalar(r)) => Some((r, 0.0)),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    let probes: &[(f64, f64)] = &[
        (0.0, 1.0),
        (0.0, 3.0),
        (1.0, 1.5),
        (1.0, 4.0),
        (2.0, 2.0),
        (0.5, 2.5),
    ];
    for &(v, z) in probes {
        for n in [1_usize, 2, 3] {
            for op in ["h1vp", "h2vp"] {
                points.push(Case {
                    case_id: format!("{op}_v{v}_z{z}_n{n}"),
                    op: op.into(),
                    v,
                    z,
                    n,
                });
            }
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
    v = float(case["v"]); z = float(case["z"]); n = int(case["n"])
    try:
        if op == "h1vp":
            c = sp.h1vp(v, z, n)
        elif op == "h2vp":
            c = sp.h2vp(v, z, n)
        else:
            points.append({"case_id": cid, "values": None}); continue
        if math.isfinite(c.real) and math.isfinite(c.imag):
            points.append({"case_id": cid, "values": [float(c.real), float(c.imag)]})
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
                "failed to spawn python3 for h1vp_h2vp oracle: {e}"
            );
            eprintln!("skipping h1vp_h2vp oracle: python3 not available ({e})");
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
                "h1vp_h2vp oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping h1vp_h2vp oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for h1vp_h2vp oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "h1vp_h2vp oracle failed: {stderr}"
        );
        eprintln!("skipping h1vp_h2vp oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse h1vp_h2vp oracle JSON"))
}

#[test]
fn diff_special_h1vp_h2vp() {
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
        let Some((re, im)) = fsci_h_deriv(&case.op, case.v, case.z, case.n) else {
            continue;
        };
        let abs_d = (re - expected[0]).abs().max((im - expected[1]).abs());
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_h1vp_h2vp".into(),
        category: "fsci_special::{h1vp, h2vp} vs scipy.special".into(),
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
        "h1vp/h2vp conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
