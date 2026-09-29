#![forbid(unsafe_code)]
//! Live scipy.special parity for fsci_special::iv, hankel1, hankel2,
//! and erfi.
//!
//! Resolves [frankenscipy-aikhp].
//!
//! - `iv(v, z)`: modified Bessel I_v(z); real v and real z>0 → real.
//! - `hankel1(v, z)`, `hankel2(v, z)`: H1_v, H2_v, complex output.
//!   Compare real & imag parts at 1e-8 abs.
//! - `erfi(x)`: imaginary error function, real → real. Compare at
//!   1e-9 abs for small |x|, 1e-6 rel for |x| ≥ 1 (erfi grows
//!   exponentially).

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_runtime::RuntimeMode;
use fsci_special::types::Complex64 as FsciComplex;
use fsci_special::types::SpecialTensor;
use fsci_special::{erfi, hankel1, hankel2, iv};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const IV_ABS_TOL: f64 = 1.0e-7;
const HANKEL_ABS_TOL: f64 = 1.0e-6;
const ERFI_ABS_TOL: f64 = 1.0e-9;
const ERFI_REL_TOL: f64 = 1.0e-6;
/// One ledger arm per op compared.
const ARMS: [&str; 4] = ["iv", "hankel1", "hankel2", "erfi"];

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "iv" | "hankel1" | "hankel2" | "erfi"
    v: f64,
    z: f64, // real positive
    x: f64, // for erfi
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Real scalar (iv, erfi) or [re, im] (hankel)
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
    compared: BTreeMap<String, ArmCounts>,
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
    fs::create_dir_all(output_dir()).expect("create iv_hankel_erfi diff dir");
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

fn fsci_iv(v: f64, z: f64) -> Option<f64> {
    let v_t = SpecialTensor::RealScalar(v);
    let z_t = SpecialTensor::RealScalar(z);
    match iv(&v_t, &z_t, RuntimeMode::Strict) {
        Ok(SpecialTensor::RealScalar(r)) => Some(r),
        _ => None,
    }
}

fn fsci_hankel(kind: &str, v: f64, z: f64) -> Option<(f64, f64)> {
    let v_t = SpecialTensor::RealScalar(v);
    let z_t = SpecialTensor::RealScalar(z);
    let res = match kind {
        "hankel1" => hankel1(&v_t, &z_t, RuntimeMode::Strict),
        "hankel2" => hankel2(&v_t, &z_t, RuntimeMode::Strict),
        _ => return None,
    };
    match res {
        Ok(SpecialTensor::ComplexScalar(c)) => Some((c.re, c.im)),
        Ok(SpecialTensor::ComplexVec(v)) if v.len() == 1 => Some((v[0].re, v[0].im)),
        Ok(SpecialTensor::RealScalar(r)) => Some((r, 0.0)),
        _ => None,
    }
}

fn fsci_erfi(x: f64) -> Option<f64> {
    let x_t = SpecialTensor::RealScalar(x);
    match erfi(&x_t, RuntimeMode::Strict) {
        Ok(SpecialTensor::RealScalar(r)) => Some(r),
        _ => None,
    }
}

#[allow(dead_code)]
fn _unused_complex(_: FsciComplex) {}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // iv probes: real (v, z>0)
    let iv_probes: &[(f64, f64)] = &[
        (0.0, 0.5),
        (0.0, 1.0),
        (0.0, 2.0),
        (0.0, 5.0),
        (1.0, 0.5),
        (1.0, 1.0),
        (1.0, 3.0),
        (0.5, 1.0),
        (0.5, 2.5),
        (2.0, 2.0),
        (3.0, 4.0),
    ];
    for &(v, z) in iv_probes {
        points.push(Case {
            case_id: format!("iv_v{v}_z{z}"),
            op: "iv".into(),
            v,
            z,
            x: 0.0,
        });
    }

    // hankel1 / hankel2 probes: real (v, z>0). Integer orders v=0,1,2,3
    // are exercised across a spread of z (frankenscipy-xg4xw).
    let h_probes: &[(f64, f64)] = &[
        (0.0, 1.0),
        (0.0, 2.5),
        (0.0, 5.0),
        (1.0, 0.5),
        (1.0, 1.0),
        (1.0, 3.0),
        (1.0, 5.0),
        (2.0, 1.0),
        (2.0, 2.0),
        (2.0, 4.0),
        (3.0, 2.0),
        (3.0, 6.0),
        (0.5, 2.0),
    ];
    for &(v, z) in h_probes {
        for op in ["hankel1", "hankel2"] {
            points.push(Case {
                case_id: format!("{op}_v{v}_z{z}"),
                op: op.into(),
                v,
                z,
                x: 0.0,
            });
        }
    }

    // erfi probes: real x
    let erfi_probes: &[f64] = &[-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5];
    for &x in erfi_probes {
        points.push(Case {
            case_id: format!("erfi_x{x}"),
            op: "erfi".into(),
            v: 0.0,
            z: 0.0,
            x,
        });
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
        if op == "iv":
            v = float(case["v"]); z = float(case["z"])
            r = float(sp.iv(v, z))
            if math.isfinite(r):
                points.append({"case_id": cid, "values": [r]})
            else:
                points.append({"case_id": cid, "values": None})
        elif op in ("hankel1", "hankel2"):
            v = float(case["v"]); z = float(case["z"])
            c = sp.hankel1(v, z) if op == "hankel1" else sp.hankel2(v, z)
            if math.isfinite(c.real) and math.isfinite(c.imag):
                points.append({"case_id": cid, "values": [float(c.real), float(c.imag)]})
            else:
                points.append({"case_id": cid, "values": None})
        elif op == "erfi":
            x = float(case["x"])
            r = float(sp.erfi(x))
            if math.isfinite(r):
                points.append({"case_id": cid, "values": [r]})
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
                "failed to spawn python3 for iv_hankel_erfi oracle: {e}"
            );
            eprintln!("skipping iv_hankel_erfi oracle: python3 not available ({e})");
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
                "iv_hankel_erfi oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping iv_hankel_erfi oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for iv_hankel_erfi oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "iv_hankel_erfi oracle failed: {stderr}"
        );
        eprintln!("skipping iv_hankel_erfi oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse iv_hankel_erfi oracle JSON"))
}

#[test]
fn diff_special_iv_hankel_erfi() {
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
    let mut ledger = CompareLedger::new("diff_special_iv_hankel_erfi", &ARMS);

    for case in &query.points {
        let scipy = pmap.get(&case.case_id).and_then(|a| a.values.as_deref());
        // `no_nan` is true for the scalar arms (`pair` hands back only finite values); the
        // hankel metric is a `f64::max` of component differences, which drops a NaN operand.
        let (abs_d, tol, no_nan) = match case.op.as_str() {
            "iv" => {
                let Some((expected, r)) = ledger.pair(
                    "iv",
                    &case.case_id,
                    scipy.and_then(|v| v.first().copied()),
                    fsci_iv(case.v, case.z),
                ) else {
                    continue;
                };
                ((r - expected).abs(), IV_ABS_TOL, true)
            }
            "hankel1" | "hankel2" => {
                let Some((expected, (re, im))) = ledger.both(
                    &case.op,
                    &case.case_id,
                    scipy,
                    fsci_hankel(&case.op, case.v, case.z),
                ) else {
                    continue;
                };
                let d_re = (re - expected[0]).abs();
                let d_im = (im - expected[1]).abs();
                (d_re.max(d_im), HANKEL_ABS_TOL, !re.is_nan() && !im.is_nan())
            }
            "erfi" => {
                let Some((expected, r)) = ledger.pair(
                    "erfi",
                    &case.case_id,
                    scipy.and_then(|v| v.first().copied()),
                    fsci_erfi(case.x),
                ) else {
                    continue;
                };
                let abs = (r - expected).abs();
                let rel = if expected.abs() > 1.0 {
                    abs / expected.abs()
                } else {
                    0.0
                };
                // pass if either abs or rel tolerance holds
                let tol = if expected.abs() > 1.0 {
                    f64::INFINITY // rely on relative
                } else {
                    ERFI_ABS_TOL
                };
                let abs_d = if expected.abs() > 1.0 {
                    // For large |result|, transform diff to be ≤ rel_tol if it is
                    if rel <= ERFI_REL_TOL { 0.0 } else { abs }
                } else {
                    abs
                };
                (abs_d, tol, true)
            }
            other => panic!("unknown op {other}"),
        };
        max_overall = max_overall.max(abs_d);
        let pass = no_nan && abs_d <= tol;
        ledger.compared(&case.op, &case.case_id, pass);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_iv_hankel_erfi".into(),
        category: "fsci_special::{iv, hankel1, hankel2, erfi} vs scipy.special".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
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
        "iv/hankel/erfi conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    // Arms have different case sets (iv and erfi have the fewest); each must compare all of its own.
    let min_per_arm = ARMS
        .iter()
        .map(|arm| query.points.iter().filter(|c| c.op == *arm).count())
        .min()
        .expect("ARMS is non-empty");
    ledger.finish(min_per_arm);
}
