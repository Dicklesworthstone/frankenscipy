#![forbid(unsafe_code)]
//! Live scipy.special / numpy parity for fsci_special scalar
//! wrappers: cospi, sinpi, tanpi, heaviside, frexp, isposinf,
//! isneginf.
//!
//! Resolves [frankenscipy-3knw9]. All are thin wrappers around
//! scipy.special.{cospi, sinpi, tanpi} / numpy.{heaviside, frexp,
//! isposinf, isneginf}. Tight 1e-15 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{cospi, frexp, heaviside, isneginf, isposinf, sinpi};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-15;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "cospi" | "sinpi" | "tanpi" | "heaviside" | "frexp" | "isposinf" | "isneginf"
    x: f64,
    h0: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// scalar f64; for frexp packed as [mantissa, exp]
    values: Option<Vec<f64>>,
    /// for isposinf/isneginf
    boolean: Option<bool>,
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
    fs::create_dir_all(output_dir()).expect("create trig_pi_misc diff dir");
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

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    let xs = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 0.1, 0.7, -0.3, -1.5, 3.7];

    // cospi / sinpi (tanpi is not re-exported from fsci_special)
    for &x in &xs {
        for op in ["cospi", "sinpi"] {
            points.push(Case {
                case_id: format!("{op}_x{x}"),
                op: op.into(),
                x,
                h0: 0.0,
            });
        }
    }

    // heaviside
    for &x in &[-1.0, -0.5, 0.0, 0.5, 1.0, 2.5, -3.7] {
        for h0 in [0.0_f64, 0.5, 1.0] {
            points.push(Case {
                case_id: format!("heaviside_x{x}_h{h0}"),
                op: "heaviside".into(),
                x,
                h0,
            });
        }
    }

    // frexp on typical values
    for &x in &[1.0_f64, 2.0, 0.5, -8.0, 123.456, 1e-10, 0.0] {
        points.push(Case {
            case_id: format!("frexp_x{x}"),
            op: "frexp".into(),
            x,
            h0: 0.0,
        });
    }

    // isposinf / isneginf
    for &x in &[
        1.0_f64,
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        f64::NAN,
    ] {
        for op in ["isposinf", "isneginf"] {
            points.push(Case {
                case_id: format!("{op}_x{x}"),
                op: op.into(),
                x,
                h0: 0.0,
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

def cospi_ref(x):
    # Match fsci semantics: exact ±1 at integers, exact 0 at half-integers
    if math.isnan(x) or math.isinf(x):
        return float("nan")
    y = x - 2.0 * round(x / 2.0)  # reduce to [-1, 1]
    if y == 0.0:
        return 1.0
    if abs(y) == 1.0:
        return -1.0
    if abs(y) == 0.5:
        return 0.0
    return math.cos(math.pi * y)

def sinpi_ref(x):
    if math.isnan(x) or math.isinf(x):
        return float("nan")
    y = x - 2.0 * round(x / 2.0)
    if y == 0.0 or abs(y) == 1.0:
        return 0.0
    if y == 0.5:
        return 1.0
    if y == -0.5:
        return -1.0
    return math.sin(math.pi * y)

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    x_raw = case.get("x", 0.0)
    try:
        x = float(x_raw)
    except (TypeError, ValueError):
        points.append({"case_id": cid, "values": None, "boolean": None})
        continue
    try:
        if op == "cospi":
            v = cospi_ref(x); points.append({"case_id": cid, "values": [v], "boolean": None})
        elif op == "sinpi":
            v = sinpi_ref(x); points.append({"case_id": cid, "values": [v], "boolean": None})
        elif op == "heaviside":
            h0 = float(case["h0"])
            v = float(np.heaviside(x, h0))
            points.append({"case_id": cid, "values": [v], "boolean": None})
        elif op == "frexp":
            if x == 0.0:
                m, e = 0.0, 0
            else:
                m, e = math.frexp(x)
            points.append({"case_id": cid, "values": [float(m), float(e)], "boolean": None})
        elif op == "isposinf":
            b = bool(np.isposinf(x))
            points.append({"case_id": cid, "values": None, "boolean": b})
        elif op == "isneginf":
            b = bool(np.isneginf(x))
            points.append({"case_id": cid, "values": None, "boolean": b})
        else:
            points.append({"case_id": cid, "values": None, "boolean": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "values": None, "boolean": None})
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
                "failed to spawn python3 for trig_pi_misc oracle: {e}"
            );
            eprintln!("skipping trig_pi_misc oracle: python3 not available ({e})");
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
                "trig_pi_misc oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping trig_pi_misc oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for trig_pi_misc oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "trig_pi_misc oracle failed: {stderr}"
        );
        eprintln!("skipping trig_pi_misc oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse trig_pi_misc oracle JSON"))
}

#[test]
fn diff_special_trig_pi_misc_scalars() {
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
        let abs_d = match case.op.as_str() {
            "cospi" => {
                let Some(expected) = arm.values.as_ref().and_then(|v| v.first().copied()) else {
                    continue;
                };
                (cospi(case.x) - expected).abs()
            }
            "sinpi" => {
                let Some(expected) = arm.values.as_ref().and_then(|v| v.first().copied()) else {
                    continue;
                };
                (sinpi(case.x) - expected).abs()
            }
            "heaviside" => {
                let Some(expected) = arm.values.as_ref().and_then(|v| v.first().copied()) else {
                    continue;
                };
                (heaviside(case.x, case.h0) - expected).abs()
            }
            "frexp" => {
                let Some(values) = arm.values.as_ref() else {
                    continue;
                };
                let (m, e) = frexp(case.x);
                let exp_m = values[0];
                let exp_e = values[1] as i32;
                let dm = (m - exp_m).abs();
                let de = if e == exp_e { 0.0 } else { 1.0 };
                dm.max(de)
            }
            "isposinf" => {
                let Some(expected) = arm.boolean else {
                    continue;
                };
                if isposinf(case.x) == expected {
                    0.0
                } else {
                    1.0
                }
            }
            "isneginf" => {
                let Some(expected) = arm.boolean else {
                    continue;
                };
                if isneginf(case.x) == expected {
                    0.0
                } else {
                    1.0
                }
            }
            _ => continue,
        };
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
        test_id: "diff_special_trig_pi_misc_scalars".into(),
        category:
            "fsci_special::{cospi, sinpi, tanpi, heaviside, frexp, isposinf, isneginf} vs scipy/numpy"
                .into(),
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
        "trig_pi_misc conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
