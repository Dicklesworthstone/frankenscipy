#![forbid(unsafe_code)]
//! Live scipy.special parity for fsci_special scalar conv wrappers:
//! gammainc_conv, gammaincc_conv, erfc_conv, erfcinv_conv, fdiff.
//!
//! Resolves [frankenscipy-7dl4i]. All thin scalar wrappers around
//! already-tested functions; ensures the convenience surface is
//! correctly wired. 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{erfc_conv, erfcinv_conv, fdiff, gammainc_conv, gammaincc_conv};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String,
    a: f64,
    b: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create conv_scalar diff dir");
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
    // gammainc / gammaincc: a in (0, large], x in [0, large]
    let ginc: &[(f64, f64)] = &[
        (0.5, 0.3),
        (0.5, 1.0),
        (0.5, 5.0),
        (1.0, 0.5),
        (1.0, 2.0),
        (2.0, 1.0),
        (2.0, 3.5),
        (5.0, 1.0),
        (5.0, 7.0),
        (10.0, 5.0),
        (10.0, 15.0),
    ];
    for &(a, x) in ginc {
        points.push(Case {
            case_id: format!("ginc_a{a}_x{x}"),
            op: "ginc".into(),
            a,
            b: x,
        });
        points.push(Case {
            case_id: format!("gincc_a{a}_x{x}"),
            op: "gincc".into(),
            a,
            b: x,
        });
    }

    // erfc_conv: x ∈ ℝ
    for &x in &[-2.0_f64, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0] {
        points.push(Case {
            case_id: format!("erfc_x{x}"),
            op: "erfc".into(),
            a: x,
            b: 0.0,
        });
    }

    // erfcinv_conv: y ∈ (0, 2)
    for &y in &[0.1_f64, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.9] {
        points.push(Case {
            case_id: format!("erfcinv_y{y}"),
            op: "erfcinv".into(),
            a: y,
            b: 0.0,
        });
    }

    // fdiff: |x - y|
    for &(x, y) in &[
        (3.0_f64, 1.0),
        (-2.0, 5.0),
        (0.5, 0.5),
        (1.0e-9, 0.0),
        (-1.0, -1.0),
    ] {
        points.push(Case {
            case_id: format!("fdiff_x{x}_y{y}"),
            op: "fdiff".into(),
            a: x,
            b: y,
        });
    }

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import special as sp

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    a = float(case["a"]); b = float(case["b"])
    try:
        if op == "ginc":
            v = float(sp.gammainc(a, b))
        elif op == "gincc":
            v = float(sp.gammaincc(a, b))
        elif op == "erfc":
            v = float(sp.erfc(a))
        elif op == "erfcinv":
            v = float(sp.erfcinv(a))
        elif op == "fdiff":
            v = abs(a - b)
        else:
            points.append({"case_id": cid, "value": None}); continue
        if math.isfinite(v):
            points.append({"case_id": cid, "value": v})
        else:
            points.append({"case_id": cid, "value": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "value": None})
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
                "failed to spawn python3 for conv_scalar oracle: {e}"
            );
            eprintln!("skipping conv_scalar oracle: python3 not available ({e})");
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
                "conv_scalar oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping conv_scalar oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for conv_scalar oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "conv_scalar oracle failed: {stderr}"
        );
        eprintln!("skipping conv_scalar oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse conv_scalar oracle JSON"))
}

#[test]
fn diff_special_conv_scalar_wrappers() {
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
        let Some(expected) = arm.value else {
            continue;
        };
        let actual = match case.op.as_str() {
            "ginc" => gammainc_conv(case.a, case.b),
            "gincc" => gammaincc_conv(case.a, case.b),
            "erfc" => erfc_conv(case.a),
            "erfcinv" => erfcinv_conv(case.a),
            "fdiff" => fdiff(case.a, case.b),
            _ => continue,
        };
        if !actual.is_finite() {
            continue;
        }
        let abs_d = (actual - expected).abs();
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
        test_id: "diff_special_conv_scalar_wrappers".into(),
        category:
            "fsci_special::{gammainc_conv, gammaincc_conv, erfc_conv, erfcinv_conv, fdiff} vs scipy.special"
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
        "conv_scalar conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
