#![forbid(unsafe_code)]
//! Live numpy differential coverage for `numpy.polyder`, `polyint`, and
//! `numpy.poly` (the HIGH-FIRST equivalent of polyfromroots).
//!
//! Resolves [frankenscipy-nagt4]. fsci uses HIGH-FIRST coefficient order
//! throughout (constant last), so:
//! - polyder ↔ np.polyder
//! - polyint ↔ np.polyint
//! - polyfromroots ↔ np.poly  (np.polynomial.polynomial.polyfromroots is
//!   the LOW-FIRST convention, which fsci does not match).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_interpolate::{polyder, polyfromroots, polyint};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    /// For polyder/polyint: input coefficients (high-first).
    /// For polyfromroots: input roots.
    input: Vec<f64>,
    /// For polyder/polyint: m (number of differentiations/integrations).
    m: usize,
    /// For polyint: integration constant.
    k: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create poly_helpers diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize poly_helpers diff log");
    fs::write(path, json).expect("write poly_helpers diff log");
}

fn generate_query() -> OracleQuery {
    let polyder_cases: &[(&str, Vec<f64>, usize)] = &[
        ("der_linear_m1", vec![3.0, 5.0], 1),
        ("der_quad_m1", vec![1.0, 2.0, 3.0], 1),
        ("der_cubic_m1", vec![2.0, -3.0, 4.0, -1.0], 1),
        ("der_quartic_m2", vec![1.0, 0.0, -2.0, 0.0, 1.0], 2),
        ("der_quintic_m3", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3),
    ];
    let polyint_cases: &[(&str, Vec<f64>, usize, f64)] = &[
        ("int_linear_m1_k0", vec![2.0, 3.0], 1, 0.0),
        ("int_quad_m1_k5", vec![1.0, 2.0, 3.0], 1, 5.0),
        ("int_cubic_m2_k0", vec![2.0, -3.0, 4.0, -1.0], 2, 0.0),
        (
            "int_quartic_m1_kneg2",
            vec![1.0, 0.0, -2.0, 0.0, 1.0],
            1,
            -2.0,
        ),
    ];
    let polyfromroots_cases: &[(&str, Vec<f64>)] = &[
        ("roots_two", vec![1.0, -1.0]),
        ("roots_three", vec![1.0, 2.0, 3.0]),
        ("roots_double", vec![2.0, 2.0]),
        ("roots_mixed", vec![0.0, 1.5, -0.5, 2.0]),
    ];

    let mut points = Vec::new();
    for (name, coeffs, m) in polyder_cases {
        points.push(PointCase {
            case_id: (*name).into(),
            op: "polyder".into(),
            input: coeffs.clone(),
            m: *m,
            k: 0.0,
        });
    }
    for (name, coeffs, m, k) in polyint_cases {
        points.push(PointCase {
            case_id: (*name).into(),
            op: "polyint".into(),
            input: coeffs.clone(),
            m: *m,
            k: *k,
        });
    }
    for (name, roots) in polyfromroots_cases {
        points.push(PointCase {
            case_id: (*name).into(),
            op: "polyfromroots".into(),
            input: roots.clone(),
            m: 0,
            k: 0.0,
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

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).flatten().tolist():
        try:
            v = float(v)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    inp = np.array(case["input"], dtype=float)
    m = int(case["m"]); k = float(case["k"])
    try:
        if op == "polyder":
            v = np.polyder(inp, m=m)
        elif op == "polyint":
            v = np.polyint(inp, m=m, k=k)
        elif op == "polyfromroots":
            # fsci's polyfromroots matches np.poly (HIGH-FIRST), not
            # np.polynomial.polynomial.polyfromroots (LOW-FIRST).
            v = np.poly(inp)
        else:
            v = None
        points.append({"case_id": cid, "values": finite_vec_or_none(v) if v is not None else None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize poly_helpers query");
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
                "failed to spawn python3 for poly_helpers oracle: {e}"
            );
            eprintln!("skipping poly_helpers oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open poly_helpers oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "poly_helpers oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping poly_helpers oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for poly_helpers oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "poly_helpers oracle failed: {stderr}"
        );
        eprintln!("skipping poly_helpers oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse poly_helpers oracle JSON"))
}

#[test]
fn diff_interpolate_poly_helpers() {
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
        let Some(expected) = scipy_arm.values.as_ref() else {
            continue;
        };
        let fsci_v: Vec<f64> = match case.op.as_str() {
            "polyder" => polyder(&case.input, case.m),
            "polyint" => polyint(&case.input, case.m, case.k),
            "polyfromroots" => polyfromroots(&case.input),
            _ => continue,
        };
        if fsci_v.len() != expected.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                op: case.op.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = fsci_v
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
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
        test_id: "diff_interpolate_poly_helpers".into(),
        category: "numpy.polyder + polyint + poly (HIGH-FIRST)".into(),
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
        "numpy.polyder/polyint/poly conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
