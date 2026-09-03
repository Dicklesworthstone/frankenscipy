#![forbid(unsafe_code)]
//! Live numpy/scipy differential coverage for fsci_special utility
//! scalar functions: kl_div, sinc_squared, log_comb, copysign,
//! nextafter, spacing, rint, fix, divmod, radian.
//!
//! Resolves [frankenscipy-hn2dz]. 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{
    copysign, divmod, fix, kl_div, log_comb, nextafter, radian, rint, sinc_squared, spacing,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    /// Primary arg (or first of pair / first of triple).
    a: f64,
    /// Secondary arg (used by kl_div, copysign, nextafter, log_comb, divmod, radian).
    b: f64,
    /// Tertiary arg (radian only).
    c: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
    /// For divmod: remainder (the second part of the tuple).
    remainder: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create utility_scalars diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize utility_scalars diff log");
    fs::write(path, json).expect("write utility_scalars diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // kl_div(x, y)
    for &(x, y) in &[
        (1.0_f64, 2.0_f64),
        (2.0, 1.0),
        (0.5, 0.5),
        (3.0, 7.0),
        (0.0, 5.0),
    ] {
        points.push(PointCase {
            case_id: format!("kl_div_{x}_{y}"),
            op: "kl_div".into(),
            a: x,
            b: y,
            c: 0.0,
        });
    }

    // sinc_squared(x)
    for x in [0.0_f64, 0.5, 1.0, 1.5, 2.0, -0.5, 3.0] {
        points.push(PointCase {
            case_id: format!("sinc_squared_{x}"),
            op: "sinc_squared".into(),
            a: x,
            b: 0.0,
            c: 0.0,
        });
    }

    // log_comb(n, k)
    for &(n, k) in &[
        (10_f64, 3_f64),
        (5.0, 2.0),
        (20.0, 7.0),
        (4.0, 0.0),
        (100.0, 50.0),
    ] {
        points.push(PointCase {
            case_id: format!("log_comb_{n}_{k}"),
            op: "log_comb".into(),
            a: n,
            b: k,
            c: 0.0,
        });
    }

    // copysign(x, y)
    for &(x, y) in &[
        (3.0_f64, -2.0_f64),
        (3.0, 2.0),
        (-3.0, 5.0),
        (0.0, -1.0),
        (1.5, 0.0),
    ] {
        points.push(PointCase {
            case_id: format!("copysign_{x}_{y}"),
            op: "copysign".into(),
            a: x,
            b: y,
            c: 0.0,
        });
    }

    // nextafter(x, y)
    for &(x, y) in &[(1.0_f64, 2.0_f64), (1.0, 0.0), (-1.0, -2.0), (0.0, 1.0)] {
        points.push(PointCase {
            case_id: format!("nextafter_{x}_{y}"),
            op: "nextafter".into(),
            a: x,
            b: y,
            c: 0.0,
        });
    }

    // spacing(x)
    for x in [1.0_f64, 2.0, 1e10, 1e-10, 100.0] {
        points.push(PointCase {
            case_id: format!("spacing_{x}"),
            op: "spacing".into(),
            a: x,
            b: 0.0,
            c: 0.0,
        });
    }

    // rint(x)
    for x in [2.5_f64, 3.5, -0.5, -1.5, 0.7, -0.7] {
        points.push(PointCase {
            case_id: format!("rint_{x}"),
            op: "rint".into(),
            a: x,
            b: 0.0,
            c: 0.0,
        });
    }

    // fix(x) = trunc towards zero
    for x in [2.7_f64, -2.7, 3.5, -3.5, 0.0, 0.999, -0.999] {
        points.push(PointCase {
            case_id: format!("fix_{x}"),
            op: "fix".into(),
            a: x,
            b: 0.0,
            c: 0.0,
        });
    }

    // divmod(x, y)
    for &(x, y) in &[(7.0_f64, 3.0_f64), (10.0, 4.0), (-7.0, 3.0), (5.5, 1.5)] {
        points.push(PointCase {
            case_id: format!("divmod_{x}_{y}"),
            op: "divmod".into(),
            a: x,
            b: y,
            c: 0.0,
        });
    }

    // radian(deg, min, sec)
    for &(d, m, s) in &[
        (45.0_f64, 30.0_f64, 0.0_f64),
        (90.0, 0.0, 0.0),
        (180.0, 0.0, 0.0),
        (45.0, 0.0, 30.0),
    ] {
        points.push(PointCase {
            case_id: format!("radian_{d}_{m}_{s}"),
            op: "radian".into(),
            a: d,
            b: m,
            c: s,
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
from scipy import special

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    a = float(case["a"]); b = float(case["b"]); c = float(case["c"])
    try:
        v = None; rem = None
        if op == "kl_div":
            v = float(special.kl_div(a, b))
        elif op == "sinc_squared":
            # sinc²(x) where sinc(x) = sin(pi*x) / (pi*x), sinc(0)=1
            if a == 0.0:
                v = 1.0
            else:
                px = math.pi * a
                s = math.sin(px) / px
                v = float(s * s)
        elif op == "log_comb":
            # log C(n, k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
            v = float(math.lgamma(a + 1.0) - math.lgamma(b + 1.0) - math.lgamma(a - b + 1.0))
        elif op == "copysign":
            v = float(math.copysign(a, b))
        elif op == "nextafter":
            v = float(math.nextafter(a, b))
        elif op == "spacing":
            v = float(np.spacing(a))
        elif op == "rint":
            v = float(np.rint(a))
        elif op == "fix":
            v = float(np.fix(a))
        elif op == "divmod":
            q_, r_ = divmod(a, b)
            v = float(q_); rem = float(r_)
        elif op == "radian":
            total_deg = a + b / 60.0 + c / 3600.0
            v = float(total_deg * math.pi / 180.0)
        if v is None or not math.isfinite(v):
            points.append({"case_id": cid, "value": None, "remainder": None})
        else:
            points.append({"case_id": cid, "value": v,
                           "remainder": rem if rem is not None else None})
    except Exception:
        points.append({"case_id": cid, "value": None, "remainder": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize utility_scalars query");
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
                "failed to spawn python3 for utility_scalars oracle: {e}"
            );
            eprintln!("skipping utility_scalars oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open utility_scalars oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "utility_scalars oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping utility_scalars oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for utility_scalars oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "utility_scalars oracle failed: {stderr}"
        );
        eprintln!("skipping utility_scalars oracle: python not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse utility_scalars oracle JSON"))
}

#[test]
fn diff_special_utility_scalars() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.value else {
            continue;
        };
        let abs_d = match case.op.as_str() {
            "kl_div" => (kl_div(case.a, case.b) - expected).abs(),
            "sinc_squared" => (sinc_squared(case.a) - expected).abs(),
            "log_comb" => (log_comb(case.a, case.b) - expected).abs(),
            "copysign" => (copysign(case.a, case.b) - expected).abs(),
            "nextafter" => (nextafter(case.a, case.b) - expected).abs(),
            "spacing" => (spacing(case.a) - expected).abs(),
            "rint" => (rint(case.a) - expected).abs(),
            "fix" => (fix(case.a) - expected).abs(),
            "divmod" => {
                let Some(rem_exp) = scipy_arm.remainder else {
                    continue;
                };
                let (q, r) = divmod(case.a, case.b);
                (q - expected).abs().max((r - rem_exp).abs())
            }
            "radian" => (radian(case.a, case.b, case.c) - expected).abs(),
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
        test_id: "diff_special_utility_scalars".into(),
        category: "fsci_special utility scalars".into(),
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
        "utility_scalars conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
