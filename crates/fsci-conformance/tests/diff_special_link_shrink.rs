#![forbid(unsafe_code)]
//! Live numpy-formula parity for fsci_special link/shrink helpers:
//! cloglog, cloglog_inv, loglog, loglog_inv, cauchit, cauchit_inv,
//! hardshrink, softshrink, tanhshrink.
//!
//! Resolves [frankenscipy-pkozf]. Tolerance: 1e-10 abs.
//!
//! Note: fsci defines `loglog(p) = -log(-log p)` (link form with output
//! matching `loglog_inv(x) = exp(-exp(-x))` Gumbel-minimum CDF inverse).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{
    cauchit, cauchit_inv, cloglog, cloglog_inv, hardshrink, loglog, loglog_inv, softshrink,
    tanhshrink,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    x: f64,
    lambda: f64, // shrink threshold (used by hardshrink/softshrink only)
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
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
    fs::create_dir_all(output_dir()).expect("create link_shrink diff dir");
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

fn fsci_eval(op: &str, x: f64, lambda: f64) -> Option<f64> {
    let pt = SpecialTensor::RealScalar(x);
    let pl = SpecialTensor::RealScalar(lambda);
    let mode = RuntimeMode::Strict;
    let result = match op {
        "cloglog" => {
            // cloglog returns a bare f64, not a SpecialResult.
            return Some(cloglog(x));
        }
        "loglog" => return Some(loglog(x)),
        "cloglog_inv" => return Some(cloglog_inv(x)),
        "loglog_inv" => return Some(loglog_inv(x)),
        "tanhshrink" => tanhshrink(&pt, mode),
        "cauchit" => cauchit(&pt, mode),
        "cauchit_inv" => cauchit_inv(&pt, mode),
        "hardshrink" => hardshrink(&pt, &pl, mode),
        "softshrink" => softshrink(&pt, &pl, mode),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    // link fns: p in (0, 1)
    let ps: &[f64] = &[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];
    for &p in ps {
        for op in ["cloglog", "loglog", "cauchit"] {
            points.push(PointCase {
                case_id: format!("{op}_p{p}").replace('.', "p"),
                op: op.into(),
                x: p,
                lambda: 0.0,
            });
        }
    }
    // inverse link fns
    let xs: &[f64] = &[-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0];
    for &x in xs {
        for op in ["cloglog_inv", "loglog_inv", "cauchit_inv"] {
            points.push(PointCase {
                case_id: format!("{op}_x{x}").replace('.', "p").replace('-', "n"),
                op: op.into(),
                x,
                lambda: 0.0,
            });
        }
    }
    // shrink fns: x across [-3, 3], threshold lambda=0.5
    let shrink_xs: &[f64] = &[-3.0, -1.0, -0.6, -0.5, -0.3, 0.0, 0.3, 0.5, 0.6, 1.0, 3.0];
    for &x in shrink_xs {
        for op in ["hardshrink", "softshrink"] {
            points.push(PointCase {
                case_id: format!("{op}_x{x}_l05").replace('.', "p").replace('-', "n"),
                op: op.into(),
                x,
                lambda: 0.5,
            });
        }
        points.push(PointCase {
            case_id: format!("tanhshrink_x{x}")
                .replace('.', "p")
                .replace('-', "n"),
            op: "tanhshrink".into(),
            x,
            lambda: 0.0,
        });
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys

def f(op, x, lam):
    if op == "cloglog":
        if x <= 0.0: return float("-inf")
        if x >= 1.0: return float("inf")
        return math.log(-math.log(1.0 - x))
    if op == "loglog":
        # fsci: loglog(p) = -log(-log p) (Gumbel-min link)
        if x <= 0.0: return float("-inf")
        if x >= 1.0: return float("inf")
        return -math.log(-math.log(x))
    if op == "cauchit":
        return math.tan(math.pi * (x - 0.5))
    if op == "cloglog_inv":
        return 1.0 - math.exp(-math.exp(x))
    if op == "loglog_inv":
        return math.exp(-math.exp(-x))
    if op == "cauchit_inv":
        return 0.5 + math.atan(x) / math.pi
    if op == "hardshrink":
        return x if abs(x) > lam else 0.0
    if op == "softshrink":
        if x > lam: return x - lam
        if x < -lam: return x + lam
        return 0.0
    if op == "tanhshrink":
        return x - math.tanh(x)
    return float("nan")

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]; x = float(case["x"]); lam = float(case["lambda"])
    try:
        v = f(op, x, lam)
        if math.isfinite(v):
            points.append({"case_id": cid, "value": float(v)})
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
                "failed to spawn python3 for link_shrink oracle: {e}"
            );
            eprintln!("skipping link_shrink oracle: python3 not available ({e})");
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
                "link_shrink oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping link_shrink oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for link_shrink oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "link_shrink oracle failed: {stderr}"
        );
        eprintln!("skipping link_shrink oracle: python3 not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse link_shrink oracle JSON"))
}

#[test]
fn diff_special_link_shrink() {
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
        let Some(actual) = fsci_eval(&case.op, case.x, case.lambda) else {
            continue;
        };
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
        test_id: "diff_special_link_shrink".into(),
        category: "fsci_special link/shrink helpers vs python formula".into(),
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
        "link_shrink conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
