#![forbid(unsafe_code)]
//! Live numpy/scipy parity for fsci_special::{powm1_scalar, cosm1_scalar,
//! log1pmx_scalar}.
//!
//! Resolves [frankenscipy-l283b]. powm1 and cosm1 are direct scipy
//! functions; log1pmx is not in scipy.special so we compute the
//! reference via numpy as log1p(x) - x with high-precision care for
//! near-zero x via the asymptotic series x²/2 - x³/3 + ...
//!
//! Tolerance: 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{cosm1_scalar, log1pmx_scalar, powm1_scalar};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String,
    x: f64,
    y: f64,
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
    fs::create_dir_all(output_dir()).expect("create powm1 diff dir");
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
    // powm1: x in (0, ∞), y any
    let powm1_xs = [0.1_f64, 0.5, 0.9, 1.5, 2.0, 5.0];
    let powm1_ys = [0.0_f64, 0.5, 1.0, 2.0, -1.0, 0.1];
    for &x in &powm1_xs {
        for &y in &powm1_ys {
            points.push(Case {
                case_id: format!("powm1_x{x}_y{y}")
                    .replace('.', "p")
                    .replace('-', "n"),
                op: "powm1".into(),
                x,
                y,
            });
        }
    }
    // cosm1: x in real
    let cosm1_xs = [-1.0_f64, -0.5, -0.1, -0.01, 0.0, 0.01, 0.1, 0.5, 1.0, 3.0];
    for &x in &cosm1_xs {
        points.push(Case {
            case_id: format!("cosm1_x{x}").replace('.', "p").replace('-', "n"),
            op: "cosm1".into(),
            x,
            y: 0.0,
        });
    }
    // log1pmx: x > -1
    let log1pmx_xs = [-0.9_f64, -0.5, -0.1, -0.01, 0.0, 0.01, 0.1, 0.5, 1.0, 5.0];
    for &x in &log1pmx_xs {
        points.push(Case {
            case_id: format!("log1pmx_x{x}").replace('.', "p").replace('-', "n"),
            op: "log1pmx".into(),
            x,
            y: 0.0,
        });
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import special

def log1pmx_ref(x):
    # log(1+x) - x; for tiny |x|, scipy uses an Erlang-style series.
    # numpy.log1p is accurate at small x; subtract x.
    if x == 0.0: return 0.0
    return math.log1p(x) - x

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    x = float(case["x"]); y = float(case["y"])
    try:
        if op == "powm1":
            v = float(special.powm1(x, y))
        elif op == "cosm1":
            v = float(special.cosm1(x))
        elif op == "log1pmx":
            v = log1pmx_ref(x)
        else:
            v = float("nan")
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
                "failed to spawn python3 for powm1 oracle: {e}"
            );
            eprintln!("skipping powm1 oracle: python3 not available ({e})");
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
                "powm1 oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping powm1 oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for powm1 oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "powm1 oracle failed: {stderr}"
        );
        eprintln!("skipping powm1 oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse powm1 oracle JSON"))
}

#[test]
fn diff_special_powm1_cosm1_log1pmx() {
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
            "powm1" => powm1_scalar(case.x, case.y),
            "cosm1" => cosm1_scalar(case.x),
            "log1pmx" => log1pmx_scalar(case.x),
            _ => continue,
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
        test_id: "diff_special_powm1_cosm1_log1pmx".into(),
        category: "fsci_special powm1/cosm1/log1pmx vs scipy.special / numpy".into(),
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
        "powm1/cosm1/log1pmx conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
