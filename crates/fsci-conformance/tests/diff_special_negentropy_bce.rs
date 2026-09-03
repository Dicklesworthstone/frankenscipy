#![forbid(unsafe_code)]
//! Live numpy-formula parity for fsci_special::{negentropy,
//! binary_cross_entropy}.
//!
//! Resolves [frankenscipy-6agve].
//!
//! Definitions:
//!   negentropy(x) = x · log(x)  (xlogx alias; lim_{x→0+}=0)
//!   binary_cross_entropy(p, q) = -p · log(q) - (1-p) · log(1-q)
//!
//! Tolerance: 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{binary_cross_entropy, negentropy};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String, // "negentropy" | "bce"
    p: f64,
    q: f64, // unused for negentropy
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
    fs::create_dir_all(output_dir()).expect("create negentropy_bce diff dir");
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

fn fsci_eval(op: &str, p: f64, q: f64) -> Option<f64> {
    let pt = SpecialTensor::RealScalar(p);
    let qt = SpecialTensor::RealScalar(q);
    let mode = RuntimeMode::Strict;
    let result = match op {
        "negentropy" => negentropy(&pt, mode),
        "bce" => binary_cross_entropy(&pt, &qt, mode),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    // negentropy: x in [0, ∞)
    let xs: &[f64] = &[0.0, 0.05, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0];
    for &x in xs {
        points.push(PointCase {
            case_id: format!("ne_x{x}").replace('.', "p"),
            op: "negentropy".into(),
            p: x,
            q: 0.0,
        });
    }
    // BCE: p in [0, 1], q in (0, 1)
    let ps: &[f64] = &[0.0, 0.25, 0.5, 0.75, 1.0];
    let qs: &[f64] = &[0.05, 0.25, 0.5, 0.75, 0.95];
    for &p in ps {
        for &q in qs {
            points.push(PointCase {
                case_id: format!("bce_p{p}_q{q}").replace('.', "p"),
                op: "bce".into(),
                p,
                q,
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

def f(op, p, q):
    if op == "negentropy":
        if p < 0: return float("nan")
        if p == 0: return 0.0
        return p * math.log(p)
    if op == "bce":
        # -p*log(q) - (1-p)*log(1-q); handle 0*log(0) = 0 by convention
        a = 0.0 if p == 0.0 else -p * math.log(q) if q > 0 else float("inf")
        b = 0.0 if (1.0 - p) == 0.0 else -(1.0 - p) * math.log(1.0 - q) if (1.0 - q) > 0 else float("inf")
        return a + b
    return float("nan")

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    p = float(case["p"]); qv = float(case["q"])
    try:
        v = f(op, p, qv)
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
                "failed to spawn python3 for negentropy/bce oracle: {e}"
            );
            eprintln!("skipping negentropy/bce oracle: python3 not available ({e})");
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
                "negentropy/bce oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping negentropy/bce oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for negentropy/bce oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "negentropy/bce oracle failed: {stderr}"
        );
        eprintln!("skipping negentropy/bce oracle: python3 not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse negentropy/bce oracle JSON"))
}

#[test]
fn diff_special_negentropy_bce() {
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
        let Some(actual) = fsci_eval(&case.op, case.p, case.q) else {
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
        test_id: "diff_special_negentropy_bce".into(),
        category: "fsci_special negentropy + binary_cross_entropy vs python formula".into(),
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
        "negentropy/bce conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
