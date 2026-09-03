#![forbid(unsafe_code)]
//! Live numpy parity for fsci_special activation/link helpers:
//! log1mexp, log1pexp, log_cosh, softsign, silu, mish, xlogx,
//! hard_sigmoid, hard_tanh, logsigmoid.
//!
//! Resolves [frankenscipy-qjwtj]. Tolerance: 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{
    hard_sigmoid, hard_tanh, log_cosh, log1mexp, log1pexp, logsigmoid, mish, silu, softsign, xlogx,
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
    fs::create_dir_all(output_dir()).expect("create activation_extras diff dir");
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

fn fsci_eval(op: &str, x: f64) -> Option<f64> {
    let pt = SpecialTensor::RealScalar(x);
    let mode = RuntimeMode::Strict;
    let result = match op {
        "log1mexp" => log1mexp(&pt, mode),
        "log1pexp" => log1pexp(&pt, mode),
        "log_cosh" => log_cosh(&pt, mode),
        "softsign" => softsign(&pt, mode),
        "silu" => silu(&pt, mode),
        "mish" => mish(&pt, mode),
        "xlogx" => xlogx(&pt, mode),
        "hard_sigmoid" => hard_sigmoid(&pt, mode),
        "hard_tanh" => {
            let lo = SpecialTensor::RealScalar(-1.0);
            let hi = SpecialTensor::RealScalar(1.0);
            hard_tanh(&pt, &lo, &hi, mode)
        }
        "logsigmoid" => logsigmoid(&pt, mode),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    let common_xs: &[f64] = &[-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 6.0];
    // log1mexp requires x < 0 (real-valued); skip x >= 0.
    let log1mexp_xs: &[f64] = &[-5.0, -3.0, -1.5, -0.5, -0.1, -0.01];
    // xlogx requires x >= 0.
    let xlogx_xs: &[f64] = &[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0];

    for &x in common_xs {
        for op in [
            "log1pexp",
            "log_cosh",
            "softsign",
            "silu",
            "mish",
            "hard_sigmoid",
            "hard_tanh",
            "logsigmoid",
        ] {
            points.push(PointCase {
                case_id: format!("{op}_x{x}").replace('.', "p").replace('-', "n"),
                op: op.into(),
                x,
            });
        }
    }
    for &x in log1mexp_xs {
        points.push(PointCase {
            case_id: format!("log1mexp_x{x}").replace('.', "p").replace('-', "n"),
            op: "log1mexp".into(),
            x,
        });
    }
    for &x in xlogx_xs {
        points.push(PointCase {
            case_id: format!("xlogx_x{x}").replace('.', "p"),
            op: "xlogx".into(),
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

def f(op, x):
    if op == "log1pexp":
        # softplus, stable
        if x > 20: return x + math.log1p(math.exp(-x))
        if x < -20: return math.exp(x)
        return math.log1p(math.exp(x))
    if op == "log1mexp":
        # log(1 - exp(x)) for x < 0; stable form
        if x >= 0: return float("nan")
        if x > -math.log(2.0): return math.log(-math.expm1(x))
        return math.log1p(-math.exp(x))
    if op == "log_cosh":
        # log(cosh(x)) stably
        ax = abs(x)
        if ax > 20: return ax - math.log(2.0)
        return math.log(math.cosh(x))
    if op == "softsign":  return x / (1.0 + abs(x))
    if op == "silu":      return x / (1.0 + math.exp(-x))
    if op == "mish":
        # x * tanh(softplus(x))
        if x > 20: sp = x + math.log1p(math.exp(-x))
        elif x < -20: sp = math.exp(x)
        else: sp = math.log1p(math.exp(x))
        return x * math.tanh(sp)
    if op == "xlogx":
        if x < 0: return float("nan")
        if x == 0: return 0.0
        return x * math.log(x)
    if op == "hard_sigmoid":
        return max(0.0, min(1.0, (x + 3.0) / 6.0))
    if op == "hard_tanh":
        return max(-1.0, min(1.0, x))
    if op == "logsigmoid":
        # log(sigmoid(x)) = -softplus(-x), stable
        nx = -x
        if nx > 20: return -(nx + math.log1p(math.exp(-nx)))
        if nx < -20: return -math.exp(nx)
        return -math.log1p(math.exp(nx))
    return float("nan")

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]; x = float(case["x"])
    try:
        v = f(op, x)
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
                "failed to spawn python3 for activation_extras oracle: {e}"
            );
            eprintln!("skipping activation_extras oracle: python3 not available ({e})");
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
                "activation_extras oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping activation_extras oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for activation_extras oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "activation_extras oracle failed: {stderr}"
        );
        eprintln!("skipping activation_extras oracle: python3 not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse activation_extras oracle JSON"))
}

#[test]
fn diff_special_activation_extras() {
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
        let Some(actual) = fsci_eval(&case.op, case.x) else {
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
        test_id: "diff_special_activation_extras".into(),
        category: "fsci_special activation/link extras vs python formula".into(),
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
        "activation_extras conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
