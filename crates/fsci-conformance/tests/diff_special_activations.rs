#![forbid(unsafe_code)]
//! Formula-derived differential coverage for fsci_special activation
//! functions with scalar signatures: elu, gelu, selu, swish,
//! leaky_relu, celu, hard_swish_scalar.
//!
//! Resolves [frankenscipy-ac42u]. 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{celu, elu, gelu, hard_swish_scalar, leaky_relu, selu, swish};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    x: f64,
    /// Auxiliary parameter (alpha for elu/leaky_relu/celu; beta for swish).
    param: f64,
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
    fs::create_dir_all(output_dir()).expect("create activations diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize activations diff log");
    fs::write(path, json).expect("write activations diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    let xs = [-3.0_f64, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 3.0];

    // Funcs that take only x
    for x in xs {
        for op in ["gelu", "selu", "hard_swish"] {
            points.push(PointCase {
                case_id: format!("{op}_x{x}"),
                op: op.into(),
                x,
                param: 0.0,
            });
        }
    }

    // elu(x, alpha)
    for x in xs {
        for alpha in [1.0_f64, 0.5, 2.0] {
            points.push(PointCase {
                case_id: format!("elu_x{x}_a{alpha}"),
                op: "elu".into(),
                x,
                param: alpha,
            });
        }
    }

    // leaky_relu(x, alpha)
    for x in xs {
        for alpha in [0.01_f64, 0.1, 0.2] {
            points.push(PointCase {
                case_id: format!("leaky_relu_x{x}_a{alpha}"),
                op: "leaky_relu".into(),
                x,
                param: alpha,
            });
        }
    }

    // swish(x, beta)
    for x in xs {
        for beta in [1.0_f64, 0.5, 2.0] {
            points.push(PointCase {
                case_id: format!("swish_x{x}_b{beta}"),
                op: "swish".into(),
                x,
                param: beta,
            });
        }
    }

    // celu(x, alpha)
    for x in xs {
        for alpha in [1.0_f64, 0.5, 2.0] {
            points.push(PointCase {
                case_id: format!("celu_x{x}_a{alpha}"),
                op: "celu".into(),
                x,
                param: alpha,
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
from scipy import special

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    x = float(case["x"]); param = float(case["param"])
    try:
        if op == "gelu":
            v = 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
        elif op == "selu":
            alpha = 1.6732632423543772
            scale = 1.0507009873554805
            v = scale * (x if x > 0.0 else alpha * (math.exp(x) - 1.0))
        elif op == "hard_swish":
            # x * relu6(x + 3) / 6
            r6 = max(0.0, min(6.0, x + 3.0))
            v = x * r6 / 6.0
        elif op == "elu":
            v = x if x > 0.0 else param * (math.exp(x) - 1.0)
        elif op == "leaky_relu":
            v = x if x > 0.0 else param * x
        elif op == "swish":
            bx = param * x
            sig = 1.0 / (1.0 + math.exp(-bx))
            v = x * sig
        elif op == "celu":
            # celu(x, alpha) = max(0, x) + min(0, alpha*(exp(x/alpha) - 1))
            pos = max(0.0, x)
            neg = min(0.0, param * (math.exp(x / param) - 1.0))
            v = pos + neg
        else:
            v = None
        if v is None or not math.isfinite(v):
            points.append({"case_id": cid, "value": None})
        else:
            points.append({"case_id": cid, "value": float(v)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize activations query");
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
                "failed to spawn python3 for activations oracle: {e}"
            );
            eprintln!("skipping activations oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open activations oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "activations oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping activations oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for activations oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "activations oracle failed: {stderr}"
        );
        eprintln!("skipping activations oracle: python not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse activations oracle JSON"))
}

#[test]
fn diff_special_activations() {
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
        let fsci_v = match case.op.as_str() {
            "gelu" => gelu(case.x),
            "selu" => selu(case.x),
            "hard_swish" => hard_swish_scalar(case.x),
            "elu" => elu(case.x, case.param),
            "leaky_relu" => leaky_relu(case.x, case.param),
            "swish" => swish(case.x, case.param),
            "celu" => celu(case.x, case.param),
            _ => continue,
        };
        let abs_d = (fsci_v - expected).abs();
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
        test_id: "diff_special_activations".into(),
        category: "fsci_special activations (formula-derived)".into(),
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
        "activations conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
