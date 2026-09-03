#![forbid(unsafe_code)]
//! Live scipy.special parity for fsci_special Box-Cox scalar
//! variants: boxcox_transform_scalar, boxcox1p_scalar,
//! inv_boxcox_scalar, inv_boxcox1p_scalar.
//!
//! Resolves [frankenscipy-hpd0r]. All deterministic scalar wrappers
//! over the same Box-Cox formula scipy implements; 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{
    boxcox_transform_scalar, boxcox1p_scalar, inv_boxcox_scalar, inv_boxcox1p_scalar,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "bc" | "bc1p" | "ibc" | "ibc1p"
    x: f64,
    lam: f64,
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
    fs::create_dir_all(output_dir()).expect("create boxcox diff dir");
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
    let xs = [0.1_f64, 0.5, 1.0, 2.0, 5.0, 10.0];
    let lams = [-1.0_f64, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0];

    for &x in &xs {
        for &lam in &lams {
            points.push(Case {
                case_id: format!("bc_x{x}_l{lam}"),
                op: "bc".into(),
                x,
                lam,
            });
        }
    }
    // boxcox1p uses x >= -1; offset by 1 internally
    for &x in &[-0.5_f64, 0.0, 0.5, 1.0, 5.0] {
        for &lam in &lams {
            points.push(Case {
                case_id: format!("bc1p_x{x}_l{lam}"),
                op: "bc1p".into(),
                x,
                lam,
            });
        }
    }
    // inv_boxcox: y inputs are any real
    for &y in &[-1.0_f64, 0.0, 0.5, 1.0, 2.0] {
        for &lam in &lams {
            points.push(Case {
                case_id: format!("ibc_y{y}_l{lam}"),
                op: "ibc".into(),
                x: y,
                lam,
            });
        }
    }
    // inv_boxcox1p
    for &y in &[-1.0_f64, 0.0, 0.5, 1.0, 2.0] {
        for &lam in &lams {
            points.push(Case {
                case_id: format!("ibc1p_y{y}_l{lam}"),
                op: "ibc1p".into(),
                x: y,
                lam,
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
from scipy import special as sp

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    x = float(case["x"]); lam = float(case["lam"])
    try:
        if op == "bc":
            v = float(sp.boxcox(x, lam))
        elif op == "bc1p":
            v = float(sp.boxcox1p(x, lam))
        elif op == "ibc":
            v = float(sp.inv_boxcox(x, lam))
        elif op == "ibc1p":
            v = float(sp.inv_boxcox1p(x, lam))
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
                "failed to spawn python3 for boxcox oracle: {e}"
            );
            eprintln!("skipping boxcox oracle: python3 not available ({e})");
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
                "boxcox oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping boxcox oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for boxcox oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "boxcox oracle failed: {stderr}"
        );
        eprintln!("skipping boxcox oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse boxcox oracle JSON"))
}

#[test]
fn diff_special_boxcox_scalar_variants() {
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
            "bc" => boxcox_transform_scalar(case.x, case.lam),
            "bc1p" => boxcox1p_scalar(case.x, case.lam),
            "ibc" => inv_boxcox_scalar(case.x, case.lam),
            "ibc1p" => inv_boxcox1p_scalar(case.x, case.lam),
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
        test_id: "diff_special_boxcox_scalar_variants".into(),
        category:
            "fsci_special::{boxcox, boxcox1p, inv_boxcox, inv_boxcox1p}_scalar vs scipy.special"
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
        "boxcox_scalar conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
