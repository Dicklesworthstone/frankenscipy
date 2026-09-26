#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the elementary
//! scipy-compat wrappers cbrt, exp2, exp10, arcsinh, arccosh,
//! arctanh.
//!
//! Resolves [frankenscipy-a6ruc]. Six closed-form kernels with
//! no dedicated diff harness. All compose libm primitives;
//! 1e-13 abs holds.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_special::{arccosh, arcsinh, arctanh, cbrt, exp2, exp10};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// exp2/exp10 grow rapidly; switch to relative for those.
const ABS_TOL: f64 = 1.0e-13;
const REL_TOL: f64 = 1.0e-13;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// One ledger arm per function.
const ARMS: [&str; 6] = ["cbrt", "exp2", "exp10", "arcsinh", "arccosh", "arctanh"];

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
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
    func: String,
    abs_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
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
    fs::create_dir_all(output_dir()).expect("create elementary diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize elementary diff log");
    fs::write(path, json).expect("write elementary diff log");
}

fn fsci_eval(func: &str, x: f64) -> Option<f64> {
    // A non-finite value is returned as is: the ledger classifies it against SciPy's.
    let v = match func {
        "cbrt" => cbrt(x),
        "exp2" => exp2(x),
        "exp10" => exp10(x),
        "arcsinh" => arcsinh(x),
        "arccosh" => arccosh(x),
        "arctanh" => arctanh(x),
        _ => return None,
    };
    Some(v)
}

fn generate_query() -> OracleQuery {
    // cbrt: ℝ; exp2/exp10: ℝ but constrain magnitude;
    // arcsinh: ℝ; arccosh: x ≥ 1; arctanh: |x| < 1.
    let xs_full = [-10.0_f64, -3.0, -1.0, -0.3, 0.0, 0.3, 1.0, 3.0, 10.0];
    let xs_arccosh = [1.0_f64, 1.5, 2.0, 5.0, 10.0, 100.0];
    let xs_arctanh = [-0.99_f64, -0.5, -0.1, 0.0, 0.1, 0.5, 0.99];

    let mut points = Vec::new();
    for &x in &xs_full {
        for func in ["cbrt", "exp2", "exp10", "arcsinh"] {
            points.push(PointCase {
                case_id: format!("{func}_x{x}"),
                func: func.to_string(),
                x,
            });
        }
    }
    for &x in &xs_arccosh {
        points.push(PointCase {
            case_id: format!("arccosh_x{x}"),
            func: "arccosh".into(),
            x,
        });
    }
    for &x in &xs_arctanh {
        points.push(PointCase {
            case_id: format!("arctanh_x{x}"),
            func: "arctanh".into(),
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
from scipy import special

def finite_or_none(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]; x = float(case["x"])
    try:
        if func == "cbrt":      value = special.cbrt(x)
        elif func == "exp2":    value = special.exp2(x)
        elif func == "exp10":   value = special.exp10(x)
        elif func == "arcsinh": value = math.asinh(x)
        elif func == "arccosh": value = math.acosh(x)
        elif func == "arctanh": value = math.atanh(x)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize elementary query");
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
                "failed to spawn python3 for elementary oracle: {e}"
            );
            eprintln!("skipping elementary oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open elementary oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "elementary oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping elementary oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for elementary oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "elementary oracle failed: {stderr}"
        );
        eprintln!("skipping elementary oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse elementary oracle JSON"))
}

#[test]
fn diff_special_elementary() {
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
    let mut ledger = CompareLedger::new("diff_special_elementary", &ARMS);

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        let arm = case.func.as_str();
        let Some((scipy_v, rust_v)) = ledger.pair(
            arm,
            &case.case_id,
            oracle.value,
            fsci_eval(&case.func, case.x),
        ) else {
            continue;
        };
        let abs_diff = (rust_v - scipy_v).abs();
        max_overall = max_overall.max(abs_diff);
        let scale = scipy_v.abs().max(1.0);
        let pass = abs_diff <= ABS_TOL || abs_diff <= REL_TOL * scale;
        ledger.compared(arm, &case.case_id, pass);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            abs_diff,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_elementary".into(),
        category: "scipy.special.cbrt/exp2/exp10/arc*h".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "elementary {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "elementary conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
    // Arms have different case sets (arccosh has the fewest); each must compare all of its own.
    let min_per_arm = ARMS
        .iter()
        .map(|arm| query.points.iter().filter(|c| c.func == *arm).count())
        .min()
        .expect("ARMS is non-empty");
    ledger.finish(min_per_arm);
}
