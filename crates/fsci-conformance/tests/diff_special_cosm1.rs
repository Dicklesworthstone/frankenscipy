#![forbid(unsafe_code)]
//! Live scipy.special.cosm1 parity for the explicit fsci_special::cosm1 API.
//!
//! Resolves [frankenscipy-13ogr]. Tolerance: 1e-13 absolute for finite
//! outputs; NaN/infinity outputs compare by classification.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::cosm1;
use fsci_special::types::SpecialTensor;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-013";
const ABS_TOL: f64 = 1.0e-13;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    input_class: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value_class: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create cosm1 diff dir");
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

fn classify(value: f64) -> &'static str {
    if value.is_nan() {
        "nan"
    } else if value == f64::INFINITY {
        "pos_inf"
    } else if value == f64::NEG_INFINITY {
        "neg_inf"
    } else {
        "finite"
    }
}

fn finite_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn case_value(case: &PointCase) -> Option<f64> {
    match case.input_class.as_str() {
        "finite" => case.value,
        "pos_inf" => Some(f64::INFINITY),
        "neg_inf" => Some(f64::NEG_INFINITY),
        "nan" => Some(f64::NAN),
        _ => None,
    }
}

fn fsci_eval(case: &PointCase) -> PointArm {
    let Some(input) = case_value(case) else {
        return PointArm {
            case_id: case.case_id.clone(),
            value_class: "error".into(),
            value: None,
        };
    };

    let result = cosm1(&SpecialTensor::RealScalar(input), RuntimeMode::Strict);

    match result {
        Ok(SpecialTensor::RealScalar(value)) => PointArm {
            case_id: case.case_id.clone(),
            value_class: classify(value).into(),
            value: finite_value(value),
        },
        _ => PointArm {
            case_id: case.case_id.clone(),
            value_class: "error".into(),
            value: None,
        },
    }
}

fn finite_case(case_id: &str, value: f64) -> PointCase {
    PointCase {
        case_id: case_id.into(),
        input_class: "finite".into(),
        value: Some(value),
    }
}

fn special_case(case_id: &str, input_class: &str) -> PointCase {
    PointCase {
        case_id: case_id.into(),
        input_class: input_class.into(),
        value: None,
    }
}

fn generate_query() -> OracleQuery {
    let points = vec![
        finite_case("neg_pi", -std::f64::consts::PI),
        finite_case("neg_2", -2.0),
        finite_case("neg_0p5", -0.5),
        finite_case("neg_1e_minus_8", -1.0e-8),
        finite_case("zero", 0.0),
        finite_case("pos_1e_minus_12", 1.0e-12),
        finite_case("pos_0p5", 0.5),
        finite_case("pos_pi", std::f64::consts::PI),
        finite_case("pos_20", 20.0),
        special_case("pos_inf", "pos_inf"),
        special_case("neg_inf", "neg_inf"),
        special_case("nan", "nan"),
    ];
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import special

def input_value(case):
    cls = case["input_class"]
    if cls == "finite":
        return float(case["value"])
    if cls == "pos_inf":
        return math.inf
    if cls == "neg_inf":
        return -math.inf
    if cls == "nan":
        return math.nan
    raise ValueError(f"unsupported input class {cls}")

def cls(x):
    if math.isnan(x):
        return "nan", None
    if math.isinf(x):
        return ("pos_inf" if x > 0 else "neg_inf"), None
    return "finite", float(x)

q = json.loads(sys.argv[1])
points = []
for case in q["points"]:
    cid = case["case_id"]
    try:
        r = float(special.cosm1(input_value(case)))
        value_class, value = cls(r)
        points.append({"case_id": cid, "value_class": value_class, "value": value})
    except Exception as exc:
        sys.stderr.write(f"oracle {cid}: {exc}\n")
        points.append({"case_id": cid, "value_class": "error", "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-")
        .arg(&query_json)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for cosm1 oracle: {e}"
            );
            eprintln!("skipping cosm1 oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(script.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "cosm1 oracle script write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping cosm1 oracle: script write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for cosm1 oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "cosm1 oracle failed: {stderr}"
        );
        eprintln!("skipping cosm1 oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse cosm1 oracle JSON"))
}

fn compare(case: &PointCase, actual: &PointArm, expected: &PointArm) -> CaseDiff {
    if actual.value_class != expected.value_class {
        return CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: f64::INFINITY,
            pass: false,
        };
    }

    let abs_diff = match (actual.value, expected.value) {
        (Some(actual_value), Some(expected_value)) => (actual_value - expected_value).abs(),
        _ => 0.0,
    };

    CaseDiff {
        case_id: case.case_id.clone(),
        abs_diff,
        pass: abs_diff <= ABS_TOL,
    }
}

#[test]
fn diff_special_cosm1() {
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
        let expected = pmap.get(&case.case_id).expect("oracle case present");
        let actual = fsci_eval(case);
        let diff = compare(case, &actual, expected);
        max_overall = max_overall.max(diff.abs_diff);
        diffs.push(diff);
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_special_cosm1".into(),
        category: "fsci_special::cosm1 vs scipy.special.cosm1".into(),
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
            eprintln!("cosm1 mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "cosm1 conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
