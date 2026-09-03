#![forbid(unsafe_code)]
//! Live scipy.special.powm1 parity for the explicit fsci_special::powm1 API.
//!
//! Resolves [frankenscipy-uadm6]. Tolerance: 1e-13 absolute for finite
//! outputs; NaN/infinity outputs compare by classification.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::powm1;
use fsci_special::types::SpecialTensor;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-015";
const ABS_TOL: f64 = 1.0e-13;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x_class: String,
    y_class: String,
    x: Option<f64>,
    y: Option<f64>,
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

fn value_for(class_name: &str, value: Option<f64>) -> Option<f64> {
    match class_name {
        "finite" => value,
        "pos_inf" => Some(f64::INFINITY),
        "neg_inf" => Some(f64::NEG_INFINITY),
        "nan" => Some(f64::NAN),
        _ => None,
    }
}

fn fsci_eval(case: &PointCase) -> PointArm {
    let Some(x) = value_for(&case.x_class, case.x) else {
        return PointArm {
            case_id: case.case_id.clone(),
            value_class: "error".into(),
            value: None,
        };
    };
    let Some(y) = value_for(&case.y_class, case.y) else {
        return PointArm {
            case_id: case.case_id.clone(),
            value_class: "error".into(),
            value: None,
        };
    };

    let result = powm1(
        &SpecialTensor::RealScalar(x),
        &SpecialTensor::RealScalar(y),
        RuntimeMode::Strict,
    );

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

fn finite_case(case_id: &str, x: f64, y: f64) -> PointCase {
    classified_case(case_id, "finite", Some(x), "finite", Some(y))
}

fn classified_case(
    case_id: &str,
    x_class: &str,
    x: Option<f64>,
    y_class: &str,
    y: Option<f64>,
) -> PointCase {
    PointCase {
        case_id: case_id.into(),
        x_class: x_class.into(),
        y_class: y_class.into(),
        x,
        y,
    }
}

fn generate_query() -> OracleQuery {
    let points = vec![
        finite_case("near_one", 1.0 + 1.0e-15, 1.0),
        finite_case("positive_fraction", 0.5, 0.5),
        finite_case("positive_negative_power", 2.0, -2.0),
        finite_case("zero_positive_power", 0.0, 2.0),
        finite_case("zero_negative_power", 0.0, -1.0),
        finite_case("negative_integer_power", -2.0, 3.0),
        finite_case("negative_fractional_power", -2.0, 0.5),
        classified_case("one_infinite_power", "finite", Some(1.0), "pos_inf", None),
        classified_case("two_infinite_power", "finite", Some(2.0), "pos_inf", None),
        classified_case(
            "two_negative_infinite_power",
            "finite",
            Some(2.0),
            "neg_inf",
            None,
        ),
        classified_case("x_pos_inf_y_one", "pos_inf", None, "finite", Some(1.0)),
        classified_case("x_neg_inf_y_fraction", "neg_inf", None, "finite", Some(0.5)),
        classified_case("x_nan_y_one", "nan", None, "finite", Some(1.0)),
        classified_case("x_two_y_nan", "finite", Some(2.0), "nan", None),
    ];
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import special

def value(class_name, raw):
    if class_name == "finite":
        return float(raw)
    if class_name == "pos_inf":
        return math.inf
    if class_name == "neg_inf":
        return -math.inf
    if class_name == "nan":
        return math.nan
    raise ValueError(f"unsupported class {class_name}")

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
        x = value(case["x_class"], case["x"])
        y = value(case["y_class"], case["y"])
        r = float(special.powm1(x, y))
        value_class, out = cls(r)
        points.append({"case_id": cid, "value_class": value_class, "value": out})
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
                "failed to spawn python3 for powm1 oracle: {e}"
            );
            eprintln!("skipping powm1 oracle: python3 not available ({e})");
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
                "powm1 oracle script write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping powm1 oracle: script write failed ({err})\n{stderr}");
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
fn diff_special_powm1() {
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
        test_id: "diff_special_powm1".into(),
        category: "fsci_special::powm1 vs scipy.special.powm1".into(),
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
            eprintln!("powm1 mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "powm1 conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
