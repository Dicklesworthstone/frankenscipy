#![forbid(unsafe_code)]
//! Live scipy.special.airye parity for fsci_special::airye.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::airye;
use fsci_special::types::{Complex64, SpecialTensor};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const ABS_TOL: f64 = 5.0e-4;
const REL_TOL: f64 = 5.0e-4;

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: &'static str,
    re: f64,
    im: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct ComponentArm {
    re: Option<f64>,
    im: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    components: Vec<ComponentArm>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    component: String,
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

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) -> Result<(), String> {
    fs::create_dir_all(output_dir()).map_err(|err| err.to_string())?;
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).map_err(|err| err.to_string())?;
    fs::write(path, json).map_err(|err| err.to_string())
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        points: vec![
            PointCase {
                case_id: "real_zero",
                re: 0.0,
                im: None,
            },
            PointCase {
                case_id: "real_one",
                re: 1.0,
                im: None,
            },
            PointCase {
                case_id: "real_five",
                re: 5.0,
                im: None,
            },
            PointCase {
                case_id: "real_ten",
                re: 10.0,
                im: None,
            },
            PointCase {
                case_id: "real_negative_one",
                re: -1.0,
                im: None,
            },
            PointCase {
                case_id: "complex_negative_one",
                re: -1.0,
                im: Some(0.0),
            },
            PointCase {
                case_id: "complex_positive",
                re: 1.0,
                im: Some(0.0),
            },
        ],
    }
}

fn scipy_required() -> bool {
    std::env::var(REQUIRE_SCIPY_ENV).is_ok()
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Result<Option<OracleResult>, String> {
    let script = r#"
import json
import math
import os
import sys

try:
    from scipy import special
except Exception as exc:
    print(f"scipy import failed: {exc}", file=sys.stderr)
    sys.exit(2)

def component(value):
    real = float(value.real)
    imag = float(value.imag)
    if math.isnan(real) or math.isnan(imag):
        return {"re": None, "im": None}
    return {"re": real, "im": imag}

q = json.loads(os.environ["FSCI_AIRYE_QUERY"])
points = []
for case in q["points"]:
    if case["im"] is None:
        x = float(case["re"])
    else:
        x = complex(float(case["re"]), float(case["im"]))
    try:
        values = special.airye(x)
    except Exception as exc:
        print(f"case {case['case_id']} failed: {exc}", file=sys.stderr)
        sys.exit(3)
    points.append({
        "case_id": case["case_id"],
        "components": [component(value) for value in values],
    })

print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).map_err(|err| err.to_string())?;
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-")
        .env("FSCI_AIRYE_QUERY", query_json)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(err) => {
            if scipy_required() {
                return Err(format!("failed to spawn python3 for airye oracle: {err}"));
            }
            eprintln!("skipping airye oracle: python3 not available ({err})");
            return Ok(None);
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| "open airye oracle stdin".to_string())?;
        if let Err(err) = stdin.write_all(script.as_bytes()) {
            let output = child
                .wait_with_output()
                .map_err(|wait_err| wait_err.to_string())?;
            let stderr = String::from_utf8_lossy(&output.stderr);
            if scipy_required() {
                return Err(format!(
                    "airye oracle stdin write failed: {err}; stderr: {stderr}"
                ));
            }
            eprintln!("skipping airye oracle: stdin write failed ({err})\n{stderr}");
            return Ok(None);
        }
    }
    let output = child.wait_with_output().map_err(|err| err.to_string())?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if scipy_required() {
            return Err(format!("airye oracle failed: {stderr}"));
        }
        eprintln!("skipping airye oracle: scipy not available\n{stderr}");
        return Ok(None);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout)
        .map(Some)
        .map_err(|err| format!("parse airye oracle JSON: {err}; stdout: {stdout}"))
}

fn fsci_eval(case: &PointCase) -> Result<Vec<Complex64>, String> {
    let input = match case.im {
        Some(im) => SpecialTensor::ComplexScalar(Complex64::new(case.re, im)),
        None => SpecialTensor::RealScalar(case.re),
    };
    let result = airye(&input, RuntimeMode::Strict).map_err(|err| err.to_string())?;
    if result.len() != 4 {
        return Err(format!("expected 4 airye outputs, got {}", result.len()));
    }
    result
        .into_iter()
        .map(|value| match value {
            SpecialTensor::RealScalar(v) => Ok(Complex64::from_real(v)),
            SpecialTensor::ComplexScalar(v) => Ok(v),
            other => Err(format!("unexpected airye output: {other:?}")),
        })
        .collect()
}

fn component_pass(actual: Complex64, expected: &ComponentArm) -> (f64, bool) {
    let Some(expected_re) = expected.re else {
        return (0.0, actual.re.is_nan());
    };
    let Some(expected_im) = expected.im else {
        return (0.0, actual.im.is_nan());
    };
    let re_diff = (actual.re - expected_re).abs();
    let im_diff = (actual.im - expected_im).abs();
    let abs_diff = re_diff.max(im_diff);
    let scale = expected_re.abs().max(expected_im.abs()).max(1.0);
    (abs_diff, abs_diff <= ABS_TOL || abs_diff / scale <= REL_TOL)
}

#[test]
fn diff_special_airye() -> Result<(), String> {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query)? else {
        return Ok(());
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let mut max_abs_diff = 0.0_f64;
    let mut diffs = Vec::new();
    let component_names = ["eAi", "eAip", "eBi", "eBip"];

    for (case, expected) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(case.case_id, expected.case_id);
        assert_eq!(expected.components.len(), 4);
        let actual = fsci_eval(case)?;
        for ((name, actual_component), expected_component) in component_names
            .iter()
            .zip(actual.iter())
            .zip(expected.components.iter())
        {
            let (abs_diff, pass) = component_pass(*actual_component, expected_component);
            max_abs_diff = max_abs_diff.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.into(),
                component: (*name).into(),
                abs_diff,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|diff| diff.pass);
    let log = DiffLog {
        test_id: "diff_special_airye".into(),
        category: "fsci_special::airye vs scipy.special.airye".into(),
        case_count: diffs.len(),
        max_abs_diff,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log)?;

    for diff in &diffs {
        if !diff.pass {
            eprintln!(
                "airye mismatch: {} {} abs_diff={}",
                diff.case_id, diff.component, diff.abs_diff
            );
        }
    }

    if all_pass {
        Ok(())
    } else {
        Err(format!(
            "scipy.special.airye conformance failed: {} cases, max_diff={max_abs_diff}",
            diffs.len()
        ))
    }
}
