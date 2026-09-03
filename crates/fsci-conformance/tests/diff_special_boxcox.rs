#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the `scipy.special.boxcox` public alias.
//!
//! Resolves [frankenscipy-ghqwr]. The underlying Rust implementation previously
//! lived behind `boxcox_transform`; this test pins the SciPy spellings and edge
//! behavior for `x == 0`, infinities, and regular finite values.

use std::error::Error;
use std::io::{Error as IoError, Write};
use std::process::Stdio;

use fsci_special::boxcox_scalar;
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const BOXCOX_TOL: f64 = 2.0e-14;

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: String,
    lam: String,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct OraclePoint {
    case_id: String,
    kind: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

fn test_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(IoError::other(message.into()))
}

fn generate_query() -> OracleQuery {
    let cases: &[(&str, &str, &str)] = &[
        ("log_lambda_zero", "2.0", "0.0"),
        ("sqrt_lambda", "4.0", "0.5"),
        ("negative_lambda", "0.5", "-1.0"),
        ("identity_at_one", "1.0", "2.0"),
        ("zero_positive_lambda", "0.0", "0.5"),
        ("zero_lambda_zero", "0.0", "0.0"),
        ("negative_domain", "-1.0", "0.5"),
        ("positive_x_pos_inf", "2.0", "inf"),
        ("small_x_pos_inf", "0.5", "inf"),
        ("positive_x_neg_inf", "2.0", "-inf"),
        ("small_x_neg_inf", "0.5", "-inf"),
    ];
    OracleQuery {
        points: cases
            .iter()
            .map(|(case_id, x, lam)| PointCase {
                case_id: (*case_id).into(),
                x: (*x).into(),
                lam: (*lam).into(),
            })
            .collect(),
    }
}

fn parse_case_f64(value: &str) -> Result<f64, Box<dyn Error>> {
    match value {
        "inf" => Ok(f64::INFINITY),
        "-inf" => Ok(f64::NEG_INFINITY),
        _ => Ok(value.parse()?),
    }
}

fn value_kind(value: f64) -> (&'static str, Option<f64>) {
    if value.is_nan() {
        ("nan", None)
    } else if value == f64::INFINITY {
        ("pos_inf", None)
    } else if value == f64::NEG_INFINITY {
        ("neg_inf", None)
    } else if value == 0.0 && value.is_sign_negative() {
        ("neg_zero", Some(value))
    } else {
        ("finite", Some(value))
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Result<Option<OracleResult>, Box<dyn Error>> {
    let script = r#"
import json
import math
import sys
from scipy import special

def parse(v):
    if v == "inf":
        return math.inf
    if v == "-inf":
        return -math.inf
    return float(v)

def encode(v):
    v = float(v)
    if math.isnan(v):
        return {"kind": "nan", "value": None}
    if math.isinf(v):
        return {"kind": "pos_inf" if v > 0 else "neg_inf", "value": None}
    if v == 0.0 and math.copysign(1.0, v) < 0:
        return {"kind": "neg_zero", "value": v}
    return {"kind": "finite", "value": v}

q = json.loads(sys.argv[1])
points = []
for case in q["points"]:
    encoded = encode(special.boxcox(parse(case["x"]), parse(case["lam"])))
    encoded["case_id"] = case["case_id"]
    points.append(encoded)
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query)?;
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-")
        .arg(query_json)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
                return Err(test_error(format!("failed to spawn scipy oracle: {e}")));
            }
            eprintln!("skipping boxcox oracle: python3 not available ({e})");
            return Ok(None);
        }
    };

    let stdin = child
        .stdin
        .as_mut()
        .ok_or_else(|| test_error("open boxcox oracle stdin"))?;
    if let Err(err) = stdin.write_all(script.as_bytes()) {
        let output = child.wait_with_output()?;
        let stderr = String::from_utf8_lossy(&output.stderr);
        if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
            return Err(test_error(format!(
                "boxcox oracle stdin write failed: {err}; stderr: {stderr}"
            )));
        }
        eprintln!("skipping boxcox oracle: stdin write failed ({err})\n{stderr}");
        return Ok(None);
    }

    let output = child.wait_with_output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
            return Err(test_error(format!("boxcox oracle failed: {stderr}")));
        }
        eprintln!("skipping boxcox oracle: scipy not available\n{stderr}");
        return Ok(None);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(Some(serde_json::from_str(&stdout)?))
}

#[test]
fn diff_special_boxcox() -> Result<(), Box<dyn Error>> {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query)? else {
        return Ok(());
    };
    if oracle.points.len() != query.points.len() {
        return Err(test_error(format!(
            "boxcox oracle returned {} points for {} queries",
            oracle.points.len(),
            query.points.len()
        )));
    }

    for (case, oracle) in query.points.iter().zip(oracle.points.iter()) {
        if case.case_id != oracle.case_id {
            return Err(test_error(format!(
                "boxcox oracle order mismatch: {} vs {}",
                case.case_id, oracle.case_id
            )));
        }
        let actual = boxcox_scalar(parse_case_f64(&case.x)?, parse_case_f64(&case.lam)?);
        let (actual_kind, actual_value) = value_kind(actual);
        if actual_kind != oracle.kind {
            return Err(test_error(format!(
                "{} kind mismatch: got {}, expected {}",
                case.case_id, actual_kind, oracle.kind
            )));
        }
        if let (Some(actual), Some(expected)) = (actual_value, oracle.value) {
            let scale = expected.abs().max(1.0);
            let abs_diff = (actual - expected).abs();
            if abs_diff > BOXCOX_TOL * scale {
                return Err(test_error(format!(
                    "{} value mismatch: got {actual}, expected {expected}, abs_diff={abs_diff}",
                    case.case_id
                )));
            }
        }
    }

    Ok(())
}
