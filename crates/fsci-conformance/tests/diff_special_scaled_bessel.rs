#![forbid(unsafe_code)]
//! Live SciPy differential coverage for scaled cylindrical Bessel/Hankel callables.
//!
//! Resolves [frankenscipy-gofn4]. These SciPy public spellings are thin scaled
//! variants over the existing Bessel and Hankel kernels, so the oracle covers
//! representative real and complex inputs for `jve`, `yve`, `hankel1e`, and
//! `hankel2e`.

use std::error::Error;
use std::io::{Error as IoError, Write};
use std::process::Stdio;

use fsci_runtime::RuntimeMode;
use fsci_special::types::{Complex64, SpecialTensor};
use fsci_special::{hankel1e, hankel2e, jve, yve};
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const SCALED_BESSEL_TOL: f64 = 5.0e-8;

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    function: String,
    order: f64,
    z_re: f64,
    z_im: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct OraclePoint {
    case_id: String,
    re: f64,
    im: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

fn test_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(IoError::other(message.into()))
}

fn generate_query() -> OracleQuery {
    let cases: &[(&str, &str, f64, f64, f64)] = &[
        ("jve_real", "jve", 0.5, 1.25, 0.0),
        ("yve_real", "yve", 0.5, 1.25, 0.0),
        ("hankel1e_real", "hankel1e", 1.0, 1.25, 0.0),
        ("hankel2e_real", "hankel2e", 1.0, 1.25, 0.0),
        ("jve_complex", "jve", 0.5, 1.25, 0.5),
        ("yve_complex", "yve", 0.5, 1.25, 0.5),
        ("hankel1e_complex", "hankel1e", 0.5, 1.25, 0.5),
        ("hankel2e_complex", "hankel2e", 0.5, 1.25, 0.5),
    ];
    OracleQuery {
        points: cases
            .iter()
            .map(|(case_id, function, order, z_re, z_im)| PointCase {
                case_id: (*case_id).into(),
                function: (*function).into(),
                order: *order,
                z_re: *z_re,
                z_im: *z_im,
            })
            .collect(),
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Result<Option<OracleResult>, Box<dyn Error>> {
    let script = r#"
import json
import sys
from scipy import special

q = json.loads(sys.argv[1])
points = []
for case in q["points"]:
    z = complex(case["z_re"], case["z_im"])
    if case["z_im"] == 0.0:
        z = case["z_re"]
    value = complex(getattr(special, case["function"])(case["order"], z))
    points.append({"case_id": case["case_id"], "re": value.real, "im": value.imag})
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
            eprintln!("skipping scaled Bessel oracle: python3 not available ({e})");
            return Ok(None);
        }
    };

    let stdin = child
        .stdin
        .as_mut()
        .ok_or_else(|| test_error("open scaled Bessel oracle stdin"))?;
    if let Err(err) = stdin.write_all(script.as_bytes()) {
        let output = child.wait_with_output()?;
        let stderr = String::from_utf8_lossy(&output.stderr);
        if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
            return Err(test_error(format!(
                "scaled Bessel oracle stdin write failed: {err}; stderr: {stderr}"
            )));
        }
        eprintln!("skipping scaled Bessel oracle: stdin write failed ({err})\n{stderr}");
        return Ok(None);
    }

    let output = child.wait_with_output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
            return Err(test_error(format!("scaled Bessel oracle failed: {stderr}")));
        }
        eprintln!("skipping scaled Bessel oracle: scipy not available\n{stderr}");
        return Ok(None);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(Some(serde_json::from_str(&stdout)?))
}

fn eval_case(case: &PointCase) -> Result<Complex64, Box<dyn Error>> {
    let order = SpecialTensor::RealScalar(case.order);
    let z = if case.z_im == 0.0 {
        SpecialTensor::RealScalar(case.z_re)
    } else {
        SpecialTensor::ComplexScalar(Complex64::new(case.z_re, case.z_im))
    };
    let output = match case.function.as_str() {
        "jve" => jve(&order, &z, RuntimeMode::Strict),
        "yve" => yve(&order, &z, RuntimeMode::Strict),
        "hankel1e" => hankel1e(&order, &z, RuntimeMode::Strict),
        "hankel2e" => hankel2e(&order, &z, RuntimeMode::Strict),
        other => return Err(test_error(format!("unknown function {other}"))),
    }
    .map_err(|err| test_error(err.to_string()))?;

    match output {
        SpecialTensor::RealScalar(value) => Ok(Complex64::new(value, 0.0)),
        SpecialTensor::ComplexScalar(value) => Ok(value),
        other => Err(test_error(format!("unexpected output tensor {other:?}"))),
    }
}

#[test]
fn diff_special_scaled_bessel() -> Result<(), Box<dyn Error>> {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query)? else {
        return Ok(());
    };
    if oracle.points.len() != query.points.len() {
        return Err(test_error(format!(
            "scaled Bessel oracle returned {} points for {} queries",
            oracle.points.len(),
            query.points.len()
        )));
    }

    for (case, expected) in query.points.iter().zip(oracle.points.iter()) {
        if case.case_id != expected.case_id {
            return Err(test_error(format!(
                "scaled Bessel oracle order mismatch: {} vs {}",
                case.case_id, expected.case_id
            )));
        }
        let actual = eval_case(case)?;
        let expected = Complex64::new(expected.re, expected.im);
        let scale = expected.abs().max(1.0);
        let diff = (actual - expected).abs();
        if diff > SCALED_BESSEL_TOL * scale {
            return Err(test_error(format!(
                "{} mismatch: got ({}, {}), expected ({}, {}), abs_diff={diff}",
                case.case_id, actual.re, actual.im, expected.re, expected.im
            )));
        }
    }

    Ok(())
}
