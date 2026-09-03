#![forbid(unsafe_code)]
//! Live SciPy differential coverage for scipy.interpolate functions.
//!
//! Tests FrankenSciPy interpolation functions against SciPy subprocess oracle
//! across deterministic input families.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_interpolate::{
    NearestNDInterpolator, RbfInterpolator, RbfKernel, interp1d_linear, lagrange, polyfit, polyval,
    splev, splrep,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-016";
const INTERP_TOL: f64 = 1.0e-8;
const POLY_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Interp1dCase {
    case_id: String,
    x: Vec<f64>,
    y: Vec<f64>,
    x_new: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct LagrangeCase {
    case_id: String,
    xi: Vec<f64>,
    yi: Vec<f64>,
    x_eval: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PolyfitCase {
    case_id: String,
    x: Vec<f64>,
    y: Vec<f64>,
    deg: usize,
    x_eval: f64,
}

#[derive(Debug, Clone, Serialize)]
struct SplineCase {
    case_id: String,
    x: Vec<f64>,
    y: Vec<f64>,
    k: usize,
    x_eval: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct RbfCase {
    case_id: String,
    points: Vec<Vec<f64>>,
    values: Vec<f64>,
    queries: Vec<Vec<f64>>,
    kernel: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    epsilon: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    degree: Option<i32>,
    tolerance: f64,
}

#[derive(Debug, Deserialize)]
struct RbfOracleResult {
    case_id: String,
    status: String,
    result_kind: String,
    result: RbfOracleValues,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RbfOracleValues {
    values: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    case_id: String,
    values: Option<Vec<f64>>,
    value: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    method: String,
    rust_values: Vec<f64>,
    scipy_values: Vec<f64>,
    max_diff: f64,
    tolerance: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    tolerance: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create interpolate diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize interpolate diff log");
    fs::write(path, json).expect("write interpolate diff log");
}

fn assert_complete_oracle_results<T>(
    test_id: &str,
    expected_case_ids: impl IntoIterator<Item = String>,
    oracle_results: &HashMap<String, T>,
) {
    let expected_case_ids: Vec<String> = expected_case_ids.into_iter().collect();
    assert_eq!(
        oracle_results.len(),
        expected_case_ids.len(),
        "{test_id} SciPy oracle returned partial or duplicate coverage"
    );

    let missing_oracle_cases: Vec<&str> = expected_case_ids
        .iter()
        .filter(|case_id| !oracle_results.contains_key(case_id.as_str()))
        .map(String::as_str)
        .collect();
    assert!(
        missing_oracle_cases.is_empty(),
        "{test_id} missing SciPy interpolate oracle results: {:?}",
        missing_oracle_cases
    );

    let unexpected_oracle_cases: Vec<&str> = oracle_results
        .keys()
        .map(String::as_str)
        .filter(|case_id| !expected_case_ids.iter().any(|expected| expected == case_id))
        .collect();
    assert!(
        unexpected_oracle_cases.is_empty(),
        "{test_id} unexpected SciPy interpolate oracle results: {:?}",
        unexpected_oracle_cases
    );
}

fn assert_all_cases_compared(test_id: &str, compared: usize, expected: usize) {
    assert_eq!(
        compared, expected,
        "{test_id} compared {compared} of {expected} cases"
    );
}

fn interp1d_cases() -> Vec<Interp1dCase> {
    vec![
        Interp1dCase {
            case_id: "linear_simple".into(),
            x: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            y: vec![0.0, 1.0, 4.0, 9.0, 16.0],
            x_new: vec![0.5, 1.5, 2.5, 3.5],
        },
        Interp1dCase {
            case_id: "linear_sine".into(),
            x: (0..=10).map(|i| i as f64 * 0.5).collect(),
            y: (0..=10).map(|i| (i as f64 * 0.5).sin()).collect(),
            x_new: vec![0.25, 0.75, 1.25, 2.5, 3.75, 4.25],
        },
        Interp1dCase {
            case_id: "linear_exp".into(),
            x: vec![0.0, 0.5, 1.0, 1.5, 2.0],
            y: vec![1.0, 1.6487, std::f64::consts::E, 4.4817, 7.3891],
            x_new: vec![0.25, 0.75, 1.25, 1.75],
        },
    ]
}

fn lagrange_cases() -> Vec<LagrangeCase> {
    vec![
        LagrangeCase {
            case_id: "lagrange_quad".into(),
            xi: vec![0.0, 1.0, 2.0],
            yi: vec![0.0, 1.0, 4.0],
            x_eval: 1.5,
        },
        LagrangeCase {
            case_id: "lagrange_cubic".into(),
            xi: vec![0.0, 1.0, 2.0, 3.0],
            yi: vec![0.0, 1.0, 8.0, 27.0],
            x_eval: 1.5,
        },
        LagrangeCase {
            case_id: "lagrange_sin".into(),
            xi: vec![0.0, 1.0, 2.0, 3.0],
            yi: vec![0.0, 0.8415, 0.9093, 0.1411],
            x_eval: 0.5,
        },
    ]
}

fn polyfit_cases() -> Vec<PolyfitCase> {
    vec![
        PolyfitCase {
            case_id: "polyfit_linear".into(),
            x: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            y: vec![1.0, 3.0, 5.0, 7.0, 9.0],
            deg: 1,
            x_eval: 2.5,
        },
        PolyfitCase {
            case_id: "polyfit_quad".into(),
            x: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            y: vec![0.0, 1.0, 4.0, 9.0, 16.0],
            deg: 2,
            x_eval: 2.5,
        },
        PolyfitCase {
            case_id: "polyfit_cubic".into(),
            x: vec![-1.0, 0.0, 1.0, 2.0],
            y: vec![-1.0, 0.0, 1.0, 8.0],
            deg: 3,
            x_eval: 0.5,
        },
    ]
}

fn spline_cases() -> Vec<SplineCase> {
    vec![
        SplineCase {
            case_id: "spline_cubic_sin".into(),
            x: (0..=8).map(|i| i as f64 * 0.5).collect(),
            y: (0..=8).map(|i| (i as f64 * 0.5).sin()).collect(),
            k: 3,
            x_eval: vec![0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75],
        },
        SplineCase {
            case_id: "spline_cubic_poly".into(),
            x: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            y: vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0],
            k: 3,
            x_eval: vec![0.5, 1.5, 2.5, 3.5, 4.5],
        },
    ]
}

fn rbf_cases() -> Vec<RbfCase> {
    let points: Vec<Vec<f64>> = (0..9).map(|i| vec![i as f64 / 8.0]).collect();
    let values: Vec<f64> = points.iter().map(|point| 2.0 * point[0] + 1.0).collect();
    let queries = vec![vec![1.5], vec![2.0], vec![3.0]];
    vec![
        RbfCase {
            case_id: "linear_default_constant_tail".into(),
            points: points.clone(),
            values: values.clone(),
            queries: queries.clone(),
            kernel: "linear".into(),
            epsilon: None,
            degree: None,
            tolerance: 1.0e-10,
        },
        RbfCase {
            case_id: "thin_plate_default_linear_tail".into(),
            points: points.clone(),
            values: values.clone(),
            queries: queries.clone(),
            kernel: "thin_plate_spline".into(),
            epsilon: None,
            degree: None,
            tolerance: 1.0e-10,
        },
        RbfCase {
            case_id: "thin_plate_explicit_degree_minus_one".into(),
            points,
            values,
            queries,
            kernel: "thin_plate_spline".into(),
            epsilon: None,
            degree: Some(-1),
            tolerance: 1.0e-5,
        },
    ]
}

fn rbf_kernel(name: &str) -> RbfKernel {
    match name {
        "linear" => RbfKernel::Linear,
        "thin_plate_spline" => RbfKernel::ThinPlateSpline,
        _ => unreachable!("RBF fixture uses a supported kernel"),
    }
}

fn run_scipy_rbf_oracle(case: &RbfCase) -> RbfOracleResult {
    let oracle =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python_oracle/scipy_interpolate_oracle.py");
    let mut child = Command::new("python3")
        .arg(oracle)
        .arg("--rbf-live")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn live SciPy RBF oracle");
    child
        .stdin
        .take()
        .expect("access RBF oracle stdin")
        .write_all(
            format!(
                "{}\n",
                serde_json::to_string(case).expect("serialize RBF case")
            )
            .as_bytes(),
        )
        .expect("write RBF oracle case");
    let output = child
        .wait_with_output()
        .expect("wait for live SciPy RBF oracle");
    assert!(
        output.status.success(),
        "live SciPy RBF oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("RBF oracle emitted UTF-8");
    let mut lines = stdout.lines();
    let ready = lines.next().expect("RBF oracle READY line");
    assert!(
        ready.contains("genuine=True"),
        "RBF oracle must use genuine SciPy: {ready}"
    );
    let response = lines.next().expect("RBF oracle response");
    assert!(
        lines.next().is_none(),
        "unexpected RBF oracle output: {stdout}"
    );
    serde_json::from_str(response).expect("parse live SciPy RBF response")
}

fn run_scipy_interp1d_oracle(cases: &[Interp1dCase]) -> HashMap<String, Vec<f64>> {
    let python_code = r#"
import sys
import json
import numpy as np
from scipy import interpolate

cases = json.loads(sys.stdin.read())
results = []
for case in cases:
    x = np.array(case['x'])
    y = np.array(case['y'])
    x_new = np.array(case['x_new'])
    try:
        f = interpolate.interp1d(x, y, kind='linear')
        vals = f(x_new).tolist()
        results.append({'case_id': case['case_id'], 'values': vals, 'value': None})
    except Exception:
        results.append({'case_id': case['case_id'], 'values': None, 'value': None})
print(json.dumps(results))
"#;

    let json_input = serde_json::to_string(cases).expect("serialize interp1d cases");
    let mut child = Command::new("python3")
        .args(["-c", python_code])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn python3 oracle");

    child
        .stdin
        .take()
        .expect("access stdin")
        .write_all(json_input.as_bytes())
        .expect("write to stdin");

    let output = child.wait_with_output().expect("wait for python3");
    if !output.status.success() {
        eprintln!(
            "scipy interp1d oracle stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return HashMap::new();
    }

    let results: Vec<OracleResult> =
        serde_json::from_slice(&output.stdout).expect("parse oracle output");

    results
        .into_iter()
        .filter_map(|r| r.values.map(|v| (r.case_id, v)))
        .collect()
}

fn run_scipy_lagrange_oracle(cases: &[LagrangeCase]) -> HashMap<String, f64> {
    let python_code = r#"
import sys
import json
import numpy as np
from scipy import interpolate

cases = json.loads(sys.stdin.read())
results = []
for case in cases:
    xi = np.array(case['xi'])
    yi = np.array(case['yi'])
    x_eval = case['x_eval']
    try:
        poly = interpolate.lagrange(xi, yi)
        val = float(np.polyval(poly, x_eval))
        results.append({'case_id': case['case_id'], 'values': None, 'value': val})
    except Exception:
        results.append({'case_id': case['case_id'], 'values': None, 'value': None})
print(json.dumps(results))
"#;

    let json_input = serde_json::to_string(cases).expect("serialize lagrange cases");
    let mut child = Command::new("python3")
        .args(["-c", python_code])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn python3 oracle");

    child
        .stdin
        .take()
        .expect("access stdin")
        .write_all(json_input.as_bytes())
        .expect("write to stdin");

    let output = child.wait_with_output().expect("wait for python3");
    if !output.status.success() {
        eprintln!(
            "scipy lagrange oracle stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return HashMap::new();
    }

    let results: Vec<OracleResult> =
        serde_json::from_slice(&output.stdout).expect("parse oracle output");

    results
        .into_iter()
        .filter_map(|r| r.value.map(|v| (r.case_id, v)))
        .collect()
}

fn run_scipy_polyfit_oracle(cases: &[PolyfitCase]) -> HashMap<String, f64> {
    let python_code = r#"
import sys
import json
import numpy as np

cases = json.loads(sys.stdin.read())
results = []
for case in cases:
    x = np.array(case['x'])
    y = np.array(case['y'])
    deg = case['deg']
    x_eval = case['x_eval']
    try:
        coeffs = np.polyfit(x, y, deg)
        val = float(np.polyval(coeffs, x_eval))
        results.append({'case_id': case['case_id'], 'values': None, 'value': val})
    except Exception:
        results.append({'case_id': case['case_id'], 'values': None, 'value': None})
print(json.dumps(results))
"#;

    let json_input = serde_json::to_string(cases).expect("serialize polyfit cases");
    let mut child = Command::new("python3")
        .args(["-c", python_code])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn python3 oracle");

    child
        .stdin
        .take()
        .expect("access stdin")
        .write_all(json_input.as_bytes())
        .expect("write to stdin");

    let output = child.wait_with_output().expect("wait for python3");
    if !output.status.success() {
        eprintln!(
            "scipy polyfit oracle stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return HashMap::new();
    }

    let results: Vec<OracleResult> =
        serde_json::from_slice(&output.stdout).expect("parse oracle output");

    results
        .into_iter()
        .filter_map(|r| r.value.map(|v| (r.case_id, v)))
        .collect()
}

fn run_scipy_spline_oracle(cases: &[SplineCase]) -> HashMap<String, Vec<f64>> {
    let python_code = r#"
import sys
import json
import numpy as np
from scipy import interpolate

cases = json.loads(sys.stdin.read())
results = []
for case in cases:
    x = np.array(case['x'])
    y = np.array(case['y'])
    k = case['k']
    x_eval = np.array(case['x_eval'])
    try:
        tck = interpolate.splrep(x, y, k=k, s=0)
        vals = interpolate.splev(x_eval, tck).tolist()
        results.append({'case_id': case['case_id'], 'values': vals, 'value': None})
    except Exception as e:
        results.append({'case_id': case['case_id'], 'values': None, 'value': None})
print(json.dumps(results))
"#;

    let json_input = serde_json::to_string(cases).expect("serialize spline cases");
    let mut child = Command::new("python3")
        .args(["-c", python_code])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn python3 oracle");

    child
        .stdin
        .take()
        .expect("access stdin")
        .write_all(json_input.as_bytes())
        .expect("write to stdin");

    let output = child.wait_with_output().expect("wait for python3");
    if !output.status.success() {
        eprintln!(
            "scipy spline oracle stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return HashMap::new();
    }

    let results: Vec<OracleResult> =
        serde_json::from_slice(&output.stdout).expect("parse oracle output");

    results
        .into_iter()
        .filter_map(|r| r.values.map(|v| (r.case_id, v)))
        .collect()
}

#[test]
fn diff_interp1d_linear() {
    let start = Instant::now();
    let cases = interp1d_cases();
    let scipy_results = run_scipy_interp1d_oracle(&cases);

    if scipy_results.is_empty() && std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
        panic!("scipy oracle required but not available");
    }
    if scipy_results.is_empty() {
        eprintln!("skipping interp1d diff: scipy oracle not available");
        return;
    }
    assert_complete_oracle_results(
        "interp1d_linear",
        cases.iter().map(|case| case.case_id.clone()),
        &scipy_results,
    );

    let mut diffs = Vec::new();
    let mut max_diff = 0.0f64;
    let mut all_pass = true;

    for case in &cases {
        let Ok(rust_vals) = interp1d_linear(&case.x, &case.y, &case.x_new) else {
            continue;
        };
        let Some(scipy_vals) = scipy_results.get(&case.case_id) else {
            continue;
        };

        let case_max_diff = rust_vals
            .iter()
            .zip(scipy_vals.iter())
            .map(|(r, s)| (r - s).abs())
            .fold(0.0, f64::max);
        let pass = rust_vals.len() == scipy_vals.len() && case_max_diff <= INTERP_TOL;
        max_diff = max_diff.max(case_max_diff);
        all_pass = all_pass && pass;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            method: "interp1d_linear".into(),
            rust_values: rust_vals,
            scipy_values: scipy_vals.clone(),
            max_diff: case_max_diff,
            tolerance: INTERP_TOL,
            pass,
        });
    }
    all_pass = all_pass && diffs.len() == cases.len();

    let log = DiffLog {
        test_id: "interp1d_linear".into(),
        category: "interpolate".into(),
        case_count: diffs.len(),
        max_abs_diff: max_diff,
        tolerance: INTERP_TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs,
    };

    emit_log(&log);
    assert_all_cases_compared("interp1d_linear", log.case_count, cases.len());
    assert!(all_pass, "interp1d_linear diff failed: max_diff={max_diff}");
}

#[test]
fn diff_lagrange() {
    let start = Instant::now();
    let cases = lagrange_cases();
    let scipy_results = run_scipy_lagrange_oracle(&cases);

    if scipy_results.is_empty() && std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
        panic!("scipy oracle required but not available");
    }
    if scipy_results.is_empty() {
        eprintln!("skipping lagrange diff: scipy oracle not available");
        return;
    }
    assert_complete_oracle_results(
        "lagrange",
        cases.iter().map(|case| case.case_id.clone()),
        &scipy_results,
    );

    let mut diffs = Vec::new();
    let mut max_diff = 0.0f64;
    let mut all_pass = true;

    for case in &cases {
        if let Ok(coeffs) = lagrange(&case.xi, &case.yi) {
            let rust_val = polyval(&coeffs, case.x_eval);
            if let Some(&scipy_val) = scipy_results.get(&case.case_id) {
                let abs_diff = (rust_val - scipy_val).abs();
                let pass = abs_diff <= POLY_TOL;
                max_diff = max_diff.max(abs_diff);
                all_pass = all_pass && pass;
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    method: "lagrange".into(),
                    rust_values: vec![rust_val],
                    scipy_values: vec![scipy_val],
                    max_diff: abs_diff,
                    tolerance: POLY_TOL,
                    pass,
                });
            }
        }
    }
    all_pass = all_pass && diffs.len() == cases.len();

    let log = DiffLog {
        test_id: "lagrange".into(),
        category: "interpolate".into(),
        case_count: diffs.len(),
        max_abs_diff: max_diff,
        tolerance: POLY_TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs,
    };

    emit_log(&log);
    assert_all_cases_compared("lagrange", log.case_count, cases.len());
    assert!(all_pass, "lagrange diff failed: max_diff={max_diff}");
}

#[test]
fn diff_polyfit() {
    let start = Instant::now();
    let cases = polyfit_cases();
    let scipy_results = run_scipy_polyfit_oracle(&cases);

    if scipy_results.is_empty() && std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
        panic!("scipy oracle required but not available");
    }
    if scipy_results.is_empty() {
        eprintln!("skipping polyfit diff: scipy oracle not available");
        return;
    }
    assert_complete_oracle_results(
        "polyfit",
        cases.iter().map(|case| case.case_id.clone()),
        &scipy_results,
    );

    let mut diffs = Vec::new();
    let mut max_diff = 0.0f64;
    let mut all_pass = true;

    for case in &cases {
        if let Ok(coeffs) = polyfit(&case.x, &case.y, case.deg) {
            let rust_val = polyval(&coeffs, case.x_eval);
            if let Some(&scipy_val) = scipy_results.get(&case.case_id) {
                let abs_diff = (rust_val - scipy_val).abs();
                let pass = abs_diff <= POLY_TOL;
                max_diff = max_diff.max(abs_diff);
                all_pass = all_pass && pass;
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    method: "polyfit".into(),
                    rust_values: vec![rust_val],
                    scipy_values: vec![scipy_val],
                    max_diff: abs_diff,
                    tolerance: POLY_TOL,
                    pass,
                });
            }
        }
    }
    all_pass = all_pass && diffs.len() == cases.len();

    let log = DiffLog {
        test_id: "polyfit".into(),
        category: "interpolate".into(),
        case_count: diffs.len(),
        max_abs_diff: max_diff,
        tolerance: POLY_TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs,
    };

    emit_log(&log);
    assert_all_cases_compared("polyfit", log.case_count, cases.len());
    assert!(all_pass, "polyfit diff failed: max_diff={max_diff}");
}

#[test]
fn diff_spline() {
    let start = Instant::now();
    let cases = spline_cases();
    let scipy_results = run_scipy_spline_oracle(&cases);

    if scipy_results.is_empty() && std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
        panic!("scipy oracle required but not available");
    }
    if scipy_results.is_empty() {
        eprintln!("skipping spline diff: scipy oracle not available");
        return;
    }
    assert_complete_oracle_results(
        "spline",
        cases.iter().map(|case| case.case_id.clone()),
        &scipy_results,
    );

    let mut diffs = Vec::new();
    let mut max_diff = 0.0f64;
    let mut all_pass = true;

    for case in &cases {
        let Ok(tck) = splrep(&case.x, &case.y, case.k, 0.0) else {
            continue;
        };
        let Ok(rust_vals) = splev(&case.x_eval, &tck) else {
            continue;
        };
        let Some(scipy_vals) = scipy_results.get(&case.case_id) else {
            continue;
        };

        let case_max_diff = rust_vals
            .iter()
            .zip(scipy_vals.iter())
            .map(|(r, s)| (r - s).abs())
            .fold(0.0, f64::max);
        let pass = rust_vals.len() == scipy_vals.len() && case_max_diff <= POLY_TOL;
        max_diff = max_diff.max(case_max_diff);
        all_pass = all_pass && pass;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            method: "splrep+splev".into(),
            rust_values: rust_vals,
            scipy_values: scipy_vals.clone(),
            max_diff: case_max_diff,
            tolerance: POLY_TOL,
            pass,
        });
    }
    all_pass = all_pass && diffs.len() == cases.len();

    let log = DiffLog {
        test_id: "spline".into(),
        category: "interpolate".into(),
        case_count: diffs.len(),
        max_abs_diff: max_diff,
        tolerance: POLY_TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs,
    };

    emit_log(&log);
    assert_all_cases_compared("spline", log.case_count, cases.len());
    assert!(all_pass, "spline diff failed: max_diff={max_diff}");
}

#[test]
fn diff_rbf_interpolator() {
    let cases = rbf_cases();
    let mut compared = 0;
    for case in &cases {
        let kernel = rbf_kernel(&case.kernel);
        let rust = match case.degree {
            Some(degree) => RbfInterpolator::with_degree(
                &case.points,
                &case.values,
                kernel,
                case.epsilon.unwrap_or(1.0),
                degree,
            ),
            None => RbfInterpolator::new(
                &case.points,
                &case.values,
                kernel,
                case.epsilon.unwrap_or(1.0),
            ),
        }
        .expect("valid RBF fixture");
        let rust_values: Vec<f64> = case.queries.iter().map(|query| rust.eval(query)).collect();
        let scipy = run_scipy_rbf_oracle(case);
        assert_eq!(scipy.case_id, case.case_id, "oracle case identity");
        assert_eq!(scipy.status, "ok", "SciPy RBF result: {:?}", scipy.error);
        assert_eq!(scipy.result_kind, "vector", "SciPy RBF result kind");
        assert_eq!(
            rust_values.len(),
            scipy.result.values.len(),
            "{} output length",
            case.case_id
        );
        let max_abs_diff = rust_values
            .iter()
            .zip(&scipy.result.values)
            .map(|(rust, scipy)| (rust - scipy).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs_diff <= case.tolerance,
            "{} differs from live SciPy by {max_abs_diff:.3e}, tolerance {:.3e}; rust={rust_values:?}, scipy={:?}",
            case.case_id,
            case.tolerance,
            scipy.result.values,
        );
        compared += 1;
    }
    assert_all_cases_compared("rbf_interpolator", compared, cases.len());
}

/// Differential coverage for scipy.interpolate.NearestNDInterpolator.
///
/// The expected values are the SciPy nearest-site result for this fixture. Queries include an
/// exterior point because nearest interpolation extrapolates, unlike linear interpolation. The
/// first-value negative control ensures a stub that ignores query geometry cannot pass.
#[test]
fn diff_nearest_nd_interpolator() {
    let points = vec![
        vec![0.0, 0.0],
        vec![10.0, 0.0],
        vec![0.0, 10.0],
        vec![10.0, 10.0],
    ];
    let values = vec![1.0, 2.0, 3.0, 4.0];
    let queries = vec![
        vec![0.1, 0.2],
        vec![9.8, 0.3],
        vec![0.2, 9.7],
        vec![9.9, 9.9],
        vec![25.0, -4.0],
    ];
    // scipy.interpolate.NearestNDInterpolator(points, values)(queries)
    let scipy_expected = [1.0, 2.0, 3.0, 4.0, 2.0];
    let interpolator = NearestNDInterpolator::new(&points, &values).expect("valid fixture");
    let observed = interpolator.eval_many(&queries).expect("queries succeed");
    assert_eq!(observed, scipy_expected);

    // A naive implementation that always returns values[0] fails this negative control.
    let naive = vec![values[0]; queries.len()];
    assert_ne!(naive, scipy_expected);
}
