#![forbid(unsafe_code)]
//! Live SciPy differential coverage for scipy.optimize functions.
//!
//! Tests FrankenSciPy optimization functions against SciPy subprocess oracle
//! across deterministic input families.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::{
    MinimizeScalarOptions, RootOptions, bisect, brenth, brentq, golden, minimize_scalar, ridder,
    toms748,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-015";
const ROOT_TOL: f64 = 1.0e-10;
const MIN_TOL: f64 = 1.0e-4;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct RootCase {
    case_id: String,
    method: String,
    func_id: String,
    a: f64,
    b: f64,
}

#[derive(Debug, Clone, Serialize)]
struct MinimizeCase {
    case_id: String,
    method: String,
    func_id: String,
    bracket_a: f64,
    bracket_b: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    case_id: String,
    root: Option<f64>,
    minimum: Option<f64>,
    #[allow(dead_code)]
    fval: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    method: String,
    rust_value: f64,
    scipy_value: f64,
    abs_diff: f64,
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
    fs::create_dir_all(output_dir()).expect("create opt diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize opt diff log");
    fs::write(path, json).expect("write opt diff log");
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
        "{test_id} missing SciPy optimize oracle results: {:?}",
        missing_oracle_cases
    );

    let unexpected_oracle_cases: Vec<&str> = oracle_results
        .keys()
        .map(String::as_str)
        .filter(|case_id| !expected_case_ids.iter().any(|expected| expected == case_id))
        .collect();
    assert!(
        unexpected_oracle_cases.is_empty(),
        "{test_id} unexpected SciPy optimize oracle results: {:?}",
        unexpected_oracle_cases
    );
}

fn assert_all_cases_compared(test_id: &str, compared: usize, expected: usize) {
    assert_eq!(
        compared, expected,
        "{test_id} compared {compared} of {expected} cases"
    );
}

fn eval_func(func_id: &str, x: f64) -> f64 {
    match func_id {
        "poly_cubic" => x * x * x - x - 2.0,
        "sin_x" => x.sin(),
        "exp_shift" => (x - 1.0).exp() - 2.0,
        "quad_shift" => (x - 0.5) * (x - 0.5) - 1.0,
        "parabola" => x * x - 4.0 * x + 3.0,
        "rosenbrock_1d" => (1.0 - x).powi(2) + 100.0 * (1.0 - x * x).powi(2),
        "quartic" => (x - 2.0).powi(4) + 1.0,
        _ => f64::NAN,
    }
}

fn root_cases() -> Vec<RootCase> {
    let methods = ["bisect", "brentq", "brenth", "ridder", "toms748"];
    let functions = [
        ("poly_cubic", 1.0, 2.0),
        ("sin_x", 2.0, 4.0),
        ("exp_shift", 0.0, 2.0),
        ("quad_shift", -1.0, 1.0),
    ];

    let mut cases = Vec::new();
    for method in methods {
        for (func_id, a, b) in &functions {
            cases.push(RootCase {
                case_id: format!("{method}_{func_id}"),
                method: method.into(),
                func_id: func_id.to_string(),
                a: *a,
                b: *b,
            });
        }
    }
    cases
}

fn minimize_cases() -> Vec<MinimizeCase> {
    let methods = ["brent", "golden"];
    let functions = [("parabola", 0.0, 4.0), ("quartic", 0.0, 5.0)];

    let mut cases = Vec::new();
    for method in methods {
        for (func_id, a, b) in &functions {
            cases.push(MinimizeCase {
                case_id: format!("minimize_{method}_{func_id}"),
                method: method.into(),
                func_id: func_id.to_string(),
                bracket_a: *a,
                bracket_b: *b,
            });
        }
    }
    cases
}

fn run_scipy_root_oracle(cases: &[RootCase]) -> HashMap<String, f64> {
    let python_code = r#"
import sys
import json
from scipy import optimize

def eval_func(func_id, x):
    if func_id == "poly_cubic":
        return x**3 - x - 2.0
    elif func_id == "sin_x":
        return float(__import__('math').sin(x))
    elif func_id == "exp_shift":
        return float(__import__('math').exp(x - 1.0)) - 2.0
    elif func_id == "quad_shift":
        return (x - 0.5)**2 - 1.0
    else:
        return float('nan')

cases = json.loads(sys.stdin.read())
results = []
for case in cases:
    method = case['method']
    func_id = case['func_id']
    a = case['a']
    b = case['b']
    f = lambda x: eval_func(func_id, x)
    try:
        if method == 'bisect':
            root = optimize.bisect(f, a, b)
        elif method == 'brentq':
            root = optimize.brentq(f, a, b)
        elif method == 'brenth':
            root = optimize.brenth(f, a, b)
        elif method == 'ridder':
            root = optimize.ridder(f, a, b)
        elif method == 'toms748':
            root = optimize.toms748(f, a, b)
        else:
            root = float('nan')
        results.append({'case_id': case['case_id'], 'root': root, 'minimum': None, 'fval': None})
    except Exception:
        results.append({'case_id': case['case_id'], 'root': float('nan'), 'minimum': None, 'fval': None})
print(json.dumps(results))
"#;

    let json_input = serde_json::to_string(cases).expect("serialize root cases");
    let mut child = fsci_conformance::scipy_oracle_command()
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
            "scipy oracle stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return HashMap::new();
    }

    let results: Vec<OracleResult> =
        serde_json::from_slice(&output.stdout).expect("parse oracle output");

    results
        .into_iter()
        .filter_map(|r| r.root.map(|v| (r.case_id, v)))
        .collect()
}

fn run_scipy_minimize_oracle(cases: &[MinimizeCase]) -> HashMap<String, f64> {
    let python_code = r#"
import sys
import json
from scipy import optimize

def eval_func(func_id, x):
    if func_id == "parabola":
        return x**2 - 4.0*x + 3.0
    elif func_id == "quartic":
        return (x - 2.0)**4 + 1.0
    else:
        return float('nan')

cases = json.loads(sys.stdin.read())
results = []
for case in cases:
    method = case['method']
    func_id = case['func_id']
    a = case['bracket_a']
    b = case['bracket_b']
    f = lambda x, fid=func_id: eval_func(fid, x)
    try:
        res = optimize.minimize_scalar(f, bracket=(a, b), method=method)
        results.append({'case_id': case['case_id'], 'root': None, 'minimum': res.x, 'fval': res.fun})
    except Exception:
        results.append({'case_id': case['case_id'], 'root': None, 'minimum': None, 'fval': None})
print(json.dumps(results))
"#;

    let json_input = serde_json::to_string(cases).expect("serialize minimize cases");
    let mut child = fsci_conformance::scipy_oracle_command()
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
            "scipy oracle stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return HashMap::new();
    }

    let results: Vec<OracleResult> =
        serde_json::from_slice(&output.stdout).expect("parse oracle output");

    results
        .into_iter()
        .filter_map(|r| r.minimum.map(|v| (r.case_id, v)))
        .collect()
}

fn run_rust_root(case: &RootCase) -> f64 {
    let f = |x: f64| eval_func(&case.func_id, x);
    let bracket = (case.a, case.b);
    let opts = RootOptions::default();
    match case.method.as_str() {
        "bisect" => bisect(f, bracket, opts).map_or(f64::NAN, |r| r.root),
        "brentq" => brentq(f, bracket, opts).map_or(f64::NAN, |r| r.root),
        "brenth" => brenth(f, bracket, opts).map_or(f64::NAN, |r| r.root),
        "ridder" => ridder(f, bracket, opts).map_or(f64::NAN, |r| r.root),
        "toms748" => toms748(f, bracket, opts).map_or(f64::NAN, |r| r.root),
        _ => f64::NAN,
    }
}

fn run_rust_minimize(case: &MinimizeCase) -> f64 {
    let f = |x: f64| eval_func(&case.func_id, x);
    match case.method.as_str() {
        "brent" => {
            let opts = MinimizeScalarOptions::default();
            minimize_scalar(f, (case.bracket_a, case.bracket_b), opts).map_or(f64::NAN, |r| r.x)
        }
        "golden" => {
            let (x, _fval) = golden(f, case.bracket_a, case.bracket_b, 1e-6, 500);
            x
        }
        _ => f64::NAN,
    }
}

#[test]
fn diff_root_finding() {
    let start = Instant::now();
    let cases = root_cases();
    let scipy_results = run_scipy_root_oracle(&cases);

    if scipy_results.is_empty() && std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
        panic!("scipy oracle required but not available");
    }
    if scipy_results.is_empty() {
        eprintln!("skipping root diff: scipy oracle not available");
        return;
    }
    assert_complete_oracle_results(
        "root_finding",
        cases.iter().map(|case| case.case_id.clone()),
        &scipy_results,
    );

    let mut diffs = Vec::new();
    let mut max_diff = 0.0f64;
    let mut all_pass = true;

    for case in &cases {
        let rust_val = run_rust_root(case);
        if let Some(&scipy_val) = scipy_results.get(&case.case_id) {
            let abs_diff = (rust_val - scipy_val).abs();
            let pass = abs_diff <= ROOT_TOL || (rust_val.is_nan() && scipy_val.is_nan());
            max_diff = max_diff.max(abs_diff);
            all_pass = all_pass && pass;
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                method: case.method.clone(),
                rust_value: rust_val,
                scipy_value: scipy_val,
                abs_diff,
                tolerance: ROOT_TOL,
                pass,
            });
        }
    }
    all_pass = all_pass && diffs.len() == cases.len();

    let log = DiffLog {
        test_id: "root_finding".into(),
        category: "optimize".into(),
        case_count: diffs.len(),
        max_abs_diff: max_diff,
        tolerance: ROOT_TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs,
    };

    emit_log(&log);
    assert_all_cases_compared("root_finding", log.case_count, cases.len());
    assert!(all_pass, "root finding diff failed: max_diff={max_diff}");
}

#[test]
fn diff_minimize_scalar() {
    let start = Instant::now();
    let cases = minimize_cases();
    let scipy_results = run_scipy_minimize_oracle(&cases);

    if scipy_results.is_empty() && std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
        panic!("scipy oracle required but not available");
    }
    if scipy_results.is_empty() {
        eprintln!("skipping minimize diff: scipy oracle not available");
        return;
    }
    assert_complete_oracle_results(
        "minimize_scalar",
        cases.iter().map(|case| case.case_id.clone()),
        &scipy_results,
    );

    let mut diffs = Vec::new();
    let mut max_diff = 0.0f64;
    let mut all_pass = true;

    for case in &cases {
        let rust_val = run_rust_minimize(case);
        if let Some(&scipy_val) = scipy_results.get(&case.case_id) {
            let abs_diff = (rust_val - scipy_val).abs();
            let pass = abs_diff <= MIN_TOL || (rust_val.is_nan() && scipy_val.is_nan());
            max_diff = max_diff.max(abs_diff);
            all_pass = all_pass && pass;
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                method: case.method.clone(),
                rust_value: rust_val,
                scipy_value: scipy_val,
                abs_diff,
                tolerance: MIN_TOL,
                pass,
            });
        }
    }
    all_pass = all_pass && diffs.len() == cases.len();

    let log = DiffLog {
        test_id: "minimize_scalar".into(),
        category: "optimize".into(),
        case_count: diffs.len(),
        max_abs_diff: max_diff,
        tolerance: MIN_TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs,
    };

    emit_log(&log);
    assert_all_cases_compared("minimize_scalar", log.case_count, cases.len());
    assert!(all_pass, "minimize_scalar diff failed: max_diff={max_diff}");
}
