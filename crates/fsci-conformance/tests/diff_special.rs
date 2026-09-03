#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.special` scalar evaluators.
//!
//! Tests FrankenSciPy orthogonal polynomial evaluators against SciPy's
//! `scipy.special.eval_*` reference functions via a subprocess oracle.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::orthopoly::{
    eval_chebyt, eval_chebyu, eval_gegenbauer, eval_genlaguerre, eval_hermite, eval_hermitenorm,
    eval_jacobi, eval_laguerre, eval_legendre,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-10;
const REL_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct SpecialCase {
    case_id: String,
    op: String,
    n: u32,
    alpha: Option<f64>,
    beta: Option<f64>,
    x: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    case_id: String,
    value: f64,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    n: u32,
    alpha: Option<f64>,
    beta: Option<f64>,
    x: f64,
    rust_value: f64,
    scipy_value: f64,
    abs_diff: f64,
    rel_diff: f64,
    tolerance_abs: f64,
    tolerance_rel: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    max_rel_diff: f64,
    tolerance_abs: f64,
    tolerance_rel: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create special diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize special diff log");
    fs::write(path, json).expect("write special diff log");
}

fn push_case(
    cases: &mut Vec<SpecialCase>,
    op: &str,
    n: u32,
    alpha: Option<f64>,
    beta: Option<f64>,
    x: f64,
) {
    let alpha_id = alpha.map_or_else(String::new, |value| format!("_a{value:.2}"));
    let beta_id = beta.map_or_else(String::new, |value| format!("_b{value:.2}"));
    cases.push(SpecialCase {
        case_id: format!("{op}_n{n}{alpha_id}{beta_id}_x{x:.2}"),
        op: op.to_string(),
        n,
        alpha,
        beta,
        x,
    });
}

fn special_cases() -> Vec<SpecialCase> {
    let mut cases = Vec::new();
    let compact_degrees = [0, 1, 2, 3, 5, 8];
    let compact_xs = [-0.75, -0.25, 0.0, 0.25, 0.75];

    for op in ["eval_legendre", "eval_chebyt", "eval_chebyu"] {
        for n in compact_degrees {
            for x in compact_xs {
                push_case(&mut cases, op, n, None, None, x);
            }
        }
    }

    for n in [0, 1, 2, 3, 5] {
        for x in [0.0, 0.5, 1.5, 3.0] {
            push_case(&mut cases, "eval_laguerre", n, None, None, x);
        }
    }

    for op in ["eval_hermite", "eval_hermitenorm"] {
        for n in [0, 1, 2, 3, 4] {
            for x in [-1.0, -0.25, 0.5] {
                push_case(&mut cases, op, n, None, None, x);
            }
        }
    }

    for n in [0, 1, 2, 4] {
        for alpha in [0.5, 2.0] {
            for x in [0.0, 0.75, 2.5] {
                push_case(&mut cases, "eval_genlaguerre", n, Some(alpha), None, x);
            }
        }
    }

    for n in [0, 1, 2, 4] {
        for alpha in [0.5, 1.5] {
            for x in [-0.5, 0.0, 0.5] {
                push_case(&mut cases, "eval_gegenbauer", n, Some(alpha), None, x);
            }
        }
    }

    for n in [0, 1, 2, 4] {
        for (alpha, beta) in [(0.0, 0.0), (0.5, 0.5), (1.0, 2.0)] {
            for x in [-0.5, 0.25] {
                push_case(&mut cases, "eval_jacobi", n, Some(alpha), Some(beta), x);
            }
        }
    }

    cases
}

fn rust_value(case: &SpecialCase) -> Option<f64> {
    match case.op.as_str() {
        "eval_legendre" => Some(eval_legendre(case.n, case.x)),
        "eval_chebyt" => Some(eval_chebyt(case.n, case.x)),
        "eval_chebyu" => Some(eval_chebyu(case.n, case.x)),
        "eval_laguerre" => Some(eval_laguerre(case.n, case.x)),
        "eval_hermite" => Some(eval_hermite(case.n, case.x)),
        "eval_hermitenorm" => Some(eval_hermitenorm(case.n, case.x)),
        "eval_genlaguerre" => Some(eval_genlaguerre(case.n, case.alpha?, case.x)),
        "eval_gegenbauer" => Some(eval_gegenbauer(case.n, case.alpha?, case.x)),
        "eval_jacobi" => Some(eval_jacobi(case.n, case.alpha?, case.beta?, case.x)),
        _ => None,
    }
}

fn run_scipy_oracle(cases: &[SpecialCase]) -> Option<HashMap<String, OracleResult>> {
    let python_code = r#"
import json
import math
import sys

try:
    from scipy import special
except Exception:
    sys.exit(42)

cases = json.loads(sys.stdin.read())
results = []
for case in cases:
    op = case["op"]
    n = int(case["n"])
    x = float(case["x"])
    alpha = case.get("alpha")
    beta = case.get("beta")
    try:
        if op == "eval_legendre":
            value = special.eval_legendre(n, x)
        elif op == "eval_chebyt":
            value = special.eval_chebyt(n, x)
        elif op == "eval_chebyu":
            value = special.eval_chebyu(n, x)
        elif op == "eval_laguerre":
            value = special.eval_laguerre(n, x)
        elif op == "eval_hermite":
            value = special.eval_hermite(n, x)
        elif op == "eval_hermitenorm":
            value = special.eval_hermitenorm(n, x)
        elif op == "eval_genlaguerre":
            value = special.eval_genlaguerre(n, float(alpha), x)
        elif op == "eval_gegenbauer":
            value = special.eval_gegenbauer(n, float(alpha), x)
        elif op == "eval_jacobi":
            value = special.eval_jacobi(n, float(alpha), float(beta), x)
        else:
            continue
        value = float(value)
        if not math.isfinite(value):
            continue
        results.append({"case_id": case["case_id"], "value": value})
    except Exception:
        continue

print(json.dumps(results))
"#;

    let mut child = fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(python_code)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    {
        let stdin = child.stdin.as_mut()?;
        let input = serde_json::to_vec(cases).expect("serialize special oracle input");
        stdin.write_all(&input).expect("write special oracle input");
    }
    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }
    let results: Vec<OracleResult> = serde_json::from_slice(&output.stdout).ok()?;
    Some(
        results
            .into_iter()
            .map(|result| (result.case_id.clone(), result))
            .collect(),
    )
}

fn assert_complete_oracle(
    test_id: &str,
    cases: &[SpecialCase],
    oracle: &HashMap<String, OracleResult>,
) {
    assert_eq!(
        oracle.len(),
        cases.len(),
        "{test_id} SciPy special oracle returned partial or duplicate coverage"
    );

    let missing: Vec<&str> = cases
        .iter()
        .filter(|case| !oracle.contains_key(case.case_id.as_str()))
        .map(|case| case.case_id.as_str())
        .collect();
    assert!(
        missing.is_empty(),
        "{test_id} missing SciPy special oracle results: {:?}",
        missing
    );
}

fn close_enough(actual: f64, expected: f64) -> (f64, f64, bool) {
    let abs_diff = (actual - expected).abs();
    let scale = actual.abs().max(expected.abs()).max(1.0);
    let rel_diff = abs_diff / scale;
    let pass = abs_diff <= ABS_TOL + REL_TOL * scale;
    (abs_diff, rel_diff, pass)
}

#[test]
fn diff_001_special_orthopoly_live_scipy() {
    let cases = special_cases();
    assert_eq!(
        cases.len(),
        212,
        "special orthopoly diff case inventory changed"
    );

    let start = Instant::now();
    let Some(oracle_results) = run_scipy_oracle(&cases) else {
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "{REQUIRE_SCIPY_ENV}=1 but SciPy special oracle unavailable"
        );
        eprintln!("skipping special orthopoly diff: scipy oracle not available");
        return;
    };
    assert_complete_oracle(
        "diff_001_special_orthopoly_live_scipy",
        &cases,
        &oracle_results,
    );

    let mut case_diffs = Vec::with_capacity(cases.len());
    for case in &cases {
        let actual = rust_value(case).expect("supported special op");
        let expected = oracle_results
            .get(case.case_id.as_str())
            .expect("complete oracle coverage")
            .value;
        let (abs_diff, rel_diff, pass) = close_enough(actual, expected);
        case_diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            n: case.n,
            alpha: case.alpha,
            beta: case.beta,
            x: case.x,
            rust_value: actual,
            scipy_value: expected,
            abs_diff,
            rel_diff,
            tolerance_abs: ABS_TOL,
            tolerance_rel: REL_TOL,
            pass,
        });
    }

    let max_abs_diff = case_diffs
        .iter()
        .map(|case| case.abs_diff)
        .fold(0.0_f64, f64::max);
    let max_rel_diff = case_diffs
        .iter()
        .map(|case| case.rel_diff)
        .fold(0.0_f64, f64::max);
    let pass = case_diffs.iter().all(|case| case.pass);
    let log = DiffLog {
        test_id: String::from("diff_001_special_orthopoly_live_scipy"),
        category: String::from("live_scipy_differential"),
        case_count: cases.len(),
        max_abs_diff,
        max_rel_diff,
        tolerance_abs: ABS_TOL,
        tolerance_rel: REL_TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: case_diffs,
    };
    emit_log(&log);

    assert!(
        pass,
        "special orthopoly live SciPy diff max_abs={max_abs_diff:.3e} max_rel={max_rel_diff:.3e}"
    );
}
