#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_linalg structured and
//! factorized solver helpers.
//!
//! Resolves [frankenscipy-66nim]. Covers the P2C-002 structured solver
//! expansion gap for:
//!   - solve_triangular, solve_banded, solveh_banded
//!   - solve_circulant, solve_toeplitz
//!   - lu_solve, cho_solve, cho_solve_banded

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{
    DecompOptions, SolveOptions, TriangularSolveOptions, TriangularTranspose, cho_factor,
    cho_solve, cho_solve_banded, lu_factor, lu_solve, solve_banded, solve_circulant,
    solve_toeplitz, solve_triangular, solveh_banded,
};
use fsci_runtime::RuntimeMode;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-002";
const ABS_TOL: f64 = 1.0e-8;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

fn require_scipy_oracle() -> bool {
    matches!(
        std::env::var(REQUIRE_SCIPY_ENV).ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

#[derive(Debug, Clone, Serialize)]
struct TriangularCase {
    case_id: String,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    lower: bool,
    trans: String,
    unit_diagonal: bool,
}

#[derive(Debug, Clone, Serialize)]
struct BandedCase {
    case_id: String,
    lower_bands: usize,
    upper_bands: usize,
    ab: Vec<Vec<f64>>,
    b: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct HermitianBandedCase {
    case_id: String,
    ab: Vec<Vec<f64>>,
    b: Vec<f64>,
    lower: bool,
}

#[derive(Debug, Clone, Serialize)]
struct VectorSolveCase {
    case_id: String,
    c: Vec<f64>,
    b: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct ToeplitzSolveCase {
    case_id: String,
    c: Vec<f64>,
    r: Option<Vec<f64>>,
    b: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct DenseSolveCase {
    case_id: String,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    triangular: Vec<TriangularCase>,
    banded: Vec<BandedCase>,
    solveh_banded: Vec<HermitianBandedCase>,
    circulant: Vec<VectorSolveCase>,
    toeplitz: Vec<ToeplitzSolveCase>,
    lu_solve: Vec<DenseSolveCase>,
    cho_solve: Vec<DenseSolveCase>,
    cho_solve_banded: Vec<HermitianBandedCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct VectorArm {
    case_id: String,
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    triangular: Vec<VectorArm>,
    banded: Vec<VectorArm>,
    solveh_banded: Vec<VectorArm>,
    circulant: Vec<VectorArm>,
    toeplitz: Vec<VectorArm>,
    lu_solve: Vec<VectorArm>,
    cho_solve: Vec<VectorArm>,
    cho_solve_banded: Vec<VectorArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    function: String,
    max_abs_diff: f64,
    pass: bool,
    detail: String,
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
    fs::create_dir_all(output_dir()).expect("create structured solver diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize structured solver diff log");
    fs::write(path, json).expect("write structured solver diff log");
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        triangular: vec![
            TriangularCase {
                case_id: "triangular_upper_no_transpose".into(),
                a: vec![
                    vec![4.0, -1.0, 2.0],
                    vec![0.0, 3.0, 0.5],
                    vec![0.0, 0.0, 2.0],
                ],
                b: vec![7.0, 8.0, 6.0],
                lower: false,
                trans: "N".into(),
                unit_diagonal: false,
            },
            TriangularCase {
                case_id: "triangular_lower_transpose".into(),
                a: vec![
                    vec![2.0, 0.0, 0.0],
                    vec![1.0, 3.0, 0.0],
                    vec![4.0, -2.0, 5.0],
                ],
                b: vec![2.0, 5.0, 7.0],
                lower: true,
                trans: "T".into(),
                unit_diagonal: false,
            },
            TriangularCase {
                case_id: "triangular_upper_unit_diagonal".into(),
                a: vec![
                    vec![99.0, 2.0, -1.0],
                    vec![0.0, 88.0, 3.0],
                    vec![0.0, 0.0, 77.0],
                ],
                b: vec![3.0, 4.0, -2.0],
                lower: false,
                trans: "N".into(),
                unit_diagonal: true,
            },
        ],
        banded: vec![
            BandedCase {
                case_id: "banded_tridiagonal_4".into(),
                lower_bands: 1,
                upper_bands: 1,
                ab: vec![
                    vec![0.0, -1.0, -1.0, -1.0],
                    vec![4.0, 4.0, 4.0, 4.0],
                    vec![-1.0, -1.0, -1.0, 0.0],
                ],
                b: vec![5.0, 5.0, 10.0, 23.0],
            },
            BandedCase {
                case_id: "banded_pentadiagonal_5".into(),
                lower_bands: 2,
                upper_bands: 2,
                ab: vec![
                    vec![0.0, 0.0, 1.0, 1.0, 1.0],
                    vec![0.0, 1.0, 1.0, 1.0, 1.0],
                    vec![4.0, 4.0, 4.0, 4.0, 4.0],
                    vec![1.0, 1.0, 1.0, 1.0, 0.0],
                    vec![1.0, 1.0, 1.0, 0.0, 0.0],
                ],
                b: vec![7.0, 8.0, 8.0, 8.0, 7.0],
            },
        ],
        solveh_banded: vec![
            HermitianBandedCase {
                case_id: "solveh_banded_lower_tridiagonal".into(),
                ab: vec![vec![4.0, 5.0, 6.0], vec![2.0, 2.0, 0.0]],
                b: vec![6.0, 9.0, 8.0],
                lower: true,
            },
            HermitianBandedCase {
                case_id: "solveh_banded_diagonal".into(),
                ab: vec![vec![4.0, 9.0, 16.0]],
                b: vec![8.0, 27.0, 32.0],
                lower: true,
            },
        ],
        circulant: vec![
            VectorSolveCase {
                case_id: "circulant_identity".into(),
                c: vec![1.0, 0.0, 0.0],
                b: vec![2.0, 3.0, 4.0],
            },
            VectorSolveCase {
                case_id: "circulant_nontrivial".into(),
                c: vec![3.0, 1.0, 0.5],
                b: vec![1.0, 2.0, 3.0],
            },
        ],
        toeplitz: vec![
            ToeplitzSolveCase {
                case_id: "toeplitz_symmetric_identity".into(),
                c: vec![1.0, 0.0, 0.0],
                r: None,
                b: vec![2.0, 3.0, 4.0],
            },
            ToeplitzSolveCase {
                case_id: "toeplitz_scipy_doc_example".into(),
                c: vec![1.0, 3.0, 6.0, 10.0],
                r: Some(vec![1.0, -1.0, -2.0, -3.0]),
                b: vec![1.0, 2.0, 2.0, 5.0],
            },
        ],
        lu_solve: vec![DenseSolveCase {
            case_id: "lu_solve_general_3".into(),
            a: vec![
                vec![3.0, 1.0, -1.0],
                vec![2.0, 4.0, 1.0],
                vec![-1.0, 2.0, 5.0],
            ],
            b: vec![4.0, 1.0, 1.0],
        }],
        cho_solve: vec![DenseSolveCase {
            case_id: "cho_solve_spd_3".into(),
            a: vec![
                vec![4.0, 1.0, 0.5],
                vec![1.0, 3.0, 0.25],
                vec![0.5, 0.25, 2.0],
            ],
            b: vec![2.0, 5.0, -1.0],
        }],
        cho_solve_banded: vec![
            HermitianBandedCase {
                case_id: "cho_solve_banded_lower_tridiagonal".into(),
                ab: vec![vec![2.0, 2.0, 5.0_f64.sqrt()], vec![1.0, 1.0, 0.0]],
                b: vec![6.0, 9.0, 8.0],
                lower: true,
            },
            HermitianBandedCase {
                case_id: "cho_solve_banded_diagonal".into(),
                ab: vec![vec![2.0, 3.0, 4.0]],
                b: vec![2.0, 6.0, 8.0],
                lower: true,
            },
        ],
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let query_json = serde_json::to_string(query).expect("serialize structured solver query");
    let query_json_literal = format!("{query_json:?}");
    let script = r#"
import json
import math
import numpy as np
from scipy import linalg

def vector_or_none(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out = []
    for value in arr.tolist():
        value = float(value)
        if not math.isfinite(value):
            return None
        out.append(value)
    return out

def arm(case_id, values):
    return {"case_id": case_id, "values": vector_or_none(values)}

q = json.loads(__QUERY_JSON__)
result = {
    "triangular": [], "banded": [], "solveh_banded": [], "circulant": [],
    "toeplitz": [], "lu_solve": [], "cho_solve": [], "cho_solve_banded": [],
}

for case in q["triangular"]:
    a = np.asarray(case["a"], dtype=np.float64)
    b = np.asarray(case["b"], dtype=np.float64)
    result["triangular"].append(
        arm(
            case["case_id"],
            linalg.solve_triangular(
                a,
                b,
                lower=bool(case["lower"]),
                trans=case["trans"],
                unit_diagonal=bool(case["unit_diagonal"]),
                check_finite=True,
            ),
        )
    )

for case in q["banded"]:
    ab = np.asarray(case["ab"], dtype=np.float64)
    b = np.asarray(case["b"], dtype=np.float64)
    result["banded"].append(
        arm(
            case["case_id"],
            linalg.solve_banded((case["lower_bands"], case["upper_bands"]), ab, b),
        )
    )

for case in q["solveh_banded"]:
    ab = np.asarray(case["ab"], dtype=np.float64)
    b = np.asarray(case["b"], dtype=np.float64)
    result["solveh_banded"].append(
        arm(case["case_id"], linalg.solveh_banded(ab, b, lower=bool(case["lower"])))
    )

for case in q["circulant"]:
    c = np.asarray(case["c"], dtype=np.float64)
    b = np.asarray(case["b"], dtype=np.float64)
    result["circulant"].append(arm(case["case_id"], linalg.solve_circulant(c, b)))

for case in q["toeplitz"]:
    c = np.asarray(case["c"], dtype=np.float64)
    b = np.asarray(case["b"], dtype=np.float64)
    if case["r"] is None:
        result["toeplitz"].append(arm(case["case_id"], linalg.solve_toeplitz(c, b)))
    else:
        r = np.asarray(case["r"], dtype=np.float64)
        result["toeplitz"].append(arm(case["case_id"], linalg.solve_toeplitz((c, r), b)))

for case in q["lu_solve"]:
    a = np.asarray(case["a"], dtype=np.float64)
    b = np.asarray(case["b"], dtype=np.float64)
    lu_and_piv = linalg.lu_factor(a, check_finite=True)
    result["lu_solve"].append(arm(case["case_id"], linalg.lu_solve(lu_and_piv, b)))

for case in q["cho_solve"]:
    a = np.asarray(case["a"], dtype=np.float64)
    b = np.asarray(case["b"], dtype=np.float64)
    factor = linalg.cho_factor(a, lower=True, check_finite=True)
    result["cho_solve"].append(arm(case["case_id"], linalg.cho_solve(factor, b)))

for case in q["cho_solve_banded"]:
    cb = np.asarray(case["ab"], dtype=np.float64)
    b = np.asarray(case["b"], dtype=np.float64)
    result["cho_solve_banded"].append(
        arm(case["case_id"], linalg.cho_solve_banded((cb, bool(case["lower"])), b))
    )

print(json.dumps(result, allow_nan=False))
"#
    .replace("__QUERY_JSON__", &query_json_literal);
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            assert!(
                !require_scipy_oracle(),
                "failed to spawn python3 for structured solver oracle: {err}"
            );
            eprintln!("skipping structured solver oracle: python3 unavailable ({err})");
            return None;
        }
    };

    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open structured solver oracle stdin");
        if let Err(err) = stdin.write_all(script.as_bytes()) {
            let stderr = child
                .stderr
                .take()
                .map_or_else(String::new, |_| String::from("stderr unavailable"));
            assert!(
                !require_scipy_oracle(),
                "structured solver oracle script write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping structured solver oracle: script write failed ({err})");
            return None;
        }
    }

    let output = child
        .wait_with_output()
        .expect("wait for structured solver oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !require_scipy_oracle(),
            "structured solver oracle failed: {stderr}"
        );
        eprintln!("skipping structured solver oracle: scipy unavailable\n{stderr}");
        return None;
    }
    serde_json::from_slice(&output.stdout).ok()
}

fn max_abs_diff_vec(actual: &[f64], expected: &[f64]) -> f64 {
    if actual.len() != expected.len() {
        return f64::INFINITY;
    }
    let mut max_diff = 0.0_f64;
    for (idx, (&actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual_value - expected_value).abs();
        if diff.is_nan() {
            return f64::NAN;
        }
        max_diff = max_diff.max(diff);
        assert!(actual_value.is_finite(), "non-finite actual at index {idx}");
    }
    max_diff
}

fn expected_map(arms: &[VectorArm]) -> HashMap<String, Option<Vec<f64>>> {
    arms.iter()
        .map(|arm| (arm.case_id.clone(), arm.values.clone()))
        .collect()
}

fn record_case(
    diffs: &mut Vec<CaseDiff>,
    expected: &HashMap<String, Option<Vec<f64>>>,
    function: &str,
    case_id: &str,
    actual: Option<Vec<f64>>,
) {
    let expected_vector = expected.get(case_id);
    let (max_abs_diff, pass, detail) = match (actual, expected_vector) {
        (Some(actual), Some(Some(expected))) => {
            let diff = max_abs_diff_vec(&actual, expected);
            let pass = diff <= ABS_TOL;
            (
                diff,
                pass,
                format!("len actual={}, expected={}", actual.len(), expected.len()),
            )
        }
        (None, Some(None)) => (0.0, true, "both implementations rejected".into()),
        (Some(_), Some(None)) => (
            f64::INFINITY,
            false,
            "Rust produced a vector but SciPy rejected".into(),
        ),
        (None, Some(Some(_))) => (
            f64::INFINITY,
            false,
            "Rust rejected but SciPy produced a vector".into(),
        ),
        (_, None) => (
            f64::INFINITY,
            false,
            "SciPy oracle did not return this case".into(),
        ),
    };

    diffs.push(CaseDiff {
        case_id: case_id.to_string(),
        function: function.to_string(),
        max_abs_diff,
        pass,
        detail,
    });
}

fn triangular_transpose(value: &str) -> TriangularTranspose {
    match value {
        "T" => TriangularTranspose::Transpose,
        "C" => TriangularTranspose::ConjugateTranspose,
        _ => TriangularTranspose::NoTranspose,
    }
}

#[test]
fn diff_linalg_structured_solvers() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    let start = Instant::now();

    let triangular_expected = expected_map(&oracle.triangular);
    let banded_expected = expected_map(&oracle.banded);
    let solveh_banded_expected = expected_map(&oracle.solveh_banded);
    let circulant_expected = expected_map(&oracle.circulant);
    let toeplitz_expected = expected_map(&oracle.toeplitz);
    let lu_solve_expected = expected_map(&oracle.lu_solve);
    let cho_solve_expected = expected_map(&oracle.cho_solve);
    let cho_solve_banded_expected = expected_map(&oracle.cho_solve_banded);

    let solve_options = SolveOptions {
        mode: RuntimeMode::Strict,
        check_finite: true,
        ..SolveOptions::default()
    };
    let decomp_options = DecompOptions {
        mode: RuntimeMode::Strict,
        check_finite: true,
    };
    let mut diffs = Vec::new();

    for case in &query.triangular {
        let actual = solve_triangular(
            &case.a,
            &case.b,
            TriangularSolveOptions {
                mode: RuntimeMode::Strict,
                check_finite: true,
                trans: triangular_transpose(&case.trans),
                lower: case.lower,
                unit_diagonal: case.unit_diagonal,
            },
        )
        .ok()
        .map(|result| result.x);
        record_case(
            &mut diffs,
            &triangular_expected,
            "solve_triangular",
            &case.case_id,
            actual,
        );
    }
    for case in &query.banded {
        let actual = solve_banded(
            (case.lower_bands, case.upper_bands),
            &case.ab,
            &case.b,
            solve_options,
        )
        .ok()
        .map(|result| result.x);
        record_case(
            &mut diffs,
            &banded_expected,
            "solve_banded",
            &case.case_id,
            actual,
        );
    }
    for case in &query.solveh_banded {
        let actual = solveh_banded(&case.ab, &case.b, case.lower)
            .ok()
            .map(|result| result.x);
        record_case(
            &mut diffs,
            &solveh_banded_expected,
            "solveh_banded",
            &case.case_id,
            actual,
        );
    }
    for case in &query.circulant {
        record_case(
            &mut diffs,
            &circulant_expected,
            "solve_circulant",
            &case.case_id,
            solve_circulant(&case.c, &case.b).ok(),
        );
    }
    for case in &query.toeplitz {
        record_case(
            &mut diffs,
            &toeplitz_expected,
            "solve_toeplitz",
            &case.case_id,
            solve_toeplitz(&case.c, case.r.as_deref(), &case.b).ok(),
        );
    }
    for case in &query.lu_solve {
        let actual = lu_factor(&case.a, decomp_options)
            .and_then(|factor| lu_solve(&factor, &case.b))
            .ok()
            .map(|result| result.x);
        record_case(
            &mut diffs,
            &lu_solve_expected,
            "lu_solve",
            &case.case_id,
            actual,
        );
    }
    for case in &query.cho_solve {
        let actual = cho_factor(&case.a, decomp_options)
            .and_then(|factor| cho_solve(&factor, &case.b))
            .ok()
            .map(|result| result.x);
        record_case(
            &mut diffs,
            &cho_solve_expected,
            "cho_solve",
            &case.case_id,
            actual,
        );
    }
    for case in &query.cho_solve_banded {
        let actual = cho_solve_banded(&case.ab, &case.b, case.lower)
            .ok()
            .map(|result| result.x);
        record_case(
            &mut diffs,
            &cho_solve_banded_expected,
            "cho_solve_banded",
            &case.case_id,
            actual,
        );
    }

    let max_abs_diff = diffs
        .iter()
        .map(|case| case.max_abs_diff)
        .fold(0.0_f64, f64::max);
    let pass = diffs.iter().all(|case| case.pass);
    let log = DiffLog {
        test_id: "diff_linalg_structured_solvers".into(),
        category: "scipy.linalg structured and factorized solvers".into(),
        case_count: diffs.len(),
        max_abs_diff,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs,
    };
    emit_log(&log);

    assert!(
        pass,
        "structured solver conformance failed: {} cases, max_diff={}",
        log.case_count, log.max_abs_diff
    );
}
