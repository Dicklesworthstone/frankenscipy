#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_linalg structured
//! matrix constructors.
//!
//! Resolves [frankenscipy-p69mf]. Covers the P2C-002 README
//! expansion gap for structured/decomposition-adjacent matrix
//! constructors:
//!   - toeplitz, circulant, hankel
//!   - hilbert, invhilbert, hadamard, companion
//!   - block_diag, pascal, helmert/full helmert, kron

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{
    block_diag, circulant, companion, hadamard, hankel, helmert, helmert_full, hilbert, invhilbert,
    kron, pascal, toeplitz,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-002";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

fn require_scipy_oracle() -> bool {
    matches!(
        std::env::var(REQUIRE_SCIPY_ENV).ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

#[derive(Debug, Clone, Serialize)]
struct VectorCase {
    case_id: String,
    values: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct ToeplitzCase {
    case_id: String,
    c: Vec<f64>,
    r: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct SizeCase {
    case_id: String,
    n: usize,
}

#[derive(Debug, Clone, Serialize)]
struct BlockDiagCase {
    case_id: String,
    blocks: Vec<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone, Serialize)]
struct PascalCase {
    case_id: String,
    n: usize,
    symmetric: bool,
}

#[derive(Debug, Clone, Serialize)]
struct HelmertCase {
    case_id: String,
    n: usize,
    full: bool,
}

#[derive(Debug, Clone, Serialize)]
struct KronCase {
    case_id: String,
    a: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    toeplitz: Vec<ToeplitzCase>,
    circulant: Vec<VectorCase>,
    hankel: Vec<ToeplitzCase>,
    hilbert: Vec<SizeCase>,
    invhilbert: Vec<SizeCase>,
    hadamard: Vec<SizeCase>,
    companion: Vec<VectorCase>,
    block_diag: Vec<BlockDiagCase>,
    pascal: Vec<PascalCase>,
    helmert: Vec<HelmertCase>,
    kron: Vec<KronCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct MatrixArm {
    case_id: String,
    matrix: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    toeplitz: Vec<MatrixArm>,
    circulant: Vec<MatrixArm>,
    hankel: Vec<MatrixArm>,
    hilbert: Vec<MatrixArm>,
    invhilbert: Vec<MatrixArm>,
    hadamard: Vec<MatrixArm>,
    companion: Vec<MatrixArm>,
    block_diag: Vec<MatrixArm>,
    pascal: Vec<MatrixArm>,
    helmert: Vec<MatrixArm>,
    kron: Vec<MatrixArm>,
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
    fs::create_dir_all(output_dir()).expect("create structured linalg diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize structured linalg diff log");
    fs::write(path, json).expect("write structured linalg diff log");
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        toeplitz: vec![
            ToeplitzCase {
                case_id: "toeplitz_default_row".into(),
                c: vec![1.0, 2.0, 3.0, 4.0],
                r: None,
            },
            ToeplitzCase {
                case_id: "toeplitz_asymmetric".into(),
                c: vec![1.0, 3.0, 5.0],
                r: Some(vec![1.0, 7.0, 9.0, 11.0]),
            },
        ],
        circulant: vec![
            VectorCase {
                case_id: "circulant_len3".into(),
                values: vec![1.0, 2.0, 3.0],
            },
            VectorCase {
                case_id: "circulant_signed_len4".into(),
                values: vec![2.0, -1.0, 0.5, 4.0],
            },
        ],
        hankel: vec![
            ToeplitzCase {
                case_id: "hankel_default_row_zero_padded".into(),
                c: vec![1.0, 2.0, 3.0],
                r: None,
            },
            ToeplitzCase {
                case_id: "hankel_explicit_row".into(),
                c: vec![1.0, 17.0, 99.0],
                r: Some(vec![99.0, 23.0, 45.0, 67.0, 89.0]),
            },
        ],
        hilbert: vec![
            SizeCase {
                case_id: "hilbert_3".into(),
                n: 3,
            },
            SizeCase {
                case_id: "hilbert_5".into(),
                n: 5,
            },
        ],
        invhilbert: vec![
            SizeCase {
                case_id: "invhilbert_3".into(),
                n: 3,
            },
            SizeCase {
                case_id: "invhilbert_5".into(),
                n: 5,
            },
        ],
        hadamard: vec![
            SizeCase {
                case_id: "hadamard_1".into(),
                n: 1,
            },
            SizeCase {
                case_id: "hadamard_4".into(),
                n: 4,
            },
        ],
        companion: vec![
            VectorCase {
                case_id: "companion_quadratic".into(),
                values: vec![1.0, -3.0, 2.0],
            },
            VectorCase {
                case_id: "companion_cubic_monic".into(),
                values: vec![1.0, -6.0, 11.0, -6.0],
            },
        ],
        block_diag: vec![
            BlockDiagCase {
                case_id: "block_diag_vector_and_matrix".into(),
                blocks: vec![
                    vec![vec![1.0, 2.0]],
                    vec![vec![3.0], vec![4.0]],
                    vec![vec![5.0, 6.0], vec![7.0, 8.0]],
                ],
            },
            BlockDiagCase {
                case_id: "block_diag_square_blocks".into(),
                blocks: vec![
                    vec![vec![1.0, 0.5], vec![0.5, 2.0]],
                    vec![vec![-1.0]],
                    vec![vec![3.0, 4.0, 5.0]],
                ],
            },
        ],
        pascal: vec![
            PascalCase {
                case_id: "pascal_lower_5".into(),
                n: 5,
                symmetric: false,
            },
            PascalCase {
                case_id: "pascal_symmetric_5".into(),
                n: 5,
                symmetric: true,
            },
        ],
        helmert: vec![
            HelmertCase {
                case_id: "helmert_default_5".into(),
                n: 5,
                full: false,
            },
            HelmertCase {
                case_id: "helmert_full_5".into(),
                n: 5,
                full: true,
            },
        ],
        kron: vec![
            KronCase {
                case_id: "kron_2x2_by_2x1".into(),
                a: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                b: vec![vec![0.0], vec![5.0]],
            },
            KronCase {
                case_id: "kron_rectangular".into(),
                a: vec![vec![1.0, -2.0, 0.5]],
                b: vec![vec![2.0, 3.0], vec![4.0, 5.0]],
            },
        ],
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let query_json = serde_json::to_string(query).expect("serialize structured linalg query");
    let query_json_literal = format!("{query_json:?}");
    let script = r#"
import json
import math
import numpy as np
from scipy import linalg

def listify_2d(arr):
    arr = np.asarray(arr, dtype=np.float64)
    out = []
    for row in arr.tolist():
        rrow = []
        for value in row:
            value = float(value)
            if not math.isfinite(value):
                return None
            rrow.append(value)
        out.append(rrow)
    return out

def arm(case_id, matrix):
    return {"case_id": case_id, "matrix": listify_2d(matrix)}

q = json.loads(__QUERY_JSON__)
result = {
    "toeplitz": [], "circulant": [], "hankel": [], "hilbert": [],
    "invhilbert": [], "hadamard": [], "companion": [], "block_diag": [],
    "pascal": [], "helmert": [], "kron": [],
}

for case in q["toeplitz"]:
    c = np.asarray(case["c"], dtype=np.float64)
    r = None if case["r"] is None else np.asarray(case["r"], dtype=np.float64)
    result["toeplitz"].append(arm(case["case_id"], linalg.toeplitz(c, r)))

for case in q["circulant"]:
    result["circulant"].append(
        arm(case["case_id"], linalg.circulant(np.asarray(case["values"], dtype=np.float64)))
    )

for case in q["hankel"]:
    c = np.asarray(case["c"], dtype=np.float64)
    r = None if case["r"] is None else np.asarray(case["r"], dtype=np.float64)
    result["hankel"].append(arm(case["case_id"], linalg.hankel(c, r)))

for case in q["hilbert"]:
    result["hilbert"].append(arm(case["case_id"], linalg.hilbert(case["n"])))

for case in q["invhilbert"]:
    result["invhilbert"].append(arm(case["case_id"], linalg.invhilbert(case["n"])))

for case in q["hadamard"]:
    result["hadamard"].append(arm(case["case_id"], linalg.hadamard(case["n"])))

for case in q["companion"]:
    result["companion"].append(
        arm(case["case_id"], linalg.companion(np.asarray(case["values"], dtype=np.float64)))
    )

for case in q["block_diag"]:
    blocks = [np.asarray(block, dtype=np.float64) for block in case["blocks"]]
    result["block_diag"].append(arm(case["case_id"], linalg.block_diag(*blocks)))

for case in q["pascal"]:
    kind = "symmetric" if case["symmetric"] else "lower"
    result["pascal"].append(arm(case["case_id"], linalg.pascal(case["n"], kind=kind, exact=False)))

for case in q["helmert"]:
    result["helmert"].append(arm(case["case_id"], linalg.helmert(case["n"], full=case["full"])))

for case in q["kron"]:
    a = np.asarray(case["a"], dtype=np.float64)
    b = np.asarray(case["b"], dtype=np.float64)
    result["kron"].append(arm(case["case_id"], np.kron(a, b)))

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
                "failed to spawn python3 for structured linalg oracle: {err}"
            );
            eprintln!("skipping structured linalg oracle: python3 unavailable ({err})");
            return None;
        }
    };

    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open structured linalg oracle stdin");
        if let Err(err) = stdin.write_all(script.as_bytes()) {
            let stderr = child
                .stderr
                .take()
                .map_or_else(String::new, |_| String::from("stderr unavailable"));
            assert!(
                !require_scipy_oracle(),
                "structured linalg oracle script write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping structured linalg oracle: script write failed ({err})");
            return None;
        }
    }

    let output = child
        .wait_with_output()
        .expect("wait for structured linalg oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !require_scipy_oracle(),
            "structured linalg oracle failed: {stderr}"
        );
        eprintln!("skipping structured linalg oracle: scipy unavailable\n{stderr}");
        return None;
    }
    serde_json::from_slice(&output.stdout).ok()
}

fn max_abs_diff_mat(actual: &[Vec<f64>], expected: &[Vec<f64>]) -> f64 {
    if actual.len() != expected.len() {
        return f64::INFINITY;
    }
    let mut max_diff = 0.0_f64;
    for (row_idx, (actual_row, expected_row)) in actual.iter().zip(expected.iter()).enumerate() {
        if actual_row.len() != expected_row.len() {
            return f64::INFINITY;
        }
        for (col_idx, (&actual_value, &expected_value)) in
            actual_row.iter().zip(expected_row.iter()).enumerate()
        {
            let diff = (actual_value - expected_value).abs();
            if diff.is_nan() {
                return f64::NAN;
            }
            max_diff = max_diff.max(diff);
            assert!(
                actual_value.is_finite(),
                "non-finite actual at row {row_idx} col {col_idx}"
            );
        }
    }
    max_diff
}

fn expected_map(arms: &[MatrixArm]) -> HashMap<String, Option<Vec<Vec<f64>>>> {
    arms.iter()
        .map(|arm| (arm.case_id.clone(), arm.matrix.clone()))
        .collect()
}

fn record_case(
    diffs: &mut Vec<CaseDiff>,
    expected: &HashMap<String, Option<Vec<Vec<f64>>>>,
    function: &str,
    case_id: &str,
    actual: Option<Vec<Vec<f64>>>,
) {
    let expected_matrix = expected.get(case_id);
    let (max_abs_diff, pass, detail) = match (actual, expected_matrix) {
        (Some(actual), Some(Some(expected))) => {
            let diff = max_abs_diff_mat(&actual, expected);
            let pass = diff <= ABS_TOL;
            (
                diff,
                pass,
                format!(
                    "shape actual={}x{}, expected={}x{}",
                    actual.len(),
                    actual.first().map_or(0, Vec::len),
                    expected.len(),
                    expected.first().map_or(0, Vec::len)
                ),
            )
        }
        (None, Some(None)) => (0.0, true, "both implementations rejected".into()),
        (Some(_), Some(None)) => (
            f64::INFINITY,
            false,
            "Rust produced a matrix but SciPy rejected".into(),
        ),
        (None, Some(Some(_))) => (
            f64::INFINITY,
            false,
            "Rust rejected but SciPy produced a matrix".into(),
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

#[test]
fn diff_linalg_structured_matrices() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    let start = Instant::now();

    let toeplitz_expected = expected_map(&oracle.toeplitz);
    let circulant_expected = expected_map(&oracle.circulant);
    let hankel_expected = expected_map(&oracle.hankel);
    let hilbert_expected = expected_map(&oracle.hilbert);
    let invhilbert_expected = expected_map(&oracle.invhilbert);
    let hadamard_expected = expected_map(&oracle.hadamard);
    let companion_expected = expected_map(&oracle.companion);
    let block_diag_expected = expected_map(&oracle.block_diag);
    let pascal_expected = expected_map(&oracle.pascal);
    let helmert_expected = expected_map(&oracle.helmert);
    let kron_expected = expected_map(&oracle.kron);

    let mut diffs = Vec::new();

    for case in &query.toeplitz {
        record_case(
            &mut diffs,
            &toeplitz_expected,
            "toeplitz",
            &case.case_id,
            Some(toeplitz(&case.c, case.r.as_deref())),
        );
    }
    for case in &query.circulant {
        record_case(
            &mut diffs,
            &circulant_expected,
            "circulant",
            &case.case_id,
            Some(circulant(&case.values)),
        );
    }
    for case in &query.hankel {
        record_case(
            &mut diffs,
            &hankel_expected,
            "hankel",
            &case.case_id,
            Some(hankel(&case.c, case.r.as_deref())),
        );
    }
    for case in &query.hilbert {
        record_case(
            &mut diffs,
            &hilbert_expected,
            "hilbert",
            &case.case_id,
            Some(hilbert(case.n)),
        );
    }
    for case in &query.invhilbert {
        record_case(
            &mut diffs,
            &invhilbert_expected,
            "invhilbert",
            &case.case_id,
            Some(invhilbert(case.n)),
        );
    }
    for case in &query.hadamard {
        record_case(
            &mut diffs,
            &hadamard_expected,
            "hadamard",
            &case.case_id,
            hadamard(case.n).ok(),
        );
    }
    for case in &query.companion {
        record_case(
            &mut diffs,
            &companion_expected,
            "companion",
            &case.case_id,
            companion(&case.values).ok(),
        );
    }
    for case in &query.block_diag {
        record_case(
            &mut diffs,
            &block_diag_expected,
            "block_diag",
            &case.case_id,
            Some(block_diag(&case.blocks)),
        );
    }
    for case in &query.pascal {
        record_case(
            &mut diffs,
            &pascal_expected,
            "pascal",
            &case.case_id,
            Some(pascal(case.n, case.symmetric)),
        );
    }
    for case in &query.helmert {
        let actual = if case.full {
            helmert_full(case.n)
        } else {
            helmert(case.n)
        };
        record_case(
            &mut diffs,
            &helmert_expected,
            "helmert",
            &case.case_id,
            Some(actual),
        );
    }
    for case in &query.kron {
        record_case(
            &mut diffs,
            &kron_expected,
            "kron",
            &case.case_id,
            Some(kron(&case.a, &case.b)),
        );
    }

    let max_abs_diff = diffs
        .iter()
        .map(|case| case.max_abs_diff)
        .fold(0.0_f64, f64::max);
    let pass = diffs.iter().all(|case| case.pass);
    let log = DiffLog {
        test_id: "diff_linalg_structured_matrices".into(),
        category: "scipy.linalg structured matrix constructors".into(),
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
        "structured linalg conformance failed: {} cases, max_diff={}",
        log.case_count, log.max_abs_diff
    );
}
