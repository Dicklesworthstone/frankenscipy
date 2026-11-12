#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_linalg matrix
//! functions.
//!
//! Resolves [frankenscipy-srhne]. Covers the 9 matrix-function
//! primitives: expm, sqrtm, logm, and the matrix-trig variants
//! (sinm, cosm, tanm, sinhm, coshm, tanhm). Each diffed against
//! the corresponding scipy.linalg.{name}.
//!
//! 4 fixtures × applicable functions ≈ 28-32 cases. Tol 1e-9
//! abs because matrix functions use Padé approximation / Schur
//! decomposition (iterative) — scipy's implementation differs
//! in detail from fsci's.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_linalg::{
    DecompOptions, coshm, cosm, expm, expm_frechet, logm, sinhm, sinm, sqrtm, tanhm, tanm,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-013";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    a: Vec<Vec<f64>>,
    fns: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct FrechetCase {
    case_id: String,
    a: Vec<Vec<f64>>,
    e: Vec<Vec<f64>>,
    direction_scale: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
    frechet: Vec<FrechetCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    fn_name: String,
    matrix: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone, Deserialize)]
struct FrechetArm {
    case_id: String,
    exponential: Vec<Vec<f64>>,
    derivative: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
    frechet: Vec<FrechetArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    fn_name: String,
    max_abs_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
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
    fs::create_dir_all(output_dir()).expect("create matrix-functions diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize matrix-functions diff log");
    fs::write(path, json).expect("write matrix-functions diff log");
}

fn dispatch_fn(name: &str, a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let opts = DecompOptions::default();
    match name {
        "expm" => expm(a, opts).ok(),
        "sqrtm" => sqrtm(a, opts).ok(),
        "logm" => logm(a, opts).ok(),
        "sinm" => sinm(a, opts).ok(),
        "cosm" => cosm(a, opts).ok(),
        "tanm" => tanm(a, opts).ok(),
        "sinhm" => sinhm(a, opts).ok(),
        "coshm" => coshm(a, opts).ok(),
        "tanhm" => tanhm(a, opts).ok(),
        _ => None,
    }
}

fn max_abs_diff_mat(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut m = 0.0_f64;
    for (ra, rb) in a.iter().zip(b.iter()) {
        for (va, vb) in ra.iter().zip(rb.iter()) {
            m = m.max((va - vb).abs());
        }
    }
    m
}

fn generate_query() -> OracleQuery {
    // 9 matrix functions; logm requires positive eigenvalues
    // (avoid the identity-only fixture for the trig fns since
    // sinm/cosm/tanm of identity is sin(1)·I which is fine, but
    // the fixture is uninteresting). Pick 4 representative
    // matrices.
    let all_fns: Vec<String> = vec![
        "expm", "sqrtm", "logm", "sinm", "cosm", "tanm", "sinhm", "coshm", "tanhm",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    OracleQuery {
        points: vec![
            // Symmetric positive-definite 2×2
            PointCase {
                case_id: "spd_2x2".into(),
                a: vec![vec![2.0, 0.5], vec![0.5, 1.5]],
                fns: all_fns.clone(),
            },
            // Diagonal 3×3 (closed-form: f(D) is diag(f(d_i)))
            PointCase {
                case_id: "diag_3x3".into(),
                a: vec![
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 0.5, 0.0],
                    vec![0.0, 0.0, 0.25],
                ],
                fns: all_fns.clone(),
            },
            // Symmetric 3×3 with distinct eigenvalues
            PointCase {
                case_id: "sym_3x3_distinct".into(),
                a: vec![
                    vec![3.0, 1.0, 0.0],
                    vec![1.0, 2.0, 0.5],
                    vec![0.0, 0.5, 1.0],
                ],
                fns: all_fns.clone(),
            },
            // Identity 4×4
            PointCase {
                case_id: "identity_4x4".into(),
                a: vec![
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0, 0.0],
                    vec![0.0, 0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 0.0, 1.0],
                ],
                fns: all_fns.clone(),
            },
        ],
        frechet: Vec::new(),
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.linalg import expm, expm_frechet, sqrtm, logm, sinm, cosm, tanm, sinhm, coshm, tanhm

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

def listify_2d(arr):
    out = []
    for row in arr.tolist():
        rrow = []
        for v in row:
            f = fnone(v)
            if f is None:
                return None
            rrow.append(f)
        out.append(rrow)
    return out

DISPATCH = {
    "expm": expm, "sqrtm": sqrtm, "logm": logm,
    "sinm": sinm, "cosm": cosm, "tanm": tanm,
    "sinhm": sinhm, "coshm": coshm, "tanhm": tanhm,
}

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    A = np.asarray(case["a"], dtype=np.float64)
    for name in case["fns"]:
        out = {"case_id": cid, "fn_name": name, "matrix": None}
        try:
            res = DISPATCH[name](A)
            # Some scipy fns return tuples (sqrtm returns (mat, errest)).
            if isinstance(res, tuple):
                res = res[0]
            res = np.asarray(res, dtype=np.float64)
            # logm/sqrtm may return complex if eigenvalues are negative
            # — drop any complex-valued result.
            if not np.all(np.isreal(res)):
                out["matrix"] = None
            else:
                out["matrix"] = listify_2d(res.real)
        except Exception:
            pass
        points.append(out)
frechet = []
for case in q["frechet"]:
    A = np.asarray(case["a"], dtype=np.float64)
    E = np.asarray(case["e"], dtype=np.float64)
    exponential, derivative = expm_frechet(A, E)
    # These cases have finite answers. Never convert a failure or NaN into a
    # skipped comparison: JSON serialization and the required Rust fields fail.
    frechet.append({"case_id": case["case_id"],
                    "exponential": exponential.tolist(),
                    "derivative": derivative.tolist()})
print(json.dumps({"points": points, "frechet": frechet}, allow_nan=False))
"#;
    let query_json = serde_json::to_string(query).expect("serialize matrix-functions query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for matrix-functions oracle: {e}"
            );
            eprintln!("skipping matrix-functions oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open matrix-functions oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "matrix-functions oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping matrix-functions oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for matrix-functions oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "matrix-functions oracle failed: {stderr}"
        );
        eprintln!("skipping matrix-functions oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse matrix-functions oracle JSON"))
}

#[test]
fn diff_linalg_matrix_functions() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    // Build (case_id, fn_name) → scipy matrix lookup.
    let mut map: HashMap<(String, String), Vec<Vec<f64>>> = HashMap::new();
    for arm in oracle.points.into_iter() {
        if let Some(mat) = arm.matrix {
            map.insert((arm.case_id, arm.fn_name), mat);
        }
    }

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;
    let arms = [
        "expm", "sqrtm", "logm", "sinm", "cosm", "tanm", "sinhm", "coshm", "tanhm",
    ];
    let mut ledger = CompareLedger::new("diff_linalg_matrix_functions", &arms);

    for case in &query.points {
        for fn_name in &case.fns {
            let scipy_mat = map.get(&(case.case_id.clone(), fn_name.clone()));
            let rust_mat = dispatch_fn(fn_name, &case.a);
            let Some((scipy_mat, rust_mat)) =
                ledger.both(fn_name, &case.case_id, scipy_mat, rust_mat)
            else {
                continue;
            };
            let shape_ok = rust_mat.len() == scipy_mat.len()
                && rust_mat
                    .iter()
                    .zip(scipy_mat.iter())
                    .all(|(rr, sr)| rr.len() == sr.len());
            // max_abs_diff_mat reads a NaN as 0.0; a NaN entry fails the case.
            let no_nan = !rust_mat.iter().flatten().any(|v| v.is_nan());
            let max_d = if shape_ok {
                max_abs_diff_mat(&rust_mat, scipy_mat)
            } else {
                f64::INFINITY
            };
            if shape_ok {
                max_overall = max_overall.max(max_d);
            }
            let pass = shape_ok && no_nan && max_d <= ABS_TOL;
            ledger.compared(fn_name, &case.case_id, pass);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                fn_name: fn_name.clone(),
                max_abs_diff: max_d,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_matrix_functions".into(),
        category: "fsci_linalg matrix functions".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "matrix-function mismatch: {} {} — max_abs={}",
                d.case_id, d.fn_name, d.max_abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "matrix-functions conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(
        arms.iter()
            .map(|arm| {
                query
                    .points
                    .iter()
                    .filter(|c| c.fns.iter().any(|f| f == arm))
                    .count()
            })
            .min()
            .unwrap_or(0),
    );
}

#[test]
fn diff_linalg_expm_frechet_direction_scaling() {
    let mut cases = Vec::new();
    for (name, a) in [
        ("diagonal", vec![vec![1.0, 0.0], vec![0.0, -2.0]]),
        ("nonnormal", vec![vec![0.0, 5.0], vec![0.0, 0.0]]),
        ("dense", vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
    ] {
        for scale in [1e-200, 1.0, 1e20, -1e20, 1e100] {
            cases.push(FrechetCase {
                case_id: format!("{name}_direction_{scale}"),
                a: a.clone(),
                e: vec![
                    vec![0.5 * scale, -0.1 * scale],
                    vec![0.2 * scale, 0.3 * scale],
                ],
                direction_scale: scale,
            });
        }
    }
    let query = OracleQuery {
        points: Vec::new(),
        frechet: cases,
    };
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    let expected: HashMap<_, _> = oracle
        .frechet
        .into_iter()
        .map(|arm| (arm.case_id.clone(), arm))
        .collect();
    assert_eq!(expected.len(), query.frechet.len(), "missing oracle cases");
    let arms = ["expm_frechet_expm", "expm_frechet_derivative"];
    let mut ledger = CompareLedger::new("diff_linalg_expm_frechet_direction_scaling", &arms);
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;
    for case in &query.frechet {
        let reference = expected.get(&case.case_id).expect("oracle case");
        let (exponential, derivative) = expm_frechet(&case.a, &case.e, DecompOptions::default())
            .expect("finite Fréchet inputs");
        for (name, actual, oracle_matrix, divisor) in [
            (arms[0], &exponential, &reference.exponential, 1.0),
            (
                arms[1],
                &derivative,
                &reference.derivative,
                case.direction_scale,
            ),
        ] {
            assert_eq!(actual.len(), oracle_matrix.len(), "{}", case.case_id);
            let mut pass = true;
            let mut max_difference = 0.0_f64;
            for (actual_row, expected_row) in actual.iter().zip(oracle_matrix) {
                assert_eq!(actual_row.len(), expected_row.len(), "{}", case.case_id);
                for (&value, &expected_value) in actual_row.iter().zip(expected_row) {
                    let value = value / divisor;
                    let expected_value = expected_value / divisor;
                    let difference = (value - expected_value).abs();
                    pass &= value.is_finite()
                        && expected_value.is_finite()
                        && difference <= 5e-12 * expected_value.abs().max(1.0);
                    max_difference = max_difference.max(difference);
                }
            }
            ledger.compared(name, &case.case_id, pass);
            max_overall = max_overall.max(max_difference);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                fn_name: name.to_string(),
                max_abs_diff: max_difference,
                pass,
            });
        }
    }
    let all_pass = diffs.iter().all(|case| case.pass);
    emit_log(&DiffLog {
        test_id: "diff_linalg_expm_frechet_direction_scaling".into(),
        category: "expm_frechet vs SciPy, derivative normalized by direction scale".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs,
    });
    assert!(
        all_pass,
        "Fréchet scaling mismatch: max difference {max_overall}"
    );
    ledger.finish(query.frechet.len());
}
