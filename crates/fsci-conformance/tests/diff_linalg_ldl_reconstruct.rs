#![forbid(unsafe_code)]
//! Differential harness for `fsci_linalg::ldl` against `scipy.linalg.ldl`, both `lower` values.
//!
//! frankenscipy-zqtde: `ldl` is SciPy's Bunch–Kaufman factorization (`sytrf` plus SciPy's own
//! post-processing), so its `lu`, `d` and `perm` are compared element by element: `perm`
//! exactly, `lu` and `d` to 1e-12 relative to the matrix scale (OpenBLAS kernels that fuse
//! `dsyr`'s multiply-add round a few ulps differently). `A ≈ lu·d·luᵀ` is checked as well.
//! The cases include symmetric indefinite matrices that need 2×2 pivots and row interchanges,
//! which the old unpivoted `ldl` factored wrongly while returning `Ok`. Every case must be
//! compared: an `Err` from either side fails the test.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{DecompOptions, ldl};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    n: usize,
    a: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct Factorization {
    lu: Vec<f64>,
    d: Vec<f64>,
    perm: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    lower: Factorization,
    upper: Factorization,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    lower: bool,
    perm_equal: bool,
    factor_rel_diff: f64,
    reconstruction_rel_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_factor_rel_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn emit_log(log: &DiffLog) {
    fs::create_dir_all(output_dir()).expect("create ldl diff output dir");
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ldl diff log");
    fs::write(path, json).expect("write ldl diff log");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn rows_of(flat: &[f64], n: usize) -> Vec<Vec<f64>> {
    (0..n).map(|r| flat[r * n..(r + 1) * n].to_vec()).collect()
}

/// A symmetric matrix from a closed form, so the Python arm needs no data.
fn symmetric(n: usize, entry: impl Fn(usize, usize) -> f64) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let value = entry(i, j);
            a[i * n + j] = value;
            a[j * n + i] = value;
        }
    }
    a
}

fn generate_query() -> OracleQuery {
    let case = |id: &str, n: usize, a: Vec<f64>| PointCase {
        case_id: id.into(),
        n,
        a,
    };
    OracleQuery {
        points: vec![
            case("spd_2x2", 2, vec![4.0, 1.0, 1.0, 3.0]),
            case(
                "spd_4x4",
                4,
                vec![
                    10.0, 2.0, 1.0, 0.5, 2.0, 8.0, 1.5, 0.7, 1.0, 1.5, 9.0, 1.2, 0.5, 0.7, 1.2, 7.0,
                ],
            ),
            case(
                "diag_3x3",
                3,
                vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0],
            ),
            case("indefinite_2x2", 2, vec![1.0, 2.0, 2.0, -3.0]),
            // Zero diagonal: only 2×2 pivots and interchanges factor it.
            case(
                "zero_diagonal_3x3",
                3,
                vec![0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 2.0, 3.0, 0.0],
            ),
            case(
                "saddle_4x4",
                4,
                vec![
                    4.0, 1.0, 0.0, 2.0, 1.0, 0.0, 3.0, 0.0, 0.0, 3.0, -1.0, 1.0, 2.0, 0.0, 1.0, 0.0,
                ],
            ),
            case(
                "zero_diagonal_9x9",
                9,
                symmetric(9, |i, j| {
                    if i == j {
                        0.0
                    } else {
                        ((i * 7 + j * 3) % 11) as f64 - 5.0
                    }
                }),
            ),
            case(
                "indefinite_16x16",
                16,
                symmetric(16, |i, j| {
                    (((i + 1) * (j + 2)) % 13) as f64 - 6.0 + 0.25 * (i == j) as u8 as f64
                }),
            ),
            // Singular: SciPy returns the factorization with a zero pivot, not an error.
            case("singular_2x2", 2, vec![1.0, 2.0, 2.0, 4.0]),
        ],
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy import linalg

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    n = int(case["n"])
    A = np.array(case["a"], dtype=float).reshape(n, n)
    arm = {"case_id": case["case_id"]}
    for key, lower in (("lower", True), ("upper", False)):
        lu, d, perm = linalg.ldl(A, lower=lower)
        arm[key] = {"lu": lu.ravel().tolist(), "d": d.ravel().tolist(),
                    "perm": [int(p) for p in perm]}
    points.append(arm)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize ldl query");
    let spawned = fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match spawned {
        Ok(child) => child,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for ldl oracle: {e}"
            );
            eprintln!("skipping ldl oracle: python3 not available ({e})");
            return None;
        }
    };
    child
        .stdin
        .as_mut()
        .expect("open ldl oracle stdin")
        .write_all(query_json.as_bytes())
        .expect("write ldl oracle stdin");
    let output = child.wait_with_output().expect("wait for ldl oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ldl oracle failed: {stderr}"
        );
        eprintln!("skipping ldl oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ldl oracle JSON"))
}

fn max_abs(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

#[test]
fn diff_linalg_ldl_reconstruct() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let mut diffs = Vec::new();
    for (case, arm) in query.points.iter().zip(&oracle.points) {
        assert_eq!(case.case_id, arm.case_id);
        let n = case.n;
        let a = rows_of(&case.a, n);
        let scale = max_abs(case.a.iter().copied()).max(f64::MIN_POSITIVE);
        for (lower, scipy) in [(true, &arm.lower), (false, &arm.upper)] {
            let ours = match ldl(&a, lower, DecompOptions::default()) {
                Ok(ours) => ours,
                Err(e) => {
                    // A failing case, not a skipped one: the final assertions count it.
                    eprintln!("ldl {} lower={lower}: {e}", case.case_id);
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        lower,
                        perm_equal: false,
                        factor_rel_diff: f64::INFINITY,
                        reconstruction_rel_diff: f64::INFINITY,
                        pass: false,
                    });
                    continue;
                }
            };
            let lu: Vec<f64> = ours.lu.concat();
            let d: Vec<f64> = ours.d.concat();
            let factor_rel_diff = max_abs(
                lu.iter()
                    .zip(&scipy.lu)
                    .chain(d.iter().zip(&scipy.d))
                    .map(|(x, y)| x - y),
            ) / scale;
            // A = lu·d·luᵀ.
            let mut reconstruction = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..n {
                        for l in 0..n {
                            sum += ours.lu[i][k] * ours.d[k][l] * ours.lu[j][l];
                        }
                    }
                    reconstruction = reconstruction.max((sum - a[i][j]).abs());
                }
            }
            let reconstruction_rel_diff = reconstruction / scale;
            let perm_equal = ours.perm == scipy.perm;
            let pass = perm_equal
                && lu.len() == scipy.lu.len()
                && d.len() == scipy.d.len()
                && factor_rel_diff <= REL_TOL
                && reconstruction_rel_diff <= 1e-12;
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                lower,
                perm_equal,
                factor_rel_diff,
                reconstruction_rel_diff,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_linalg_ldl_reconstruct".into(),
        category: "fsci_linalg.ldl vs scipy.linalg.ldl (lu, d, perm; both triangles)".into(),
        case_count: diffs.len(),
        max_factor_rel_diff: max_abs(diffs.iter().map(|d| d.factor_rel_diff)),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);
    for d in diffs.iter().filter(|d| !d.pass) {
        eprintln!("ldl mismatch: {d:?}");
    }
    assert_eq!(
        diffs.len(),
        2 * query.points.len(),
        "every case must be compared"
    );
    assert!(all_pass, "ldl conformance failed: {log:?}");
}
