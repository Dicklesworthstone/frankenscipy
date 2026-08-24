#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the interpolative-decomposition reconstruction
//! functions in `fsci_linalg::interpolative`, against `scipy.linalg.interpolative`.
//!
//! ## The randomness is sidestepped, not fought
//!
//! `interp_decomp` is randomized on both sides — SciPy draws from its own RNG, this crate uses
//! a seeded SRHT — so the two will never agree on which columns end up in the skeleton, and
//! comparing them directly would be comparing two coin flips. Everything tested here is a pure
//! function of `(idx, proj)`, so the oracle computes ONE decomposition and both arms are handed
//! that same `(idx, proj)` to build from. What is compared is the reconstruction, which is
//! deterministic.
//!
//! ## Which comparisons are EXACT and which carry a tolerance, and why
//!
//! Mixing the two silently would let a tolerance hide a bug in the half that ought to be exact.
//! They are separated deliberately:
//!
//!   * `reconstruct_interp_matrix` and `reconstruct_skel_matrix` do NO ARITHMETIC — they place
//!     ones and copy entries. Compared BIT-EXACTLY. A tolerance here would mask precisely the
//!     failure these functions are prone to, which is putting the right numbers in the wrong
//!     columns.
//!   * `reconstruct_matrix_from_id` is a matrix product. NumPy dispatches to BLAS, which
//!     reassociates and may fuse multiply-add; a naive triple loop legitimately differs in the
//!     last bits. Compared to a relative tolerance.
//!   * `id_to_svd` is iterative, and its singular VECTORS are only determined up to sign and up
//!     to rotation within equal singular values. So the singular VALUES are compared, and the
//!     reconstruction `U·diag(s)·Vᵀ` is compared — never `U` entry by entry, which would be a
//!     test of LAPACK's sign conventions rather than of this code.
//!
//! Floats cross the process boundary as IEEE-754 bit patterns, since `serde_json` without
//! `float_roundtrip` misreads some 17-digit decimals by one ULP — which would defeat the
//! bit-exact half of the above.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_linalg::interpolative::{
    id_to_svd, reconstruct_interp_matrix, reconstruct_matrix_from_id, reconstruct_skel_matrix,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

/// Relative tolerance for the arithmetic comparisons. Generous enough for BLAS reassociation
/// over the small ranks used here, tight enough that a wrong formula cannot pass.
const REL_TOL: f64 = 1e-11;

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    rows: usize,
    cols: usize,
    /// Row-major, as IEEE-754 bit patterns.
    a_bits: Vec<u64>,
    k: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// The decomposition the oracle produced, which BOTH arms then build from.
    idx: Option<Vec<usize>>,
    proj_bits: Option<Vec<u64>>,
    proj_rows: Option<usize>,
    proj_cols: Option<usize>,
    /// The incumbent's reconstructions, row-major.
    p_bits: Option<Vec<u64>>,
    b_bits: Option<Vec<u64>>,
    c_bits: Option<Vec<u64>>,
    s_bits: Option<Vec<u64>>,
    /// `U·diag(s)·Vᵀ` from the incumbent's own SVD, row-major `m × n`.
    svd_reconstruction_bits: Option<Vec<u64>>,
    error: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    rows: usize,
    cols: usize,
    k: usize,
    exact_entries_compared: usize,
    toleranced_entries_compared: usize,
    max_relative_difference: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared_cases: usize,
    total_exact_entries: usize,
    total_toleranced_entries: usize,
    max_relative_difference: f64,
    pass: bool,
    timestamp_ms: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn emit_log(log: &DiffLog) {
    fs::create_dir_all(output_dir()).expect("create interpolative diff output dir");
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize interpolative diff log");
    fs::write(path, json).expect("write interpolative diff log");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

/// A deterministic, well-conditioned test matrix. Generated in Rust so the two arms are given
/// bit-identical input rather than each generating "the same" matrix from a shared seed.
fn matrix(rows: usize, cols: usize, kind: &str) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| {
                    let (x, y) = (i as f64, j as f64);
                    match kind {
                        // Exactly rank 2: every column is a combination of two.
                        "rank2" => (x + 1.0) * (y + 1.0) + (x - 1.0) * (2.0 * y - 3.0),
                        // Exactly rank 3.
                        "rank3" => {
                            (x + 1.0) * (y + 1.0) + x * x * (y - 2.0) + (2.0 * x - 1.0) * (y * y)
                        }
                        // Smoothly decaying spectrum — the case IDs are actually used on.
                        "decay" => 1.0 / (1.0 + x + 2.0 * y),
                        // Full rank, well separated.
                        _ => {
                            if i == j {
                                (i + 2) as f64
                            } else {
                                0.5 / (1.0 + (x - y).abs())
                            }
                        }
                    }
                })
                .collect()
        })
        .collect()
}

fn case(case_id: &str, rows: usize, cols: usize, k: usize, kind: &str) -> Case {
    let a = matrix(rows, cols, kind);
    Case {
        case_id: case_id.to_string(),
        rows,
        cols,
        a_bits: a
            .iter()
            .flat_map(|row| row.iter().map(|v| v.to_bits()))
            .collect(),
        k,
    }
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        points: vec![
            case("rank2-6x5-k2", 6, 5, 2, "rank2"),
            case("rank2-8x8-k3", 8, 8, 3, "rank2"),
            case("rank3-7x6-k3", 7, 6, 3, "rank3"),
            case("rank3-10x9-k4", 10, 9, 4, "rank3"),
            case("decay-6x6-k2", 6, 6, 2, "decay"),
            case("decay-9x7-k4", 9, 7, 4, "decay"),
            case("decay-12x10-k6", 12, 10, 6, "decay"),
            case("full-5x5-k3", 5, 5, 3, "full"),
            // k == n: proj is k×0 and the interpolation matrix is a pure permutation. An
            // implementation that assumed a non-empty proj falls over exactly here.
            case("full-6x4-k4", 6, 4, 4, "full"),
            case("decay-5x3-k3", 5, 3, 3, "decay"),
            // k == 1: the narrowest non-degenerate rank.
            case("decay-6x5-k1", 6, 5, 1, "decay"),
            // Tall and wide extremes.
            case("decay-20x4-k3", 20, 4, 3, "decay"),
            case("decay-4x14-k3", 4, 14, 3, "decay"),
        ],
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import struct
import sys
import numpy as np
import scipy.linalg.interpolative as ii

def bits(values):
    return [struct.unpack("<Q", struct.pack("<d", float(v)))[0] for v in values]

def unbits(values):
    return [struct.unpack("<d", struct.pack("<Q", int(v)))[0] for v in values]

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    try:
        m, n, k = int(case["rows"]), int(case["cols"]), int(case["k"])
        A = np.array(unbits(case["a_bits"]), dtype=float).reshape(m, n)

        # ONE decomposition, handed to both arms. Its randomness never leaves this block.
        idx, proj = ii.interp_decomp(A, k)
        # scipy's reconstruction entry points are Fortran extensions that require the ORIGINAL
        # ndarray for `perms`; a Python list of the same integers is rejected outright. So the
        # array is what scipy is called with, and the list exists only for transport.
        idx_list = [int(v) for v in np.asarray(idx).ravel()]
        proj = np.atleast_2d(np.asarray(proj, dtype=float))
        if proj.size == 0:
            proj = proj.reshape(k, 0)

        P = ii.reconstruct_interp_matrix(idx, proj)
        B = ii.reconstruct_skel_matrix(A, k, idx)
        C = ii.reconstruct_matrix_from_id(B, idx, proj)
        U, S, V = ii.id_to_svd(B, idx, proj)
        svd_rec = U @ np.diag(S) @ V.T

        vals = (list(np.asarray(P).ravel()) + list(np.asarray(B).ravel())
                + list(np.asarray(C).ravel()) + list(np.asarray(S).ravel())
                + list(np.asarray(svd_rec).ravel()))
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("case produced a non-finite value")

        points.append({
            "case_id": cid,
            "idx": idx_list,
            "proj_bits": bits(np.asarray(proj).ravel()),
            "proj_rows": int(proj.shape[0]),
            "proj_cols": int(proj.shape[1]),
            "p_bits": bits(np.asarray(P).ravel()),
            "b_bits": bits(np.asarray(B).ravel()),
            "c_bits": bits(np.asarray(C).ravel()),
            "s_bits": bits(np.asarray(S).ravel()),
            "svd_reconstruction_bits": bits(np.asarray(svd_rec).ravel()),
            "error": None,
        })
    except Exception as exc:
        points.append({
            "case_id": cid, "idx": None, "proj_bits": None, "proj_rows": None,
            "proj_cols": None, "p_bits": None, "b_bits": None, "c_bits": None,
            "s_bits": None, "svd_reconstruction_bits": None,
            "error": f"{type(exc).__name__}: {exc}",
        })
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize interpolative query");
    let mut child = match Command::new("python3")
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
                "failed to spawn python3 for interpolative oracle: {e}"
            );
            eprintln!("skipping interpolative oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open interpolative oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "interpolative oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping interpolative oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for interpolative oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "interpolative oracle failed: {stderr}"
        );
        eprintln!("skipping interpolative oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse interpolative oracle JSON"))
}

fn as_floats(bits: &[u64]) -> Vec<f64> {
    bits.iter().copied().map(f64::from_bits).collect()
}

fn reshape(flat: &[f64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|i| flat[i * cols..(i + 1) * cols].to_vec())
        .collect()
}

fn flatten(m: &[Vec<f64>]) -> Vec<f64> {
    m.iter().flat_map(|row| row.iter().copied()).collect()
}

/// Bit-exact comparison, for the two functions that do no arithmetic.
fn same_bits(ours: &[f64], theirs: &[u64]) -> bool {
    ours.len() == theirs.len() && ours.iter().zip(theirs).all(|(x, y)| x.to_bits() == *y)
}

/// Largest relative difference between two sequences, scaled by the magnitude present so that
/// a near-zero entry is not judged against an absolute floor it can never meet.
///
/// Returns `f64::INFINITY` on a length mismatch, so a shape error can never read as agreement.
fn max_relative_difference(ours: &[f64], theirs: &[f64]) -> f64 {
    if ours.len() != theirs.len() {
        return f64::INFINITY;
    }
    let scale = theirs
        .iter()
        .fold(0.0f64, |acc, v| acc.max(v.abs()))
        .max(1.0);
    ours.iter()
        .zip(theirs)
        .fold(0.0f64, |acc, (a, b)| acc.max((a - b).abs() / scale))
}

#[test]
fn diff_linalg_interpolative_reconstruct() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(
        oracle.points.len(),
        query.points.len(),
        "the oracle must answer every case"
    );

    let mut cases = Vec::new();
    let mut compared = 0usize;
    let mut total_exact = 0usize;
    let mut total_toleranced = 0usize;
    let mut worst = 0.0f64;

    for (case, arm) in query.points.iter().zip(&oracle.points) {
        assert_eq!(
            case.case_id, arm.case_id,
            "oracle answers must stay in order"
        );
        assert!(
            arm.error.is_none(),
            "case {} raised on the incumbent: {:?}. Every case here is one scipy supports; \
             an error means the query is malformed, not that the case may be skipped.",
            case.case_id,
            arm.error
        );
        let (
            Some(idx),
            Some(proj_bits),
            Some(proj_rows),
            Some(proj_cols),
            Some(p_bits),
            Some(b_bits),
            Some(c_bits),
            Some(s_bits),
            Some(svd_rec_bits),
        ) = (
            arm.idx.as_ref(),
            arm.proj_bits.as_ref(),
            arm.proj_rows,
            arm.proj_cols,
            arm.p_bits.as_ref(),
            arm.b_bits.as_ref(),
            arm.c_bits.as_ref(),
            arm.s_bits.as_ref(),
            arm.svd_reconstruction_bits.as_ref(),
        )
        else {
            panic!(
                "case {} came back with a null field and no error; a half-populated arm \
                 would compare vacuously",
                case.case_id
            );
        };

        let (m, n, k) = (case.rows, case.cols, case.k);
        assert_eq!(
            proj_rows, k,
            "case {}: oracle proj has wrong rank",
            case.case_id
        );
        assert_eq!(
            proj_cols,
            n - k,
            "case {}: oracle proj has wrong width",
            case.case_id
        );
        let a = reshape(&as_floats(&case.a_bits), m, n);
        let proj = reshape(&as_floats(proj_bits), proj_rows, proj_cols);

        // --- EXACT: pure placement, no arithmetic -----------------------------------------
        let p = reconstruct_interp_matrix(idx, &proj).unwrap_or_else(|e| {
            panic!(
                "case {}: reconstruct_interp_matrix failed: {e}",
                case.case_id
            )
        });
        assert_eq!(p.len(), k, "case {}: P must be k×n", case.case_id);
        assert!(
            same_bits(&flatten(&p), p_bits),
            "case {}: interpolation matrix differs from the incumbent's BIT-EXACTLY. \
             This function only places ones and copies proj entries, so any difference is a \
             wrong column, not rounding.\n  ours:  {:?}\n  scipy: {:?}",
            case.case_id,
            &flatten(&p)[..flatten(&p).len().min(16)],
            &as_floats(p_bits)[..p_bits.len().min(16)]
        );

        let b = reconstruct_skel_matrix(&a, k, idx).unwrap_or_else(|e| {
            panic!("case {}: reconstruct_skel_matrix failed: {e}", case.case_id)
        });
        assert!(
            same_bits(&flatten(&b), b_bits),
            "case {}: skeleton matrix differs from the incumbent's BIT-EXACTLY; it is a pure \
             column selection",
            case.case_id
        );
        total_exact += p_bits.len() + b_bits.len();

        // --- TOLERANCED: matrix product ----------------------------------------------------
        let c = reconstruct_matrix_from_id(&b, idx, &proj).unwrap_or_else(|e| {
            panic!(
                "case {}: reconstruct_matrix_from_id failed: {e}",
                case.case_id
            )
        });
        let c_diff = max_relative_difference(&flatten(&c), &as_floats(c_bits));
        assert!(
            c_diff <= REL_TOL,
            "case {}: reconstructed matrix differs by {c_diff:e}, above {REL_TOL:e}",
            case.case_id
        );

        // --- TOLERANCED: SVD, compared on values and reconstruction only -------------------
        let result = id_to_svd(&b, idx, &proj)
            .unwrap_or_else(|e| panic!("case {}: id_to_svd failed: {e}", case.case_id));
        assert_eq!(result.u.len(), m, "case {}: U must be m×k", case.case_id);
        assert_eq!(
            result.s.len(),
            k,
            "case {}: k singular values",
            case.case_id
        );
        assert_eq!(result.v.len(), n, "case {}: V must be n×k", case.case_id);

        let s_diff = max_relative_difference(&result.s, &as_floats(s_bits));
        assert!(
            s_diff <= REL_TOL,
            "case {}: singular values differ by {s_diff:e}, above {REL_TOL:e}.\n  ours:  {:?}\n  scipy: {:?}",
            case.case_id,
            result.s,
            as_floats(s_bits)
        );

        // U·diag(s)·Vᵀ, which is well defined even though U and V individually are not.
        let mut ours_rec = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                ours_rec.push(
                    (0..k)
                        .map(|t| result.u[i][t] * result.s[t] * result.v[j][t])
                        .sum::<f64>(),
                );
            }
        }
        let rec_diff = max_relative_difference(&ours_rec, &as_floats(svd_rec_bits));
        assert!(
            rec_diff <= REL_TOL,
            "case {}: SVD reconstruction differs by {rec_diff:e}, above {REL_TOL:e}",
            case.case_id
        );
        // And it must equal the matrix product route, which is the whole claim of id_to_svd.
        let self_consistency = max_relative_difference(&ours_rec, &flatten(&c));
        assert!(
            self_consistency <= REL_TOL,
            "case {}: id_to_svd disagrees with reconstruct_matrix_from_id by {self_consistency:e}",
            case.case_id
        );

        total_toleranced += c_bits.len() + s_bits.len() + svd_rec_bits.len();
        let case_worst = c_diff.max(s_diff).max(rec_diff);
        worst = worst.max(case_worst);
        compared += 1;
        cases.push(CaseDiff {
            case_id: case.case_id.clone(),
            rows: m,
            cols: n,
            k,
            exact_entries_compared: p_bits.len() + b_bits.len(),
            toleranced_entries_compared: c_bits.len() + s_bits.len() + svd_rec_bits.len(),
            max_relative_difference: case_worst,
            pass: true,
        });
    }

    emit_log(&DiffLog {
        test_id: "diff_linalg_interpolative_reconstruct".to_string(),
        category: "linalg.interpolative".to_string(),
        case_count: query.points.len(),
        compared_cases: compared,
        total_exact_entries: total_exact,
        total_toleranced_entries: total_toleranced,
        max_relative_difference: worst,
        pass: true,
        timestamp_ms: timestamp_ms(),
        cases,
    });

    // GREEN-BY-SKIPPING GUARD.
    eprintln!(
        "interpolative diff: compared={compared} exact_entries={total_exact} \
         toleranced_entries={total_toleranced} max_rel_diff={worst:e}"
    );
    assert_eq!(
        compared,
        query.points.len(),
        "every case must be compared, not skipped"
    );
    assert!(
        total_exact > 500,
        "only {total_exact} bit-exact entries compared; the exact half barely ran"
    );
    assert!(
        total_toleranced > 500,
        "only {total_toleranced} toleranced entries compared"
    );
}

/// MUST-HIT / MUST-MISS control for the bit-exact comparator.
#[test]
fn bit_comparator_rejects_the_differences_it_exists_to_catch() {
    let w = |xs: &[f64]| -> Vec<u64> { xs.iter().map(|x| x.to_bits()).collect() };
    assert!(same_bits(&[1.0, 0.0, 2.5], &w(&[1.0, 0.0, 2.5])));
    assert!(!same_bits(&[1.0, 0.0], &w(&[1.0, 0.0, 2.5])), "length");
    assert!(!same_bits(&[1.0, 2.5], &w(&[1.0, 2.0])), "value");
    assert!(!same_bits(&[0.0], &w(&[-0.0])), "signed zero");
    // THE failure mode of these functions: right numbers, wrong order.
    assert!(
        !same_bits(&[1.0, 0.0, 2.5], &w(&[1.0, 2.5, 0.0])),
        "permuted"
    );
    let one_ulp = 1.0f64.to_bits() + 1;
    assert!(!same_bits(&[1.0], &[one_ulp]), "one ULP");
}

/// MUST-HIT / MUST-MISS control for the toleranced comparator, including the case that would
/// otherwise let a shape error read as perfect agreement.
#[test]
fn relative_comparator_rejects_the_differences_it_exists_to_catch() {
    assert_eq!(max_relative_difference(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    assert!(max_relative_difference(&[1.0, 2.0], &[1.0, 2.0 + 1e-15]) < REL_TOL);

    // MUST-MISS: a difference just above tolerance, at the scale of the data.
    assert!(max_relative_difference(&[100.0, 0.0], &[100.0, 1e-8]) > REL_TOL);
    // MUST-MISS: a length mismatch is infinite, never zero.
    assert_eq!(
        max_relative_difference(&[1.0], &[1.0, 2.0]),
        f64::INFINITY,
        "a shape mismatch must never read as agreement"
    );
    assert_eq!(max_relative_difference(&[], &[1.0]), f64::INFINITY);
    // Two empty sequences agree, and must not be infinite.
    assert_eq!(max_relative_difference(&[], &[]), 0.0);
    // MUST-MISS: a wholly wrong value.
    assert!(max_relative_difference(&[1.0], &[2.0]) > REL_TOL);
}
