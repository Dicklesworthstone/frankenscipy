#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `fsci_opt::LbfgsInvHessProduct` against
//! `scipy.optimize.LbfgsInvHessProduct`.
//!
//! ## What is compared, and why it is a tolerance rather than bits
//!
//! Both arms run the same two-loop recursion over the same correction pairs, but the inner
//! products differ in summation order — NumPy dispatches `np.dot` to BLAS, which reassociates,
//! while ours is an ordered fold. So the comparison is relative, at 1e-12. The correction pairs
//! themselves cross as IEEE-754 bit patterns, so the two arms are provably operating on
//! identical input and any difference is the RECURSION, not the data.
//!
//! Three things are checked per case:
//!
//!   * `matvec` against several probe vectors, including the unit vectors;
//!   * `todense`, entry by entry;
//!   * that our dense form and our `matvec` agree with each other, which catches a drift
//!     between the two paths that would otherwise cancel out against the incumbent.
//!
//! ## H₀ = I is the whole point of this test
//!
//! `fsci_opt::minimize::lbfgs_two_loop` — the crate's SEARCH-DIRECTION recursion — scales the
//! initial inverse Hessian by `γ = sᵀy / yᵀy`. SciPy's `LbfgsInvHessProduct` does not; it starts
//! from the identity. Both are correct for their own purpose and they compute DIFFERENT
//! operators from the same history. An implementation that reached for the existing helper
//! would produce entirely reasonable numbers that disagree with the incumbent on every case
//! here, which is exactly what this test exists to prevent.
//!
//! ## One deliberate divergence, not covered by any case
//!
//! SciPy does not check the curvature condition `sᵢ · yᵢ > 0`. Measured on scipy 1.17.1,
//! `sk = [[1, 0]]`, `yk = [[0, 1]]` yields `rho = [inf]` and an all-NaN `matvec` and `todense`,
//! announced only by a `RuntimeWarning`. We return an error instead. That case is therefore
//! absent from the query — comparing an error against NaN would test nothing — and is pinned
//! on our side by `inputs_that_would_silently_produce_nan_are_rejected` in the unit tests.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_opt::LbfgsInvHessProduct;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

/// Relative tolerance. Loose enough for BLAS reassociation over these correction counts, tight
/// enough that a wrong recursion cannot pass — a gamma-scaled H₀, for instance, moves entries
/// by whole percent.
const REL_TOL: f64 = 1e-12;

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    n_corrs: usize,
    n: usize,
    /// Row-major `n_corrs × n`, as IEEE-754 bit patterns.
    sk_bits: Vec<u64>,
    yk_bits: Vec<u64>,
    /// Row-major `probe_count × n` vectors to apply the operator to.
    probes_bits: Vec<u64>,
    probe_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Row-major `probe_count × n`.
    matvec_bits: Option<Vec<u64>>,
    /// Row-major `n × n`.
    dense_bits: Option<Vec<u64>>,
    error: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    n_corrs: usize,
    n: usize,
    entries_compared: usize,
    max_relative_difference: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared_cases: usize,
    total_entries_compared: usize,
    max_relative_difference: f64,
    pass: bool,
    timestamp_ms: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn emit_log(log: &DiffLog) {
    fs::create_dir_all(output_dir()).expect("create lbfgs inv hess diff output dir");
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize lbfgs inv hess diff log");
    fs::write(path, json).expect("write lbfgs inv hess diff log");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn flat_bits(m: &[Vec<f64>]) -> Vec<u64> {
    m.iter()
        .flat_map(|row| row.iter().map(|v| v.to_bits()))
        .collect()
}

fn as_floats(bits: &[u64]) -> Vec<f64> {
    bits.iter().copied().map(f64::from_bits).collect()
}

fn reshape(bits: &[u64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|i| as_floats(&bits[i * cols..(i + 1) * cols]))
        .collect()
}

/// Correction pairs that satisfy the curvature condition by construction.
///
/// `y = D s` with `D` diagonal and positive makes `s · y = Σ dᵢ sᵢ² > 0` for any non-zero `s`,
/// so every case is one the incumbent handles without its unchecked-`rho` failure mode.
fn corrections(n_corrs: usize, n: usize, flavour: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let sk: Vec<Vec<f64>> = (0..n_corrs)
        .map(|k| {
            (0..n)
                .map(|i| match flavour {
                    0 => ((i + 1) as f64).sin() + 0.5 * (k + 1) as f64,
                    1 => ((i as f64) - (n as f64) / 2.0) * (k as f64 + 1.0) + 0.25,
                    2 => 1.0 / (1.0 + i as f64 + k as f64),
                    _ => ((i * 7 + k * 3) as f64).cos() * 2.0 + 3.0,
                })
                .collect()
        })
        .collect();
    let yk: Vec<Vec<f64>> = sk
        .iter()
        .enumerate()
        .map(|(k, s)| {
            s.iter()
                .enumerate()
                .map(|(i, v)| v * (0.5 + i as f64 * 0.25 + k as f64 * 0.1))
                .collect()
        })
        .collect();
    (sk, yk)
}

fn probes(count: usize, n: usize) -> Vec<Vec<f64>> {
    (0..count)
        .map(|p| {
            (0..n)
                .map(|i| {
                    if p < n {
                        // The unit vectors, whose images are the dense form's columns.
                        if i == p { 1.0 } else { 0.0 }
                    } else {
                        ((i + p) as f64).cos() * (1.0 + p as f64)
                    }
                })
                .collect()
        })
        .collect()
}

fn case(case_id: &str, n_corrs: usize, n: usize, flavour: usize) -> Case {
    let (sk, yk) = corrections(n_corrs, n, flavour);
    // Every unit vector, plus three arbitrary directions.
    let probe_count = n + 3;
    let probe_rows = probes(probe_count, n);
    Case {
        case_id: case_id.to_string(),
        n_corrs,
        n,
        sk_bits: flat_bits(&sk),
        yk_bits: flat_bits(&yk),
        probes_bits: flat_bits(&probe_rows),
        probe_count,
    }
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        points: vec![
            // A single correction: the closed-form BFGS update.
            case("1corr-3", 1, 3, 0),
            case("1corr-6", 1, 6, 1),
            // The usual case: several corrections, order-sensitive.
            case("3corr-4", 3, 4, 0),
            case("5corr-5", 5, 5, 0),
            case("5corr-5-alt", 5, 5, 1),
            case("8corr-6", 8, 6, 2),
            // L-BFGS-B's default memory is 10 corrections.
            case("10corr-7", 10, 7, 0),
            case("10corr-12", 10, 12, 3),
            // More corrections than dimensions, and the reverse.
            case("12corr-3", 12, 3, 1),
            case("2corr-15", 2, 15, 2),
            // n = 1, where every vector is a scalar and an index slip has nowhere to hide.
            case("4corr-1", 4, 1, 0),
            // Widely varying magnitudes across corrections, which stresses the rho scaling.
            case("6corr-9", 6, 9, 3),
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
from scipy.optimize import LbfgsInvHessProduct

def bits(values):
    return [struct.unpack("<Q", struct.pack("<d", float(v)))[0] for v in values]

def unbits(values):
    return [struct.unpack("<d", struct.pack("<Q", int(v)))[0] for v in values]

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    try:
        n_corrs, n = int(case["n_corrs"]), int(case["n"])
        pc = int(case["probe_count"])
        sk = np.array(unbits(case["sk_bits"]), dtype=float).reshape(n_corrs, n)
        yk = np.array(unbits(case["yk_bits"]), dtype=float).reshape(n_corrs, n)
        probes = np.array(unbits(case["probes_bits"]), dtype=float).reshape(pc, n)

        op = LbfgsInvHessProduct(sk, yk)
        mv = np.stack([np.asarray(op.matvec(probes[p])).ravel() for p in range(pc)])
        dense = np.asarray(op.todense())

        vals = list(mv.ravel()) + list(dense.ravel())
        if not all(math.isfinite(v) for v in vals):
            # Every case here satisfies the curvature condition, so a non-finite value would
            # mean the query is malformed rather than that the incumbent misbehaved.
            raise ValueError("case produced a non-finite value")

        points.append({
            "case_id": cid,
            "matvec_bits": bits(mv.ravel()),
            "dense_bits": bits(dense.ravel()),
            "error": None,
        })
    except Exception as exc:
        points.append({
            "case_id": cid, "matvec_bits": None, "dense_bits": None,
            "error": f"{type(exc).__name__}: {exc}",
        })
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize lbfgs inv hess query");
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
                "failed to spawn python3 for lbfgs inv hess oracle: {e}"
            );
            eprintln!("skipping lbfgs inv hess oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open lbfgs inv hess oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "lbfgs inv hess oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping lbfgs inv hess oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for lbfgs inv hess oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "lbfgs inv hess oracle failed: {stderr}"
        );
        eprintln!("skipping lbfgs inv hess oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse lbfgs inv hess oracle JSON"))
}

/// Largest relative difference, scaled by the magnitude actually present so a near-zero entry
/// is not judged against a floor it cannot meet. Infinite on a length mismatch, so a shape
/// error can never read as agreement.
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
fn diff_optimize_lbfgs_inv_hess() {
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
    let mut total_entries = 0usize;
    let mut worst = 0.0f64;

    for (case, arm) in query.points.iter().zip(&oracle.points) {
        assert_eq!(
            case.case_id, arm.case_id,
            "oracle answers must stay in order"
        );
        assert!(
            arm.error.is_none(),
            "case {} raised on the incumbent: {:?}. Every case satisfies the curvature \
             condition, so an error means the query is malformed.",
            case.case_id,
            arm.error
        );
        let (Some(matvec_bits), Some(dense_bits)) =
            (arm.matvec_bits.as_ref(), arm.dense_bits.as_ref())
        else {
            panic!(
                "case {} came back with a null field and no error; a half-populated arm \
                 would compare vacuously",
                case.case_id
            );
        };

        let (n_corrs, n, probe_count) = (case.n_corrs, case.n, case.probe_count);
        let sk = reshape(&case.sk_bits, n_corrs, n);
        let yk = reshape(&case.yk_bits, n_corrs, n);
        let probe_rows = reshape(&case.probes_bits, probe_count, n);

        let operator = LbfgsInvHessProduct::new(sk, yk)
            .unwrap_or_else(|e| panic!("case {}: construction failed: {e}", case.case_id));
        assert_eq!(operator.shape(), (n, n), "case {}", case.case_id);
        assert_eq!(operator.n_corrs(), n_corrs, "case {}", case.case_id);

        let ours_matvec: Vec<f64> = probe_rows
            .iter()
            .flat_map(|p| {
                operator
                    .matvec(p)
                    .unwrap_or_else(|e| panic!("case {}: matvec failed: {e}", case.case_id))
            })
            .collect();
        let matvec_diff = max_relative_difference(&ours_matvec, &as_floats(matvec_bits));
        assert!(
            matvec_diff <= REL_TOL,
            "case {}: matvec differs by {matvec_diff:e}, above {REL_TOL:e}",
            case.case_id
        );

        let dense = operator.todense();
        let ours_dense: Vec<f64> = dense.iter().flat_map(|row| row.iter().copied()).collect();
        let dense_diff = max_relative_difference(&ours_dense, &as_floats(dense_bits));
        assert!(
            dense_diff <= REL_TOL,
            "case {}: todense differs by {dense_diff:e}, above {REL_TOL:e}",
            case.case_id
        );

        // SELF-CONSISTENCY, which the comparison against the incumbent cannot supply: our dense
        // form and our matvec must agree with EACH OTHER. If both drifted together they could
        // still match scipy on one path and not the other.
        for (j, probe) in probe_rows.iter().enumerate().take(n) {
            // The first `n` probes are the unit vectors, so their images are dense's columns.
            let expected: Vec<f64> = (0..n).map(|i| dense[i][j]).collect();
            let got = operator.matvec(probe).expect("right length");
            let column_diff = max_relative_difference(&got, &expected);
            assert!(
                column_diff == 0.0,
                "case {}: our dense column {j} disagrees with our own matvec by {column_diff:e}",
                case.case_id
            );
        }

        let case_worst = matvec_diff.max(dense_diff);
        worst = worst.max(case_worst);
        total_entries += matvec_bits.len() + dense_bits.len();
        compared += 1;
        cases.push(CaseDiff {
            case_id: case.case_id.clone(),
            n_corrs,
            n,
            entries_compared: matvec_bits.len() + dense_bits.len(),
            max_relative_difference: case_worst,
            pass: true,
        });
    }

    emit_log(&DiffLog {
        test_id: "diff_optimize_lbfgs_inv_hess".to_string(),
        category: "optimize".to_string(),
        case_count: query.points.len(),
        compared_cases: compared,
        total_entries_compared: total_entries,
        max_relative_difference: worst,
        pass: true,
        timestamp_ms: timestamp_ms(),
        cases,
    });

    // GREEN-BY-SKIPPING GUARD.
    eprintln!(
        "lbfgs inv hess diff: compared={compared} entries={total_entries} \
         max_rel_diff={worst:e}"
    );
    assert_eq!(
        compared,
        query.points.len(),
        "every case must be compared, not skipped"
    );
    assert!(
        total_entries > 1000,
        "only {total_entries} entries compared; the larger cases did not run"
    );
}

/// MUST-HIT / MUST-MISS control for the comparator, including the shape case that would
/// otherwise let a length error read as perfect agreement.
#[test]
fn relative_comparator_rejects_the_differences_it_exists_to_catch() {
    assert_eq!(max_relative_difference(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    assert_eq!(max_relative_difference(&[], &[]), 0.0);
    assert!(max_relative_difference(&[1.0, 2.0], &[1.0, 2.0 + 1e-16]) < REL_TOL);

    assert!(max_relative_difference(&[100.0, 0.0], &[100.0, 1e-9]) > REL_TOL);
    assert!(max_relative_difference(&[1.0], &[2.0]) > REL_TOL);
    assert_eq!(
        max_relative_difference(&[1.0], &[1.0, 2.0]),
        f64::INFINITY,
        "a shape mismatch must never read as agreement"
    );
    assert_eq!(max_relative_difference(&[1.0, 2.0], &[1.0]), f64::INFINITY);
}

/// The divergence this suite deliberately does not compare, asserted on our side so it cannot
/// quietly change: SciPy returns NaN where we return an error.
#[test]
fn the_curvature_condition_is_enforced_here_even_though_the_incumbent_ignores_it() {
    // scipy 1.17.1 on this input: rho = [inf], matvec = [nan, nan], todense all NaN.
    assert!(LbfgsInvHessProduct::new(vec![vec![1.0, 0.0]], vec![vec![0.0, 1.0]]).is_err());
    assert!(LbfgsInvHessProduct::new(vec![vec![1.0, 0.0]], vec![vec![-1.0, 0.0]]).is_err());
    // And a valid pair still constructs, so the guard is not simply rejecting everything.
    assert!(LbfgsInvHessProduct::new(vec![vec![1.0, 0.0]], vec![vec![2.0, 0.0]]).is_ok());
}
