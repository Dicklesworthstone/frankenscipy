#![forbid(unsafe_code)]
//! Public-API regression and live SciPy checks for sparse exponential actions.

use std::io::Write;
use std::process::Stdio;

use fsci_sparse::{
    CooMatrix, CsrMatrix, FormatConvertible, Shape2D, SparseError, expm_multiply,
    expm_multiply_interval, expm_multiply_matrix, expm_multiply_matrix_interval,
};

// The resolver is std-only. Including it keeps this focused integration target
// independent of the conformance crate and the runtime benchmark feature.
#[path = "../../fsci-runtime/src/scipy_incumbent.rs"]
mod scipy_incumbent;

fn diag(values: &[f64]) -> CsrMatrix {
    let n = values.len();
    CooMatrix::from_triplets(
        Shape2D::new(n, n),
        values.to_vec(),
        (0..n).collect(),
        (0..n).collect(),
        true,
    )
    .expect("coo")
    .to_csr()
    .expect("csr")
}

#[test]
fn implicit_zero_diagonal_is_included_in_the_shifted_norm() {
    // A = diag(0, 64, ..., 64), mu = 63. The old norm only visited stored
    // diagonal entries and found 1 instead of 63. Its unscaled Taylor series
    // for exp(-63) then suffered catastrophic cancellation, despite exp(A)b=b.
    let n = 64;
    let a = CooMatrix::from_triplets(
        Shape2D::new(n, n),
        vec![64.0; n - 1],
        (1..n).collect(),
        (1..n).collect(),
        true,
    )
    .unwrap()
    .to_csr()
    .unwrap();
    let mut b = vec![0.0; n];
    b[0] = 1.0;
    let got = expm_multiply(&a, &b, 1.0, 0.0).unwrap();
    assert!((got[0] - 1.0).abs() < 2e-13, "{got:?}");
    assert!(got[1..].iter().all(|value| *value == 0.0));
    let explicit = diag(&[vec![0.0], vec![64.0; n - 1]].concat());
    let with_zero = expm_multiply(&explicit, &b, 1.0, 0.0).unwrap();
    assert_eq!(got, with_zero, "stored zeros cannot change the answer");
}

#[test]
fn block_action_matches_independent_columns_at_different_scales() {
    let a = CooMatrix::from_triplets(
        Shape2D::new(3, 3),
        vec![-2.0, 0.4, -1.0, 0.7, -3.0],
        vec![0, 0, 1, 2, 2],
        vec![0, 1, 1, 0, 2],
        true,
    )
    .unwrap()
    .to_csr()
    .unwrap();
    let b = vec![
        vec![1e100, 1e-100, 0.0],
        vec![0.0, -2e-100, 0.0],
        vec![-1e100, 3e-100, 0.0],
    ];
    for t in [-0.5, 0.0, 1.25] {
        let block = expm_multiply_matrix(&a, &b, t, 0.0).unwrap();
        for k in 0..3 {
            let column: Vec<_> = b.iter().map(|row| row[k]).collect();
            let separate = expm_multiply(&a, &column, t, 0.0).unwrap();
            for (row, want) in block.iter().zip(separate) {
                assert!((row[k] - want).abs() <= 2e-14 * want.abs().max(1e-110));
            }
        }
    }
    assert_eq!(
        expm_multiply_matrix(&a, &[vec![], vec![], vec![]], 1.0, 0.0).unwrap(),
        vec![Vec::<f64>::new(); 3]
    );
}

#[test]
fn scaled_products_preserve_finite_exponential_actions() {
    for (entry, rhs, time, want) in [(1e308, 2.0, 1e-308, 2.0), (1e-200, 1e300, 1e-300, 1e-200)] {
        let a = CooMatrix::from_triplets(Shape2D::new(2, 2), vec![entry], vec![0], vec![1], true)
            .unwrap()
            .to_csr()
            .unwrap();
        let result = expm_multiply(&a, &[0.0, rhs], time, 0.0).unwrap();
        assert!((result[0] / want - 1.0).abs() < 2e-15, "{result:?}");
        assert_eq!(result[1], rhs);
    }
    let result = expm_multiply(&diag(&[-1000.0]), &[1e300], 1.0, 0.0).unwrap();
    let want = (-1000.0 + 1e300_f64.ln()).exp();
    assert!((result[0] / want - 1.0).abs() < 1e-13);
}

#[test]
fn interval_samples_do_not_inherit_underflow_from_previous_times() {
    let samples =
        expm_multiply_interval(&diag(&[-1000.0]), &[1.0], 1.0, 0.5, 2, true, 0.0).unwrap();
    assert_eq!(samples[0], vec![0.0]);
    assert!((samples[1][0] / (-500.0_f64).exp() - 1.0).abs() < 1e-14);
    let zero = diag(&[0.0]);
    assert_eq!(
        expm_multiply_interval(&zero, &[2.0], 1e308, -1e308, 2, true, 0.0).unwrap(),
        vec![vec![2.0]; 2]
    );
}

#[test]
fn identity_and_zero_actions_do_not_require_a_representable_norm() {
    let a = CooMatrix::from_triplets(
        Shape2D::new(2, 2),
        vec![1e308, 1e308, 1e308, -1e308],
        vec![0, 0, 1, 1],
        vec![0, 1, 0, 1],
        true,
    )
    .unwrap()
    .to_csr()
    .unwrap();
    assert_eq!(
        expm_multiply(&a, &[1.0, 2.0], 0.0, 0.0).unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        expm_multiply(&a, &[0.0, 0.0], 1.0, 0.0).unwrap(),
        vec![0.0, 0.0]
    );
    assert!(expm_multiply(&a, &[1.0, 2.0], 1.0, 0.0).is_err());
}

#[test]
fn interval_samples_match_closed_form_in_both_directions() {
    let d = [-2.0_f64, 0.0, 0.75];
    let a = diag(&d);
    let b = [1.0, -0.0, -2.0];
    for (start, stop, endpoint) in [(-1.0, 2.0, true), (2.0, -1.0, false), (0.5, 0.5, true)] {
        let samples = expm_multiply_interval(&a, &b, start, stop, 7, endpoint, 0.0).unwrap();
        assert_eq!(samples.len(), 7);
        for (index, sample) in samples.iter().enumerate() {
            let t = start + (stop - start) * index as f64 / if endpoint { 6.0 } else { 7.0 };
            for i in 0..3 {
                let want = (t * d[i]).exp() * b[i];
                assert!((sample[i] - want).abs() <= 2e-13 * want.abs().max(1.0));
            }
            if t == 0.0 {
                assert_eq!(
                    sample.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn block_interval_propagates_nondiagonal_systems() {
    // A nilpotent chain: exp(t*A) = I + t*A + t^2*A^2/2, an independent
    // closed form that exercises coupled states and all three output axes.
    let a = CooMatrix::from_triplets(
        Shape2D::new(3, 3),
        vec![2.0, -3.0],
        vec![0, 1],
        vec![1, 2],
        true,
    )
    .unwrap()
    .to_csr()
    .unwrap();
    let b = vec![vec![1.0, -2.0], vec![2.0, 0.5], vec![-1.0, 3.0]];
    let samples = expm_multiply_matrix_interval(&a, &b, -0.5, 1.5, 9, true, 0.0).unwrap();
    for (index, sample) in samples.iter().enumerate() {
        let t = -0.5 + index as f64 / 4.0;
        for k in 0..2 {
            let want = [
                b[0][k] + 2.0 * t * b[1][k] - 3.0 * t * t * b[2][k],
                b[1][k] - 3.0 * t * b[2][k],
                b[2][k],
            ];
            for i in 0..3 {
                assert!(
                    (sample[i][k] - want[i]).abs() < 3e-13,
                    "t={t} row={i} column={k}"
                );
            }
        }
    }
}

#[test]
fn block_and_interval_validation_and_scaling_budget() {
    let a = diag(&[1.0, -1.0]);
    assert!(expm_multiply_matrix(&a, &[vec![1.0], vec![2.0, 3.0]], 1.0, 0.0).is_err());
    assert!(expm_multiply_matrix(&a, &[vec![1.0]], 1.0, 0.0).is_err());
    assert!(expm_multiply_interval(&a, &[1.0, 2.0], 0.0, 1.0, 1, true, 0.0).is_err());
    assert!(expm_multiply_interval(&a, &[1.0, 2.0], 0.0, f64::NAN, 3, true, 0.0).is_err());
    assert!(expm_multiply(&a, &[1.0, 2.0], 1.0, f64::INFINITY).is_err());
    assert!(expm_multiply(&diag(&[f64::NAN, 1.0]), &[1.0, 2.0], 1.0, 0.0).is_err());
    assert!(matches!(
        expm_multiply(&a, &[1.0, 2.0], 1e10, 0.0),
        Err(SparseError::ResourceExhausted { .. })
    ));
    // A zero block needs no scaling work, including at extreme finite times.
    assert_eq!(
        expm_multiply(&a, &[0.0, -0.0], 1e10, 0.0).unwrap(),
        vec![0.0, -0.0]
    );
}

#[derive(Clone)]
struct ActionMatrix {
    case_id: &'static str,
    n: usize,
    triplets: Vec<(usize, usize, f64)>,
}

struct ActionCase {
    matrix: ActionMatrix,
    b: Vec<Vec<f64>>,
    endpoint: bool,
}

fn action_cases() -> Vec<ActionCase> {
    let matrices = vec![
        ActionMatrix {
            case_id: "3x3_spd_tridiag",
            n: 3,
            triplets: vec![
                (0, 0, 1.0),
                (0, 1, 0.5),
                (1, 0, 0.5),
                (1, 1, 1.0),
                (1, 2, 0.5),
                (2, 1, 0.5),
                (2, 2, 1.0),
            ],
        },
        ActionMatrix {
            case_id: "4x4_diag",
            n: 4,
            triplets: (0..4).map(|i| (i, i, 0.5 + i as f64 * 0.25)).collect(),
        },
        ActionMatrix {
            case_id: "3x3_skew_sym",
            n: 3,
            triplets: vec![
                (0, 1, 1.0),
                (1, 0, -1.0),
                (0, 2, 0.5),
                (2, 0, -0.5),
                (1, 2, 0.3),
                (2, 1, -0.3),
            ],
        },
        ActionMatrix {
            case_id: "4x4_dense_small",
            n: 4,
            triplets: vec![
                (0, 0, 0.5),
                (0, 1, 0.2),
                (1, 0, 0.2),
                (1, 1, 0.5),
                (1, 2, 0.2),
                (2, 1, 0.2),
                (2, 2, 0.5),
                (2, 3, 0.2),
                (3, 2, 0.2),
                (3, 3, 0.5),
            ],
        },
        ActionMatrix {
            case_id: "implicit_zero_diagonal",
            n: 64,
            triplets: (1..64).map(|i| (i, i, 64.0)).collect(),
        },
    ];
    let mut cases = Vec::new();
    for matrix in matrices {
        let b: Vec<_> = (0..matrix.n)
            .map(|i| {
                if matrix.case_id == "implicit_zero_diagonal" {
                    if i == 0 {
                        vec![1.0, -2.0]
                    } else {
                        vec![0.0, 0.0]
                    }
                } else {
                    vec![(i as f64 + 0.3).sin(), (2.0 * i as f64 - 0.7).cos()]
                }
            })
            .collect();
        for endpoint in [false, true] {
            cases.push(ActionCase {
                matrix: matrix.clone(),
                b: b.clone(),
                endpoint,
            });
        }
    }
    cases
}

#[test]
fn diff_sparse_expm_block_and_interval_actions() {
    let required = std::env::var_os("FSCI_REQUIRE_SCIPY_ORACLE").is_some();
    let incumbent = match scipy_incumbent::ScipyIncumbent::resolve() {
        Ok(incumbent) => incumbent,
        Err(error) => {
            assert!(!required, "required SciPy oracle unavailable: {error}");
            eprintln!("SKIP sparse exponential actions: {error}; compared_cases=0");
            return;
        }
    };
    eprintln!("{}", incumbent.provenance_line());
    if let Some(reason) = incumbent.disagreement() {
        assert!(!required, "required pinned SciPy oracle refused: {reason}");
        eprintln!("SKIP sparse exponential actions: {reason}; compared_cases=0");
        return;
    }
    let cases = action_cases();
    assert_eq!(cases.len(), 10);
    let query: Vec<_> = cases
        .iter()
        .map(|case| {
            serde_json::json!({
                "matrix": {
                    "case_id": case.matrix.case_id,
                    "n": case.matrix.n,
                    "triplets": case.matrix.triplets,
                },
                "b": case.b,
                "endpoint": case.endpoint,
            })
        })
        .collect();
    let script = r#"
import json, sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
answer = []
for case in json.load(sys.stdin):
    matrix = case['matrix']
    entries = matrix['triplets']
    rows, cols, data = zip(*entries)
    A = csr_matrix((data, (rows, cols)), shape=(matrix['n'], matrix['n']))
    B = np.asarray(case['b'])
    kw = dict(start=-0.25, stop=1.25, num=7, endpoint=case['endpoint'])
    answer.append(dict(
        block=expm_multiply(0.7*A, B).ravel().tolist(),
        block_interval=expm_multiply(A, B, **kw).ravel().tolist(),
        vector_interval=expm_multiply(A, B[:, 0], **kw).ravel().tolist()))
print(json.dumps(answer, allow_nan=False))
"#;
    let mut child = incumbent
        .command()
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn proven SciPy oracle");
    child
        .stdin
        .take()
        .expect("oracle stdin")
        .write_all(&serde_json::to_vec(&query).expect("serialize action cases"))
        .expect("write action cases");
    let output = child.wait_with_output().expect("wait for SciPy oracle");
    assert!(
        output.status.success(),
        "SciPy oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("SciPy oracle JSON");
    let oracle = oracle.as_array().expect("oracle results array");
    assert_eq!(oracle.len(), cases.len());
    let mut compared = [0_usize; 3];
    let mut scipy_compared = [0_usize; 3];
    let mut exact_compared = [0_usize; 3];
    let mut reference_losses = 0_usize;
    for (case, expected) in cases.iter().zip(oracle) {
        let matrix = &case.matrix;
        let a = CooMatrix::from_triplets(
            Shape2D::new(matrix.n, matrix.n),
            matrix.triplets.iter().map(|entry| entry.2).collect(),
            matrix.triplets.iter().map(|entry| entry.0).collect(),
            matrix.triplets.iter().map(|entry| entry.1).collect(),
            true,
        )
        .unwrap()
        .to_csr()
        .unwrap();
        // Select this reference from a structural proof BEFORE observing any
        // SciPy disagreement: every product A_ij * B_jk is exactly zero.
        // Thus AB=0 and exp(tA)B=B for every finite t. No floating-point
        // cancellation or approximate matrix multiplication is used as proof.
        let stationary = a.data().iter().zip(a.indices()).all(|(&value, &column)| {
            value.is_finite() && (value == 0.0 || case.b[column].iter().all(|&rhs| rhs == 0.0))
        }) && case.b.iter().flatten().all(|value| value.is_finite());
        let column: Vec<_> = case.b.iter().map(|row| row[0]).collect();
        let block = expm_multiply_matrix(&a, &case.b, 0.7, 0.0).unwrap();
        let block_interval =
            expm_multiply_matrix_interval(&a, &case.b, -0.25, 1.25, 7, case.endpoint, 0.0).unwrap();
        let vector_interval =
            expm_multiply_interval(&a, &column, -0.25, 1.25, 7, case.endpoint, 0.0).unwrap();
        let id = format!("{}_endpoint_{}", matrix.case_id, case.endpoint);
        for (index, (arm, actual)) in [
            ("block", block.into_iter().flatten().collect::<Vec<_>>()),
            (
                "block_interval",
                block_interval.into_iter().flatten().flatten().collect(),
            ),
            (
                "vector_interval",
                vector_interval.into_iter().flatten().collect(),
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let reference: Vec<f64> = expected[arm]
                .as_array()
                .expect("every oracle arm must contain an array")
                .iter()
                .map(|value| value.as_f64().expect("finite oracle number"))
                .collect();
            assert!(!reference.is_empty(), "empty oracle arm {arm}/{id}");
            assert_eq!(actual.len(), reference.len(), "shape mismatch {arm}/{id}");
            assert!(
                actual
                    .iter()
                    .chain(&reference)
                    .all(|value| value.is_finite()),
                "nonfinite value in {arm}/{id}"
            );
            // SciPy 1.17.1 loses accuracy on the implicit-zero-diagonal
            // stationary fixture. Preserve that fixture and its raw oracle
            // values, but gate it against the proved exact identity. All
            // other fixtures remain gated directly against live SciPy.
            let exact = stationary.then(|| {
                let initial: Vec<f64> = if index == 2 {
                    column.clone()
                } else {
                    case.b.iter().flatten().copied().collect()
                };
                initial.repeat(if index == 0 { 1 } else { 7 })
            });
            let target = exact.as_ref().unwrap_or(&reference);
            assert_eq!(
                target.len(),
                actual.len(),
                "exact-reference shape {arm}/{id}"
            );
            let scaled_error = |lhs: &[f64], rhs: &[f64]| {
                lhs.iter()
                    .zip(rhs)
                    .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
                    .fold(0.0_f64, f64::max)
            };
            let error = scaled_error(&actual, target);
            assert!(
                error <= 1e-11,
                "expm_multiply {arm}/{id}: max_scaled_error={error:.3e}"
            );
            compared[index] += 1;
            if stationary {
                exact_compared[index] += 1;
                let reference_error = scaled_error(&reference, target);
                reference_losses += usize::from(reference_error > 1e-11);
                eprintln!(
                    "expm_multiply {arm}/{id}: reference=exact_AB_zero max_scaled_error={error:.3e}; SciPy_reference={} max_scaled_reference_error={reference_error:.3e}; raw_SciPy={reference:?}",
                    if reference_error > 1e-11 {
                        "LOSS"
                    } else {
                        "PASS"
                    }
                );
            } else {
                scipy_compared[index] += 1;
                eprintln!(
                    "expm_multiply {arm}/{id}: reference=live_SciPy max_scaled_error={error:.3e}"
                );
            }
        }
    }
    assert_eq!(compared, [10, 10, 10], "every arm must compare every case");
    assert_eq!(
        scipy_compared,
        [8, 8, 8],
        "all 24 live SciPy checks are required"
    );
    assert_eq!(
        exact_compared,
        [2, 2, 2],
        "all six exact identity checks are required"
    );
    let compared_cases: usize = compared.iter().sum();
    assert_eq!(compared_cases, 30);
    eprintln!(
        "expm_multiply compared_cases={compared_cases} scipy_compared_cases=24 exact_invariant_cases=6 SciPy_reference_losses={reference_losses}"
    );
}

#[test]
fn trace_mean_of_extreme_finite_diagonal_stays_finite() {
    // The trace overflows; its mean and the requested action stay finite.
    for sign in [-1.0, 1.0] {
        let a = diag(&[sign * f64::MAX; 3]);
        let t = 1.0 / f64::MAX;
        let b = [1.0, -2.0, 0.5];
        let result = expm_multiply(&a, &b, t, 0.0).unwrap();
        for (actual, initial) in result.iter().zip(b) {
            let expected = (sign * f64::MAX * t).exp() * initial;
            assert!((actual / expected - 1.0).abs() < 4e-15);
        }
    }
}

#[test]
fn trace_shifting_preserves_finite_actions_across_the_f64_range() {
    // Logarithms supply an independent finite scalar answer when computing
    // exp(rate) first would overflow or underflow.
    for (rate, initial) in [(-1000.0_f64, 1e300_f64), (1000.0, 1e-300)] {
        let result = expm_multiply(&diag(&[rate]), &[initial], 1.0, 0.0).unwrap();
        let expected = (rate + initial.ln()).exp();
        assert!((result[0] / expected - 1.0).abs() < 2e-13);
    }
    for diagonal in [vec![0.0, 2000.0], vec![0.0, -2000.0]] {
        let result = expm_multiply(&diag(&diagonal), &[1.0, 0.0], 1.0, 0.0).unwrap();
        assert!((result[0] - 1.0).abs() < 2e-12, "{result:?}");
        assert_eq!(result[1], 0.0);
    }
}

#[test]
fn scaled_fallback_keeps_mixed_range_components_and_final_rounding() {
    // One shared exponent cannot represent both initial states, and losing
    // the growing component would produce a wrong finite final answer.
    let a = diag(&[1000.0, 0.0]);
    let b = [1e-300_f64, 1e300];
    let result = expm_multiply(&a, &b, 1.0, 0.0).unwrap();
    let expected = (1000.0 + b[0].ln()).exp();
    assert!((result[0] / expected - 1.0).abs() < 2e-12, "{result:?}");
    assert!((result[1] / b[1] - 1.0).abs() < 2e-12, "{result:?}");

    // Subnormal results must round at the output rather than disappearing
    // as small intermediate mantissas.
    let smallest = f64::from_bits(1);
    let growth = expm_multiply(&diag(&[4.0_f64.ln()]), &[smallest], 1.0, 0.0).unwrap();
    assert_eq!(growth[0].to_bits(), 4);
    let decay = expm_multiply(&diag(&[-4.0_f64.ln()]), &[8.0 * smallest], 1.0, 0.0).unwrap();
    assert_eq!(decay[0].to_bits(), 2);
    assert!(expm_multiply(&diag(&[1000.0]), &[1.0], 1.0, 0.0).is_err());
}

#[test]
fn scaled_fallback_preserves_subnormal_contributions_through_sparse_couplings() {
    // x'=1000*x+y, y'=0 with x(0)=0: x(t)=y*expm1(1000*t)/1000.
    // The trace shift would underflow the initial y in ordinary f64.
    let a = CooMatrix::from_triplets(
        Shape2D::new(2, 2),
        vec![1000.0, 1.0],
        vec![0, 0],
        vec![0, 1],
        true,
    )
    .unwrap()
    .to_csr()
    .unwrap();
    for initial in [1e-300_f64, f64::from_bits(1)] {
        let result = expm_multiply(&a, &[0.0, initial], 1.0, 0.0).unwrap();
        // exp(-1000) is below rounding significance, so replacing
        // expm1(1000) with exp(1000) loses less than one output ULP.
        let expected = (1000.0 + initial.ln() - 1000.0_f64.ln()).exp();
        assert!((result[0] / expected - 1.0).abs() < 2e-12, "{result:?}");
        assert!((result[1] / initial - 1.0).abs() < 2e-12, "{result:?}");
    }
}

#[test]
fn growing_states_do_not_hide_small_components_in_the_same_rhs() {
    // A growing state can dominate norm-based stopping while a smaller
    // state remains inaccurate. Mixed RHS entries rule out AB=0 here.
    let n = 64;
    let a = CooMatrix::from_triplets(
        Shape2D::new(n, n),
        vec![64.0; n - 1],
        (1..n).collect(),
        (1..n).collect(),
        true,
    )
    .unwrap()
    .to_csr()
    .unwrap();
    let b: Vec<_> = (0..n)
        .map(|i| vec![(i as f64 + 0.3).sin(), (2.0 * i as f64 - 0.7).cos()])
        .collect();
    for time in [-0.25_f64, 0.7, 1.25] {
        let block = expm_multiply_matrix(&a, &b, time, 0.0).unwrap();
        let column: Vec<_> = b.iter().map(|row| row[0]).collect();
        let vector = expm_multiply(&a, &column, time, 0.0).unwrap();
        for i in 0..n {
            for k in 0..2 {
                let expected = if i == 0 {
                    b[i][k]
                } else {
                    (64.0 * time).exp() * b[i][k]
                };
                assert!((block[i][k] - expected).abs() <= 1e-11 * expected.abs().max(1.0));
                if k == 0 {
                    assert!((vector[i] - expected).abs() <= 1e-11 * expected.abs().max(1.0));
                }
            }
        }
    }
}
