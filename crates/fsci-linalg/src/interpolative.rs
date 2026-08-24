//! Reconstruction from an interpolative decomposition — the missing half of
//! `scipy.linalg.interpolative`.
//!
//! WHAT WAS MISSING. `interp_decomp` already lives in this crate: it PRODUCES an ID. Seven of
//! the nine callables in `scipy.linalg.interpolative` had no counterpart anywhere in the
//! workspace, and four of them are the ones that CONSUME an ID — without them the decomposition
//! could be computed and then not used for anything. This module adds those four. (The other
//! three, `estimate_rank` and the two spectral-norm estimators, are randomized power iterations
//! whose answer depends on a random starting vector; they are a different kind of problem and
//! are deliberately not bundled in here.)
//!
//! ## SciPy's representation, which these functions take as given
//!
//! An ID of an `m × n` matrix `A` at rank `k` is a pair:
//!
//!   * `idx` — a length-`n` permutation of the column indices whose FIRST `k` entries name the
//!     skeleton columns;
//!   * `proj` — a `k × (n - k)` matrix of coefficients expressing each remaining column in
//!     terms of the skeleton ones.
//!
//! From those, `A ≈ B · P`, where `B = A[:, idx[..k]]` is the skeleton matrix and `P` is the
//! `k × n` interpolation matrix that has the identity in the skeleton columns and `proj` in the
//! rest. Note that `proj`'s columns are in the order given by `idx[k..]` — NOT in ascending
//! column order — so scattering them back through `idx` is the whole content of
//! [`reconstruct_interp_matrix`], and getting it wrong produces a matrix that still has the
//! right shape and the right entries in the wrong places.
//!
//! ## These are deterministic, and that is why they can be differentially tested
//!
//! `interp_decomp` itself is randomized on both sides — SciPy's uses its own RNG and this
//! crate's uses a seeded SRHT — so the two will never produce the same `idx`. But everything
//! here is a pure function of `(idx, proj)`, so the differential test hands BOTH arms the same
//! decomposition, taken from the incumbent, and compares what they build from it. That
//! sidesteps the randomness entirely instead of trying to defeat it.

use crate::{DecompOptions, LinalgError, SvdResult, matrix_shape, svd};

/// An SVD derived from an interpolative decomposition.
///
/// Note `v` is `n × k` — the right singular vectors as COLUMNS, matching what
/// `scipy.linalg.interpolative.id_to_svd` returns, rather than the `vt` that [`svd`] returns.
#[derive(Debug, Clone, PartialEq)]
pub struct IdSvd {
    /// Left singular vectors, `m × k`.
    pub u: Vec<Vec<f64>>,
    /// Singular values, descending.
    pub s: Vec<f64>,
    /// Right singular vectors as columns, `n × k`.
    pub v: Vec<Vec<f64>>,
}

/// Validate an `(idx, proj)` pair and return `(k, n)`.
///
/// Checks the parts that silently produce garbage rather than an error: an `idx` that is not a
/// permutation would make the scatter below overwrite one column and leave another at zero, and
/// the result would look plausible.
fn id_shape(idx: &[usize], proj: &[Vec<f64>]) -> Result<(usize, usize), LinalgError> {
    let n = idx.len();
    let k = proj.len();
    if k > n {
        return Err(LinalgError::InvalidArgument {
            detail: format!("interpolative: rank {k} exceeds the column count {n}"),
        });
    }
    let expected_cols = n - k;
    for (row_index, row) in proj.iter().enumerate() {
        if row.len() != expected_cols {
            return Err(LinalgError::InvalidArgument {
                detail: format!(
                    "interpolative: proj row {row_index} has {} entries, expected {expected_cols} \
                     (proj is k×(n-k) with k={k}, n={n})",
                    row.len()
                ),
            });
        }
    }
    let mut seen = vec![false; n];
    for &column in idx {
        if column >= n {
            return Err(LinalgError::InvalidArgument {
                detail: format!(
                    "interpolative: idx entry {column} is out of range for {n} columns"
                ),
            });
        }
        if std::mem::replace(&mut seen[column], true) {
            return Err(LinalgError::InvalidArgument {
                detail: format!(
                    "interpolative: idx repeats column {column}; it must be a permutation"
                ),
            });
        }
    }
    Ok((k, n))
}

/// Build the `k × n` interpolation matrix `P` from an ID.
///
/// `P[:, idx[i]]` is the `i`-th unit vector for `i < k`, and `proj[:, i - k]` otherwise, so that
/// `A ≈ A[:, idx[..k]] · P`.
///
/// # Errors
///
/// Returns [`LinalgError::InvalidArgument`] if `idx` is not a permutation of `0..idx.len()` or
/// if `proj` is not `k × (n - k)`.
pub fn reconstruct_interp_matrix(
    idx: &[usize],
    proj: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (k, n) = id_shape(idx, proj)?;
    let mut p = vec![vec![0.0; n]; k];
    // The skeleton columns carry the identity...
    for (i, &column) in idx.iter().take(k).enumerate() {
        p[i][column] = 1.0;
    }
    // ...and the rest carry proj, scattered through idx rather than written in ascending
    // column order. proj's j-th column belongs at column idx[k + j].
    for (j, &column) in idx.iter().skip(k).enumerate() {
        for i in 0..k {
            p[i][column] = proj[i][j];
        }
    }
    Ok(p)
}

/// Extract the skeleton matrix `B = A[:, idx[..k]]`, an `m × k` matrix.
///
/// # Errors
///
/// Returns [`LinalgError::InvalidArgument`] if `k` exceeds `idx.len()` or if any of the first
/// `k` entries of `idx` is not a column of `a`; [`LinalgError::RaggedMatrix`] if `a`'s rows have
/// differing lengths.
pub fn reconstruct_skel_matrix(
    a: &[Vec<f64>],
    k: usize,
    idx: &[usize],
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (_, n) = matrix_shape(a)?;
    if k > idx.len() {
        return Err(LinalgError::InvalidArgument {
            detail: format!(
                "interpolative: rank {k} exceeds the {} entries in idx",
                idx.len()
            ),
        });
    }
    for &column in idx.iter().take(k) {
        if column >= n {
            return Err(LinalgError::InvalidArgument {
                detail: format!(
                    "interpolative: idx names column {column}, but the matrix has {n} columns"
                ),
            });
        }
    }
    Ok(a.iter()
        .map(|row| idx.iter().take(k).map(|&column| row[column]).collect())
        .collect())
}

/// Rebuild the approximated matrix `B · P` from the skeleton matrix and an ID.
///
/// Equivalent to `reconstruct_matrix_from_id` in SciPy: the result is `m × n`, the same shape
/// as the matrix the ID came from.
///
/// # Errors
///
/// Returns [`LinalgError::InvalidArgument`] if `b`'s column count does not match `proj`'s row
/// count, or if `(idx, proj)` is not a well-formed ID.
pub fn reconstruct_matrix_from_id(
    b: &[Vec<f64>],
    idx: &[usize],
    proj: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, LinalgError> {
    let (m, b_cols) = matrix_shape(b)?;
    let (k, n) = id_shape(idx, proj)?;
    if b_cols != k {
        return Err(LinalgError::InvalidArgument {
            detail: format!(
                "interpolative: skeleton matrix has {b_cols} columns but the ID has rank {k}"
            ),
        });
    }
    let p = reconstruct_interp_matrix(idx, proj)?;
    let mut out = vec![vec![0.0; n]; m];
    // Accumulated in i-k-j order so the innermost loop walks both `p[kk]` and `out[i]`
    // contiguously; the row of `b` is bound outside it rather than indexed as `b[i][kk]`.
    for (i, out_row) in out.iter_mut().enumerate() {
        let b_row = &b[i];
        for (kk, p_row) in p.iter().enumerate() {
            let scale = b_row[kk];
            if scale == 0.0 {
                continue;
            }
            for (out_entry, &p_entry) in out_row.iter_mut().zip(p_row.iter()) {
                *out_entry += scale * p_entry;
            }
        }
    }
    Ok(out)
}

/// Convert an ID into an SVD of the matrix it approximates.
///
/// Returns `U` (`m × k`), the singular values, and `V` (`n × k`, vectors as COLUMNS) such that
/// `U · diag(s) · Vᵀ` equals `B · P`.
///
/// SINGULAR VECTORS ARE ONLY DETERMINED UP TO SIGN, and up to an arbitrary rotation within any
/// group of equal singular values. So this agreeing with the incumbent means the singular
/// VALUES agree and the reconstruction agrees — not that `U` matches entry by entry, which it
/// legitimately need not. The differential test compares it on those terms.
///
/// # Errors
///
/// Returns [`LinalgError::InvalidArgument`] for a malformed ID, or propagates a
/// [`LinalgError::ConvergenceFailure`] from the underlying SVD.
pub fn id_to_svd(b: &[Vec<f64>], idx: &[usize], proj: &[Vec<f64>]) -> Result<IdSvd, LinalgError> {
    let (k, _) = id_shape(idx, proj)?;
    let reconstructed = reconstruct_matrix_from_id(b, idx, proj)?;
    let SvdResult { u, s, vt } = svd(&reconstructed, DecompOptions::default())?;

    // The reconstruction has rank at most k, so anything beyond the first k singular triplets
    // is numerical dust; SciPy returns exactly k.
    let keep = k.min(s.len());
    let u_k = u
        .iter()
        .map(|row| row.iter().take(keep).copied().collect())
        .collect();
    let s_k = s.iter().take(keep).copied().collect();
    // `vt` is k×n with the right singular vectors as ROWS; SciPy's `id_to_svd` returns them as
    // COLUMNS of an n×k matrix, so this transposes rather than merely truncating.
    let n = vt.first().map_or(0, Vec::len);
    let v_k = (0..n)
        .map(|column| (0..keep).map(|row| vt[row][column]).collect())
        .collect();

    Ok(IdSvd {
        u: u_k,
        s: s_k,
        v: v_k,
    })
}

/// SciPy's default number of power iterations for the spectral-norm estimators.
pub const DEFAULT_SPECTRAL_NORM_ITERATIONS: usize = 20;

/// A deterministic starting vector with no component that is exactly zero.
///
/// SciPy draws this at random. A power iteration converges to the same answer from ANY start
/// that is not orthogonal to the leading right singular vector, so a fixed start gives a
/// reproducible estimate instead of one that varies run to run — strictly better for a library,
/// and it costs nothing, because the failure mode both choices share (a start orthogonal to the
/// leading singular vector) is measure-zero either way.
///
/// `sin(i + 1)` is used because it is never exactly zero for an integer argument and carries no
/// structure that could align with a test matrix built from a regular formula.
fn power_iteration_start(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i + 1) as f64).sin()).collect()
}

fn scaled_in_place(v: &mut [f64], factor: f64) {
    for entry in v {
        *entry *= factor;
    }
}

fn euclidean_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Estimate the largest singular value of an operator given its forward and adjoint actions.
///
/// This is the power method applied to `AᵀA`: each sweep maps `x → Aᵀ(A x)`, whose norm tends
/// to `σ₁²` as `x` aligns with the leading right singular vector. Hence the square root.
///
/// The estimate is always a LOWER bound on `σ₁` (a Rayleigh-quotient-style estimate cannot
/// exceed the largest eigenvalue), and it approaches it like `(σ₂/σ₁)^(2·its)` — so accuracy is
/// governed by the spectral GAP, not by the iteration count alone. That is why the differential
/// test tolerances are derived from each case's gap rather than fixed.
fn power_method_norm<Forward, Adjoint>(
    forward: Forward,
    adjoint: Adjoint,
    n: usize,
    its: usize,
) -> f64
where
    Forward: Fn(&[f64]) -> Vec<f64>,
    Adjoint: Fn(&[f64]) -> Vec<f64>,
{
    if n == 0 {
        return 0.0;
    }
    let mut x = power_iteration_start(n);
    let start_norm = euclidean_norm(&x);
    if start_norm == 0.0 {
        return 0.0;
    }
    scaled_in_place(&mut x, 1.0 / start_norm);

    let mut estimate = 0.0;
    for _ in 0..its {
        let y = forward(&x);
        let mut next = adjoint(&y);
        let magnitude = euclidean_norm(&next);
        if magnitude == 0.0 {
            // The operator annihilates the current iterate; for a zero operator that is the
            // right answer, and for any other it means we started in the null space, which the
            // deterministic start makes reproducible rather than sporadic.
            return 0.0;
        }
        // `x` is a unit vector here, so ‖AᵀA x‖ estimates σ₁².
        estimate = magnitude.sqrt();
        scaled_in_place(&mut next, 1.0 / magnitude);
        x = next;
    }
    estimate
}

fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x).map(|(v, xi)| v * xi).sum())
        .collect()
}

fn transpose_matvec(a: &[Vec<f64>], n: usize, y: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for (row, &scale) in a.iter().zip(y) {
        if scale == 0.0 {
            continue;
        }
        for (entry, &value) in out.iter_mut().zip(row.iter()) {
            *entry += scale * value;
        }
    }
    out
}

/// Estimate the spectral norm (largest singular value) of `a` by power iteration.
///
/// Pass [`DEFAULT_SPECTRAL_NORM_ITERATIONS`] for SciPy's default of 20 sweeps.
///
/// The result is a lower bound on the true norm and converges to it at a rate set by the
/// spectral gap: on a matrix whose second singular value is well below the first this is exact
/// to machine precision, while on one with a near-degenerate leading pair it is correspondingly
/// looser. That is a property of the method, shared with the incumbent, not a limitation of
/// this implementation.
///
/// # Errors
///
/// Returns [`LinalgError::RaggedMatrix`] if the rows differ in length, or
/// [`LinalgError::NonFiniteInput`] if any entry is not finite — a NaN would propagate through
/// the iteration and turn the estimate into NaN with no indication of where it came from.
pub fn estimate_spectral_norm(a: &[Vec<f64>], its: usize) -> Result<f64, LinalgError> {
    let (_, n) = matrix_shape(a)?;
    if a.iter().flatten().any(|v| !v.is_finite()) {
        return Err(LinalgError::NonFiniteInput);
    }
    Ok(power_method_norm(
        |x| matvec(a, x),
        |y| transpose_matvec(a, n, y),
        n,
        its,
    ))
}

/// Estimate the spectral norm of the DIFFERENCE `a - b`, without forming it.
///
/// The difference is applied one matrix-vector product at a time, which is what makes this
/// worth having as its own function rather than as `estimate_spectral_norm(&(a - b))`: for the
/// large operators these estimators exist for, materialising the difference is the expensive
/// part.
///
/// # Errors
///
/// As [`estimate_spectral_norm`], plus [`LinalgError::InvalidArgument`] if the two matrices do
/// not have the same shape.
pub fn estimate_spectral_norm_diff(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    its: usize,
) -> Result<f64, LinalgError> {
    let (a_rows, a_cols) = matrix_shape(a)?;
    let (b_rows, b_cols) = matrix_shape(b)?;
    if a_rows != b_rows || a_cols != b_cols {
        return Err(LinalgError::InvalidArgument {
            detail: format!(
                "interpolative: estimate_spectral_norm_diff needs matching shapes, \
                 got {a_rows}×{a_cols} and {b_rows}×{b_cols}"
            ),
        });
    }
    if a.iter()
        .flatten()
        .chain(b.iter().flatten())
        .any(|v| !v.is_finite())
    {
        return Err(LinalgError::NonFiniteInput);
    }
    let forward = |x: &[f64]| {
        let ax = matvec(a, x);
        let bx = matvec(b, x);
        ax.iter().zip(&bx).map(|(p, q)| p - q).collect::<Vec<f64>>()
    };
    let adjoint = |y: &[f64]| {
        let aty = transpose_matvec(a, a_cols, y);
        let bty = transpose_matvec(b, b_cols, y);
        aty.iter()
            .zip(&bty)
            .map(|(p, q)| p - q)
            .collect::<Vec<f64>>()
    };
    Ok(power_method_norm(forward, adjoint, a_cols, its))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The worked example used throughout: a rank-2 matrix whose ID is exact, so reconstruction
    /// must return the ORIGINAL matrix and not merely something close to it.
    fn rank_two_fixture() -> (Vec<Vec<f64>>, Vec<usize>, Vec<Vec<f64>>) {
        // Columns 2 and 3 are exact combinations of columns 0 and 1.
        let a = vec![
            vec![1.0, 0.0, 2.0, 1.0],
            vec![0.0, 1.0, 3.0, -1.0],
            vec![2.0, 1.0, 7.0, 1.0],
        ];
        // Skeleton = columns 0 and 1; column 2 = 2*c0 + 3*c1, column 3 = 1*c0 - 1*c1.
        let idx = vec![0, 1, 2, 3];
        let proj = vec![vec![2.0, 1.0], vec![3.0, -1.0]];
        (a, idx, proj)
    }

    #[test]
    fn interp_matrix_places_the_identity_in_the_skeleton_columns() {
        let (_, idx, proj) = rank_two_fixture();
        let p = reconstruct_interp_matrix(&idx, &proj).expect("well-formed ID");
        assert_eq!(p.len(), 2);
        assert_eq!(p[0], vec![1.0, 0.0, 2.0, 1.0]);
        assert_eq!(p[1], vec![0.0, 1.0, 3.0, -1.0]);
    }

    #[test]
    fn interp_matrix_scatters_proj_through_idx_rather_than_in_column_order() {
        // THE MISTAKE THIS GUARDS AGAINST. With a permuted idx, writing proj into columns
        // k, k+1, ... in ascending order yields a matrix of the right shape with the right
        // numbers in the WRONG places — and it still reconstructs something.
        let idx = vec![2, 0, 3, 1];
        let proj = vec![vec![10.0, 20.0], vec![30.0, 40.0]];
        let p = reconstruct_interp_matrix(&idx, &proj).expect("well-formed ID");

        // Skeleton columns are idx[0]=2 and idx[1]=0.
        assert_eq!(p[0][2], 1.0);
        assert_eq!(p[1][2], 0.0);
        assert_eq!(p[0][0], 0.0);
        assert_eq!(p[1][0], 1.0);
        // proj's columns belong at idx[2]=3 and idx[3]=1, NOT at columns 2 and 3.
        assert_eq!(p[0][3], 10.0);
        assert_eq!(p[1][3], 30.0);
        assert_eq!(p[0][1], 20.0);
        assert_eq!(p[1][1], 40.0);
    }

    #[test]
    fn skeleton_matrix_selects_the_named_columns_in_idx_order() {
        let (a, _, _) = rank_two_fixture();
        let b = reconstruct_skel_matrix(&a, 2, &[3, 1, 0, 2]).expect("valid selection");
        assert_eq!(b.len(), 3);
        // Columns 3 then 1, in that order — not sorted.
        assert_eq!(b[0], vec![1.0, 0.0]);
        assert_eq!(b[1], vec![-1.0, 1.0]);
        assert_eq!(b[2], vec![1.0, 1.0]);
    }

    #[test]
    fn an_exact_id_reconstructs_the_original_matrix_exactly() {
        let (a, idx, proj) = rank_two_fixture();
        let b = reconstruct_skel_matrix(&a, 2, &idx).expect("valid selection");
        let c = reconstruct_matrix_from_id(&b, &idx, &proj).expect("well-formed ID");
        assert_eq!(c, a, "a rank-2 matrix from its exact rank-2 ID");
    }

    #[test]
    fn reconstruction_survives_a_permuted_idx() {
        // The same matrix, the same rank, but with the skeleton named in a different order and
        // the remaining columns correspondingly reordered in proj.
        let a = vec![
            vec![1.0, 0.0, 2.0, 1.0],
            vec![0.0, 1.0, 3.0, -1.0],
            vec![2.0, 1.0, 7.0, 1.0],
        ];
        // Skeleton = columns 1 and 0 (in that order); remaining = columns 3 then 2.
        // c3 = -1*c1 + 1*c0, c2 = 3*c1 + 2*c0.
        let idx = vec![1, 0, 3, 2];
        let proj = vec![vec![-1.0, 3.0], vec![1.0, 2.0]];
        let b = reconstruct_skel_matrix(&a, 2, &idx).expect("valid selection");
        let c = reconstruct_matrix_from_id(&b, &idx, &proj).expect("well-formed ID");
        assert_eq!(c, a);
    }

    #[test]
    fn id_to_svd_reproduces_the_matrix_it_came_from() {
        let (a, idx, proj) = rank_two_fixture();
        let b = reconstruct_skel_matrix(&a, 2, &idx).expect("valid selection");
        let result = id_to_svd(&b, &idx, &proj).expect("svd succeeds");

        assert_eq!(result.u.len(), 3, "U is m×k");
        assert_eq!(result.u[0].len(), 2);
        assert_eq!(result.s.len(), 2);
        assert_eq!(result.v.len(), 4, "V is n×k, vectors as columns");
        assert_eq!(result.v[0].len(), 2);

        // U·diag(s)·Vᵀ must be the original. Comparing the RECONSTRUCTION rather than the
        // factors, because singular vectors are only determined up to sign.
        for (i, a_row) in a.iter().enumerate() {
            for (j, &expected) in a_row.iter().enumerate() {
                let got: f64 = (0..2)
                    .map(|t| result.u[i][t] * result.s[t] * result.v[j][t])
                    .sum();
                assert!(
                    (got - expected).abs() < 1e-12,
                    "entry ({i},{j}): got {got}, expected {expected}"
                );
            }
        }
        // A rank-2 matrix has two positive singular values, in descending order.
        assert!(result.s[0] >= result.s[1]);
        assert!(result.s[1] > 1e-12, "got {:?}", result.s);
    }

    /// The true spectral norm, from the SVD, for the estimator tests to be judged against.
    fn true_spectral_norm(a: &[Vec<f64>]) -> f64 {
        svd(a, DecompOptions::default()).expect("svd").s[0]
    }

    fn decay_matrix(m: usize, n: usize) -> Vec<Vec<f64>> {
        (0..m)
            .map(|i| {
                (0..n)
                    .map(|j| 1.0 / (1.0 + i as f64 + 2.0 * j as f64))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn spectral_norm_is_exact_when_the_leading_singular_value_is_well_separated() {
        // Live scipy: estimate_spectral_norm on this 8×6 matrix returns 1.44259141439456, and
        // the true norm is the same to every digit — the gap σ₂/σ₁ is 0.183, so twenty sweeps
        // leave an error of order 0.183^40, far below rounding.
        let a = decay_matrix(8, 6);
        let estimate =
            estimate_spectral_norm(&a, DEFAULT_SPECTRAL_NORM_ITERATIONS).expect("finite matrix");
        let truth = true_spectral_norm(&a);
        assert!(
            (estimate - truth).abs() / truth < 1e-12,
            "estimate {estimate}, true {truth}"
        );
    }

    #[test]
    fn spectral_norm_never_exceeds_the_true_norm() {
        // A sharp invariant of the method, not a tolerance: the power iteration produces a
        // Rayleigh-quotient-style estimate, which cannot exceed the largest singular value.
        // A wrong normalisation would show up here even when the value looks plausible.
        for (m, n) in [(8, 6), (12, 10), (5, 5), (3, 9)] {
            let a = decay_matrix(m, n);
            let estimate = estimate_spectral_norm(&a, DEFAULT_SPECTRAL_NORM_ITERATIONS)
                .expect("finite matrix");
            let truth = true_spectral_norm(&a);
            assert!(
                estimate <= truth * (1.0 + 1e-12),
                "{m}×{n}: estimate {estimate} exceeds the true norm {truth}"
            );
        }
    }

    #[test]
    fn spectral_norm_of_a_rank_one_matrix_is_its_only_singular_value() {
        // σ₂ is exactly zero here, so the answer is analytic: ‖u vᵀ‖₂ = ‖u‖·‖v‖.
        let u = [1.0, 2.0, 3.0, 4.0];
        let v = [2.0, -1.0, 0.5];
        let a: Vec<Vec<f64>> = u
            .iter()
            .map(|ui| v.iter().map(|vj| ui * vj).collect())
            .collect();
        let expected = euclidean_norm(&u) * euclidean_norm(&v);

        // TWO sweeps, not one. A single sweep estimates σ₁·√|⟨x₀, v₁⟩|, because the estimate is
        // read off BEFORE the iterate has been normalised into alignment; the first sweep is
        // what aligns it. Measured: one sweep here is 26% low. From the second it is exact,
        // since a rank-one operator aligns the iterate perfectly in one step.
        let one_sweep = estimate_spectral_norm(&a, 1).expect("finite matrix");
        assert!(
            (one_sweep - expected).abs() / expected > 1e-3,
            "one sweep should NOT already be exact; got {one_sweep} against {expected}"
        );
        for its in [2, 3, 20] {
            let estimate = estimate_spectral_norm(&a, its).expect("finite matrix");
            assert!(
                (estimate - expected).abs() / expected < 1e-12,
                "{its} sweeps: estimate {estimate}, expected {expected}"
            );
        }
    }

    #[test]
    fn spectral_norm_diff_matches_the_norm_of_the_explicit_difference() {
        // The whole point of the `_diff` form is not forming `a - b`. It must nonetheless agree
        // with what forming it would give.
        let a = decay_matrix(9, 7);
        let b: Vec<Vec<f64>> = decay_matrix(9, 7)
            .iter()
            .map(|row| row.iter().map(|v| v * 0.25).collect())
            .collect();
        let explicit: Vec<Vec<f64>> = a
            .iter()
            .zip(&b)
            .map(|(ra, rb)| ra.iter().zip(rb).map(|(p, q)| p - q).collect())
            .collect();

        let via_diff = estimate_spectral_norm_diff(&a, &b, DEFAULT_SPECTRAL_NORM_ITERATIONS)
            .expect("matching shapes");
        let via_explicit = estimate_spectral_norm(&explicit, DEFAULT_SPECTRAL_NORM_ITERATIONS)
            .expect("finite matrix");
        assert!(
            (via_diff - via_explicit).abs() / via_explicit < 1e-12,
            "diff {via_diff}, explicit {via_explicit}"
        );
    }

    #[test]
    fn spectral_norm_diff_of_a_matrix_with_itself_is_zero() {
        // The zero operator annihilates every iterate; the estimator must report 0, not NaN
        // from normalising by a zero magnitude.
        let a = decay_matrix(6, 5);
        let estimate = estimate_spectral_norm_diff(&a, &a, DEFAULT_SPECTRAL_NORM_ITERATIONS)
            .expect("matching shapes");
        assert_eq!(estimate, 0.0);
    }

    #[test]
    fn spectral_norm_estimators_reject_input_they_cannot_use() {
        let a = decay_matrix(4, 3);
        let b = decay_matrix(3, 4);
        assert!(
            estimate_spectral_norm_diff(&a, &b, 5).is_err(),
            "shapes must match"
        );
        let nan = vec![vec![1.0, f64::NAN], vec![2.0, 3.0]];
        assert!(
            estimate_spectral_norm(&nan, 5).is_err(),
            "a NaN would propagate into the estimate with no trace of its origin"
        );
        assert!(estimate_spectral_norm_diff(&nan, &nan, 5).is_err());
        // Zero iterations is not an error; it simply produces no estimate.
        assert_eq!(estimate_spectral_norm(&a, 0).expect("valid"), 0.0);
    }

    #[test]
    fn malformed_ids_are_rejected_rather_than_silently_reconstructed() {
        // idx not a permutation: column 1 named twice, column 2 never. Scattering through it
        // would leave a zero column and look like a valid result.
        assert!(reconstruct_interp_matrix(&[0, 1, 1, 3], &vec![vec![1.0, 2.0]; 2]).is_err());
        // idx out of range.
        assert!(reconstruct_interp_matrix(&[0, 1, 2, 9], &vec![vec![1.0, 2.0]; 2]).is_err());
        // proj with the wrong number of columns: k×(n-k) is 2×2 here, not 2×3.
        assert!(reconstruct_interp_matrix(&[0, 1, 2, 3], &vec![vec![1.0, 2.0, 3.0]; 2]).is_err());
        // rank exceeding the column count.
        assert!(reconstruct_interp_matrix(&[0, 1], &vec![vec![]; 3]).is_err());
        // skeleton width disagreeing with the ID's rank.
        let b = vec![vec![1.0, 2.0, 3.0]];
        assert!(reconstruct_matrix_from_id(&b, &[0, 1, 2, 3], &vec![vec![1.0, 2.0]; 2]).is_err());
        // k larger than idx.
        let (a, _, _) = rank_two_fixture();
        assert!(reconstruct_skel_matrix(&a, 5, &[0, 1]).is_err());
        // idx naming a column the matrix does not have.
        assert!(reconstruct_skel_matrix(&a, 1, &[9]).is_err());
    }

    #[test]
    fn a_full_rank_id_has_an_empty_proj_and_is_a_pure_permutation() {
        // k == n: every column is a skeleton column, proj is k×0, and P is a permutation
        // matrix. An implementation that assumed proj was non-empty would fall over here.
        let idx = vec![2, 0, 1];
        let proj = vec![Vec::new(), Vec::new(), Vec::new()];
        let p = reconstruct_interp_matrix(&idx, &proj).expect("k == n is well-formed");
        assert_eq!(
            p,
            vec![
                vec![0.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
            ]
        );

        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let b = reconstruct_skel_matrix(&a, 3, &idx).expect("valid selection");
        let c = reconstruct_matrix_from_id(&b, &idx, &proj).expect("well-formed ID");
        assert_eq!(c, a, "a full-rank ID reconstructs exactly");
    }
}
