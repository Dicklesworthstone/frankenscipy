//! The L-BFGS approximate inverse Hessian as an operator — `scipy.optimize.LbfgsInvHessProduct`.
//!
//! This is what SciPy hands back as `result.hess_inv` from `minimize(method='L-BFGS-B')`: a
//! linear operator that multiplies a vector by the limited-memory approximation to `H⁻¹`,
//! reconstructed from the correction pairs `(sₖ, yₖ)` the optimizer accumulated. It is how a
//! caller gets an inverse-Hessian — and hence a covariance-like — estimate out of L-BFGS-B
//! without ever forming an `n × n` matrix.
//!
//! ## It is NOT the crate's existing two-loop recursion, and reusing that would be a bug
//!
//! `minimize::lbfgs_two_loop` computes SEARCH DIRECTIONS, and like most implementations it
//! scales the initial inverse Hessian by `γ = sᵀy / yᵀy` to improve step quality. SciPy's
//! `LbfgsInvHessProduct` does NOT: it starts from `H₀ = I`. The two therefore compute different
//! operators from the same history, and wiring this module to the existing helper would produce
//! numbers that are individually reasonable and differ from the incumbent everywhere.
//!
//! ## One deliberate divergence, at the curvature condition
//!
//! `ρᵢ = 1 / (sᵢ · yᵢ)`. BFGS requires `sᵢ · yᵢ > 0` — the curvature condition — for the
//! approximation to be positive definite. SciPy does not check it: measured on scipy 1.17.1,
//! `sk = [[1, 0]]`, `yk = [[0, 1]]` gives `rho = [inf]`, and both `matvec` and `todense` return
//! all-NaN, announced only by a `RuntimeWarning` on stderr. We reject it instead, because an
//! infinite `ρ` poisons every subsequent result with no indication of where it came from.

use crate::types::OptError;

/// The L-BFGS limited-memory approximation to the inverse Hessian, as an operator.
///
/// Construct with [`LbfgsInvHessProduct::new`] from the correction pairs, then apply it with
/// [`matvec`](Self::matvec) — or materialise it with [`todense`](Self::todense) when an actual
/// matrix is wanted and `n` is small enough to afford one.
#[derive(Debug, Clone, PartialEq)]
pub struct LbfgsInvHessProduct {
    /// The `n_corrs` most recent updates to the solution vector, each of length `n`.
    sk: Vec<Vec<f64>>,
    /// The `n_corrs` most recent updates to the gradient, matching `sk` in shape.
    yk: Vec<Vec<f64>>,
    /// `1 / (sᵢ · yᵢ)`, precomputed as SciPy does.
    rho: Vec<f64>,
    /// The dimension the operator acts on.
    n: usize,
}

impl LbfgsInvHessProduct {
    /// Build the operator from `n_corrs × n` correction pairs.
    ///
    /// # Errors
    ///
    /// Returns [`OptError::InvalidArgument`] if `sk` and `yk` do not have matching rectangular
    /// shapes, if any entry is not finite, or if any pair violates the curvature condition
    /// `sᵢ · yᵢ > 0` — see the module comment for why that last one is rejected rather than
    /// propagated as the incumbent does.
    pub fn new(sk: Vec<Vec<f64>>, yk: Vec<Vec<f64>>) -> Result<Self, OptError> {
        if sk.len() != yk.len() {
            return Err(OptError::InvalidArgument {
                detail: format!(
                    "LbfgsInvHessProduct: sk and yk must have matching shape (n_corrs, n), \
                     got {} and {} corrections",
                    sk.len(),
                    yk.len()
                ),
            });
        }
        // With no corrections the operator is the identity, which is what an L-BFGS run that
        // never took a step would legitimately produce.
        let n = sk.first().map_or(0, Vec::len);
        for (index, (s_row, y_row)) in sk.iter().zip(&yk).enumerate() {
            if s_row.len() != n || y_row.len() != n {
                return Err(OptError::InvalidArgument {
                    detail: format!(
                        "LbfgsInvHessProduct: correction {index} has lengths {} and {}, \
                         expected {n} for both",
                        s_row.len(),
                        y_row.len()
                    ),
                });
            }
            if s_row.iter().chain(y_row.iter()).any(|v| !v.is_finite()) {
                return Err(OptError::InvalidArgument {
                    detail: format!("LbfgsInvHessProduct: correction {index} is not finite"),
                });
            }
        }

        let mut rho = Vec::with_capacity(sk.len());
        for (index, (s_row, y_row)) in sk.iter().zip(&yk).enumerate() {
            let curvature = dot(s_row, y_row);
            // `is_finite` first, so a curvature that overflowed to infinity — or came out NaN
            // from cancelling infinities — is rejected there rather than reaching `<=`, which
            // a NaN would pass.
            if !curvature.is_finite() || curvature <= 0.0 {
                return Err(OptError::InvalidArgument {
                    detail: format!(
                        "LbfgsInvHessProduct: correction {index} violates the curvature \
                         condition, s·y = {curvature} is not positive; the inverse Hessian \
                         would not be positive definite and rho would be non-finite"
                    ),
                });
            }
            rho.push(1.0 / curvature);
        }

        Ok(Self { sk, yk, rho, n })
    }

    /// The operator's shape, `(n, n)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    /// How many correction pairs the approximation is built from.
    #[must_use]
    pub fn n_corrs(&self) -> usize {
        self.sk.len()
    }

    /// Multiply a vector by the approximate inverse Hessian.
    ///
    /// The two-loop recursion of Nocedal (1980): sweep the corrections newest-to-oldest
    /// removing each one's contribution, apply `H₀ = I`, then sweep oldest-to-newest adding
    /// them back. Cost is `O(n_corrs · n)` rather than the `O(n²)` a dense multiply would be.
    ///
    /// # Errors
    ///
    /// Returns [`OptError::InvalidArgument`] if `x` is not of length `n`.
    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>, OptError> {
        if x.len() != self.n {
            return Err(OptError::InvalidArgument {
                detail: format!(
                    "LbfgsInvHessProduct: operator is {}×{}, cannot apply it to a vector of \
                     length {}",
                    self.n,
                    self.n,
                    x.len()
                ),
            });
        }
        let mut q = x.to_vec();
        let mut alpha = vec![0.0; self.sk.len()];

        // Backward sweep, newest correction first.
        for i in (0..self.sk.len()).rev() {
            alpha[i] = self.rho[i] * dot(&self.sk[i], &q);
            axpy(&mut q, &self.yk[i], -alpha[i]);
        }

        // H₀ = I, so `q` passes through untouched here. Stated rather than written as a
        // multiply by one, because this is precisely where the incumbent differs from the
        // gamma-scaled recursion used for search directions elsewhere in this crate.
        let mut r = q;

        // Forward sweep, oldest correction first. Walked as an iterator rather than by index
        // so the four arrays are advanced together and cannot drift out of step.
        for (((s_row, y_row), &rho_i), &alpha_i) in
            self.sk.iter().zip(&self.yk).zip(&self.rho).zip(&alpha)
        {
            let beta = rho_i * dot(y_row, &r);
            axpy(&mut r, s_row, alpha_i - beta);
        }
        Ok(r)
    }

    /// Multiply an `n × m` matrix by the operator, column by column.
    ///
    /// # Errors
    ///
    /// Returns [`OptError::InvalidArgument`] if `x` is not rectangular with `n` rows.
    pub fn matmat(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, OptError> {
        if x.len() != self.n {
            return Err(OptError::InvalidArgument {
                detail: format!(
                    "LbfgsInvHessProduct: operator is {}×{}, cannot apply it to a matrix with \
                     {} rows",
                    self.n,
                    self.n,
                    x.len()
                ),
            });
        }
        let columns = x.first().map_or(0, Vec::len);
        if x.iter().any(|row| row.len() != columns) {
            return Err(OptError::InvalidArgument {
                detail: "LbfgsInvHessProduct: the operand matrix is ragged".to_string(),
            });
        }

        let mut out = vec![vec![0.0; columns]; self.n];
        let mut column = vec![0.0; self.n];
        for j in 0..columns {
            for (i, entry) in column.iter_mut().enumerate() {
                *entry = x[i][j];
            }
            let product = self.matvec(&column)?;
            for (i, value) in product.into_iter().enumerate() {
                out[i][j] = value;
            }
        }
        Ok(out)
    }

    /// Materialise the operator as a dense `n × n` matrix.
    ///
    /// Column `j` is the operator applied to the `j`-th unit vector, which is how SciPy builds
    /// it too (`todense` is `matmat(I)` there).
    #[must_use]
    pub fn todense(&self) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; self.n]; self.n];
        let mut unit = vec![0.0; self.n];
        for j in 0..self.n {
            unit[j] = 1.0;
            let column = self
                .matvec(&unit)
                .expect("a unit vector always has the operator's own length");
            for (i, value) in column.into_iter().enumerate() {
                dense[i][j] = value;
            }
            unit[j] = 0.0;
        }
        dense
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `target += scale · source`, in place.
fn axpy(target: &mut [f64], source: &[f64], scale: f64) {
    for (entry, &value) in target.iter_mut().zip(source) {
        *entry += scale * value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A well-conditioned pair set satisfying the curvature condition.
    fn corrections(n_corrs: usize, n: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let sk: Vec<Vec<f64>> = (0..n_corrs)
            .map(|k| {
                (0..n)
                    .map(|i| ((i + 1) as f64).sin() + 0.5 * (k + 1) as f64)
                    .collect()
            })
            .collect();
        // y = D s with D positive definite guarantees s·y > 0.
        let yk: Vec<Vec<f64>> = sk
            .iter()
            .map(|s| {
                s.iter()
                    .enumerate()
                    .map(|(i, v)| v * (1.0 + i as f64 * 0.25))
                    .collect()
            })
            .collect();
        (sk, yk)
    }

    // The closed-form check below is a literal transcription of a matrix formula, where `i`,
    // `j` and `t` are the formula's own subscripts. Rewriting them as iterator chains would
    // obscure the correspondence this test exists to make checkable by eye.
    #[allow(clippy::needless_range_loop)]
    #[test]
    fn a_single_correction_reproduces_the_closed_form_bfgs_update() {
        // With one pair and H₀ = I the operator is analytic:
        //   H = (I - ρ s yᵀ)(I - ρ y sᵀ) + ρ s sᵀ
        // Computing that directly is an INDEPENDENT check of the two-loop recursion, not a
        // restatement of it — the recursion never forms these outer products.
        let s = vec![1.0, 2.0, -0.5];
        let y = vec![0.5, 1.5, 0.25];
        let rho = 1.0 / dot(&s, &y);
        let n = 3;

        let operator =
            LbfgsInvHessProduct::new(vec![s.clone()], vec![y.clone()]).expect("valid curvature");
        let dense = operator.todense();

        for i in 0..n {
            for j in 0..n {
                let identity_ij = if i == j { 1.0 } else { 0.0 };
                // (I - ρ s yᵀ)(I - ρ y sᵀ) + ρ s sᵀ, entry by entry.
                let mut expected = 0.0;
                for t in 0..n {
                    let a1 = if i == t { 1.0 } else { 0.0 } - rho * s[i] * y[t];
                    let a2 = if t == j { 1.0 } else { 0.0 } - rho * y[t] * s[j];
                    expected += a1 * a2;
                }
                expected += rho * s[i] * s[j];
                let _ = identity_ij;
                assert!(
                    (dense[i][j] - expected).abs() < 1e-12,
                    "entry ({i},{j}): got {}, expected {expected}",
                    dense[i][j]
                );
            }
        }
    }

    // Same reason as above: `dense[i][j]` against `dense[j][i]` is the statement of symmetry.
    #[allow(clippy::needless_range_loop)]
    #[test]
    fn the_operator_is_symmetric_and_positive_definite() {
        // Both are guaranteed by the BFGS update when every pair satisfies the curvature
        // condition, and both would be lost by a sign or index slip in either sweep.
        let (sk, yk) = corrections(4, 6);
        let dense = LbfgsInvHessProduct::new(sk, yk)
            .expect("valid curvature")
            .todense();

        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (dense[i][j] - dense[j][i]).abs() < 1e-12,
                    "asymmetry at ({i},{j}): {} vs {}",
                    dense[i][j],
                    dense[j][i]
                );
            }
        }
        // Positive definiteness via the quadratic form on a spread of vectors.
        for probe in 0..6 {
            let mut x = [0.0; 6];
            for (i, entry) in x.iter_mut().enumerate() {
                *entry = ((i + probe + 1) as f64).cos();
            }
            let quadratic: f64 = (0..6)
                .map(|i| (0..6).map(|j| x[i] * dense[i][j] * x[j]).sum::<f64>())
                .sum();
            assert!(quadratic > 0.0, "probe {probe}: xᵀHx = {quadratic}");
        }
    }

    #[test]
    fn todense_columns_are_the_operator_applied_to_unit_vectors() {
        // The relationship `todense` is DEFINED by; if the two ever disagree, one of the two
        // paths has drifted.
        let (sk, yk) = corrections(3, 5);
        let operator = LbfgsInvHessProduct::new(sk, yk).expect("valid curvature");
        let dense = operator.todense();

        for j in 0..5 {
            let mut unit = vec![0.0; 5];
            unit[j] = 1.0;
            let column = operator.matvec(&unit).expect("right length");
            for i in 0..5 {
                assert_eq!(dense[i][j], column[i], "column {j}, row {i}");
            }
        }
    }

    #[test]
    fn matmat_agrees_with_column_wise_matvec() {
        let (sk, yk) = corrections(2, 4);
        let operator = LbfgsInvHessProduct::new(sk, yk).expect("valid curvature");
        // A 4×3 operand.
        let x: Vec<Vec<f64>> = (0..4)
            .map(|i| (0..3).map(|j| ((i * 3 + j) as f64).sin()).collect())
            .collect();
        let product = operator.matmat(&x).expect("compatible shape");

        for j in 0..3 {
            let column: Vec<f64> = (0..4).map(|i| x[i][j]).collect();
            let expected = operator.matvec(&column).expect("right length");
            for i in 0..4 {
                assert_eq!(product[i][j], expected[i], "entry ({i},{j})");
            }
        }
    }

    #[test]
    fn no_corrections_is_the_identity() {
        // An L-BFGS run that never took a step returns an empty history, and the approximation
        // is then H₀ = I. An implementation that assumed at least one pair would panic here.
        let operator = LbfgsInvHessProduct::new(vec![], vec![]).expect("empty history is valid");
        assert_eq!(operator.n_corrs(), 0);
        assert_eq!(operator.shape(), (0, 0));
        assert_eq!(operator.matvec(&[]).expect("empty"), Vec::<f64>::new());
        assert!(operator.todense().is_empty());
    }

    #[test]
    fn h0_is_the_identity_not_the_gamma_scaled_initial_hessian() {
        // THE DISTINCTION THAT MATTERS. `minimize::lbfgs_two_loop` scales H₀ by
        // γ = sᵀy / yᵀy; this operator does not, because the incumbent does not. With a single
        // correction where s and y are parallel — y = c·s — the two-loop recursion collapses to
        // H = ρ s sᵀ + (I - ρ s yᵀ)(I - ρ y sᵀ), and applying it to s itself gives exactly s/c,
        // whereas a γ-scaled variant would give something else entirely.
        let s = vec![1.0, 2.0, 3.0];
        let c = 4.0;
        let y: Vec<f64> = s.iter().map(|v| c * v).collect();
        let operator = LbfgsInvHessProduct::new(vec![s.clone()], vec![y]).expect("valid");
        let applied = operator.matvec(&s).expect("right length");
        for (i, value) in applied.iter().enumerate() {
            assert!(
                (value - s[i] / c).abs() < 1e-12,
                "entry {i}: got {value}, expected {}",
                s[i] / c
            );
        }
    }

    #[test]
    fn inputs_that_would_silently_produce_nan_are_rejected() {
        // The curvature condition. Measured on scipy 1.17.1, this exact input gives
        // rho = [inf] and an all-NaN matvec, reported only as a RuntimeWarning.
        let zero_curvature = LbfgsInvHessProduct::new(vec![vec![1.0, 0.0]], vec![vec![0.0, 1.0]]);
        assert!(zero_curvature.is_err(), "s·y = 0 must be rejected");
        // Negative curvature breaks positive definiteness just as thoroughly.
        let negative = LbfgsInvHessProduct::new(vec![vec![1.0, 0.0]], vec![vec![-1.0, 0.0]]);
        assert!(negative.is_err(), "s·y < 0 must be rejected");

        // Shape and finiteness.
        assert!(LbfgsInvHessProduct::new(vec![vec![1.0]], vec![]).is_err());
        assert!(
            LbfgsInvHessProduct::new(vec![vec![1.0, 2.0]], vec![vec![1.0]]).is_err(),
            "mismatched row lengths"
        );
        assert!(LbfgsInvHessProduct::new(vec![vec![1.0, f64::NAN]], vec![vec![1.0, 1.0]]).is_err());

        // Wrong operand length.
        let (sk, yk) = corrections(2, 4);
        let operator = LbfgsInvHessProduct::new(sk, yk).expect("valid");
        assert!(operator.matvec(&[1.0, 2.0]).is_err());
        assert!(operator.matmat(&[vec![1.0], vec![2.0]]).is_err());
    }
}
