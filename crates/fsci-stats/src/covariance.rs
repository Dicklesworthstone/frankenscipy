#![forbid(unsafe_code)]

//! Representation of covariance matrices matching `scipy.stats.Covariance`.

// SciPy's _PSD uses this relative spectral cutoff for double precision. The
// explicit diagonal/eigendecomposition constructors instead retain every
// strictly positive variance; applying the _PSD cutoff to them changes rank.
const PSD_RCOND: f64 = 1e6 * f64::EPSILON;

#[derive(Debug, Clone, PartialEq)]
enum CovarianceTransform {
    Diagonal(Vec<f64>),
    Cholesky,
    Precision(Vec<Vec<f64>>),
    Spectral {
        vectors: Vec<Vec<f64>>,
        scales: Vec<f64>,
        null: Vec<bool>,
        support_tolerance: f64,
    },
}

/// Representation of a covariance matrix supporting whitening, colorizing, and determinants.
#[derive(Debug, Clone, PartialEq)]
pub struct Covariance {
    dim: usize,
    cov: Vec<Vec<f64>>,
    cholesky_factor: Option<Vec<Vec<f64>>>,
    log_pdet_val: f64,
    rank_val: usize,
    transform: CovarianceTransform,
}

impl Covariance {
    /// Construct covariance representation from a diagonal variance vector.
    pub fn from_diagonal(diag: &[f64]) -> Result<Self, String> {
        let dim = diag.len();
        if dim == 0 {
            return Err("diagonal must not be empty".to_string());
        }
        let mut cov = vec![vec![0.0; dim]; dim];
        let mut log_pdet = 0.0;
        let mut rank = 0;
        let mut chol = vec![vec![0.0; dim]; dim];
        let mut scales = vec![0.0; dim];

        for i in 0..dim {
            let v = diag[i];
            if !v.is_finite() || v < 0.0 {
                return Err(format!(
                    "diagonal variance must be non-negative and finite, got {v} at index {i}"
                ));
            }
            cov[i][i] = v;
            // gh#3: positive diagonal variances retain rank at every representable scale.
            if v > 0.0 {
                log_pdet += v.ln();
                rank += 1;
                chol[i][i] = v.sqrt();
                scales[i] = chol[i][i];
            }
        }

        Ok(Self {
            dim,
            cov,
            cholesky_factor: Some(chol),
            log_pdet_val: log_pdet,
            rank_val: rank,
            transform: CovarianceTransform::Diagonal(scales),
        })
    }

    /// Construct from lower-triangular Cholesky factor L (where Cov = L * L^T).
    pub fn from_cholesky(l: &[Vec<f64>]) -> Result<Self, String> {
        let dim = validate_square(l, "cholesky factor")?;
        let mut cov = vec![vec![0.0; dim]; dim];
        let mut log_pdet = 0.0;

        for i in 0..dim {
            let diag = l[i][i];
            if diag <= 0.0 {
                return Err("cholesky factor must have a positive diagonal".to_string());
            }
            if l[i].iter().skip(i + 1).any(|&v| v != 0.0) {
                return Err("cholesky factor must be lower triangular".to_string());
            }
            log_pdet += 2.0 * diag.ln();
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..=j {
                    sum += l[i][k] * l[j][k];
                }
                if !sum.is_finite() || (i == j && sum <= 0.0) {
                    return Err(
                        "cholesky covariance is not representable as finite f64".to_string()
                    );
                }
                cov[i][j] = sum;
                cov[j][i] = sum;
            }
        }

        Ok(Self {
            dim,
            cov,
            cholesky_factor: Some(l.to_vec()),
            log_pdet_val: log_pdet,
            rank_val: dim,
            transform: CovarianceTransform::Cholesky,
        })
    }

    /// Construct from a finite symmetric positive-semidefinite matrix.
    ///
    /// Rank and the pseudo-determinant use SciPy's relative `_PSD` eigenvalue
    /// cutoff, `1e6 * f64::EPSILON * max(abs(eigenvalues))`. Whitening uses a
    /// spectral pseudoinverse, including for non-axis-aligned singular inputs.
    pub fn from_psd(psd: &[Vec<f64>]) -> Result<Self, String> {
        let dim = validate_square(psd, "PSD matrix")?;
        validate_symmetric(psd, "PSD matrix")?;
        let scale = psd.iter().flatten().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let (eigenvalues, eigenvectors) = symmetric_eigen_scaled(psd, scale)?;
        let largest = eigenvalues.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let cutoff = PSD_RCOND * largest;
        if eigenvalues.iter().any(|&value| value < -cutoff) {
            return Err("PSD matrix must be positive semidefinite".to_string());
        }
        let mut log_pdet = 0.0;
        let mut rank = 0;
        let mut scales = vec![0.0; dim];
        let mut null = vec![true; dim];
        for (i, &value) in eigenvalues.iter().enumerate() {
            if value > cutoff {
                // Taking the square roots separately preserves finite factors
                // even when an unscaled eigenvalue overflows or underflows.
                scales[i] = value.sqrt() * scale.sqrt();
                log_pdet += value.ln() + scale.ln();
                null[i] = false;
                rank += 1;
            }
        }
        Ok(Self {
            dim,
            cov: psd.to_vec(),
            cholesky_factor: if rank == dim {
                cholesky(psd).ok()
            } else {
                None
            },
            log_pdet_val: log_pdet,
            rank_val: rank,
            transform: CovarianceTransform::Spectral {
                vectors: eigenvectors,
                scales,
                null,
                support_tolerance: (1e3 * cutoff) * scale,
            },
        })
    }

    /// Construct covariance representation from its precision matrix (inverse of covariance).
    ///
    /// The precision matrix must be symmetric positive-definite.
    /// Matches `scipy.stats.Covariance.from_precision`.
    pub fn from_precision(precision: &[Vec<f64>]) -> Result<Self, String> {
        Self::from_precision_with_covariance(precision, None)
    }

    /// Construct covariance representation from precision matrix with optional precomputed covariance.
    pub fn from_precision_with_covariance(
        precision: &[Vec<f64>],
        covariance: Option<&[Vec<f64>]>,
    ) -> Result<Self, String> {
        let dim = validate_square(precision, "precision matrix")?;
        validate_symmetric(precision, "precision matrix")?;
        let l_p = cholesky(precision)?;
        let log_det_p = 2.0 * (0..dim).map(|i| l_p[i][i].ln()).sum::<f64>();

        // The transforms retain L_P itself. In particular, whitening is
        // L_P^T x, not a solve with a Cholesky factor of the inverse; those
        // have equal norms but different coordinates.
        let mut inverse_lower = vec![vec![0.0; dim]; dim];
        for j in 0..dim {
            inverse_lower[j][j] = 1.0 / l_p[j][j];
            for i in (j + 1)..dim {
                let sum: f64 = (j..i).map(|k| l_p[i][k] * inverse_lower[k][j]).sum();
                inverse_lower[i][j] = -sum / l_p[i][i];
            }
        }
        let mut inverse = vec![vec![0.0; dim]; dim];
        for i in 0..dim {
            for j in 0..=i {
                let value: f64 = (i..dim)
                    .map(|k| inverse_lower[k][i] * inverse_lower[k][j])
                    .sum();
                if !value.is_finite() || (i == j && value <= 0.0) {
                    return Err(
                        "inverse precision is not representable as a finite covariance".to_string(),
                    );
                }
                inverse[i][j] = value;
                inverse[j][i] = value;
            }
        }

        let cov_matrix = if let Some(c) = covariance {
            if validate_square(c, "covariance matrix")? != dim {
                return Err(format!(
                    "covariance length {} != precision dimension {dim}",
                    c.len()
                ));
            }
            validate_symmetric(c, "covariance matrix")?;
            let tolerance = 128.0 * dim as f64 * f64::EPSILON;
            for i in 0..dim {
                for j in 0..=i {
                    // Normalize by the implied standard deviations, rather
                    // than one global absolute tolerance that hides mistakes
                    // in small-variance coordinates of an anisotropic matrix.
                    let left = inverse[i][i].sqrt();
                    let right = inverse[j][j].sqrt();
                    let expected = scale_covariance_entry(inverse[i][j], left, right);
                    let supplied = scale_covariance_entry(c[i][j], left, right);
                    if !supplied.is_finite() || (supplied - expected).abs() > tolerance {
                        return Err(
                            "supplied covariance is inconsistent with precision".to_string()
                        );
                    }
                }
            }
            c.to_vec()
        } else {
            inverse
        };
        let cholesky_factor = cholesky(&cov_matrix).ok();
        Ok(Self {
            dim,
            cov: cov_matrix,
            cholesky_factor,
            log_pdet_val: -log_det_p,
            rank_val: dim,
            transform: CovarianceTransform::Precision(l_p),
        })
    }

    /// Construct covariance representation from eigendecomposition (eigenvalues, eigenvectors).
    ///
    /// `eigenvalues` has length n.
    /// `eigenvectors` has shape (n, n) where column `j` (`eigenvectors[i][j]`) is the eigenvector for `eigenvalues[j]`.
    /// Matches `scipy.stats.Covariance.from_eigendecomposition((w, v))`.
    pub fn from_eigendecomposition(
        eigenvalues: &[f64],
        eigenvectors: &[Vec<f64>],
    ) -> Result<Self, String> {
        let dim = eigenvalues.len();
        if dim == 0 {
            return Err("eigenvalues must not be empty".to_string());
        }
        if validate_square(eigenvectors, "eigenvectors")? != dim {
            return Err(format!(
                "eigenvectors outer dimension {} != eigenvalues length {dim}",
                eigenvectors.len()
            ));
        }
        for (i, &w) in eigenvalues.iter().enumerate() {
            if !w.is_finite() || w < 0.0 {
                return Err(format!(
                    "eigenvalues must be non-negative and finite, got {w} at index {i}"
                ));
            }
        }
        let orthogonality_tolerance = 64.0 * dim as f64 * f64::EPSILON;
        for i in 0..dim {
            for j in 0..=i {
                let product: f64 = eigenvectors.iter().map(|row| row[i] * row[j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                if !product.is_finite() || (product - expected).abs() > orthogonality_tolerance {
                    return Err("eigenvectors must form an orthonormal basis".to_string());
                }
            }
        }

        // Reconstruct covariance matrix: Sigma = V * diag(W) * V^T
        // Sigma_{i, j} = sum_k w_k * V_{i, k} * V_{j, k}
        let mut cov = vec![vec![0.0; dim]; dim];
        let mut log_pdet = 0.0;
        let mut rank = 0;
        let mut scales = vec![0.0; dim];
        let mut null = vec![true; dim];

        for (i, &w) in eigenvalues.iter().enumerate() {
            if w > 0.0 {
                log_pdet += w.ln();
                rank += 1;
                scales[i] = w.sqrt();
                null[i] = false;
            }
        }

        for i in 0..dim {
            for j in 0..=i {
                let mut sum = 0.0;
                for (k, &w) in eigenvalues.iter().enumerate() {
                    if w > 0.0 {
                        sum += w * eigenvectors[i][k] * eigenvectors[j][k];
                    }
                }
                if !sum.is_finite() {
                    return Err(
                        "eigendecomposition covariance is not representable as finite f64"
                            .to_string(),
                    );
                }
                cov[i][j] = sum;
                cov[j][i] = sum;
            }
        }

        let cholesky_factor = if rank == dim {
            cholesky(&cov).ok()
        } else {
            None
        };
        let largest = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
        Ok(Self {
            dim,
            cov,
            cholesky_factor,
            log_pdet_val: log_pdet,
            rank_val: rank,
            transform: CovarianceTransform::Spectral {
                vectors: eigenvectors.to_vec(),
                scales,
                null,
                support_tolerance: (1e3 * PSD_RCOND) * largest,
            },
        })
    }

    /// Get shape (dim, dim).
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.dim, self.dim)
    }

    /// Dimension of the random vector.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Full covariance matrix.
    #[must_use]
    pub fn covariance(&self) -> &[Vec<f64>] {
        &self.cov
    }

    /// Log pseudo-determinant of covariance matrix.
    #[must_use]
    pub fn log_pdet(&self) -> f64 {
        self.log_pdet_val
    }

    /// Numerical rank.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank_val
    }

    /// Lower-triangular Cholesky factor L (if available), where Cov = L * L^T.
    #[must_use]
    pub fn cholesky_factor(&self) -> Option<&[Vec<f64>]> {
        self.cholesky_factor.as_deref()
    }

    /// Whiten a vector using the supplied representation.
    ///
    /// Diagonal covariance divides by standard deviations, Cholesky solves
    /// `L y = x`, precision multiplies by `L_P^T`, and eigendecomposition
    /// projects into its eigenbasis before dividing by square roots. Singular
    /// directions map to zero. These transformations have equal Mahalanobis
    /// norms but need not have equal coordinates across representations.
    pub fn whiten(&self, x: &[f64]) -> Result<Vec<f64>, String> {
        if x.len() != self.dim {
            return Err(format!("vector length {} != dim {}", x.len(), self.dim));
        }
        match &self.transform {
            CovarianceTransform::Diagonal(scales) => Ok(x
                .iter()
                .zip(scales)
                .map(|(&v, &s)| if s == 0.0 { 0.0 } else { v / s })
                .collect()),
            CovarianceTransform::Cholesky => {
                let lower = self.cholesky_factor.as_ref().ok_or("No Cholesky factor")?;
                let mut y = vec![0.0; self.dim];
                for i in 0..self.dim {
                    let sum: f64 = (0..i).map(|j| lower[i][j] * y[j]).sum();
                    y[i] = (x[i] - sum) / lower[i][i];
                }
                Ok(y)
            }
            CovarianceTransform::Precision(lower) => Ok((0..self.dim)
                .map(|j| (j..self.dim).map(|i| x[i] * lower[i][j]).sum())
                .collect()),
            CovarianceTransform::Spectral {
                vectors, scales, ..
            } => Ok((0..self.dim)
                .map(|j| {
                    if scales[j] == 0.0 {
                        0.0
                    } else {
                        x.iter()
                            .enumerate()
                            .map(|(i, &v)| v * vectors[i][j])
                            .sum::<f64>()
                            / scales[j]
                    }
                })
                .collect()),
        }
    }

    /// Colorize a standard normal vector using the supplied decomposition.
    ///
    /// For singular covariance the result lies in its column space. In
    /// particular, colorizing after whitening projects a vector into that
    /// space; it need not recover an input outside the support.
    pub fn colorize(&self, x: &[f64]) -> Result<Vec<f64>, String> {
        if x.len() != self.dim {
            return Err(format!("vector length {} != dim {}", x.len(), self.dim));
        }
        match &self.transform {
            CovarianceTransform::Diagonal(scales) => {
                Ok(x.iter().zip(scales).map(|(&v, &s)| v * s).collect())
            }
            CovarianceTransform::Cholesky => {
                let lower = self.cholesky_factor.as_ref().ok_or("No Cholesky factor")?;
                Ok((0..self.dim)
                    .map(|i| (0..=i).map(|j| lower[i][j] * x[j]).sum())
                    .collect())
            }
            CovarianceTransform::Precision(lower) => {
                let mut y = vec![0.0; self.dim];
                for i in (0..self.dim).rev() {
                    let sum: f64 = ((i + 1)..self.dim).map(|j| lower[j][i] * y[j]).sum();
                    y[i] = (x[i] - sum) / lower[i][i];
                }
                Ok(y)
            }
            CovarianceTransform::Spectral {
                vectors, scales, ..
            } => Ok((0..self.dim)
                .map(|i| {
                    (0..self.dim)
                        .map(|j| x[j] * (vectors[i][j] * scales[j]))
                        .sum()
                })
                .collect()),
        }
    }

    /// Whether a centered vector lies in the support of this covariance.
    ///
    /// Diagonal null directions require exactly zero coordinates. Spectral
    /// representations use SciPy's eigenvalue-scaled null-space tolerance.
    /// The all-zero vector is in support even when the entire spectrum is
    /// zero (or the support tolerance underflows to zero).
    pub fn in_support(&self, x: &[f64]) -> Result<bool, String> {
        if x.len() != self.dim {
            return Err(format!("vector length {} != dim {}", x.len(), self.dim));
        }
        if self.rank_val == self.dim {
            return Ok(true);
        }
        match &self.transform {
            CovarianceTransform::Diagonal(scales) => {
                Ok(x.iter().zip(scales).all(|(&v, &s)| s != 0.0 || v == 0.0))
            }
            CovarianceTransform::Spectral {
                vectors,
                null,
                support_tolerance,
                ..
            } => {
                let mut residual = 0.0_f64;
                for (j, &is_null) in null.iter().enumerate() {
                    if is_null {
                        let component: f64 =
                            x.iter().enumerate().map(|(i, &v)| v * vectors[i][j]).sum();
                        residual = residual.hypot(component);
                    }
                }
                Ok(residual == 0.0 || residual < *support_tolerance)
            }
            CovarianceTransform::Cholesky | CovarianceTransform::Precision(_) => Ok(true),
        }
    }
}

fn validate_square(matrix: &[Vec<f64>], name: &str) -> Result<usize, String> {
    let dim = matrix.len();
    if dim == 0 {
        return Err(format!("{name} must not be empty"));
    }
    if matrix.iter().any(|row| row.len() != dim) {
        return Err(format!("{name} must be square"));
    }
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(format!("{name} entries must be finite"));
    }
    Ok(dim)
}

fn validate_symmetric(matrix: &[Vec<f64>], name: &str) -> Result<(), String> {
    let tolerance = 64.0 * matrix.len() as f64 * f64::EPSILON;
    for i in 0..matrix.len() {
        for j in 0..i {
            let a = matrix[i][j];
            let b = matrix[j][i];
            let scale = a.abs().max(b.abs());
            if scale != 0.0 && (a / scale - b / scale).abs() > tolerance {
                return Err(format!("{name} must be symmetric"));
            }
        }
    }
    Ok(())
}

/// Strict positive-definite Cholesky; the sign of a pivot, never an absolute
/// magnitude cutoff, determines whether the supplied matrix is admissible.
fn cholesky(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let dim = matrix.len();
    let mut lower = vec![vec![0.0; dim]; dim];
    for i in 0..dim {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| lower[i][k] * lower[j][k]).sum();
            let residual = matrix[i][j] - sum;
            let value = if i == j {
                if residual <= 0.0 || !residual.is_finite() {
                    return Err("matrix must be positive definite".to_string());
                }
                residual.sqrt()
            } else {
                residual / lower[j][j]
            };
            if !value.is_finite() {
                return Err("Cholesky factor is not finite".to_string());
            }
            lower[i][j] = value;
        }
    }
    Ok(lower)
}

fn scale_covariance_entry(value: f64, left: f64, right: f64) -> f64 {
    // Divide by the smaller root first. On a valid covariance, Cauchy-Schwarz
    // bounds this intermediate by the larger root, while dividing by the
    // larger first could erase a small but representable correlation.
    if left <= right {
        value / left / right
    } else {
        value / right / left
    }
}

/// Cyclic Jacobi on a matrix scaled to unit maximum entry. Scaling before
/// deciding convergence makes rank independent of the caller's units and
/// prevents both tiny-matrix early exits and large-matrix rotation overflow.
/// The returned eigenvalues are still in scaled units.
fn symmetric_eigen_scaled(
    input: &[Vec<f64>],
    scale: f64,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), String> {
    let dim = input.len();
    let mut matrix = vec![vec![0.0; dim]; dim];
    let mut vectors = vec![vec![0.0; dim]; dim];
    for i in 0..dim {
        vectors[i][i] = 1.0;
        if scale != 0.0 {
            for j in 0..=i {
                // Symmetry has been checked. Use the lower triangle, as
                // scipy.linalg.eigh does by default.
                matrix[i][j] = input[i][j] / scale;
                matrix[j][i] = matrix[i][j];
            }
        }
    }
    for _ in 0..64 {
        let mut largest_off = 0.0_f64;
        let mut largest_diag = 0.0_f64;
        for i in 0..dim {
            largest_diag = largest_diag.max(matrix[i][i].abs());
            for j in 0..i {
                largest_off = largest_off.max(matrix[i][j].abs());
            }
        }
        if largest_off == 0.0 || largest_off <= 4.0 * f64::EPSILON * largest_diag {
            return Ok(((0..dim).map(|i| matrix[i][i]).collect(), vectors));
        }
        for p in 0..dim {
            for q in (p + 1)..dim {
                let apq = matrix[p][q];
                if apq == 0.0 {
                    continue;
                }
                let app = matrix[p][p];
                let aqq = matrix[q][q];
                let half_difference = 0.5 * (aqq - app);
                // Unlike tau / sqrt(1 + tau^2), this remains finite for a
                // tiny off-diagonal and handles equal diagonal entries.
                let t =
                    apq / (half_difference + half_difference.hypot(apq).copysign(half_difference));
                let c = 1.0 / 1.0_f64.hypot(t);
                let s = t * c;
                for i in 0..dim {
                    if i != p && i != q {
                        let aip = matrix[i][p];
                        let aiq = matrix[i][q];
                        matrix[i][p] = c * aip - s * aiq;
                        matrix[p][i] = matrix[i][p];
                        matrix[i][q] = s * aip + c * aiq;
                        matrix[q][i] = matrix[i][q];
                    }
                    let vip = vectors[i][p];
                    let viq = vectors[i][q];
                    vectors[i][p] = c * vip - s * viq;
                    vectors[i][q] = s * vip + c * viq;
                }
                matrix[p][p] = app - t * apq;
                matrix[q][q] = aqq + t * apq;
                matrix[p][q] = 0.0;
                matrix[q][p] = 0.0;
            }
        }
    }
    Err("covariance eigendecomposition did not converge".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_diagonal() {
        let cov = Covariance::from_diagonal(&[4.0, 9.0]).expect("from_diagonal");
        assert_eq!(cov.shape(), (2, 2));
        assert_eq!(cov.dim(), 2);
        assert_eq!(cov.rank(), 2);
        assert!((cov.log_pdet() - (4.0_f64.ln() + 9.0_f64.ln())).abs() < 1e-12);
        assert_eq!(cov.covariance(), &[vec![4.0, 0.0], vec![0.0, 9.0]]);

        let x = vec![2.0, 6.0];
        let w = cov.whiten(&x).expect("whiten");
        assert!((w[0] - 1.0).abs() < 1e-12);
        assert!((w[1] - 2.0).abs() < 1e-12);

        let c = cov.colorize(&w).expect("colorize");
        assert!((c[0] - x[0]).abs() < 1e-12);
        assert!((c[1] - x[1]).abs() < 1e-12);

        assert!(Covariance::from_diagonal(&[]).is_err());
        assert!(Covariance::from_diagonal(&[1.0, -2.0]).is_err());
    }

    #[test]
    fn test_from_cholesky() {
        let l = vec![vec![2.0, 0.0], vec![1.0, 3.0]];
        let cov = Covariance::from_cholesky(&l).expect("from_cholesky");
        assert_eq!(cov.dim(), 2);
        assert_eq!(cov.rank(), 2);
        // Cov = L * L^T = [[4, 2], [2, 10]]
        assert_eq!(cov.covariance(), &[vec![4.0, 2.0], vec![2.0, 10.0]]);
        assert!((cov.log_pdet() - (2.0 * (2.0_f64.ln() + 3.0_f64.ln()))).abs() < 1e-12);

        let x = vec![1.0, 2.0];
        let w = cov.whiten(&x).expect("whiten");
        let c = cov.colorize(&w).expect("colorize");
        assert!((c[0] - x[0]).abs() < 1e-12);
        assert!((c[1] - x[1]).abs() < 1e-12);
    }

    #[test]
    fn tiny_positive_diagonal_and_cholesky_keep_scale() {
        for variance in [1e-16_f64, 1e-40, f64::MIN_POSITIVE] {
            let scale = variance.sqrt();
            let diagonal = Covariance::from_diagonal(&[variance]).expect("positive variance");
            let cholesky = Covariance::from_cholesky(&[vec![scale]]).expect("positive factor");
            for covariance in [diagonal, cholesky] {
                assert_eq!(covariance.rank(), 1);
                assert!((covariance.log_pdet() - variance.ln()).abs() < 1e-12);
                assert_eq!(covariance.whiten(&[scale]).expect("whiten"), vec![1.0]);
                assert_eq!(covariance.colorize(&[1.0]).expect("colorize"), vec![scale]);
            }
        }
        let singular = Covariance::from_diagonal(&[0.0]).expect("zero variance");
        assert_eq!(singular.rank(), 0);
        assert_eq!(singular.whiten(&[1.0]).expect("singular whiten"), vec![0.0]);
    }

    #[test]
    fn cholesky_checks_later_rows_before_matrix_product() {
        assert!(Covariance::from_cholesky(&[vec![1.0, 0.0], vec![0.0]]).is_err());
        assert!(Covariance::from_cholesky(&[vec![1.0], vec![0.0, 1.0]]).is_err());
    }

    #[test]
    fn test_from_psd() {
        let psd = vec![vec![4.0, 2.0], vec![2.0, 10.0]];
        let cov = Covariance::from_psd(&psd).expect("from_psd");
        assert_eq!(cov.covariance(), &psd);
        assert_eq!(cov.rank(), 2);
        let chol = cov.cholesky_factor().expect("cholesky factor");
        assert!((chol[0][0] - 2.0).abs() < 1e-12);
        assert!((chol[1][0] - 1.0).abs() < 1e-12);
        assert!((chol[1][1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_from_precision() {
        // P = [[2, 1], [1, 3]], det(P) = 5
        // P^-1 = [[0.6, -0.2], [-0.2, 0.4]]
        let p = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let cov = Covariance::from_precision(&p).expect("from_precision");
        assert_eq!(cov.dim(), 2);
        assert_eq!(cov.rank(), 2);
        let c = cov.covariance();
        assert!((c[0][0] - 0.6).abs() < 1e-12);
        assert!((c[0][1] - (-0.2)).abs() < 1e-12);
        assert!((c[1][0] - (-0.2)).abs() < 1e-12);
        assert!((c[1][1] - 0.4).abs() < 1e-12);

        // log_pdet = -ln(5)
        assert!((cov.log_pdet() - (-5.0_f64.ln())).abs() < 1e-12);

        // Roundtrip whiten / colorize
        let x = vec![1.5, -0.5];
        let w = cov.whiten(&x).expect("whiten");
        let rec = cov.colorize(&w).expect("colorize");
        assert!((rec[0] - x[0]).abs() < 1e-12);
        assert!((rec[1] - x[1]).abs() < 1e-12);

        // With supplied covariance
        let cov_pre = Covariance::from_precision_with_covariance(&p, Some(c)).expect("with cov");
        assert_eq!(cov_pre.covariance(), c);

        // Non-positive definite rejected
        let non_pd = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert!(Covariance::from_precision(&non_pd).is_err());

        // Asymmetric rejected
        let asym = vec![vec![2.0, 1.0], vec![0.0, 3.0]];
        assert!(Covariance::from_precision(&asym).is_err());

        // Empty rejected
        assert!(Covariance::from_precision(&[]).is_err());
    }

    #[test]
    fn test_from_eigendecomposition() {
        // A = [[0.6, -0.2], [-0.2, 0.4]]
        // trace = 1.0, det = 0.20
        // eigenvalues: lambda^2 - lambda + 0.2 = 0 => (1 +- sqrt(1 - 0.8)) / 2 = 0.5 +- sqrt(0.05)
        let l1 = 0.5 + 0.05_f64.sqrt();
        let l2 = 0.5 - 0.05_f64.sqrt();
        let theta = 0.5 * ((-0.4_f64) / (0.6 - 0.4)).atan();
        let v = vec![
            vec![theta.cos(), -theta.sin()],
            vec![theta.sin(), theta.cos()],
        ];

        let cov =
            Covariance::from_eigendecomposition(&[l1, l2], &v).expect("from_eigendecomposition");
        assert_eq!(cov.dim(), 2);
        assert_eq!(cov.rank(), 2);
        assert!((cov.log_pdet() - (l1.ln() + l2.ln())).abs() < 1e-12);

        let x = vec![0.8, -0.3];
        let w = cov.whiten(&x).expect("whiten");
        let c = cov.colorize(&w).expect("colorize");
        assert!((c[0] - x[0]).abs() < 1e-12);
        assert!((c[1] - x[1]).abs() < 1e-12);

        // Singular: one zero eigenvalue
        let sing_cov = Covariance::from_eigendecomposition(&[2.0, 0.0], &v).expect("singular");
        assert_eq!(sing_cov.rank(), 1);
        assert!((sing_cov.log_pdet() - 2.0_f64.ln()).abs() < 1e-12);

        // Mismatched dimensions rejected
        assert!(Covariance::from_eigendecomposition(&[1.0], &v).is_err());
        assert!(Covariance::from_eigendecomposition(&[], &[]).is_err());
    }

    #[test]
    fn covariance_representation_specific_transforms() {
        let precision = Covariance::from_precision(&[vec![2.0, 1.0], vec![1.0, 3.0]])
            .expect("positive precision");
        let x = [1.5, -0.5];
        let whitened = precision.whiten(&x).expect("precision whitening");
        assert!((whitened[0] - 1.767766952966369).abs() < 1e-14);
        assert!((whitened[1] + 0.7905694150420949).abs() < 1e-14);

        let q = 0.5_f64.sqrt();
        let basis = [vec![q, -q], vec![q, q]];
        let eigen = Covariance::from_eigendecomposition(&[4.0, 0.0], &basis)
            .expect("singular eigendecomposition");
        let whitened = eigen.whiten(&[2.0, 2.0]).expect("eigen whitening");
        assert!((whitened[0] - 2.0_f64.sqrt()).abs() < 1e-14);
        assert_eq!(whitened[1], 0.0);
        let colored = eigen.colorize(&[1.0, 0.0]).expect("eigen colorizing");
        for value in colored {
            assert!((value - 2.0_f64.sqrt()).abs() < 1e-14);
        }
    }

    #[test]
    fn covariance_scale_invariant_rank_and_pseudodeterminant() {
        let q = 0.5_f64.sqrt();
        let basis = [vec![q, -q], vec![q, q]];
        for scale in [1e-300_f64, 1e-20, 1.0, 1e300] {
            let matrix = [vec![2.0 * scale, scale], vec![scale, 2.0 * scale]];
            let psd = Covariance::from_psd(&matrix).expect("positive PSD");
            let precision = Covariance::from_precision(&matrix).expect("positive precision");
            let eigen = Covariance::from_eigendecomposition(&[scale, 3.0 * scale], &basis)
                .expect("positive eigenvalues");
            let expected = 3.0_f64.ln() + 2.0 * scale.ln();
            for covariance in [&psd, &eigen] {
                assert_eq!(covariance.rank(), 2, "scale={scale}");
                assert!((covariance.log_pdet() - expected).abs() < 1e-12);
            }
            assert_eq!(precision.rank(), 2);
            assert!((precision.log_pdet() + expected).abs() < 1e-12);

            let singular = Covariance::from_psd(&[vec![scale, scale], vec![scale, scale]])
                .expect("rank-one PSD");
            assert_eq!(singular.rank(), 1, "scale={scale}");
            assert!((singular.log_pdet() - (scale.ln() + 2.0_f64.ln())).abs() < 1e-12);
            let whitened = singular
                .whiten(&[scale.sqrt(), scale.sqrt()])
                .expect("whiten");
            let norm_squared: f64 = whitened.iter().map(|v| v * v).sum();
            assert!((norm_squared - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn covariance_rejects_invalid_and_inconsistent_decompositions() {
        for invalid in [
            vec![vec![1.0, 2.0], vec![2.0, 1.0]],
            vec![vec![1.0, 0.5], vec![0.0, 1.0]],
            vec![vec![1.0, f64::NAN], vec![f64::NAN, 1.0]],
            vec![vec![1.0, 0.0], vec![0.0, f64::INFINITY]],
        ] {
            assert!(Covariance::from_psd(&invalid).is_err());
        }
        assert!(Covariance::from_cholesky(&[vec![1.0, 0.5], vec![0.0, 1.0]]).is_err());
        assert!(Covariance::from_cholesky(&[vec![0.0]]).is_err());
        assert!(Covariance::from_cholesky(&[vec![f64::NAN]]).is_err());
        assert!(Covariance::from_eigendecomposition(&[-1.0], &[vec![1.0]]).is_err());
        assert!(Covariance::from_eigendecomposition(&[1.0], &[vec![2.0]]).is_err());
        assert!(
            Covariance::from_precision_with_covariance(&[vec![2.0]], Some(&[vec![3.0]])).is_err()
        );
        for scale in [1e-300, 1.0, 1e300] {
            let indefinite = [vec![scale, 2.0 * scale], vec![2.0 * scale, scale]];
            assert!(Covariance::from_psd(&indefinite).is_err());
            assert!(Covariance::from_precision(&indefinite).is_err());
            let asymmetric = [vec![scale, 0.5 * scale], vec![0.0, scale]];
            assert!(Covariance::from_psd(&asymmetric).is_err());
        }
    }

    #[test]
    fn covariance_rotated_three_dimensional_repeated_and_singular_spectra() {
        let a = 1.0 / 3.0_f64.sqrt();
        let b = 1.0 / 2.0_f64.sqrt();
        let c = 1.0 / 6.0_f64.sqrt();
        let basis = [[a, b, c], [a, -b, c], [a, 0.0, -2.0 * c]];
        for eigenvalues in [[3.0_f64, 1.0, 1.0], [3.0, 1.0, 0.0]] {
            for scale in [1e-300_f64, 1.0, 1e300] {
                let mut matrix = vec![vec![0.0; 3]; 3];
                for i in 0..3 {
                    for j in 0..=i {
                        let value = (0..3)
                            .map(|k| basis[i][k] * eigenvalues[k] * basis[j][k])
                            .sum::<f64>()
                            * scale;
                        matrix[i][j] = value;
                        matrix[j][i] = value;
                    }
                }
                let covariance = Covariance::from_psd(&matrix).expect("rotated PSD");
                let expected_rank = eigenvalues.iter().filter(|&&w| w > 0.0).count();
                assert_eq!(covariance.rank(), expected_rank);
                let expected_logdet = eigenvalues
                    .iter()
                    .filter(|&&w| w > 0.0)
                    .map(|w| w.ln() + scale.ln())
                    .sum::<f64>();
                assert!((covariance.log_pdet() - expected_logdet).abs() < 1e-12);

                let CovarianceTransform::Spectral { vectors, .. } = &covariance.transform else {
                    panic!("PSD must retain its spectral representation");
                };
                for i in 0..3 {
                    for j in 0..3 {
                        let dot: f64 = vectors.iter().map(|row| row[i] * row[j]).sum();
                        let expected = if i == j { 1.0 } else { 0.0 };
                        assert!((dot - expected).abs() < 1e-14);
                    }
                }

                let columns: Vec<_> = (0..3)
                    .map(|i| {
                        let mut unit = [0.0; 3];
                        unit[i] = 1.0;
                        covariance.colorize(&unit).expect("spectral coloring")
                    })
                    .collect();
                for i in 0..3 {
                    for j in 0..3 {
                        let reconstructed: f64 = columns
                            .iter()
                            .map(|column| (column[i] / scale.sqrt()) * (column[j] / scale.sqrt()))
                            .sum();
                        assert!((reconstructed - matrix[i][j] / scale).abs() < 1e-13);
                    }
                }
                for column in &columns {
                    let roundtrip = covariance
                        .colorize(&covariance.whiten(column).expect("spectral whitening"))
                        .expect("spectral projection");
                    for (&actual, &expected) in roundtrip.iter().zip(column) {
                        assert!(((actual - expected) / scale.sqrt()).abs() < 1e-13);
                    }
                }
            }
        }
    }

    #[test]
    fn covariance_rank_cutoffs_follow_the_representation_contract() {
        let basis = [
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        for scale in [1e-200, 1.0, 1e200] {
            let values = [scale, 1e-9 * scale, 1e-11 * scale];
            let diagonal = Covariance::from_diagonal(&values).expect("explicit diagonal");
            let eigen =
                Covariance::from_eigendecomposition(&values, &basis).expect("explicit eigen");
            let psd = Covariance::from_psd(diagonal.covariance()).expect("numerical PSD");
            assert_eq!(diagonal.rank(), 3);
            assert_eq!(eigen.rank(), 3);
            assert_eq!(psd.rank(), 2);
            let expected = 2.0 * scale.ln() + 1e-9_f64.ln();
            assert!((psd.log_pdet() - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn covariance_singular_support_and_zero_spectrum() {
        let q = 0.5_f64.sqrt();
        let basis = [vec![q, -q], vec![q, q]];
        for covariance in [
            Covariance::from_psd(&[vec![1.0, 1.0], vec![1.0, 1.0]]).unwrap(),
            Covariance::from_eigendecomposition(&[2.0, 0.0], &basis).unwrap(),
        ] {
            assert!(covariance.in_support(&[2.0, 2.0]).unwrap());
            assert!(!covariance.in_support(&[1.0, -1.0]).unwrap());
            assert!(covariance.cholesky_factor().is_none());
            let projection = covariance
                .colorize(&covariance.whiten(&[2.0, 0.0]).unwrap())
                .unwrap();
            for value in projection {
                assert!((value - 1.0).abs() < 1e-14);
            }
        }
        for covariance in [
            Covariance::from_diagonal(&[0.0, 0.0]).unwrap(),
            Covariance::from_psd(&[vec![0.0, 0.0], vec![0.0, 0.0]]).unwrap(),
            Covariance::from_eigendecomposition(&[0.0, 0.0], &basis).unwrap(),
        ] {
            assert_eq!(covariance.rank(), 0);
            assert_eq!(covariance.log_pdet(), 0.0);
            assert!(covariance.in_support(&[0.0, 0.0]).unwrap());
            assert!(!covariance.in_support(&[1.0, 0.0]).unwrap());
            assert_eq!(covariance.whiten(&[1.0, 2.0]).unwrap(), vec![0.0, 0.0]);
            assert_eq!(covariance.colorize(&[1.0, 2.0]).unwrap(), vec![0.0, 0.0]);
            assert!(covariance.in_support(&[0.0]).is_err());
        }
    }

    #[test]
    fn covariance_precision_consistency_keeps_anisotropic_coordinates() {
        let precision = [vec![1e300, 0.0], vec![0.0, 1e-300]];
        let correct = [vec![1e-300, 0.0], vec![0.0, 1e300]];
        let covariance = Covariance::from_precision_with_covariance(&precision, Some(&correct))
            .expect("consistent anisotropic covariance");
        assert_eq!(covariance.rank(), 2);
        let wrong_small_coordinate = [vec![2e-300, 0.0], vec![0.0, 1e300]];
        assert!(
            Covariance::from_precision_with_covariance(&precision, Some(&wrong_small_coordinate))
                .is_err()
        );
        let x = [1e-150, 1e150];
        let whitened = covariance.whiten(&x).unwrap();
        for value in whitened {
            assert!((value - 1.0).abs() < 1e-14);
        }
    }
}
