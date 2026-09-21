#![forbid(unsafe_code)]

//! Representation of covariance matrices matching `scipy.stats.Covariance`.

/// Representation of a covariance matrix supporting whitening, colorizing, and determinants.
#[derive(Debug, Clone, PartialEq)]
pub struct Covariance {
    dim: usize,
    cov: Vec<Vec<f64>>,
    cholesky_factor: Option<Vec<Vec<f64>>>,
    log_pdet_val: f64,
    rank_val: usize,
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

        for i in 0..dim {
            let v = diag[i];
            if !v.is_finite() || v < 0.0 {
                return Err(format!(
                    "diagonal variance must be non-negative and finite, got {v} at index {i}"
                ));
            }
            cov[i][i] = v;
            if v > 1e-15 {
                log_pdet += v.ln();
                rank += 1;
                chol[i][i] = v.sqrt();
            }
        }

        Ok(Self {
            dim,
            cov,
            cholesky_factor: Some(chol),
            log_pdet_val: log_pdet,
            rank_val: rank,
        })
    }

    /// Construct from lower-triangular Cholesky factor L (where Cov = L * L^T).
    pub fn from_cholesky(l: &[Vec<f64>]) -> Result<Self, String> {
        let dim = l.len();
        if dim == 0 {
            return Err("cholesky matrix must not be empty".to_string());
        }
        let mut cov = vec![vec![0.0; dim]; dim];
        let mut log_pdet = 0.0;
        let mut rank = 0;

        for i in 0..dim {
            if l[i].len() != dim {
                return Err("cholesky factor must be square".to_string());
            }
            let diag = l[i][i];
            if diag > 1e-15 {
                log_pdet += 2.0 * diag.ln();
                rank += 1;
            }
            for j in 0..dim {
                let mut sum = 0.0;
                for k in 0..dim {
                    sum += l[i][k] * l[j][k];
                }
                cov[i][j] = sum;
            }
        }

        Ok(Self {
            dim,
            cov,
            cholesky_factor: Some(l.to_vec()),
            log_pdet_val: log_pdet,
            rank_val: rank,
        })
    }

    /// Construct from full positive semi-definite matrix.
    pub fn from_psd(psd: &[Vec<f64>]) -> Result<Self, String> {
        let dim = psd.len();
        if dim == 0 {
            return Err("PSD matrix must not be empty".to_string());
        }
        for row in psd {
            if row.len() != dim {
                return Err("PSD matrix must be square".to_string());
            }
        }

        let mut l = vec![vec![0.0; dim]; dim];
        let mut log_pdet = 0.0;
        let mut rank = 0;

        for i in 0..dim {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i][k] * l[j][k];
                }
                if i == j {
                    let d = psd[i][i] - sum;
                    if d > 1e-15 {
                        let sqrt_d = d.sqrt();
                        l[i][j] = sqrt_d;
                        log_pdet += 2.0 * sqrt_d.ln();
                        rank += 1;
                    } else {
                        l[i][j] = 0.0;
                    }
                } else {
                    let denom = l[j][j];
                    if denom > 1e-15 {
                        l[i][j] = (psd[i][j] - sum) / denom;
                    } else {
                        l[i][j] = 0.0;
                    }
                }
            }
        }

        Ok(Self {
            dim,
            cov: psd.to_vec(),
            cholesky_factor: Some(l),
            log_pdet_val: log_pdet,
            rank_val: rank,
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
        let dim = precision.len();
        if dim == 0 {
            return Err("precision matrix must not be empty".to_string());
        }
        for (i, row) in precision.iter().enumerate() {
            if row.len() != dim {
                return Err(format!(
                    "precision matrix must be square, row {i} has length {}",
                    row.len()
                ));
            }
            for (j, &val) in row.iter().enumerate() {
                if !val.is_finite() {
                    return Err(format!(
                        "precision matrix entries must be finite, got {val} at ({i}, {j})"
                    ));
                }
            }
        }
        for i in 0..dim {
            for j in 0..i {
                if (precision[i][j] - precision[j][i]).abs() > 1e-10 {
                    return Err("precision matrix must be symmetric".to_string());
                }
            }
        }

        // Cholesky decomposition of precision matrix: P = L_P * L_P^T
        let mut l_p = vec![vec![0.0; dim]; dim];
        let mut log_det_p = 0.0;
        for i in 0..dim {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l_p[i][k] * l_p[j][k];
                }
                if i == j {
                    let d = precision[i][i] - sum;
                    if d <= 1e-15 || !d.is_finite() {
                        return Err("precision matrix must be positive definite".to_string());
                    }
                    let sqrt_d = d.sqrt();
                    l_p[i][j] = sqrt_d;
                    log_det_p += 2.0 * sqrt_d.ln();
                } else {
                    let denom = l_p[j][j];
                    if denom.abs() <= 1e-15 {
                        return Err("precision matrix must be positive definite".to_string());
                    }
                    l_p[i][j] = (precision[i][j] - sum) / denom;
                }
            }
        }

        let cov_matrix = if let Some(c) = covariance {
            if c.len() != dim {
                return Err(format!(
                    "covariance length {} != precision dimension {dim}",
                    c.len()
                ));
            }
            for (i, row) in c.iter().enumerate() {
                if row.len() != dim {
                    return Err(format!(
                        "covariance row {i} length {} != precision dimension {dim}",
                        row.len()
                    ));
                }
            }
            c.to_vec()
        } else {
            // Invert L_P to get M = L_P^{-1}
            let mut m = vec![vec![0.0; dim]; dim];
            for j in 0..dim {
                m[j][j] = 1.0 / l_p[j][j];
                for i in (j + 1)..dim {
                    let mut s = 0.0;
                    for k in j..i {
                        s += l_p[i][k] * m[k][j];
                    }
                    m[i][j] = -s / l_p[i][i];
                }
            }
            // P^{-1} = M^T * M
            let mut inv_p = vec![vec![0.0; dim]; dim];
            for i in 0..dim {
                for j in 0..dim {
                    let mut s = 0.0;
                    for k in 0..dim {
                        s += m[k][i] * m[k][j];
                    }
                    inv_p[i][j] = s;
                }
            }
            inv_p
        };

        // Cholesky factor of the covariance matrix
        let mut l_cov = vec![vec![0.0; dim]; dim];
        for i in 0..dim {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l_cov[i][k] * l_cov[j][k];
                }
                if i == j {
                    let d = cov_matrix[i][i] - sum;
                    if d > 1e-15 {
                        l_cov[i][j] = d.sqrt();
                    } else {
                        l_cov[i][j] = 0.0;
                    }
                } else {
                    let denom = l_cov[j][j];
                    if denom > 1e-15 {
                        l_cov[i][j] = (cov_matrix[i][j] - sum) / denom;
                    } else {
                        l_cov[i][j] = 0.0;
                    }
                }
            }
        }

        Ok(Self {
            dim,
            cov: cov_matrix,
            cholesky_factor: Some(l_cov),
            log_pdet_val: -log_det_p,
            rank_val: dim,
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
        if eigenvectors.len() != dim {
            return Err(format!(
                "eigenvectors outer dimension {} != eigenvalues length {dim}",
                eigenvectors.len()
            ));
        }
        for (i, row) in eigenvectors.iter().enumerate() {
            if row.len() != dim {
                return Err(format!(
                    "eigenvectors row {i} length {} != eigenvalues length {dim}",
                    row.len()
                ));
            }
            for (j, &val) in row.iter().enumerate() {
                if !val.is_finite() {
                    return Err(format!(
                        "eigenvectors entries must be finite, got {val} at ({i}, {j})"
                    ));
                }
            }
        }
        for (i, &w) in eigenvalues.iter().enumerate() {
            if !w.is_finite() {
                return Err(format!("eigenvalues must be finite, got {w} at index {i}"));
            }
        }

        // Reconstruct covariance matrix: Sigma = V * diag(W) * V^T
        // Sigma_{i, j} = sum_k w_k * V_{i, k} * V_{j, k}
        let mut cov = vec![vec![0.0; dim]; dim];
        let mut log_pdet = 0.0;
        let mut rank = 0;

        for &w in eigenvalues {
            if w > 1e-15 {
                log_pdet += w.ln();
                rank += 1;
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
                cov[i][j] = sum;
                cov[j][i] = sum;
            }
        }

        // Compute Cholesky factor of the reconstructed covariance matrix
        let mut l = vec![vec![0.0; dim]; dim];
        for i in 0..dim {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i][k] * l[j][k];
                }
                if i == j {
                    let d = cov[i][i] - sum;
                    if d > 1e-15 {
                        l[i][j] = d.sqrt();
                    } else {
                        l[i][j] = 0.0;
                    }
                } else {
                    let denom = l[j][j];
                    if denom > 1e-15 {
                        l[i][j] = (cov[i][j] - sum) / denom;
                    } else {
                        l[i][j] = 0.0;
                    }
                }
            }
        }

        Ok(Self {
            dim,
            cov,
            cholesky_factor: Some(l),
            log_pdet_val: log_pdet,
            rank_val: rank,
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

    /// Whiten a vector: computes L^{-1} x.
    pub fn whiten(&self, x: &[f64]) -> Result<Vec<f64>, String> {
        if x.len() != self.dim {
            return Err(format!("vector length {} != dim {}", x.len(), self.dim));
        }
        let chol = self.cholesky_factor.as_ref().ok_or("No Cholesky factor")?;
        let mut y = vec![0.0; self.dim];
        for i in 0..self.dim {
            let mut sum = 0.0;
            for j in 0..i {
                sum += chol[i][j] * y[j];
            }
            let denom = chol[i][i];
            if denom.abs() < 1e-15 {
                y[i] = 0.0;
            } else {
                y[i] = (x[i] - sum) / denom;
            }
        }
        Ok(y)
    }

    /// Colorize a standard normal vector: computes L * x.
    pub fn colorize(&self, x: &[f64]) -> Result<Vec<f64>, String> {
        if x.len() != self.dim {
            return Err(format!("vector length {} != dim {}", x.len(), self.dim));
        }
        let chol = self.cholesky_factor.as_ref().ok_or("No Cholesky factor")?;
        let mut y = vec![0.0; self.dim];
        for i in 0..self.dim {
            let mut sum = 0.0;
            for j in 0..=i {
                sum += chol[i][j] * x[j];
            }
            y[i] = sum;
        }
        Ok(y)
    }
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
}
