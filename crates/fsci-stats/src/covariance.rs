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
