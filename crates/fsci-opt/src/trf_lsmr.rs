#![forbid(unsafe_code)]

//! Iterative least-squares subproblems for TRF and Dogbox.
//!
//! LSMR supplies an approximate Gauss–Newton direction using only products by
//! the Jacobian and its transpose. TRF minimizes its quadratic model in the
//! span of that direction and the scaled gradient. The projected Hessian is
//! positive semidefinite; a safeguarded secular solve handles its at-most-two
//! eigenvalues, including a one-dimensional or rank-deficient subspace.

use fsci_runtime::RuntimeMode;
use fsci_sparse::linalg::{LsmrOptions, lsmr_with_options};
use fsci_sparse::{LinearOperator, Shape2D, SparseError, SparseResult};

/// Linear subproblem solver for nonlinear least squares.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TrustRegionSolver {
    /// Factor the dense Jacobian using an SVD. This remains the default.
    #[default]
    Exact,
    /// Iterative Gauss–Newton solves, with a two-dimensional TRF subspace.
    Lsmr(TrustRegionLsmrOptions),
}

/// SciPy's LSMR `tr_options` for `least_squares`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrustRegionLsmrOptions {
    /// Relative accuracy of the linear operator.
    pub atol: f64,
    /// Relative accuracy of its right-hand side.
    pub btol: f64,
    /// Stop the linear solve when its estimated condition number exceeds this
    /// value. Zero disables this limit.
    pub conlim: f64,
    /// Linear iterations per nonlinear iteration. None uses min(rows, columns).
    pub max_iter: Option<usize>,
    /// Tikhonov damping for the approximate Gauss–Newton direction.
    pub damp: f64,
    /// Add TRF's adaptive regularization to its linear solve. Dogbox ignores
    /// this setting. It does not change the nonlinear objective.
    pub regularize: bool,
}

impl Default for TrustRegionLsmrOptions {
    fn default() -> Self {
        Self {
            atol: 1.0e-6,
            btol: 1.0e-6,
            conlim: 1.0e8,
            max_iter: None,
            damp: 0.0,
            regularize: true,
        }
    }
}

impl TrustRegionLsmrOptions {
    pub(crate) fn validate(self) -> bool {
        [self.atol, self.btol, self.conlim]
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
            && self.damp.is_finite()
    }

    fn linear_options(self, mode: RuntimeMode, damp: f64) -> LsmrOptions<'static> {
        LsmrOptions {
            mode,
            atol: self.atol,
            btol: self.btol,
            conlim: self.conlim,
            max_iter: self.max_iter,
            damp,
            x0: None,
        }
    }
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

fn norm(x: &[f64]) -> f64 {
    x.iter().fold(0.0_f64, |value, &x| value.hypot(x))
}

fn product(matrix: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    matrix.iter().map(|row| dot(row, x)).collect()
}

/// [J; diag(diagonal)], without allocating an n-by-n diagonal block.
struct AugmentedJacobian<'a> {
    matrix: &'a [Vec<f64>],
    diagonal: &'a [f64],
    columns: usize,
}

impl LinearOperator for AugmentedJacobian<'_> {
    fn shape(&self) -> Shape2D {
        Shape2D {
            rows: self.matrix.len() + self.diagonal.len(),
            cols: self.columns,
        }
    }

    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        if x.len() != self.columns || y.len() != self.shape().rows {
            return Err(SparseError::IncompatibleShape {
                message: String::from("least-squares operator matvec dimensions"),
            });
        }
        for (out, row) in y.iter_mut().zip(self.matrix) {
            *out = dot(row, x);
        }
        for ((out, &d), &value) in y[self.matrix.len()..]
            .iter_mut()
            .zip(self.diagonal)
            .zip(x)
        {
            *out = d * value;
        }
        Ok(())
    }

    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        if x.len() != self.shape().rows || y.len() != self.columns {
            return Err(SparseError::IncompatibleShape {
                message: String::from("least-squares operator rmatvec dimensions"),
            });
        }
        y.fill(0.0);
        for (row, &value) in self.matrix.iter().zip(x) {
            for (out, &entry) in y.iter_mut().zip(row) {
                *out += entry * value;
            }
        }
        for ((out, &d), &value) in y.iter_mut().zip(self.diagonal).zip(&x[self.matrix.len()..]) {
            *out += d * value;
        }
        Ok(())
    }
}

pub(crate) struct Subspace {
    basis: Vec<Vec<f64>>,
    hessian: [[f64; 2]; 2],
    gradient: [f64; 2],
    columns: usize,
}

impl Subspace {
    pub(crate) fn step(&self, radius: f64) -> Vec<f64> {
        let coefficients = solve_projected(self.hessian, self.gradient, radius);
        let mut step = vec![0.0; self.columns];
        for (column, &coefficient) in self.basis.iter().zip(&coefficients) {
            for (value, &entry) in step.iter_mut().zip(column) {
                *value += coefficient * entry;
            }
        }
        step
    }
}

/// Minimize a PSD quadratic over a two-dimensional ball. The eigenbasis
/// reduces the boundary condition to a monotone scalar secular equation.
fn solve_projected(hessian: [[f64; 2]; 2], gradient: [f64; 2], radius: f64) -> [f64; 2] {
    if radius == 0.0 || gradient == [0.0, 0.0] {
        return [0.0, 0.0];
    }
    let scale = hessian
        .iter()
        .flatten()
        .fold(0.0_f64, |value, x| value.max(x.abs()));
    if scale == 0.0 {
        let factor = -radius / norm(&gradient);
        return [factor * gradient[0], factor * gradient[1]];
    }
    let a = hessian[0][0] / scale;
    let b = hessian[0][1] / scale;
    let c = hessian[1][1] / scale;
    let (cos, sin) = if b == 0.0 {
        if a >= c { (1.0, 0.0) } else { (0.0, 1.0) }
    } else {
        let angle = 0.5 * (2.0 * b).atan2(a - c);
        let (sin, cos) = angle.sin_cos();
        (cos, sin)
    };
    let largest = 0.5 * a + 0.5 * c + (0.5 * (a - c)).hypot(b);
    // B is a Gram matrix. Its determinant can become a tiny negative number
    // solely from rounding; preserve the zero eigenvalue in that case.
    let smallest = ((a * c - b * b) / largest).max(0.0);
    let eigenvalues = [largest * scale, smallest * scale];
    let rotated = [
        cos * gradient[0] + sin * gradient[1],
        -sin * gradient[0] + cos * gradient[1],
    ];
    let at = |shift: f64| {
        std::array::from_fn::<_, 2, _>(|i| {
            let denominator = eigenvalues[i] + shift;
            if denominator == 0.0 && rotated[i] == 0.0 {
                0.0
            } else {
                -rotated[i] / denominator
            }
        })
    };
    let mut step = at(0.0);
    if norm(&step) > radius {
        let mut lower = 0.0;
        let mut upper = norm(&rotated) / radius;
        // For a PSD model this upper bound always puts the step inside the
        // ball. Bisect until the boundary is resolved to floating precision.
        for _ in 0..100 {
            let middle = 0.5 * lower + 0.5 * upper;
            if middle == lower || middle == upper {
                break;
            }
            let candidate = at(middle);
            if norm(&candidate) > radius {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        step = at(upper);
    }
    [cos * step[0] - sin * step[1], sin * step[0] + cos * step[1]]
}

pub(crate) fn subspace(
    jacobian: &[Vec<f64>],
    residuals: &[f64],
    gradient: &[f64],
    diagonal: &[f64],
    radius: f64,
    bounded: bool,
    options: TrustRegionLsmrOptions,
    mode: RuntimeMode,
) -> SparseResult<Subspace> {
    let n = gradient.len();
    let gradient_norm = norm(gradient);
    if gradient_norm == 0.0 {
        return Ok(Subspace {
            basis: Vec::new(),
            hessian: [[0.0; 2]; 2],
            gradient: [0.0; 2],
            columns: n,
        });
    }

    let mut regularization = 0.0;
    if options.regularize {
        // Minimize q(-t*g) for 0 <= t <= radius/||g||.
        let image = product(jacobian, gradient);
        let quadratic = dot(&image, &image)
            + gradient
                .iter()
                .zip(diagonal)
                .map(|(g, d)| g * g * d)
                .sum::<f64>();
        let linear = dot(gradient, gradient);
        let to_boundary = radius / gradient_norm;
        let length = if quadratic > 0.0 {
            (linear / quadratic).min(to_boundary)
        } else {
            to_boundary
        };
        regularization = (length * (linear - 0.5 * length * quadratic) / radius) / radius;
    }
    let extra: Vec<f64> = if bounded {
        diagonal
            .iter()
            .map(|d| (d + regularization).sqrt())
            .collect()
    } else {
        Vec::new()
    };
    let damp = if bounded {
        options.damp
    } else {
        options.damp.hypot(regularization.sqrt())
    };
    if !damp.is_finite() || extra.iter().any(|value| !value.is_finite()) {
        return Err(SparseError::NonFiniteInput {
            message: String::from("non-finite least-squares regularization"),
        });
    }
    let operator = AugmentedJacobian {
        matrix: jacobian,
        diagonal: &extra,
        columns: n,
    };
    let mut rhs = residuals.to_vec();
    rhs.resize(operator.shape().rows, 0.0);
    let gauss_newton = lsmr_with_options(&operator, &rhs, options.linear_options(mode, damp))?.solution;

    let first: Vec<f64> = gradient.iter().map(|value| value / gradient_norm).collect();
    let mut second = gauss_newton.clone();
    // Reorthogonalization is important when LSMR's direction nearly coincides
    // with the gradient. Do not manufacture a second direction for rank one.
    for _ in 0..2 {
        let projection = dot(&first, &second);
        for (value, &entry) in second.iter_mut().zip(&first) {
            *value -= projection * entry;
        }
    }
    let second_norm = norm(&second);
    let mut basis = vec![first];
    if second_norm > 32.0 * f64::EPSILON * norm(&gauss_newton) {
        for value in &mut second {
            *value /= second_norm;
        }
        basis.push(second);
    }
    let images: Vec<_> = basis.iter().map(|column| product(jacobian, column)).collect();
    let mut hessian = [[0.0; 2]; 2];
    let mut projected_gradient = [0.0; 2];
    for i in 0..basis.len() {
        projected_gradient[i] = dot(&basis[i], gradient);
        for j in 0..basis.len() {
            hessian[i][j] = dot(&images[i], &images[j])
                + basis[i]
                    .iter()
                    .zip(diagonal)
                    .zip(&basis[j])
                    .map(|((x, d), y)| x * d * y)
                    .sum::<f64>();
        }
    }
    if hessian
        .iter()
        .flatten()
        .chain(&projected_gradient)
        .any(|x| !x.is_finite())
    {
        return Err(SparseError::NonFiniteInput {
            message: String::from("non-finite projected least-squares model"),
        });
    }
    Ok(Subspace {
        basis,
        hessian,
        gradient: projected_gradient,
        columns: n,
    })
}

/// Dogbox solves in scaled free-variable coordinates and then transforms the
/// step back. A finite iteration budget may legitimately give an approximate
/// direction; the outer trust-region ratio still decides whether to accept it.
pub(crate) fn dogbox_step(
    jacobian: &[Vec<f64>],
    residuals: &[f64],
    scales: &[f64],
    options: TrustRegionLsmrOptions,
    mode: RuntimeMode,
) -> SparseResult<Vec<f64>> {
    let scaled: Vec<Vec<f64>> = jacobian
        .iter()
        .map(|row| row.iter().zip(scales).map(|(a, b)| a * b).collect())
        .collect();
    if scaled.iter().flatten().any(|x| !x.is_finite()) {
        return Err(SparseError::NonFiniteInput {
            message: String::from("non-finite scaled least-squares Jacobian"),
        });
    }
    let operator = AugmentedJacobian {
        matrix: &scaled,
        diagonal: &[],
        columns: scales.len(),
    };
    let result = lsmr_with_options(
        &operator,
        residuals,
        options.linear_options(mode, options.damp),
    )?;
    Ok(result
        .solution
        .iter()
        .zip(scales)
        .map(|(x, d)| -x * d)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::solve_projected;

    #[test]
    fn projected_solver_handles_interior_boundary_and_rank_one() {
        let interior = solve_projected([[4.0, 0.0], [0.0, 1.0]], [2.0, -0.5], 2.0);
        assert!((interior[0] + 0.5).abs() < 1.0e-14);
        assert!((interior[1] - 0.5).abs() < 1.0e-14);
        let singular = solve_projected([[4.0, 0.0], [0.0, 0.0]], [2.0, 0.0], 2.0);
        assert!((singular[0] + 0.5).abs() < 1.0e-14);
        assert!(singular[1].abs() < 1.0e-14);
        let swapped = solve_projected([[0.0, 0.0], [0.0, 4.0]], [0.0, 2.0], 2.0);
        assert_eq!(swapped[0], 0.0);
        assert!((swapped[1] + 0.5).abs() < 1.0e-14);

        let hessian = [[4.0, 1.0], [1.0, 2.0]];
        let gradient = [3.0, -4.0];
        let step = solve_projected(hessian, gradient, 0.25);
        assert!((step[0].hypot(step[1]) - 0.25).abs() < 1.0e-14);
        // KKT: Bp + g must equal -lambda*p for a nonnegative lambda.
        let residual = [
            4.0 * step[0] + step[1] + 3.0,
            step[0] + 2.0 * step[1] - 4.0,
        ];
        assert!((residual[0] * step[1] - residual[1] * step[0]).abs() < 1.0e-13);
        assert!(residual[0] * step[0] + residual[1] * step[1] < 0.0);
    }

    #[test]
    fn projected_solver_has_a_real_null_space_descent_direction() {
        let step = solve_projected([[1.0, 0.0], [0.0, 0.0]], [0.0, 2.0], 0.5);
        assert!(step[0].abs() < 1.0e-14);
        assert!((step[1] + 0.5).abs() < 1.0e-14);
    }
}
