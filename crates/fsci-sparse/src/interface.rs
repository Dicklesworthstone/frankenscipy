//! Matrix-free linear operators: `scipy.sparse.linalg.LinearOperator` and `aslinearoperator`,
//! after SciPy 1.17.1's `scipy/sparse/linalg/_interface.py` (frankenscipy-6j5tz).
//!
//! A [`LinearOperator`] is anything that can apply `y = A·x` (and, optionally, `y = Aᴴ·x`)
//! without exposing its entries. Every sparse format of this crate is one, as are
//! [`FunctionOperator`] (SciPy's `LinearOperator(shape, matvec, rmatvec)`), the dense
//! [`DenseLinearOperator`], and the algebra SciPy builds from them: [`SumOperator`] (`A + B`),
//! [`ProductOperator`] (`A @ B`), [`ScaledOperator`] (`alpha * A`), [`PowerOperator`]
//! (`A ** p`), [`AdjointOperator`] (`A.H`), [`TransposeOperator`] (`A.T`) and
//! [`IdentityOperator`].
//!
//! The iterative solvers of [`crate::linalg`] take `&dyn LinearOperator`, so a concrete matrix
//! (`&CsrMatrix` coerces) and a closure-defined operator that is never materialized go through
//! the same Krylov kernels. Operators here are real (`dtype = float64`): SciPy's adjoint and
//! transpose coincide, and `conj` is the identity.

use std::rc::Rc;
use std::sync::Arc;

use crate::formats::{
    BsrMatrix, CooMatrix, CscMatrix, CsrMatrix, DiaMatrix, DokMatrix, LilMatrix, Shape2D,
    SparseArray2D, SparseError, SparseResult,
};
use crate::linalg::{csc_matvec_into, csr_matvec_into};

/// SciPy's `ValueError('dimension mismatch')` for an operand of the wrong length.
fn dimension_mismatch(operation: &str, operand: &str, got: usize, expected: usize) -> SparseError {
    SparseError::IncompatibleShape {
        message: format!(
            "dimension mismatch: {operation} {operand} has length {got}, expected {expected}"
        ),
    }
}

/// Checks the two vector lengths of one `y = op(x)` application.
fn check_application(
    operation: &str,
    x: &[f64],
    y: &[f64],
    x_len: usize,
    y_len: usize,
) -> SparseResult<()> {
    if x.len() != x_len {
        return Err(dimension_mismatch(operation, "input", x.len(), x_len));
    }
    if y.len() != y_len {
        return Err(dimension_mismatch(operation, "output", y.len(), y_len));
    }
    Ok(())
}

/// SciPy's `NotImplementedError("rmatvec is not defined")`.
fn rmatvec_not_defined() -> SparseError {
    SparseError::Unsupported {
        feature: "rmatvec is not defined for this LinearOperator".to_string(),
    }
}

/// Validates a row-major block `x` with `rows` rows and returns its column count.
fn block_columns(operation: &str, x: &[Vec<f64>], rows: usize) -> SparseResult<usize> {
    if x.len() != rows {
        return Err(SparseError::IncompatibleShape {
            message: format!(
                "dimension mismatch: {operation} operand has {} rows, expected {rows}",
                x.len()
            ),
        });
    }
    let columns = x.first().map_or(0, Vec::len);
    if x.iter().any(|row| row.len() != columns) {
        return Err(SparseError::InvalidShape {
            message: format!("{operation} operand rows must all have the same length"),
        });
    }
    Ok(columns)
}

/// Applies `apply` to every column of the row-major block `x` (`x_rows × k`) and returns the
/// row-major `y_rows × k` result: SciPy's default `_matmat`, `hstack([matvec(col) ...])`.
fn apply_columns(
    x: &[Vec<f64>],
    x_rows: usize,
    y_rows: usize,
    columns: usize,
    mut apply: impl FnMut(&[f64], &mut [f64]) -> SparseResult<()>,
) -> SparseResult<Vec<Vec<f64>>> {
    let mut y = vec![vec![0.0; columns]; y_rows];
    let mut column = vec![0.0; x_rows];
    let mut image = vec![0.0; y_rows];
    for j in 0..columns {
        for (slot, row) in column.iter_mut().zip(x) {
            *slot = row[j];
        }
        apply(&column, &mut image)?;
        for (row, &value) in y.iter_mut().zip(&image) {
            row[j] = value;
        }
    }
    Ok(y)
}

/// A real linear map `A: ℝⁿ → ℝᵐ` given by its action: `scipy.sparse.linalg.LinearOperator`.
///
/// Implementors supply [`shape`](Self::shape) and [`matvec_into`](Self::matvec_into), and
/// [`rmatvec_into`](Self::rmatvec_into) when `Aᴴ·x` is available; the default refuses it with
/// [`SparseError::Unsupported`], SciPy's `NotImplementedError("rmatvec is not defined")`.
/// `matvec`, `rmatvec`, `matmat` and `rmatmat` are provided on top, with SciPy's dimension
/// checks.
///
/// Dense blocks (`matmat`/`rmatmat` operands and results) are row-major `&[Vec<f64>]`, the
/// dense convention of this crate: `x[i][j]` is SciPy's `X[i, j]`.
pub trait LinearOperator {
    /// `(rows, cols)`: the operator maps vectors of length `cols` to length `rows`.
    fn shape(&self) -> Shape2D;

    /// `y = A·x`. `x.len()` must be `shape().cols` and `y.len()` `shape().rows`; the previous
    /// contents of `y` are overwritten. The provided [`matvec`](Self::matvec) checks the
    /// lengths, as does every implementation in this crate (`IncompatibleShape`).
    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()>;

    /// `y = Aᴴ·x` (`Aᵀ·x` for a real operator), with `x.len() == rows` and `y.len() == cols`.
    ///
    /// # Errors
    /// The default is [`SparseError::Unsupported`]: SciPy raises `NotImplementedError` for an
    /// operator built without `rmatvec`.
    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        let _ = (x, y);
        Err(rmatvec_not_defined())
    }

    /// The operator as a concrete CSR matrix, when it is one. Solvers use this to keep their
    /// matrix-specific fast paths (cached transposes, threaded kernels, finite-entry checks,
    /// factorizations); every other operator goes through `matvec`/`rmatvec`.
    fn as_csr(&self) -> Option<&CsrMatrix> {
        None
    }

    /// `A·x` with SciPy's `matvec` check: `x.len()` must equal `shape().cols`.
    fn matvec(&self, x: &[f64]) -> SparseResult<Vec<f64>> {
        let shape = self.shape();
        if x.len() != shape.cols {
            return Err(dimension_mismatch("matvec", "x", x.len(), shape.cols));
        }
        let mut y = vec![0.0; shape.rows];
        self.matvec_into(x, &mut y)?;
        Ok(y)
    }

    /// `Aᴴ·x` with SciPy's `rmatvec` check: `x.len()` must equal `shape().rows`.
    fn rmatvec(&self, x: &[f64]) -> SparseResult<Vec<f64>> {
        let shape = self.shape();
        if x.len() != shape.rows {
            return Err(dimension_mismatch("rmatvec", "x", x.len(), shape.rows));
        }
        let mut y = vec![0.0; shape.cols];
        self.rmatvec_into(x, &mut y)?;
        Ok(y)
    }

    /// `A·X` for a row-major `cols × k` block `X`, column by column (SciPy's default
    /// `_matmat`). Returns the row-major `rows × k` product.
    fn matmat(&self, x: &[Vec<f64>]) -> SparseResult<Vec<Vec<f64>>> {
        let shape = self.shape();
        let columns = block_columns("matmat", x, shape.cols)?;
        apply_columns(x, shape.cols, shape.rows, columns, |column, image| {
            self.matvec_into(column, image)
        })
    }

    /// `Aᴴ·X` for a row-major `rows × k` block `X`. Returns the row-major `cols × k` product.
    fn rmatmat(&self, x: &[Vec<f64>]) -> SparseResult<Vec<Vec<f64>>> {
        let shape = self.shape();
        let columns = block_columns("rmatmat", x, shape.rows)?;
        apply_columns(x, shape.rows, shape.cols, columns, |column, image| {
            self.rmatvec_into(column, image)
        })
    }

    /// The operator as a dense row-major matrix, `A @ eye(cols)`: one `matvec` per column.
    fn to_dense(&self) -> SparseResult<Vec<Vec<f64>>> {
        let shape = self.shape();
        let mut dense = vec![vec![0.0; shape.cols]; shape.rows];
        let mut unit = vec![0.0; shape.cols];
        let mut image = vec![0.0; shape.rows];
        for j in 0..shape.cols {
            unit[j] = 1.0;
            self.matvec_into(&unit, &mut image)?;
            unit[j] = 0.0;
            for (row, &value) in dense.iter_mut().zip(&image) {
                row[j] = value;
            }
        }
        Ok(dense)
    }
}

/// Forwards every method, so an override of a provided method in `T` is kept.
macro_rules! forward_linear_operator {
    ($($wrapper:ty),* $(,)?) => {$(
        impl<T: LinearOperator + ?Sized> LinearOperator for $wrapper {
            fn shape(&self) -> Shape2D {
                (**self).shape()
            }
            fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
                (**self).matvec_into(x, y)
            }
            fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
                (**self).rmatvec_into(x, y)
            }
            fn as_csr(&self) -> Option<&CsrMatrix> {
                (**self).as_csr()
            }
            fn matvec(&self, x: &[f64]) -> SparseResult<Vec<f64>> {
                (**self).matvec(x)
            }
            fn rmatvec(&self, x: &[f64]) -> SparseResult<Vec<f64>> {
                (**self).rmatvec(x)
            }
            fn matmat(&self, x: &[Vec<f64>]) -> SparseResult<Vec<Vec<f64>>> {
                (**self).matmat(x)
            }
            fn rmatmat(&self, x: &[Vec<f64>]) -> SparseResult<Vec<Vec<f64>>> {
                (**self).rmatmat(x)
            }
            fn to_dense(&self) -> SparseResult<Vec<Vec<f64>>> {
                (**self).to_dense()
            }
        }
    )*};
}

forward_linear_operator!(&T, Box<T>, Rc<T>, Arc<T>);

impl<M: LinearOperator> LinearOperator for SparseArray2D<M> {
    fn shape(&self) -> Shape2D {
        LinearOperator::shape(self.as_matrix())
    }
    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        self.as_matrix().matvec_into(x, y)
    }
    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        self.as_matrix().rmatvec_into(x, y)
    }
    fn as_csr(&self) -> Option<&CsrMatrix> {
        self.as_matrix().as_csr()
    }
}

impl LinearOperator for CsrMatrix {
    fn shape(&self) -> Shape2D {
        self.shape
    }

    /// The threaded row-dot kernel every Krylov solver used before operators existed, so a
    /// solve on a concrete CSR matrix is unchanged bit for bit.
    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("matvec", x, y, self.shape.cols, self.shape.rows)?;
        csr_matvec_into(self, x, y);
        Ok(())
    }

    /// `Aᵀ·x` as a row scatter, accumulating each output in increasing row order.
    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("rmatvec", x, y, self.shape.rows, self.shape.cols)?;
        y.fill(0.0);
        for (row, &xi) in x.iter().enumerate() {
            for idx in self.indptr[row]..self.indptr[row + 1] {
                y[self.indices[idx]] += self.data[idx] * xi;
            }
        }
        Ok(())
    }

    fn as_csr(&self) -> Option<&CsrMatrix> {
        Some(self)
    }
}

impl LinearOperator for CscMatrix {
    fn shape(&self) -> Shape2D {
        self.shape
    }

    /// `A·x` as a column scatter (`spmv_csc`'s order).
    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("matvec", x, y, self.shape.cols, self.shape.rows)?;
        y.fill(0.0);
        for (col, &xc) in x.iter().enumerate() {
            for idx in self.indptr[col]..self.indptr[col + 1] {
                y[self.indices[idx]] += self.data[idx] * xc;
            }
        }
        Ok(())
    }

    /// `Aᵀ·x` as a per-column gather.
    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("rmatvec", x, y, self.shape.rows, self.shape.cols)?;
        csc_matvec_into(self, x, y);
        Ok(())
    }
}

impl LinearOperator for CooMatrix {
    fn shape(&self) -> Shape2D {
        self.shape
    }

    /// `A·x` in stored order, duplicates summed (`spmv_coo`'s order).
    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("matvec", x, y, self.shape.cols, self.shape.rows)?;
        y.fill(0.0);
        for ((&row, &col), &value) in self
            .row_indices
            .iter()
            .zip(&self.col_indices)
            .zip(&self.data)
        {
            y[row] += value * x[col];
        }
        Ok(())
    }

    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("rmatvec", x, y, self.shape.rows, self.shape.cols)?;
        y.fill(0.0);
        for ((&row, &col), &value) in self
            .row_indices
            .iter()
            .zip(&self.col_indices)
            .zip(&self.data)
        {
            y[col] += value * x[row];
        }
        Ok(())
    }
}

impl LinearOperator for BsrMatrix {
    fn shape(&self) -> Shape2D {
        self.shape
    }

    /// `A·x` block row by block row; a block is stored row-major (`r · block_cols + c`).
    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("matvec", x, y, self.shape.cols, self.shape.rows)?;
        y.fill(0.0);
        let (br, bc) = (self.block_shape.rows, self.block_shape.cols);
        for block_row in 0..self.indptr.len().saturating_sub(1) {
            for k in self.indptr[block_row]..self.indptr[block_row + 1] {
                let block = &self.data[k];
                let col0 = self.indices[k] * bc;
                for r in 0..br {
                    let mut sum = 0.0;
                    for c in 0..bc {
                        sum += block[r * bc + c] * x[col0 + c];
                    }
                    y[block_row * br + r] += sum;
                }
            }
        }
        Ok(())
    }

    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("rmatvec", x, y, self.shape.rows, self.shape.cols)?;
        y.fill(0.0);
        let (br, bc) = (self.block_shape.rows, self.block_shape.cols);
        for block_row in 0..self.indptr.len().saturating_sub(1) {
            for k in self.indptr[block_row]..self.indptr[block_row + 1] {
                let block = &self.data[k];
                let col0 = self.indices[k] * bc;
                for r in 0..br {
                    let xr = x[block_row * br + r];
                    for c in 0..bc {
                        y[col0 + c] += block[r * bc + c] * xr;
                    }
                }
            }
        }
        Ok(())
    }
}

impl LinearOperator for DiaMatrix {
    fn shape(&self) -> Shape2D {
        self.shape
    }

    /// `A·x` diagonal by diagonal (the inherent `DiaMatrix::matvec` order); `data[k][j]` is
    /// the entry in column `j` of diagonal `offsets[k]`, as SciPy stores it.
    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("matvec", x, y, self.shape.cols, self.shape.rows)?;
        y.fill(0.0);
        for (k, &offset) in self.offsets.iter().enumerate() {
            let diagonal = &self.data[k];
            for (i, yi) in y.iter_mut().enumerate() {
                let j = i as isize + offset;
                if j >= 0 && (j as usize) < self.shape.cols {
                    let j = j as usize;
                    let idx = if offset >= 0 { i } else { j };
                    if idx < diagonal.len() {
                        *yi += diagonal[idx] * x[j];
                    }
                }
            }
        }
        Ok(())
    }

    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("rmatvec", x, y, self.shape.rows, self.shape.cols)?;
        y.fill(0.0);
        for (k, &offset) in self.offsets.iter().enumerate() {
            let diagonal = &self.data[k];
            for (i, &xi) in x.iter().enumerate() {
                let j = i as isize + offset;
                if j >= 0 && (j as usize) < self.shape.cols {
                    let j = j as usize;
                    let idx = if offset >= 0 { i } else { j };
                    if idx < diagonal.len() {
                        y[j] += diagonal[idx] * xi;
                    }
                }
            }
        }
        Ok(())
    }
}

impl LinearOperator for DokMatrix {
    fn shape(&self) -> Shape2D {
        self.shape
    }

    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("matvec", x, y, self.shape.cols, self.shape.rows)?;
        y.fill(0.0);
        for (&(row, col), &value) in &self.entries {
            y[row] += value * x[col];
        }
        Ok(())
    }

    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("rmatvec", x, y, self.shape.rows, self.shape.cols)?;
        y.fill(0.0);
        for (&(row, col), &value) in &self.entries {
            y[col] += value * x[row];
        }
        Ok(())
    }
}

impl LinearOperator for LilMatrix {
    fn shape(&self) -> Shape2D {
        self.shape
    }

    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("matvec", x, y, self.shape.cols, self.shape.rows)?;
        for ((yi, columns), values) in y.iter_mut().zip(&self.row_indices).zip(&self.row_data) {
            let mut sum = 0.0;
            for (&col, &value) in columns.iter().zip(values) {
                sum += value * x[col];
            }
            *yi = sum;
        }
        Ok(())
    }

    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("rmatvec", x, y, self.shape.rows, self.shape.cols)?;
        y.fill(0.0);
        for ((&xi, columns), values) in x.iter().zip(&self.row_indices).zip(&self.row_data) {
            for (&col, &value) in columns.iter().zip(values) {
                y[col] += value * xi;
            }
        }
        Ok(())
    }
}

/// A fallible vector map `x ↦ y`, the shape of a user-supplied `matvec`/`rmatvec`.
type VectorMap<'a> = Box<dyn Fn(&[f64]) -> SparseResult<Vec<f64>> + 'a>;

/// An operator defined by closures: SciPy's `LinearOperator(shape, matvec, rmatvec=None)`
/// (`_CustomLinearOperator`). Nothing is ever materialized.
///
/// ```
/// use fsci_sparse::{FunctionOperator, LinearOperator, Shape2D};
/// // The 1-D Dirichlet Laplacian of size 4, never stored.
/// let laplacian = FunctionOperator::new(Shape2D::new(4, 4), |x: &[f64]| {
///     Ok((0..x.len())
///         .map(|i| {
///             let left = if i > 0 { x[i - 1] } else { 0.0 };
///             let right = if i + 1 < x.len() { x[i + 1] } else { 0.0 };
///             2.0 * x[i] - left - right
///         })
///         .collect())
/// });
/// assert_eq!(laplacian.matvec(&[1.0, 1.0, 1.0, 1.0]).unwrap(), vec![1.0, 0.0, 0.0, 1.0]);
/// ```
pub struct FunctionOperator<'a> {
    shape: Shape2D,
    matvec: VectorMap<'a>,
    rmatvec: Option<VectorMap<'a>>,
}

impl<'a> FunctionOperator<'a> {
    /// An operator of `shape` whose `A·x` is `matvec(x)`; `rmatvec` is undefined until
    /// [`with_rmatvec`](Self::with_rmatvec) supplies it.
    pub fn new<F>(shape: Shape2D, matvec: F) -> Self
    where
        F: Fn(&[f64]) -> SparseResult<Vec<f64>> + 'a,
    {
        Self {
            shape,
            matvec: Box::new(matvec),
            rmatvec: None,
        }
    }

    /// Supplies `Aᴴ·x` (SciPy's `rmatvec=` argument).
    #[must_use]
    pub fn with_rmatvec<G>(mut self, rmatvec: G) -> Self
    where
        G: Fn(&[f64]) -> SparseResult<Vec<f64>> + 'a,
    {
        self.rmatvec = Some(Box::new(rmatvec));
        self
    }
}

impl std::fmt::Debug for FunctionOperator<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionOperator")
            .field("shape", &self.shape)
            .field("rmatvec", &self.rmatvec.is_some())
            .finish_non_exhaustive()
    }
}

/// Copies a user closure's result into `y`, refusing a result of the wrong length as SciPy
/// does (`'invalid shape returned by user-defined matvec()'`).
fn copy_user_result(name: &str, result: &[f64], y: &mut [f64]) -> SparseResult<()> {
    if result.len() != y.len() {
        return Err(SparseError::InvalidShape {
            message: format!(
                "invalid shape returned by user-defined {name}(): length {}, expected {}",
                result.len(),
                y.len()
            ),
        });
    }
    y.copy_from_slice(result);
    Ok(())
}

impl LinearOperator for FunctionOperator<'_> {
    fn shape(&self) -> Shape2D {
        self.shape
    }

    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("matvec", x, y, self.shape.cols, self.shape.rows)?;
        let result = (self.matvec)(x)?;
        copy_user_result("matvec", &result, y)
    }

    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("rmatvec", x, y, self.shape.rows, self.shape.cols)?;
        let rmatvec = self.rmatvec.as_ref().ok_or_else(rmatvec_not_defined)?;
        let result = rmatvec(x)?;
        copy_user_result("rmatvec", &result, y)
    }
}

/// A dense matrix as an operator: SciPy's `aslinearoperator(ndarray)` (`MatrixLinearOperator`).
#[derive(Debug, Clone, PartialEq)]
pub struct DenseLinearOperator {
    shape: Shape2D,
    /// Row-major entries.
    data: Vec<f64>,
}

impl DenseLinearOperator {
    /// Wraps the row-major matrix `rows` (`rows[i][j] = A[i, j]`). An empty slice is the 0×0
    /// operator.
    ///
    /// # Errors
    /// [`SparseError::InvalidShape`] when the rows do not all have the same length.
    pub fn from_rows(rows: &[Vec<f64>]) -> SparseResult<Self> {
        let cols = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|row| row.len() != cols) {
            return Err(SparseError::InvalidShape {
                message: "dense operator rows must all have the same length".to_string(),
            });
        }
        Ok(Self {
            shape: Shape2D::new(rows.len(), cols),
            data: rows.iter().flatten().copied().collect(),
        })
    }
}

impl LinearOperator for DenseLinearOperator {
    fn shape(&self) -> Shape2D {
        self.shape
    }

    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("matvec", x, y, self.shape.cols, self.shape.rows)?;
        let cols = self.shape.cols;
        for (i, yi) in y.iter_mut().enumerate() {
            let row = &self.data[i * cols..(i + 1) * cols];
            *yi = row.iter().zip(x).map(|(a, b)| a * b).sum();
        }
        Ok(())
    }

    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("rmatvec", x, y, self.shape.rows, self.shape.cols)?;
        y.fill(0.0);
        let cols = self.shape.cols;
        for (i, &xi) in x.iter().enumerate() {
            let row = &self.data[i * cols..(i + 1) * cols];
            for (yj, &aij) in y.iter_mut().zip(row) {
                *yj += aij * xi;
            }
        }
        Ok(())
    }
}

/// The `n × n` identity: SciPy's `IdentityOperator((n, n))`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdentityOperator {
    n: usize,
}

impl IdentityOperator {
    #[must_use]
    pub const fn new(n: usize) -> Self {
        Self { n }
    }
}

impl LinearOperator for IdentityOperator {
    fn shape(&self) -> Shape2D {
        Shape2D::new(self.n, self.n)
    }

    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("matvec", x, y, self.n, self.n)?;
        y.copy_from_slice(x);
        Ok(())
    }

    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        check_application("rmatvec", x, y, self.n, self.n)?;
        y.copy_from_slice(x);
        Ok(())
    }
}

/// `A + B`: SciPy's `_SumLinearOperator`, `A.matvec(x) + B.matvec(x)`.
#[derive(Debug, Clone)]
pub struct SumOperator<A, B> {
    a: A,
    b: B,
}

impl<A: LinearOperator, B: LinearOperator> SumOperator<A, B> {
    /// # Errors
    /// [`SparseError::IncompatibleShape`] when the shapes differ (SciPy: "shape mismatch").
    pub fn new(a: A, b: B) -> SparseResult<Self> {
        let (sa, sb) = (a.shape(), b.shape());
        if sa != sb {
            return Err(SparseError::IncompatibleShape {
                message: format!(
                    "cannot add {}x{} and {}x{} operators: shape mismatch",
                    sa.rows, sa.cols, sb.rows, sb.cols
                ),
            });
        }
        Ok(Self { a, b })
    }
}

impl<A: LinearOperator, B: LinearOperator> LinearOperator for SumOperator<A, B> {
    fn shape(&self) -> Shape2D {
        self.a.shape()
    }

    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        self.a.matvec_into(x, y)?;
        let second = self.b.matvec(x)?;
        for (yi, si) in y.iter_mut().zip(second) {
            *yi += si;
        }
        Ok(())
    }

    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        self.a.rmatvec_into(x, y)?;
        let second = self.b.rmatvec(x)?;
        for (yi, si) in y.iter_mut().zip(second) {
            *yi += si;
        }
        Ok(())
    }
}

/// `A @ B`: SciPy's `_ProductLinearOperator`, `A.matvec(B.matvec(x))`.
#[derive(Debug, Clone)]
pub struct ProductOperator<A, B> {
    a: A,
    b: B,
}

impl<A: LinearOperator, B: LinearOperator> ProductOperator<A, B> {
    /// # Errors
    /// [`SparseError::IncompatibleShape`] unless `A.cols == B.rows`.
    pub fn new(a: A, b: B) -> SparseResult<Self> {
        let (sa, sb) = (a.shape(), b.shape());
        if sa.cols != sb.rows {
            return Err(SparseError::IncompatibleShape {
                message: format!(
                    "cannot multiply {}x{} and {}x{} operators: shape mismatch",
                    sa.rows, sa.cols, sb.rows, sb.cols
                ),
            });
        }
        Ok(Self { a, b })
    }
}

impl<A: LinearOperator, B: LinearOperator> LinearOperator for ProductOperator<A, B> {
    fn shape(&self) -> Shape2D {
        Shape2D::new(self.a.shape().rows, self.b.shape().cols)
    }

    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        let inner = self.b.matvec(x)?;
        self.a.matvec_into(&inner, y)
    }

    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        let inner = self.a.rmatvec(x)?;
        self.b.rmatvec_into(&inner, y)
    }
}

/// `alpha * A`: SciPy's `_ScaledLinearOperator`.
#[derive(Debug, Clone)]
pub struct ScaledOperator<A> {
    a: A,
    alpha: f64,
}

impl<A: LinearOperator> ScaledOperator<A> {
    #[must_use]
    pub const fn new(a: A, alpha: f64) -> Self {
        Self { a, alpha }
    }

    /// `beta * (alpha * A)` as one scaling by `beta * alpha`, the collapse SciPy performs when
    /// a scaled operator is scaled again.
    #[must_use]
    pub fn rescale(self, beta: f64) -> Self {
        Self {
            a: self.a,
            alpha: beta * self.alpha,
        }
    }

    /// The scale factor.
    #[must_use]
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }
}

impl<A: LinearOperator> LinearOperator for ScaledOperator<A> {
    fn shape(&self) -> Shape2D {
        self.a.shape()
    }

    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        self.a.matvec_into(x, y)?;
        y.iter_mut().for_each(|yi| *yi *= self.alpha);
        Ok(())
    }

    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        self.a.rmatvec_into(x, y)?;
        y.iter_mut().for_each(|yi| *yi *= self.alpha);
        Ok(())
    }
}

/// `A ** p` for a square `A` and integer `p ≥ 0`: SciPy's `_PowerLinearOperator`, `p`
/// successive applications (`A ** 0` is the identity).
#[derive(Debug, Clone)]
pub struct PowerOperator<A> {
    a: A,
    p: usize,
}

impl<A: LinearOperator> PowerOperator<A> {
    /// # Errors
    /// [`SparseError::InvalidShape`] for a non-square `A` (SciPy: "square LinearOperator
    /// expected").
    pub fn new(a: A, p: usize) -> SparseResult<Self> {
        let shape = a.shape();
        if !shape.is_square() {
            return Err(SparseError::InvalidShape {
                message: format!(
                    "square LinearOperator expected, got {}x{}",
                    shape.rows, shape.cols
                ),
            });
        }
        Ok(Self { a, p })
    }

    fn power(
        &self,
        x: &[f64],
        y: &mut [f64],
        apply: impl Fn(&A, &[f64], &mut [f64]) -> SparseResult<()>,
    ) -> SparseResult<()> {
        if y.len() != x.len() {
            return Err(dimension_mismatch("power", "output", y.len(), x.len()));
        }
        y.copy_from_slice(x);
        let mut scratch = vec![0.0; x.len()];
        for _ in 0..self.p {
            apply(&self.a, y, &mut scratch)?;
            y.copy_from_slice(&scratch);
        }
        Ok(())
    }
}

impl<A: LinearOperator> LinearOperator for PowerOperator<A> {
    fn shape(&self) -> Shape2D {
        self.a.shape()
    }

    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        self.power(x, y, |a, v, out| a.matvec_into(v, out))
    }

    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        self.power(x, y, |a, v, out| a.rmatvec_into(v, out))
    }
}

/// `Aᴴ` (SciPy's `A.H`, `_AdjointLinearOperator`): `matvec` is `A.rmatvec` and vice versa.
#[derive(Debug, Clone)]
pub struct AdjointOperator<A> {
    a: A,
}

impl<A: LinearOperator> AdjointOperator<A> {
    #[must_use]
    pub const fn new(a: A) -> Self {
        Self { a }
    }
}

impl<A: LinearOperator> LinearOperator for AdjointOperator<A> {
    fn shape(&self) -> Shape2D {
        let shape = self.a.shape();
        Shape2D::new(shape.cols, shape.rows)
    }

    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        self.a.rmatvec_into(x, y)
    }

    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        self.a.matvec_into(x, y)
    }
}

/// `Aᵀ` (SciPy's `A.T`, `_TransposedLinearOperator`: `conj(A.rmatvec(conj(x)))`, which for a
/// real operator is `A.rmatvec(x)`, the same map as [`AdjointOperator`]).
#[derive(Debug, Clone)]
pub struct TransposeOperator<A> {
    a: A,
}

impl<A: LinearOperator> TransposeOperator<A> {
    #[must_use]
    pub const fn new(a: A) -> Self {
        Self { a }
    }
}

impl<A: LinearOperator> LinearOperator for TransposeOperator<A> {
    fn shape(&self) -> Shape2D {
        let shape = self.a.shape();
        Shape2D::new(shape.cols, shape.rows)
    }

    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        self.a.rmatvec_into(x, y)
    }

    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        self.a.matvec_into(x, y)
    }
}

/// `scipy.sparse.linalg.aslinearoperator(A)`: `A` as a boxed [`LinearOperator`].
///
/// Every sparse format, [`FunctionOperator`], [`DenseLinearOperator`] (SciPy's ndarray branch)
/// and the operator algebra are accepted; a reference works too (`aslinearoperator(&matrix)`),
/// so the matrix is borrowed rather than moved.
pub fn aslinearoperator<'a, T: LinearOperator + 'a>(a: T) -> Box<dyn LinearOperator + 'a> {
    Box::new(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FormatConvertible;

    fn sample_csr() -> CsrMatrix {
        // [[1, 0, 2], [0, 3, 0]]
        CsrMatrix::from_components(
            Shape2D::new(2, 3),
            vec![1.0, 2.0, 3.0],
            vec![0, 2, 1],
            vec![0, 2, 3],
            true,
        )
        .expect("valid csr")
    }

    fn dense_matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        a.iter()
            .map(|row| row.iter().zip(x).map(|(p, q)| p * q).sum())
            .collect()
    }

    #[test]
    fn every_format_applies_the_same_matrix_and_its_transpose() {
        let csr = sample_csr();
        let dense = vec![vec![1.0, 0.0, 2.0], vec![0.0, 3.0, 0.0]];
        let x = [0.5, -1.25, 2.0];
        let u = [3.0, -0.5];
        let expected = dense_matvec(&dense, &x);
        let expected_t: Vec<f64> = (0..3)
            .map(|j| dense[0][j] * u[0] + dense[1][j] * u[1])
            .collect();
        let operators: Vec<Box<dyn LinearOperator>> = vec![
            Box::new(csr.clone()),
            Box::new(csr.to_csc().unwrap()),
            Box::new(csr.to_coo().unwrap()),
            Box::new(SparseArray2D::new(csr.clone())),
            Box::new(DenseLinearOperator::from_rows(&dense).unwrap()),
            Box::new(
                crate::DokMatrix::from_triplets(
                    Shape2D::new(2, 3),
                    vec![1.0, 2.0, 3.0],
                    vec![0, 0, 1],
                    vec![0, 2, 1],
                )
                .unwrap(),
            ),
        ];
        for op in &operators {
            assert_eq!(op.shape(), Shape2D::new(2, 3));
            assert_eq!(op.matvec(&x).unwrap(), expected);
            assert_eq!(op.rmatvec(&u).unwrap(), expected_t);
            assert_eq!(op.to_dense().unwrap(), dense);
        }
        let lil = csr.to_coo().unwrap();
        let lil = crate::LilMatrix::from_triplets(
            lil.shape(),
            lil.data().to_vec(),
            lil.row_indices().to_vec(),
            lil.col_indices().to_vec(),
        )
        .unwrap();
        assert_eq!(lil.matvec(&x).unwrap(), expected);
        assert_eq!(LinearOperator::rmatvec(&lil, &u).unwrap(), expected_t);
    }

    #[test]
    fn dia_and_bsr_operators_match_their_dense_form() {
        let dense = vec![
            vec![4.0, 1.0, 0.0, 0.0],
            vec![2.0, 5.0, 1.0, 0.0],
            vec![0.0, 2.0, 6.0, 1.0],
            vec![0.0, 0.0, 2.0, 7.0],
        ];
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for (i, row) in dense.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    rows.push(i);
                    cols.push(j);
                    vals.push(v);
                }
            }
        }
        let shape = Shape2D::new(4, 4);
        let dia =
            DiaMatrix::from_triplets(shape, vals.clone(), rows.clone(), cols.clone()).expect("dia");
        let bsr =
            BsrMatrix::from_triplets(shape, Shape2D::new(2, 2), vals, rows, cols).expect("bsr");
        let x = [1.0, -2.0, 0.5, 3.0];
        let expected = dense_matvec(&dense, &x);
        let transposed: Vec<Vec<f64>> = (0..4)
            .map(|j| (0..4).map(|i| dense[i][j]).collect())
            .collect();
        let expected_t = dense_matvec(&transposed, &x);
        for op in [&dia as &dyn LinearOperator, &bsr] {
            assert_eq!(op.matvec(&x).unwrap(), expected);
            assert_eq!(op.rmatvec(&x).unwrap(), expected_t);
        }
    }

    #[test]
    fn matvec_refuses_a_wrong_length_operand() {
        let csr = sample_csr();
        let op: &dyn LinearOperator = &csr;
        assert!(matches!(
            op.matvec(&[1.0, 2.0]),
            Err(SparseError::IncompatibleShape { .. })
        ));
        assert!(matches!(
            op.rmatvec(&[1.0, 2.0, 3.0]),
            Err(SparseError::IncompatibleShape { .. })
        ));
        let mut short = [0.0; 1];
        assert!(op.matvec_into(&[1.0, 2.0, 3.0], &mut short).is_err());
    }

    #[test]
    fn function_operator_without_rmatvec_refuses_it_like_scipy() {
        let op = FunctionOperator::new(Shape2D::new(2, 2), |x: &[f64]| Ok(vec![x[1], x[0]]));
        assert_eq!(op.matvec(&[1.0, 2.0]).unwrap(), vec![2.0, 1.0]);
        assert!(matches!(
            op.rmatvec(&[1.0, 2.0]),
            Err(SparseError::Unsupported { .. })
        ));
        let with = op.with_rmatvec(|x: &[f64]| Ok(vec![x[1], x[0]]));
        assert_eq!(with.rmatvec(&[1.0, 2.0]).unwrap(), vec![2.0, 1.0]);
    }

    #[test]
    fn function_operator_refuses_a_wrong_length_result() {
        let op = FunctionOperator::new(Shape2D::new(3, 2), |_x: &[f64]| Ok(vec![1.0, 2.0]));
        assert!(matches!(
            op.matvec(&[1.0, 2.0]),
            Err(SparseError::InvalidShape { .. })
        ));
    }

    #[test]
    fn operator_algebra_matches_dense_arithmetic() {
        let a = DenseLinearOperator::from_rows(&[vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let b = DenseLinearOperator::from_rows(&[vec![0.0, 1.0], vec![-1.0, 0.5]]).unwrap();
        let x = [2.0, -1.0];
        let ax = a.matvec(&x).unwrap();
        let bx = b.matvec(&x).unwrap();

        let sum = SumOperator::new(&a, &b).unwrap();
        assert_eq!(sum.matvec(&x).unwrap(), vec![ax[0] + bx[0], ax[1] + bx[1]]);

        let product = ProductOperator::new(&a, &b).unwrap();
        assert_eq!(product.matvec(&x).unwrap(), a.matvec(&bx).unwrap());
        // (AB)ᵀ = BᵀAᵀ
        assert_eq!(
            product.rmatvec(&x).unwrap(),
            b.rmatvec(&a.rmatvec(&x).unwrap()).unwrap()
        );

        let scaled = ScaledOperator::new(&a, -2.0);
        assert_eq!(scaled.matvec(&x).unwrap(), vec![-2.0 * ax[0], -2.0 * ax[1]]);
        assert_eq!(scaled.rescale(0.5).alpha(), -1.0);

        let squared = PowerOperator::new(&a, 2).unwrap();
        assert_eq!(squared.matvec(&x).unwrap(), a.matvec(&ax).unwrap());
        let zeroth = PowerOperator::new(&a, 0).unwrap();
        assert_eq!(zeroth.matvec(&x).unwrap(), x.to_vec());

        let adjoint = AdjointOperator::new(&a);
        assert_eq!(adjoint.matvec(&x).unwrap(), a.rmatvec(&x).unwrap());
        assert_eq!(adjoint.rmatvec(&x).unwrap(), ax);
        let transpose = TransposeOperator::new(&a);
        assert_eq!(transpose.matvec(&x).unwrap(), vec![-1.0, 0.0]);

        let identity = IdentityOperator::new(2);
        assert_eq!(identity.matvec(&x).unwrap(), x.to_vec());
    }

    #[test]
    fn operator_algebra_refuses_shape_mismatches() {
        let csr = sample_csr(); // 2x3
        let square = IdentityOperator::new(2);
        assert!(matches!(
            SumOperator::new(&csr, &square),
            Err(SparseError::IncompatibleShape { .. })
        ));
        assert!(matches!(
            ProductOperator::new(&csr, &square),
            Err(SparseError::IncompatibleShape { .. })
        ));
        assert!(ProductOperator::new(&square, &csr).is_ok());
        assert!(matches!(
            PowerOperator::new(&csr, 2),
            Err(SparseError::InvalidShape { .. })
        ));
        let adjoint = AdjointOperator::new(&csr);
        assert_eq!(adjoint.shape(), Shape2D::new(3, 2));
    }

    #[test]
    fn matmat_and_rmatmat_apply_column_by_column() {
        let csr = sample_csr();
        let x = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![2.0, -1.0]];
        let y = csr.matmat(&x).unwrap();
        assert_eq!(y, vec![vec![5.0, -2.0], vec![0.0, 3.0]]);
        let u = vec![vec![1.0], vec![2.0]];
        assert_eq!(
            csr.rmatmat(&u).unwrap(),
            vec![vec![1.0], vec![6.0], vec![2.0]]
        );
        assert!(csr.matmat(&u).is_err());
        let ragged = vec![vec![1.0], vec![2.0, 3.0], vec![4.0]];
        assert!(matches!(
            csr.matmat(&ragged),
            Err(SparseError::InvalidShape { .. })
        ));
    }

    #[test]
    fn empty_and_one_by_one_operators() {
        let empty = DenseLinearOperator::from_rows(&[]).unwrap();
        assert_eq!(empty.shape(), Shape2D::new(0, 0));
        assert_eq!(empty.matvec(&[]).unwrap(), Vec::<f64>::new());
        assert_eq!(empty.to_dense().unwrap(), Vec::<Vec<f64>>::new());
        let one = DenseLinearOperator::from_rows(&[vec![-3.0]]).unwrap();
        assert_eq!(one.matvec(&[2.0]).unwrap(), vec![-6.0]);
        assert_eq!(one.rmatvec(&[2.0]).unwrap(), vec![-6.0]);
        assert!(DenseLinearOperator::from_rows(&[vec![1.0], vec![1.0, 2.0]]).is_err());
    }

    #[test]
    fn aslinearoperator_borrows_or_owns() {
        let csr = sample_csr();
        let borrowed = aslinearoperator(&csr);
        assert!(borrowed.as_csr().is_some());
        let owned = aslinearoperator(csr.clone());
        assert_eq!(
            owned.matvec(&[1.0, 1.0, 1.0]).unwrap(),
            borrowed.matvec(&[1.0, 1.0, 1.0]).unwrap()
        );
        let function =
            aslinearoperator(FunctionOperator::new(Shape2D::new(1, 1), |x: &[f64]| {
                Ok(vec![2.0 * x[0]])
            }));
        assert!(function.as_csr().is_none());
        assert_eq!(function.matvec(&[4.0]).unwrap(), vec![8.0]);
    }
}
