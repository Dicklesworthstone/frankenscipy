#![forbid(unsafe_code)]

//! Jacobian storage shared by the nonlinear least-squares algorithms.
//!
//! Sparse and matrix-free Jacobians retain their representation through robust-loss
//! scaling, parameter scaling, active-variable selection, and iterative trust-region
//! subproblems. Dense storage is materialized only when explicitly requested.

use fsci_sparse::{CsrMatrix, LinearOperator, Shape2D, SparseError, SparseResult};
use std::sync::Arc;

/// An owned matrix-free Jacobian, including any scaling applied by the solver.
///
/// Both products must be implemented by the supplied [`LinearOperator`]. The wrapper
/// validates dimensions and finite outputs on every application, including after
/// robust-loss scaling and active-variable selection. Clones share the underlying
/// operator; equality means the same operator and the same transformations.
#[derive(Clone)]
pub struct LeastSquaresOperator {
    operator: Arc<dyn LinearOperator + Send + Sync>,
    original_shape: Shape2D,
    row_scale: Option<Arc<[f64]>>,
    column_scale: Option<Arc<[f64]>>,
    columns: Option<Arc<[usize]>>,
}

impl std::fmt::Debug for LeastSquaresOperator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LeastSquaresOperator")
            .field("shape", &self.shape())
            .field("row_scale", &self.row_scale)
            .field("column_scale", &self.column_scale)
            .field("columns", &self.columns)
            .finish_non_exhaustive()
    }
}

impl PartialEq for LeastSquaresOperator {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.operator, &other.operator)
            && self.original_shape == other.original_shape
            && self.row_scale == other.row_scale
            && self.column_scale == other.column_scale
            && self.columns == other.columns
    }
}

impl LeastSquaresOperator {
    fn new<O>(operator: O) -> Self
    where
        O: LinearOperator + Send + Sync + 'static,
    {
        Self {
            original_shape: operator.shape(),
            operator: Arc::new(operator),
            row_scale: None,
            column_scale: None,
            columns: None,
        }
    }

    fn validate_application(&self, x: &[f64], y: &[f64], transpose: bool) -> SparseResult<()> {
        if self.operator.shape() != self.original_shape {
            return Err(SparseError::IncompatibleShape {
                message: String::from("matrix-free Jacobian changed shape after evaluation"),
            });
        }
        let shape = self.shape();
        let (input, output) = if transpose {
            (shape.rows, shape.cols)
        } else {
            (shape.cols, shape.rows)
        };
        if x.len() != input || y.len() != output {
            return Err(SparseError::IncompatibleShape {
                message: format!(
                    "matrix-free Jacobian product requires input length {input} and output length {output}"
                ),
            });
        }
        finite_product(x)
    }

    fn scale_rows(&mut self, scale: &[f64]) {
        self.row_scale = Some(
            match &self.row_scale {
                Some(old) => old.iter().zip(scale).map(|(a, b)| a * b).collect(),
                None => scale.to_vec(),
            }
            .into(),
        );
    }

    fn scaled_columns(&self, scale: &[f64]) -> Self {
        let mut out = self.clone();
        out.column_scale = Some(
            match &self.column_scale {
                Some(old) => old.iter().zip(scale).map(|(a, b)| a * b).collect(),
                None => scale.to_vec(),
            }
            .into(),
        );
        out
    }

    fn select_columns(&self, columns: &[usize]) -> Self {
        let mut out = self.clone();
        out.columns = Some(
            columns
                .iter()
                .map(|&column| self.columns.as_ref().map_or(column, |old| old[column]))
                .collect::<Vec<_>>()
                .into(),
        );
        out.column_scale = self.column_scale.as_ref().map(|old| {
            columns
                .iter()
                .map(|&column| old[column])
                .collect::<Vec<_>>()
                .into()
        });
        out
    }

    fn scales_finite(&self) -> bool {
        self.row_scale
            .iter()
            .chain(&self.column_scale)
            .flat_map(|scale| scale.iter())
            .all(|value| value.is_finite())
    }
}

fn finite_product(values: &[f64]) -> SparseResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SparseError::NonFiniteInput {
            message: String::from("matrix-free Jacobian product contains NaN or Inf"),
        });
    }
    Ok(())
}

impl LinearOperator for LeastSquaresOperator {
    fn shape(&self) -> Shape2D {
        Shape2D::new(
            self.original_shape.rows,
            self.columns
                .as_ref()
                .map_or(self.original_shape.cols, |columns| columns.len()),
        )
    }

    fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        self.validate_application(x, y, false)?;
        let transformed;
        let input = if self.columns.is_some() || self.column_scale.is_some() {
            let mut full = vec![0.0; self.original_shape.cols];
            for (i, &value) in x.iter().enumerate() {
                let column = self.columns.as_ref().map_or(i, |columns| columns[i]);
                full[column] = value * self.column_scale.as_ref().map_or(1.0, |scale| scale[i]);
            }
            finite_product(&full)?;
            transformed = full;
            &transformed
        } else {
            x
        };
        // Unwritten outputs must not masquerade as valid zeros.
        y.fill(f64::NAN);
        self.operator.matvec_into(input, y)?;
        finite_product(y)?;
        if let Some(scale) = &self.row_scale {
            for (value, &factor) in y.iter_mut().zip(scale.iter()) {
                *value *= factor;
            }
            finite_product(y)?;
        }
        Ok(())
    }

    fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
        self.validate_application(x, y, true)?;
        let transformed;
        let input = if let Some(scale) = &self.row_scale {
            transformed = x
                .iter()
                .zip(scale.iter())
                .map(|(a, b)| a * b)
                .collect::<Vec<_>>();
            finite_product(&transformed)?;
            &transformed
        } else {
            x
        };
        if self.columns.is_some() || self.column_scale.is_some() {
            let mut full = vec![f64::NAN; self.original_shape.cols];
            self.operator.rmatvec_into(input, &mut full)?;
            finite_product(&full)?;
            for (i, value) in y.iter_mut().enumerate() {
                let column = self.columns.as_ref().map_or(i, |columns| columns[i]);
                *value = full[column] * self.column_scale.as_ref().map_or(1.0, |scale| scale[i]);
            }
        } else {
            y.fill(f64::NAN);
            self.operator.rmatvec_into(input, y)?;
        }
        finite_product(y)
    }
}

/// A residual Jacobian, with one row per residual and one column per parameter.
///
/// `Sparse` stores a compressed sparse row matrix. `Operator` holds matrix-vector
/// products without materialized entries. Least-squares entry points validate shapes,
/// stored entries, and operator products. [`Self::to_dense`] is an explicit conversion;
/// the LSMR solver does not call it.
#[derive(Debug, Clone, PartialEq)]
pub enum LeastSquaresMatrix {
    Dense(Vec<Vec<f64>>),
    Sparse(CsrMatrix),
    Operator(LeastSquaresOperator),
}

impl LeastSquaresMatrix {
    /// Retain an analytic Jacobian as matrix-vector and transpose products.
    ///
    /// Shape is captured now and must remain fixed for the lifetime of this matrix.
    /// The operator must represent the Jacobian at this evaluation point, independently
    /// of subsequent Jacobian callbacks. No basis vectors or dense entries are probed.
    pub fn from_operator<O>(operator: O) -> Self
    where
        O: LinearOperator + Send + Sync + 'static,
    {
        Self::Operator(LeastSquaresOperator::new(operator))
    }

    /// Normalize CSR rows by sorting column indices and summing duplicates.
    /// Already sorted, duplicate-free input is retained without copying. The
    /// actual indices are checked, because callers can construct CSR storage
    /// whose metadata claims it is canonical without reordering its entries.
    pub fn from_sparse(matrix: CsrMatrix) -> SparseResult<Self> {
        let ordered = (0..matrix.shape().rows).all(|row| {
            matrix.indices()[matrix.indptr()[row]..matrix.indptr()[row + 1]]
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        });
        if ordered {
            return Ok(Self::Sparse(matrix));
        }
        let mut data = Vec::with_capacity(matrix.nnz());
        let mut indices = Vec::with_capacity(matrix.nnz());
        let mut indptr = Vec::with_capacity(matrix.shape().rows + 1);
        let mut row_entries = Vec::new();
        indptr.push(0);
        for row in 0..matrix.shape().rows {
            row_entries.clear();
            row_entries.extend(
                (matrix.indptr()[row]..matrix.indptr()[row + 1])
                    .map(|entry| (matrix.indices()[entry], matrix.data()[entry])),
            );
            // Stable ordering retains the caller's accumulation order within
            // a duplicate column, independent of the sorting implementation.
            row_entries.sort_by_key(|&(column, _)| column);
            let row_start = data.len();
            for &(column, value) in &row_entries {
                if data.len() > row_start && indices.last() == Some(&column) {
                    *data.last_mut().expect("this row already has an entry") += value;
                } else {
                    indices.push(column);
                    data.push(value);
                }
            }
            indptr.push(data.len());
        }
        CsrMatrix::from_components(matrix.shape(), data, indices, indptr, true).map(Self::Sparse)
    }

    /// Matrix dimensions. As with a nested vector, an empty dense matrix has
    /// shape `(0, 0)`; CSR storage also preserves a zero-row matrix's column count.
    #[must_use]
    pub fn shape(&self) -> Shape2D {
        match self {
            Self::Dense(rows) => Shape2D::new(rows.len(), rows.first().map_or(0, Vec::len)),
            Self::Sparse(matrix) => matrix.shape(),
            Self::Operator(operator) => operator.shape(),
        }
    }

    #[must_use]
    pub const fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse(_))
    }

    #[must_use]
    pub const fn is_operator(&self) -> bool {
        matches!(self, Self::Operator(_))
    }

    /// Access the retained, possibly loss-scaled Jacobian through [`LinearOperator`].
    #[must_use]
    pub const fn as_operator(&self) -> Option<&LeastSquaresOperator> {
        match self {
            Self::Operator(operator) => Some(operator),
            Self::Dense(_) | Self::Sparse(_) => None,
        }
    }

    #[must_use]
    pub fn as_dense(&self) -> Option<&[Vec<f64>]> {
        match self {
            Self::Dense(rows) => Some(rows),
            Self::Sparse(_) | Self::Operator(_) => None,
        }
    }

    #[must_use]
    pub const fn as_sparse(&self) -> Option<&CsrMatrix> {
        match self {
            Self::Dense(_) | Self::Operator(_) => None,
            Self::Sparse(matrix) => Some(matrix),
        }
    }

    /// Explicitly allocate a row-major dense copy, summing duplicate CSR entries or
    /// applying an operator to each basis vector. This requires `rows * columns` storage.
    /// Operator application errors and nonfinite outputs are returned to the caller.
    pub fn to_dense(&self) -> SparseResult<Vec<Vec<f64>>> {
        match self {
            Self::Dense(rows) => Ok(rows.clone()),
            Self::Sparse(matrix) => {
                let shape = matrix.shape();
                let mut rows = vec![vec![0.0; shape.cols]; shape.rows];
                for (row, values) in rows.iter_mut().enumerate() {
                    for entry in matrix.indptr()[row]..matrix.indptr()[row + 1] {
                        values[matrix.indices()[entry]] += matrix.data()[entry];
                    }
                }
                Ok(rows)
            }
            Self::Operator(operator) => operator.to_dense(),
        }
    }

    /// Whether stored entries and scaling factors are finite. An operator has no
    /// inspectable entries; its callback outputs are validated when products are taken.
    #[must_use]
    pub fn all_finite(&self) -> bool {
        match self {
            Self::Dense(rows) => rows.iter().flatten().all(|value| value.is_finite()),
            Self::Sparse(matrix) => matrix.data().iter().all(|value| value.is_finite()),
            Self::Operator(operator) => operator.scales_finite(),
        }
    }

    pub(crate) fn product(&self, vector: &[f64]) -> SparseResult<Vec<f64>> {
        let mut result = vec![0.0; self.shape().rows];
        self.product_into(vector, &mut result)?;
        Ok(result)
    }

    pub(crate) fn product_into(&self, vector: &[f64], result: &mut [f64]) -> SparseResult<()> {
        match self {
            Self::Dense(rows) => {
                for (out, row) in result.iter_mut().zip(rows) {
                    *out = row.iter().zip(vector).map(|(a, b)| a * b).sum();
                }
            }
            Self::Sparse(matrix) => {
                for (row, out) in result.iter_mut().enumerate() {
                    *out = (matrix.indptr()[row]..matrix.indptr()[row + 1])
                        .map(|entry| matrix.data()[entry] * vector[matrix.indices()[entry]])
                        .sum();
                }
            }
            Self::Operator(operator) => operator.matvec_into(vector, result)?,
        }
        Ok(())
    }

    pub(crate) fn transpose_product(&self, vector: &[f64]) -> SparseResult<Vec<f64>> {
        let mut result = vec![0.0; self.shape().cols];
        self.transpose_product_into(vector, &mut result)?;
        Ok(result)
    }

    pub(crate) fn transpose_product_into(
        &self,
        vector: &[f64],
        result: &mut [f64],
    ) -> SparseResult<()> {
        result.fill(0.0);
        match self {
            Self::Dense(rows) => {
                for (row, &value) in rows.iter().zip(vector) {
                    for (out, &entry) in result.iter_mut().zip(row) {
                        *out += entry * value;
                    }
                }
            }
            Self::Sparse(matrix) => {
                for (row, &value) in vector.iter().enumerate() {
                    for entry in matrix.indptr()[row]..matrix.indptr()[row + 1] {
                        result[matrix.indices()[entry]] += matrix.data()[entry] * value;
                    }
                }
            }
            Self::Operator(operator) => operator.rmatvec_into(vector, result)?,
        }
        Ok(())
    }

    /// Column norms in the same summation order as the dense trust-region
    /// implementation. Sparse inputs have been canonicalized at the API boundary.
    pub(crate) fn column_norms(&self) -> SparseResult<Vec<f64>> {
        let mut norms = vec![0.0; self.shape().cols];
        match self {
            Self::Dense(rows) => {
                for row in rows {
                    for (sum, &value) in norms.iter_mut().zip(row) {
                        *sum += value * value;
                    }
                }
            }
            Self::Sparse(matrix) => {
                for (&column, &value) in matrix.indices().iter().zip(matrix.data()) {
                    norms[column] += value * value;
                }
            }
            Self::Operator(_) => {
                return Err(SparseError::Unsupported {
                    feature: String::from("x_scale='jac' requires explicit Jacobian entries"),
                });
            }
        }
        for value in &mut norms {
            *value = value.sqrt();
        }
        Ok(norms)
    }

    pub(crate) fn scale_rows(&mut self, scale: &[f64]) {
        match self {
            Self::Dense(rows) => {
                for (row, &factor) in rows.iter_mut().zip(scale) {
                    for value in row {
                        *value *= factor;
                    }
                }
            }
            Self::Sparse(matrix) => {
                let mut values = matrix.data().to_vec();
                for (row, &factor) in scale.iter().enumerate() {
                    for value in &mut values[matrix.indptr()[row]..matrix.indptr()[row + 1]] {
                        *value *= factor;
                    }
                }
                *matrix = with_values(matrix, values);
            }
            Self::Operator(operator) => operator.scale_rows(scale),
        }
    }

    pub(crate) fn scaled_columns(&self, scale: &[f64]) -> Self {
        match self {
            Self::Dense(rows) => Self::Dense(
                rows.iter()
                    .map(|row| row.iter().zip(scale).map(|(a, b)| a * b).collect())
                    .collect(),
            ),
            Self::Sparse(matrix) => {
                let values = matrix
                    .data()
                    .iter()
                    .zip(matrix.indices())
                    .map(|(&value, &column)| value * scale[column])
                    .collect();
                Self::Sparse(with_values(matrix, values))
            }
            Self::Operator(operator) => Self::Operator(operator.scaled_columns(scale)),
        }
    }

    /// Select an increasing set of columns, preserving sparse storage for the
    /// free variables in Dogbox. No `rows * columns` temporary is needed.
    pub(crate) fn select_columns(&self, columns: &[usize]) -> Self {
        match self {
            Self::Dense(rows) => Self::Dense(
                rows.iter()
                    .map(|row| columns.iter().map(|&column| row[column]).collect())
                    .collect(),
            ),
            Self::Sparse(matrix) => {
                let mut mapping = vec![usize::MAX; matrix.shape().cols];
                for (new, &old) in columns.iter().enumerate() {
                    mapping[old] = new;
                }
                let mut data = Vec::with_capacity(matrix.nnz());
                let mut indices = Vec::with_capacity(matrix.nnz());
                let mut indptr = Vec::with_capacity(matrix.shape().rows + 1);
                indptr.push(0);
                for row in 0..matrix.shape().rows {
                    for entry in matrix.indptr()[row]..matrix.indptr()[row + 1] {
                        let column = mapping[matrix.indices()[entry]];
                        if column != usize::MAX {
                            data.push(matrix.data()[entry]);
                            indices.push(column);
                        }
                    }
                    indptr.push(data.len());
                }
                Self::Sparse(
                    CsrMatrix::from_components(
                        Shape2D::new(matrix.shape().rows, columns.len()),
                        data,
                        indices,
                        indptr,
                        false,
                    )
                    .expect("selecting columns preserves validated CSR structure"),
                )
            }
            Self::Operator(operator) => Self::Operator(operator.select_columns(columns)),
        }
    }
}

fn with_values(matrix: &CsrMatrix, values: Vec<f64>) -> CsrMatrix {
    CsrMatrix::from_components(
        matrix.shape(),
        values,
        matrix.indices().to_vec(),
        matrix.indptr().to_vec(),
        false,
    )
    .expect("scaling values preserves validated CSR structure")
}

#[cfg(test)]
mod tests {
    use super::LeastSquaresMatrix;
    use fsci_sparse::{CsrMatrix, DenseLinearOperator, LinearOperator, Shape2D, SparseError};

    fn matrices() -> (LeastSquaresMatrix, LeastSquaresMatrix) {
        (
            LeastSquaresMatrix::Dense(vec![
                vec![2.0, 0.0, -1.0],
                vec![0.0, 3.0, 0.0],
                vec![4.0, 0.0, 5.0],
                vec![0.0, 0.0, 0.0],
            ]),
            LeastSquaresMatrix::Sparse(
                CsrMatrix::from_components(
                    Shape2D::new(4, 3),
                    vec![2.0, -1.0, 3.0, 4.0, 5.0],
                    vec![0, 2, 1, 0, 2],
                    vec![0, 2, 3, 5, 5],
                    true,
                )
                .unwrap(),
            ),
        )
    }

    #[test]
    fn sparse_products_scaling_and_active_columns_preserve_the_matrix() {
        let (mut dense, mut sparse) = matrices();
        assert_eq!(sparse.to_dense(), dense.to_dense());
        assert_eq!(
            sparse.product(&[1.0, 2.0, 3.0]).unwrap(),
            vec![-1.0, 6.0, 19.0, 0.0]
        );
        assert_eq!(
            sparse.transpose_product(&[1.0, -2.0, 3.0, 4.0]).unwrap(),
            vec![14.0, -6.0, 14.0]
        );
        assert_eq!(sparse.column_norms(), dense.column_norms());
        for matrix in [&mut dense, &mut sparse] {
            matrix.scale_rows(&[0.5, -1.0, 2.0, 3.0]);
            *matrix = matrix.scaled_columns(&[3.0, 4.0, -2.0]);
            *matrix = matrix.select_columns(&[0, 2]);
        }
        assert_eq!(sparse.to_dense(), dense.to_dense());
        assert_eq!(sparse.shape(), Shape2D::new(4, 2));
        assert_eq!(sparse.as_sparse().unwrap().nnz(), 4);
        assert!(sparse.as_dense().is_none());
        assert!(sparse.all_finite());
    }

    #[test]
    fn sparse_empty_rows_and_empty_column_selection_remain_sparse() {
        let (_, sparse) = matrices();
        let empty = sparse.select_columns(&[]);
        assert_eq!(empty.shape(), Shape2D::new(4, 0));
        assert_eq!(empty.product(&[]).unwrap(), vec![0.0; 4]);
        assert!(empty.transpose_product(&[1.0; 4]).unwrap().is_empty());
        assert_eq!(empty.as_sparse().unwrap().nnz(), 0);
        let zero_rows = LeastSquaresMatrix::Sparse(
            CsrMatrix::from_components(Shape2D::new(0, 3), vec![], vec![], vec![0], true).unwrap(),
        );
        assert_eq!(zero_rows.shape(), Shape2D::new(0, 3));
        assert_eq!(zero_rows.transpose_product(&[]).unwrap(), vec![0.0; 3]);
        assert!(zero_rows.product(&[1.0; 3]).unwrap().is_empty());
    }

    #[test]
    fn sparse_constructor_sums_duplicates_even_when_metadata_claims_canonical() {
        let input = CsrMatrix::from_components(
            Shape2D::new(2, 3),
            vec![2.0, 3.0, -1.0, 4.0],
            vec![2, 0, 2, 1],
            vec![0, 3, 4],
            true,
        )
        .unwrap();
        let matrix = LeastSquaresMatrix::from_sparse(input).unwrap();
        assert_eq!(matrix.as_sparse().unwrap().nnz(), 3);
        assert_eq!(
            matrix.to_dense().unwrap(),
            vec![vec![3.0, 0.0, 1.0], vec![0.0, 4.0, 0.0]]
        );
        assert_eq!(matrix.column_norms().unwrap(), vec![3.0, 4.0, 1.0]);
    }

    #[test]
    fn operator_scaling_and_column_selection_preserve_products_and_adjoint() {
        let (mut dense, _) = matrices();
        let mut operator = LeastSquaresMatrix::from_operator(
            DenseLinearOperator::from_rows(&dense.to_dense().unwrap()).unwrap(),
        );
        assert_eq!(operator, operator.clone());
        for matrix in [&mut dense, &mut operator] {
            matrix.scale_rows(&[0.5, -1.0, 2.0, 3.0]);
            *matrix = matrix.scaled_columns(&[3.0, 4.0, -2.0]);
            *matrix = matrix.select_columns(&[0, 2]);
            // Repeated operations must compose with the previous column projection.
            matrix.scale_rows(&[2.0, 0.25, 0.5, 1.0]);
            *matrix = matrix.scaled_columns(&[0.5, -3.0]);
            *matrix = matrix.select_columns(&[1]);
        }
        let x = [1.25];
        let y = [2.0, 3.0, -1.0, 4.0];
        let image = operator.product(&x).unwrap();
        let adjoint = operator.transpose_product(&y).unwrap();
        assert_eq!(image, dense.product(&x).unwrap());
        assert_eq!(adjoint, dense.transpose_product(&y).unwrap());
        assert_eq!(
            image.iter().zip(y).map(|(a, b)| a * b).sum::<f64>(),
            x[0] * adjoint[0]
        );
        assert_eq!(operator.to_dense().unwrap(), dense.to_dense().unwrap());
        assert!(operator.as_dense().is_none());
        assert!(operator.as_sparse().is_none());
        assert!(matches!(
            operator.column_norms(),
            Err(SparseError::Unsupported { .. })
        ));
        let retained = operator.as_operator().unwrap();
        assert!(matches!(
            retained.matvec(&[1.0, 2.0]),
            Err(SparseError::IncompatibleShape { .. })
        ));
        assert!(matches!(
            retained.rmatvec(&[1.0]),
            Err(SparseError::IncompatibleShape { .. })
        ));

        let empty = operator.select_columns(&[]);
        assert_eq!(empty.shape(), Shape2D::new(4, 0));
        assert_eq!(empty.product(&[]).unwrap(), vec![0.0; 4]);
        assert!(empty.transpose_product(&y).unwrap().is_empty());
    }
}
