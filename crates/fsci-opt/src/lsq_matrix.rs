#![forbid(unsafe_code)]

//! Jacobian storage shared by the nonlinear least-squares algorithms.
//!
//! Sparse Jacobians remain compressed through robust-loss scaling, parameter
//! scaling, active-variable selection, and iterative trust-region subproblems.
//! Dense storage is materialized only when a caller explicitly requests it.

use fsci_sparse::{CsrMatrix, Shape2D, SparseResult};

/// A residual Jacobian, with one row per residual and one column per parameter.
///
/// `Sparse` stores a compressed sparse row matrix. Least-squares entry points
/// validate its shape and entries before use. [`Self::to_dense`] is an explicit
/// conversion; the LSMR solver does not call it.
#[derive(Debug, Clone, PartialEq)]
pub enum LeastSquaresMatrix {
    Dense(Vec<Vec<f64>>),
    Sparse(CsrMatrix),
}

impl LeastSquaresMatrix {
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
        }
    }

    #[must_use]
    pub const fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse(_))
    }

    #[must_use]
    pub fn as_dense(&self) -> Option<&[Vec<f64>]> {
        match self {
            Self::Dense(rows) => Some(rows),
            Self::Sparse(_) => None,
        }
    }

    #[must_use]
    pub const fn as_sparse(&self) -> Option<&CsrMatrix> {
        match self {
            Self::Dense(_) => None,
            Self::Sparse(matrix) => Some(matrix),
        }
    }

    /// Explicitly allocate a row-major dense copy, summing any duplicate CSR
    /// entries. This may require `rows * columns` storage for a sparse matrix.
    #[must_use]
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        match self {
            Self::Dense(rows) => rows.clone(),
            Self::Sparse(matrix) => {
                let shape = matrix.shape();
                let mut rows = vec![vec![0.0; shape.cols]; shape.rows];
                for (row, values) in rows.iter_mut().enumerate() {
                    for entry in matrix.indptr()[row]..matrix.indptr()[row + 1] {
                        values[matrix.indices()[entry]] += matrix.data()[entry];
                    }
                }
                rows
            }
        }
    }

    #[must_use]
    pub fn all_finite(&self) -> bool {
        match self {
            Self::Dense(rows) => rows.iter().flatten().all(|value| value.is_finite()),
            Self::Sparse(matrix) => matrix.data().iter().all(|value| value.is_finite()),
        }
    }

    pub(crate) fn product(&self, vector: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.shape().rows];
        self.product_into(vector, &mut result);
        result
    }

    pub(crate) fn product_into(&self, vector: &[f64], result: &mut [f64]) {
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
        }
    }

    pub(crate) fn transpose_product(&self, vector: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.shape().cols];
        self.transpose_product_into(vector, &mut result);
        result
    }

    pub(crate) fn transpose_product_into(&self, vector: &[f64], result: &mut [f64]) {
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
        }
    }

    /// Column norms in the same summation order as the dense trust-region
    /// implementation. Sparse inputs have been canonicalized at the API boundary.
    pub(crate) fn column_norms(&self) -> Vec<f64> {
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
        }
        for value in &mut norms {
            *value = value.sqrt();
        }
        norms
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
    use fsci_sparse::{CsrMatrix, Shape2D};

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
        assert_eq!(sparse.product(&[1.0, 2.0, 3.0]), vec![-1.0, 6.0, 19.0, 0.0]);
        assert_eq!(
            sparse.transpose_product(&[1.0, -2.0, 3.0, 4.0]),
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
        assert_eq!(empty.product(&[]), vec![0.0; 4]);
        assert!(empty.transpose_product(&[1.0; 4]).is_empty());
        assert_eq!(empty.as_sparse().unwrap().nnz(), 0);
        let zero_rows = LeastSquaresMatrix::Sparse(
            CsrMatrix::from_components(Shape2D::new(0, 3), vec![], vec![], vec![0], true).unwrap(),
        );
        assert_eq!(zero_rows.shape(), Shape2D::new(0, 3));
        assert_eq!(zero_rows.transpose_product(&[]), vec![0.0; 3]);
        assert!(zero_rows.product(&[1.0; 3]).is_empty());
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
            matrix.to_dense(),
            vec![vec![3.0, 0.0, 1.0], vec![0.0, 4.0, 0.0]]
        );
        assert_eq!(matrix.column_norms(), vec![3.0, 4.0, 1.0]);
    }
}
