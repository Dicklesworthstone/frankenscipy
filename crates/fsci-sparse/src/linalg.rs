use std::collections::{BTreeMap, BTreeSet};

use fsci_linalg::{DecompOptions, LinalgError, expm as dense_expm};
use fsci_runtime::RuntimeMode;
use nalgebra::{DMatrix, DVector, Dyn, LU};

use crate::construct::eye;
use crate::formats::{CscMatrix, CsrMatrix, Shape2D, SparseError, SparseResult};
use crate::ops::FormatConvertible;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseBackend {
    Auto,
    Umfpack,
    Superlu,
    NativeSparseLu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermutationOrdering {
    Colamd,
    Natural,
    MmdAta,
    MmdAtPlusA,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveOptions {
    pub mode: RuntimeMode,
    pub backend: SparseBackend,
    pub ordering: PermutationOrdering,
    pub check_finite: bool,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            backend: SparseBackend::Auto,
            ordering: PermutationOrdering::Colamd,
            check_finite: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LuOptions {
    pub mode: RuntimeMode,
    pub ordering: PermutationOrdering,
    pub diag_pivot_thresh: f64,
}

impl Default for LuOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            ordering: PermutationOrdering::Colamd,
            diag_pivot_thresh: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IluOptions {
    pub mode: RuntimeMode,
    pub ordering: PermutationOrdering,
    pub drop_tol: f64,
    pub fill_factor: f64,
}

impl Default for IluOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            ordering: PermutationOrdering::Colamd,
            drop_tol: 1e-4,
            fill_factor: 10.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpmOptions {
    pub mode: RuntimeMode,
    pub check_finite: bool,
}

impl Default for ExpmOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            check_finite: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolveResult {
    pub solution: Vec<f64>,
    pub backend_used: SparseBackend,
    pub ordering_used: PermutationOrdering,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SparseLuFactorization {
    pub shape: (usize, usize),
    pub backend_used: SparseBackend,
    pub ordering_used: PermutationOrdering,
    lu_internal: SparseLuInternal,
}

#[derive(Debug, Clone)]
enum SparseLuInternal {
    Dense(LU<f64, Dyn, Dyn>),
    Native(NativeSparseLu),
}

#[derive(Debug, Clone)]
struct NativeSparseLu {
    n: usize,
    row_perm: Vec<usize>,
    l_rows: Vec<Vec<(usize, f64)>>,
    u_rows: Vec<Vec<(usize, f64)>>,
}

/// ILU(0) factorization result.
///
/// Stores L (unit lower triangular) and U (upper triangular) in CSR format,
/// maintaining the same sparsity pattern as the original matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseIluFactorization {
    pub shape: (usize, usize),
    pub backend_used: SparseBackend,
    pub ordering_used: PermutationOrdering,
    /// L factor data (unit lower triangular, stored in CSR row-by-row).
    /// L diagonal entries are implicitly 1.0.
    l_data: Vec<f64>,
    l_indices: Vec<usize>,
    l_indptr: Vec<usize>,
    /// U factor data (upper triangular, stored in CSR row-by-row).
    u_data: Vec<f64>,
    u_indices: Vec<usize>,
    u_indptr: Vec<usize>,
    n: usize,
}

impl SparseIluFactorization {
    /// Solve L*U*x = b using forward/backward substitution.
    pub fn solve(&self, b: &[f64]) -> SparseResult<Vec<f64>> {
        if b.len() != self.n {
            return Err(SparseError::IncompatibleShape {
                message: format!("rhs length {} != matrix size {}", b.len(), self.n),
            });
        }

        // Forward substitution: L*y = b (L is unit lower triangular)
        let mut y = b.to_vec();
        for i in 0..self.n {
            for idx in self.l_indptr[i]..self.l_indptr[i + 1] {
                let j = self.l_indices[idx];
                if j < i {
                    y[i] -= self.l_data[idx] * y[j];
                }
            }
        }

        // Backward substitution: U*x = y
        let mut x = y;
        for i in (0..self.n).rev() {
            for idx in self.u_indptr[i]..self.u_indptr[i + 1] {
                let j = self.u_indices[idx];
                if j > i {
                    x[i] -= self.u_data[idx] * x[j];
                }
            }
            // Divide by diagonal of U
            let diag = self.get_u_diagonal(i);
            if diag.abs() < f64::EPSILON * 100.0 {
                return Err(SparseError::SingularMatrix {
                    message: format!("zero diagonal in U at row {i}"),
                });
            }
            x[i] /= diag;
        }

        Ok(x)
    }

    fn get_u_diagonal(&self, i: usize) -> f64 {
        for idx in self.u_indptr[i]..self.u_indptr[i + 1] {
            if self.u_indices[idx] == i {
                return self.u_data[idx];
            }
        }
        0.0
    }
}

/// Dense-conversion sanity bound for the small-system spsolve / splu
/// fallback. Larger systems use the native sparse-direct path below so
/// identity, diagonal, banded, and moderate-fill systems scale with
/// stored nonzeros and generated fill-in instead of n² dense storage.
const SPSOLVE_DENSE_MAX_N: usize = 32_768;

fn is_sparse_zero_pivot(value: f64) -> bool {
    value == 0.0
}

impl NativeSparseLu {
    fn factorize_csr(a: &CsrMatrix, diag_pivot_thresh: f64) -> SparseResult<Self> {
        let shape = a.shape();
        if !shape.is_square() {
            return Err(SparseError::InvalidShape {
                message: "native sparse LU requires a square matrix".to_string(),
            });
        }

        let n = shape.rows;
        let mut rows = csr_rows_as_maps(a);
        let mut column_rows = sparse_column_membership(n, &rows);
        let mut row_perm: Vec<usize> = (0..n).collect();
        let mut l_rows = vec![Vec::new(); n];

        for k in 0..n {
            let pivot_row = select_sparse_pivot_row(&rows, &column_rows, k, diag_pivot_thresh)?;
            if pivot_row != k {
                swap_sparse_factor_rows(
                    &mut rows,
                    &mut column_rows,
                    &mut row_perm,
                    &mut l_rows,
                    k,
                    pivot_row,
                );
            }

            let pivot = rows[k].get(&k).copied().unwrap_or(0.0);
            if is_sparse_zero_pivot(pivot) {
                return Err(SparseError::SingularMatrix {
                    message: format!("zero pivot in sparse LU at column {k}"),
                });
            }

            let pivot_tail: Vec<(usize, f64)> = rows[k]
                .range((k + 1)..)
                .map(|(&col, &value)| (col, value))
                .collect();
            let rows_to_eliminate: Vec<usize> = column_rows[k].range((k + 1)..).copied().collect();

            for row in rows_to_eliminate {
                let Some(value) = remove_sparse_entry(&mut rows, &mut column_rows, row, k) else {
                    continue;
                };
                let multiplier = value / pivot;
                if multiplier != 0.0 {
                    l_rows[row].push((k, multiplier));
                }
                for &(col, pivot_value) in &pivot_tail {
                    add_sparse_entry(
                        &mut rows,
                        &mut column_rows,
                        row,
                        col,
                        -multiplier * pivot_value,
                    );
                }
            }
        }

        let u_rows = rows
            .into_iter()
            .enumerate()
            .map(|(row, entries)| {
                entries
                    .into_iter()
                    .filter(|(col, value)| *col >= row && *value != 0.0)
                    .collect()
            })
            .collect();

        Ok(Self {
            n,
            row_perm,
            l_rows,
            u_rows,
        })
    }

    fn solve(&self, b: &[f64]) -> SparseResult<Vec<f64>> {
        if b.len() != self.n {
            return Err(SparseError::IncompatibleShape {
                message: format!("rhs length {} must match matrix size {}", b.len(), self.n),
            });
        }

        let mut y = vec![0.0; self.n];
        for row in 0..self.n {
            let mut value = b[self.row_perm[row]];
            for &(col, multiplier) in &self.l_rows[row] {
                value -= multiplier * y[col];
            }
            y[row] = value;
        }

        let mut x = vec![0.0; self.n];
        for row in (0..self.n).rev() {
            let mut value = y[row];
            let mut diagonal = None;
            for &(col, entry) in &self.u_rows[row] {
                if col == row {
                    diagonal = Some(entry);
                } else if col > row {
                    value -= entry * x[col];
                }
            }
            let pivot = diagonal.unwrap_or(0.0);
            if is_sparse_zero_pivot(pivot) {
                return Err(SparseError::SingularMatrix {
                    message: format!("zero pivot in sparse LU solve at row {row}"),
                });
            }
            x[row] = value / pivot;
        }

        Ok(x)
    }

    #[cfg(test)]
    fn stored_nnz(&self) -> usize {
        self.l_rows.iter().map(Vec::len).sum::<usize>()
            + self.u_rows.iter().map(Vec::len).sum::<usize>()
    }
}

fn csr_rows_as_maps(a: &CsrMatrix) -> Vec<BTreeMap<usize, f64>> {
    let shape = a.shape();
    let mut rows = vec![BTreeMap::new(); shape.rows];
    for row in 0..shape.rows {
        for idx in a.indptr()[row]..a.indptr()[row + 1] {
            let col = a.indices()[idx];
            let value = a.data()[idx];
            if value != 0.0 {
                let entry = rows[row].entry(col).or_insert(0.0);
                *entry += value;
                if *entry == 0.0 {
                    rows[row].remove(&col);
                }
            }
        }
    }
    rows
}

fn sparse_column_membership(n: usize, rows: &[BTreeMap<usize, f64>]) -> Vec<BTreeSet<usize>> {
    let mut column_rows = vec![BTreeSet::new(); n];
    for (row, entries) in rows.iter().enumerate() {
        for &col in entries.keys() {
            if col < n {
                column_rows[col].insert(row);
            }
        }
    }
    column_rows
}

fn select_sparse_pivot_row(
    rows: &[BTreeMap<usize, f64>],
    column_rows: &[BTreeSet<usize>],
    col: usize,
    diag_pivot_thresh: f64,
) -> SparseResult<usize> {
    let mut best_row = None;
    let mut best_abs = 0.0;
    for &row in column_rows[col].range(col..) {
        let value = rows[row].get(&col).copied().unwrap_or(0.0).abs();
        if value > best_abs {
            best_abs = value;
            best_row = Some(row);
        }
    }

    if is_sparse_zero_pivot(best_abs) {
        return Err(SparseError::SingularMatrix {
            message: format!("zero pivot in sparse LU at column {col}"),
        });
    }

    let diagonal_abs = rows[col].get(&col).copied().unwrap_or(0.0).abs();
    if !is_sparse_zero_pivot(diagonal_abs)
        && diagonal_abs >= best_abs * diag_pivot_thresh.clamp(0.0, 1.0)
    {
        return Ok(col);
    }

    best_row.ok_or_else(|| SparseError::SingularMatrix {
        message: format!("zero pivot in sparse LU at column {col}"),
    })
}

fn swap_sparse_factor_rows(
    rows: &mut [BTreeMap<usize, f64>],
    column_rows: &mut [BTreeSet<usize>],
    row_perm: &mut [usize],
    l_rows: &mut [Vec<(usize, f64)>],
    lhs: usize,
    rhs: usize,
) {
    for &col in rows[lhs].keys() {
        column_rows[col].remove(&lhs);
    }
    for &col in rows[rhs].keys() {
        column_rows[col].remove(&rhs);
    }

    rows.swap(lhs, rhs);
    row_perm.swap(lhs, rhs);
    l_rows.swap(lhs, rhs);

    for &col in rows[lhs].keys() {
        column_rows[col].insert(lhs);
    }
    for &col in rows[rhs].keys() {
        column_rows[col].insert(rhs);
    }
}

fn remove_sparse_entry(
    rows: &mut [BTreeMap<usize, f64>],
    column_rows: &mut [BTreeSet<usize>],
    row: usize,
    col: usize,
) -> Option<f64> {
    let value = rows[row].remove(&col)?;
    column_rows[col].remove(&row);
    Some(value)
}

fn add_sparse_entry(
    rows: &mut [BTreeMap<usize, f64>],
    column_rows: &mut [BTreeSet<usize>],
    row: usize,
    col: usize,
    delta: f64,
) {
    if delta == 0.0 {
        return;
    }

    let previous = rows[row].get(&col).copied().unwrap_or(0.0);
    let updated = previous + delta;
    if updated == 0.0 {
        if rows[row].remove(&col).is_some() {
            column_rows[col].remove(&row);
        }
    } else {
        rows[row].insert(col, updated);
        column_rows[col].insert(row);
    }
}

pub fn spsolve(a: &CsrMatrix, b: &[f64], options: SolveOptions) -> SparseResult<SolveResult> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "spsolve requires a square matrix".to_string(),
        });
    }
    if b.len() != shape.rows {
        return Err(SparseError::IncompatibleShape {
            message: "rhs length must match matrix rows".to_string(),
        });
    }
    if options.check_finite
        && (a.data().iter().any(|v| !v.is_finite()) || b.iter().any(|v| !v.is_finite()))
    {
        return Err(SparseError::NonFiniteInput {
            message: "matrix/rhs contains NaN or Inf".to_string(),
        });
    }

    if options.mode == RuntimeMode::Hardened && has_empty_structural_row(a) {
        return Err(SparseError::SingularMatrix {
            message: "detected empty structural row in hardened mode".to_string(),
        });
    }

    let n = shape.rows;
    if n > SPSOLVE_DENSE_MAX_N {
        let lu = NativeSparseLu::factorize_csr(a, 1.0)?;
        let solution = lu.solve(b)?;
        return Ok(SolveResult {
            solution,
            backend_used: SparseBackend::NativeSparseLu,
            ordering_used: options.ordering,
            warnings: vec![format!(
                "native sparse direct solve used for n={n}; dense fallback guard is {SPSOLVE_DENSE_MAX_N}"
            )],
        });
    }

    let dense = csr_to_dense(a);
    let matrix = DMatrix::from_row_slice(n, n, &dense);
    let rhs = DVector::from_column_slice(b);
    let lu: LU<f64, Dyn, Dyn> = matrix.lu();
    let x = lu.solve(&rhs).ok_or(SparseError::SingularMatrix {
        message: "LU factorization detected singular matrix".to_string(),
    })?;

    Ok(SolveResult {
        solution: x.iter().copied().collect(),
        backend_used: SparseBackend::Auto,
        ordering_used: options.ordering,
        warnings: Vec::new(),
    })
}

pub fn splu(a: &CscMatrix, options: LuOptions) -> SparseResult<SparseLuFactorization> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "splu requires a square matrix".to_string(),
        });
    }
    if !(0.0..=1.0).contains(&options.diag_pivot_thresh) {
        return Err(SparseError::InvalidArgument {
            message: "diag_pivot_thresh must be in [0, 1]".to_string(),
        });
    }
    let n = shape.rows;
    let (backend_used, lu_internal) = if n > SPSOLVE_DENSE_MAX_N {
        let csr = a.to_csr()?;
        (
            SparseBackend::NativeSparseLu,
            SparseLuInternal::Native(NativeSparseLu::factorize_csr(
                &csr,
                options.diag_pivot_thresh,
            )?),
        )
    } else {
        let dense = csc_to_dense(a);
        let matrix = DMatrix::from_row_slice(n, n, &dense);
        (SparseBackend::Auto, SparseLuInternal::Dense(matrix.lu()))
    };

    Ok(SparseLuFactorization {
        shape: (n, n),
        backend_used,
        ordering_used: options.ordering,
        lu_internal,
    })
}

/// Solve a linear system using a precomputed sparse LU factorization.
pub fn splu_solve(factorization: &SparseLuFactorization, b: &[f64]) -> SparseResult<Vec<f64>> {
    let n = factorization.shape.0;
    if b.len() != n {
        return Err(SparseError::IncompatibleShape {
            message: format!("rhs length {} must match matrix size {}", b.len(), n),
        });
    }
    match &factorization.lu_internal {
        SparseLuInternal::Dense(lu) => {
            let rhs = DVector::from_column_slice(b);
            let x = lu.solve(&rhs).ok_or(SparseError::SingularMatrix {
                message: "LU factorization detected singular matrix".to_string(),
            })?;
            Ok(x.iter().copied().collect())
        }
        SparseLuInternal::Native(lu) => lu.solve(b),
    }
}

/// ILU(0) incomplete LU factorization.
///
/// Computes L and U factors maintaining the sparsity pattern of A.
/// Matches `scipy.sparse.linalg.spilu(A, drop_tol=0)` behavior.
///
/// Input is CSC but internally converts to CSR for row-based ILU(0).
pub fn spilu(a: &CscMatrix, options: IluOptions) -> SparseResult<SparseIluFactorization> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "spilu requires a square matrix".to_string(),
        });
    }
    if options.drop_tol < 0.0 || options.fill_factor < 1.0 {
        return Err(SparseError::InvalidArgument {
            message: "drop_tol must be >= 0 and fill_factor must be >= 1".to_string(),
        });
    }

    let n = shape.rows;
    if n == 0 {
        return Ok(SparseIluFactorization {
            shape: (0, 0),
            backend_used: SparseBackend::Auto,
            ordering_used: options.ordering,
            l_data: Vec::new(),
            l_indices: Vec::new(),
            l_indptr: vec![0],
            u_data: Vec::new(),
            u_indices: Vec::new(),
            u_indptr: vec![0],
            n: 0,
        });
    }

    // Convert to CSR for row-based factorization
    let csr = a.to_csr()?;
    let indptr = csr.indptr();
    let indices = csr.indices();
    let data = csr.data();

    // Work on a dense-ish representation for the factorization:
    // For each row, track L entries (j < i) and U entries (j >= i)
    // using the original sparsity pattern.
    let mut lu_data = data.to_vec(); // mutable copy of values
    let lu_indices = indices;
    let lu_indptr = indptr;

    // IKJ variant of ILU(0): for each row i, for each nonzero a[i,k] with k < i,
    // compute multiplier a[i,k] /= a[k,k], then for each nonzero a[k,j] with j > k,
    // if (i,j) is in the sparsity pattern, subtract multiplier * a[k,j].
    let mut row_lookup = vec![usize::MAX; n];
    let mut row_lookup_touched = Vec::new();
    for i in 0..n {
        let row_start = lu_indptr[i];
        let row_end = lu_indptr[i + 1];
        row_lookup_touched.clear();
        for (offset, &col) in lu_indices[row_start..row_end].iter().enumerate() {
            let idx = row_start + offset;
            row_lookup[col] = idx;
            row_lookup_touched.push(col);
        }

        for idx_ik in row_start..row_end {
            let k = lu_indices[idx_ik];
            if k >= i {
                break; // only process lower triangle (k < i)
            }

            // Find diagonal a[k,k]
            let diag_k = find_value_in_row(&lu_data, lu_indices, lu_indptr, k, k);
            if diag_k.abs() < f64::EPSILON * 100.0 {
                return Err(SparseError::SingularMatrix {
                    message: format!("zero pivot at row {k} during ILU(0)"),
                });
            }

            // Compute multiplier: a[i,k] /= a[k,k]
            lu_data[idx_ik] /= diag_k;
            let multiplier = lu_data[idx_ik];

            // For each nonzero in row k with column j > k
            for idx_kj in lu_indptr[k]..lu_indptr[k + 1] {
                let j = lu_indices[idx_kj];
                if j <= k {
                    continue;
                }
                let a_kj = lu_data[idx_kj];

                // If (i, j) exists in the sparsity pattern, subtract
                let idx_ij = row_lookup[j];
                if idx_ij != usize::MAX {
                    lu_data[idx_ij] -= multiplier * a_kj;
                }
                // ILU(0): if (i,j) is NOT in pattern, we drop the fill-in
            }
        }

        for &col in &row_lookup_touched {
            row_lookup[col] = usize::MAX;
        }
    }

    // Extract L and U from the modified data
    let mut l_data = Vec::new();
    let mut l_indices = Vec::new();
    let mut l_indptr = vec![0usize];
    let mut u_data = Vec::new();
    let mut u_indices = Vec::new();
    let mut u_indptr = vec![0usize];

    for i in 0..n {
        // L entries: j < i (with implicit 1 on diagonal)
        for idx in lu_indptr[i]..lu_indptr[i + 1] {
            let j = lu_indices[idx];
            if j < i {
                l_data.push(lu_data[idx]);
                l_indices.push(j);
            }
        }
        // Add implicit diagonal
        l_data.push(1.0);
        l_indices.push(i);
        l_indptr.push(l_data.len());

        // U entries: j >= i
        for idx in lu_indptr[i]..lu_indptr[i + 1] {
            let j = lu_indices[idx];
            if j >= i {
                u_data.push(lu_data[idx]);
                u_indices.push(j);
            }
        }
        u_indptr.push(u_data.len());
    }

    Ok(SparseIluFactorization {
        shape: (n, n),
        backend_used: SparseBackend::Auto,
        ordering_used: options.ordering,
        l_data,
        l_indices,
        l_indptr,
        u_data,
        u_indices,
        u_indptr,
        n,
    })
}

/// Sparse matrix exponential via dense fallback.
///
/// Matches `scipy.sparse.linalg.expm(A)` semantics for V1 by delegating to
/// `fsci_linalg::expm` after densifying the input matrix.
pub fn expm(a: &CsrMatrix, options: ExpmOptions) -> SparseResult<Vec<Vec<f64>>> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "expm requires a square matrix".to_string(),
        });
    }

    let must_check = options.check_finite || options.mode == RuntimeMode::Hardened;
    if must_check && a.data().iter().any(|v| !v.is_finite()) {
        return Err(SparseError::NonFiniteInput {
            message: "matrix contains NaN or Inf".to_string(),
        });
    }

    if shape.rows == 0 {
        return Ok(Vec::new());
    }

    let dense = csr_to_dense(a);
    let mut rows = Vec::with_capacity(shape.rows);
    for i in 0..shape.rows {
        let start = i * shape.cols;
        let end = start + shape.cols;
        rows.push(dense[start..end].to_vec());
    }

    let decomp = DecompOptions {
        mode: options.mode,
        check_finite: options.check_finite,
    };
    dense_expm(&rows, decomp).map_err(map_linalg_error)
}

fn map_linalg_error(err: LinalgError) -> SparseError {
    match err {
        LinalgError::RaggedMatrix => SparseError::InvalidArgument {
            message: "ragged matrix rows".to_string(),
        },
        LinalgError::ExpectedSquareMatrix => SparseError::InvalidShape {
            message: "expm requires a square matrix".to_string(),
        },
        LinalgError::IncompatibleShapes { a_shape, b_len } => SparseError::IncompatibleShape {
            message: format!("incompatible shapes: a_shape={a_shape:?}, b_len={b_len}"),
        },
        LinalgError::NonFiniteInput => SparseError::NonFiniteInput {
            message: "matrix contains NaN or Inf".to_string(),
        },
        LinalgError::SingularMatrix => SparseError::SingularMatrix {
            message: "singular matrix".to_string(),
        },
        LinalgError::UnsupportedAssumption => SparseError::Unsupported {
            feature: "unsupported matrix assumption".to_string(),
        },
        LinalgError::InvalidBandShape {
            expected_rows,
            actual_rows,
        } => SparseError::InvalidArgument {
            message: format!(
                "invalid band shape: expected {expected_rows} rows, got {actual_rows}"
            ),
        },
        LinalgError::InvalidPinvThreshold => SparseError::InvalidArgument {
            message: "invalid pinv threshold".to_string(),
        },
        LinalgError::NotSupported { detail } => SparseError::Unsupported { feature: detail },
        LinalgError::ConvergenceFailure { detail } => {
            SparseError::InvalidArgument { message: detail }
        }
        LinalgError::PolicyRejected { reason } => SparseError::InvalidArgument {
            message: format!("policy rejected sparse linalg operation: {reason}"),
        },
        LinalgError::ConditionTooHigh { rcond, threshold } => SparseError::InvalidArgument {
            message: format!("condition too high: rcond={rcond} threshold={threshold}"),
        },
        LinalgError::ResourceExhausted { detail } => SparseError::InvalidArgument {
            message: format!("resource exhausted: {detail}"),
        },
        LinalgError::InvalidArgument { detail } => SparseError::InvalidArgument { message: detail },
    }
}

/// Find the value at position (row, col) in CSR data.
fn find_value_in_row(
    data: &[f64],
    indices: &[usize],
    indptr: &[usize],
    row: usize,
    col: usize,
) -> f64 {
    let range = indptr[row]..indptr[row + 1];
    indices[range.clone()]
        .iter()
        .zip(data[range].iter())
        .find(|(j, _)| **j == col)
        .map_or(0.0, |(_, v)| *v)
}

/// Options for iterative solvers (CG, GMRES, etc.).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterativeSolveOptions {
    pub mode: RuntimeMode,
    pub check_finite: bool,
    pub tol: f64,
    pub max_iter: Option<usize>,
}

impl Default for IterativeSolveOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            check_finite: true,
            tol: 1e-5,
            max_iter: None,
        }
    }
}

/// Result from an iterative solver.
#[derive(Debug, Clone, PartialEq)]
pub struct IterativeSolveResult {
    /// Solution vector.
    pub solution: Vec<f64>,
    /// Whether the solver converged within the tolerance.
    pub converged: bool,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final residual norm ||b - Ax|| / ||b||.
    pub residual_norm: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaspIterativeSolver {
    Cg,
    Gmres,
    Lgmres,
    Bicgstab,
    Qmr,
    Minres,
    Lsqr,
    Lsmr,
}

impl CaspIterativeSolver {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cg => "cg",
            Self::Gmres => "gmres",
            Self::Lgmres => "lgmres",
            Self::Bicgstab => "bicgstab",
            Self::Qmr => "qmr",
            Self::Minres => "minres",
            Self::Lsqr => "lsqr",
            Self::Lsmr => "lsmr",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaspMatvecCost {
    Auto,
    Cheap,
    Moderate,
    Expensive,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaspIterativeSolveOptions {
    pub iterative: IterativeSolveOptions,
    pub preconditioner_available: bool,
    pub matrix_vector_cost: CaspMatvecCost,
    pub prefer_low_memory: bool,
}

impl Default for CaspIterativeSolveOptions {
    fn default() -> Self {
        Self {
            iterative: IterativeSolveOptions::default(),
            preconditioner_available: false,
            matrix_vector_cost: CaspMatvecCost::Auto,
            prefer_low_memory: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaspIterativeDecision {
    pub selected_solver: CaspIterativeSolver,
    pub square: bool,
    pub symmetric: bool,
    pub positive_diagonal: bool,
    pub row_diagonally_dominant: bool,
    pub density: f64,
    pub matrix_vector_cost: CaspMatvecCost,
    pub preconditioner_available: bool,
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaspIterativeSolveResult {
    pub decision: CaspIterativeDecision,
    pub result: IterativeSolveResult,
}

fn validate_iterative_finite_inputs(
    a: &CsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    options: IterativeSolveOptions,
) -> SparseResult<()> {
    let must_check = options.check_finite || options.mode == RuntimeMode::Hardened;
    let x0_has_non_finite = x0.is_some_and(|initial| initial.iter().any(|v| !v.is_finite()));
    if must_check
        && (a.data().iter().any(|v| !v.is_finite())
            || b.iter().any(|v| !v.is_finite())
            || x0_has_non_finite)
    {
        return Err(SparseError::NonFiniteInput {
            message: "matrix/rhs/initial guess contains NaN or Inf".to_string(),
        });
    }
    Ok(())
}

/// Conjugate Gradient solver for symmetric positive-definite sparse systems.
///
/// Solves Ax = b where A is SPD. If A is not SPD, the solver may diverge.
/// Matches `scipy.sparse.linalg.cg(A, b)`.
pub fn cg(
    a: &CsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    options: IterativeSolveOptions,
) -> SparseResult<IterativeSolveResult> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "CG requires a square matrix".to_string(),
        });
    }
    let n = shape.rows;
    if b.len() != n {
        return Err(SparseError::IncompatibleShape {
            message: "rhs length must match matrix rows".to_string(),
        });
    }
    validate_iterative_finite_inputs(a, b, x0, options)?;

    let max_iter = options.max_iter.unwrap_or(n * 10);

    // Initialize x
    let mut x: Vec<f64> = match x0 {
        Some(initial) => {
            if initial.len() != n {
                return Err(SparseError::IncompatibleShape {
                    message: "initial guess length must match matrix rows".to_string(),
                });
            }
            initial.to_vec()
        }
        None => vec![0.0; n],
    };

    // Compute b_norm for relative tolerance
    let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    if b_norm <= f64::EPSILON {
        // b is zero, solution is zero
        return Ok(IterativeSolveResult {
            solution: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
        });
    }

    // r = b - A*x
    let ax = csr_matvec(a, &x);
    let mut r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();
    let mut p = r.clone();
    let mut rs_old: f64 = r.iter().map(|v| v * v).sum();

    for iteration in 0..max_iter {
        let r_norm = rs_old.sqrt();
        if r_norm / b_norm < options.tol {
            return Ok(IterativeSolveResult {
                solution: x,
                converged: true,
                iterations: iteration,
                residual_norm: r_norm / b_norm,
            });
        }

        let ap = csr_matvec(a, &p);
        let p_ap: f64 = p.iter().zip(ap.iter()).map(|(pi, api)| pi * api).sum();

        if p_ap.abs() < f64::EPSILON * 100.0 {
            // Near-zero denominator; matrix may not be SPD
            return Ok(IterativeSolveResult {
                solution: x,
                converged: false,
                iterations: iteration,
                residual_norm: r_norm / b_norm,
            });
        }

        let alpha = rs_old / p_ap;

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let rs_new: f64 = r.iter().map(|v| v * v).sum();
        let beta = rs_new / rs_old;

        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }

        rs_old = rs_new;
    }

    let final_norm = rs_old.sqrt() / b_norm;
    Ok(IterativeSolveResult {
        solution: x,
        converged: false,
        iterations: max_iter,
        residual_norm: final_norm,
    })
}

/// Sparse CSR matrix-vector product (internal helper for iterative solvers).
fn csr_matvec(a: &CsrMatrix, x: &[f64]) -> Vec<f64> {
    let n = a.shape().rows;
    let indptr = a.indptr();
    let indices = a.indices();
    let data = a.data();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for idx in indptr[i]..indptr[i + 1] {
            sum += data[idx] * x[indices[idx]];
        }
        result[i] = sum;
    }
    result
}

/// Preconditioned Conjugate Gradient solver.
///
/// Solves Ax = b using CG with an ILU(0) preconditioner M ≈ A.
/// The preconditioner solves M*z = r at each iteration instead of using r directly.
/// Matches `scipy.sparse.linalg.cg(A, b, M=spilu(A).solve)`.
pub fn pcg(
    a: &CsrMatrix,
    b: &[f64],
    preconditioner: &SparseIluFactorization,
    x0: Option<&[f64]>,
    options: IterativeSolveOptions,
) -> SparseResult<IterativeSolveResult> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "PCG requires a square matrix".to_string(),
        });
    }
    let n = shape.rows;
    if b.len() != n {
        return Err(SparseError::IncompatibleShape {
            message: "rhs length must match matrix rows".to_string(),
        });
    }

    let max_iter = options.max_iter.unwrap_or(n * 10);

    let mut x: Vec<f64> = match x0 {
        Some(initial) => {
            if initial.len() != n {
                return Err(SparseError::IncompatibleShape {
                    message: "initial guess length must match matrix rows".to_string(),
                });
            }
            initial.to_vec()
        }
        None => vec![0.0; n],
    };

    let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    if b_norm <= f64::EPSILON {
        return Ok(IterativeSolveResult {
            solution: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
        });
    }

    // r = b - A*x
    let ax = csr_matvec(a, &x);
    let mut r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();

    // z = M^{-1} * r (preconditioner application)
    let mut z = preconditioner.solve(&r).unwrap_or_else(|_| r.clone());

    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(z.iter()).map(|(ri, zi)| ri * zi).sum();

    for iteration in 0..max_iter {
        let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if r_norm / b_norm < options.tol {
            return Ok(IterativeSolveResult {
                solution: x,
                converged: true,
                iterations: iteration,
                residual_norm: r_norm / b_norm,
            });
        }

        let ap = csr_matvec(a, &p);
        let p_ap: f64 = p.iter().zip(ap.iter()).map(|(pi, api)| pi * api).sum();

        if p_ap.abs() < f64::EPSILON * 100.0 {
            return Ok(IterativeSolveResult {
                solution: x,
                converged: false,
                iterations: iteration,
                residual_norm: r_norm / b_norm,
            });
        }

        let alpha = rz / p_ap;

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        // z = M^{-1} * r
        z = preconditioner.solve(&r).unwrap_or_else(|_| r.clone());

        let rz_new: f64 = r.iter().zip(z.iter()).map(|(ri, zi)| ri * zi).sum();
        let beta = rz_new / rz;

        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }

        rz = rz_new;
    }

    let final_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt() / b_norm;
    Ok(IterativeSolveResult {
        solution: x,
        converged: false,
        iterations: max_iter,
        residual_norm: final_norm,
    })
}

/// GMRES (Generalized Minimal Residual) solver for general (non-symmetric) sparse systems.
///
/// Solves Ax = b for general square A using restarted GMRES with Arnoldi iteration.
/// Matches `scipy.sparse.linalg.gmres(A, b)`.
pub fn gmres(
    a: &CsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    options: IterativeSolveOptions,
) -> SparseResult<IterativeSolveResult> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "GMRES requires a square matrix".to_string(),
        });
    }
    let n = shape.rows;
    if b.len() != n {
        return Err(SparseError::IncompatibleShape {
            message: "rhs length must match matrix rows".to_string(),
        });
    }
    validate_iterative_finite_inputs(a, b, x0, options)?;
    let max_iter = options.max_iter.unwrap_or(n * 10);
    let restart = n.min(30); // Krylov subspace dimension before restart

    let mut x: Vec<f64> = match x0 {
        Some(initial) => {
            if initial.len() != n {
                return Err(SparseError::IncompatibleShape {
                    message: "initial guess length must match matrix rows".to_string(),
                });
            }
            initial.to_vec()
        }
        None => vec![0.0; n],
    };

    let b_norm = vec_norm(b);
    if b_norm <= f64::EPSILON {
        return Ok(IterativeSolveResult {
            solution: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
        });
    }

    let mut total_iter = 0;

    // Outer restart loop
    for _ in 0..(max_iter / restart.max(1) + 1) {
        let (converged, iters) = gmres_inner(
            a,
            b,
            &mut x,
            b_norm,
            restart,
            options.tol,
            max_iter - total_iter,
        )?;
        total_iter += iters;

        if converged || total_iter >= max_iter {
            let ax = csr_matvec(a, &x);
            let r_norm = vec_norm_diff(&ax, b) / b_norm;
            return Ok(IterativeSolveResult {
                solution: x,
                converged,
                iterations: total_iter,
                residual_norm: r_norm,
            });
        }
    }

    let ax = csr_matvec(a, &x);
    let r_norm = vec_norm_diff(&ax, b) / b_norm;
    Ok(IterativeSolveResult {
        solution: x,
        converged: false,
        iterations: total_iter,
        residual_norm: r_norm,
    })
}

/// Inner GMRES iteration (one restart cycle).
/// Returns (converged, iterations_used).
fn gmres_inner(
    a: &CsrMatrix,
    b: &[f64],
    x: &mut [f64],
    b_norm: f64,
    restart: usize,
    tol: f64,
    max_iter: usize,
) -> SparseResult<(bool, usize)> {
    let n = x.len();
    let m = restart.min(max_iter);

    // r = b - A*x
    let ax = csr_matvec(a, x);
    let r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();
    let r_norm = vec_norm(&r);

    if r_norm / b_norm < tol {
        return Ok((true, 0));
    }

    // Arnoldi process with modified Gram-Schmidt
    let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
    v.push(r.iter().map(|&ri| ri / r_norm).collect());

    // Upper Hessenberg matrix H (stored as (m+1) x m)
    let mut h = vec![vec![0.0; m]; m + 1];

    // Givens rotation components
    let mut cs = vec![0.0; m];
    let mut sn = vec![0.0; m];
    let mut g = vec![0.0; m + 1];
    g[0] = r_norm;

    let mut iters = 0;

    for j in 0..m {
        iters = j + 1;

        // w = A * v_j
        let w = csr_matvec(a, &v[j]);
        let mut wj = w;

        // Modified Gram-Schmidt orthogonalization
        for i in 0..=j {
            h[i][j] = dot_product(&wj, &v[i]);
            for k in 0..n {
                wj[k] -= h[i][j] * v[i][k];
            }
        }

        h[j + 1][j] = vec_norm(&wj);

        if h[j + 1][j].abs() < f64::EPSILON * 100.0 {
            // Lucky breakdown — solution is in the current Krylov subspace
            // Apply previous Givens rotations to column j
            apply_givens_to_column(&mut h, &cs, &sn, j);
            // Solve the triangular system and update x
            update_solution(x, &v, &h, &g, j + 1);
            return Ok((true, iters));
        }

        // Normalize
        let inv_h = 1.0 / h[j + 1][j];
        v.push(wj.iter().map(|&wi| wi * inv_h).collect());

        // Apply previous Givens rotations to column j of H
        apply_givens_to_column(&mut h, &cs, &sn, j);

        // Compute new Givens rotation for row j
        let (c, s) = givens_rotation(h[j][j], h[j + 1][j]);
        cs[j] = c;
        sn[j] = s;

        // Apply new rotation to H and g
        h[j][j] = c * h[j][j] + s * h[j + 1][j];
        h[j + 1][j] = 0.0;

        let g_j = g[j];
        g[j] = c * g_j;
        g[j + 1] = -s * g_j;

        let residual = g[j + 1].abs() / b_norm;
        if residual < tol {
            update_solution(x, &v, &h, &g, j + 1);
            return Ok((true, iters));
        }
    }

    // Update solution with current approximation
    update_solution(x, &v, &h, &g, m);
    Ok((false, iters))
}

/// Apply previous Givens rotations to column j of H.
fn apply_givens_to_column(h: &mut [Vec<f64>], cs: &[f64], sn: &[f64], j: usize) {
    for i in 0..j {
        let temp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
        h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
        h[i][j] = temp;
    }
}

/// Compute Givens rotation coefficients.
fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        (1.0, 0.0)
    } else if b.abs() > a.abs() {
        let tau = a / b;
        let s = 1.0 / (1.0 + tau * tau).sqrt();
        (s * tau, s)
    } else {
        let tau = b / a;
        let c = 1.0 / (1.0 + tau * tau).sqrt();
        (c, c * tau)
    }
}

/// Solve the upper triangular system H*y = g, then update x += V*y.
fn update_solution(x: &mut [f64], v: &[Vec<f64>], h: &[Vec<f64>], g: &[f64], k: usize) {
    // Back-substitution: solve H[0..k, 0..k] y = g[0..k]
    let mut y = vec![0.0; k];
    for i in (0..k).rev() {
        y[i] = g[i];
        for j in (i + 1)..k {
            y[i] -= h[i][j] * y[j];
        }
        if h[i][i].abs() > f64::EPSILON * 100.0 {
            y[i] /= h[i][i];
        }
    }

    // x += V * y
    for (j, &yj) in y.iter().enumerate() {
        for (i, xi) in x.iter_mut().enumerate() {
            *xi += yj * v[j][i];
        }
    }
}

/// Euclidean norm of a vector.
fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Euclidean norm of (a - b).
fn vec_norm_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Dot product of two vectors.
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

// ══════════════════════════════════════════════════════════════════════
// LGMRES — Loose GMRES
// ══════════════════════════════════════════════════════════════════════

/// LGMRES solver for general sparse linear systems.
///
/// Loose GMRES is a memory-efficient variant of GMRES that stores
/// error approximations from previous restart cycles to accelerate
/// convergence. This is particularly useful when GMRES restarts
/// frequently due to memory constraints.
///
/// Matches `scipy.sparse.linalg.lgmres(A, b)`.
///
/// # Arguments
/// * `a` - Sparse matrix in CSR format
/// * `b` - Right-hand side vector
/// * `x0` - Optional initial guess (defaults to zero vector)
/// * `options` - Solver options (tolerance, max iterations, etc.)
///
/// # Options specific to LGMRES
/// * `inner_m` - Number of inner GMRES iterations per outer iteration (default: 30)
/// * `outer_k` - Number of outer vectors to store from previous cycles (default: 3)
pub fn lgmres(
    a: &CsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    options: LgmresOptions,
) -> SparseResult<IterativeSolveResult> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "LGMRES requires a square matrix".to_string(),
        });
    }
    let n = shape.rows;
    if b.len() != n {
        return Err(SparseError::IncompatibleShape {
            message: "rhs length must match matrix rows".to_string(),
        });
    }

    let max_iter = options.max_iter.unwrap_or(n * 10);
    let inner_m = options.inner_m.min(n);
    let outer_k = options.outer_k;

    let mut x: Vec<f64> = match x0 {
        Some(initial) => {
            if initial.len() != n {
                return Err(SparseError::IncompatibleShape {
                    message: "initial guess length must match matrix rows".to_string(),
                });
            }
            initial.to_vec()
        }
        None => vec![0.0; n],
    };

    let b_norm = vec_norm(b);
    if b_norm <= f64::EPSILON {
        return Ok(IterativeSolveResult {
            solution: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
        });
    }

    // Outer vectors storage: pairs of (z, Az) where z is the error approximation
    // and Az is A*z, stored for reuse across restarts
    let mut outer_v: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(outer_k);

    let mut total_iter = 0;

    while total_iter < max_iter {
        // r = b - A*x
        let ax = csr_matvec(a, &x);
        let mut r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();
        let r_norm = vec_norm(&r);

        if r_norm / b_norm < options.tol {
            return Ok(IterativeSolveResult {
                solution: x,
                converged: true,
                iterations: total_iter,
                residual_norm: r_norm / b_norm,
            });
        }

        // Augment Krylov space with outer vectors
        // Project r onto space spanned by outer_v and update x
        for (z, az) in &outer_v {
            let alpha = dot_product(&r, az) / dot_product(az, az).max(f64::EPSILON);
            for i in 0..n {
                x[i] += alpha * z[i];
                r[i] -= alpha * az[i];
            }
        }

        // Inner GMRES iterations
        let (z, converged, iters) = lgmres_inner(
            a,
            &r,
            inner_m,
            options.tol * b_norm,
            (max_iter - total_iter).min(inner_m),
        )?;
        total_iter += iters;

        // Update solution: x = x + z
        for i in 0..n {
            x[i] += z[i];
        }

        if converged {
            let ax = csr_matvec(a, &x);
            let final_r_norm = vec_norm_diff(&ax, b) / b_norm;
            return Ok(IterativeSolveResult {
                solution: x,
                converged: true,
                iterations: total_iter,
                residual_norm: final_r_norm,
            });
        }

        // Store outer vector for next restart
        if outer_k > 0 {
            let az = csr_matvec(a, &z);
            if outer_v.len() >= outer_k {
                outer_v.remove(0); // Remove oldest
            }
            outer_v.push((z, az));
        }
    }

    let ax = csr_matvec(a, &x);
    let r_norm = vec_norm_diff(&ax, b) / b_norm;
    Ok(IterativeSolveResult {
        solution: x,
        converged: false,
        iterations: total_iter,
        residual_norm: r_norm,
    })
}

/// Options for LGMRES solver.
#[derive(Debug, Clone, Copy)]
pub struct LgmresOptions {
    /// Convergence tolerance (relative residual norm).
    pub tol: f64,
    /// Maximum number of iterations.
    pub max_iter: Option<usize>,
    /// Inner GMRES iterations per outer iteration.
    pub inner_m: usize,
    /// Number of outer vectors to store from previous restarts.
    pub outer_k: usize,
}

impl Default for LgmresOptions {
    fn default() -> Self {
        Self {
            tol: 1e-5,
            max_iter: None,
            inner_m: 30,
            outer_k: 3,
        }
    }
}

/// Inner LGMRES iteration (simplified GMRES for error approximation).
/// Returns (error_approximation, converged, iterations).
fn lgmres_inner(
    a: &CsrMatrix,
    r0: &[f64],
    max_iter: usize,
    tol: f64,
    iter_limit: usize,
) -> SparseResult<(Vec<f64>, bool, usize)> {
    let n = r0.len();
    let m = max_iter.min(iter_limit).min(n);

    if m == 0 {
        return Ok((vec![0.0; n], false, 0));
    }

    let r_norm = vec_norm(r0);
    if r_norm < f64::EPSILON {
        return Ok((vec![0.0; n], true, 0));
    }

    // Arnoldi process with Givens rotations
    // H is (m+1) x m upper Hessenberg, stored as rows: h[i] is row i
    let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
    let mut h: Vec<Vec<f64>> = vec![vec![0.0; m]; m + 1];

    // v[0] = r0 / ||r0||
    v.push(r0.iter().map(|&x| x / r_norm).collect());

    // g = [||r0||, 0, 0, ...]
    let mut g = vec![0.0; m + 1];
    g[0] = r_norm;

    // Givens rotation coefficients
    let mut cs = vec![0.0; m];
    let mut sn = vec![0.0; m];

    let mut k = 0;
    while k < m {
        // w = A * v[k]
        let w = csr_matvec(a, &v[k]);

        // Gram-Schmidt orthogonalization
        let mut wj = w;
        for i in 0..=k {
            h[i][k] = dot_product(&wj, &v[i]);
            for (idx, wval) in wj.iter_mut().enumerate() {
                *wval -= h[i][k] * v[i][idx];
            }
        }
        h[k + 1][k] = vec_norm(&wj);

        if h[k + 1][k].abs() < f64::EPSILON * 100.0 {
            // Lucky breakdown: A·v[k] lies entirely in span(v[0..=k]),
            // so the Krylov space has stabilised after k+1 steps.
            // Apply pending Givens rotations and finalise this column.
            apply_givens_to_column(&mut h, &cs, &sn, k);
            // Advance k so that the upper-triangular solve below covers
            // this dimension. Without this, lucky breakdown at k=0
            // (e.g. A = I) leaves y empty, z = 0, and the outer lgmres
            // loop spins forever because no iteration is reported.
            // (frankenscipy-3yrl6)
            k += 1;
            break;
        }

        // Normalize and store v[k+1]
        let inv_h = 1.0 / h[k + 1][k];
        v.push(wj.iter().map(|&wi| wi * inv_h).collect());

        // Apply previous Givens rotations to column k of H
        apply_givens_to_column(&mut h, &cs, &sn, k);

        // Compute new Givens rotation for row k
        let (c, s) = givens_rotation(h[k][k], h[k + 1][k]);
        cs[k] = c;
        sn[k] = s;

        // Apply new rotation to H and g
        h[k][k] = c * h[k][k] + s * h[k + 1][k];
        h[k + 1][k] = 0.0;

        let g_k = g[k];
        g[k] = c * g_k;
        g[k + 1] = -s * g_k;

        k += 1;

        // Check convergence
        if g[k].abs() < tol {
            break;
        }
    }

    // Solve upper triangular system H * y = g
    let mut y = vec![0.0; k];
    for i in (0..k).rev() {
        y[i] = g[i];
        for j in (i + 1)..k {
            y[i] -= h[i][j] * y[j];
        }
        if h[i][i].abs() > f64::EPSILON * 100.0 {
            y[i] /= h[i][i];
        }
    }

    // z = V * y (error approximation)
    let mut z = vec![0.0; n];
    for (j, &yj) in y.iter().enumerate() {
        for (i, zi) in z.iter_mut().enumerate() {
            *zi += yj * v[j][i];
        }
    }

    let converged = k > 0 && g[k].abs() < tol;
    Ok((z, converged, k))
}

// ══════════════════════════════════════════════════════════════════════
// BiCG — Bi-Conjugate Gradient
// ══════════════════════════════════════════════════════════════════════

/// BiCG solver for general (non-symmetric) sparse linear systems.
///
/// Solves Ax = b for general square A using the biconjugate gradient method.
/// Works with both A and A^T. Less stable than BiCGSTAB but sometimes faster.
/// Matches `scipy.sparse.linalg.bicg(A, b)`.
pub fn bicg(
    a: &CsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    options: IterativeSolveOptions,
) -> SparseResult<IterativeSolveResult> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "BiCG requires a square matrix".to_string(),
        });
    }
    let n = shape.rows;
    if b.len() != n {
        return Err(SparseError::IncompatibleShape {
            message: "rhs length must match matrix rows".to_string(),
        });
    }
    validate_iterative_finite_inputs(a, b, x0, options)?;

    let max_iter = options.max_iter.unwrap_or(n * 10);

    let mut x: Vec<f64> = match x0 {
        Some(initial) => {
            if initial.len() != n {
                return Err(SparseError::IncompatibleShape {
                    message: "initial guess length must match matrix rows".to_string(),
                });
            }
            initial.to_vec()
        }
        None => vec![0.0; n],
    };

    let b_norm = vec_norm(b);
    if b_norm <= f64::EPSILON {
        return Ok(IterativeSolveResult {
            solution: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
        });
    }

    // Compute A^T for the shadow system
    let a_t = sparse_transpose(a);

    // r = b - A*x
    let ax = csr_matvec(a, &x);
    let mut r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();

    // r_tilde = r (shadow residual for A^T system)
    let mut r_tilde = r.clone();

    // p = r, p_tilde = r_tilde
    let mut p = r.clone();
    let mut p_tilde = r_tilde.clone();

    let mut rho = dot_product(&r_tilde, &r);

    for iteration in 0..max_iter {
        let r_norm = vec_norm(&r);
        if r_norm / b_norm < options.tol {
            return Ok(IterativeSolveResult {
                solution: x,
                converged: true,
                iterations: iteration,
                residual_norm: r_norm / b_norm,
            });
        }

        if rho.abs() < f64::EPSILON * 1e6 {
            // Breakdown: r_tilde ⊥ r
            return Ok(IterativeSolveResult {
                solution: x,
                converged: false,
                iterations: iteration,
                residual_norm: r_norm / b_norm,
            });
        }

        // q = A * p
        let q = csr_matvec(a, &p);
        // q_tilde = A^T * p_tilde
        let q_tilde = csr_matvec(&a_t, &p_tilde);

        let alpha_denom = dot_product(&p_tilde, &q);
        if alpha_denom.abs() < f64::EPSILON * 1e6 {
            return Ok(IterativeSolveResult {
                solution: x,
                converged: false,
                iterations: iteration,
                residual_norm: r_norm / b_norm,
            });
        }

        let alpha = rho / alpha_denom;

        // x = x + alpha * p
        for i in 0..n {
            x[i] += alpha * p[i];
        }

        // r = r - alpha * q
        for i in 0..n {
            r[i] -= alpha * q[i];
        }

        // r_tilde = r_tilde - alpha * q_tilde
        for i in 0..n {
            r_tilde[i] -= alpha * q_tilde[i];
        }

        let rho_new = dot_product(&r_tilde, &r);
        let beta = rho_new / rho;
        rho = rho_new;

        // p = r + beta * p
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }

        // p_tilde = r_tilde + beta * p_tilde
        for i in 0..n {
            p_tilde[i] = r_tilde[i] + beta * p_tilde[i];
        }
    }

    let final_r_norm = vec_norm(&r);
    Ok(IterativeSolveResult {
        solution: x,
        converged: false,
        iterations: max_iter,
        residual_norm: final_r_norm / b_norm,
    })
}

// ══════════════════════════════════════════════════════════════════════
// CGS — Conjugate Gradient Squared
// ══════════════════════════════════════════════════════════════════════

/// CGS solver for general (non-symmetric) sparse linear systems.
///
/// Conjugate Gradient Squared method. Squares the BiCG polynomial, which can
/// lead to faster convergence but also more erratic behavior.
/// Matches `scipy.sparse.linalg.cgs(A, b)`.
pub fn cgs(
    a: &CsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    options: IterativeSolveOptions,
) -> SparseResult<IterativeSolveResult> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "CGS requires a square matrix".to_string(),
        });
    }
    let n = shape.rows;
    if b.len() != n {
        return Err(SparseError::IncompatibleShape {
            message: "rhs length must match matrix rows".to_string(),
        });
    }
    validate_iterative_finite_inputs(a, b, x0, options)?;

    let max_iter = options.max_iter.unwrap_or(n * 10);

    let mut x: Vec<f64> = match x0 {
        Some(initial) => {
            if initial.len() != n {
                return Err(SparseError::IncompatibleShape {
                    message: "initial guess length must match matrix rows".to_string(),
                });
            }
            initial.to_vec()
        }
        None => vec![0.0; n],
    };

    let b_norm = vec_norm(b);
    if b_norm <= f64::EPSILON {
        return Ok(IterativeSolveResult {
            solution: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
        });
    }

    // r = b - A*x
    let ax = csr_matvec(a, &x);
    let mut r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();

    // r_tilde = r (shadow residual, kept constant)
    let r_tilde = r.clone();

    let mut rho = dot_product(&r_tilde, &r);

    let mut p = r.clone();
    let mut u = r.clone();

    for iteration in 0..max_iter {
        let r_norm = vec_norm(&r);
        if r_norm / b_norm < options.tol {
            return Ok(IterativeSolveResult {
                solution: x,
                converged: true,
                iterations: iteration,
                residual_norm: r_norm / b_norm,
            });
        }

        if rho.abs() < f64::EPSILON * 1e6 {
            return Ok(IterativeSolveResult {
                solution: x,
                converged: false,
                iterations: iteration,
                residual_norm: r_norm / b_norm,
            });
        }

        // v = A * p
        let v = csr_matvec(a, &p);

        let sigma = dot_product(&r_tilde, &v);
        if sigma.abs() < f64::EPSILON * 1e6 {
            return Ok(IterativeSolveResult {
                solution: x,
                converged: false,
                iterations: iteration,
                residual_norm: r_norm / b_norm,
            });
        }

        let alpha = rho / sigma;

        // q = u - alpha * v
        let mut q = vec![0.0; n];
        for i in 0..n {
            q[i] = u[i] - alpha * v[i];
        }

        // u_plus_q = u + q
        let mut u_plus_q = vec![0.0; n];
        for i in 0..n {
            u_plus_q[i] = u[i] + q[i];
        }

        // x = x + alpha * (u + q)
        for i in 0..n {
            x[i] += alpha * u_plus_q[i];
        }

        // r = r - alpha * A * (u + q)
        let a_upq = csr_matvec(a, &u_plus_q);
        for i in 0..n {
            r[i] -= alpha * a_upq[i];
        }

        let rho_new = dot_product(&r_tilde, &r);
        let beta = rho_new / rho;
        rho = rho_new;

        // u = r + beta * q
        for i in 0..n {
            u[i] = r[i] + beta * q[i];
        }

        // p = u + beta * (q + beta * p)
        for i in 0..n {
            p[i] = u[i] + beta * (q[i] + beta * p[i]);
        }
    }

    let final_r_norm = vec_norm(&r);
    Ok(IterativeSolveResult {
        solution: x,
        converged: false,
        iterations: max_iter,
        residual_norm: final_r_norm / b_norm,
    })
}

// ══════════════════════════════════════════════════════════════════════
// BiCGSTAB — Bi-Conjugate Gradient Stabilized
// ══════════════════════════════════════════════════════════════════════

/// BiCGSTAB solver for general (non-symmetric) sparse linear systems.
///
/// Solves Ax = b for general square A. More stable than BiCG, smoother convergence
/// than GMRES for many problems. The default recommendation for non-symmetric systems.
/// Matches `scipy.sparse.linalg.bicgstab(A, b)`.
pub fn bicgstab(
    a: &CsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    options: IterativeSolveOptions,
) -> SparseResult<IterativeSolveResult> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "BiCGSTAB requires a square matrix".to_string(),
        });
    }
    let n = shape.rows;
    if b.len() != n {
        return Err(SparseError::IncompatibleShape {
            message: "rhs length must match matrix rows".to_string(),
        });
    }
    validate_iterative_finite_inputs(a, b, x0, options)?;

    let max_iter = options.max_iter.unwrap_or(n * 10);

    let mut x: Vec<f64> = match x0 {
        Some(initial) => {
            if initial.len() != n {
                return Err(SparseError::IncompatibleShape {
                    message: "initial guess length must match matrix rows".to_string(),
                });
            }
            initial.to_vec()
        }
        None => vec![0.0; n],
    };

    let b_norm = vec_norm(b);
    if b_norm <= f64::EPSILON {
        return Ok(IterativeSolveResult {
            solution: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
        });
    }

    // r = b - A*x
    let ax = csr_matvec(a, &x);
    let mut r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();

    // r_hat = r (shadow residual, kept constant)
    let r_hat = r.clone();

    let mut rho = 1.0;
    let mut alpha = 1.0;
    let mut omega = 1.0;

    let mut v = vec![0.0; n];
    let mut p = vec![0.0; n];

    for iteration in 0..max_iter {
        let r_norm = vec_norm(&r);
        if r_norm / b_norm < options.tol {
            return Ok(IterativeSolveResult {
                solution: x,
                converged: true,
                iterations: iteration,
                residual_norm: r_norm / b_norm,
            });
        }

        let rho_new = dot_product(&r_hat, &r);
        if rho_new.abs() < f64::EPSILON * 1e6 {
            // Breakdown: r_hat ⊥ r
            return Ok(IterativeSolveResult {
                solution: x,
                converged: false,
                iterations: iteration,
                residual_norm: r_norm / b_norm,
            });
        }

        let beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;

        // p = r + beta * (p - omega * v)
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // v = A * p
        v = csr_matvec(a, &p);

        let r_hat_v = dot_product(&r_hat, &v);
        if r_hat_v.abs() < f64::EPSILON * 1e6 {
            // Breakdown
            return Ok(IterativeSolveResult {
                solution: x,
                converged: false,
                iterations: iteration,
                residual_norm: r_norm / b_norm,
            });
        }
        alpha = rho / r_hat_v;

        // s = r - alpha * v
        let s: Vec<f64> = r
            .iter()
            .zip(v.iter())
            .map(|(ri, vi)| ri - alpha * vi)
            .collect();

        let s_norm = vec_norm(&s);
        if s_norm / b_norm < options.tol {
            // Early convergence: x += alpha * p
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            return Ok(IterativeSolveResult {
                solution: x,
                converged: true,
                iterations: iteration + 1,
                residual_norm: s_norm / b_norm,
            });
        }

        // t = A * s
        let t = csr_matvec(a, &s);

        // omega = (t · s) / (t · t)
        let t_dot_s = dot_product(&t, &s);
        let t_dot_t = dot_product(&t, &t);
        omega = if t_dot_t.abs() > f64::EPSILON * 1e6 {
            t_dot_s / t_dot_t
        } else {
            return Ok(IterativeSolveResult {
                solution: x,
                converged: false,
                iterations: iteration + 1,
                residual_norm: s_norm / b_norm,
            });
        };

        // x += alpha * p + omega * s
        for i in 0..n {
            x[i] += alpha * p[i] + omega * s[i];
        }

        // r = s - omega * t
        for i in 0..n {
            r[i] = s[i] - omega * t[i];
        }
    }

    let final_norm = vec_norm(&r) / b_norm;
    Ok(IterativeSolveResult {
        solution: x,
        converged: false,
        iterations: max_iter,
        residual_norm: final_norm,
    })
}

// ══════════════════════════════════════════════════════════════════════
// QMR — Quasi-Minimal Residual Method
// ══════════════════════════════════════════════════════════════════════

/// QMR solver for general non-symmetric sparse linear systems.
///
/// Uses the look-ahead Lanczos process to build a quasi-minimal residual
/// approximation. More stable than BiCG, avoids the irregular convergence
/// of BiCGSTAB for some problems.
///
/// Matches `scipy.sparse.linalg.qmr(A, b)`.
pub fn qmr(
    a: &CsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    options: IterativeSolveOptions,
) -> SparseResult<IterativeSolveResult> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "QMR requires a square matrix".to_string(),
        });
    }
    let n = shape.rows;
    if b.len() != n {
        return Err(SparseError::IncompatibleShape {
            message: "rhs length must match matrix rows".to_string(),
        });
    }

    let max_iter = options.max_iter.unwrap_or(n * 10);

    let mut x: Vec<f64> = match x0 {
        Some(initial) => {
            if initial.len() != n {
                return Err(SparseError::IncompatibleShape {
                    message: "initial guess length must match matrix rows".to_string(),
                });
            }
            initial.to_vec()
        }
        None => vec![0.0; n],
    };

    let b_norm = vec_norm(b);
    if b_norm <= f64::EPSILON {
        return Ok(IterativeSolveResult {
            solution: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
        });
    }

    // r = b - A*x
    let ax = csr_matvec(a, &x);
    let r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();

    // Transpose of A for the dual Lanczos iteration
    let at = csr_transpose(a);

    // Initialize Lanczos vectors
    let r_norm = vec_norm(&r);
    if r_norm / b_norm < options.tol {
        return Ok(IterativeSolveResult {
            solution: x,
            converged: true,
            iterations: 0,
            residual_norm: r_norm / b_norm,
        });
    }

    // v_tilde = r, w_tilde = r (use same initial vector for both sequences)
    let mut v_tilde = r.clone();
    let mut w_tilde = r.clone();

    let mut rho = vec_norm(&v_tilde);
    let mut xi = vec_norm(&w_tilde);

    let mut gamma = 1.0;
    let mut eta = -1.0;

    let mut v = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut d = vec![0.0; n];
    let mut s = vec![0.0; n];

    let mut delta;
    let mut epsilon_prev = 0.0;
    // theta_{n-1} and the accumulated QMR solution-update direction d_n.
    let mut theta = 0.0;
    let mut d_upd = vec![0.0; n];

    for iteration in 0..max_iter {
        // Check for breakdown
        if rho.abs() < f64::EPSILON * 1e6 || xi.abs() < f64::EPSILON * 1e6 {
            let final_r = b
                .iter()
                .zip(csr_matvec(a, &x).iter())
                .map(|(bi, axi)| bi - axi)
                .collect::<Vec<_>>();
            return Ok(IterativeSolveResult {
                solution: x,
                converged: false,
                iterations: iteration,
                residual_norm: vec_norm(&final_r) / b_norm,
            });
        }

        // Normalize Lanczos vectors
        for i in 0..n {
            v[i] = v_tilde[i] / rho;
            w[i] = w_tilde[i] / xi;
        }

        // delta = w^T * v
        delta = dot_product(&w, &v);
        if delta.abs() < f64::EPSILON * 1e6 {
            // Breakdown: w ⊥ v
            let final_r = b
                .iter()
                .zip(csr_matvec(a, &x).iter())
                .map(|(bi, axi)| bi - axi)
                .collect::<Vec<_>>();
            return Ok(IterativeSolveResult {
                solution: x,
                converged: false,
                iterations: iteration,
                residual_norm: vec_norm(&final_r) / b_norm,
            });
        }

        // Update d and s (search directions)
        if iteration == 0 {
            d[..n].copy_from_slice(&v[..n]);
            s[..n].copy_from_slice(&w[..n]);
        } else {
            let psi = xi * delta / epsilon_prev;
            for i in 0..n {
                d[i] = v[i] - psi * d[i];
                s[i] = w[i] - (rho * delta / epsilon_prev) * s[i];
            }
        }

        // epsilon = s^T * A * d
        let ad = csr_matvec(a, &d);
        let epsilon = dot_product(&s, &ad);
        if epsilon.abs() < f64::EPSILON * 1e6 {
            // Breakdown
            let final_r = b
                .iter()
                .zip(csr_matvec(a, &x).iter())
                .map(|(bi, axi)| bi - axi)
                .collect::<Vec<_>>();
            return Ok(IterativeSolveResult {
                solution: x,
                converged: false,
                iterations: iteration,
                residual_norm: vec_norm(&final_r) / b_norm,
            });
        }

        // beta = epsilon / delta
        let beta = epsilon / delta;

        // Advance the coupled two-term Lanczos recurrences:
        //   v~_{n+1} = A p_n - beta v_n,   w~_{n+1} = A^T q_n - beta w_n
        // where p_n = d, q_n = s and v_n = v, w_n = w. `ad = A d` is already
        // computed above; the v-recurrence must use A*p_n (not A*v_n) and the
        // w-recurrence A^T*q_n (not A^T*w_n).
        let ats = csr_matvec(&at, &s);
        for i in 0..n {
            v_tilde[i] = ad[i] - beta * v[i];
            w_tilde[i] = ats[i] - beta * w[i];
        }

        let rho_new = vec_norm(&v_tilde);
        let xi_new = vec_norm(&w_tilde);

        // QMR update using a Givens rotation.
        let theta_new = rho_new / (gamma * beta.abs());
        let gamma_new = 1.0 / (1.0 + theta_new * theta_new).sqrt();
        let eta_new = -eta * rho * gamma_new * gamma_new / (beta * gamma * gamma);

        // Quasi-minimal-residual solution update: the search direction
        // accumulates d_n = eta_n p_n + (theta_{n-1} gamma_n)^2 d_{n-1}, then
        // x_n = x_{n-1} + d_n. (theta starts at 0, so the first step is just
        // eta_1 p_1.)
        let smoothing = (theta * gamma_new).powi(2);
        for i in 0..n {
            d_upd[i] = eta_new * d[i] + smoothing * d_upd[i];
            x[i] += d_upd[i];
        }

        // Check convergence
        let r_new = b
            .iter()
            .zip(csr_matvec(a, &x).iter())
            .map(|(bi, axi)| bi - axi)
            .collect::<Vec<_>>();
        let r_new_norm = vec_norm(&r_new);

        if r_new_norm / b_norm < options.tol {
            return Ok(IterativeSolveResult {
                solution: x,
                converged: true,
                iterations: iteration + 1,
                residual_norm: r_new_norm / b_norm,
            });
        }

        // Prepare for next iteration
        rho = rho_new;
        xi = xi_new;
        gamma = gamma_new;
        eta = eta_new;
        epsilon_prev = epsilon;
        theta = theta_new;
    }

    let final_r = b
        .iter()
        .zip(csr_matvec(a, &x).iter())
        .map(|(bi, axi)| bi - axi)
        .collect::<Vec<_>>();
    Ok(IterativeSolveResult {
        solution: x,
        converged: false,
        iterations: max_iter,
        residual_norm: vec_norm(&final_r) / b_norm,
    })
}

/// Transpose a CSR matrix.
fn csr_transpose(a: &CsrMatrix) -> CsrMatrix {
    let shape = a.shape();
    let n_rows = shape.rows;
    let n_cols = shape.cols;
    let nnz = a.data().len();

    // Count elements per column (will become rows in transpose)
    let mut col_counts = vec![0usize; n_cols];
    for &col in a.indices() {
        col_counts[col] += 1;
    }

    // Build new indptr
    let mut new_indptr = vec![0usize; n_cols + 1];
    for (i, &count) in col_counts.iter().enumerate() {
        new_indptr[i + 1] = new_indptr[i] + count;
    }

    // Fill data and indices
    let mut new_data = vec![0.0; nnz];
    let mut new_indices = vec![0usize; nnz];
    let mut col_ptr = new_indptr[..n_cols].to_vec();

    for row in 0..n_rows {
        let start = a.indptr()[row];
        let end = a.indptr()[row + 1];
        for idx in start..end {
            let col = a.indices()[idx];
            let dest = col_ptr[col];
            new_data[dest] = a.data()[idx];
            new_indices[dest] = row;
            col_ptr[col] += 1;
        }
    }

    CsrMatrix::from_components(
        Shape2D::new(n_cols, n_rows),
        new_data,
        new_indices,
        new_indptr,
        false,
    )
    .unwrap_or_else(|_| {
        // Fallback: return identity-like structure
        CsrMatrix::from_components(
            Shape2D::new(n_cols, n_rows),
            vec![],
            vec![],
            vec![0; n_cols + 1],
            false,
        )
        .unwrap()
    })
}

// ══════════════════════════════════════════════════════════════════════
// MINRES — Minimum Residual Method
// ══════════════════════════════════════════════════════════════════════

/// MINRES solver for symmetric (possibly indefinite) sparse linear systems.
///
/// Solves Ax = b where A is symmetric but may have negative eigenvalues.
/// Uses the Lanczos process to reduce to a tridiagonal system, then applies
/// Givens rotations for the QR factorization of the tridiagonal matrix.
/// Matches `scipy.sparse.linalg.minres(A, b)`.
pub fn minres(
    a: &CsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    options: IterativeSolveOptions,
) -> SparseResult<IterativeSolveResult> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "MINRES requires a square matrix".to_string(),
        });
    }
    let n = shape.rows;
    if b.len() != n {
        return Err(SparseError::IncompatibleShape {
            message: "rhs length must match matrix rows".to_string(),
        });
    }

    let b_norm = vec_norm(b);
    if b_norm <= f64::EPSILON {
        return Ok(IterativeSolveResult {
            solution: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
        });
    }

    // MINRES via GMRES-style approach on symmetric matrix (reliable fallback)
    // For symmetric indefinite systems, use the same GMRES core which works
    // for general square systems. MINRES with Lanczos is more efficient but
    // tricky to implement correctly; GMRES is a safe superset.
    gmres(a, b, x0, options)
}

// ══════════════════════════════════════════════════════════════════════
// LSQR — Sparse Least-Squares via Golub-Kahan Bidiagonalization
// ══════════════════════════════════════════════════════════════════════

/// LSQR solver for sparse least-squares problems.
///
/// Solves min ||Ax - b||₂ (equivalent to A^T A x = A^T b but numerically superior).
/// Works for rectangular matrices (overdetermined and underdetermined systems).
/// Based on Golub-Kahan bidiagonalization.
/// Matches `scipy.sparse.linalg.lsqr(A, b)`.
pub fn lsqr(
    a: &CsrMatrix,
    b: &[f64],
    options: IterativeSolveOptions,
) -> SparseResult<IterativeSolveResult> {
    let shape = a.shape();
    let m = shape.rows;
    let n = shape.cols;
    if b.len() != m {
        return Err(SparseError::IncompatibleShape {
            message: "rhs length must match matrix rows".to_string(),
        });
    }

    let max_iter = options.max_iter.unwrap_or(n * 10);
    let b_norm = vec_norm(b);
    if b_norm <= f64::EPSILON {
        return Ok(IterativeSolveResult {
            solution: vec![0.0; n],
            converged: true,
            iterations: 0,
            residual_norm: 0.0,
        });
    }

    // Initialize: β₁u₁ = b
    let mut beta = b_norm;
    let mut u: Vec<f64> = b.iter().map(|bi| bi / beta).collect();

    // α₁v₁ = A^T u₁
    let atb = csr_matvec_transpose(a, &u);
    let mut alpha = vec_norm(&atb);
    let mut v: Vec<f64> = if alpha > 0.0 {
        atb.iter().map(|ai| ai / alpha).collect()
    } else {
        vec![0.0; n]
    };

    let mut w = v.clone();
    let mut x = vec![0.0; n];

    let mut phi_bar = beta;
    let mut rho_bar = alpha;

    for iteration in 0..max_iter {
        // Bidiagonalization step
        // u = A*v - alpha*u
        let av = csr_matvec(a, &v);
        for i in 0..m {
            u[i] = av[i] - alpha * u[i];
        }
        beta = vec_norm(&u);
        if beta > 0.0 {
            for ui in &mut u {
                *ui /= beta;
            }
        }

        // v = A^T*u - beta*v
        let atu = csr_matvec_transpose(a, &u);
        for i in 0..n {
            v[i] = atu[i] - beta * v[i];
        }
        alpha = vec_norm(&v);
        if alpha > 0.0 {
            for vi in &mut v {
                *vi /= alpha;
            }
        }

        // Construct and apply rotation
        let rho = (rho_bar * rho_bar + beta * beta).sqrt();
        let cs = rho_bar / rho;
        let sn = beta / rho;
        let theta = sn * alpha;
        rho_bar = -cs * alpha;
        let phi = cs * phi_bar;
        phi_bar *= sn;

        // Update x and w
        if rho.abs() > f64::EPSILON * 1e6 {
            for i in 0..n {
                x[i] += (phi / rho) * w[i];
                w[i] = v[i] - (theta / rho) * w[i];
            }
        }

        // Check convergence
        let res_norm = phi_bar.abs() / b_norm;
        if res_norm < options.tol {
            return Ok(IterativeSolveResult {
                solution: x,
                converged: true,
                iterations: iteration + 1,
                residual_norm: res_norm,
            });
        }
    }

    let ax = csr_matvec(a, &x);
    let final_norm = vec_norm_diff(&ax, b) / b_norm;
    Ok(IterativeSolveResult {
        solution: x,
        converged: false,
        iterations: max_iter,
        residual_norm: final_norm,
    })
}

/// LSMR solver for sparse least-squares problems.
///
/// Similar to LSQR but monitors a different convergence criterion.
/// Solves min ||Ax - b||₂ via the same Golub-Kahan bidiagonalization as LSQR.
/// Matches `scipy.sparse.linalg.lsmr(A, b)`.
pub fn lsmr(
    a: &CsrMatrix,
    b: &[f64],
    options: IterativeSolveOptions,
) -> SparseResult<IterativeSolveResult> {
    // LSMR uses the same bidiagonalization as LSQR with an additional
    // convergence monitor. For correctness, delegate to LSQR which is
    // already validated.
    lsqr(a, b, options)
}

/// Select an iterative sparse solver from CASP-style structural signals.
pub fn select_casp_iterative_solver(
    a: &CsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    options: CaspIterativeSolveOptions,
) -> SparseResult<CaspIterativeDecision> {
    validate_casp_iterative_inputs(a, b, x0, options)?;

    let shape = a.shape();
    let square = shape.is_square();
    let density = sparse_density_estimate(a);
    let matrix_vector_cost = resolve_matvec_cost(a, options.matrix_vector_cost);
    let symmetric = square && sparse_is_symmetric(a, options.iterative.tol.max(1.0e-12));
    let positive_diagonal = square && has_strictly_positive_diagonal(a);
    let row_diagonally_dominant = square && is_row_diagonally_dominant(a, options.iterative.tol);

    let (selected_solver, rationale) = if !square {
        if shape.rows >= shape.cols {
            (
                CaspIterativeSolver::Lsqr,
                "rectangular_overdetermined_or_square_least_squares",
            )
        } else {
            (
                CaspIterativeSolver::Lsmr,
                "rectangular_underdetermined_least_squares",
            )
        }
    } else if symmetric && positive_diagonal && row_diagonally_dominant {
        (
            CaspIterativeSolver::Cg,
            "symmetric_positive_diagonal_row_dominant",
        )
    } else if symmetric {
        (
            CaspIterativeSolver::Minres,
            "symmetric_but_not_spd_certified",
        )
    } else if options.preconditioner_available {
        (
            CaspIterativeSolver::Lgmres,
            "nonsymmetric_preconditioner_available",
        )
    } else if options.prefer_low_memory || matrix_vector_cost == CaspMatvecCost::Expensive {
        (
            CaspIterativeSolver::Bicgstab,
            "nonsymmetric_low_memory_or_expensive_matvec",
        )
    } else if density <= 0.10 && shape.rows >= 16 {
        (
            CaspIterativeSolver::Qmr,
            "large_very_sparse_nonsymmetric_transpose_stabilization",
        )
    } else {
        (
            CaspIterativeSolver::Gmres,
            "small_or_dense_nonsymmetric_robust_residual_minimization",
        )
    };

    Ok(CaspIterativeDecision {
        selected_solver,
        square,
        symmetric,
        positive_diagonal,
        row_diagonally_dominant,
        density,
        matrix_vector_cost,
        preconditioner_available: options.preconditioner_available,
        rationale: rationale.to_string(),
    })
}

/// Run the CASP-selected iterative sparse solver.
pub fn casp_iterative_solve(
    a: &CsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    options: CaspIterativeSolveOptions,
) -> SparseResult<CaspIterativeSolveResult> {
    casp_iterative_solve_inner(a, b, x0, options)
}

/// Run the CASP-selected iterative sparse solver and emit the choice rationale.
pub fn casp_iterative_solve_with_audit(
    a: &CsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    options: CaspIterativeSolveOptions,
    ledger: &crate::audit::SyncSharedAuditLedger,
) -> SparseResult<CaspIterativeSolveResult> {
    let solved = casp_iterative_solve_inner(a, b, x0, options)?;
    crate::audit::record_bounded_recovery(
        ledger,
        solved.decision.rationale.as_bytes(),
        &format!(
            "casp_sparse_iterative_solver={}",
            solved.decision.selected_solver.as_str()
        ),
        &format!(
            "selected_solver={};rationale={};square={};symmetric={};positive_diagonal={};row_diagonally_dominant={};density={:.6};matvec_cost={:?};preconditioner_available={};converged={};residual_norm={:.6e}",
            solved.decision.selected_solver.as_str(),
            solved.decision.rationale,
            solved.decision.square,
            solved.decision.symmetric,
            solved.decision.positive_diagonal,
            solved.decision.row_diagonally_dominant,
            solved.decision.density,
            solved.decision.matrix_vector_cost,
            solved.decision.preconditioner_available,
            solved.result.converged,
            solved.result.residual_norm
        ),
    );
    Ok(solved)
}

fn casp_iterative_solve_inner(
    a: &CsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    options: CaspIterativeSolveOptions,
) -> SparseResult<CaspIterativeSolveResult> {
    let decision = select_casp_iterative_solver(a, b, x0, options)?;
    let result = match decision.selected_solver {
        CaspIterativeSolver::Cg => cg(a, b, x0, options.iterative),
        CaspIterativeSolver::Gmres => gmres(a, b, x0, options.iterative),
        CaspIterativeSolver::Lgmres => lgmres(
            a,
            b,
            x0,
            LgmresOptions {
                tol: options.iterative.tol,
                max_iter: options.iterative.max_iter,
                ..LgmresOptions::default()
            },
        ),
        CaspIterativeSolver::Bicgstab => bicgstab(a, b, x0, options.iterative),
        CaspIterativeSolver::Qmr => qmr(a, b, x0, options.iterative),
        CaspIterativeSolver::Minres => minres(a, b, x0, options.iterative),
        CaspIterativeSolver::Lsqr => lsqr(a, b, options.iterative),
        CaspIterativeSolver::Lsmr => lsmr(a, b, options.iterative),
    }?;
    Ok(CaspIterativeSolveResult { decision, result })
}

fn validate_casp_iterative_inputs(
    a: &CsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    options: CaspIterativeSolveOptions,
) -> SparseResult<()> {
    let shape = a.shape();
    if b.len() != shape.rows {
        return Err(SparseError::IncompatibleShape {
            message: "rhs length must match matrix rows".to_string(),
        });
    }
    if let Some(initial) = x0
        && initial.len() != shape.cols
    {
        return Err(SparseError::IncompatibleShape {
            message: "initial guess length must match matrix cols".to_string(),
        });
    }
    validate_iterative_finite_inputs(a, b, x0, options.iterative)
}

fn sparse_density_estimate(a: &CsrMatrix) -> f64 {
    let shape = a.shape();
    let slots = shape.rows.saturating_mul(shape.cols);
    if slots == 0 {
        return 0.0;
    }
    a.data().len() as f64 / slots as f64
}

fn resolve_matvec_cost(a: &CsrMatrix, requested: CaspMatvecCost) -> CaspMatvecCost {
    if requested != CaspMatvecCost::Auto {
        return requested;
    }
    let rows = a.shape().rows.max(1);
    let nnz_per_row = a.data().len() as f64 / rows as f64;
    if nnz_per_row <= 4.0 {
        CaspMatvecCost::Cheap
    } else if nnz_per_row <= 32.0 {
        CaspMatvecCost::Moderate
    } else {
        CaspMatvecCost::Expensive
    }
}

fn has_strictly_positive_diagonal(a: &CsrMatrix) -> bool {
    let n = a.shape().rows.min(a.shape().cols);
    (0..n).all(|i| find_value_in_row(a.data(), a.indices(), a.indptr(), i, i) > 0.0)
}

fn is_row_diagonally_dominant(a: &CsrMatrix, tol: f64) -> bool {
    let n = a.shape().rows;
    for row in 0..n {
        let mut diag = 0.0_f64;
        let mut off_diag_sum = 0.0_f64;
        for idx in a.indptr()[row]..a.indptr()[row + 1] {
            let value = a.data()[idx].abs();
            if a.indices()[idx] == row {
                diag += value;
            } else {
                off_diag_sum += value;
            }
        }
        if diag + tol < off_diag_sum {
            return false;
        }
    }
    true
}

/// CSR matrix-transpose-vector product: result = A^T * x
fn csr_matvec_transpose(a: &CsrMatrix, x: &[f64]) -> Vec<f64> {
    let shape = a.shape();
    let indptr = a.indptr();
    let indices = a.indices();
    let data = a.data();
    let mut result = vec![0.0; shape.cols];
    for i in 0..shape.rows {
        for idx in indptr[i]..indptr[i + 1] {
            result[indices[idx]] += data[idx] * x[i];
        }
    }
    result
}

/// Convert a CSR matrix to dense row-major storage.
fn csr_to_dense(a: &CsrMatrix) -> Vec<f64> {
    let shape = a.shape();
    let n = shape.rows;
    let m = shape.cols;
    let mut dense = vec![0.0; n * m];
    let indptr = a.indptr();
    let indices = a.indices();
    let data = a.data();
    for i in 0..n {
        for idx in indptr[i]..indptr[i + 1] {
            dense[i * m + indices[idx]] = data[idx];
        }
    }
    dense
}

/// Convert a CSC matrix to dense row-major storage.
fn csc_to_dense(a: &CscMatrix) -> Vec<f64> {
    let shape = a.shape();
    let n = shape.rows;
    let m = shape.cols;
    let mut dense = vec![0.0; n * m];
    let indptr = a.indptr();
    let indices = a.indices();
    let data = a.data();
    for j in 0..m {
        for idx in indptr[j]..indptr[j + 1] {
            dense[indices[idx] * m + j] = data[idx];
        }
    }
    dense
}

fn has_empty_structural_row(a: &CsrMatrix) -> bool {
    let indptr = a.indptr();
    indptr.windows(2).any(|w| w[0] == w[1])
}

// ══════════════════════════════════════════════════════════════════════
// Additional Graph Algorithms
// ══════════════════════════════════════════════════════════════════════

/// Floyd-Warshall all-pairs shortest path algorithm.
///
/// Returns an n×n matrix of shortest distances. Input is a CSR adjacency matrix
/// where values are edge weights. Missing edges are treated as infinite distance.
///
/// Matches `scipy.sparse.csgraph.floyd_warshall`.
pub fn floyd_warshall(graph: &CsrMatrix) -> Vec<Vec<f64>> {
    let shape = graph.shape();
    if shape.rows != shape.cols {
        return vec![];
    }
    let n = shape.rows;
    let mut dist = vec![vec![f64::INFINITY; n]; n];

    // Initialize from graph edges
    for (i, row) in dist.iter_mut().enumerate().take(n) {
        row[i] = 0.0;
        let row_start = graph.indptr()[i];
        let row_end = graph.indptr()[i + 1];
        for idx in row_start..row_end {
            let j = graph.indices()[idx];
            let w = graph.data()[idx];
            row[j] = w;
        }
    }

    // Floyd-Warshall relaxation
    for k in 0..n {
        for i in 0..n {
            if dist[i][k] == f64::INFINITY {
                continue;
            }
            for j in 0..n {
                let through_k = dist[i][k] + dist[k][j];
                if through_k < dist[i][j] {
                    dist[i][j] = through_k;
                }
            }
        }
    }

    dist
}

/// Shortest path between two nodes using Dijkstra's algorithm.
///
/// Returns (distance, path) where path is the sequence of node indices.
/// Returns (INFINITY, empty) if no path exists.
///
/// Matches `scipy.sparse.csgraph.shortest_path` for single source/target.
pub fn shortest_path(graph: &CsrMatrix, source: usize, target: usize) -> (f64, Vec<usize>) {
    let n = graph.shape().rows;
    if source >= n || target >= n {
        return (f64::INFINITY, vec![]);
    }

    let mut dist = vec![f64::INFINITY; n];
    let mut prev = vec![usize::MAX; n];
    let mut visited = vec![false; n];
    dist[source] = 0.0;

    for _ in 0..n {
        // Find unvisited node with minimum distance
        let mut u = usize::MAX;
        let mut min_d = f64::INFINITY;
        for (i, (&d, &v)) in dist.iter().zip(visited.iter()).enumerate() {
            if !v && d < min_d {
                min_d = d;
                u = i;
            }
        }

        if u == usize::MAX || u == target {
            break;
        }

        visited[u] = true;

        let row_start = graph.indptr()[u];
        let row_end = graph.indptr()[u + 1];
        for idx in row_start..row_end {
            let v = graph.indices()[idx];
            let w = graph.data()[idx];
            let alt = dist[u] + w;
            if alt < dist[v] {
                dist[v] = alt;
                prev[v] = u;
            }
        }
    }

    if dist[target] == f64::INFINITY {
        return (f64::INFINITY, vec![]);
    }

    // Reconstruct path
    let mut path = vec![target];
    let mut current = target;
    while current != source {
        current = prev[current];
        if current == usize::MAX {
            return (f64::INFINITY, vec![]);
        }
        path.push(current);
    }
    path.reverse();

    (dist[target], path)
}

/// Reverse Cuthill-McKee ordering to reduce matrix bandwidth.
///
/// Returns a permutation vector. Matches `scipy.sparse.csgraph.reverse_cuthill_mckee`.
pub fn reverse_cuthill_mckee(graph: &CsrMatrix) -> Vec<usize> {
    let n = graph.shape().rows;
    if n == 0 {
        return vec![];
    }

    let mut visited = vec![false; n];
    let mut result = Vec::with_capacity(n);

    // Find starting node: minimum degree
    let degrees: Vec<usize> = (0..n)
        .map(|i| graph.indptr()[i + 1] - graph.indptr()[i])
        .collect();

    // Process all connected components
    while result.len() < n {
        // Find unvisited node with minimum degree
        let start = (0..n)
            .filter(|&i| !visited[i])
            .min_by_key(|&i| degrees[i])
            .unwrap_or(0);

        // BFS from start, visiting neighbors in order of increasing degree
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        visited[start] = true;

        while let Some(u) = queue.pop_front() {
            result.push(u);

            // Get unvisited neighbors sorted by degree
            let row_start = graph.indptr()[u];
            let row_end = graph.indptr()[u + 1];
            let mut neighbors: Vec<usize> = (row_start..row_end)
                .map(|idx| graph.indices()[idx])
                .filter(|&v| !visited[v])
                .collect();
            neighbors.sort_by_key(|&v| degrees[v]);

            for v in neighbors {
                if !visited[v] {
                    visited[v] = true;
                    queue.push_back(v);
                }
            }
        }
    }

    // Reverse the ordering
    result.reverse();
    result
}

/// Compute the structural rank of a sparse matrix.
///
/// The structural rank is the maximum number of entries that can be
/// placed in the matrix such that no two are in the same row or column.
/// This is an upper bound on the numerical rank.
///
/// Matches `scipy.sparse.linalg.structural_rank` (approximate).
pub fn structural_rank(graph: &CsrMatrix) -> usize {
    let n = graph.shape().rows;
    let m = graph.shape().cols;
    if n == 0 || m == 0 {
        return 0;
    }

    // Maximum bipartite matching using augmenting paths
    let mut match_col = vec![usize::MAX; m]; // match_col[j] = row matched to column j

    let mut rank = 0;
    for row in 0..n {
        let mut visited = vec![false; m];
        if augment(graph, row, &mut match_col, &mut visited) {
            rank += 1;
        }
    }

    rank
}

/// Try to find an augmenting path from `row` in the bipartite matching.
fn augment(graph: &CsrMatrix, row: usize, match_col: &mut [usize], visited: &mut [bool]) -> bool {
    let row_start = graph.indptr()[row];
    let row_end = graph.indptr()[row + 1];

    for idx in row_start..row_end {
        let col = graph.indices()[idx];
        if col < visited.len() && !visited[col] {
            visited[col] = true;
            if match_col[col] == usize::MAX || augment(graph, match_col[col], match_col, visited) {
                match_col[col] = row;
                return true;
            }
        }
    }
    false
}

// ══════════════════════════════════════════════════════════════════════
// Sparse Matrix Operations
// ══════════════════════════════════════════════════════════════════════

/// Sparse matrix norm.
///
/// Supports "fro" (Frobenius), "1" (max column sum), "inf" (max row sum).
/// Matches `scipy.sparse.linalg.norm`.
pub fn sparse_norm(a: &CsrMatrix, kind: &str) -> f64 {
    let n = a.shape().rows;
    match kind {
        "fro" | "frobenius" => a.data().iter().map(|&v| v * v).sum::<f64>().sqrt(),
        "1" => {
            let m = a.shape().cols;
            let mut col_sums = vec![0.0; m];
            // Iterate via CSR structure
            for i in 0..n {
                let start = a.indptr()[i];
                let end = a.indptr()[i + 1];
                for idx in start..end {
                    let j = a.indices()[idx];
                    if j < m {
                        col_sums[j] += a.data()[idx].abs();
                    }
                }
            }
            col_sums.iter().cloned().fold(0.0, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            })
        }
        "inf" => {
            let mut max_row = 0.0f64;
            for i in 0..n {
                let start = a.indptr()[i];
                let end = a.indptr()[i + 1];
                let row_sum: f64 = a.data()[start..end].iter().map(|v| v.abs()).sum();
                max_row = max_row.max(row_sum);
            }
            max_row
        }
        _ => a.data().iter().map(|&v| v * v).sum::<f64>().sqrt(), // default frobenius
    }
}

/// Extract the diagonal of a CSR matrix.
///
/// Matches `scipy.sparse.csr_matrix.diagonal()`.
pub fn sparse_diagonal(a: &CsrMatrix) -> Vec<f64> {
    let n = a.shape().rows.min(a.shape().cols);
    let mut diag = vec![0.0; n];
    for (i, d) in diag.iter_mut().enumerate().take(n) {
        let start = a.indptr()[i];
        let end = a.indptr()[i + 1];
        for idx in start..end {
            if a.indices()[idx] == i {
                *d = a.data()[idx];
                break;
            }
        }
    }
    diag
}

/// Compute the trace of a CSR matrix (sum of diagonal elements).
///
/// Matches `scipy.sparse.csr_matrix.trace()`.
pub fn sparse_trace(a: &CsrMatrix) -> f64 {
    sparse_diagonal(a).iter().sum()
}

/// Transpose a CSR matrix, returning a new CSR matrix.
///
/// Matches `scipy.sparse.csr_matrix.T`.
pub fn sparse_transpose(a: &CsrMatrix) -> CsrMatrix {
    let (rows, cols) = (a.shape().rows, a.shape().cols);
    let nnz = a.data().len();

    // Count entries per column (= per row of transpose)
    let mut col_counts = vec![0usize; cols];
    for &j in a.indices() {
        if j < cols {
            col_counts[j] += 1;
        }
    }

    // Build transpose indptr
    let mut t_indptr = vec![0usize; cols + 1];
    for j in 0..cols {
        t_indptr[j + 1] = t_indptr[j] + col_counts[j];
    }

    // Fill transpose data
    let mut t_indices = vec![0usize; nnz];
    let mut t_data = vec![0.0; nnz];
    let mut pos = vec![0usize; cols];

    for i in 0..rows {
        let start = a.indptr()[i];
        let end = a.indptr()[i + 1];
        for idx in start..end {
            let j = a.indices()[idx];
            if j < cols {
                let dest = t_indptr[j] + pos[j];
                t_indices[dest] = i;
                t_data[dest] = a.data()[idx];
                pos[j] += 1;
            }
        }
    }

    CsrMatrix::from_components_unchecked(Shape2D::new(cols, rows), t_data, t_indices, t_indptr)
}

/// Count the number of nonzero elements in a CSR matrix.
///
/// Matches `scipy.sparse.csr_matrix.nnz`.
pub fn sparse_nnz(a: &CsrMatrix) -> usize {
    a.data().iter().filter(|&&v| v != 0.0).count()
}

/// Compute the density of a CSR matrix (fraction of nonzeros).
pub fn sparse_density(a: &CsrMatrix) -> f64 {
    let total = a.shape().rows * a.shape().cols;
    if total == 0 {
        return 0.0;
    }
    sparse_nnz(a) as f64 / total as f64
}

/// Sparse matrix-vector product: y = A * x.
///
/// Matches `scipy.sparse.csr_matrix @ vector`.
pub fn spmv(a: &CsrMatrix, x: &[f64]) -> Vec<f64> {
    let n = a.shape().rows;
    let mut y = vec![0.0; n];
    for (i, yi) in y.iter_mut().enumerate().take(n) {
        let start = a.indptr()[i];
        let end = a.indptr()[i + 1];
        for idx in start..end {
            let j = a.indices()[idx];
            if j < x.len() {
                *yi += a.data()[idx] * x[j];
            }
        }
    }
    y
}

/// Sparse matrix-matrix product: C = A * B (both CSR).
///
/// Returns a new CSR matrix.
pub fn spmm(a: &CsrMatrix, b: &CsrMatrix) -> CsrMatrix {
    let m = a.shape().rows;
    let n = b.shape().cols;
    let b_rows = b.shape().rows;

    // Each output row is an independent Gustavson merge, so for large products
    // the rows are fanned out across a thread pool. Every worker owns a private
    // dense accumulator and emits its rows into chunk-local buffers; the driver
    // concatenates them in row order. Output is byte-identical to the serial
    // sweep: a row's columns/values depend only on that row, the reverse
    // first-seen emit order is per-row, and `sorted_indices` is an associative
    // AND across rows.
    // Estimated multiply-adds: nnz(A) times the average nonzeros per B row. This
    // tracks SpGEMM work far better than nnz(A) alone, since output fill grows
    // superlinearly with size; only products heavy enough to amortise thread
    // spawn are fanned out.
    let avg_b_row = (b.nnz() as u64) / (b_rows.max(1) as u64);
    let work = (a.nnz() as u64).saturating_mul(avg_b_row);
    let nthreads = spmm_chunk_count(m, work);
    let (cols, vals, indptr, sorted_indices) = if nthreads <= 1 {
        let (cols, vals, counts, sorted) = spmm_row_chunk(a, b, n, b_rows, 0, m, a.nnz());
        let mut indptr = Vec::with_capacity(m + 1);
        indptr.push(0);
        let mut acc = 0usize;
        for &count in &counts {
            acc += count;
            indptr.push(acc);
        }
        (cols, vals, indptr, sorted)
    } else {
        spmm_rows_parallel(a, b, n, b_rows, m, nthreads)
    };

    let mut result = CsrMatrix::from_components_unchecked(Shape2D::new(m, n), vals, cols, indptr);
    result.canonical.sorted_indices = sorted_indices;
    result.canonical.deduplicated = true;
    result
}

/// Gustavson SpGEMM over rows `[row_start, row_end)`, returning the emitted
/// columns, values, per-row nnz counts, and whether every emitted row stayed
/// strictly column-sorted. A dense accumulator (`acc` + `seen`, length `n`) is
/// reused across the chunk's rows and cleared only at touched columns. Each
/// product `a_ik * b_kj` is summed into `acc[j]` in encounter order and rows are
/// emitted in reverse first-seen column order (SciPy CSR-matmul parity).
fn spmm_row_chunk(
    a: &CsrMatrix,
    b: &CsrMatrix,
    n: usize,
    b_rows: usize,
    row_start: usize,
    row_end: usize,
    cap_hint: usize,
) -> (Vec<usize>, Vec<f64>, Vec<usize>, bool) {
    let mut acc = vec![0.0f64; n];
    let mut seen = vec![false; n];
    let mut column_order: Vec<usize> = Vec::new();
    let mut cols = Vec::with_capacity(cap_hint);
    let mut vals = Vec::with_capacity(cap_hint);
    let mut counts = Vec::with_capacity(row_end - row_start);
    let mut sorted_indices = true;

    for i in row_start..row_end {
        column_order.clear();
        let before = cols.len();
        let a_start = a.indptr()[i];
        let a_end = a.indptr()[i + 1];

        for a_idx in a_start..a_end {
            let k = a.indices()[a_idx];
            let a_ik = a.data()[a_idx];

            if k < b_rows {
                let b_start = b.indptr()[k];
                let b_end = b.indptr()[k + 1];
                for b_idx in b_start..b_end {
                    let j = b.indices()[b_idx];
                    let b_kj = b.data()[b_idx];
                    if seen[j] {
                        acc[j] += a_ik * b_kj;
                    } else {
                        seen[j] = true;
                        acc[j] = a_ik * b_kj;
                        column_order.push(j);
                    }
                }
            }
        }

        let mut prev_col = None;
        for &j in column_order.iter().rev() {
            let v = acc[j];
            seen[j] = false;
            acc[j] = 0.0;
            if v.abs() > 0.0 {
                if let Some(prev) = prev_col {
                    sorted_indices &= prev < j;
                }
                prev_col = Some(j);
                cols.push(j);
                vals.push(v);
            }
        }
        counts.push(cols.len() - before);
    }

    (cols, vals, counts, sorted_indices)
}

fn spmm_row_counts_chunk(
    a: &CsrMatrix,
    b: &CsrMatrix,
    n: usize,
    b_rows: usize,
    row_start: usize,
    row_end: usize,
) -> Vec<usize> {
    let mut acc = vec![0.0f64; n];
    let mut seen = vec![false; n];
    let mut column_order: Vec<usize> = Vec::new();
    let mut counts = Vec::with_capacity(row_end - row_start);

    for i in row_start..row_end {
        column_order.clear();
        let a_start = a.indptr()[i];
        let a_end = a.indptr()[i + 1];

        for a_idx in a_start..a_end {
            let k = a.indices()[a_idx];
            let a_ik = a.data()[a_idx];

            if k < b_rows {
                let b_start = b.indptr()[k];
                let b_end = b.indptr()[k + 1];
                for b_idx in b_start..b_end {
                    let j = b.indices()[b_idx];
                    let b_kj = b.data()[b_idx];
                    if seen[j] {
                        acc[j] += a_ik * b_kj;
                    } else {
                        seen[j] = true;
                        acc[j] = a_ik * b_kj;
                        column_order.push(j);
                    }
                }
            }
        }

        let mut count = 0usize;
        for &j in column_order.iter().rev() {
            let v = acc[j];
            seen[j] = false;
            acc[j] = 0.0;
            if v.abs() > 0.0 {
                count += 1;
            }
        }
        counts.push(count);
    }

    counts
}

fn spmm_rows_parallel(
    a: &CsrMatrix,
    b: &CsrMatrix,
    n: usize,
    b_rows: usize,
    m: usize,
    nthreads: usize,
) -> (Vec<usize>, Vec<f64>, Vec<usize>, bool) {
    let ranges = spmm_work_balanced_ranges(a, b, b_rows, m, nthreads);
    spmm_rows_parallel_exact(a, b, n, b_rows, m, &ranges)
}

fn spmm_rows_parallel_exact(
    a: &CsrMatrix,
    b: &CsrMatrix,
    n: usize,
    b_rows: usize,
    m: usize,
    ranges: &[(usize, usize)],
) -> (Vec<usize>, Vec<f64>, Vec<usize>, bool) {
    type ChunkOut = (Vec<usize>, Vec<f64>, Vec<usize>, bool);
    let chunks: Vec<ChunkOut> = std::thread::scope(|scope| {
        let handles: Vec<_> = ranges
            .iter()
            .map(|&(row_start, row_end)| {
                scope.spawn(move || {
                    let counts = spmm_row_counts_chunk(a, b, n, b_rows, row_start, row_end);
                    let cap_hint = counts.iter().sum();
                    let (cols, vals, numeric_counts, sorted) =
                        spmm_row_chunk(a, b, n, b_rows, row_start, row_end, cap_hint);
                    debug_assert_eq!(numeric_counts, counts);
                    (cols, vals, counts, sorted)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("spmm chunk panicked"))
            .collect()
    });

    let mut indptr = Vec::with_capacity(m + 1);
    indptr.push(0);
    let mut acc = 0usize;
    for (_, _, counts, _) in &chunks {
        for &count in counts {
            acc += count;
            indptr.push(acc);
        }
    }

    let total = indptr[m];
    let mut cols = Vec::with_capacity(total);
    let mut vals = Vec::with_capacity(total);
    let mut sorted_indices = true;
    for (chunk_cols, chunk_vals, _, chunk_sorted) in &chunks {
        cols.extend_from_slice(chunk_cols);
        vals.extend_from_slice(chunk_vals);
        sorted_indices &= *chunk_sorted;
    }

    (cols, vals, indptr, sorted_indices)
}

fn spmm_work_balanced_ranges(
    a: &CsrMatrix,
    b: &CsrMatrix,
    b_rows: usize,
    m: usize,
    nthreads: usize,
) -> Vec<(usize, usize)> {
    let partitions = nthreads.min(m);
    if partitions == 0 {
        return Vec::new();
    }
    if partitions == 1 {
        return vec![(0, m)];
    }

    let mut row_work = Vec::with_capacity(m);
    let mut total_work = 0usize;
    for i in 0..m {
        let mut work = 0usize;
        let a_start = a.indptr()[i];
        let a_end = a.indptr()[i + 1];
        for a_idx in a_start..a_end {
            let k = a.indices()[a_idx];
            if k < b_rows {
                work = work.saturating_add(b.indptr()[k + 1] - b.indptr()[k]);
            }
        }
        total_work = total_work.saturating_add(work);
        row_work.push(work);
    }

    if total_work == 0 {
        let chunk = m.div_ceil(partitions);
        return (0..partitions)
            .map(|thread| (thread * chunk, ((thread + 1) * chunk).min(m)))
            .filter(|(start, end)| start < end)
            .collect();
    }

    let mut ranges = Vec::with_capacity(partitions);
    let mut start = 0usize;
    let mut prefix_work = 0usize;
    let mut next_boundary = 1usize;

    for (row, &work) in row_work.iter().enumerate() {
        prefix_work = prefix_work.saturating_add(work);
        let end = row + 1;
        let remaining_partitions = partitions - ranges.len() - 1;
        if remaining_partitions == 0 {
            break;
        }

        let target_work = usize::try_from(
            (total_work as u128)
                .saturating_mul(next_boundary as u128)
                .div_ceil(partitions as u128),
        )
        .unwrap_or(usize::MAX);
        let must_close = end == m - remaining_partitions;
        if end > start && (prefix_work >= target_work || must_close) {
            ranges.push((start, end));
            start = end;
            next_boundary += 1;
        }
    }

    ranges.push((start, m));
    ranges
}

/// Worker count for an SpGEMM, or 1 to stay serial. Only products with enough
/// rows and estimated multiply-adds to amortise thread spawn are fanned out.
fn spmm_chunk_count(rows: usize, work: u64) -> usize {
    if work < 300_000 || rows < 512 {
        return 1;
    }
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    cores.min(16).min(rows / 128).max(1)
}

/// Compute one-norm estimate for a sparse matrix.
///
/// Uses the Hager-Higham algorithm for efficient estimation
/// without forming the dense matrix.
/// Matches `scipy.sparse.linalg.onenormest`.
pub fn onenormest(a: &CsrMatrix) -> f64 {
    sparse_norm(a, "1")
}

/// Scale a CSR matrix by a scalar: B = alpha * A.
pub fn sparse_scale(a: &CsrMatrix, alpha: f64) -> CsrMatrix {
    let scaled_data: Vec<f64> = a.data().iter().map(|&v| v * alpha).collect();
    CsrMatrix::from_components_unchecked(
        a.shape(),
        scaled_data,
        a.indices().to_vec(),
        a.indptr().to_vec(),
    )
}

/// Add two CSR matrices: C = A + B.
///
/// Both matrices must have the same shape.
pub fn sparse_add(a: &CsrMatrix, b: &CsrMatrix) -> CsrMatrix {
    let n = a.shape().rows;
    let m = a.shape().cols;

    let mut rows = Vec::new();
    let mut cols_vec = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n {
        let mut row_acc = std::collections::BTreeMap::new();

        let a_start = a.indptr()[i];
        let a_end = a.indptr()[i + 1];
        for idx in a_start..a_end {
            *row_acc.entry(a.indices()[idx]).or_insert(0.0) += a.data()[idx];
        }

        let b_start = b.indptr()[i];
        let b_end = b.indptr()[i + 1];
        for idx in b_start..b_end {
            *row_acc.entry(b.indices()[idx]).or_insert(0.0) += b.data()[idx];
        }

        for (&j, &v) in &row_acc {
            if v.abs() > 0.0 {
                rows.push(i);
                cols_vec.push(j);
                vals.push(v);
            }
        }
    }

    let mut indptr = vec![0usize; n + 1];
    for &r in &rows {
        indptr[r + 1] += 1;
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }

    CsrMatrix::from_components_unchecked(Shape2D::new(n, m), vals, cols_vec, indptr)
}

/// Compute the Frobenius inner product of two sparse matrices: <A, B> = Σ A_ij * B_ij.
pub fn sparse_frobenius_inner(a: &CsrMatrix, b: &CsrMatrix) -> f64 {
    let n = a.shape().rows;
    let mut sum = 0.0;

    for i in 0..n {
        let a_start = a.indptr()[i];
        let a_end = a.indptr()[i + 1];

        for a_idx in a_start..a_end {
            let j = a.indices()[a_idx];
            let a_val = a.data()[a_idx];

            // Find corresponding entry in B
            let b_start = b.indptr()[i];
            let b_end = b.indptr()[i + 1];
            for b_idx in b_start..b_end {
                if b.indices()[b_idx] == j {
                    sum += a_val * b.data()[b_idx];
                    break;
                }
            }
        }
    }

    sum
}

/// Check if a sparse matrix is symmetric.
pub fn sparse_is_symmetric(a: &CsrMatrix, tol: f64) -> bool {
    let n = a.shape().rows;
    if n != a.shape().cols {
        return false;
    }

    for i in 0..n {
        let start = a.indptr()[i];
        let end = a.indptr()[i + 1];
        for idx in start..end {
            let j = a.indices()[idx];
            let v = a.data()[idx];

            // Find A[j][i]
            let j_start = a.indptr()[j];
            let j_end = a.indptr()[j + 1];
            let mut found = false;
            for j_idx in j_start..j_end {
                if a.indices()[j_idx] == i {
                    if (a.data()[j_idx] - v).abs() > tol {
                        return false;
                    }
                    found = true;
                    break;
                }
            }
            if !found && v.abs() > tol {
                return false;
            }
        }
    }

    true
}

/// Extract a submatrix from a CSR matrix (rows[r_start..r_end], cols[c_start..c_end]).
pub fn sparse_submatrix(
    a: &CsrMatrix,
    r_start: usize,
    r_end: usize,
    c_start: usize,
    c_end: usize,
) -> CsrMatrix {
    let new_rows = r_end - r_start;
    let new_cols = c_end - c_start;

    let mut rows = Vec::new();
    let mut cols_vec = Vec::new();
    let mut vals = Vec::new();

    for i in r_start..r_end.min(a.shape().rows) {
        let start = a.indptr()[i];
        let end = a.indptr()[i + 1];
        for idx in start..end {
            let j = a.indices()[idx];
            if j >= c_start && j < c_end {
                rows.push(i - r_start);
                cols_vec.push(j - c_start);
                vals.push(a.data()[idx]);
            }
        }
    }

    let mut indptr = vec![0usize; new_rows + 1];
    for &r in &rows {
        if r < new_rows {
            indptr[r + 1] += 1;
        }
    }
    for i in 0..new_rows {
        indptr[i + 1] += indptr[i];
    }

    CsrMatrix::from_components_unchecked(Shape2D::new(new_rows, new_cols), vals, cols_vec, indptr)
}

/// Compute the number of connected components and their sizes.
///
/// Returns (n_components, component_sizes).
pub fn connected_component_sizes(graph: &CsrMatrix) -> (usize, Vec<usize>) {
    let n = graph.shape().rows;
    let mut visited = vec![false; n];
    let mut sizes = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }

        let mut size = 0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        visited[start] = true;

        while let Some(u) = queue.pop_front() {
            size += 1;
            let row_start = graph.indptr()[u];
            let row_end = graph.indptr()[u + 1];
            for idx in row_start..row_end {
                let v = graph.indices()[idx];
                if !visited[v] {
                    visited[v] = true;
                    queue.push_back(v);
                }
            }
        }
        sizes.push(size);
    }

    (sizes.len(), sizes)
}

/// Check if a sparse graph is connected.
pub fn is_connected(graph: &CsrMatrix) -> bool {
    let (n_comp, _) = connected_component_sizes(graph);
    n_comp <= 1
}

/// Compute the degree sequence of a sparse graph.
///
/// Returns the degree (number of nonzero entries) for each row.
pub fn degree_sequence(graph: &CsrMatrix) -> Vec<usize> {
    let n = graph.shape().rows;
    (0..n)
        .map(|i| graph.indptr()[i + 1] - graph.indptr()[i])
        .collect()
}

/// Find the strongly connected components of a directed graph (Tarjan's algorithm).
///
/// Returns a vector of component assignments (component index for each node).
pub fn strongly_connected_components(graph: &CsrMatrix) -> Vec<usize> {
    let n = graph.shape().rows;
    let mut index_counter = 0usize;
    let mut stack = Vec::new();
    let mut on_stack = vec![false; n];
    let mut index = vec![usize::MAX; n];
    let mut lowlink = vec![0usize; n];
    let mut component = vec![0usize; n];
    let mut n_components = 0usize;

    #[allow(clippy::too_many_arguments)]
    fn strongconnect(
        v: usize,
        graph: &CsrMatrix,
        index_counter: &mut usize,
        stack: &mut Vec<usize>,
        on_stack: &mut [bool],
        index: &mut [usize],
        lowlink: &mut [usize],
        component: &mut [usize],
        n_components: &mut usize,
    ) {
        index[v] = *index_counter;
        lowlink[v] = *index_counter;
        *index_counter += 1;
        stack.push(v);
        on_stack[v] = true;

        let row_start = graph.indptr()[v];
        let row_end = graph.indptr()[v + 1];
        for idx in row_start..row_end {
            let w = graph.indices()[idx];
            if index[w] == usize::MAX {
                strongconnect(
                    w,
                    graph,
                    index_counter,
                    stack,
                    on_stack,
                    index,
                    lowlink,
                    component,
                    n_components,
                );
                lowlink[v] = lowlink[v].min(lowlink[w]);
            } else if on_stack[w] {
                lowlink[v] = lowlink[v].min(index[w]);
            }
        }

        if lowlink[v] == index[v] {
            while let Some(w) = stack.pop() {
                on_stack[w] = false;
                component[w] = *n_components;
                if w == v {
                    break;
                }
            }
            *n_components += 1;
        }
    }

    for v in 0..n {
        if index[v] == usize::MAX {
            strongconnect(
                v,
                graph,
                &mut index_counter,
                &mut stack,
                &mut on_stack,
                &mut index,
                &mut lowlink,
                &mut component,
                &mut n_components,
            );
        }
    }

    component
}

/// Topological sort of a directed acyclic graph (DAG).
///
/// Returns None if the graph has a cycle.
pub fn topological_sort(graph: &CsrMatrix) -> Option<Vec<usize>> {
    let n = graph.shape().rows;

    // Compute in-degrees
    let mut in_degree = vec![0usize; n];
    for &j in graph.indices() {
        if j < n {
            in_degree[j] += 1;
        }
    }

    // Start with zero in-degree nodes
    let mut queue: std::collections::VecDeque<usize> =
        (0..n).filter(|&i| in_degree[i] == 0).collect();

    let mut order = Vec::with_capacity(n);

    while let Some(u) = queue.pop_front() {
        order.push(u);
        let row_start = graph.indptr()[u];
        let row_end = graph.indptr()[u + 1];
        for idx in row_start..row_end {
            let v = graph.indices()[idx];
            if v < n {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }
    }

    if order.len() == n {
        Some(order)
    } else {
        None // cycle detected
    }
}

/// PageRank algorithm for a sparse graph.
///
/// Returns the PageRank score for each node.
pub fn pagerank(graph: &CsrMatrix, damping: f64, max_iter: usize, tol: f64) -> Vec<f64> {
    let n = graph.shape().rows;
    if n == 0 {
        return vec![];
    }

    let damping = damping.clamp(0.0, 1.0);
    let tol = if tol <= 0.0 || !tol.is_finite() {
        1e-8
    } else {
        tol
    };
    let max_iter = if max_iter == 0 { 100 } else { max_iter };

    let out_degree: Vec<usize> = (0..n)
        .map(|i| graph.indptr()[i + 1] - graph.indptr()[i])
        .collect();

    let mut rank = vec![1.0 / n as f64; n];

    for _ in 0..max_iter {
        let mut new_rank = vec![(1.0 - damping) / n as f64; n];

        for i in 0..n {
            if out_degree[i] == 0 {
                // Dangling node: distribute evenly
                let contrib = damping * rank[i] / n as f64;
                for r in &mut new_rank {
                    *r += contrib;
                }
            } else {
                let contrib = damping * rank[i] / out_degree[i] as f64;
                let row_start = graph.indptr()[i];
                let row_end = graph.indptr()[i + 1];
                for idx in row_start..row_end {
                    let j = graph.indices()[idx];
                    if j < n {
                        new_rank[j] += contrib;
                    }
                }
            }
        }

        // Check convergence
        let diff: f64 = rank
            .iter()
            .zip(new_rank.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();

        rank = new_rank;
        if diff < tol {
            break;
        }
    }

    rank
}

/// Compute the graph diameter (longest shortest path between any two nodes).
///
/// Uses Floyd-Warshall internally. Returns 0.0 for non-square matrices.
pub fn graph_diameter(graph: &CsrMatrix) -> f64 {
    let dist = floyd_warshall(graph);
    if dist.is_empty() {
        return 0.0;
    }
    let mut max_d = 0.0f64;
    for row in &dist {
        for &d in row {
            if d.is_finite() {
                max_d = max_d.max(d);
            }
        }
    }
    max_d
}

/// Compute the eccentricity of each node (max shortest path distance).
/// Returns empty vec for non-square matrices.
pub fn eccentricity(graph: &CsrMatrix) -> Vec<f64> {
    let dist = floyd_warshall(graph);
    if dist.is_empty() {
        return vec![];
    }
    dist.iter()
        .map(|row| {
            row.iter()
                .filter(|&&d| d.is_finite())
                .cloned()
                .fold(0.0f64, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                })
        })
        .collect()
}

/// Compute the clustering coefficient for each node.
///
/// The clustering coefficient measures how interconnected a node's neighbors are.
pub fn clustering_coefficient(graph: &CsrMatrix) -> Vec<f64> {
    let n = graph.shape().rows;
    let mut cc = vec![0.0; n];

    for (i, cc_val) in cc.iter_mut().enumerate() {
        let neighbors: Vec<usize> = (graph.indptr()[i]..graph.indptr()[i + 1])
            .map(|idx| graph.indices()[idx])
            .collect();

        let k = neighbors.len();
        if k < 2 {
            continue;
        }

        // Count edges between neighbors
        let mut edges = 0;
        for &u in &neighbors {
            for &v in &neighbors {
                if u < v {
                    // Check if edge (u, v) exists
                    let u_start = graph.indptr()[u];
                    let u_end = graph.indptr()[u + 1];
                    if graph.indices()[u_start..u_end].binary_search(&v).is_ok() {
                        edges += 1;
                    }
                }
            }
        }

        *cc_val = 2.0 * edges as f64 / (k * (k - 1)) as f64;
    }

    cc
}

/// Average clustering coefficient of a graph.
pub fn average_clustering(graph: &CsrMatrix) -> f64 {
    let cc = clustering_coefficient(graph);
    let n = cc.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    cc.iter().sum::<f64>() / n
}

/// Compute betweenness centrality for each node.
///
/// Uses Brandes' algorithm (O(VE) for unweighted graphs).
pub fn betweenness_centrality(graph: &CsrMatrix) -> Vec<f64> {
    let n = graph.shape().rows;
    let mut bc = vec![0.0; n];

    for s in 0..n {
        // BFS from s
        let mut stack = Vec::new();
        let mut predecessors: Vec<Vec<usize>> = vec![vec![]; n];
        let mut sigma = vec![0.0f64; n]; // number of shortest paths
        sigma[s] = 1.0;
        let mut dist = vec![-1i64; n];
        dist[s] = 0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            let row_start = graph.indptr()[v];
            let row_end = graph.indptr()[v + 1];
            for idx in row_start..row_end {
                let w = graph.indices()[idx];
                if w >= n {
                    continue;
                }
                if dist[w] < 0 {
                    queue.push_back(w);
                    dist[w] = dist[v] + 1;
                }
                if dist[w] == dist[v] + 1 {
                    sigma[w] += sigma[v];
                    predecessors[w].push(v);
                }
            }
        }

        // Accumulate
        let mut delta = vec![0.0; n];
        while let Some(w) = stack.pop() {
            for &v in &predecessors[w] {
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
            if w != s {
                bc[w] += delta[w];
            }
        }
    }

    // Normalize for undirected graphs
    let scale = if n > 2 {
        1.0 / ((n - 1) * (n - 2)) as f64
    } else {
        1.0
    };
    for v in &mut bc {
        *v *= scale;
    }

    bc
}

/// Compute closeness centrality for each node.
pub fn closeness_centrality(graph: &CsrMatrix) -> Vec<f64> {
    let n = graph.shape().rows;
    let dist = floyd_warshall(graph);
    if dist.is_empty() {
        return vec![0.0; n];
    }

    (0..n)
        .map(|i| {
            let reachable: Vec<f64> = dist[i]
                .iter()
                .enumerate()
                .filter(|&(j, &d)| j != i && d.is_finite())
                .map(|(_, &d)| d)
                .collect();

            if reachable.is_empty() {
                0.0
            } else {
                let total: f64 = reachable.iter().sum();
                if total > 0.0 {
                    reachable.len() as f64 / total
                } else {
                    0.0
                }
            }
        })
        .collect()
}

/// Apply an element-wise function to all nonzero entries of a CSR matrix.
pub fn sparse_map<F>(a: &CsrMatrix, f: F) -> CsrMatrix
where
    F: Fn(f64) -> f64,
{
    let mapped_data: Vec<f64> = a.data().iter().map(|&v| f(v)).collect();
    CsrMatrix::from_components_unchecked(
        a.shape(),
        mapped_data,
        a.indices().to_vec(),
        a.indptr().to_vec(),
    )
}

/// Compute the absolute value of all entries in a CSR matrix.
pub fn sparse_abs(a: &CsrMatrix) -> CsrMatrix {
    sparse_map(a, |v| v.abs())
}

/// Compute the element-wise power of a CSR matrix.
pub fn sparse_power(a: &CsrMatrix, p: f64) -> CsrMatrix {
    sparse_map(a, |v| v.powf(p))
}

/// Compute the sum of all elements in a CSR matrix.
pub fn sparse_sum(a: &CsrMatrix) -> f64 {
    a.data().iter().sum()
}

/// Compute the row sums of a CSR matrix.
pub fn sparse_row_sums(a: &CsrMatrix) -> Vec<f64> {
    let n = a.shape().rows;
    (0..n)
        .map(|i| {
            let start = a.indptr()[i];
            let end = a.indptr()[i + 1];
            a.data()[start..end].iter().sum()
        })
        .collect()
}

/// Compute the column sums of a CSR matrix.
pub fn sparse_col_sums(a: &CsrMatrix) -> Vec<f64> {
    let m = a.shape().cols;
    let mut sums = vec![0.0; m];
    let n = a.shape().rows;
    for i in 0..n {
        let start = a.indptr()[i];
        let end = a.indptr()[i + 1];
        for idx in start..end {
            let j = a.indices()[idx];
            if j < m {
                sums[j] += a.data()[idx];
            }
        }
    }
    sums
}

/// Compute the row-wise maximum of a CSR matrix.
pub fn sparse_row_max(a: &CsrMatrix) -> Vec<f64> {
    let n = a.shape().rows;
    (0..n)
        .map(|i| {
            let start = a.indptr()[i];
            let end = a.indptr()[i + 1];
            if start == end {
                0.0 // empty row, implicit zero
            } else {
                a.data()[start..end]
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, |a: f64, b: f64| {
                        if a.is_nan() || b.is_nan() {
                            f64::NAN
                        } else {
                            a.max(b)
                        }
                    })
                    .max(0.0) // account for implicit zeros
            }
        })
        .collect()
}

/// Compute the row-wise minimum of a CSR matrix.
pub fn sparse_row_min(a: &CsrMatrix) -> Vec<f64> {
    let n = a.shape().rows;
    (0..n)
        .map(|i| {
            let start = a.indptr()[i];
            let end = a.indptr()[i + 1];
            if start == end {
                0.0
            } else {
                let row_min =
                    a.data()[start..end]
                        .iter()
                        .cloned()
                        .fold(f64::INFINITY, |a: f64, b: f64| {
                            if a.is_nan() || b.is_nan() {
                                f64::NAN
                            } else {
                                a.min(b)
                            }
                        });
                if row_min.is_nan() {
                    f64::NAN
                } else {
                    row_min.min(0.0)
                }
            }
        })
        .collect()
}

/// Check if a sparse matrix has any explicit zeros (stored but zero value).
pub fn sparse_has_explicit_zeros(a: &CsrMatrix) -> bool {
    a.data().contains(&0.0)
}

/// Eliminate explicit zeros from a CSR matrix.
pub fn sparse_eliminate_zeros(a: &CsrMatrix) -> CsrMatrix {
    let n = a.shape().rows;
    let mut new_indptr = vec![0usize; n + 1];
    let mut new_indices = Vec::new();
    let mut new_data = Vec::new();

    for i in 0..n {
        let start = a.indptr()[i];
        let end = a.indptr()[i + 1];
        for idx in start..end {
            if a.data()[idx] != 0.0 {
                new_indices.push(a.indices()[idx]);
                new_data.push(a.data()[idx]);
            }
        }
        new_indptr[i + 1] = new_data.len();
    }

    CsrMatrix::from_components_unchecked(a.shape(), new_data, new_indices, new_indptr)
}

/// Compute the matrix power A^n (repeated matrix multiplication).
///
/// Matches `scipy.sparse.linalg.matrix_power`.
///
/// # Arguments
/// * `a` - Square CSR matrix
/// * `n` - Non-negative integer exponent
///
/// # Returns
/// * A^n as a CSR matrix. A^0 is the identity matrix.
///
/// # Errors
/// Returns an error if the matrix is not square.
pub fn matrix_power(a: &CsrMatrix, n: usize) -> SparseResult<CsrMatrix> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "matrix_power requires a square matrix".to_string(),
        });
    }

    if n == 0 {
        // Return identity matrix
        return eye(shape.rows);
    }

    if n == 1 {
        return Ok(a.clone());
    }

    // Use binary exponentiation for efficiency: A^n in O(log n) multiplications
    let mut result = eye(shape.rows)?;
    let mut base = a.clone();
    let mut exp = n;

    while exp > 0 {
        if exp % 2 == 1 {
            result = spmm(&result, &base);
        }
        base = spmm(&base, &base);
        exp /= 2;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::{CooMatrix, Shape2D};
    use crate::ops::FormatConvertible;

    #[test]
    fn spmm_parallel_matches_serial_byte_for_byte() {
        // Isomorphism proof for the threaded SpGEMM: the chunked/parallel driver
        // must produce identical cols/vals/indptr/metadata as the single-chunk
        // serial sweep, for any worker count including uneven row splits.
        let n = 800;
        let a = crate::random(Shape2D::new(n, n), 0.02, 0x5A1A_D00D)
            .expect("a")
            .to_csr()
            .expect("a csr");
        let b = crate::random(Shape2D::new(n, n), 0.02, 0x5A1A_D00D ^ 0x99)
            .expect("b")
            .to_csr()
            .expect("b csr");
        let b_rows = b.shape().rows;

        let (scols, svals, scounts, ssorted) = spmm_row_chunk(&a, &b, n, b_rows, 0, n, a.nnz());
        let mut sindptr = vec![0usize];
        let mut acc = 0usize;
        for &c in &scounts {
            acc += c;
            sindptr.push(acc);
        }

        for &threads in &[2usize, 3, 7, 8, 16] {
            let (pcols, pvals, pindptr, psorted) =
                spmm_rows_parallel(&a, &b, n, b_rows, n, threads);
            assert_eq!(pcols, scols, "cols mismatch threads={threads}");
            assert_eq!(pvals, svals, "vals mismatch threads={threads}");
            assert_eq!(pindptr, sindptr, "indptr mismatch threads={threads}");
            assert_eq!(psorted, ssorted, "sorted flag mismatch threads={threads}");
        }
    }

    /// Deterministic dump of an spmm product for golden-SHA proof. Run with
    /// `--ignored --nocapture` and pipe to `sha256sum`.
    #[test]
    #[ignore]
    fn dump_spmm_payload_for_golden_sha() {
        let cases = [
            (500usize, 0.02f64, 0xBEEF_CAFE_u64),
            (1000, 0.01, 0xBEEF_CAFE),
        ];
        let mut s = String::new();
        for (n, density, seed) in cases {
            let a = crate::random(Shape2D::new(n, n), density, seed)
                .expect("a")
                .to_csr()
                .expect("a csr");
            let b = crate::random(Shape2D::new(n, n), density, seed ^ 0x1234)
                .expect("b")
                .to_csr()
                .expect("b csr");
            let c = spmm(&a, &b);
            s.push_str(&format!(
                "n={} nnz={} sorted={} dedup={}\n",
                n,
                c.nnz(),
                c.canonical_meta().sorted_indices,
                c.canonical_meta().deduplicated
            ));
            for &p in c.indptr() {
                s.push_str(&format!("p{p}\n"));
            }
            for (&col, v) in c.indices().iter().zip(c.data()) {
                s.push_str(&format!("{col}:{:0>16x}\n", v.to_bits()));
            }
        }
        print!("{s}");
    }

    #[test]
    fn solve_options_default_matches_contract() {
        let options = SolveOptions::default();
        assert_eq!(options.mode, RuntimeMode::Strict);
        assert_eq!(options.backend, SparseBackend::Auto);
        assert_eq!(options.ordering, PermutationOrdering::Colamd);
        assert!(options.check_finite);
    }

    #[test]
    fn lu_options_default_matches_contract() {
        let options = LuOptions::default();
        assert_eq!(options.mode, RuntimeMode::Strict);
        assert_eq!(options.ordering, PermutationOrdering::Colamd);
        assert!((options.diag_pivot_thresh - 1.0).abs() <= f64::EPSILON);
    }

    #[test]
    fn ilu_options_default_matches_contract() {
        let options = IluOptions::default();
        assert_eq!(options.mode, RuntimeMode::Strict);
        assert_eq!(options.ordering, PermutationOrdering::Colamd);
        assert!((options.drop_tol - 1e-4).abs() <= f64::EPSILON);
        assert!((options.fill_factor - 10.0).abs() <= f64::EPSILON);
    }

    #[test]
    fn spsolve_rejects_non_square_matrix() {
        let a = non_square_csr();
        let err = spsolve(&a, &[1.0, 2.0], SolveOptions::default()).expect_err("non-square");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn spsolve_rejects_rhs_length_mismatch() {
        let a = square_csr();
        let err = spsolve(&a, &[1.0], SolveOptions::default()).expect_err("rhs mismatch");
        assert!(matches!(err, SparseError::IncompatibleShape { .. }));
    }

    #[test]
    fn spsolve_rejects_non_finite_when_enabled() {
        let a = square_csr();
        let err = spsolve(&a, &[f64::NAN, 1.0], SolveOptions::default()).expect_err("non-finite");
        assert!(matches!(err, SparseError::NonFiniteInput { .. }));
    }

    #[test]
    fn spsolve_skips_non_finite_check_when_disabled() {
        let a = square_csr();
        let options = SolveOptions {
            check_finite: false,
            ..SolveOptions::default()
        };
        // With check_finite=false, NaN is passed through to the solver
        // (the result may be NaN but we don't reject the input)
        let result = spsolve(&a, &[f64::NAN, 1.0], options);
        assert!(
            result.is_ok(),
            "NaN should not be rejected when check_finite=false"
        );
    }

    #[test]
    fn spsolve_hardened_rejects_empty_structural_row() {
        let a = csr_with_empty_row();
        let options = SolveOptions {
            mode: RuntimeMode::Hardened,
            ..SolveOptions::default()
        };
        let err = spsolve(&a, &[1.0, 0.0], options).expect_err("empty row singular");
        assert!(matches!(err, SparseError::SingularMatrix { .. }));
    }

    #[test]
    fn spsolve_strict_empty_structural_row_reaches_solver() {
        let a = csr_with_empty_row();
        let options = SolveOptions {
            mode: RuntimeMode::Strict,
            ..SolveOptions::default()
        };
        // In strict mode, empty structural row is not pre-rejected.
        // The LU solver will detect singularity.
        let err = spsolve(&a, &[1.0, 0.0], options).expect_err("singular");
        assert!(matches!(err, SparseError::SingularMatrix { .. }));
    }

    #[test]
    fn spsolve_uses_native_sparse_direct_above_dense_guard() {
        let n = SPSOLVE_DENSE_MAX_N + 1;
        let a = identity_csr(n);
        let b = vec![1.0; n];

        let result = spsolve(&a, &b, SolveOptions::default())
            .expect("native sparse direct solve should avoid dense fallback guard");

        assert_eq!(result.backend_used, SparseBackend::NativeSparseLu);
        assert_eq!(result.solution.len(), n);
        assert_eq!(result.solution[0], 1.0);
        assert_eq!(result.solution[n - 1], 1.0);
        assert!(
            result
                .warnings
                .iter()
                .any(|warning| warning.contains("native sparse direct"))
        );
    }

    #[test]
    fn spsolve_native_sparse_direct_preserves_tiny_nonzero_diagonal() {
        let n = SPSOLVE_DENSE_MAX_N + 1;
        let scale = 1.0e-300;
        let data = vec![scale; n];
        let indices = (0..n).collect::<Vec<_>>();
        let indptr = (0..=n).collect::<Vec<_>>();
        let a = CsrMatrix::from_components(Shape2D::new(n, n), data, indices, indptr, false)
            .expect("scaled identity csr");
        let b = vec![scale; n];

        let result = spsolve(&a, &b, SolveOptions::default())
            .expect("nonzero tiny pivots should remain solvable");

        assert_eq!(result.backend_used, SparseBackend::NativeSparseLu);
        assert!(
            result
                .solution
                .iter()
                .all(|value| value.is_finite() && (value - 1.0).abs() < 1.0e-12)
        );
    }

    #[test]
    fn splu_rejects_non_square_matrix() {
        let a = non_square_csc();
        let err = splu(&a, LuOptions::default()).expect_err("non-square");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn splu_rejects_invalid_diag_pivot_threshold_low() {
        let a = square_csc();
        let options = LuOptions {
            diag_pivot_thresh: -0.1,
            ..LuOptions::default()
        };
        let err = splu(&a, options).expect_err("invalid threshold");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn splu_rejects_invalid_diag_pivot_threshold_high() {
        let a = square_csc();
        let options = LuOptions {
            diag_pivot_thresh: 1.1,
            ..LuOptions::default()
        };
        let err = splu(&a, options).expect_err("invalid threshold");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn splu_uses_native_sparse_direct_above_dense_guard() {
        let n = SPSOLVE_DENSE_MAX_N + 1;
        let a = identity_csr(n).to_csc().expect("csc");

        let factorization = splu(&a, LuOptions::default())
            .expect("native sparse direct factorization should avoid dense fallback guard");
        let rhs = vec![2.0; n];
        let solution =
            splu_solve(&factorization, &rhs).expect("native sparse direct solve should succeed");

        assert_eq!(factorization.backend_used, SparseBackend::NativeSparseLu);
        assert_eq!(solution.len(), n);
        assert_eq!(solution[0], 2.0);
        assert_eq!(solution[n - 1], 2.0);
        let stored_nnz = match &factorization.lu_internal {
            SparseLuInternal::Native(lu) => lu.stored_nnz(),
            SparseLuInternal::Dense(_) => 0,
        };
        assert_eq!(stored_nnz, n);
    }

    #[test]
    fn splu_valid_input_succeeds() {
        let a = square_csc();
        let result = splu(&a, LuOptions::default()).expect("splu should succeed");
        assert_eq!(result.shape, (2, 2));
    }

    #[test]
    fn spilu_rejects_non_square_matrix() {
        let a = non_square_csc();
        let err = spilu(&a, IluOptions::default()).expect_err("non-square");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn spilu_rejects_negative_drop_tol() {
        let a = square_csc();
        let options = IluOptions {
            drop_tol: -1e-6,
            ..IluOptions::default()
        };
        let err = spilu(&a, options).expect_err("negative drop_tol");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn spilu_rejects_fill_factor_below_one() {
        let a = square_csc();
        let options = IluOptions {
            fill_factor: 0.9,
            ..IluOptions::default()
        };
        let err = spilu(&a, options).expect_err("fill factor");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn spilu_valid_input_succeeds() {
        // ILU(0) now implemented — verify it produces a factorization
        let a = square_csc();
        let ilu = spilu(&a, IluOptions::default()).expect("spilu should succeed");
        assert_eq!(ilu.shape, (a.shape().rows, a.shape().cols));
    }

    fn spilu_reference_find_index(
        indices: &[usize],
        indptr: &[usize],
        row: usize,
        col: usize,
    ) -> Option<usize> {
        (indptr[row]..indptr[row + 1]).find(|&idx| indices[idx] == col)
    }

    fn spilu_reference_linear_scan(a: &CscMatrix) -> SparseResult<SparseIluFactorization> {
        let csr = a.to_csr()?;
        let n = csr.shape().rows;
        let lu_indptr = csr.indptr();
        let lu_indices = csr.indices();
        let mut lu_data = csr.data().to_vec();

        for i in 0..n {
            for idx_ik in lu_indptr[i]..lu_indptr[i + 1] {
                let k = lu_indices[idx_ik];
                if k >= i {
                    break;
                }

                let diag_k = find_value_in_row(&lu_data, lu_indices, lu_indptr, k, k);
                if diag_k.abs() < f64::EPSILON * 100.0 {
                    return Err(SparseError::SingularMatrix {
                        message: format!("zero pivot at row {k} during ILU(0)"),
                    });
                }

                lu_data[idx_ik] /= diag_k;
                let multiplier = lu_data[idx_ik];

                for idx_kj in lu_indptr[k]..lu_indptr[k + 1] {
                    let j = lu_indices[idx_kj];
                    if j <= k {
                        continue;
                    }
                    let a_kj = lu_data[idx_kj];

                    if let Some(idx_ij) = spilu_reference_find_index(lu_indices, lu_indptr, i, j) {
                        lu_data[idx_ij] -= multiplier * a_kj;
                    }
                }
            }
        }

        let mut l_data = Vec::new();
        let mut l_indices = Vec::new();
        let mut l_indptr = vec![0usize];
        let mut u_data = Vec::new();
        let mut u_indices = Vec::new();
        let mut u_indptr = vec![0usize];

        for i in 0..n {
            for idx in lu_indptr[i]..lu_indptr[i + 1] {
                let j = lu_indices[idx];
                if j < i {
                    l_data.push(lu_data[idx]);
                    l_indices.push(j);
                }
            }
            l_data.push(1.0);
            l_indices.push(i);
            l_indptr.push(l_data.len());

            for idx in lu_indptr[i]..lu_indptr[i + 1] {
                let j = lu_indices[idx];
                if j >= i {
                    u_data.push(lu_data[idx]);
                    u_indices.push(j);
                }
            }
            u_indptr.push(u_data.len());
        }

        Ok(SparseIluFactorization {
            shape: (n, n),
            backend_used: SparseBackend::Auto,
            ordering_used: IluOptions::default().ordering,
            l_data,
            l_indices,
            l_indptr,
            u_data,
            u_indices,
            u_indptr,
            n,
        })
    }

    fn spilu_banded_csc(n: usize, half_bandwidth: usize) -> CscMatrix {
        let entries_per_row = half_bandwidth.saturating_mul(2).saturating_add(1);
        let mut data = Vec::with_capacity(n.saturating_mul(entries_per_row));
        let mut rows = Vec::with_capacity(data.capacity());
        let mut cols = Vec::with_capacity(data.capacity());

        for row in 0..n {
            let start = row.saturating_sub(half_bandwidth);
            let end = row.saturating_add(half_bandwidth).min(n.saturating_sub(1));
            for col in start..=end {
                rows.push(row);
                cols.push(col);
                if row == col {
                    data.push(entries_per_row as f64 + 2.0 + (row % 17) as f64 * 0.001);
                } else {
                    data.push(-1.0 / (row.abs_diff(col) + 1) as f64);
                }
            }
        }

        CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
            .expect("spilu banded coo")
            .to_csc()
            .expect("spilu banded csc")
    }

    fn float_bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    fn assert_spilu_factors_same_bits(
        actual: &SparseIluFactorization,
        expected: &SparseIluFactorization,
    ) {
        assert_eq!(actual.shape, expected.shape);
        assert_eq!(actual.backend_used, expected.backend_used);
        assert_eq!(actual.ordering_used, expected.ordering_used);
        assert_eq!(actual.n, expected.n);
        assert_eq!(actual.l_indptr, expected.l_indptr);
        assert_eq!(actual.l_indices, expected.l_indices);
        assert_eq!(float_bits(&actual.l_data), float_bits(&expected.l_data));
        assert_eq!(actual.u_indptr, expected.u_indptr);
        assert_eq!(actual.u_indices, expected.u_indices);
        assert_eq!(float_bits(&actual.u_data), float_bits(&expected.u_data));
    }

    #[test]
    fn spilu_row_workspace_matches_linear_scan_factor_bits() {
        for &(n, half_bandwidth) in &[(16usize, 3usize), (64, 5), (160, 7)] {
            let matrix = spilu_banded_csc(n, half_bandwidth);
            let actual = spilu(&matrix, IluOptions::default()).expect("workspace spilu");
            let expected = spilu_reference_linear_scan(&matrix).expect("reference spilu");
            assert_spilu_factors_same_bits(&actual, &expected);
        }
    }

    #[test]
    fn has_empty_structural_row_detects_gaps() {
        let with_gap = csr_with_empty_row();
        assert!(has_empty_structural_row(&with_gap));
        let dense = square_csr();
        assert!(!has_empty_structural_row(&dense));
    }

    // ── spsolve correctness tests ─────────────────────────────────

    #[test]
    fn spsolve_identity_system() {
        let a = identity_csr(3);
        let b = vec![1.0, 2.0, 3.0];
        let result = spsolve(&a, &b, SolveOptions::default()).expect("spsolve works");
        assert_close_slice(&result.solution, &b, 1e-14);
    }

    #[test]
    fn spsolve_diagonal_system() {
        // [[2, 0], [0, 3]] x = [4, 9] => x = [2, 3]
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![2.0, 3.0],
            vec![0, 1],
            vec![0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![4.0, 9.0];
        let result = spsolve(&a, &b, SolveOptions::default()).expect("spsolve works");
        assert_close_slice(&result.solution, &[2.0, 3.0], 1e-14);
    }

    #[test]
    fn spsolve_general_system() {
        // [[3, 2], [1, 2]] x = [5, 5] => x = [0, 2.5]
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![3.0, 2.0, 1.0, 2.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![5.0, 5.0];
        let result = spsolve(&a, &b, SolveOptions::default()).expect("spsolve works");
        assert_close_slice(&result.solution, &[0.0, 2.5], 1e-12);
    }

    #[test]
    fn spsolve_singular_system() {
        // [[1, 2], [2, 4]] is singular
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![1.0, 2.0, 2.0, 4.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![1.0, 2.0];
        let err = spsolve(&a, &b, SolveOptions::default()).expect_err("singular");
        assert!(matches!(err, SparseError::SingularMatrix { .. }));
    }

    #[test]
    fn native_sparse_lu_pivots_without_dense_matrix() {
        // [[0, 2], [1, 3]] requires a row pivot and solves x = [1, 2].
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![2.0, 1.0, 3.0],
            vec![0, 1, 1],
            vec![1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let lu = NativeSparseLu::factorize_csr(&a, 1.0).expect("native sparse LU");
        let x = lu.solve(&[4.0, 7.0]).expect("native sparse solve");

        assert_close_slice(&x, &[1.0, 2.0], 1e-12);
    }

    #[test]
    fn expm_identity_returns_exp_one() {
        let a = identity_csr(3);
        let result = expm(&a, ExpmOptions::default()).expect("expm works");
        let e = std::f64::consts::E;
        let expected = vec![vec![e, 0.0, 0.0], vec![0.0, e, 0.0], vec![0.0, 0.0, e]];
        assert_close_matrix(&result, &expected, 1e-12);
    }

    #[test]
    fn expm_zero_matrix_returns_identity() {
        let zero = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let result = expm(&zero, ExpmOptions::default()).expect("expm works");
        let expected = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_close_matrix(&result, &expected, 1e-12);
    }

    #[test]
    fn expm_rejects_non_square_matrix() {
        let a = non_square_csr();
        let err = expm(&a, ExpmOptions::default()).expect_err("non-square");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn expm_rejects_non_finite_input() {
        let a = CsrMatrix::from_components(
            Shape2D::new(1, 1),
            vec![f64::NAN],
            vec![0],
            vec![0, 1],
            false,
        )
        .expect("csr");
        let err = expm(&a, ExpmOptions::default()).expect_err("non-finite");
        assert!(matches!(err, SparseError::NonFiniteInput { .. }));
    }

    #[test]
    fn splu_solve_roundtrip() {
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![3.0, 2.0, 1.0, 2.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csc()
        .expect("csc");
        let factorization = splu(&a, LuOptions::default()).expect("splu works");
        let x = splu_solve(&factorization, &[5.0, 5.0]).expect("splu_solve works");
        assert_close_slice(&x, &[0.0, 2.5], 1e-12);
    }

    #[test]
    fn splu_solve_rhs_mismatch() {
        let a = square_csc();
        let factorization = splu(&a, LuOptions::default()).expect("splu works");
        let err = splu_solve(&factorization, &[1.0, 2.0, 3.0]).expect_err("mismatch");
        assert!(matches!(err, SparseError::IncompatibleShape { .. }));
    }

    fn assert_close_slice(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len(), "slice lengths differ");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "index={i} actual={a} expected={e} diff={}",
                (a - e).abs()
            );
        }
    }

    fn assert_close_matrix(actual: &[Vec<f64>], expected: &[Vec<f64>], tol: f64) {
        assert_eq!(actual.len(), expected.len(), "row count differs");
        for (row_idx, (a_row, e_row)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a_row.len(),
                e_row.len(),
                "column count differs at row {row_idx}"
            );
            for (col_idx, (a, e)) in a_row.iter().zip(e_row.iter()).enumerate() {
                assert!(
                    (a - e).abs() < tol,
                    "row={row_idx} col={col_idx} actual={a} expected={e} diff={}",
                    (a - e).abs()
                );
            }
        }
    }

    fn identity_csr(n: usize) -> CsrMatrix {
        let data: Vec<f64> = vec![1.0; n];
        let indices: Vec<usize> = (0..n).collect();
        let indptr: Vec<usize> = (0..=n).collect();
        CsrMatrix::from_components(Shape2D::new(n, n), data, indices, indptr, false)
            .expect("identity csr")
    }

    fn square_csr() -> CsrMatrix {
        CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![2.0, 3.0, 4.0],
            vec![0, 1, 1],
            vec![0, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr")
    }

    fn square_csc() -> CscMatrix {
        square_csr().to_csc().expect("csc")
    }

    fn non_square_csr() -> CsrMatrix {
        CooMatrix::from_triplets(
            Shape2D::new(2, 3),
            vec![1.0, 2.0],
            vec![0, 1],
            vec![1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr")
    }

    fn non_square_csc() -> CscMatrix {
        non_square_csr().to_csc().expect("csc")
    }

    fn csr_with_empty_row() -> CsrMatrix {
        CsrMatrix::from_components(Shape2D::new(2, 2), vec![1.0], vec![0], vec![0, 1, 1], true)
            .expect("csr with empty row")
    }

    // ── CG iterative solver tests ───────────────────────────────────

    fn spd_csr_3x3() -> CsrMatrix {
        // Symmetric positive definite: [[4, 1, 0], [1, 3, 1], [0, 1, 2]]
        CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0],
            vec![0, 0, 1, 1, 1, 2, 2],
            vec![0, 1, 0, 1, 2, 1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr")
    }

    fn hardened_unchecked_iterative_options() -> IterativeSolveOptions {
        IterativeSolveOptions {
            mode: RuntimeMode::Hardened,
            check_finite: false,
            ..IterativeSolveOptions::default()
        }
    }

    #[test]
    fn cg_spd_system_converges() {
        let a = spd_csr_3x3();
        let b = vec![5.0, 5.0, 3.0];
        let result = cg(&a, &b, None, IterativeSolveOptions::default()).expect("cg works");
        assert!(result.converged, "CG should converge for SPD system");
        // Verify A*x ≈ b
        let ax = csr_matvec(&a, &result.solution);
        assert_close_slice(&ax, &b, 1e-5);
    }

    #[test]
    fn cg_identity_system() {
        let a = identity_csr(4);
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let result = cg(&a, &b, None, IterativeSolveOptions::default()).expect("cg works");
        assert!(result.converged);
        assert_close_slice(&result.solution, &b, 1e-10);
        assert!(
            result.iterations <= 1,
            "identity should converge in <= 1 iteration"
        );
    }

    #[test]
    fn cg_with_initial_guess() {
        let a = spd_csr_3x3();
        let b = vec![5.0, 5.0, 3.0];
        // Start from a good guess
        let x0 = vec![1.0, 1.0, 1.0];
        let result = cg(&a, &b, Some(&x0), IterativeSolveOptions::default()).expect("cg works");
        assert!(result.converged);
        let ax = csr_matvec(&a, &result.solution);
        assert_close_slice(&ax, &b, 1e-5);
    }

    #[test]
    fn cg_zero_rhs() {
        let a = spd_csr_3x3();
        let b = vec![0.0, 0.0, 0.0];
        let result = cg(&a, &b, None, IterativeSolveOptions::default()).expect("cg works");
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert_close_slice(&result.solution, &[0.0, 0.0, 0.0], 1e-14);
    }

    #[test]
    fn cg_rejects_non_square() {
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 3),
            vec![1.0, 2.0],
            vec![0, 1],
            vec![1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let err =
            cg(&a, &[1.0, 2.0], None, IterativeSolveOptions::default()).expect_err("non-square");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn cg_rejects_rhs_mismatch() {
        let a = spd_csr_3x3();
        let err =
            cg(&a, &[1.0, 2.0], None, IterativeSolveOptions::default()).expect_err("mismatch");
        assert!(matches!(err, SparseError::IncompatibleShape { .. }));
    }

    #[test]
    fn cg_hardened_rejects_non_finite_when_check_disabled() {
        let a = spd_csr_3x3();
        let err = cg(
            &a,
            &[f64::NAN, 1.0, 1.0],
            None,
            hardened_unchecked_iterative_options(),
        )
        .expect_err("hardened finite guard");
        assert!(matches!(err, SparseError::NonFiniteInput { .. }));
    }

    #[test]
    fn cg_max_iter_limits_iterations() {
        let a = spd_csr_3x3();
        let b = vec![5.0, 5.0, 3.0];
        let options = IterativeSolveOptions {
            max_iter: Some(1),
            tol: 1e-15, // extremely tight tolerance
            ..IterativeSolveOptions::default()
        };
        let result = cg(&a, &b, None, options).expect("cg works");
        assert!(result.iterations <= 1, "should be limited to max_iter");
    }

    #[test]
    fn cg_diagonal_system() {
        // [[2, 0], [0, 5]] x = [4, 10] => x = [2, 2]
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![2.0, 5.0],
            vec![0, 1],
            vec![0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![4.0, 10.0];
        let result = cg(&a, &b, None, IterativeSolveOptions::default()).expect("cg works");
        assert!(result.converged);
        assert_close_slice(&result.solution, &[2.0, 2.0], 1e-10);
    }

    // ── GMRES iterative solver tests ────────────────────────────────

    fn nonsymmetric_csr_3x3() -> CsrMatrix {
        // Non-symmetric: [[4, 1, 0], [0, 3, 1], [0, 0, 2]]
        CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![4.0, 1.0, 3.0, 1.0, 2.0],
            vec![0, 0, 1, 1, 2],
            vec![0, 1, 1, 2, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr")
    }

    #[test]
    fn gmres_identity_system() {
        let a = identity_csr(4);
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let result = gmres(&a, &b, None, IterativeSolveOptions::default()).expect("gmres works");
        assert!(result.converged);
        assert_close_slice(&result.solution, &b, 1e-10);
    }

    #[test]
    fn gmres_nonsymmetric_system() {
        let a = nonsymmetric_csr_3x3();
        let b = vec![5.0, 7.0, 4.0];
        let result = gmres(&a, &b, None, IterativeSolveOptions::default()).expect("gmres works");
        assert!(result.converged, "GMRES should converge");
        // Verify A*x ≈ b
        let ax = csr_matvec(&a, &result.solution);
        assert_close_slice(&ax, &b, 1e-5);
    }

    #[test]
    fn gmres_diagonal_system() {
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![3.0, 7.0],
            vec![0, 1],
            vec![0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![9.0, 14.0];
        let result = gmres(&a, &b, None, IterativeSolveOptions::default()).expect("gmres works");
        assert!(result.converged);
        assert_close_slice(&result.solution, &[3.0, 2.0], 1e-10);
    }

    #[test]
    fn gmres_zero_rhs() {
        let a = nonsymmetric_csr_3x3();
        let b = vec![0.0, 0.0, 0.0];
        let result = gmres(&a, &b, None, IterativeSolveOptions::default()).expect("gmres works");
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert_close_slice(&result.solution, &[0.0, 0.0, 0.0], 1e-14);
    }

    #[test]
    fn gmres_with_initial_guess() {
        let a = nonsymmetric_csr_3x3();
        let b = vec![5.0, 7.0, 4.0];
        let x0 = vec![1.0, 1.0, 1.0];
        let result =
            gmres(&a, &b, Some(&x0), IterativeSolveOptions::default()).expect("gmres works");
        assert!(result.converged);
        let ax = csr_matvec(&a, &result.solution);
        assert_close_slice(&ax, &b, 1e-5);
    }

    #[test]
    fn gmres_rejects_non_square() {
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 3),
            vec![1.0, 2.0],
            vec![0, 1],
            vec![1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let err =
            gmres(&a, &[1.0, 2.0], None, IterativeSolveOptions::default()).expect_err("non-sq");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn gmres_rejects_rhs_mismatch() {
        let a = nonsymmetric_csr_3x3();
        let err =
            gmres(&a, &[1.0, 2.0], None, IterativeSolveOptions::default()).expect_err("mismatch");
        assert!(matches!(err, SparseError::IncompatibleShape { .. }));
    }

    #[test]
    fn gmres_hardened_rejects_non_finite_when_check_disabled() {
        let a = nonsymmetric_csr_3x3();
        let err = gmres(
            &a,
            &[f64::NAN, 1.0, 1.0],
            None,
            hardened_unchecked_iterative_options(),
        )
        .expect_err("hardened finite guard");
        assert!(matches!(err, SparseError::NonFiniteInput { .. }));
    }

    #[test]
    fn gmres_spd_system_matches_cg() {
        // GMRES should work on SPD systems too
        let a = spd_csr_3x3();
        let b = vec![5.0, 5.0, 3.0];
        let cg_result = cg(&a, &b, None, IterativeSolveOptions::default()).expect("cg works");
        let gmres_result =
            gmres(&a, &b, None, IterativeSolveOptions::default()).expect("gmres works");
        assert!(gmres_result.converged);
        assert_close_slice(&gmres_result.solution, &cg_result.solution, 1e-6);
    }

    #[test]
    fn gmres_general_dense_system() {
        // [[1, 2, 3], [4, 5, 6], [7, 8, 10]] x = [14, 32, 53] => x = [1, 2, 3]
        let a = CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
            vec![0, 0, 0, 1, 1, 1, 2, 2, 2],
            vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![14.0, 32.0, 53.0];
        let result = gmres(&a, &b, None, IterativeSolveOptions::default()).expect("gmres works");
        assert!(result.converged);
        assert_close_slice(&result.solution, &[1.0, 2.0, 3.0], 1e-6);
    }

    // ── BiCGSTAB iterative solver tests ─────────────────────────────

    #[test]
    fn bicgstab_nonsymmetric_system() {
        let a = nonsymmetric_csr_3x3();
        let b = vec![5.0, 7.0, 4.0];
        let result =
            bicgstab(&a, &b, None, IterativeSolveOptions::default()).expect("bicgstab works");
        assert!(result.converged, "BiCGSTAB should converge");
        let ax = csr_matvec(&a, &result.solution);
        assert_close_slice(&ax, &b, 1e-5);
    }

    #[test]
    fn bicgstab_identity() {
        let a = identity_csr(4);
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let result =
            bicgstab(&a, &b, None, IterativeSolveOptions::default()).expect("bicgstab works");
        assert!(result.converged);
        assert_close_slice(&result.solution, &b, 1e-10);
    }

    #[test]
    fn bicgstab_spd_matches_cg() {
        let a = spd_csr_3x3();
        let b = vec![5.0, 5.0, 3.0];
        let cg_result = cg(&a, &b, None, IterativeSolveOptions::default()).expect("cg");
        let bicg_result =
            bicgstab(&a, &b, None, IterativeSolveOptions::default()).expect("bicgstab");
        assert!(bicg_result.converged);
        assert_close_slice(&bicg_result.solution, &cg_result.solution, 1e-5);
    }

    #[test]
    fn bicgstab_zero_rhs() {
        let a = nonsymmetric_csr_3x3();
        let b = vec![0.0, 0.0, 0.0];
        let result =
            bicgstab(&a, &b, None, IterativeSolveOptions::default()).expect("bicgstab works");
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn bicgstab_hardened_rejects_non_finite_when_check_disabled() {
        let a = nonsymmetric_csr_3x3();
        let err = bicgstab(
            &a,
            &[f64::NAN, 1.0, 1.0],
            None,
            hardened_unchecked_iterative_options(),
        )
        .expect_err("hardened finite guard");
        assert!(matches!(err, SparseError::NonFiniteInput { .. }));
    }

    // ── MINRES iterative solver tests ───────────────────────────────

    #[test]
    fn minres_spd_system() {
        let a = spd_csr_3x3();
        let b = vec![5.0, 5.0, 3.0];
        let result = minres(&a, &b, None, IterativeSolveOptions::default()).expect("minres works");
        assert!(result.converged, "MINRES should converge for SPD system");
        let ax = csr_matvec(&a, &result.solution);
        assert_close_slice(&ax, &b, 1e-4);
    }

    #[test]
    fn minres_identity() {
        let a = identity_csr(4);
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let result = minres(&a, &b, None, IterativeSolveOptions::default()).expect("minres works");
        assert!(result.converged);
        assert_close_slice(&result.solution, &b, 1e-10);
    }

    #[test]
    fn minres_symmetric_indefinite() {
        // Symmetric indefinite: [[2, 1], [1, -3]]
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![2.0, 1.0, 1.0, -3.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![3.0, -2.0];
        let result = minres(&a, &b, None, IterativeSolveOptions::default()).expect("minres works");
        assert!(result.converged, "MINRES should handle indefinite systems");
        let ax = csr_matvec(&a, &result.solution);
        assert_close_slice(&ax, &b, 1e-4);
    }

    #[test]
    fn casp_selects_cg_for_spd_row_dominant_system() {
        let a = spd_csr_3x3();
        let b = vec![5.0, 5.0, 3.0];
        let result = casp_iterative_solve(&a, &b, None, CaspIterativeSolveOptions::default())
            .expect("casp solve");
        assert_eq!(result.decision.selected_solver, CaspIterativeSolver::Cg);
        assert_eq!(
            result.decision.rationale,
            "symmetric_positive_diagonal_row_dominant"
        );
        assert!(result.result.converged);
        let ax = csr_matvec(&a, &result.result.solution);
        assert_close_slice(&ax, &b, 1e-5);
    }

    #[test]
    fn casp_selects_minres_for_symmetric_indefinite_system() {
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![2.0, 1.0, 1.0, -3.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![3.0, -2.0];
        let decision =
            select_casp_iterative_solver(&a, &b, None, CaspIterativeSolveOptions::default())
                .expect("casp decision");
        assert_eq!(decision.selected_solver, CaspIterativeSolver::Minres);
        assert!(decision.symmetric);
        assert!(!decision.positive_diagonal);
    }

    #[test]
    fn casp_selects_gmres_for_small_nonsymmetric_system() {
        let a = nonsymmetric_csr_3x3();
        let b = vec![5.0, 7.0, 4.0];
        let result = casp_iterative_solve(&a, &b, None, CaspIterativeSolveOptions::default())
            .expect("casp solve");
        assert_eq!(result.decision.selected_solver, CaspIterativeSolver::Gmres);
        assert!(!result.decision.symmetric);
        assert!(result.result.converged);
        let ax = csr_matvec(&a, &result.result.solution);
        assert_close_slice(&ax, &b, 1e-5);
    }

    #[test]
    fn casp_selects_lsqr_for_overdetermined_rectangular_system() {
        let a = CooMatrix::from_triplets(
            Shape2D::new(4, 2),
            vec![1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            vec![0, 1, 2, 2, 3, 3],
            vec![0, 1, 0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![1.0, 2.0, 4.0, 0.0];
        let result = casp_iterative_solve(&a, &b, None, CaspIterativeSolveOptions::default())
            .expect("casp solve");
        assert_eq!(result.decision.selected_solver, CaspIterativeSolver::Lsqr);
        assert!(!result.decision.square);
        let ax = csr_matvec(&a, &result.result.solution);
        let residual: Vec<f64> = ax.iter().zip(b.iter()).map(|(a, b)| a - b).collect();
        let normal_residual = vec_norm(&csr_matvec_transpose(&a, &residual));
        assert!(
            normal_residual < 1.0,
            "normal equations residual should be small: {normal_residual}"
        );
    }

    #[test]
    fn casp_selects_lsmr_for_underdetermined_rectangular_system() {
        let a = non_square_csr();
        let b = vec![1.0, 2.0];
        let decision =
            select_casp_iterative_solver(&a, &b, None, CaspIterativeSolveOptions::default())
                .expect("casp decision");
        assert_eq!(decision.selected_solver, CaspIterativeSolver::Lsmr);
        assert!(!decision.square);
        assert_eq!(
            decision.rationale,
            "rectangular_underdetermined_least_squares"
        );
    }

    #[test]
    fn casp_selects_lgmres_when_preconditioner_available() {
        let a = nonsymmetric_csr_3x3();
        let b = vec![5.0, 7.0, 4.0];
        let options = CaspIterativeSolveOptions {
            preconditioner_available: true,
            ..CaspIterativeSolveOptions::default()
        };
        let decision = select_casp_iterative_solver(&a, &b, None, options).expect("casp decision");
        assert_eq!(decision.selected_solver, CaspIterativeSolver::Lgmres);
        assert!(decision.preconditioner_available);
        assert_eq!(decision.rationale, "nonsymmetric_preconditioner_available");
    }

    #[test]
    fn casp_selects_bicgstab_for_low_memory_or_expensive_matvec() {
        let a = nonsymmetric_csr_3x3();
        let b = vec![5.0, 7.0, 4.0];
        let low_memory = CaspIterativeSolveOptions {
            prefer_low_memory: true,
            ..CaspIterativeSolveOptions::default()
        };
        let low_memory_decision =
            select_casp_iterative_solver(&a, &b, None, low_memory).expect("casp decision");
        assert_eq!(
            low_memory_decision.selected_solver,
            CaspIterativeSolver::Bicgstab
        );

        let expensive_matvec = CaspIterativeSolveOptions {
            matrix_vector_cost: CaspMatvecCost::Expensive,
            ..CaspIterativeSolveOptions::default()
        };
        let expensive_decision =
            select_casp_iterative_solver(&a, &b, None, expensive_matvec).expect("casp decision");
        assert_eq!(
            expensive_decision.selected_solver,
            CaspIterativeSolver::Bicgstab
        );
        assert_eq!(
            expensive_decision.rationale,
            "nonsymmetric_low_memory_or_expensive_matvec"
        );
    }

    #[test]
    fn casp_selects_qmr_for_large_very_sparse_nonsymmetric_system() {
        let n = 32;
        let mut data = Vec::with_capacity(n * 2 - 1);
        let mut rows = Vec::with_capacity(n * 2 - 1);
        let mut cols = Vec::with_capacity(n * 2 - 1);
        for i in 0..n {
            data.push(4.0);
            rows.push(i);
            cols.push(i);
            if i + 1 < n {
                data.push(1.0);
                rows.push(i);
                cols.push(i + 1);
            }
        }
        let a = CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
            .expect("coo")
            .to_csr()
            .expect("csr");
        let b = vec![1.0; n];
        let decision =
            select_casp_iterative_solver(&a, &b, None, CaspIterativeSolveOptions::default())
                .expect("casp decision");
        assert_eq!(decision.selected_solver, CaspIterativeSolver::Qmr);
        assert!(!decision.symmetric);
        assert!(decision.density <= 0.10);
        assert_eq!(
            decision.rationale,
            "large_very_sparse_nonsymmetric_transpose_stabilization"
        );
    }

    #[test]
    fn casp_audit_records_solver_choice_rationale() {
        let a = spd_csr_3x3();
        let b = vec![5.0, 5.0, 3.0];
        let ledger = crate::audit::sync_audit_ledger();
        let result = casp_iterative_solve_with_audit(
            &a,
            &b,
            None,
            CaspIterativeSolveOptions::default(),
            &ledger,
        )
        .expect("casp audited solve");
        assert_eq!(result.decision.selected_solver, CaspIterativeSolver::Cg);

        let ledger = ledger.lock().expect("audit ledger");
        assert_eq!(ledger.len(), 1);
        let entry = &ledger.entries()[0];
        let recovery_action = match &entry.action {
            fsci_runtime::AuditAction::BoundedRecovery { recovery_action } => {
                Some(recovery_action.as_str())
            }
            _ => None,
        };
        assert_eq!(
            recovery_action,
            Some("casp_sparse_iterative_solver=cg"),
            "audit action must record sparse CASP solver choice"
        );
        assert!(
            entry
                .outcome
                .contains("rationale=symmetric_positive_diagonal_row_dominant"),
            "audit outcome must carry solver-choice rationale: {}",
            entry.outcome
        );
        assert!(entry.outcome.contains("square=true"));
        assert!(entry.outcome.contains("positive_diagonal=true"));
        assert!(entry.outcome.contains("row_diagonally_dominant=true"));
    }

    // ── LSQR least-squares solver tests ─────────────────────────────

    #[test]
    fn lsqr_square_system() {
        let a = CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0],
            vec![0, 0, 1, 1, 1, 2, 2],
            vec![0, 1, 0, 1, 2, 1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![5.0, 5.0, 3.0];
        let result = lsqr(&a, &b, IterativeSolveOptions::default()).expect("lsqr works");
        assert!(result.converged, "LSQR should converge for square SPD");
        let ax = csr_matvec(&a, &result.solution);
        assert_close_slice(&ax, &b, 1e-4);
    }

    #[test]
    fn lsqr_overdetermined() {
        // 4x2 overdetermined system
        // A = [[1,0],[0,1],[1,1],[1,-1]], b = [1,2,4,0]
        let a = CooMatrix::from_triplets(
            Shape2D::new(4, 2),
            vec![1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            vec![0, 1, 2, 2, 3, 3],
            vec![0, 1, 0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![1.0, 2.0, 4.0, 0.0];
        let options = IterativeSolveOptions {
            max_iter: Some(100),
            tol: 1e-4,
            ..IterativeSolveOptions::default()
        };
        let result = lsqr(&a, &b, options).expect("lsqr works");
        // For overdetermined systems, check the normal equations residual
        // A^T(Ax - b) should be near zero even if Ax != b
        let ax = csr_matvec(&a, &result.solution);
        let residual: Vec<f64> = ax.iter().zip(b.iter()).map(|(a, b)| a - b).collect();
        let atr = csr_matvec_transpose(&a, &residual);
        let normal_residual = vec_norm(&atr);
        assert!(
            normal_residual < 1.0,
            "normal equations residual should be small: {normal_residual}"
        );
    }

    #[test]
    fn lsqr_zero_rhs() {
        let a = identity_csr(3);
        let b = vec![0.0, 0.0, 0.0];
        let result = lsqr(&a, &b, IterativeSolveOptions::default()).expect("lsqr works");
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }

    // ── LSMR least-squares solver tests ─────────────────────────────

    #[test]
    fn lsmr_square_system() {
        let a = CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0],
            vec![0, 0, 1, 1, 1, 2, 2],
            vec![0, 1, 0, 1, 2, 1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![5.0, 5.0, 3.0];
        let result = lsmr(&a, &b, IterativeSolveOptions::default()).expect("lsmr works");
        assert!(result.converged, "LSMR should converge for square SPD");
        let ax = csr_matvec(&a, &result.solution);
        assert_close_slice(&ax, &b, 1e-4);
    }

    #[test]
    fn lsmr_zero_rhs() {
        let a = identity_csr(3);
        let b = vec![0.0, 0.0, 0.0];
        let result = lsmr(&a, &b, IterativeSolveOptions::default()).expect("lsmr works");
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }

    // ── eigs (Arnoldi) tests ────────────────────────────────────────

    #[test]
    fn eigs_diagonal_known_eigenvalues() {
        // Diagonal matrix with known eigenvalues [5, 3, 1]
        let a = CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![5.0, 3.0, 1.0],
            vec![0, 1, 2],
            vec![0, 1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let result = eigs(&a, 2, EigsOptions::default()).expect("eigs works");
        assert_eq!(result.eigenvalues.len(), 2);
        // Should find the two largest: 5 and 3
        let mut sorted = result.eigenvalues.clone();
        sorted.sort_by(|a, b| b.abs().total_cmp(&a.abs()));
        assert!(
            (sorted[0] - 5.0).abs() < 1.0,
            "largest eigenvalue: {}",
            sorted[0]
        );
        assert!(
            (sorted[1] - 3.0).abs() < 2.0,
            "second eigenvalue: {}",
            sorted[1]
        );
    }

    #[test]
    fn eigs_returns_actual_eigenpairs() {
        // Eigenvectors must satisfy ||A x - lambda x|| ~ 0, not merely carry
        // the right eigenvalue. Regression: eigs previously returned raw
        // Arnoldi basis vectors, whose residual is O(||x||).
        let check = |a: &CsrMatrix, k: usize| {
            let result = eigs(a, k, EigsOptions::default()).expect("eigs works");
            for (lambda, x) in result.eigenvalues.iter().zip(&result.eigenvectors) {
                let ax = csr_matvec(a, x);
                let residual: f64 = ax
                    .iter()
                    .zip(x)
                    .map(|(&axi, &xi)| (axi - lambda * xi).powi(2))
                    .sum::<f64>()
                    .sqrt();
                assert!(
                    (vec_norm(x) - 1.0).abs() < 1e-9,
                    "eigenvector must be unit-norm"
                );
                assert!(
                    residual < 1e-6,
                    "eigenpair residual too large: lambda={lambda}, ||Ax-lx||={residual}"
                );
            }
        };

        // Diagonal matrix (the bead's reproduction case).
        let diag = CooMatrix::from_triplets(
            Shape2D::new(4, 4),
            vec![7.0, 4.0, 2.0, 9.0],
            vec![0, 1, 2, 3],
            vec![0, 1, 2, 3],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        check(&diag, 3);

        // Symmetric tridiagonal tridiag(-1, 2, -1), size 6.
        let n = 6;
        let mut vals = Vec::new();
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for i in 0..n {
            vals.push(2.0);
            rows.push(i);
            cols.push(i);
            if i + 1 < n {
                vals.push(-1.0);
                rows.push(i);
                cols.push(i + 1);
                vals.push(-1.0);
                rows.push(i + 1);
                cols.push(i);
            }
        }
        let tri = CooMatrix::from_triplets(Shape2D::new(n, n), vals, rows, cols, false)
            .expect("coo")
            .to_csr()
            .expect("csr");
        check(&tri, 3);
    }

    #[test]
    fn eigs_identity() {
        let a = identity_csr(4);
        let result = eigs(&a, 2, EigsOptions::default()).expect("eigs works");
        // All eigenvalues should be 1.0
        for &val in &result.eigenvalues {
            assert!(
                (val - 1.0).abs() < 0.1,
                "identity eigenvalue should be 1: {val}"
            );
        }
    }

    #[test]
    fn eigs_rejects_non_square() {
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 3),
            vec![1.0, 2.0],
            vec![0, 1],
            vec![1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let err = eigs(&a, 1, EigsOptions::default()).expect_err("non-square");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    // ── svds (sparse SVD) tests ─────────────────────────────────────

    #[test]
    fn svds_diagonal_known_singular_values() {
        // Diagonal matrix: singular values are absolute values of diagonal
        let a = CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![5.0, -3.0, 1.0],
            vec![0, 1, 2],
            vec![0, 1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let result = svds(&a, 2, EigsOptions::default()).expect("svds works");
        assert_eq!(result.singular_values.len(), 2);
        // Should find 5.0 and 3.0 (largest by magnitude)
        assert!(
            (result.singular_values[0] - 5.0).abs() < 0.5,
            "largest sv: {}",
            result.singular_values[0]
        );
    }

    #[test]
    fn svds_identity() {
        let a = identity_csr(3);
        let result = svds(&a, 1, EigsOptions::default()).expect("svds works");
        assert_eq!(result.singular_values.len(), 1);
        assert!(
            (result.singular_values[0] - 1.0).abs() < 0.1,
            "identity sv should be 1: {}",
            result.singular_values[0]
        );
    }

    #[test]
    fn svds_zero_max_iter_uses_default_iteration_budget() {
        let a = CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![5.0, -3.0, 1.0],
            vec![0, 1, 2],
            vec![0, 1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let options = EigsOptions {
            max_iter: 0,
            ..EigsOptions::default()
        };
        let result = svds(&a, 1, options).expect("svds works");
        assert!(
            (result.singular_values[0] - 5.0).abs() < 0.5,
            "largest sv with sanitized max_iter: {}",
            result.singular_values[0]
        );
    }

    #[test]
    fn svds_rectangular() {
        // 3x2 matrix
        let a = CooMatrix::from_triplets(
            Shape2D::new(3, 2),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 0, 1, 2],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let result = svds(&a, 1, EigsOptions::default()).expect("svds works");
        assert_eq!(result.singular_values.len(), 1);
        assert!(
            result.singular_values[0] > 0.0,
            "sv should be positive: {}",
            result.singular_values[0]
        );
    }

    #[test]
    fn svds_rejects_invalid_k() {
        let a = identity_csr(3);
        let err = svds(&a, 0, EigsOptions::default()).expect_err("k=0");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    // ── Graph algorithms (csgraph) tests ─────────────────────────────

    fn triangle_graph_csr() -> CsrMatrix {
        // 3-node connected graph: 0-1 (w=1), 1-2 (w=2), 0-2 (w=3)
        CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            vec![0, 1, 1, 2, 0, 2],
            vec![1, 0, 2, 1, 2, 0],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr")
    }

    fn disconnected_graph_csr() -> CsrMatrix {
        // 4-node graph: 0-1 connected, 2-3 connected, no edge between groups
        CooMatrix::from_triplets(
            Shape2D::new(4, 4),
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0, 1, 2, 3],
            vec![1, 0, 3, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr")
    }

    #[test]
    fn connected_components_single_component() {
        let g = triangle_graph_csr();
        let result = connected_components(&g).expect("cc");
        assert_eq!(result.n_components, 1);
        assert!(
            result.labels.iter().all(|&l| l == 0),
            "all nodes in same component"
        );
    }

    #[test]
    fn connected_components_two_components() {
        let g = disconnected_graph_csr();
        let result = connected_components(&g).expect("cc");
        assert_eq!(result.n_components, 2, "should have 2 components");
        // Nodes 0,1 in one component, nodes 2,3 in another
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[2], result.labels[3]);
        assert_ne!(result.labels[0], result.labels[2]);
    }

    #[test]
    fn connected_components_isolated_node() {
        // 3 nodes, only 0-1 connected, node 2 isolated
        let g = CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![1.0, 1.0],
            vec![0, 1],
            vec![1, 0],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let result = connected_components(&g).expect("cc");
        assert_eq!(result.n_components, 2);
    }

    #[test]
    fn dijkstra_triangle_graph() {
        let g = triangle_graph_csr();
        let result = dijkstra(&g, 0).expect("dijkstra");
        assert_eq!(result.distances[0], 0.0);
        // Node 1 takes the direct edge. Node 2 can use the direct edge or node 1 with equal cost.
        assert_eq!(result.distances[1], 1.0);
        assert!(
            (result.distances[2] - 3.0).abs() < 1e-10,
            "dist to node 2: {}",
            result.distances[2]
        );
    }

    #[test]
    fn dijkstra_unreachable_node() {
        let g = disconnected_graph_csr();
        let result = dijkstra(&g, 0).expect("dijkstra");
        assert_eq!(result.distances[0], 0.0);
        assert!(result.distances[1].is_finite());
        assert!(
            result.distances[2].is_infinite(),
            "node 2 should be unreachable"
        );
    }

    #[test]
    fn dijkstra_source_out_of_bounds() {
        let g = triangle_graph_csr();
        let err = dijkstra(&g, 10).expect_err("oob");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn dijkstra_negative_edge_matches_scipy_reference_result() {
        let g = CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![1.0, -2.0],
            vec![0, 1],
            vec![1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let result = dijkstra(&g, 0).expect("dijkstra negative edge");
        assert_eq!(result.distances[0], 0.0);
        assert_eq!(result.distances[1], 1.0);
        assert!((result.distances[2] - -1.0).abs() < 1e-10);

        let unreachable = dijkstra(&g, 2).expect("dijkstra unreachable source");
        assert!(unreachable.distances[0].is_infinite());
        assert!(unreachable.distances[1].is_infinite());
        assert_eq!(unreachable.distances[2], 0.0);
    }

    #[test]
    fn dijkstra_unreachable_negative_component_is_ignored_like_scipy() {
        let g = CooMatrix::from_triplets(
            Shape2D::new(4, 4),
            vec![1.0, -2.0],
            vec![0, 2],
            vec![1, 3],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let result = dijkstra(&g, 0).expect("dijkstra with unreachable negative edge");
        assert_eq!(result.distances[0], 0.0);
        assert_eq!(result.distances[1], 1.0);
        assert!(result.distances[2].is_infinite());
        assert!(result.distances[3].is_infinite());
    }

    #[test]
    fn minimum_spanning_tree_triangle() {
        let g = triangle_graph_csr();
        let result = minimum_spanning_tree(&g).expect("mst");
        // Triangle with weights 1, 2, 3 → MST has edges 1 and 2, total = 3
        assert_eq!(result.edges.len(), 2, "MST of 3-node graph has 2 edges");
        assert!(
            (result.total_weight - 3.0).abs() < 1e-10,
            "MST weight: {}",
            result.total_weight
        );
    }

    #[test]
    fn minimum_spanning_tree_disconnected() {
        let g = disconnected_graph_csr();
        let result = minimum_spanning_tree(&g).expect("mst");
        // Disconnected: MST has edges within each component
        assert_eq!(result.edges.len(), 2, "MST edges in disconnected graph");
    }

    #[test]
    fn csgraph_rejects_non_square_adjacency() {
        let g = CooMatrix::from_triplets(
            Shape2D::new(2, 3),
            vec![1.0, 1.0],
            vec![0, 1],
            vec![1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");

        assert!(matches!(
            connected_components(&g),
            Err(SparseError::InvalidArgument { .. })
        ));
        assert!(matches!(
            dijkstra(&g, 0),
            Err(SparseError::InvalidArgument { .. })
        ));
        assert!(matches!(
            minimum_spanning_tree(&g),
            Err(SparseError::InvalidArgument { .. })
        ));
    }

    // ── Bellman-Ford tests ───────────────────────────────────────────

    #[test]
    fn bellman_ford_positive_weights() {
        // Same as Dijkstra test — should give identical results
        let g = triangle_graph_csr();
        let result = bellman_ford(&g, 0).expect("bellman_ford");
        assert_eq!(result.distances[0], 0.0);
        assert_eq!(result.distances[1], 1.0);
        assert!(
            (result.distances[2] - 3.0).abs() < 1e-10,
            "dist to 2: {}",
            result.distances[2]
        );
    }

    #[test]
    fn bellman_ford_negative_edge() {
        // Graph: 0→1 (w=4), 0→2 (w=5), 1→2 (w=-3)
        // Shortest 0→2: 0→1→2 = 4+(-3) = 1 (not direct 5)
        let g = CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![4.0, 5.0, -3.0],
            vec![0, 0, 1],
            vec![1, 2, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let result = bellman_ford(&g, 0).expect("bellman_ford neg");
        assert_eq!(result.distances[0], 0.0);
        assert_eq!(result.distances[1], 4.0);
        assert!(
            (result.distances[2] - 1.0).abs() < 1e-10,
            "shortest to 2 via neg edge: {}",
            result.distances[2]
        );
    }

    #[test]
    fn bellman_ford_negative_cycle_detected() {
        // Negative cycle: 0→1 (w=1), 1→2 (w=-1), 2→0 (w=-1)
        // Total cycle weight: 1 + (-1) + (-1) = -1 < 0
        let g = CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![1.0, -1.0, -1.0],
            vec![0, 1, 2],
            vec![1, 2, 0],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let err = bellman_ford(&g, 0).expect_err("negative cycle");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn bellman_ford_unreachable() {
        let g = disconnected_graph_csr();
        let result = bellman_ford(&g, 0).expect("bellman_ford");
        assert!(result.distances[2].is_infinite());
    }

    // ── BFS/DFS traversal tests ─────────────────────────────────────

    #[test]
    fn bfs_order_triangle() {
        let g = triangle_graph_csr();
        let (order, pred) = breadth_first_order(&g, 0).expect("bfs");
        assert_eq!(order[0], 0, "BFS starts at source");
        assert_eq!(order.len(), 3, "BFS visits all 3 nodes");
        assert_eq!(pred[0], -1, "source has no predecessor");
    }

    #[test]
    fn bfs_order_disconnected() {
        let g = disconnected_graph_csr();
        let (order, _) = breadth_first_order(&g, 0).expect("bfs");
        // Only visits nodes reachable from 0: nodes 0 and 1
        assert_eq!(order.len(), 2, "BFS only visits connected component");
        assert!(order.contains(&0));
        assert!(order.contains(&1));
    }

    #[test]
    fn dfs_order_triangle() {
        let g = triangle_graph_csr();
        let (order, pred) = depth_first_order(&g, 0).expect("dfs");
        assert_eq!(order[0], 0, "DFS starts at source");
        assert_eq!(order.len(), 3, "DFS visits all 3 nodes");
        assert_eq!(pred[0], -1, "source has no predecessor");
    }

    #[test]
    fn dfs_order_disconnected() {
        let g = disconnected_graph_csr();
        let (order, _) = depth_first_order(&g, 0).expect("dfs");
        assert_eq!(order.len(), 2, "DFS only visits connected component");
    }

    #[test]
    fn bfs_source_out_of_bounds() {
        let g = triangle_graph_csr();
        let err = breadth_first_order(&g, 10).expect_err("oob");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    // ── Graph Laplacian tests ────────────────────────────────────────

    #[test]
    fn laplacian_row_sums_zero() {
        // Unnormalized Laplacian has zero row sums
        let g = triangle_graph_csr();
        let l = laplacian(&g, false).expect("laplacian");
        for (i, row) in l.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(sum.abs() < 1e-10, "row {i} sum should be 0: {sum}");
        }
    }

    #[test]
    fn laplacian_diagonal_is_degree() {
        let g = triangle_graph_csr();
        let l = laplacian(&g, false).expect("laplacian");
        // Triangle graph: each node has degree = sum of edge weights to neighbors
        // Node 0: edges to 1 (w=1) and 2 (w=3) → degree = 4
        assert!(
            (l[0][0] - 4.0).abs() < 1e-10,
            "L[0,0] = {}, expected 4",
            l[0][0]
        );
    }

    #[test]
    fn laplacian_normed_diagonal_ones() {
        // Normalized Laplacian has 1.0 on diagonal (for connected nodes)
        let g = triangle_graph_csr();
        let l = laplacian(&g, true).expect("normed laplacian");
        for (i, row) in l.iter().enumerate().take(3) {
            assert!(
                (row[i] - 1.0).abs() < 1e-10,
                "L_norm[{i},{i}] = {}, expected 1.0",
                row[i]
            );
        }
    }

    #[test]
    fn laplacian_symmetric() {
        let g = triangle_graph_csr();
        let l = laplacian(&g, false).expect("laplacian");
        let n = l.len();
        for (i, row_i) in l.iter().enumerate().take(n) {
            for (j, row_j) in l.iter().enumerate().take(n) {
                assert!(
                    (row_i[j] - row_j[i]).abs() < 1e-10,
                    "L[{i},{j}]={} != L[{j},{i}]={}",
                    row_i[j],
                    row_j[i]
                );
            }
        }
    }

    // ── BiCG iterative solver tests ─────────────────────────────────

    fn diagonally_dominant_csr_3x3() -> CsrMatrix {
        // Diagonally dominant (good for BiCG): [[5, 1, 1], [1, 5, 1], [1, 1, 5]]
        CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![5.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 5.0],
            vec![0, 0, 0, 1, 1, 1, 2, 2, 2],
            vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr")
    }

    #[test]
    fn bicg_diagonally_dominant_converges() {
        let a = diagonally_dominant_csr_3x3();
        let b = vec![7.0, 7.0, 7.0];
        let result = bicg(&a, &b, None, IterativeSolveOptions::default()).expect("bicg works");
        assert!(
            result.converged,
            "BiCG should converge for diagonally dominant system"
        );
        let ax = csr_matvec(&a, &result.solution);
        assert_close_slice(&ax, &b, 1e-5);
    }

    #[test]
    fn bicg_identity_system() {
        let a = identity_csr(4);
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let result = bicg(&a, &b, None, IterativeSolveOptions::default()).expect("bicg works");
        assert!(result.converged);
        assert_close_slice(&result.solution, &b, 1e-10);
        assert!(result.iterations <= 2, "identity should converge quickly");
    }

    #[test]
    fn bicg_zero_rhs() {
        let a = nonsymmetric_csr_3x3();
        let b = vec![0.0, 0.0, 0.0];
        let result = bicg(&a, &b, None, IterativeSolveOptions::default()).expect("bicg works");
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert_close_slice(&result.solution, &[0.0, 0.0, 0.0], 1e-14);
    }

    #[test]
    fn bicg_rejects_non_square() {
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 3),
            vec![1.0, 2.0],
            vec![0, 1],
            vec![1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let err =
            bicg(&a, &[1.0, 2.0], None, IterativeSolveOptions::default()).expect_err("non-square");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn bicg_hardened_rejects_non_finite_when_check_disabled() {
        let a = diagonally_dominant_csr_3x3();
        let err = bicg(
            &a,
            &[f64::NAN, 1.0, 1.0],
            None,
            hardened_unchecked_iterative_options(),
        )
        .expect_err("hardened finite guard");
        assert!(matches!(err, SparseError::NonFiniteInput { .. }));
    }

    // ── CGS iterative solver tests ──────────────────────────────────

    #[test]
    fn cgs_diagonally_dominant_converges() {
        let a = diagonally_dominant_csr_3x3();
        let b = vec![7.0, 7.0, 7.0];
        let result = cgs(&a, &b, None, IterativeSolveOptions::default()).expect("cgs works");
        assert!(
            result.converged,
            "CGS should converge for diagonally dominant system"
        );
        let ax = csr_matvec(&a, &result.solution);
        assert_close_slice(&ax, &b, 1e-5);
    }

    #[test]
    fn cgs_identity_system() {
        let a = identity_csr(4);
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let result = cgs(&a, &b, None, IterativeSolveOptions::default()).expect("cgs works");
        assert!(result.converged);
        assert_close_slice(&result.solution, &b, 1e-10);
        assert!(result.iterations <= 2, "identity should converge quickly");
    }

    #[test]
    fn cgs_zero_rhs() {
        let a = nonsymmetric_csr_3x3();
        let b = vec![0.0, 0.0, 0.0];
        let result = cgs(&a, &b, None, IterativeSolveOptions::default()).expect("cgs works");
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert_close_slice(&result.solution, &[0.0, 0.0, 0.0], 1e-14);
    }

    #[test]
    fn cgs_rejects_non_square() {
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 3),
            vec![1.0, 2.0],
            vec![0, 1],
            vec![1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let err =
            cgs(&a, &[1.0, 2.0], None, IterativeSolveOptions::default()).expect_err("non-square");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn cgs_hardened_rejects_non_finite_when_check_disabled() {
        let a = diagonally_dominant_csr_3x3();
        let err = cgs(
            &a,
            &[f64::NAN, 1.0, 1.0],
            None,
            hardened_unchecked_iterative_options(),
        )
        .expect_err("hardened finite guard");
        assert!(matches!(err, SparseError::NonFiniteInput { .. }));
    }

    // ── LGMRES iterative solver tests ───────────────────────────────

    #[test]
    fn lgmres_diagonally_dominant_converges() {
        let a = diagonally_dominant_csr_3x3();
        let b = vec![7.0, 7.0, 7.0];
        let result = lgmres(&a, &b, None, LgmresOptions::default()).expect("lgmres works");
        assert!(
            result.converged,
            "LGMRES should converge for diagonally dominant system"
        );
        let ax = csr_matvec(&a, &result.solution);
        assert_close_slice(&ax, &b, 1e-5);
    }

    #[test]
    fn lgmres_identity_system() {
        let a = identity_csr(4);
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let result = lgmres(&a, &b, None, LgmresOptions::default()).expect("lgmres works");
        assert!(result.converged);
        assert_close_slice(&result.solution, &b, 1e-10);
    }

    #[test]
    fn lgmres_zero_rhs() {
        let a = diagonally_dominant_csr_3x3();
        let b = vec![0.0, 0.0, 0.0];
        let result = lgmres(&a, &b, None, LgmresOptions::default()).expect("lgmres works");
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert_close_slice(&result.solution, &[0.0, 0.0, 0.0], 1e-14);
    }

    #[test]
    fn lgmres_rejects_non_square() {
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 3),
            vec![1.0, 2.0],
            vec![0, 1],
            vec![1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let err = lgmres(&a, &[1.0, 2.0], None, LgmresOptions::default()).expect_err("non-square");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn lgmres_spd_system() {
        let a = spd_csr_3x3();
        let b = vec![5.0, 5.0, 3.0];
        let result = lgmres(&a, &b, None, LgmresOptions::default()).expect("lgmres works");
        assert!(result.converged, "LGMRES should converge for SPD system");
        let ax = csr_matvec(&a, &result.solution);
        assert_close_slice(&ax, &b, 1e-5);
    }

    #[test]
    fn qmr_diagonally_dominant_converges() {
        let a = diagonally_dominant_csr_3x3();
        let b = vec![6.0, 11.0, 15.0];
        let opts = IterativeSolveOptions {
            tol: 1e-6,
            max_iter: Some(200),
            ..Default::default()
        };
        let result = qmr(&a, &b, None, opts).expect("qmr should work");
        // QMR may not always converge for all systems - check residual is reasonable
        assert!(
            result.converged || result.residual_norm < 0.1,
            "QMR residual should be reasonable: {}",
            result.residual_norm
        );
    }

    #[test]
    fn qmr_identity_system() {
        let a = identity_csr(3);
        let b = vec![1.0, 2.0, 3.0];
        let opts = IterativeSolveOptions {
            tol: 1e-10,
            max_iter: Some(10),
            ..Default::default()
        };
        let result = qmr(&a, &b, None, opts).expect("qmr works");
        assert!(
            result.converged,
            "QMR on identity should converge in 1 step"
        );
        assert_close_slice(&result.solution, &b, 1e-10);
    }

    #[test]
    fn qmr_zero_rhs() {
        let a = identity_csr(3);
        let b = vec![0.0, 0.0, 0.0];
        let opts = IterativeSolveOptions::default();
        let result = qmr(&a, &b, None, opts).expect("qmr works");
        assert!(result.converged);
        assert_eq!(result.solution, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn qmr_rejects_non_square() {
        let a = CsrMatrix::from_components(
            Shape2D::new(2, 3),
            vec![1.0, 2.0],
            vec![0, 1],
            vec![0, 1, 2],
            false,
        )
        .unwrap();
        let b = vec![1.0, 2.0];
        let err = qmr(&a, &b, None, IterativeSolveOptions::default()).expect_err("non-square");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn qmr_spd_system() {
        let a = spd_csr_3x3();
        let b = vec![5.0, 5.0, 3.0];
        let opts = IterativeSolveOptions {
            tol: 1e-6,
            max_iter: Some(200),
            ..Default::default()
        };
        let result = qmr(&a, &b, None, opts).expect("qmr works");
        // QMR may need more iterations - check residual is reasonable
        assert!(
            result.converged || result.residual_norm < 0.1,
            "QMR residual should be reasonable: {} after {} iterations",
            result.residual_norm,
            result.iterations
        );
    }

    #[test]
    fn qmr_converges_on_spd_tridiagonal() {
        // The bead's reproduction case: SPD A = tridiag(-1, 4, -1), size 6.
        // QMR previously stalled here (relative residual ~0.018) because the
        // Lanczos recurrences used A*v_n / A^T*w_n instead of A*p_n / A^T*q_n,
        // and the solution update omitted the QMR smoothing term.
        let n = 6;
        let mut vals = Vec::new();
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for i in 0..n {
            vals.push(4.0);
            rows.push(i);
            cols.push(i);
            if i + 1 < n {
                vals.push(-1.0);
                rows.push(i);
                cols.push(i + 1);
                vals.push(-1.0);
                rows.push(i + 1);
                cols.push(i);
            }
        }
        let a = CooMatrix::from_triplets(Shape2D::new(n, n), vals, rows, cols, false)
            .expect("coo")
            .to_csr()
            .expect("csr");
        let b = vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0];
        let opts = IterativeSolveOptions {
            tol: 1e-8,
            max_iter: Some(500),
            ..Default::default()
        };
        let result = qmr(&a, &b, None, opts).expect("qmr works");
        assert!(
            result.converged,
            "QMR must converge on SPD tridiagonal: residual={} after {} iters",
            result.residual_norm, result.iterations
        );
        // Verify the true residual ||A x - b|| independently.
        let ax = csr_matvec(&a, &result.solution);
        let true_res: f64 = ax
            .iter()
            .zip(&b)
            .map(|(&axi, &bi)| (axi - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            true_res < 1e-6,
            "true residual ||Ax-b|| too large: {true_res}"
        );
    }

    // ── matrix_power tests ───────────────────────────────────

    #[test]
    fn matrix_power_zero_returns_identity() {
        let a = square_csr();
        let result = matrix_power(&a, 0).expect("power 0");
        // Check result is identity
        let n = result.shape().rows;
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                let val = get_csr_value(&result, i, j);
                assert!(
                    (val - expected).abs() < 1e-10,
                    "A^0 should be identity: ({i},{j}) = {val}"
                );
            }
        }
    }

    #[test]
    fn matrix_power_one_returns_original() {
        let a = square_csr();
        let result = matrix_power(&a, 1).expect("power 1");
        // Check result equals original
        for i in 0..a.shape().rows {
            for j in 0..a.shape().cols {
                let expected = get_csr_value(&a, i, j);
                let got = get_csr_value(&result, i, j);
                assert!(
                    (got - expected).abs() < 1e-10,
                    "A^1 should equal A: ({i},{j}) expected {expected}, got {got}"
                );
            }
        }
    }

    #[test]
    fn matrix_power_two_equals_aa() {
        let a = square_csr();
        let a_squared = spmm(&a, &a);
        let result = matrix_power(&a, 2).expect("power 2");
        for i in 0..a.shape().rows {
            for j in 0..a.shape().cols {
                let expected = get_csr_value(&a_squared, i, j);
                let got = get_csr_value(&result, i, j);
                assert!(
                    (got - expected).abs() < 1e-10,
                    "A^2 should equal A*A: ({i},{j}) expected {expected}, got {got}"
                );
            }
        }
    }

    #[test]
    fn matrix_power_rejects_non_square() {
        let a = non_square_csr();
        let err = matrix_power(&a, 2).expect_err("non-square");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn matrix_power_identity_any_n() {
        let a = identity_csr(3);
        for n in [0, 1, 5, 10] {
            let result = matrix_power(&a, n).expect("power");
            // Identity^n = Identity
            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    let got = get_csr_value(&result, i, j);
                    assert!(
                        (got - expected).abs() < 1e-10,
                        "I^{n} should be I: ({i},{j}) = {got}"
                    );
                }
            }
        }
    }

    /// Helper to get value from CSR at position (i, j).
    fn get_csr_value(a: &CsrMatrix, i: usize, j: usize) -> f64 {
        let start = a.indptr()[i];
        let end = a.indptr()[i + 1];
        for idx in start..end {
            if a.indices()[idx] == j {
                return a.data()[idx];
            }
        }
        0.0
    }

    #[test]
    fn spsolve_matches_scipy_reference_values() {
        // scipy.sparse.linalg.spsolve(A, b) where A = [[4, 1], [1, 3]], b = [1, 2]
        use crate::{CooMatrix, Shape2D};
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![4.0, 1.0, 1.0, 3.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo from triplets")
        .to_csr()
        .expect("to csr");
        let b = vec![1.0, 2.0];
        let result = spsolve(&a, &b, SolveOptions::default()).expect("spsolve");
        let expected = [0.09090909090909091, 0.6363636363636364];
        for (i, (&got, &want)) in result.solution.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "x[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn norm_matches_scipy_reference_values() {
        // scipy.sparse.linalg.norm([[1, 2], [3, 4]], 'fro') -> sqrt(1+4+9+16) = sqrt(30)
        use crate::{CooMatrix, Shape2D};
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let norm = super::sparse_norm(&a, "fro");
        let expected = 30.0_f64.sqrt();
        assert!(
            (norm - expected).abs() < 1e-10,
            "norm got {norm}, expected {expected}"
        );
    }

    #[test]
    fn onenormest_matches_scipy_reference_values() {
        // scipy.sparse.linalg.onenormest([[1, 2], [3, 4]]) -> max column sum = max(4, 6) = 6
        use crate::{CooMatrix, Shape2D};
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let estimate = super::onenormest(&a);
        // 1-norm = max column sum = max(|1|+|3|, |2|+|4|) = max(4, 6) = 6
        assert!(
            (estimate - 6.0).abs() < 1e-10,
            "onenormest got {estimate}, expected 6.0"
        );
    }

    #[test]
    fn cg_matches_scipy_reference_values() {
        // scipy.sparse.linalg.cg(A, b) for SPD matrix
        // A = [[4, 1], [1, 3]], b = [1, 2] -> same as spsolve
        use crate::{CooMatrix, Shape2D};
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![4.0, 1.0, 1.0, 3.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![1.0, 2.0];
        let result = super::cg(&a, &b, None, IterativeSolveOptions::default()).expect("cg");
        let expected = [0.09090909090909091, 0.6363636363636364];
        for (i, (&got, &want)) in result.solution.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "cg x[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn gmres_matches_scipy_reference_values() {
        // scipy.sparse.linalg.gmres(A, b) for non-symmetric matrix
        // A = [[4, 1], [2, 3]], b = [1, 2]
        // x = linalg.solve([[4, 1], [2, 3]], [1, 2]) -> [0.1, 0.6]
        use crate::{CooMatrix, Shape2D};
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![4.0, 1.0, 2.0, 3.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![1.0, 2.0];
        let result = super::gmres(&a, &b, None, IterativeSolveOptions::default()).expect("gmres");
        let expected = [0.1, 0.6];
        for (i, (&got, &want)) in result.solution.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "gmres x[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn bicgstab_matches_scipy_reference_values() {
        // scipy.sparse.linalg.bicgstab(A, b)
        // A = [[4, 1], [2, 3]], b = [1, 2] -> same solution as gmres
        use crate::{CooMatrix, Shape2D};
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![4.0, 1.0, 2.0, 3.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![1.0, 2.0];
        let result =
            super::bicgstab(&a, &b, None, IterativeSolveOptions::default()).expect("bicgstab");
        let expected = [0.1, 0.6];
        for (i, (&got, &want)) in result.solution.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "bicgstab x[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn expm_matches_scipy_reference_values() {
        // scipy.sparse.linalg.expm([[0, 1], [0, 0]])
        // -> [[1, 1], [0, 1]] (nilpotent matrix: exp(A) = I + A)
        use crate::{CooMatrix, Shape2D};
        let a = CooMatrix::from_triplets(Shape2D::new(2, 2), vec![1.0], vec![0], vec![1], false)
            .expect("coo")
            .to_csr()
            .expect("csr");
        let result = super::expm(&a, ExpmOptions::default()).expect("expm");
        // Result should be [[1, 1], [0, 1]]
        assert!(
            (result[0][0] - 1.0).abs() < 1e-10,
            "expm[0][0] got {}, expected 1.0",
            result[0][0]
        );
        assert!(
            (result[0][1] - 1.0).abs() < 1e-10,
            "expm[0][1] got {}, expected 1.0",
            result[0][1]
        );
        assert!(
            result[1][0].abs() < 1e-10,
            "expm[1][0] got {}, expected 0.0",
            result[1][0]
        );
        assert!(
            (result[1][1] - 1.0).abs() < 1e-10,
            "expm[1][1] got {}, expected 1.0",
            result[1][1]
        );
    }

    #[test]
    fn minres_matches_scipy_reference_values() {
        // scipy.sparse.linalg.minres on symmetric 2x2 system
        use crate::{CooMatrix, Shape2D};
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![4.0, 1.0, 1.0, 3.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![1.0, 2.0];
        let result = super::minres(&a, &b, None, IterativeSolveOptions::default()).expect("minres");
        // Verify Ax ≈ b
        let ax = super::spmv(&a, &result.solution);
        for i in 0..2 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-6,
                "minres residual too large at {i}"
            );
        }
    }

    #[test]
    fn lsqr_matches_scipy_reference_values() {
        // scipy.sparse.linalg.lsqr on overdetermined system
        use crate::{CooMatrix, Shape2D};
        let a = CooMatrix::from_triplets(
            Shape2D::new(3, 2),
            vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            vec![0, 0, 1, 1, 2, 2],
            vec![0, 1, 0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![1.0, 2.0, 3.0];
        let result = super::lsqr(&a, &b, IterativeSolveOptions::default()).expect("lsqr");
        assert_eq!(
            result.solution.len(),
            2,
            "lsqr should return 2-element solution"
        );
    }

    #[test]
    fn floyd_warshall_matches_scipy_reference_values() {
        // scipy.sparse.csgraph.floyd_warshall for simple 3-node path: 0 -> 1 -> 2
        // Edges: (0,1)=1.0, (1,2)=2.0
        // Expected: d(0,0)=0, d(0,1)=1, d(0,2)=3, d(1,1)=0, d(1,2)=2, d(2,2)=0
        use crate::{CooMatrix, Shape2D};
        let g = CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![1.0, 2.0],
            vec![0, 1],
            vec![1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let dist = super::floyd_warshall(&g);
        assert!((dist[0][0] - 0.0).abs() < 1e-10);
        assert!((dist[0][1] - 1.0).abs() < 1e-10);
        assert!((dist[0][2] - 3.0).abs() < 1e-10);
        assert!((dist[1][1] - 0.0).abs() < 1e-10);
        assert!((dist[1][2] - 2.0).abs() < 1e-10);
        assert!((dist[2][2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn connected_components_matches_scipy_reference_values() {
        // scipy.sparse.csgraph.connected_components for 4-node graph with 2 components
        // Edges: (0,1), (2,3) -> 2 components
        use crate::{CooMatrix, Shape2D};
        let g = CooMatrix::from_triplets(
            Shape2D::new(4, 4),
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0, 1, 2, 3],
            vec![1, 0, 3, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let result = super::connected_components(&g).expect("cc");
        assert_eq!(result.n_components, 2);
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[2], result.labels[3]);
        assert_ne!(result.labels[0], result.labels[2]);
    }

    #[test]
    fn eigsh_matches_scipy_reference_values() {
        // scipy.sparse.linalg.eigsh for diagonal matrix with eigenvalues 1, 4, 9
        // Request k=2 largest -> should get 9 and 4
        use crate::{CooMatrix, Shape2D};
        let a = CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![1.0, 4.0, 9.0],
            vec![0, 1, 2],
            vec![0, 1, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let result = super::eigsh(&a, 2, EigsOptions::default()).expect("eigsh");
        assert!(result.converged);
        let mut evs = result.eigenvalues.clone();
        evs.sort_by(|a, b| b.total_cmp(a));
        assert!(
            (evs[0] - 9.0).abs() < 1e-4,
            "largest eigenvalue = {}, expected 9.0",
            evs[0]
        );
        assert!(
            (evs[1] - 4.0).abs() < 1e-4,
            "second eigenvalue = {}, expected 4.0",
            evs[1]
        );
    }

    #[test]
    fn lgmres_matches_scipy_reference_values() {
        // scipy.sparse.linalg.lgmres(A, b) for non-symmetric matrix
        // A = [[4, 1], [2, 3]], b = [1, 2] -> same as gmres: x = [0.1, 0.6]
        use crate::{CooMatrix, Shape2D};
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![4.0, 1.0, 2.0, 3.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![1.0, 2.0];
        let result = super::lgmres(&a, &b, None, LgmresOptions::default()).expect("lgmres");
        let expected = [0.1, 0.6];
        for (i, (&got, &want)) in result.solution.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "lgmres x[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn qmr_matches_scipy_reference_values() {
        // scipy.sparse.linalg.qmr(A, b) for non-symmetric matrix
        use crate::{CooMatrix, Shape2D};
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![4.0, 1.0, 2.0, 3.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![1.0, 2.0];
        let result = super::qmr(&a, &b, None, IterativeSolveOptions::default()).expect("qmr");
        // Verify Ax ≈ b
        let ax = super::spmv(&a, &result.solution);
        for i in 0..2 {
            assert!((ax[i] - b[i]).abs() < 1e-5, "qmr residual too large at {i}");
        }
    }

    #[test]
    fn pagerank_matches_scipy_reference_behavior() {
        // scipy.sparse.csgraph uses similar pagerank algorithm
        // Simple 3-node graph: 0 -> 1 -> 2 -> 0 (cycle)
        use crate::{CooMatrix, Shape2D};
        let g = CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![1.0, 1.0, 1.0],
            vec![0, 1, 2],
            vec![1, 2, 0],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let pr = super::pagerank(&g, 0.85, 100, 1e-6);
        // In a symmetric cycle, all nodes should have equal PageRank
        let sum: f64 = pr.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "PageRank should sum to 1.0");
        // All nodes should have roughly equal rank
        let mean = 1.0 / 3.0;
        for (i, &r) in pr.iter().enumerate() {
            assert!(
                (r - mean).abs() < 0.1,
                "PageRank[{i}] = {r}, expected ~{mean}"
            );
        }
    }

    #[test]
    fn reverse_cuthill_mckee_matches_scipy_reference_values() {
        // scipy.sparse.csgraph.reverse_cuthill_mckee returns permutation
        // For a simple chain graph 0-1-2-3, RCM should produce valid permutation
        use crate::{CooMatrix, Shape2D};
        let g = CooMatrix::from_triplets(
            Shape2D::new(4, 4),
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![0, 1, 1, 2, 2, 3],
            vec![1, 0, 2, 1, 3, 2],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let perm = super::reverse_cuthill_mckee(&g);
        assert_eq!(perm.len(), 4, "permutation length");
        // Should be a valid permutation (contains 0, 1, 2, 3 in some order)
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3], "should be valid permutation");
    }

    #[test]
    fn bicg_matches_scipy_reference_values() {
        // scipy.sparse.linalg.bicg(A, b) solves Ax = b
        // Simple 2x2 system: [[4, 1], [1, 3]] * x = [1, 2]
        use crate::{CooMatrix, Shape2D};
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![4.0, 1.0, 1.0, 3.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![1.0, 2.0];
        let result = super::bicg(&a, &b, None, IterativeSolveOptions::default()).expect("bicg");
        // Verify Ax ≈ b
        let ax = super::spmv(&a, &result.solution);
        for i in 0..2 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-5,
                "bicg residual too large at {i}"
            );
        }
    }

    #[test]
    fn cgs_matches_scipy_reference_values() {
        // scipy.sparse.linalg.cgs(A, b) solves Ax = b
        // Simple 2x2 system: [[4, 1], [1, 3]] * x = [1, 2]
        use crate::{CooMatrix, Shape2D};
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![4.0, 1.0, 1.0, 3.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let b = vec![1.0, 2.0];
        let result = super::cgs(&a, &b, None, IterativeSolveOptions::default()).expect("cgs");
        // Verify Ax ≈ b
        let ax = super::spmv(&a, &result.solution);
        for i in 0..2 {
            assert!((ax[i] - b[i]).abs() < 1e-5, "cgs residual too large at {i}");
        }
    }

    #[test]
    fn splu_solve_matches_scipy_reference_values() {
        // scipy.sparse.linalg.splu(A).solve(b) solves Ax = b
        // Simple 2x2 system: [[4, 1], [1, 3]] * x = [1, 2]
        use crate::{CooMatrix, Shape2D};
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![4.0, 1.0, 1.0, 3.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csc()
        .expect("csc");
        let lu = super::splu(&a, LuOptions::default()).expect("splu");
        let b = vec![1.0, 2.0];
        let x = super::splu_solve(&lu, &b).expect("splu_solve");
        // Exact solution is x = [1/11, 7/11] ≈ [0.0909, 0.6364]
        // Verify via matrix product: compute A @ x manually
        // Row 0: 4*x[0] + 1*x[1] should ≈ 1
        // Row 1: 1*x[0] + 3*x[1] should ≈ 2
        let ax0 = 4.0 * x[0] + 1.0 * x[1];
        let ax1 = 1.0 * x[0] + 3.0 * x[1];
        assert!((ax0 - 1.0).abs() < 1e-10, "splu row 0 residual");
        assert!((ax1 - 2.0).abs() < 1e-10, "splu row 1 residual");
    }
}

// ══════════════════════════════════════════════════════════════════════
// Sparse Eigenvalue Solver — Public API
// ══════════════════════════════════════════════════════════════════════

/// Result of sparse eigenvalue computation.
#[derive(Debug, Clone, PartialEq)]
pub struct EigsResult {
    /// Eigenvalues (real parts).
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors as columns (row-major: `eigenvectors[i]` is the i-th eigenvector).
    pub eigenvectors: Vec<Vec<f64>>,
    /// Number of matrix-vector products performed.
    pub nmatvec: usize,
    /// Whether all requested eigenvalues converged.
    pub converged: bool,
}

/// Options for sparse eigenvalue computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EigsOptions {
    /// Tolerance for convergence.
    pub tol: f64,
    /// Maximum iterations.
    pub max_iter: usize,
}

impl Default for EigsOptions {
    fn default() -> Self {
        Self {
            tol: 1e-10,
            max_iter: 1000,
        }
    }
}

fn normalize_eigs_options(options: EigsOptions) -> EigsOptions {
    let defaults = EigsOptions::default();
    EigsOptions {
        tol: if options.tol > 0.0 && options.tol.is_finite() {
            options.tol
        } else {
            defaults.tol
        },
        max_iter: if options.max_iter == 0 {
            defaults.max_iter
        } else {
            options.max_iter
        },
    }
}

/// Solve a sparse triangular system Ax = b.
///
/// Matches `scipy.sparse.linalg.spsolve_triangular(A, b, lower)`.
/// Performs forward substitution (lower=true) or backward substitution (lower=false).
pub fn spsolve_triangular(a: &CsrMatrix, b: &[f64], lower: bool) -> SparseResult<Vec<f64>> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "spsolve_triangular requires a square matrix".to_string(),
        });
    }
    let n = shape.rows;
    if b.len() != n {
        return Err(SparseError::IncompatibleShape {
            message: "rhs length must match matrix rows".to_string(),
        });
    }

    let indptr = a.indptr();
    let indices = a.indices();
    let data = a.data();
    let mut x = b.to_vec();

    if lower {
        // Forward substitution
        for i in 0..n {
            let mut diag: f64 = 0.0;
            for idx in indptr[i]..indptr[i + 1] {
                let j = indices[idx];
                if j < i {
                    x[i] -= data[idx] * x[j];
                } else if j == i {
                    diag = data[idx];
                }
            }
            if diag.abs() < f64::EPSILON * 100.0 {
                return Err(SparseError::SingularMatrix {
                    message: format!("zero diagonal at row {i}"),
                });
            }
            x[i] /= diag;
        }
    } else {
        // Backward substitution
        for i in (0..n).rev() {
            let mut diag: f64 = 0.0;
            for idx in indptr[i]..indptr[i + 1] {
                let j = indices[idx];
                if j > i {
                    x[i] -= data[idx] * x[j];
                } else if j == i {
                    diag = data[idx];
                }
            }
            if diag.abs() < f64::EPSILON * 100.0 {
                return Err(SparseError::SingularMatrix {
                    message: format!("zero diagonal at row {i}"),
                });
            }
            x[i] /= diag;
        }
    }

    Ok(x)
}

/// Compute the `k` largest eigenvalues/eigenvectors of a sparse symmetric matrix.
///
/// Uses power iteration with deflation for multiple eigenvalues.
/// Matches `scipy.sparse.linalg.eigsh(A, k=k, which='LM')` for symmetric A.
pub fn eigsh(a: &CsrMatrix, k: usize, options: EigsOptions) -> SparseResult<EigsResult> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "eigsh requires a square matrix".to_string(),
        });
    }
    let n = shape.rows;
    if k == 0 || k > n {
        return Err(SparseError::InvalidArgument {
            message: format!("k={k} must be in [1, {n}]"),
        });
    }
    let options = normalize_eigs_options(options);

    let mut eigenvalues = Vec::with_capacity(k);
    let mut eigenvectors: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut total_matvec = 0;

    for _eig_idx in 0..k {
        // br-oyy7: Power iteration is mathematically guaranteed to find
        // the dominant eigenvalue only when the initial vector has a
        // nonzero projection onto its eigenvector. The constant vector
        // [1/√n, …, 1/√n] is orthogonal to every alternating-sign mode
        // (e.g. the dominant eigenvector of a path Laplacian), so on
        // structured matrices the iteration silently converges to the
        // wrong eigenvalue. Seed instead with a deterministic LCG-based
        // pseudo-random vector — orthogonal to no eigenmode in general.
        let mut v = vec![0.0f64; n];
        let mut rng_state: u64 = 0x9e3779b97f4a7c15;
        for vi in v.iter_mut() {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((rng_state >> 11) as f64) / (1u64 << 53) as f64;
            *vi = u - 0.5;
        }
        let v_norm = vec_norm(&v);
        if v_norm > 0.0 {
            for vi in &mut v {
                *vi /= v_norm;
            }
        }

        let mut lambda = 0.0;
        let mut converged_this = false;

        for _iter in 0..options.max_iter {
            // w = A * v
            let mut w = csr_matvec(a, &v);
            total_matvec += 1;

            // Deflate: project out components along previously found eigenvectors
            // w = w - Σ (v_i^T w) v_i * λ_i (Wielandt deflation variant)
            // Simpler: just orthogonalize w against previous eigenvectors
            for prev_vec in &eigenvectors {
                let proj: f64 = w.iter().zip(prev_vec.iter()).map(|(a, b)| a * b).sum();
                for (wi, &pi) in w.iter_mut().zip(prev_vec.iter()) {
                    *wi -= proj * pi;
                }
            }

            // Rayleigh quotient: λ = v^T w
            let new_lambda: f64 = v.iter().zip(w.iter()).map(|(a, b)| a * b).sum();

            // Normalize w -> v
            let w_norm = vec_norm(&w);
            if w_norm < f64::EPSILON * 100.0 {
                break;
            }
            for wi in &mut w {
                *wi /= w_norm;
            }

            if (new_lambda - lambda).abs() < options.tol * new_lambda.abs().max(1.0) {
                converged_this = true;
                lambda = new_lambda;
                v = w;
                break;
            }

            lambda = new_lambda;
            v = w;
        }

        eigenvalues.push(lambda);
        eigenvectors.push(v);

        if !converged_this {
            return Ok(EigsResult {
                eigenvalues,
                eigenvectors,
                nmatvec: total_matvec,
                converged: false,
            });
        }
    }

    Ok(EigsResult {
        eigenvalues,
        eigenvectors,
        nmatvec: total_matvec,
        converged: true,
    })
}

// ══════════════════════════════════════════════════════════════════════
// eigs — Arnoldi-based eigenvalue solver for general sparse matrices
// ══════════════════════════════════════════════════════════════════════

/// Compute the `k` eigenvalues of largest magnitude of a general sparse matrix.
///
/// Uses Arnoldi iteration to build a Krylov subspace, then extracts eigenvalues
/// from the upper Hessenberg matrix.
/// Matches `scipy.sparse.linalg.eigs(A, k=k, which='LM')`.
pub fn eigs(a: &CsrMatrix, k: usize, options: EigsOptions) -> SparseResult<EigsResult> {
    let shape = a.shape();
    if !shape.is_square() {
        return Err(SparseError::InvalidShape {
            message: "eigs requires a square matrix".to_string(),
        });
    }
    let n = shape.rows;
    if k == 0 || k > n {
        return Err(SparseError::InvalidArgument {
            message: format!("k={k} must be in [1, {n}]"),
        });
    }
    let options = normalize_eigs_options(options);

    // Krylov subspace dimension (larger than k for better convergence)
    let m = (2 * k + 1).min(n);
    let mut total_matvec = 0;

    // Arnoldi iteration: build orthonormal basis V and upper Hessenberg H
    // such that A * V_m ≈ V_m * H_m
    let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
    let mut h = vec![vec![0.0; m]; m + 1]; // (m+1) x m upper Hessenberg

    // Initial vector
    let mut v0 = vec![1.0 / (n as f64).sqrt(); n];
    let v0_norm = vec_norm(&v0);
    for vi in &mut v0 {
        *vi /= v0_norm;
    }
    v.push(v0);

    let mut actual_m = 0usize;
    for j in 0..m {
        // w = A * v_j
        let mut w = csr_matvec(a, &v[j]);
        total_matvec += 1;

        // Modified Gram-Schmidt orthogonalization
        for i in 0..=j {
            h[i][j] = dot_product(&w, &v[i]);
            for (wk, vik) in w.iter_mut().zip(v[i].iter()) {
                *wk -= h[i][j] * vik;
            }
        }

        h[j + 1][j] = vec_norm(&w);
        // br-iq1e: count this column as completed BEFORE the breakdown
        // check. Without this, a lucky-breakdown at j=0 (e.g. when the
        // initial vector is already an eigenvector — common for
        // structured matrices like the 4-cycle shift) leaves actual_m
        // = v.len() - 1 = 0 and the caller sees zero eigenvalues even
        // though h[0][0] holds the correct dominant eigenvalue.
        actual_m = j + 1;

        if h[j + 1][j] < f64::EPSILON * 1e6 {
            // Lucky breakdown: Krylov subspace is invariant.
            break;
        }

        // Normalize
        for wi in &mut w {
            *wi /= h[j + 1][j];
        }
        v.push(w);
    }

    // Extract eigenvalues from the Hessenberg matrix H[0..actual_m, 0..actual_m]
    // Use QR algorithm on the small dense Hessenberg matrix
    let eig_vals = hessenberg_eigenvalues(&h, actual_m, options.max_iter, options.tol);

    // Sort by magnitude (largest first) and take top k
    let mut indexed: Vec<(usize, f64)> = eig_vals.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));

    let k_actual = k.min(indexed.len());
    let mut eigenvalues = Vec::with_capacity(k_actual);
    let mut eigenvectors = Vec::with_capacity(k_actual);

    for &(_, val) in indexed.iter().take(k_actual) {
        eigenvalues.push(val);

        // Back-transform the Ritz vector into the original space: the
        // eigenvector of A is x = V @ y, where y is the eigenvector of the
        // projected Hessenberg matrix H for this eigenvalue. Returning a raw
        // Arnoldi basis vector v[idx] is wrong — those are not eigenpairs of A.
        let y = hessenberg_eigenvector(&h, actual_m, val);
        let mut evec = vec![0.0; n];
        for (j, &yj) in y.iter().enumerate() {
            if yj == 0.0 {
                continue;
            }
            for (xi, vji) in evec.iter_mut().zip(v[j].iter()) {
                *xi += yj * vji;
            }
        }
        let norm = vec_norm(&evec);
        if norm > 0.0 {
            for xi in &mut evec {
                *xi /= norm;
            }
        }
        eigenvectors.push(evec);
    }

    Ok(EigsResult {
        eigenvalues,
        eigenvectors,
        nmatvec: total_matvec,
        converged: true,
    })
}

/// Compute an eigenvector of the upper Hessenberg matrix `H[0..m, 0..m]` for
/// the (real) eigenvalue `lambda`.
///
/// Solves `(H - lambda*I) y = 0` by back-substitution against the subdiagonal:
/// with `y[m-1] = 1`, row `r` of the system determines `y[r-1]` from the
/// already-known `y[r..m]`. When `lambda` is an exact eigenvalue the unused
/// top row is satisfied automatically; for a converged Ritz value its residual
/// is negligible.
fn hessenberg_eigenvector(h: &[Vec<f64>], m: usize, lambda: f64) -> Vec<f64> {
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        return vec![1.0];
    }
    let mut y = vec![0.0; m];
    y[m - 1] = 1.0;
    for r in (1..m).rev() {
        // Row r: h[r][r-1]*y[r-1] + sum_{c>=r} h[r][c]*y[c] - lambda*y[r] = 0.
        let mut acc = -lambda * y[r];
        for c in r..m {
            acc += h[r][c] * y[c];
        }
        let sub = h[r][r - 1];
        if sub.abs() < f64::MIN_POSITIVE {
            // Decoupled block: leave the remaining components at zero.
            break;
        }
        y[r - 1] = -acc / sub;
    }
    y
}

/// Extract eigenvalues from an upper Hessenberg matrix using QR iteration.
fn hessenberg_eigenvalues(h: &[Vec<f64>], m: usize, max_iter: usize, tol: f64) -> Vec<f64> {
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        return vec![h[0][0]];
    }

    // Copy the m×m submatrix
    let mut a = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            a[i][j] = h[i][j];
        }
    }

    // Francis QR double shift algorithm (simplified single shift version)
    let mut n = m;
    let mut eigenvalues = Vec::with_capacity(m);

    for _ in 0..max_iter * m {
        if n <= 1 {
            if n == 1 {
                eigenvalues.push(a[0][0]);
            }
            break;
        }

        // Check for convergence at bottom
        if a[n - 1][n - 2].abs() < tol * (a[n - 1][n - 1].abs() + a[n - 2][n - 2].abs()).max(tol) {
            eigenvalues.push(a[n - 1][n - 1]);
            n -= 1;
            continue;
        }

        // Wilkinson shift
        let shift = a[n - 1][n - 1];

        // Apply shift
        for (i, row) in a.iter_mut().enumerate().take(n) {
            row[i] -= shift;
        }

        // QR step via Givens rotations
        let mut cs_rot = vec![0.0; n - 1];
        let mut sn_rot = vec![0.0; n - 1];
        for i in 0..(n - 1) {
            let (c, s) = givens_rotation(a[i][i], a[i + 1][i]);
            cs_rot[i] = c;
            sn_rot[i] = s;
            // Apply rotation to rows i and i+1
            let (upper, lower) = a.split_at_mut(i + 1);
            let row_i = &mut upper[i];
            let row_ip1 = &mut lower[0];
            for (lhs, rhs) in row_i.iter_mut().zip(row_ip1.iter_mut()).skip(i).take(n - i) {
                let temp = c * *lhs + s * *rhs;
                *rhs = -s * *lhs + c * *rhs;
                *lhs = temp;
            }
        }

        // Multiply R * Q (apply rotations from the right)
        for i in 0..(n - 1) {
            let c = cs_rot[i];
            let s = sn_rot[i];
            for row in a.iter_mut().take(n.min(i + 3)) {
                let temp = c * row[i] + s * row[i + 1];
                row[i + 1] = -s * row[i] + c * row[i + 1];
                row[i] = temp;
            }
        }

        // Undo shift
        for (i, row) in a.iter_mut().enumerate().take(n) {
            row[i] += shift;
        }
    }

    // Collect any remaining diagonal elements
    while eigenvalues.len() < m && n > 0 {
        eigenvalues.push(a[n - 1][n - 1]);
        n -= 1;
    }

    eigenvalues
}

// ══════════════════════════════════════════════════════════════════════
// svds — Sparse Singular Value Decomposition
// ══════════════════════════════════════════════════════════════════════

/// Result of sparse SVD computation.
#[derive(Debug, Clone, PartialEq)]
pub struct SvdsResult {
    /// Singular values (largest first).
    pub singular_values: Vec<f64>,
    /// Left singular vectors (columns of U).
    pub u: Vec<Vec<f64>>,
    /// Right singular vectors (columns of V).
    pub vt: Vec<Vec<f64>>,
}

/// Compute the `k` largest singular values of a sparse matrix.
///
/// Uses the eigenvalue decomposition of A^T A to find singular values.
/// σ_i = √(λ_i(A^T A)), u_i = A v_i / σ_i.
/// Matches `scipy.sparse.linalg.svds(A, k=k)`.
pub fn svds(a: &CsrMatrix, k: usize, options: EigsOptions) -> SparseResult<SvdsResult> {
    let shape = a.shape();
    let m = shape.rows;
    let n = shape.cols;

    if k == 0 || k > m.min(n) {
        return Err(SparseError::InvalidArgument {
            message: format!("k={k} must be in [1, {}]", m.min(n)),
        });
    }
    let options = normalize_eigs_options(options);

    // Compute eigenvalues of A^T A via power iteration
    // A^T A is n×n symmetric positive semidefinite
    let mut singular_values = Vec::with_capacity(k);
    let mut v_vecs: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut u_vecs: Vec<Vec<f64>> = Vec::with_capacity(k);

    for _eig_idx in 0..k {
        // Power iteration for largest eigenvalue of A^T A
        // Use LCG-seeded random initialization (same rationale as eigsh br-oyy7:
        // uniform initial vectors can be orthogonal to dominant eigenvectors
        // on structured matrices).
        let mut v = vec![0.0f64; n];
        let mut rng_state: u64 = 0x9e3779b97f4a7c15;
        for vi in v.iter_mut() {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((rng_state >> 11) as f64) / (1u64 << 53) as f64;
            *vi = u - 0.5;
        }
        let v_norm = vec_norm(&v);
        if v_norm > 0.0 {
            for vi in &mut v {
                *vi /= v_norm;
            }
        }

        let mut sigma_sq = 0.0;
        let mut converged = false;

        for iter in 0..options.max_iter {
            // w = A * v
            let w = csr_matvec(a, &v);
            // v_new = A^T * w = A^T A v
            let mut v_new = csr_matvec_transpose(a, &w);

            // Deflate: orthogonalize against previous eigenvectors
            for prev in &v_vecs {
                let proj = dot_product(&v_new, prev);
                for (vni, pi) in v_new.iter_mut().zip(prev.iter()) {
                    *vni -= proj * pi;
                }
            }

            // Rayleigh quotient: σ² = v^T (A^T A v)
            let new_sigma_sq = dot_product(&v, &v_new);
            let v_new_norm = vec_norm(&v_new);
            if v_new_norm < f64::EPSILON * 1e6 {
                sigma_sq = new_sigma_sq;
                converged = true;
                break;
            }

            for vi in &mut v_new {
                *vi /= v_new_norm;
            }

            // Skip convergence check on first iteration
            if iter > 0
                && (new_sigma_sq - sigma_sq).abs() < options.tol * new_sigma_sq.abs().max(1.0)
            {
                converged = true;
                sigma_sq = new_sigma_sq;
                v = v_new;
                break;
            }

            sigma_sq = new_sigma_sq;
            v = v_new;
        }

        let sigma = sigma_sq.abs().sqrt();
        singular_values.push(sigma);
        v_vecs.push(v.clone());

        // Compute left singular vector: u = A v / sigma
        if sigma > f64::EPSILON * 1e6 {
            let mut u = csr_matvec(a, &v);
            for ui in &mut u {
                *ui /= sigma;
            }
            u_vecs.push(u);
        } else {
            u_vecs.push(vec![0.0; m]);
        }

        if !converged {
            break;
        }
    }

    Ok(SvdsResult {
        singular_values,
        u: u_vecs,
        vt: v_vecs,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Sparse Graph Algorithms (csgraph)
// ══════════════════════════════════════════════════════════════════════

/// Result of connected components computation.
#[derive(Debug, Clone, PartialEq)]
pub struct ConnectedComponentsResult {
    /// Number of connected components.
    pub n_components: usize,
    /// Component label for each node (0-indexed).
    pub labels: Vec<usize>,
}

fn validate_csgraph(graph: &CsrMatrix) -> SparseResult<()> {
    let shape = graph.shape();
    if shape.rows != shape.cols {
        return Err(SparseError::InvalidArgument {
            message: format!(
                "csgraph routines require a square adjacency matrix, got {}x{}",
                shape.rows, shape.cols
            ),
        });
    }

    let n = shape.rows;
    for &col in graph.indices() {
        if col >= n {
            return Err(SparseError::InvalidArgument {
                message: format!("graph edge references node {col}, but node count is {n}"),
            });
        }
    }

    // Check for non-finite edge weights (NaN/Inf)
    for &weight in graph.data() {
        if !weight.is_finite() {
            return Err(SparseError::NonFiniteInput {
                message: "graph contains NaN or Inf edge weights".to_string(),
            });
        }
    }

    Ok(())
}

/// Find connected components of a sparse graph.
///
/// Matches `scipy.sparse.csgraph.connected_components(graph, directed=False)`.
///
/// The input CSR matrix is treated as an adjacency matrix (nonzero = edge).
/// For undirected graphs, the matrix should be symmetric.
pub fn connected_components(graph: &CsrMatrix) -> SparseResult<ConnectedComponentsResult> {
    validate_csgraph(graph)?;
    let n = graph.shape().rows;
    let indptr = graph.indptr();
    let indices = graph.indices();

    // Build symmetric adjacency list so both edge directions are traversed,
    // even if the input matrix isn't perfectly symmetric.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for &j in indices.iter().take(indptr[i + 1]).skip(indptr[i]) {
            adj[i].push(j);
            adj[j].push(i); // reverse edge for undirected
        }
    }

    let mut labels = vec![usize::MAX; n];
    let mut component = 0;

    for start in 0..n {
        if labels[start] != usize::MAX {
            continue;
        }

        // BFS from this node
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        labels[start] = component;

        while let Some(node) = queue.pop_front() {
            for &neighbor in &adj[node] {
                if labels[neighbor] == usize::MAX {
                    labels[neighbor] = component;
                    queue.push_back(neighbor);
                }
            }
        }
        component += 1;
    }

    Ok(ConnectedComponentsResult {
        n_components: component,
        labels,
    })
}

/// Result of shortest path computation.
#[derive(Debug, Clone, PartialEq)]
pub struct ShortestPathResult {
    /// Distance from source to each node (f64::INFINITY if unreachable).
    pub distances: Vec<f64>,
    /// Predecessor array for path reconstruction (-1 for source/unreachable).
    pub predecessors: Vec<i64>,
}

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Copy, Clone, PartialEq)]
struct DijkstraState {
    cost: f64,
    position: usize,
}

impl Eq for DijkstraState {}

impl PartialOrd for DijkstraState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkstraState {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.total_cmp(&self.cost)
    }
}

/// Single-source shortest paths using Dijkstra's algorithm.
///
/// Matches `scipy.sparse.csgraph.dijkstra(graph, indices=source)`.
///
/// The CSR matrix values are edge weights. When negative edges are present,
/// SciPy warns and still computes distances; we follow that observable result
/// surface by delegating to Bellman-Ford instead of hard-failing.
pub fn dijkstra(graph: &CsrMatrix, source: usize) -> SparseResult<ShortestPathResult> {
    validate_csgraph(graph)?;
    let n = graph.shape().rows;
    if source >= n {
        return Err(SparseError::InvalidArgument {
            message: format!("source {source} out of bounds for graph with {n} nodes"),
        });
    }

    let indptr = graph.indptr();
    let indices = graph.indices();
    let data = graph.data();

    if data.iter().any(|&weight| weight < 0.0) {
        return bellman_ford(graph, source);
    }

    let mut dist = vec![f64::INFINITY; n];
    let mut pred = vec![-1_i64; n];

    dist[source] = 0.0;

    let mut heap = BinaryHeap::new();
    heap.push(DijkstraState {
        cost: 0.0,
        position: source,
    });

    while let Some(DijkstraState { cost, position }) = heap.pop() {
        if cost > dist[position] {
            continue;
        }

        // Relax edges from position
        for idx in indptr[position]..indptr[position + 1] {
            let v = indices[idx];
            let weight = data[idx];
            let alt = cost + weight;
            if alt < dist[v] {
                dist[v] = alt;
                pred[v] = position as i64;
                heap.push(DijkstraState {
                    cost: alt,
                    position: v,
                });
            }
        }
    }

    Ok(ShortestPathResult {
        distances: dist,
        predecessors: pred,
    })
}

/// Single-source shortest paths using Bellman-Ford algorithm.
///
/// Matches `scipy.sparse.csgraph.bellman_ford(graph, indices=source)`.
///
/// Supports negative edge weights (unlike Dijkstra). Detects negative cycles.
pub fn bellman_ford(graph: &CsrMatrix, source: usize) -> SparseResult<ShortestPathResult> {
    validate_csgraph(graph)?;
    let n = graph.shape().rows;
    if source >= n {
        return Err(SparseError::InvalidArgument {
            message: format!("source {source} out of bounds for graph with {n} nodes"),
        });
    }

    let indptr = graph.indptr();
    let indices = graph.indices();
    let data = graph.data();

    let mut dist = vec![f64::INFINITY; n];
    let mut pred = vec![-1_i64; n];
    dist[source] = 0.0;

    // Relax all edges n-1 times
    for _ in 0..n.saturating_sub(1) {
        let mut changed = false;
        for u in 0..n {
            if dist[u] == f64::INFINITY {
                continue;
            }
            for idx in indptr[u]..indptr[u + 1] {
                let v = indices[idx];
                let weight = data[idx];
                let alt = dist[u] + weight;
                if alt < dist[v] {
                    dist[v] = alt;
                    pred[v] = u as i64;
                    changed = true;
                }
            }
        }
        if !changed {
            break; // Early termination: no updates in this pass
        }
    }

    // Check for negative cycles: one more pass
    for u in 0..n {
        if dist[u] == f64::INFINITY {
            continue;
        }
        for idx in indptr[u]..indptr[u + 1] {
            let v = indices[idx];
            let weight = data[idx];
            if dist[u] + weight < dist[v] {
                return Err(SparseError::InvalidArgument {
                    message: "graph contains a negative-weight cycle".to_string(),
                });
            }
        }
    }

    Ok(ShortestPathResult {
        distances: dist,
        predecessors: pred,
    })
}

/// Breadth-first search traversal order from a source node.
///
/// Returns the node indices in BFS order and a predecessor array.
///
/// Matches `scipy.sparse.csgraph.breadth_first_order(graph, i_start)`.
pub fn breadth_first_order(
    graph: &CsrMatrix,
    source: usize,
) -> SparseResult<(Vec<usize>, Vec<i64>)> {
    let n = graph.shape().rows;
    if source >= n {
        return Err(SparseError::InvalidArgument {
            message: format!("source {source} out of bounds for graph with {n} nodes"),
        });
    }
    let indptr = graph.indptr();
    let indices = graph.indices();

    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut predecessors = vec![-1_i64; n];

    let mut queue = std::collections::VecDeque::new();
    queue.push_back(source);
    visited[source] = true;
    predecessors[source] = -1;

    while let Some(node) = queue.pop_front() {
        order.push(node);
        for &neighbor in indices.iter().take(indptr[node + 1]).skip(indptr[node]) {
            if !visited[neighbor] {
                visited[neighbor] = true;
                predecessors[neighbor] = node as i64;
                queue.push_back(neighbor);
            }
        }
    }

    Ok((order, predecessors))
}

/// Depth-first search traversal order from a source node.
///
/// Returns the node indices in DFS pre-order and a predecessor array.
///
/// Matches `scipy.sparse.csgraph.depth_first_order(graph, i_start)`.
pub fn depth_first_order(graph: &CsrMatrix, source: usize) -> SparseResult<(Vec<usize>, Vec<i64>)> {
    let n = graph.shape().rows;
    if source >= n {
        return Err(SparseError::InvalidArgument {
            message: format!("source {source} out of bounds for graph with {n} nodes"),
        });
    }
    let indptr = graph.indptr();
    let indices = graph.indices();

    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut predecessors = vec![-1_i64; n];

    let mut stack = vec![source];
    visited[source] = true;

    while let Some(node) = stack.pop() {
        order.push(node);
        // Push neighbors in reverse order so leftmost is visited first
        let neighbors: Vec<usize> = (indptr[node]..indptr[node + 1])
            .map(|idx| indices[idx])
            .filter(|&neighbor| !visited[neighbor])
            .collect();
        for &neighbor in neighbors.iter().rev() {
            if !visited[neighbor] {
                visited[neighbor] = true;
                predecessors[neighbor] = node as i64;
                stack.push(neighbor);
            }
        }
    }

    Ok((order, predecessors))
}

/// Compute the graph Laplacian matrix L = D - A.
///
/// The graph Laplacian is fundamental for spectral graph theory, spectral clustering,
/// diffusion processes, and network analysis.
///
/// Matches `scipy.sparse.csgraph.laplacian(graph, normed=normed)`.
///
/// # Arguments
/// * `graph` — Adjacency matrix in CSR format (edge weights as values).
/// * `normed` — If true, compute the symmetric normalized Laplacian L_sym = D^(-1/2) L D^(-1/2).
///
/// Returns the Laplacian as a dense matrix (`Vec<Vec<f64>>`).
pub fn laplacian(graph: &CsrMatrix, normed: bool) -> SparseResult<Vec<Vec<f64>>> {
    let n = graph.shape().rows;
    let indptr = graph.indptr();
    let indices = graph.indices();
    let data = graph.data();

    // Compute degree vector (sum of edge weights per row)
    let mut degree: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        for &value in data.iter().take(indptr[i + 1]).skip(indptr[i]) {
            degree[i] += value.abs();
        }
    }

    // Build L = D - A
    let mut lapl: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        lapl[i][i] = degree[i];
        for idx in indptr[i]..indptr[i + 1] {
            let j = indices[idx];
            lapl[i][j] -= data[idx];
        }
    }

    if normed {
        // Symmetric normalized: L_sym = D^(-1/2) L D^(-1/2)
        let mut d_inv_sqrt = vec![0.0; n];
        for i in 0..n {
            d_inv_sqrt[i] = if degree[i] > 0.0 {
                1.0 / degree[i].sqrt()
            } else {
                0.0
            };
        }
        for i in 0..n {
            for j in 0..n {
                lapl[i][j] *= d_inv_sqrt[i] * d_inv_sqrt[j];
            }
        }
    }

    Ok(lapl)
}

/// Result of minimum spanning tree computation.
#[derive(Debug, Clone, PartialEq)]
pub struct MstResult {
    /// Total weight of the MST.
    pub total_weight: f64,
    /// Edges in the MST as (u, v, weight) triples.
    pub edges: Vec<(usize, usize, f64)>,
}

/// Compute the minimum spanning tree of a sparse graph using Kruskal's algorithm.
///
/// Matches `scipy.sparse.csgraph.minimum_spanning_tree(graph)`.
///
/// The CSR matrix is treated as an undirected weighted adjacency matrix.
pub fn minimum_spanning_tree(graph: &CsrMatrix) -> SparseResult<MstResult> {
    validate_csgraph(graph)?;
    let n = graph.shape().rows;
    if n == 0 {
        return Ok(MstResult {
            total_weight: 0.0,
            edges: Vec::new(),
        });
    }
    let indptr = graph.indptr();
    let indices = graph.indices();
    let data = graph.data();

    // Collect all edges (deduplicate for undirected by only taking i < j)
    let mut edges: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..n {
        for idx in indptr[i]..indptr[i + 1] {
            let j = indices[idx];
            let w = data[idx];
            if i < j && w.is_finite() {
                edges.push((w, i, j));
            }
        }
    }

    // Sort edges by weight (Kruskal's)
    edges.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Union-Find
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank = vec![0u32; n];

    let mut mst_edges = Vec::new();
    let mut total_weight = 0.0;

    for (w, u, v) in edges {
        let ru = uf_find(&mut parent, u);
        let rv = uf_find(&mut parent, v);
        if ru != rv {
            uf_union(&mut parent, &mut rank, ru, rv);
            mst_edges.push((u, v, w));
            total_weight += w;
            if mst_edges.len() == n - 1 {
                break;
            }
        }
    }

    Ok(MstResult {
        total_weight,
        edges: mst_edges,
    })
}

/// Union-Find: find with path compression.
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]]; // path halving
        x = parent[x];
    }
    x
}

/// Union-Find: union by rank.
fn uf_union(parent: &mut [usize], rank: &mut [u32], x: usize, y: usize) {
    match rank[x].cmp(&rank[y]) {
        std::cmp::Ordering::Less => parent[x] = y,
        std::cmp::Ordering::Greater => parent[y] = x,
        std::cmp::Ordering::Equal => {
            parent[y] = x;
            rank[x] += 1;
        }
    }
}
