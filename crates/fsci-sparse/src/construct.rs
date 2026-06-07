use std::collections::HashSet;

use crate::formats::{
    BsrMatrix, CooMatrix, CscMatrix, CsrMatrix, DiaMatrix, DokMatrix, LilMatrix, Shape2D,
    SparseError, SparseFormat, SparseResult,
};
use crate::ops::{FormatConvertible, add_csr};

pub fn eye(size: usize) -> SparseResult<CsrMatrix> {
    let shape = Shape2D::new(size, size);
    let data = vec![1.0; size];
    let rows: Vec<usize> = (0..size).collect();
    let cols = rows.clone();
    let coo = CooMatrix::from_triplets(shape, data, rows, cols, false)?;
    coo.to_csr()
}

/// Construct a square sparse identity matrix.
///
/// Mirrors `scipy.sparse.identity(n)`. The Rust constructor returns CSR to
/// match the existing `eye` constructor convention in this crate.
pub fn identity(size: usize) -> SparseResult<CsrMatrix> {
    eye(size)
}

/// Construct a `rows × cols` sparse identity-like matrix with the
/// `k`-th diagonal set to one (all other entries zero).
///
/// Matches `scipy.sparse.eye(m, n, k)`. `k = 0` is the main diagonal,
/// `k > 0` shifts to the upper diagonal (col index = row index + k),
/// `k < 0` shifts to the lower diagonal. For empty matrices or
/// out-of-range `k`, returns an explicit-zero CSR.
///
/// Resolves [frankenscipy-xj9sq].
pub fn eye_rectangular(rows: usize, cols: usize, k: isize) -> SparseResult<CsrMatrix> {
    let shape = Shape2D::new(rows, cols);
    // Determine the valid range of indices on the k-th diagonal:
    //   row range: [max(0, -k), min(rows, cols - k))
    let (row_start, length) = if k >= 0 {
        let k_us = k as usize;
        if k_us >= cols {
            (0usize, 0usize)
        } else {
            (0usize, rows.min(cols - k_us))
        }
    } else {
        let k_abs = (-k) as usize;
        if k_abs >= rows {
            (0usize, 0usize)
        } else {
            (k_abs, (rows - k_abs).min(cols))
        }
    };
    let data = vec![1.0; length];
    let r: Vec<usize> = (row_start..row_start + length).collect();
    let c: Vec<usize> = if k >= 0 {
        (k as usize..k as usize + length).collect()
    } else {
        (0..length).collect()
    };
    let coo = CooMatrix::from_triplets(shape, data, r, c, false)?;
    coo.to_csr()
}

pub fn diags(
    diagonals: &[Vec<f64>],
    offsets: &[isize],
    shape: Option<Shape2D>,
) -> SparseResult<CsrMatrix> {
    if diagonals.len() != offsets.len() {
        return Err(SparseError::InvalidArgument {
            message: "diagonals and offsets lengths must match".to_string(),
        });
    }

    let mut seen = HashSet::new();
    for &offset in offsets {
        if !seen.insert(offset) {
            return Err(SparseError::InvalidArgument {
                message: "repeated diagonal offsets are not allowed".to_string(),
            });
        }
    }

    let inferred = infer_shape(diagonals, offsets)?;
    let shape = shape.unwrap_or(inferred);

    for (diag, &offset) in diagonals.iter().zip(offsets.iter()) {
        let start_row = if offset < 0 { (-offset) as usize } else { 0 };
        let start_col = if offset > 0 { offset as usize } else { 0 };
        for k in 0..diag.len() {
            let row = start_row + k;
            let col = start_col + k;
            if row >= shape.rows || col >= shape.cols {
                return Err(SparseError::InvalidShape {
                    message: "diagonal length exceeds matrix shape bounds".to_string(),
                });
            }
        }
    }

    let capacity = diagonals
        .iter()
        .try_fold(0usize, |total, diag| total.checked_add(diag.len()))
        .ok_or_else(|| SparseError::IndexOverflow {
            message: "diagonal entry count overflows usize".to_string(),
        })?;
    let mut diagonal_order: Vec<(isize, &[f64])> = diagonals
        .iter()
        .zip(offsets.iter().copied())
        .map(|(diag, offset)| (offset, diag.as_slice()))
        .collect();
    diagonal_order.sort_unstable_by_key(|&(offset, _)| offset);

    let mut data = Vec::with_capacity(capacity);
    let mut indices = Vec::with_capacity(capacity);
    let mut indptr = Vec::with_capacity(shape.rows + 1);
    indptr.push(0);

    for row in 0..shape.rows {
        for &(offset, diag) in &diagonal_order {
            let entry = if offset >= 0 {
                let col = row + offset as usize;
                if col < shape.cols && row < diag.len() {
                    Some((col, diag[row]))
                } else {
                    None
                }
            } else {
                let row_offset = (-offset) as usize;
                row.checked_sub(row_offset)
                    .filter(|&k| k < diag.len())
                    .map(|k| (k, diag[k]))
            };
            if let Some((col, value)) = entry {
                indices.push(col);
                data.push(value);
            }
        }
        indptr.push(data.len());
    }

    CsrMatrix::from_components(shape, data, indices, indptr, true)
}

/// Construct a sparse DIA matrix from SciPy-style padded diagonal rows.
///
/// Mirrors `scipy.sparse.spdiags(data, diags, m, n)`: each row in `data`
/// contains values indexed by output column, including any leading padding for
/// positive offsets.
pub fn spdiags(
    data: &[Vec<f64>],
    offsets: &[isize],
    rows: usize,
    cols: usize,
) -> SparseResult<DiaMatrix> {
    DiaMatrix::new(Shape2D::new(rows, cols), data.to_vec(), offsets.to_vec())
}

pub fn random(shape: Shape2D, density: f64, seed: u64) -> SparseResult<CooMatrix> {
    if !(0.0..=1.0).contains(&density) {
        return Err(SparseError::InvalidArgument {
            message: "density must be in [0.0, 1.0]".to_string(),
        });
    }
    let total = shape
        .rows
        .checked_mul(shape.cols)
        .ok_or_else(|| SparseError::IndexOverflow {
            message: "rows * cols overflows usize".to_string(),
        })?;

    let mut state = seed.max(1);
    let mut rows: Vec<usize>;
    let mut cols: Vec<usize>;
    let mut data: Vec<f64>;

    // Per frankenscipy-sw4o: switch from O(rows*cols) dense Bernoulli
    // sampling to O(nnz) position sampling whenever density * total would
    // produce fewer than total/8 expected entries. That threshold keeps
    // the legacy O(total) path for dense-like densities (where position
    // rejection sampling is dominated by collisions) and takes the
    // O(nnz) path everywhere else.
    let expected_nnz_f = density * total as f64;
    let expected_nnz = expected_nnz_f.round().max(0.0).min(total as f64) as usize;

    if expected_nnz > 0 && expected_nnz <= total / 8 {
        // O(nnz) path via flat-index sampling with dedupe.
        let mut seen: HashSet<usize> = HashSet::with_capacity(expected_nnz);
        rows = Vec::with_capacity(expected_nnz);
        cols = Vec::with_capacity(expected_nnz);
        data = Vec::with_capacity(expected_nnz);
        while seen.len() < expected_nnz {
            state = xorshift64(state);
            let flat = (state as usize) % total;
            if seen.insert(flat) {
                let row = flat / shape.cols.max(1);
                let col = flat % shape.cols.max(1);
                state = xorshift64(state);
                let value = ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0;
                rows.push(row.min(shape.rows.saturating_sub(1)));
                cols.push(col.min(shape.cols.saturating_sub(1)));
                data.push(value);
            }
        }
    } else {
        // O(rows*cols) legacy path for dense-like densities.
        rows = Vec::new();
        cols = Vec::new();
        data = Vec::new();
        for index in 0..total {
            state = xorshift64(state);
            let sample = (state as f64) / (u64::MAX as f64);
            if sample <= density {
                let row = index / shape.cols.max(1);
                let col = index % shape.cols.max(1);
                state = xorshift64(state);
                let value = ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0;
                rows.push(row.min(shape.rows.saturating_sub(1)));
                cols.push(col.min(shape.cols.saturating_sub(1)));
                data.push(value);
            }
        }
    }

    CooMatrix::from_triplets(shape, data, rows, cols, true)
}

fn infer_shape(diagonals: &[Vec<f64>], offsets: &[isize]) -> SparseResult<Shape2D> {
    let mut rows = 0usize;
    let mut cols = 0usize;
    for (diag, &offset) in diagonals.iter().zip(offsets.iter()) {
        let len = diag.len();
        if offset >= 0 {
            rows = rows.max(len);
            cols = cols.max(len + offset as usize);
        } else {
            let abs = (-offset) as usize;
            rows = rows.max(len + abs);
            cols = cols.max(len);
        }
    }
    if rows == 0 && cols == 0 {
        return Err(SparseError::InvalidShape {
            message: "cannot infer shape from empty diagonals".to_string(),
        });
    }
    Ok(Shape2D::new(rows, cols))
}

/// Construct a block diagonal sparse matrix from a list of sparse matrices.
///
/// Matches `scipy.sparse.block_diag(mats)`.
pub fn block_diag(matrices: &[&CsrMatrix]) -> SparseResult<CsrMatrix> {
    if matrices.is_empty() {
        return eye(0);
    }

    // Per frankenscipy-sw4o: checked_add instead of plain .sum() so huge
    // matrices can't silently wrap to a small total (debug panics instead
    // of release-wrapping; either path is worse than an explicit error).
    let total_rows: usize = matrices.iter().try_fold(0usize, |acc, m| {
        acc.checked_add(m.shape().rows)
            .ok_or_else(|| SparseError::IndexOverflow {
                message: "block_diag total rows overflow".to_string(),
            })
    })?;
    let total_cols: usize = matrices.iter().try_fold(0usize, |acc, m| {
        acc.checked_add(m.shape().cols)
            .ok_or_else(|| SparseError::IndexOverflow {
                message: "block_diag total cols overflow".to_string(),
            })
    })?;
    let shape = Shape2D::new(total_rows, total_cols);

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    let mut row_offset = 0;
    let mut col_offset = 0;

    for &mat in matrices {
        let s = mat.shape();
        let indptr = mat.indptr();
        let indices = mat.indices();
        let mat_data = mat.data();
        for i in 0..s.rows {
            for idx in indptr[i]..indptr[i + 1] {
                rows.push(row_offset + i);
                cols.push(col_offset + indices[idx]);
                data.push(mat_data[idx]);
            }
        }
        row_offset += s.rows;
        col_offset += s.cols;
    }

    let coo = CooMatrix::from_triplets(shape, data, rows, cols, false)?;
    coo.to_csr()
}

/// Construct a sparse matrix from a block layout of sparse matrices.
///
/// `blocks` is a row-major grid where `None` entries represent zero blocks.
/// All blocks in the same row must have the same number of rows,
/// and all blocks in the same column must have the same number of columns.
/// Matches `scipy.sparse.bmat(blocks)`.
pub fn bmat(blocks: &[Vec<Option<&CsrMatrix>>]) -> SparseResult<CsrMatrix> {
    if blocks.is_empty() {
        return eye(0);
    }

    let n_block_rows = blocks.len();
    let n_block_cols = blocks[0].len();
    for row in blocks {
        if row.len() != n_block_cols {
            return Err(SparseError::InvalidArgument {
                message: "all block rows must have the same number of columns".to_string(),
            });
        }
    }

    // Determine row heights and column widths
    let mut row_heights = vec![0usize; n_block_rows];
    let mut col_widths = vec![0usize; n_block_cols];

    for (i, row) in blocks.iter().enumerate() {
        for (j, block) in row.iter().enumerate() {
            if let Some(mat) = block {
                let s = mat.shape();
                if row_heights[i] == 0 {
                    row_heights[i] = s.rows;
                } else if row_heights[i] != s.rows {
                    return Err(SparseError::IncompatibleShape {
                        message: format!(
                            "block row {i}: height mismatch {} vs {}",
                            row_heights[i], s.rows
                        ),
                    });
                }
                if col_widths[j] == 0 {
                    col_widths[j] = s.cols;
                } else if col_widths[j] != s.cols {
                    return Err(SparseError::IncompatibleShape {
                        message: format!(
                            "block col {j}: width mismatch {} vs {}",
                            col_widths[j], s.cols
                        ),
                    });
                }
            }
        }
    }

    let total_rows: usize = row_heights.iter().try_fold(0usize, |acc, &height| {
        acc.checked_add(height)
            .ok_or_else(|| SparseError::IndexOverflow {
                message: "bmat total rows overflow".to_string(),
            })
    })?;
    let total_cols: usize = col_widths.iter().try_fold(0usize, |acc, &width| {
        acc.checked_add(width)
            .ok_or_else(|| SparseError::IndexOverflow {
                message: "bmat total cols overflow".to_string(),
            })
    })?;
    let shape = Shape2D::new(total_rows, total_cols);

    let mut all_rows = Vec::new();
    let mut all_cols = Vec::new();
    let mut all_data = Vec::new();

    let mut row_offset = 0;
    for (i, block_row) in blocks.iter().enumerate() {
        let mut col_offset = 0;
        for (j, block) in block_row.iter().enumerate() {
            if let Some(mat) = block {
                let indptr = mat.indptr();
                let indices = mat.indices();
                let mat_data = mat.data();
                for r in 0..mat.shape().rows {
                    for idx in indptr[r]..indptr[r + 1] {
                        all_rows.push(row_offset + r);
                        all_cols.push(col_offset + indices[idx]);
                        all_data.push(mat_data[idx]);
                    }
                }
            }
            col_offset += col_widths[j];
        }
        row_offset += row_heights[i];
    }

    let coo = CooMatrix::from_triplets(shape, all_data, all_rows, all_cols, false)?;
    coo.to_csr()
}

/// Stack sparse matrices vertically (row wise).
///
/// Mirrors `scipy.sparse.vstack` for sparse inputs. The Rust API returns CSR
/// directly; call `.to_coo()` when coordinate output is desired.
pub fn vstack(blocks: &[&dyn FormatConvertible]) -> SparseResult<CsrMatrix> {
    stack_sparse_blocks(blocks, StackAxis::Rows)
}

/// Stack sparse matrices horizontally (column wise).
///
/// Mirrors `scipy.sparse.hstack` for sparse inputs. The Rust API returns CSR
/// directly; call `.to_coo()` when coordinate output is desired.
pub fn hstack(blocks: &[&dyn FormatConvertible]) -> SparseResult<CsrMatrix> {
    stack_sparse_blocks(blocks, StackAxis::Cols)
}

#[derive(Debug, Clone, PartialEq)]
pub enum HstackOutput {
    Csr(CsrMatrix),
    Csc(CscMatrix),
    Coo(CooMatrix),
    Bsr(BsrMatrix),
    Dia(DiaMatrix),
    Dok(DokMatrix),
    Lil(LilMatrix),
}

impl HstackOutput {
    #[must_use]
    pub const fn format(&self) -> SparseFormat {
        match self {
            Self::Csr(_) => SparseFormat::Csr,
            Self::Csc(_) => SparseFormat::Csc,
            Self::Coo(_) => SparseFormat::Coo,
            Self::Bsr(_) => SparseFormat::Bsr,
            Self::Dia(_) => SparseFormat::Dia,
            Self::Dok(_) => SparseFormat::Dok,
            Self::Lil(_) => SparseFormat::Lil,
        }
    }

    #[must_use]
    pub const fn format_name(&self) -> &'static str {
        sparse_format_name(self.format())
    }

    pub fn to_coo(&self) -> SparseResult<CooMatrix> {
        match self {
            Self::Csr(matrix) => matrix.to_coo(),
            Self::Csc(matrix) => matrix.to_coo(),
            Self::Coo(matrix) => Ok(matrix.clone()),
            Self::Bsr(matrix) => matrix.to_coo(),
            Self::Dia(matrix) => matrix.to_coo(),
            Self::Dok(matrix) => matrix.to_coo(),
            Self::Lil(matrix) => matrix.to_coo(),
        }
    }
}

/// Stack sparse matrices horizontally and honor SciPy's `format=` kwarg.
///
/// When `format` is `None`, this matches SciPy's default container choice and
/// returns a COO matrix. Supported format strings are `"csr"`, `"csc"`,
/// `"coo"`, `"bsr"`, `"dia"`, `"dok"`, and `"lil"`.
pub fn hstack_with_format(
    blocks: &[&dyn FormatConvertible],
    format: Option<&str>,
) -> SparseResult<HstackOutput> {
    let requested_format = parse_hstack_format(format)?;
    let csr = stack_sparse_blocks(blocks, StackAxis::Cols)?;
    hstack_output_from_csr(csr, requested_format)
}

/// Kronecker product of two sparse matrices.
///
/// For A (m×n) and B (p×q), produces an (m*p × n*q) matrix.
/// Matches `scipy.sparse.kron(A, B)`.
pub fn kron(a: &CsrMatrix, b: &CsrMatrix) -> SparseResult<CsrMatrix> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let out_rows =
        a_shape
            .rows
            .checked_mul(b_shape.rows)
            .ok_or_else(|| SparseError::IndexOverflow {
                message: "kron output rows overflow".to_string(),
            })?;
    let out_cols =
        a_shape
            .cols
            .checked_mul(b_shape.cols)
            .ok_or_else(|| SparseError::IndexOverflow {
                message: "kron output cols overflow".to_string(),
            })?;
    let shape = Shape2D::new(out_rows, out_cols);

    let a_indptr = a.indptr();
    let a_indices = a.indices();
    let a_data = a.data();
    let b_indptr = b.indptr();
    let b_indices = b.indices();
    let b_data = b.data();

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for ai in 0..a_shape.rows {
        for a_idx in a_indptr[ai]..a_indptr[ai + 1] {
            let aj = a_indices[a_idx];
            let a_val = a_data[a_idx];
            for bi in 0..b_shape.rows {
                for b_idx in b_indptr[bi]..b_indptr[bi + 1] {
                    let bj = b_indices[b_idx];
                    let b_val = b_data[b_idx];
                    rows.push(ai * b_shape.rows + bi);
                    cols.push(aj * b_shape.cols + bj);
                    data.push(a_val * b_val);
                }
            }
        }
    }

    let coo = CooMatrix::from_triplets(shape, data, rows, cols, false)?;
    coo.to_csr()
}

/// Kronecker sum of two square sparse matrices.
///
/// For A (m×m) and B (n×n), produces `kron(I_n, A) + kron(B, I_m)`.
/// Matches `scipy.sparse.kronsum(A, B)`.
pub fn kronsum(a: &CsrMatrix, b: &CsrMatrix) -> SparseResult<CsrMatrix> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if !a_shape.is_square() {
        return Err(SparseError::IncompatibleShape {
            message: "kronsum first input must be square".to_string(),
        });
    }
    if !b_shape.is_square() {
        return Err(SparseError::IncompatibleShape {
            message: "kronsum second input must be square".to_string(),
        });
    }

    let left_identity = eye(b_shape.rows)?;
    let right_identity = eye(a_shape.rows)?;
    let left = kron(&left_identity, a)?;
    let right = kron(b, &right_identity)?;
    add_csr(&left, &right)
}

fn xorshift64(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StackAxis {
    Rows,
    Cols,
}

fn stack_sparse_blocks(
    blocks: &[&dyn FormatConvertible],
    axis: StackAxis,
) -> SparseResult<CsrMatrix> {
    if blocks.is_empty() {
        return Err(SparseError::InvalidArgument {
            message: format!(
                "{} requires at least one block",
                match axis {
                    StackAxis::Rows => "vstack",
                    StackAxis::Cols => "hstack",
                }
            ),
        });
    }

    let first = blocks[0].to_coo()?;
    let mut total_rows = first.shape().rows;
    let mut total_cols = first.shape().cols;
    let mut row_offset = 0usize;
    let mut col_offset = 0usize;
    let shared_extent = match axis {
        StackAxis::Rows => first.shape().cols,
        StackAxis::Cols => first.shape().rows,
    };

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    append_coo_entries(
        &first, row_offset, col_offset, &mut rows, &mut cols, &mut data,
    );

    match axis {
        StackAxis::Rows => row_offset += first.shape().rows,
        StackAxis::Cols => col_offset += first.shape().cols,
    }

    for block in blocks.iter().skip(1) {
        let coo = block.to_coo()?;
        match axis {
            StackAxis::Rows => {
                if coo.shape().cols != shared_extent {
                    return Err(SparseError::IncompatibleShape {
                        message: "vstack requires all blocks to have matching column counts"
                            .to_string(),
                    });
                }
                total_rows += coo.shape().rows;
            }
            StackAxis::Cols => {
                if coo.shape().rows != shared_extent {
                    return Err(SparseError::IncompatibleShape {
                        message: "hstack requires all blocks to have matching row counts"
                            .to_string(),
                    });
                }
                total_cols += coo.shape().cols;
            }
        }

        append_coo_entries(
            &coo, row_offset, col_offset, &mut rows, &mut cols, &mut data,
        );

        match axis {
            StackAxis::Rows => row_offset += coo.shape().rows,
            StackAxis::Cols => col_offset += coo.shape().cols,
        }
    }

    let shape = Shape2D::new(total_rows, total_cols);
    let coo = CooMatrix::from_triplets(shape, data, rows, cols, false)?;
    coo.to_csr()
}

fn parse_hstack_format(format: Option<&str>) -> SparseResult<SparseFormat> {
    match format.unwrap_or("coo") {
        "csr" => Ok(SparseFormat::Csr),
        "csc" => Ok(SparseFormat::Csc),
        "coo" => Ok(SparseFormat::Coo),
        "bsr" => Ok(SparseFormat::Bsr),
        "dia" => Ok(SparseFormat::Dia),
        "dok" => Ok(SparseFormat::Dok),
        "lil" => Ok(SparseFormat::Lil),
        _ => Err(SparseError::InvalidArgument {
            message: "format must be one of {'csr', 'csc', 'coo', 'bsr', 'dia', 'dok', 'lil'}"
                .to_string(),
        }),
    }
}

fn hstack_output_from_csr(csr: CsrMatrix, format: SparseFormat) -> SparseResult<HstackOutput> {
    let shape = csr.shape();
    let coo = match format {
        SparseFormat::Csr => return Ok(HstackOutput::Csr(csr)),
        SparseFormat::Csc => return Ok(HstackOutput::Csc(csr.to_csc()?)),
        SparseFormat::Coo => return Ok(HstackOutput::Coo(csr.to_coo()?)),
        _ => csr.to_coo()?,
    };

    let data = coo.data().to_vec();
    let rows = coo.row_indices().to_vec();
    let cols = coo.col_indices().to_vec();

    match format {
        SparseFormat::Bsr => Ok(HstackOutput::Bsr(BsrMatrix::from_triplets(
            shape,
            Shape2D::new(1, 1),
            data,
            rows,
            cols,
        )?)),
        SparseFormat::Dia => Ok(HstackOutput::Dia(DiaMatrix::from_triplets(
            shape, data, rows, cols,
        )?)),
        SparseFormat::Dok => Ok(HstackOutput::Dok(DokMatrix::from_triplets(
            shape, data, rows, cols,
        )?)),
        SparseFormat::Lil => Ok(HstackOutput::Lil(LilMatrix::from_triplets(
            shape, data, rows, cols,
        )?)),
        SparseFormat::Csr | SparseFormat::Csc | SparseFormat::Coo => {
            unreachable!("compressed and COO formats returned early")
        }
    }
}

const fn sparse_format_name(format: SparseFormat) -> &'static str {
    match format {
        SparseFormat::Csr => "csr",
        SparseFormat::Csc => "csc",
        SparseFormat::Coo => "coo",
        SparseFormat::Bsr => "bsr",
        SparseFormat::Dia => "dia",
        SparseFormat::Dok => "dok",
        SparseFormat::Lil => "lil",
    }
}

fn append_coo_entries(
    coo: &CooMatrix,
    row_offset: usize,
    col_offset: usize,
    rows: &mut Vec<usize>,
    cols: &mut Vec<usize>,
    data: &mut Vec<f64>,
) {
    for idx in 0..coo.nnz() {
        rows.push(row_offset + coo.row_indices()[idx]);
        cols.push(col_offset + coo.col_indices()[idx]);
        data.push(coo.data()[idx]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::FormatConvertible;

    #[test]
    fn eye_constructs_expected_identity() {
        let id = eye(4).expect("identity");
        assert_eq!(id.shape(), Shape2D::new(4, 4));
        assert_eq!(id.nnz(), 4);

        let coo = id.to_coo().expect("csr->coo");
        for idx in 0..coo.nnz() {
            assert_eq!(coo.row_indices()[idx], coo.col_indices()[idx]);
            assert!((coo.data()[idx] - 1.0).abs() <= f64::EPSILON);
        }
    }

    #[test]
    fn identity_matches_eye() {
        let id = identity(5).expect("identity");
        let eye = eye(5).expect("eye");
        assert_eq!(id.shape(), eye.shape());
        assert_eq!(id.data(), eye.data());
        assert_eq!(id.indices(), eye.indices());
        assert_eq!(id.indptr(), eye.indptr());
    }

    #[test]
    fn eye_zero_size_is_empty() {
        let id = eye(0).expect("identity");
        assert_eq!(id.shape(), Shape2D::new(0, 0));
        assert_eq!(id.nnz(), 0);
    }

    #[test]
    fn eye_rectangular_square_k0_matches_eye() {
        // eye_rectangular(n, n, 0) ≡ eye(n).
        for &n in &[1usize, 3, 7] {
            let lhs = eye_rectangular(n, n, 0).expect("eye_rect");
            let rhs = eye(n).expect("eye");
            assert_eq!(lhs.shape(), rhs.shape());
            assert_eq!(lhs.nnz(), rhs.nnz());
            let lhs_coo = lhs.to_coo().expect("coo");
            let rhs_coo = rhs.to_coo().expect("coo");
            for i in 0..lhs_coo.nnz() {
                assert_eq!(lhs_coo.row_indices()[i], rhs_coo.row_indices()[i]);
                assert_eq!(lhs_coo.col_indices()[i], rhs_coo.col_indices()[i]);
            }
        }
    }

    #[test]
    fn eye_rectangular_super_diagonal() {
        // eye_rectangular(3, 3, 1): super-diagonal — entries at
        // (0,1), (1,2). nnz = 2.
        let m = eye_rectangular(3, 3, 1).expect("eye_rect");
        assert_eq!(m.shape(), Shape2D::new(3, 3));
        assert_eq!(m.nnz(), 2);
        let coo = m.to_coo().expect("coo");
        let mut pairs: Vec<(usize, usize)> = (0..coo.nnz())
            .map(|i| (coo.row_indices()[i], coo.col_indices()[i]))
            .collect();
        pairs.sort();
        assert_eq!(pairs, vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn eye_rectangular_wide_negative_k() {
        // eye_rectangular(3, 4, -1): 3×4 sub-diagonal — entries at
        // (1,0), (2,1). nnz = 2.
        let m = eye_rectangular(3, 4, -1).expect("eye_rect");
        assert_eq!(m.shape(), Shape2D::new(3, 4));
        assert_eq!(m.nnz(), 2);
        let coo = m.to_coo().expect("coo");
        let mut pairs: Vec<(usize, usize)> = (0..coo.nnz())
            .map(|i| (coo.row_indices()[i], coo.col_indices()[i]))
            .collect();
        pairs.sort();
        assert_eq!(pairs, vec![(1, 0), (2, 1)]);
    }

    #[test]
    fn eye_rectangular_out_of_range_k_is_empty() {
        // |k| >= max(rows, cols) yields an all-zero matrix.
        let m = eye_rectangular(3, 3, 5).expect("k > cols");
        assert_eq!(m.shape(), Shape2D::new(3, 3));
        assert_eq!(m.nnz(), 0);
        let m = eye_rectangular(3, 3, -5).expect("|k| > rows");
        assert_eq!(m.nnz(), 0);
    }

    #[test]
    fn diags_rejects_length_mismatch() {
        let err = diags(&[vec![1.0]], &[0, 1], None).expect_err("length mismatch");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn diags_rejects_repeated_offsets() {
        let err = diags(&[vec![1.0], vec![2.0]], &[0, 0], None).expect_err("repeated offsets");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn diags_rejects_empty_shape_inference() {
        let err = diags(&[], &[], None).expect_err("empty inference");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn diags_infers_shape_with_positive_and_negative_offsets() {
        let csr = diags(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0]], &[1, -2], None).expect("diags");
        assert_eq!(csr.shape(), Shape2D::new(4, 4));

        let dense = dense_from_csr(&csr);
        let expected = vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 2.0, 0.0],
            vec![4.0, 0.0, 0.0, 3.0],
            vec![0.0, 5.0, 0.0, 0.0],
        ];
        assert_eq!(dense, expected);
    }

    #[test]
    fn diags_honors_explicit_shape() {
        let csr = diags(
            &[vec![1.0, 2.0], vec![3.0]],
            &[0, 2],
            Some(Shape2D::new(5, 5)),
        )
        .expect("diags");
        assert_eq!(csr.shape(), Shape2D::new(5, 5));
        assert_eq!(csr.nnz(), 3);
    }

    #[test]
    fn diags_rejects_out_of_bounds_for_explicit_shape() {
        let err = diags(&[vec![1.0, 2.0, 3.0]], &[1], Some(Shape2D::new(3, 3)))
            .expect_err("bounds violation");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn spdiags_uses_column_indexed_padding_for_positive_offsets() {
        let dia = spdiags(&[vec![1.0, 2.0, 3.0, 4.0]], &[1], 4, 4).expect("spdiags");
        assert_eq!(dia.shape(), Shape2D::new(4, 4));
        assert_eq!(dia.offsets(), &[1]);
        assert_eq!(
            dense_from_coo(&dia.to_coo().expect("dia->coo")),
            vec![
                vec![0.0, 2.0, 0.0, 0.0],
                vec![0.0, 0.0, 3.0, 0.0],
                vec![0.0, 0.0, 0.0, 4.0],
                vec![0.0, 0.0, 0.0, 0.0],
            ]
        );
    }

    #[test]
    fn spdiags_uses_column_indexed_values_for_negative_offsets() {
        let dia = spdiags(&[vec![1.0, 2.0, 3.0, 4.0]], &[-1], 4, 4).expect("spdiags");
        assert_eq!(
            dense_from_coo(&dia.to_coo().expect("dia->coo")),
            vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 2.0, 0.0, 0.0],
                vec![0.0, 0.0, 3.0, 0.0],
            ]
        );
    }

    #[test]
    fn spdiags_rejects_repeated_offsets() {
        let err = spdiags(&[vec![1.0], vec![2.0]], &[0, 0], 2, 2).expect_err("repeated offsets");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn random_rejects_density_out_of_range() {
        let shape = Shape2D::new(2, 2);
        assert!(matches!(
            random(shape, -0.1, 7),
            Err(SparseError::InvalidArgument { .. })
        ));
        assert!(matches!(
            random(shape, 1.1, 7),
            Err(SparseError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn random_rejects_shape_overflow() {
        let err = random(Shape2D::new(usize::MAX, 2), 0.5, 9).expect_err("overflow");
        assert!(matches!(err, SparseError::IndexOverflow { .. }));
    }

    #[test]
    fn random_density_zero_returns_empty_matrix() {
        let coo = random(Shape2D::new(4, 5), 0.0, 11).expect("random");
        assert_eq!(coo.shape(), Shape2D::new(4, 5));
        assert_eq!(coo.nnz(), 0);
    }

    #[test]
    fn random_density_one_fills_every_position() {
        let coo = random(Shape2D::new(3, 2), 1.0, 11).expect("random");
        assert_eq!(coo.nnz(), 6);
        for idx in 0..coo.nnz() {
            assert!(coo.row_indices()[idx] < 3);
            assert!(coo.col_indices()[idx] < 2);
            assert!((-1.0..=1.0).contains(&coo.data()[idx]));
        }
    }

    #[test]
    fn random_is_deterministic_for_same_seed() {
        let first = random(Shape2D::new(5, 4), 0.35, 12345).expect("random first");
        let second = random(Shape2D::new(5, 4), 0.35, 12345).expect("random second");
        assert_eq!(first.row_indices(), second.row_indices());
        assert_eq!(first.col_indices(), second.col_indices());
        assert_eq!(first.data(), second.data());
    }

    #[test]
    fn random_large_sparse_samples_expected_nnz() {
        let coo = random(Shape2D::new(1_000_000, 1_000_000), 1e-9, 42).expect("random");
        assert_eq!(coo.shape(), Shape2D::new(1_000_000, 1_000_000));
        assert_eq!(coo.nnz(), 1_000);

        let mut coordinates = HashSet::with_capacity(coo.nnz());
        for idx in 0..coo.nnz() {
            let row = coo.row_indices()[idx];
            let col = coo.col_indices()[idx];
            assert!(row < 1_000_000);
            assert!(col < 1_000_000);
            assert!(coordinates.insert((row, col)));
        }
    }

    #[test]
    fn random_zero_dimension_returns_empty() {
        let coo = random(Shape2D::new(0, 7), 1.0, 99).expect("random");
        assert_eq!(coo.nnz(), 0);
    }

    #[test]
    fn xorshift_changes_nonzero_values() {
        assert_eq!(xorshift64(0), 0);
        assert_ne!(xorshift64(1), 1);
    }

    // ── block_diag tests ─────────────────────────────────────────────

    #[test]
    fn block_diag_two_matrices() {
        let a = eye(2).expect("eye(2)");
        let b = diags(&[vec![3.0, 4.0]], &[0], None).expect("diags");
        let result = block_diag(&[&a, &b]).expect("block_diag");
        assert_eq!(result.shape(), Shape2D::new(4, 4));
        let dense = dense_from_csr(&result);
        assert_eq!(
            dense,
            vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 3.0, 0.0],
                vec![0.0, 0.0, 0.0, 4.0],
            ]
        );
    }

    #[test]
    fn block_diag_empty() {
        let result = block_diag(&[]).expect("block_diag empty");
        assert_eq!(result.shape(), Shape2D::new(0, 0));
    }

    #[test]
    fn block_diag_single() {
        let a = eye(3).expect("eye(3)");
        let result = block_diag(&[&a]).expect("block_diag single");
        assert_eq!(result.shape(), Shape2D::new(3, 3));
        assert_eq!(result.nnz(), 3);
    }

    #[test]
    fn block_diag_rejects_dimension_overflow() {
        let huge = CsrMatrix::from_components_unchecked(
            Shape2D::new(usize::MAX / 2 + 1, 1),
            Vec::new(),
            Vec::new(),
            vec![0],
        );

        let err = block_diag(&[&huge, &huge]).expect_err("overflow");
        assert!(matches!(err, SparseError::IndexOverflow { .. }));
    }

    // ── bmat tests ──────────────────────────────────────────────────

    #[test]
    fn bmat_2x2_grid() {
        let a = eye(2).expect("eye(2)");
        let b = diags(&[vec![5.0, 6.0]], &[0], None).expect("diags");
        let result = bmat(&[vec![Some(&a), None], vec![None, Some(&b)]]).expect("bmat");
        assert_eq!(result.shape(), Shape2D::new(4, 4));
        let dense = dense_from_csr(&result);
        assert_eq!(
            dense,
            vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 5.0, 0.0],
                vec![0.0, 0.0, 0.0, 6.0],
            ]
        );
    }

    #[test]
    fn bmat_with_off_diagonal_blocks() {
        let a = eye(2).expect("eye(2)");
        let result = bmat(&[vec![Some(&a), Some(&a)], vec![Some(&a), Some(&a)]]).expect("bmat");
        assert_eq!(result.shape(), Shape2D::new(4, 4));
        let dense = dense_from_csr(&result);
        assert_eq!(
            dense,
            vec![
                vec![1.0, 0.0, 1.0, 0.0],
                vec![0.0, 1.0, 0.0, 1.0],
                vec![1.0, 0.0, 1.0, 0.0],
                vec![0.0, 1.0, 0.0, 1.0],
            ]
        );
    }

    #[test]
    fn bmat_empty() {
        let result = bmat(&[]).expect("bmat empty");
        assert_eq!(result.shape(), Shape2D::new(0, 0));
    }

    #[test]
    fn bmat_ragged_block_rows_rejected() {
        let a = eye(2).expect("eye(2)");
        let err = bmat(&[vec![Some(&a), Some(&a)], vec![Some(&a)]]).expect_err("ragged");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn bmat_rejects_dimension_overflow() {
        let huge = CsrMatrix::from_components_unchecked(
            Shape2D::new(usize::MAX / 2 + 1, 1),
            Vec::new(),
            Vec::new(),
            vec![0],
        );

        let err = bmat(&[vec![Some(&huge)], vec![Some(&huge)]]).expect_err("overflow");
        assert!(matches!(err, SparseError::IndexOverflow { .. }));
    }

    #[test]
    fn vstack_accepts_mixed_sparse_formats() {
        let top = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo");
        let bottom = CooMatrix::from_triplets(
            Shape2D::new(1, 2),
            vec![5.0, 6.0],
            vec![0, 0],
            vec![0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");

        let result = vstack(&[&top, &bottom]).expect("vstack");
        assert_eq!(result.shape(), Shape2D::new(3, 2));
        assert_eq!(
            dense_from_csr(&result),
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]
        );
    }

    #[test]
    fn hstack_accepts_mixed_sparse_formats() {
        let left = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo");
        let right = CooMatrix::from_triplets(
            Shape2D::new(2, 1),
            vec![5.0, 6.0],
            vec![0, 1],
            vec![0, 0],
            false,
        )
        .expect("coo")
        .to_csc()
        .expect("csc");

        let result = hstack(&[&left, &right]).expect("hstack");
        assert_eq!(result.shape(), Shape2D::new(2, 3));
        assert_eq!(
            dense_from_csr(&result),
            vec![vec![1.0, 2.0, 5.0], vec![3.0, 4.0, 6.0]]
        );
    }

    #[test]
    fn hstack_with_format_defaults_to_coo_output() {
        let left = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo");
        let right = CooMatrix::from_triplets(
            Shape2D::new(2, 1),
            vec![5.0, 6.0],
            vec![0, 1],
            vec![0, 0],
            false,
        )
        .expect("coo");

        let result = hstack_with_format(&[&left, &right], None).expect("default hstack format");
        assert!(matches!(result, HstackOutput::Coo(_)));
        assert_eq!(result.format(), SparseFormat::Coo);
        assert_eq!(
            dense_from_coo(&result.to_coo().expect("output->coo")),
            vec![vec![1.0, 2.0, 5.0], vec![3.0, 4.0, 6.0]]
        );
    }

    #[test]
    fn hstack_with_format_supports_all_sparse_output_kinds() {
        let left = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo");
        let right = CooMatrix::from_triplets(
            Shape2D::new(2, 1),
            vec![5.0, 6.0],
            vec![0, 1],
            vec![0, 0],
            false,
        )
        .expect("coo")
        .to_csc()
        .expect("csc");

        for (format, expected) in [
            ("csr", SparseFormat::Csr),
            ("csc", SparseFormat::Csc),
            ("coo", SparseFormat::Coo),
            ("bsr", SparseFormat::Bsr),
            ("dia", SparseFormat::Dia),
            ("dok", SparseFormat::Dok),
            ("lil", SparseFormat::Lil),
        ] {
            let result = hstack_with_format(&[&left, &right], Some(format));
            assert!(
                result.is_ok(),
                "hstack format {format} failed: {:?}",
                result.as_ref().err()
            );
            let Ok(result) = result else {
                continue;
            };
            assert_eq!(result.format(), expected, "format mismatch for {format}");
            assert_eq!(
                dense_from_coo(&result.to_coo().expect("output->coo")),
                vec![vec![1.0, 2.0, 5.0], vec![3.0, 4.0, 6.0]],
                "dense mismatch for {format}"
            );
            if let HstackOutput::Bsr(matrix) = &result {
                assert_eq!(matrix.block_shape(), Shape2D::new(1, 1));
            }
        }
    }

    #[test]
    fn hstack_with_format_rejects_invalid_format() {
        let left = eye(2).expect("eye");
        let right = eye(2).expect("eye");
        let err = hstack_with_format(&[&left, &right], Some("bad")).expect_err("invalid format");
        assert!(
            matches!(err, SparseError::InvalidArgument { message } if message.contains("format must be one of"))
        );
    }

    #[test]
    fn vstack_rejects_empty_input() {
        let err = vstack(&[]).expect_err("empty vstack");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn vstack_rejects_mismatched_column_counts() {
        let left = eye(2).expect("eye");
        let right = eye(3).expect("eye");
        let err = vstack(&[&left, &right]).expect_err("mismatched cols");
        assert!(matches!(err, SparseError::IncompatibleShape { .. }));
    }

    #[test]
    fn hstack_rejects_mismatched_row_counts() {
        let left = eye(2).expect("eye");
        let right = eye(3).expect("eye");
        let err = hstack(&[&left, &right]).expect_err("mismatched rows");
        assert!(matches!(err, SparseError::IncompatibleShape { .. }));
    }

    // ── kron tests ──────────────────────────────────────────────────

    #[test]
    fn kron_identity_times_identity() {
        let i2 = eye(2).expect("eye(2)");
        let result = kron(&i2, &i2).expect("kron");
        assert_eq!(result.shape(), Shape2D::new(4, 4));
        // I2 ⊗ I2 = I4
        let dense = dense_from_csr(&result);
        for (i, row) in dense.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((val - expected).abs() < 1e-14, "kron I2⊗I2 at [{i}][{j}]");
            }
        }
    }

    #[test]
    fn kron_scalar_times_matrix() {
        // scalar [[2]] ⊗ [[1, 0], [0, 3]] = [[2, 0], [0, 6]]
        let scalar =
            CooMatrix::from_triplets(Shape2D::new(1, 1), vec![2.0], vec![0], vec![0], false)
                .expect("coo")
                .to_csr()
                .expect("csr");
        let b = diags(&[vec![1.0, 3.0]], &[0], None).expect("diags");
        let result = kron(&scalar, &b).expect("kron");
        assert_eq!(result.shape(), Shape2D::new(2, 2));
        let dense = dense_from_csr(&result);
        assert_eq!(dense, vec![vec![2.0, 0.0], vec![0.0, 6.0]]);
    }

    #[test]
    fn kron_known_result() {
        // [[1, 2], [3, 4]] ⊗ [[0, 5], [6, 7]]
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
        let b = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![5.0, 6.0, 7.0],
            vec![0, 1, 1],
            vec![1, 0, 1],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let result = kron(&a, &b).expect("kron");
        assert_eq!(result.shape(), Shape2D::new(4, 4));
        let dense = dense_from_csr(&result);
        // Expected: [[0, 5, 0, 10], [6, 7, 12, 14], [0, 15, 0, 20], [18, 21, 24, 28]]
        let expected = [
            [0.0, 5.0, 0.0, 10.0],
            [6.0, 7.0, 12.0, 14.0],
            [0.0, 15.0, 0.0, 20.0],
            [18.0, 21.0, 24.0, 28.0],
        ];
        for (i, (d_row, e_row)) in dense.iter().zip(expected.iter()).enumerate() {
            for (j, (&d_val, &e_val)) in d_row.iter().zip(e_row.iter()).enumerate() {
                assert!(
                    (d_val - e_val).abs() < 1e-12,
                    "kron at [{i}][{j}]: {d_val} vs {e_val}"
                );
            }
        }
    }

    #[test]
    fn kronsum_scalar_inputs_sum_values() {
        // kronsum([[a]], [[b]]) = [[a + b]] since kron(I_1,A) = A and
        // kron(B,I_1) = B for 1×1 inputs.
        let a = CooMatrix::from_triplets(Shape2D::new(1, 1), vec![3.0], vec![0], vec![0], false)
            .expect("coo")
            .to_csr()
            .expect("csr");
        let b = CooMatrix::from_triplets(Shape2D::new(1, 1), vec![4.0], vec![0], vec![0], false)
            .expect("coo")
            .to_csr()
            .expect("csr");
        let result = kronsum(&a, &b).expect("kronsum");
        assert_eq!(result.shape(), Shape2D::new(1, 1));
        let dense = dense_from_csr(&result);
        assert!((dense[0][0] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn kronsum_two_by_two_matches_scipy_reference() {
        // /testing-conformance-harnesses: scipy.sparse.kronsum
        //
        //   A = [[1, 2], [3, 4]]      (2×2)
        //   B = [[10, 0], [0, 20]]    (2×2)
        //
        //   kron(I_2, A) = [[1, 2, 0, 0],
        //                   [3, 4, 0, 0],
        //                   [0, 0, 1, 2],
        //                   [0, 0, 3, 4]]
        //   kron(B, I_2) = [[10,  0,  0,  0],
        //                   [ 0, 10,  0,  0],
        //                   [ 0,  0, 20,  0],
        //                   [ 0,  0,  0, 20]]
        //   sum         = [[11,  2,  0,  0],
        //                   [ 3, 14,  0,  0],
        //                   [ 0,  0, 21,  2],
        //                   [ 0,  0,  3, 24]]
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
        let b = diags(&[vec![10.0, 20.0]], &[0], None).expect("diags");
        let result = kronsum(&a, &b).expect("kronsum");
        assert_eq!(result.shape(), Shape2D::new(4, 4));
        let dense = dense_from_csr(&result);
        let expected = [
            [11.0, 2.0, 0.0, 0.0],
            [3.0, 14.0, 0.0, 0.0],
            [0.0, 0.0, 21.0, 2.0],
            [0.0, 0.0, 3.0, 24.0],
        ];
        for (i, (d_row, e_row)) in dense.iter().zip(expected.iter()).enumerate() {
            for (j, (&d_val, &e_val)) in d_row.iter().zip(e_row.iter()).enumerate() {
                assert!(
                    (d_val - e_val).abs() < 1e-12,
                    "kronsum at [{i}][{j}]: {d_val} vs {e_val}"
                );
            }
        }
    }

    #[test]
    fn kronsum_identity_doubles_to_2_eye() {
        // kronsum(I_m, I_n) = kron(I_n, I_m) + kron(I_n, I_m) = 2·I_{mn}
        let i2 = eye(2).expect("eye(2)");
        let i3 = eye(3).expect("eye(3)");
        let result = kronsum(&i2, &i3).expect("kronsum");
        assert_eq!(result.shape(), Shape2D::new(6, 6));
        let dense = dense_from_csr(&result);
        for (i, row) in dense.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                let expected = if i == j { 2.0 } else { 0.0 };
                assert!(
                    (val - expected).abs() < 1e-12,
                    "kronsum I_2⊕I_3 at [{i}][{j}]: got {val}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn kronsum_rejects_non_square() {
        let rect = CooMatrix::from_triplets(Shape2D::new(2, 3), vec![1.0], vec![0], vec![0], false)
            .expect("coo")
            .to_csr()
            .expect("csr");
        let sq = eye(2).expect("eye");
        assert!(kronsum(&rect, &sq).is_err());
        assert!(kronsum(&sq, &rect).is_err());
    }

    fn dense_from_csr(csr: &CsrMatrix) -> Vec<Vec<f64>> {
        let shape = csr.shape();
        let mut dense = vec![vec![0.0; shape.cols]; shape.rows];
        for (row, row_dense) in dense.iter_mut().enumerate().take(shape.rows) {
            for idx in csr.indptr()[row]..csr.indptr()[row + 1] {
                row_dense[csr.indices()[idx]] += csr.data()[idx];
            }
        }
        dense
    }

    fn dense_from_coo(coo: &CooMatrix) -> Vec<Vec<f64>> {
        let shape = coo.shape();
        let mut dense = vec![vec![0.0; shape.cols]; shape.rows];
        for idx in 0..coo.nnz() {
            dense[coo.row_indices()[idx]][coo.col_indices()[idx]] += coo.data()[idx];
        }
        dense
    }

    #[test]
    fn eye_matches_scipy_reference_values() {
        // scipy.sparse.eye(3).toarray()
        let result = eye(3).expect("eye");
        let dense = dense_from_csr(&result);
        let expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for (i, row) in dense.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                assert!(
                    (*val - expected[i][j]).abs() < 1e-10,
                    "eye[{i}][{j}] got {val}, expected {}",
                    expected[i][j]
                );
            }
        }
    }

    #[test]
    fn diags_matches_scipy_reference_values() {
        // scipy.sparse.diags([1,2,3], 0, shape=(3,3)).toarray()
        let result = diags(&[vec![1.0, 2.0, 3.0]], &[0], Some(Shape2D::new(3, 3))).expect("diags");
        let dense = dense_from_csr(&result);
        let expected = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
        for (i, row) in dense.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                assert!(
                    (*val - expected[i][j]).abs() < 1e-10,
                    "diags[{i}][{j}] got {val}, expected {}",
                    expected[i][j]
                );
            }
        }
    }

    #[test]
    fn kron_matches_scipy_reference_values() {
        // scipy.sparse.kron([[1,2],[3,4]], [[1,0],[0,1]])
        let a = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo_a")
        .to_csr()
        .expect("csr_a");
        let b = eye(2).expect("eye_b");
        let result = kron(&a, &b).expect("kron");
        let dense = dense_from_csr(&result);
        let expected = [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 2.0],
            [3.0, 0.0, 4.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
        ];
        for (i, row) in dense.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                assert!(
                    (*val - expected[i][j]).abs() < 1e-10,
                    "kron[{i}][{j}] got {val}, expected {}",
                    expected[i][j]
                );
            }
        }
    }

    #[test]
    fn block_diag_matches_scipy_reference_values() {
        // scipy.sparse.block_diag([[[1]], [[2,3],[4,5]]]).toarray()
        // -> [[1, 0, 0], [0, 2, 3], [0, 4, 5]]
        let a = CooMatrix::from_triplets(Shape2D::new(1, 1), vec![1.0], vec![0], vec![0], false)
            .expect("coo_a")
            .to_csr()
            .expect("csr_a");
        let b = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![2.0, 3.0, 4.0, 5.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            false,
        )
        .expect("coo_b")
        .to_csr()
        .expect("csr_b");
        let result = block_diag(&[&a, &b]).expect("block_diag");
        let dense = dense_from_csr(&result);
        let expected = [[1.0, 0.0, 0.0], [0.0, 2.0, 3.0], [0.0, 4.0, 5.0]];
        for (i, row) in dense.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                assert!(
                    (*val - expected[i][j]).abs() < 1e-10,
                    "block_diag[{i}][{j}] got {val}, expected {}",
                    expected[i][j]
                );
            }
        }
    }

    #[test]
    fn random_matches_scipy_density() {
        // scipy.sparse.random(10, 10, density=0.3)
        // Just verify shape and approximate density
        let result = random(Shape2D::new(10, 10), 0.3, 42).expect("random");
        assert_eq!(result.shape().rows, 10);
        assert_eq!(result.shape().cols, 10);
        // Density should be approximately 0.3 (30 non-zeros in 100 elements)
        let actual_density = result.nnz() as f64 / 100.0;
        assert!(
            (actual_density - 0.3).abs() < 0.1,
            "density got {actual_density}, expected ~0.3"
        );
    }
}
