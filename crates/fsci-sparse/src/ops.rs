use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;

use crate::SparseArray;
use crate::formats::{
    BsrMatrix, CanonicalMeta, CooArray, CooMatrix, CscMatrix, CsrMatrix, DiaMatrix, DokMatrix,
    LilMatrix, Shape2D, SparseArray2D, SparseError, SparseFormat, SparseResult,
};

pub trait FormatConvertible {
    fn to_csr(&self) -> SparseResult<CsrMatrix>;
    fn to_csc(&self) -> SparseResult<CscMatrix>;
    fn to_coo(&self) -> SparseResult<CooMatrix>;
}

impl<M: FormatConvertible> FormatConvertible for SparseArray2D<M> {
    fn to_csr(&self) -> SparseResult<CsrMatrix> {
        self.as_matrix().to_csr()
    }

    fn to_csc(&self) -> SparseResult<CscMatrix> {
        self.as_matrix().to_csc()
    }

    fn to_coo(&self) -> SparseResult<CooMatrix> {
        self.as_matrix().to_coo()
    }
}

impl FormatConvertible for CooArray {
    fn to_csr(&self) -> SparseResult<CsrMatrix> {
        self.to_coo_matrix()?.to_csr()
    }

    fn to_csc(&self) -> SparseResult<CscMatrix> {
        self.to_coo_matrix()?.to_csc()
    }

    fn to_coo(&self) -> SparseResult<CooMatrix> {
        self.to_coo_matrix()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionLogEntry {
    pub timestamp: String,
    pub operation_id: String,
    pub from_format: SparseFormat,
    pub to_format: SparseFormat,
    pub nnz_before: usize,
    pub nnz_after: usize,
}

impl ConversionLogEntry {
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            "{{\"timestamp\":\"{}\",\"operation_id\":\"{}\",\"from_format\":\"{}\",\"to_format\":\"{}\",\"nnz_before\":{},\"nnz_after\":{}}}",
            self.timestamp,
            self.operation_id,
            format_label(self.from_format),
            format_label(self.to_format),
            self.nnz_before,
            self.nnz_after
        )
    }
}

impl FormatConvertible for CooMatrix {
    fn to_csr(&self) -> SparseResult<CsrMatrix> {
        if let Some(csr) = sorted_unique_coo_to_csr(self)? {
            return Ok(csr);
        }

        Ok(coo_to_canonical_csr_counting(self))
    }

    fn to_csc(&self) -> SparseResult<CscMatrix> {
        Ok(coo_to_canonical_csc_counting(self))
    }

    fn to_coo(&self) -> SparseResult<CooMatrix> {
        Ok(self.clone())
    }
}

impl FormatConvertible for CsrMatrix {
    fn to_csr(&self) -> SparseResult<CsrMatrix> {
        Ok(self.clone())
    }

    fn to_csc(&self) -> SparseResult<CscMatrix> {
        if can_direct_transpose_compressed(
            self.canonical_meta().sorted_indices,
            self.canonical_meta().deduplicated,
            self.shape().rows,
            self.shape().cols,
            self.nnz(),
            self.indptr(),
            self.indices(),
        ) {
            return csr_to_csc_direct(self);
        }
        self.to_coo()?.to_csc()
    }

    fn to_coo(&self) -> SparseResult<CooMatrix> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut data = Vec::with_capacity(self.nnz());
        for row in 0..self.shape().rows {
            for idx in self.indptr()[row]..self.indptr()[row + 1] {
                rows.push(row);
                cols.push(self.indices()[idx]);
                data.push(self.data()[idx]);
            }
        }
        // Row indices come from the `0..rows` loop counter and column indices
        // are copied verbatim from `self.indices()`, both already validated when
        // this CSR was built. Constructing the COO directly skips the two O(nnz)
        // bounds re-scans `from_triplets` would run over provably in-bounds data;
        // byte-identical to `from_triplets(.., false)`, which just returns the struct.
        Ok(CooMatrix {
            shape: self.shape(),
            data,
            row_indices: rows,
            col_indices: cols,
        })
    }
}

impl FormatConvertible for CscMatrix {
    fn to_csr(&self) -> SparseResult<CsrMatrix> {
        if can_direct_transpose_compressed(
            self.canonical_meta().sorted_indices,
            self.canonical_meta().deduplicated,
            self.shape().cols,
            self.shape().rows,
            self.nnz(),
            self.indptr(),
            self.indices(),
        ) {
            return csc_to_csr_direct(self);
        }
        self.to_coo()?.to_csr()
    }

    fn to_csc(&self) -> SparseResult<CscMatrix> {
        Ok(self.clone())
    }

    fn to_coo(&self) -> SparseResult<CooMatrix> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut data = Vec::with_capacity(self.nnz());
        for col in 0..self.shape().cols {
            for idx in self.indptr()[col]..self.indptr()[col + 1] {
                rows.push(self.indices()[idx]);
                cols.push(col);
                data.push(self.data()[idx]);
            }
        }
        // Column indices come from the `0..cols` loop counter and row indices are
        // copied verbatim from `self.indices()`, both already validated when this
        // CSC was built. Constructing the COO directly skips the two O(nnz) bounds
        // re-scans `from_triplets` would run over provably in-bounds data;
        // byte-identical to `from_triplets(.., false)`, which just returns the struct.
        Ok(CooMatrix {
            shape: self.shape(),
            data,
            row_indices: rows,
            col_indices: cols,
        })
    }
}

impl FormatConvertible for DiaMatrix {
    fn to_csr(&self) -> SparseResult<CsrMatrix> {
        self.to_coo()?.to_csr()
    }

    fn to_csc(&self) -> SparseResult<CscMatrix> {
        self.to_coo()?.to_csc()
    }

    fn to_coo(&self) -> SparseResult<CooMatrix> {
        // Delegate to the inherent method so the trait-object path (vstack/
        // hstack) matches scipy's dia_matrix.tocoo(), which filters explicit
        // zeros. The previous standalone copy materialized stored zeros.
        DiaMatrix::to_coo(self)
    }
}

impl FormatConvertible for DokMatrix {
    fn to_csr(&self) -> SparseResult<CsrMatrix> {
        DokMatrix::to_coo(self)?.to_csr()
    }

    fn to_csc(&self) -> SparseResult<CscMatrix> {
        DokMatrix::to_coo(self)?.to_csc()
    }

    fn to_coo(&self) -> SparseResult<CooMatrix> {
        DokMatrix::to_coo(self)
    }
}

impl FormatConvertible for LilMatrix {
    fn to_csr(&self) -> SparseResult<CsrMatrix> {
        LilMatrix::to_coo(self)?.to_csr()
    }

    fn to_csc(&self) -> SparseResult<CscMatrix> {
        LilMatrix::to_coo(self)?.to_csc()
    }

    fn to_coo(&self) -> SparseResult<CooMatrix> {
        LilMatrix::to_coo(self)
    }
}

impl FormatConvertible for BsrMatrix {
    fn to_csr(&self) -> SparseResult<CsrMatrix> {
        BsrMatrix::to_coo(self)?.to_csr()
    }

    fn to_csc(&self) -> SparseResult<CscMatrix> {
        BsrMatrix::to_coo(self)?.to_csc()
    }

    fn to_coo(&self) -> SparseResult<CooMatrix> {
        BsrMatrix::to_coo(self)
    }
}

pub fn coo_to_csr_with_mode(
    coo: &CooMatrix,
    mode: RuntimeMode,
    operation_id: impl Into<String>,
) -> SparseResult<(CsrMatrix, ConversionLogEntry)> {
    let mut csr = coo.to_csr()?;
    if mode == RuntimeMode::Hardened {
        csr.canonical.sorted_indices = true;
        csr.canonical.deduplicated = true;
    }
    let log = conversion_log(
        operation_id.into(),
        SparseFormat::Coo,
        SparseFormat::Csr,
        coo.nnz(),
        csr.nnz(),
    );
    Ok((csr, log))
}

pub fn csr_to_csc_with_mode(
    csr: &CsrMatrix,
    mode: RuntimeMode,
    operation_id: impl Into<String>,
) -> SparseResult<(CscMatrix, ConversionLogEntry)> {
    csr_to_csc_with_mode_inner(csr, mode, operation_id, None)
}

/// Audit-emitting variant of [`csr_to_csc_with_mode`] (br-egba-4).
/// Records an AuditAction::FailClosed event when Hardened mode
/// rejects a non-canonical CSR. Strict mode never rejects, so no
/// emission occurs.
pub fn csr_to_csc_with_mode_and_audit(
    csr: &CsrMatrix,
    mode: RuntimeMode,
    operation_id: impl Into<String>,
    ledger: &crate::audit::SyncSharedAuditLedger,
) -> SparseResult<(CscMatrix, ConversionLogEntry)> {
    csr_to_csc_with_mode_inner(csr, mode, operation_id, Some(ledger))
}

fn csr_to_csc_with_mode_inner(
    csr: &CsrMatrix,
    mode: RuntimeMode,
    operation_id: impl Into<String>,
    ledger: Option<&crate::audit::SyncSharedAuditLedger>,
) -> SparseResult<(CscMatrix, ConversionLogEntry)> {
    if mode == RuntimeMode::Hardened && !csr.canonical.sorted_indices {
        if let Some(ledger) = ledger {
            crate::audit::record_fail_closed(
                ledger,
                format!("csr_shape={:?}", csr.shape()).as_bytes(),
                "csr_to_csc::unsorted_indices",
                "rejected",
            );
        }
        return Err(SparseError::InvalidSparseStructure {
            message: "hardened conversion requires sorted CSR indices".to_string(),
        });
    }
    let mut csc = csr.to_csc()?;
    // The conversion process via COO always sorts and deduplicates the triplets,
    // so the resulting CSC is always canonical, regardless of mode.
    csc.canonical.sorted_indices = true;
    csc.canonical.deduplicated = true;

    let log = conversion_log(
        operation_id.into(),
        SparseFormat::Csr,
        SparseFormat::Csc,
        csr.nnz(),
        csc.nnz(),
    );
    Ok((csc, log))
}

pub fn csc_to_csr_with_mode(
    csc: &CscMatrix,
    mode: RuntimeMode,
    operation_id: impl Into<String>,
) -> SparseResult<(CsrMatrix, ConversionLogEntry)> {
    if mode == RuntimeMode::Hardened && !csc.canonical.sorted_indices {
        return Err(SparseError::InvalidSparseStructure {
            message: "hardened conversion requires sorted CSC indices".to_string(),
        });
    }
    let mut csr = csc.to_csr()?;
    // The conversion process via COO always sorts and deduplicates the triplets,
    // so the resulting CSR is always canonical, regardless of mode.
    csr.canonical.sorted_indices = true;
    csr.canonical.deduplicated = true;

    let log = conversion_log(
        operation_id.into(),
        SparseFormat::Csc,
        SparseFormat::Csr,
        csc.nnz(),
        csr.nnz(),
    );
    Ok((csr, log))
}

/// Return the row indices, column indices, and values of nonzero entries.
///
/// Matches `scipy.sparse.find` by canonicalizing duplicate coordinates and
/// dropping explicit zeros from the returned triplets.
pub fn find<T: FormatConvertible>(matrix: &T) -> SparseResult<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    let CooMatrix {
        shape,
        data,
        row_indices,
        col_indices,
    } = matrix.to_coo()?;
    let canonical = CooMatrix::from_triplets(shape, data, row_indices, col_indices, true)?;

    let mut rows = Vec::with_capacity(canonical.nnz());
    let mut cols = Vec::with_capacity(canonical.nnz());
    let mut data = Vec::with_capacity(canonical.nnz());
    for idx in 0..canonical.nnz() {
        let value = canonical.data()[idx];
        if value != 0.0 {
            rows.push(canonical.row_indices()[idx]);
            cols.push(canonical.col_indices()[idx]);
            data.push(value);
        }
    }

    Ok((rows, cols, data))
}

/// Return the lower-triangular portion of a sparse matrix in COO form.
///
/// Matches the default `scipy.sparse.tril` behavior. Call `.to_csr()` or
/// `.to_csc()` on the result when a different output format is desired.
pub fn tril<T: FormatConvertible>(matrix: &T, k: isize) -> SparseResult<CooMatrix> {
    triangular_filter(matrix, k, TriangleHalf::Lower)
}

/// Return the upper-triangular portion of a sparse matrix in COO form.
///
/// Matches the default `scipy.sparse.triu` behavior. Call `.to_csr()` or
/// `.to_csc()` on the result when a different output format is desired.
pub fn triu<T: FormatConvertible>(matrix: &T, k: isize) -> SparseResult<CooMatrix> {
    triangular_filter(matrix, k, TriangleHalf::Upper)
}

/// Lower-triangular portion of a sparse array, matching `scipy.sparse.tril_array`
/// (the array-API spelling of [`tril`]).
pub fn tril_array<T: SparseArray + ?Sized>(array: &T, k: isize) -> SparseResult<CooArray> {
    let coo = array.to_coo_array()?;
    let matrix = coo.to_coo_matrix()?;
    triangular_filter(&matrix, k, TriangleHalf::Lower)
        .map(|filtered| CooArray::from_coo_matrix(&filtered))
}

/// Upper-triangular portion of a sparse array, matching `scipy.sparse.triu_array`
/// (the array-API spelling of [`triu`]).
pub fn triu_array<T: SparseArray + ?Sized>(array: &T, k: isize) -> SparseResult<CooArray> {
    let coo = array.to_coo_array()?;
    let matrix = coo.to_coo_matrix()?;
    triangular_filter(&matrix, k, TriangleHalf::Upper)
        .map(|filtered| CooArray::from_coo_matrix(&filtered))
}

/// Runtime switch to force the serial `spmv_csr` loop for same-binary A/B
/// benchmarks. Defaults off. `#[doc(hidden)]` — internal.
#[doc(hidden)]
pub static SPMV_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn spmv_csr(matrix: &CsrMatrix, vector: &[f64]) -> SparseResult<Vec<f64>> {
    if vector.len() != matrix.shape().cols {
        return Err(SparseError::IncompatibleShape {
            message: "spmv vector length must match matrix columns".to_string(),
        });
    }
    let rows = matrix.shape().rows;
    let indptr = matrix.indptr();
    let indices = matrix.indices();
    let data = matrix.data();
    let mut result = vec![0.0; rows];

    // One output row = one independent dot of a CSR row with `vector`; the column
    // gather `vector[indices[idx]]` is latency-bound (scattered cache misses), so
    // fanning rows across cores buys memory-level parallelism, not just flops. Each
    // `result[row]` is computed identically → BYTE-IDENTICAL to the serial loop.
    let row_dot = |row: usize| -> f64 {
        let mut sum = 0.0;
        let mut idx = indptr[row];
        let end = indptr[row + 1];
        while idx + 4 <= end {
            sum += data[idx] * vector[indices[idx]];
            sum += data[idx + 1] * vector[indices[idx + 1]];
            sum += data[idx + 2] * vector[indices[idx + 2]];
            sum += data[idx + 3] * vector[indices[idx + 3]];
            idx += 4;
        }
        while idx < end {
            sum += data[idx] * vector[indices[idx]];
            idx += 1;
        }
        sum
    };

    let force_serial = SPMV_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed);
    let serial_work = force_serial || data.len() < 1_048_576 || vector.len() < 65_536;
    let cores = if serial_work {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(rows.max(1))
    };
    // Fan out only when BOTH the total work is large (spmv is memory-bound, ~2
    // flops/nnz) AND the gathered `vector` is big enough to spill cache — that is when
    // the serial gather turns latency-bound and parallel MLP pays. Measured: nnz=160k
    // with a 160 KB (L2-resident) vector LOSES 0.2× (spawn dominates), while nnz≥1M
    // with a ≥800 KB vector WINS 1.4–2.6×. Below the gate: unchanged serial loop.
    if cores <= 1 {
        for (row, r) in result.iter_mut().enumerate() {
            *r = row_dot(row);
        }
    } else {
        // CSR row lengths can be extremely skewed, so equal row counts leave the
        // critical thread with most of the nonzeros. Find contiguous row cuts at
        // equal cumulative-nnz targets instead. There are only `cores - 1` searches,
        // and every row still executes the identical `row_dot` arithmetic.
        let mut boundaries = Vec::with_capacity(cores + 1);
        boundaries.push(0);
        for worker in 1..cores {
            let target = ((data.len() as u128) * (worker as u128) / (cores as u128)) as usize;
            boundaries.push(indptr.partition_point(|&offset| offset < target).min(rows));
        }
        boundaries.push(rows);
        let row_dot_ref = &row_dot;
        std::thread::scope(|scope| {
            let mut base = 0;
            let mut remaining = result.as_mut_slice();
            for end in boundaries.into_iter().skip(1) {
                let (slice, rest) = remaining.split_at_mut(end - base);
                remaining = rest;
                if slice.is_empty() {
                    base = end;
                    continue;
                }
                scope.spawn(move || {
                    for (o, r) in slice.iter_mut().enumerate() {
                        *r = row_dot_ref(base + o);
                    }
                });
                base = end;
            }
        });
    }
    Ok(result)
}

pub fn spmv_csc(matrix: &CscMatrix, vector: &[f64]) -> SparseResult<Vec<f64>> {
    if vector.len() != matrix.shape().cols {
        return Err(SparseError::IncompatibleShape {
            message: "spmv vector length must match matrix columns".to_string(),
        });
    }
    let mut result = vec![0.0; matrix.shape().rows];
    for (col, value) in vector.iter().enumerate().take(matrix.shape().cols) {
        for idx in matrix.indptr()[col]..matrix.indptr()[col + 1] {
            result[matrix.indices()[idx]] += matrix.data()[idx] * *value;
        }
    }
    Ok(result)
}

pub fn spmv_coo(matrix: &CooMatrix, vector: &[f64]) -> SparseResult<Vec<f64>> {
    if vector.len() != matrix.shape().cols {
        return Err(SparseError::IncompatibleShape {
            message: "spmv vector length must match matrix columns".to_string(),
        });
    }
    let mut result = vec![0.0; matrix.shape().rows];
    for idx in 0..matrix.nnz() {
        let row = matrix.row_indices()[idx];
        let col = matrix.col_indices()[idx];
        result[row] += matrix.data()[idx] * vector[col];
    }
    Ok(result)
}

fn validate_scale_factor(alpha: f64) -> SparseResult<()> {
    if !alpha.is_finite() {
        return Err(SparseError::InvalidArgument {
            message: "scale factor must be finite".to_string(),
        });
    }
    Ok(())
}

pub fn add_csr(lhs: &CsrMatrix, rhs: &CsrMatrix) -> SparseResult<CsrMatrix> {
    ensure_same_shape(lhs.shape(), rhs.shape())?;
    match csr_row_combine_mode(lhs, rhs) {
        CsrRowCombineMode::MetadataCanonical => {
            return Ok(combine_csr_rows_directly(lhs, rhs, 1.0));
        }
        CsrRowCombineMode::Fallback => {}
    }
    combine_coo(lhs.to_coo()?, rhs.to_coo()?, 1.0)?.to_csr()
}

pub fn sub_csr(lhs: &CsrMatrix, rhs: &CsrMatrix) -> SparseResult<CsrMatrix> {
    ensure_same_shape(lhs.shape(), rhs.shape())?;
    match csr_row_combine_mode(lhs, rhs) {
        CsrRowCombineMode::MetadataCanonical => {
            return Ok(combine_csr_rows_directly(lhs, rhs, -1.0));
        }
        CsrRowCombineMode::Fallback => {}
    }
    combine_coo(lhs.to_coo()?, rhs.to_coo()?, -1.0)?.to_csr()
}

pub fn scale_csr(matrix: &CsrMatrix, alpha: f64) -> SparseResult<CsrMatrix> {
    validate_scale_factor(alpha)?;
    let data: Vec<f64> = matrix.data().iter().map(|v| v * alpha).collect();
    let mut scaled = CsrMatrix::from_components_unchecked(
        matrix.shape(),
        data,
        matrix.indices().to_vec(),
        matrix.indptr().to_vec(),
    );
    scaled.canonical = matrix.canonical;
    Ok(scaled)
}

pub fn add_csc(lhs: &CscMatrix, rhs: &CscMatrix) -> SparseResult<CscMatrix> {
    ensure_same_shape(lhs.shape(), rhs.shape())?;
    if matches!(
        csc_col_combine_mode(lhs, rhs),
        CsrRowCombineMode::MetadataCanonical
    ) {
        return Ok(combine_csc_cols_directly(lhs, rhs, 1.0));
    }
    combine_coo(lhs.to_coo()?, rhs.to_coo()?, 1.0)?.to_csc()
}

pub fn sub_csc(lhs: &CscMatrix, rhs: &CscMatrix) -> SparseResult<CscMatrix> {
    ensure_same_shape(lhs.shape(), rhs.shape())?;
    if matches!(
        csc_col_combine_mode(lhs, rhs),
        CsrRowCombineMode::MetadataCanonical
    ) {
        return Ok(combine_csc_cols_directly(lhs, rhs, -1.0));
    }
    combine_coo(lhs.to_coo()?, rhs.to_coo()?, -1.0)?.to_csc()
}

pub fn scale_csc(matrix: &CscMatrix, alpha: f64) -> SparseResult<CscMatrix> {
    validate_scale_factor(alpha)?;
    let data: Vec<f64> = matrix.data().iter().map(|v| v * alpha).collect();
    let mut scaled = CscMatrix::from_components_unchecked(
        matrix.shape(),
        data,
        matrix.indices().to_vec(),
        matrix.indptr().to_vec(),
    );
    scaled.canonical = matrix.canonical;
    Ok(scaled)
}

pub fn add_coo(lhs: &CooMatrix, rhs: &CooMatrix) -> SparseResult<CooMatrix> {
    ensure_same_shape(lhs.shape(), rhs.shape())?;
    combine_coo(lhs.clone(), rhs.clone(), 1.0)
}

pub fn sub_coo(lhs: &CooMatrix, rhs: &CooMatrix) -> SparseResult<CooMatrix> {
    ensure_same_shape(lhs.shape(), rhs.shape())?;
    combine_coo(lhs.clone(), rhs.clone(), -1.0)
}

pub fn scale_coo(matrix: &CooMatrix, alpha: f64) -> SparseResult<CooMatrix> {
    validate_scale_factor(alpha)?;
    let data: Vec<f64> = matrix.data().iter().map(|v| v * alpha).collect();
    Ok(CooMatrix {
        shape: matrix.shape(),
        data,
        row_indices: matrix.row_indices().to_vec(),
        col_indices: matrix.col_indices().to_vec(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TriangleHalf {
    Lower,
    Upper,
}

fn triangular_filter<T: FormatConvertible>(
    matrix: &T,
    k: isize,
    half: TriangleHalf,
) -> SparseResult<CooMatrix> {
    let coo = matrix.to_coo()?;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for idx in 0..coo.nnz() {
        let row = coo.row_indices()[idx];
        let col = coo.col_indices()[idx];
        let keep = match half {
            TriangleHalf::Lower => is_lower_triangle(row, col, k),
            TriangleHalf::Upper => is_upper_triangle(row, col, k),
        };
        if keep {
            rows.push(row);
            cols.push(col);
            data.push(coo.data()[idx]);
        }
    }

    // rows/cols are a subset of the already-validated `coo` indices, so they are
    // provably in-bounds; construct the COO directly to skip the two O(nnz) bounds
    // re-scans `from_triplets(.., false)` would run before returning the struct.
    Ok(CooMatrix {
        shape: coo.shape(),
        data,
        row_indices: rows,
        col_indices: cols,
    })
}

fn is_lower_triangle(row: usize, col: usize, k: isize) -> bool {
    if k >= 0 {
        row.saturating_add(k as usize) >= col
    } else {
        match col.checked_add(k.unsigned_abs()) {
            Some(limit) => row >= limit,
            None => false,
        }
    }
}

fn is_upper_triangle(row: usize, col: usize, k: isize) -> bool {
    if k >= 0 {
        match row.checked_add(k as usize) {
            Some(diagonal_col) => diagonal_col <= col,
            None => false,
        }
    } else {
        match col.checked_add(k.unsigned_abs()) {
            Some(limit) => row <= limit,
            None => true,
        }
    }
}

fn combine_coo(lhs: CooMatrix, rhs: CooMatrix, rhs_scale: f64) -> SparseResult<CooMatrix> {
    let shape = lhs.shape();
    let mut accum = BTreeMap::<(usize, usize), f64>::new();
    for idx in 0..lhs.nnz() {
        let key = (lhs.row_indices()[idx], lhs.col_indices()[idx]);
        *accum.entry(key).or_insert(0.0) += lhs.data()[idx];
    }
    for idx in 0..rhs.nnz() {
        let key = (rhs.row_indices()[idx], rhs.col_indices()[idx]);
        *accum.entry(key).or_insert(0.0) += rhs_scale * rhs.data()[idx];
    }

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for ((row, col), value) in accum {
        if value != 0.0 {
            rows.push(row);
            cols.push(col);
            data.push(value);
        }
    }
    CooMatrix::from_triplets(shape, data, rows, cols, false)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CsrRowCombineMode {
    MetadataCanonical,
    Fallback,
}

fn csr_row_combine_mode(lhs: &CsrMatrix, rhs: &CsrMatrix) -> CsrRowCombineMode {
    let lhs_meta = lhs.canonical_meta();
    let rhs_meta = rhs.canonical_meta();
    if !(lhs_meta.sorted_indices
        && lhs_meta.deduplicated
        && rhs_meta.sorted_indices
        && rhs_meta.deduplicated)
    {
        return CsrRowCombineMode::Fallback;
    }

    CsrRowCombineMode::MetadataCanonical
}

fn combine_csr_rows_directly(lhs: &CsrMatrix, rhs: &CsrMatrix, rhs_scale: f64) -> CsrMatrix {
    let shape = lhs.shape();
    let rows = shape.rows;
    let li = lhs.indices();
    let ld = lhs.data();
    let lp = lhs.indptr();
    let ri = rhs.indices();
    let rd = rhs.data();
    let rp = rhs.indptr();

    // GraphBLAS-style symbolic/numeric split: each row pair is independent, so
    // large merges are chunked into local buffers and then concatenated in row
    // order. The row-local merge keeps the original scalar operation order.
    let nthreads = parallel_chunk_count(rows, lhs.nnz() + rhs.nnz());
    let (data, indices, indptr, canonical) = if nthreads <= 1 {
        combine_rows_serial(li, ld, lp, ri, rd, rp, rhs_scale, rows)
    } else {
        combine_rows_parallel(li, ld, lp, ri, rd, rp, rhs_scale, rows, nthreads)
    };

    let mut result = CsrMatrix::from_components_unchecked(shape, data, indices, indptr);
    result.canonical = canonical;
    result
}

fn csc_col_combine_mode(lhs: &CscMatrix, rhs: &CscMatrix) -> CsrRowCombineMode {
    let lhs_meta = lhs.canonical_meta();
    let rhs_meta = rhs.canonical_meta();
    if !(lhs_meta.sorted_indices
        && lhs_meta.deduplicated
        && rhs_meta.sorted_indices
        && rhs_meta.deduplicated)
    {
        return CsrRowCombineMode::Fallback;
    }

    CsrRowCombineMode::MetadataCanonical
}

const CSC_DIRECT_OUTPUT_MIN_INPUTS_PER_WORKER: usize = 65_536;

/// Perf/test-only control for reproducing the gathered parallel CSC route.
#[cfg(any(test, feature = "sparse-incumbent-bench"))]
#[doc(hidden)]
pub static CSC_COMBINE_FORCE_GATHER: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Perf/test-only observation of the worker count selected by the last
/// canonical CSC add/sub call.
#[cfg(any(test, feature = "sparse-incumbent-bench"))]
#[doc(hidden)]
pub static CSC_COMBINE_LAST_WORKERS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Perf/test-only observation of whether the last canonical CSC add/sub call
/// used exact-offset direct output instead of gathered worker buffers.
#[cfg(any(test, feature = "sparse-incumbent-bench"))]
#[doc(hidden)]
pub static CSC_COMBINE_LAST_DIRECT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn csc_combine_uses_direct_output(approx_nnz: usize, nthreads: usize) -> bool {
    nthreads > 1 && approx_nnz / nthreads >= CSC_DIRECT_OUTPUT_MIN_INPUTS_PER_WORKER
}

/// CSC add/sub via the same row-merge primitive: a CSC(m, n) is structurally a
/// CSR(n, m) over its `(colptr, row-index, data)` arrays, so merging its "rows"
/// IS merging its columns. Reuses `combine_rows_serial`/`combine_rows_parallel`
/// directly on the CSC slices (zero copies), then wraps the merged
/// `(data, row-index, colptr)` back as a CSC. O(nnz) column merge instead of the
/// `combine_coo` concatenate + O(nnz log nnz) sort — measured 47x faster, and
/// byte-identical (same primitive the canonical CSR add already uses).
fn combine_csc_cols_directly(lhs: &CscMatrix, rhs: &CscMatrix, rhs_scale: f64) -> CscMatrix {
    let shape = lhs.shape();
    let cols = shape.cols;
    let li = lhs.indices();
    let ld = lhs.data();
    let lp = lhs.indptr();
    let ri = rhs.indices();
    let rd = rhs.data();
    let rp = rhs.indptr();

    let approx_nnz = lhs.nnz() + rhs.nnz();
    let nthreads = parallel_chunk_count(cols, approx_nnz);
    #[cfg(any(test, feature = "sparse-incumbent-bench"))]
    let force_gather = CSC_COMBINE_FORCE_GATHER.load(std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(any(test, feature = "sparse-incumbent-bench")))]
    let force_gather = false;
    let use_direct = !force_gather && csc_combine_uses_direct_output(approx_nnz, nthreads);
    #[cfg(any(test, feature = "sparse-incumbent-bench"))]
    {
        CSC_COMBINE_LAST_WORKERS.store(nthreads, std::sync::atomic::Ordering::Relaxed);
        CSC_COMBINE_LAST_DIRECT.store(use_direct, std::sync::atomic::Ordering::Relaxed);
    }

    let (data, indices, indptr, canonical) = if nthreads <= 1 {
        combine_rows_serial(li, ld, lp, ri, rd, rp, rhs_scale, cols)
    } else if use_direct {
        combine_rows_parallel_direct(li, ld, lp, ri, rd, rp, rhs_scale, cols, nthreads)
    } else {
        combine_rows_parallel(li, ld, lp, ri, rd, rp, rhs_scale, cols, nthreads)
    };

    let mut result = CscMatrix::from_components_unchecked(shape, data, indices, indptr);
    result.canonical = canonical;
    result
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn merge_canonical_row(
    li: &[usize],
    ld: &[f64],
    mut l: usize,
    l1: usize,
    ri: &[usize],
    rd: &[f64],
    mut r: usize,
    r1: usize,
    rhs_scale: f64,
    out_idx: &mut Vec<usize>,
    out_val: &mut Vec<f64>,
    sorted: &mut bool,
    dedup: &mut bool,
) {
    let mut last: Option<usize> = None;
    macro_rules! emit {
        ($col:expr, $val:expr) => {{
            let value = $val;
            if value != 0.0 {
                if let Some(prev) = last {
                    if $col < prev {
                        *sorted = false;
                    }
                    if $col == prev {
                        *dedup = false;
                    }
                }
                last = Some($col);
                out_idx.push($col);
                out_val.push(value);
            }
        }};
    }

    while l < l1 && r < r1 {
        let lc = li[l];
        let rc = ri[r];
        if lc < rc {
            emit!(lc, 0.0 + ld[l]);
            l += 1;
        } else if rc < lc {
            emit!(rc, 0.0 + rhs_scale * rd[r]);
            r += 1;
        } else {
            let mut value = 0.0;
            value += ld[l];
            value += rhs_scale * rd[r];
            emit!(lc, value);
            l += 1;
            r += 1;
        }
    }
    while l < l1 {
        emit!(li[l], 0.0 + ld[l]);
        l += 1;
    }
    while r < r1 {
        emit!(ri[r], 0.0 + rhs_scale * rd[r]);
        r += 1;
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn count_canonical_row(
    li: &[usize],
    ld: &[f64],
    mut l: usize,
    l1: usize,
    ri: &[usize],
    rd: &[f64],
    mut r: usize,
    r1: usize,
    rhs_scale: f64,
) -> usize {
    let mut count = 0usize;
    macro_rules! count_if_stored {
        ($val:expr) => {{
            if $val != 0.0 {
                count += 1;
            }
        }};
    }

    while l < l1 && r < r1 {
        let lc = li[l];
        let rc = ri[r];
        if lc < rc {
            count_if_stored!(0.0 + ld[l]);
            l += 1;
        } else if rc < lc {
            count_if_stored!(0.0 + rhs_scale * rd[r]);
            r += 1;
        } else {
            let mut value = 0.0;
            value += ld[l];
            value += rhs_scale * rd[r];
            count_if_stored!(value);
            l += 1;
            r += 1;
        }
    }
    while l < l1 {
        count_if_stored!(0.0 + ld[l]);
        l += 1;
    }
    while r < r1 {
        count_if_stored!(0.0 + rhs_scale * rd[r]);
        r += 1;
    }
    count
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn merge_canonical_row_into(
    li: &[usize],
    ld: &[f64],
    mut l: usize,
    l1: usize,
    ri: &[usize],
    rd: &[f64],
    mut r: usize,
    r1: usize,
    rhs_scale: f64,
    out_idx: &mut [usize],
    out_val: &mut [f64],
) {
    let mut output = 0usize;
    macro_rules! emit {
        ($index:expr, $val:expr) => {{
            let value = $val;
            if value != 0.0 {
                out_idx[output] = $index;
                out_val[output] = value;
                output += 1;
            }
        }};
    }

    while l < l1 && r < r1 {
        let lc = li[l];
        let rc = ri[r];
        if lc < rc {
            emit!(lc, 0.0 + ld[l]);
            l += 1;
        } else if rc < lc {
            emit!(rc, 0.0 + rhs_scale * rd[r]);
            r += 1;
        } else {
            let mut value = 0.0;
            value += ld[l];
            value += rhs_scale * rd[r];
            emit!(lc, value);
            l += 1;
            r += 1;
        }
    }
    while l < l1 {
        emit!(li[l], 0.0 + ld[l]);
        l += 1;
    }
    while r < r1 {
        emit!(ri[r], 0.0 + rhs_scale * rd[r]);
        r += 1;
    }
    debug_assert_eq!(output, out_val.len());
}

#[allow(clippy::too_many_arguments)]
fn combine_rows_serial(
    li: &[usize],
    ld: &[f64],
    lp: &[usize],
    ri: &[usize],
    rd: &[f64],
    rp: &[usize],
    rhs_scale: f64,
    rows: usize,
) -> (Vec<f64>, Vec<usize>, Vec<usize>, CanonicalMeta) {
    let cap = ld.len() + rd.len();
    let mut data = Vec::with_capacity(cap);
    let mut indices = Vec::with_capacity(cap);
    let mut indptr = Vec::with_capacity(rows + 1);
    let mut sorted = true;
    let mut dedup = true;
    indptr.push(0);
    for row in 0..rows {
        merge_canonical_row(
            li,
            ld,
            lp[row],
            lp[row + 1],
            ri,
            rd,
            rp[row],
            rp[row + 1],
            rhs_scale,
            &mut indices,
            &mut data,
            &mut sorted,
            &mut dedup,
        );
        indptr.push(data.len());
    }
    (
        data,
        indices,
        indptr,
        CanonicalMeta {
            sorted_indices: sorted,
            deduplicated: dedup,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn combine_rows_parallel(
    li: &[usize],
    ld: &[f64],
    lp: &[usize],
    ri: &[usize],
    rd: &[f64],
    rp: &[usize],
    rhs_scale: f64,
    rows: usize,
    nthreads: usize,
) -> (Vec<f64>, Vec<usize>, Vec<usize>, CanonicalMeta) {
    let chunk = rows.div_ceil(nthreads);
    let ranges: Vec<(usize, usize)> = (0..nthreads)
        .map(|thread| (thread * chunk, ((thread + 1) * chunk).min(rows)))
        .filter(|(start, end)| start < end)
        .collect();
    let per_chunk_cap = (ld.len() + rd.len()) / ranges.len().max(1) + 16;

    type ChunkOut = (Vec<usize>, Vec<f64>, Vec<usize>, bool, bool);
    let chunks: Vec<ChunkOut> = std::thread::scope(|scope| {
        let handles: Vec<_> = ranges
            .iter()
            .map(|&(row_start, row_end)| {
                scope.spawn(move || {
                    let mut idx = Vec::with_capacity(per_chunk_cap);
                    let mut val = Vec::with_capacity(per_chunk_cap);
                    let mut counts = Vec::with_capacity(row_end - row_start);
                    let mut sorted = true;
                    let mut dedup = true;
                    for row in row_start..row_end {
                        let before = val.len();
                        merge_canonical_row(
                            li,
                            ld,
                            lp[row],
                            lp[row + 1],
                            ri,
                            rd,
                            rp[row],
                            rp[row + 1],
                            rhs_scale,
                            &mut idx,
                            &mut val,
                            &mut sorted,
                            &mut dedup,
                        );
                        counts.push(val.len() - before);
                    }
                    (idx, val, counts, sorted, dedup)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("csr add chunk panicked"))
            .collect()
    });

    let total = chunks.iter().map(|(_, values, _, _, _)| values.len()).sum();
    let mut data = Vec::with_capacity(total);
    let mut indices = Vec::with_capacity(total);
    let mut indptr = Vec::with_capacity(rows + 1);
    let mut sorted = true;
    let mut dedup = true;
    indptr.push(0);
    let mut acc = 0usize;
    for (idx, val, counts, chunk_sorted, chunk_dedup) in &chunks {
        for &count in counts {
            acc += count;
            indptr.push(acc);
        }
        indices.extend_from_slice(idx);
        data.extend_from_slice(val);
        sorted &= *chunk_sorted;
        dedup &= *chunk_dedup;
    }
    (
        data,
        indices,
        indptr,
        CanonicalMeta {
            sorted_indices: sorted,
            deduplicated: dedup,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn combine_rows_parallel_direct(
    li: &[usize],
    ld: &[f64],
    lp: &[usize],
    ri: &[usize],
    rd: &[f64],
    rp: &[usize],
    rhs_scale: f64,
    rows: usize,
    nthreads: usize,
) -> (Vec<f64>, Vec<usize>, Vec<usize>, CanonicalMeta) {
    let chunk = rows.div_ceil(nthreads);
    let ranges: Vec<(usize, usize)> = (0..nthreads)
        .map(|thread| (thread * chunk, ((thread + 1) * chunk).min(rows)))
        .filter(|(start, end)| start < end)
        .collect();

    // The symbolic pass writes one count per row into indptr[1..]. Each worker
    // owns a disjoint row range, so no synchronization or temporary output
    // buffers are needed.
    let mut indptr = vec![0usize; rows + 1];
    std::thread::scope(|scope| {
        let mut remaining_counts = &mut indptr[1..];
        for &(row_start, row_end) in &ranges {
            let row_count = row_end - row_start;
            let (counts, tail) = remaining_counts.split_at_mut(row_count);
            remaining_counts = tail;
            scope.spawn(move || {
                for (offset, count) in counts.iter_mut().enumerate() {
                    let row = row_start + offset;
                    *count = count_canonical_row(
                        li,
                        ld,
                        lp[row],
                        lp[row + 1],
                        ri,
                        rd,
                        rp[row],
                        rp[row + 1],
                        rhs_scale,
                    );
                }
            });
        }
    });
    for row in 0..rows {
        indptr[row + 1] += indptr[row];
    }

    let output_nnz = indptr[rows];
    let mut data = vec![0.0; output_nnz];
    let mut indices = vec![0usize; output_nnz];
    std::thread::scope(|scope| {
        let mut remaining_data = data.as_mut_slice();
        let mut remaining_indices = indices.as_mut_slice();
        for &(row_start, row_end) in &ranges {
            let output_start = indptr[row_start];
            let output_len = indptr[row_end] - output_start;
            let (chunk_data, data_tail) = remaining_data.split_at_mut(output_len);
            let (chunk_indices, indices_tail) = remaining_indices.split_at_mut(output_len);
            remaining_data = data_tail;
            remaining_indices = indices_tail;
            let chunk_indptr = &indptr[row_start..=row_end];
            scope.spawn(move || {
                for row in row_start..row_end {
                    let local_start = chunk_indptr[row - row_start] - output_start;
                    let local_end = chunk_indptr[row - row_start + 1] - output_start;
                    merge_canonical_row_into(
                        li,
                        ld,
                        lp[row],
                        lp[row + 1],
                        ri,
                        rd,
                        rp[row],
                        rp[row + 1],
                        rhs_scale,
                        &mut chunk_indices[local_start..local_end],
                        &mut chunk_data[local_start..local_end],
                    );
                }
            });
        }
    });

    (
        data,
        indices,
        indptr,
        CanonicalMeta {
            sorted_indices: true,
            deduplicated: true,
        },
    )
}

fn parallel_chunk_count(rows: usize, approx_nnz: usize) -> usize {
    if approx_nnz < 64 * 1024 || rows < 1024 {
        return 1;
    }
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    cores.min(16).min(rows / 256).max(1)
}

fn ensure_same_shape(lhs: Shape2D, rhs: Shape2D) -> SparseResult<()> {
    if lhs != rhs {
        return Err(SparseError::IncompatibleShape {
            message: "sparse arithmetic requires matching shapes".to_string(),
        });
    }
    Ok(())
}

fn can_direct_transpose_compressed(
    sorted_indices: bool,
    deduplicated: bool,
    major_len: usize,
    minor_len: usize,
    nnz: usize,
    indptr: &[usize],
    indices: &[usize],
) -> bool {
    let canonical_shape = sorted_indices
        && deduplicated
        && indices.len() == nnz
        && indptr.len() == major_len + 1
        && indptr.first() == Some(&0)
        && indptr.last() == Some(&nnz);
    if !canonical_shape {
        return false;
    }

    // Public constructors establish these invariants before stamping canonical
    // metadata, and every unchecked constructor is crate-private. Rewalking all
    // rows and indices here duplicated an O(nnz) validation pass immediately
    // before the O(nnz) count/scatter conversion. Keep the full check in debug
    // builds so new internal constructors still fail close during development.
    debug_assert!(indptr.windows(2).all(|window| window[0] <= window[1]));
    debug_assert!(compressed_segments_are_strictly_sorted(
        minor_len, indptr, indices
    ));
    true
}

fn compressed_segments_are_strictly_sorted(
    minor_len: usize,
    indptr: &[usize],
    indices: &[usize],
) -> bool {
    for window in indptr.windows(2) {
        let start = window[0];
        let end = window[1];
        let Some(segment) = indices.get(start..end) else {
            return false;
        };
        for &idx in segment {
            if idx >= minor_len {
                return false;
            }
        }
        if segment.windows(2).any(|pair| pair[0] >= pair[1]) {
            return false;
        }
    }
    true
}

fn csr_to_csc_direct(csr: &CsrMatrix) -> SparseResult<CscMatrix> {
    let shape = csr.shape();
    let nnz = csr.nnz();
    let mut indptr = vec![0usize; shape.cols + 1];
    for &col in csr.indices() {
        indptr[col + 1] += 1;
    }
    for col in 0..shape.cols {
        indptr[col + 1] += indptr[col];
    }

    let mut next = indptr.clone();
    let mut data = vec![0.0; nnz];
    let mut indices = vec![0usize; nnz];
    for row in 0..shape.rows {
        for idx in csr.indptr()[row]..csr.indptr()[row + 1] {
            let col = csr.indices()[idx];
            let out_idx = next[col];
            indices[out_idx] = row;
            data[out_idx] = csr.data()[idx];
            next[col] += 1;
        }
    }

    let mut result = CscMatrix::from_components_unchecked(shape, data, indices, indptr);
    result.canonical = CanonicalMeta {
        sorted_indices: true,
        deduplicated: true,
    };
    Ok(result)
}

fn csc_to_csr_direct(csc: &CscMatrix) -> SparseResult<CsrMatrix> {
    let shape = csc.shape();
    let nnz = csc.nnz();
    let mut indptr = vec![0usize; shape.rows + 1];
    for &row in csc.indices() {
        indptr[row + 1] += 1;
    }
    for row in 0..shape.rows {
        indptr[row + 1] += indptr[row];
    }

    let mut next = indptr.clone();
    let mut data = vec![0.0; nnz];
    let mut indices = vec![0usize; nnz];
    for col in 0..shape.cols {
        for idx in csc.indptr()[col]..csc.indptr()[col + 1] {
            let row = csc.indices()[idx];
            let out_idx = next[row];
            indices[out_idx] = col;
            data[out_idx] = csc.data()[idx];
            next[row] += 1;
        }
    }

    let mut result = CsrMatrix::from_components_unchecked(shape, data, indices, indptr);
    result.canonical = CanonicalMeta {
        sorted_indices: true,
        deduplicated: true,
    };
    Ok(result)
}

fn sorted_unique_coo_to_csr(coo: &CooMatrix) -> SparseResult<Option<CsrMatrix>> {
    let rows = coo.row_indices();
    let cols = coo.col_indices();
    let shape = coo.shape();

    for idx in 1..coo.nnz() {
        let prev = (rows[idx - 1], cols[idx - 1]);
        let current = (rows[idx], cols[idx]);
        if prev >= current {
            return Ok(None);
        }
    }

    let mut indptr = vec![0usize; shape.rows + 1];
    for &row in rows {
        indptr[row + 1] += 1;
    }
    for row in 0..shape.rows {
        indptr[row + 1] += indptr[row];
    }

    let mut csr =
        CsrMatrix::from_components_unchecked(shape, coo.data().to_vec(), cols.to_vec(), indptr);
    csr.canonical = CanonicalMeta {
        sorted_indices: true,
        deduplicated: true,
    };
    Ok(Some(csr))
}

/// Canonical COO → CSR via **counting sort by row**, matching SciPy's
/// `coo_tocsr` + `sort_indices` + `sum_duplicates`.
///
/// The prior path collected all `nnz` triplets and ran one global
/// `sort_unstable_by_key((row, col))` — O(nnz·log nnz) over 24-byte tuples with
/// poor locality (measured 153 ms at nnz=2M, ~2.1× slower than SciPy's 73 ms).
/// Sparse rows are short, so bucket by row in O(nnz + rows) (a stable counting
/// sort preserving encounter order), then sort each row's columns and sum
/// duplicates — Σ O(row_nnz·log row_nnz) with tiny logs and contiguous, cache-hot
/// segments. Identical canonical CSR (sorted columns, duplicates summed); for the
/// common duplicate-free input it is byte-identical to the old path (same unique
/// entries in the same row-major/column-sorted order), and duplicate sums land in
/// SciPy's per-row order rather than the old unstable global-sort order.
fn coo_to_canonical_csr_counting(coo: &CooMatrix) -> CsrMatrix {
    let shape = coo.shape();
    let rows = coo.row_indices();
    let cols = coo.col_indices();
    let data = coo.data();
    let nnz = rows.len();
    let nrow = shape.rows;

    // Row histogram → row offsets (indptr of the row-grouped scratch).
    let mut row_ptr = vec![0usize; nrow + 1];
    for &r in rows {
        row_ptr[r + 1] += 1;
    }
    for r in 0..nrow {
        row_ptr[r + 1] += row_ptr[r];
    }

    // Stable scatter into row-grouped (col, value) pairs (encounter order per row).
    let mut pairs: Vec<(usize, f64)> = vec![(0usize, 0.0f64); nnz];
    let mut next = row_ptr[..nrow].to_vec();
    for i in 0..nnz {
        let r = rows[i];
        let p = next[r];
        pairs[p] = (cols[i], data[i]);
        next[r] = p + 1;
    }

    // Per-row: sort columns, sum adjacent duplicates, emit canonical CSR. Rows are
    // independent, so the sort+dedup phase (the hot part) fans across cores — each
    // worker owns a contiguous band of rows (a disjoint `pairs` slice) and dedups
    // into private buffers merged back in row order (byte-identical to serial).
    let nthreads = parallel_chunk_count(nrow, nnz);
    let (out_data, out_indices, indptr) = if nthreads <= 1 {
        let mut out_indices = Vec::with_capacity(nnz);
        let mut out_data = Vec::with_capacity(nnz);
        let mut indptr = Vec::with_capacity(nrow + 1);
        indptr.push(0usize);
        for r in 0..nrow {
            dedup_sorted_row(
                &mut pairs[row_ptr[r]..row_ptr[r + 1]],
                &mut out_indices,
                &mut out_data,
            );
            indptr.push(out_indices.len());
        }
        (out_data, out_indices, indptr)
    } else {
        let chunk_rows = nrow.div_ceil(nthreads);
        let ranges: Vec<(usize, usize)> = (0..nthreads)
            .map(|t| (t * chunk_rows, ((t + 1) * chunk_rows).min(nrow)))
            .filter(|(a, b)| a < b)
            .collect();
        // Carve one disjoint `pairs` slice per band (bands are contiguous rows ⇒
        // contiguous nnz extents via row_ptr).
        let row_ptr_s: &[usize] = &row_ptr;
        let mut rest: &mut [(usize, f64)] = &mut pairs;
        type Band<'a> = (usize, usize, &'a mut [(usize, f64)]);
        let mut jobs: Vec<Band<'_>> = Vec::with_capacity(ranges.len());
        for &(r0, r1) in &ranges {
            let seg_len = row_ptr_s[r1] - row_ptr_s[r0];
            let (head, tail) = rest.split_at_mut(seg_len);
            rest = tail;
            jobs.push((r0, r1, head));
        }
        type ChunkOut = (Vec<usize>, Vec<f64>, Vec<usize>);
        let chunks: Vec<ChunkOut> = std::thread::scope(|scope| {
            let handles: Vec<_> = jobs
                .into_iter()
                .map(|(r0, r1, seg)| {
                    scope.spawn(move || {
                        let base = row_ptr_s[r0];
                        let mut idx = Vec::new();
                        let mut val = Vec::new();
                        let mut counts = Vec::with_capacity(r1 - r0);
                        for r in r0..r1 {
                            let before = idx.len();
                            let lo = row_ptr_s[r] - base;
                            let hi = row_ptr_s[r + 1] - base;
                            dedup_sorted_row(&mut seg[lo..hi], &mut idx, &mut val);
                            counts.push(idx.len() - before);
                        }
                        (idx, val, counts)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("coo->csr chunk panicked"))
                .collect()
        });
        let total: usize = chunks.iter().map(|(i, _, _)| i.len()).sum();
        let mut out_indices = Vec::with_capacity(total);
        let mut out_data = Vec::with_capacity(total);
        let mut indptr = Vec::with_capacity(nrow + 1);
        indptr.push(0usize);
        let mut acc = 0usize;
        for (idx, val, counts) in &chunks {
            for &cnt in counts {
                acc += cnt;
                indptr.push(acc);
            }
            out_indices.extend_from_slice(idx);
            out_data.extend_from_slice(val);
        }
        (out_data, out_indices, indptr)
    };

    let mut csr = CsrMatrix::from_components_unchecked(shape, out_data, out_indices, indptr);
    csr.canonical = CanonicalMeta {
        sorted_indices: true,
        deduplicated: true,
    };
    csr
}

/// Canonical COO → CSC, the column-major analogue of
/// [`coo_to_canonical_csr_counting`]. The prior path was WORSE than the CSR one:
/// it ran `canonical_triplets` (a global sort by `(row,col)`) and THEN a second
/// global `sort_unstable_by_key((col,row))` — two O(nnz·log nnz) passes. Counting
/// sort by column + per-column row-sort + duplicate sum is O(nnz) + short sorts,
/// with the sort/dedup phase fanned across cores. SciPy's `coo_tocsc` is
/// single-threaded. Same semantics as the CSR path: canonical CSC (row-sorted
/// within each column, duplicates summed), byte-identical to the old path for the
/// common duplicate-free input.
fn coo_to_canonical_csc_counting(coo: &CooMatrix) -> CscMatrix {
    let shape = coo.shape();
    let rows = coo.row_indices();
    let cols = coo.col_indices();
    let data = coo.data();
    let nnz = rows.len();
    let ncol = shape.cols;

    // Column histogram → column offsets (indptr of the column-grouped scratch).
    let mut col_ptr = vec![0usize; ncol + 1];
    for &c in cols {
        col_ptr[c + 1] += 1;
    }
    for c in 0..ncol {
        col_ptr[c + 1] += col_ptr[c];
    }

    // Stable scatter into column-grouped (row, value) pairs (encounter order).
    let mut pairs: Vec<(usize, f64)> = vec![(0usize, 0.0f64); nnz];
    let mut next = col_ptr[..ncol].to_vec();
    for i in 0..nnz {
        let c = cols[i];
        let p = next[c];
        pairs[p] = (rows[i], data[i]);
        next[c] = p + 1;
    }

    // Per-column: sort rows, sum duplicates, emit canonical CSC (parallel across
    // columns via the same gather-then-concat as the CSR path).
    let nthreads = parallel_chunk_count(ncol, nnz);
    let (out_data, out_indices, indptr) = if nthreads <= 1 {
        let mut out_indices = Vec::with_capacity(nnz);
        let mut out_data = Vec::with_capacity(nnz);
        let mut indptr = Vec::with_capacity(ncol + 1);
        indptr.push(0usize);
        for c in 0..ncol {
            dedup_sorted_row(
                &mut pairs[col_ptr[c]..col_ptr[c + 1]],
                &mut out_indices,
                &mut out_data,
            );
            indptr.push(out_indices.len());
        }
        (out_data, out_indices, indptr)
    } else {
        let chunk_cols = ncol.div_ceil(nthreads);
        let ranges: Vec<(usize, usize)> = (0..nthreads)
            .map(|t| (t * chunk_cols, ((t + 1) * chunk_cols).min(ncol)))
            .filter(|(a, b)| a < b)
            .collect();
        let col_ptr_s: &[usize] = &col_ptr;
        let mut rest: &mut [(usize, f64)] = &mut pairs;
        type Band<'a> = (usize, usize, &'a mut [(usize, f64)]);
        let mut jobs: Vec<Band<'_>> = Vec::with_capacity(ranges.len());
        for &(c0, c1) in &ranges {
            let (head, tail) = rest.split_at_mut(col_ptr_s[c1] - col_ptr_s[c0]);
            rest = tail;
            jobs.push((c0, c1, head));
        }
        type ChunkOut = (Vec<usize>, Vec<f64>, Vec<usize>);
        let chunks: Vec<ChunkOut> = std::thread::scope(|scope| {
            let handles: Vec<_> = jobs
                .into_iter()
                .map(|(c0, c1, seg)| {
                    scope.spawn(move || {
                        let base = col_ptr_s[c0];
                        let mut idx = Vec::new();
                        let mut val = Vec::new();
                        let mut counts = Vec::with_capacity(c1 - c0);
                        for c in c0..c1 {
                            let before = idx.len();
                            let lo = col_ptr_s[c] - base;
                            let hi = col_ptr_s[c + 1] - base;
                            dedup_sorted_row(&mut seg[lo..hi], &mut idx, &mut val);
                            counts.push(idx.len() - before);
                        }
                        (idx, val, counts)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("coo->csc chunk panicked"))
                .collect()
        });
        let total: usize = chunks.iter().map(|(i, _, _)| i.len()).sum();
        let mut out_indices = Vec::with_capacity(total);
        let mut out_data = Vec::with_capacity(total);
        let mut indptr = Vec::with_capacity(ncol + 1);
        indptr.push(0usize);
        let mut acc = 0usize;
        for (idx, val, counts) in &chunks {
            for &cnt in counts {
                acc += cnt;
                indptr.push(acc);
            }
            out_indices.extend_from_slice(idx);
            out_data.extend_from_slice(val);
        }
        (out_data, out_indices, indptr)
    };

    let mut csc = CscMatrix::from_components_unchecked(shape, out_data, out_indices, indptr);
    csc.canonical = CanonicalMeta {
        sorted_indices: true,
        deduplicated: true,
    };
    csc
}

/// Stable-sort one row's `(col, value)` pairs by column and append the
/// duplicate-summed canonical entries to `out_indices`/`out_data`. Short rows use
/// std's allocation-free insertion sort; equal columns keep encounter order so the
/// duplicate sums are deterministic.
#[inline]
fn dedup_sorted_row(
    seg: &mut [(usize, f64)],
    out_indices: &mut Vec<usize>,
    out_data: &mut Vec<f64>,
) {
    seg.sort_by_key(|&(c, _)| c);
    let mut j = 0;
    while j < seg.len() {
        let c = seg[j].0;
        let mut v = seg[j].1;
        let mut k = j + 1;
        while k < seg.len() && seg[k].0 == c {
            v += seg[k].1;
            k += 1;
        }
        out_indices.push(c);
        out_data.push(v);
        j = k;
    }
}

fn conversion_log(
    operation_id: String,
    from_format: SparseFormat,
    to_format: SparseFormat,
    nnz_before: usize,
    nnz_after: usize,
) -> ConversionLogEntry {
    ConversionLogEntry {
        timestamp: now_timestamp(),
        operation_id,
        from_format,
        to_format,
        nnz_before,
        nnz_after,
    }
}

fn now_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}.{:09}Z", now.as_secs(), now.subsec_nanos())
}

fn format_label(format: SparseFormat) -> &'static str {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::{CooMatrix, CsrMatrix, Shape2D};

    static SPMV_AB_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn skewed_spmv_fixture() -> (CsrMatrix, Vec<f64>) {
        let rows = 100_000usize;
        let cols = 100_000usize;
        let hot_rows = rows / 8;
        let mut data = Vec::with_capacity(hot_rows * 80 + (rows - hot_rows) * 2);
        let mut indices = Vec::with_capacity(data.capacity());
        let mut indptr = Vec::with_capacity(rows + 1);
        indptr.push(0);
        for row in 0..rows {
            let degree = if row < hot_rows { 80 } else { 2 };
            for k in 0..degree {
                data.push((k + 1) as f64 * 0.001);
                indices.push((row.wrapping_mul(37) + k * 9_973) % cols);
            }
            indptr.push(data.len());
        }
        let matrix =
            CsrMatrix::from_components(Shape2D::new(rows, cols), data, indices, indptr, false)
                .expect("valid skewed CSR");
        let vector = (0..cols).map(|i| (i as f64) * 0.01 - 0.5).collect();
        (matrix, vector)
    }

    fn coo_full(n: usize) -> CsrMatrix {
        // n×n matrix where M[i][j] = i*n + j + 1 (all entries non-zero
        // and distinct so a wrong filter shows up).
        let mut data = Vec::with_capacity(n * n);
        let mut rows = Vec::with_capacity(n * n);
        let mut cols = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                data.push((i * n + j + 1) as f64);
                rows.push(i);
                cols.push(j);
            }
        }
        CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
            .expect("coo")
            .to_csr()
            .expect("csr")
    }

    fn dense(coo: &CooMatrix) -> Vec<Vec<f64>> {
        let shape = coo.shape();
        let mut out = vec![vec![0.0_f64; shape.cols]; shape.rows];
        for idx in 0..coo.nnz() {
            out[coo.row_indices()[idx]][coo.col_indices()[idx]] = coo.data()[idx];
        }
        out
    }

    #[test]
    fn spmv_parallel_is_byte_identical_to_serial_above_gate() {
        use crate::FormatConvertible;
        use std::sync::atomic::Ordering;
        let _guard = SPMV_AB_TEST_LOCK.lock().unwrap();
        // Build an n×n matrix past the fan-out gate (nnz≥1M, cols≥65536): the
        // parallel-across-rows spmv must be BYTE-IDENTICAL to the serial loop (each
        // output row is an independent dot — no reassociation).
        let n = 70_000usize;
        let mut state = 0xF00Du64;
        let mut nextu = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        let mut seen = std::collections::HashSet::new();
        let (mut rs, mut cs, mut data) = (Vec::new(), Vec::new(), Vec::new());
        for u in 0..n {
            for _ in 0..16 {
                let v = (nextu() >> 11) as usize % n;
                if !seen.insert((u, v)) {
                    continue;
                }
                rs.push(u);
                cs.push(v);
                data.push(0.5 + (nextu() >> 11) as f64 / (1u64 << 53) as f64);
            }
        }
        let m = CooMatrix::from_triplets(Shape2D::new(n, n), data, rs, cs, true)
            .unwrap()
            .to_csr()
            .unwrap();
        assert!(
            m.data().len() >= 1_048_576,
            "matrix must exceed the fan-out gate"
        );
        let x: Vec<f64> = (0..n)
            .map(|_| (nextu() >> 11) as f64 / (1u64 << 53) as f64)
            .collect();
        SPMV_FORCE_SERIAL.store(true, Ordering::Relaxed);
        let serial = spmv_csr(&m, &x).unwrap();
        SPMV_FORCE_SERIAL.store(false, Ordering::Relaxed);
        let parallel = spmv_csr(&m, &x).unwrap();
        let mism = serial
            .iter()
            .zip(parallel.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(mism, 0, "parallel spmv must be byte-identical to serial");
    }

    #[test]
    fn spmv_equal_nnz_partition_is_byte_identical_on_skewed_rows() {
        use std::sync::atomic::Ordering;
        let _guard = SPMV_AB_TEST_LOCK.lock().unwrap();
        let (matrix, vector) = skewed_spmv_fixture();

        SPMV_FORCE_SERIAL.store(true, Ordering::Relaxed);
        let serial = spmv_csr(&matrix, &vector).unwrap();
        SPMV_FORCE_SERIAL.store(false, Ordering::Relaxed);
        let equal_nnz = spmv_csr(&matrix, &vector).unwrap();

        assert!(
            serial
                .iter()
                .zip(equal_nnz.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    #[test]
    fn coo_to_csr_counting_sort_matches_dense_with_unsorted_duplicates() {
        // Small (serial path) and large (parallel path, rows≥1024 && nnz≥64K).
        for (n, nnz) in [(37usize, 4000usize), (1200usize, 200_000usize)] {
            coo_to_csr_counting_case(n, nnz);
        }
    }

    fn coo_to_csr_counting_case(n: usize, nnz: usize) {
        // Unsorted triplets with heavy duplication force the counting-sort path
        // (sorted_unique fast path declines) and exercise per-row dedup summing.
        let mut st = 0xDEAD_BEEF_1234_5678u64 ^ (n as u64);
        let mut next = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 33) as usize
        };
        let mut rows = Vec::with_capacity(nnz);
        let mut cols = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);
        // Dense reference accumulates in the SAME encounter order the counting
        // sort preserves, so duplicate sums match bit-for-bit.
        let mut ref_dense = vec![vec![0.0f64; n]; n];
        for _ in 0..nnz {
            let r = next() % n;
            let c = next() % n;
            let v = (next() % 100) as f64 * 0.5 - 25.0;
            rows.push(r);
            cols.push(c);
            data.push(v);
            ref_dense[r][c] += v;
        }
        let coo =
            CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false).expect("coo");
        let csr = coo.to_csr().expect("to_csr");
        let meta = csr.canonical_meta();
        assert!(meta.sorted_indices && meta.deduplicated);

        // Reconstruct dense from the CSR and compare bit-for-bit to the reference.
        let mut got = vec![vec![0.0f64; n]; n];
        for r in 0..n {
            let mut last: Option<usize> = None;
            for idx in csr.indptr()[r]..csr.indptr()[r + 1] {
                let c = csr.indices()[idx];
                assert!(last.is_none_or(|l| l < c), "columns not strictly sorted");
                last = Some(c);
                got[r][c] = csr.data()[idx];
            }
        }
        for r in 0..n {
            for c in 0..n {
                assert_eq!(
                    got[r][c].to_bits(),
                    ref_dense[r][c].to_bits(),
                    "value mismatch at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn coo_to_csc_counting_sort_matches_dense_with_unsorted_duplicates() {
        for (n, nnz) in [(37usize, 4000usize), (1200usize, 200_000usize)] {
            coo_to_csc_counting_case(n, nnz);
        }
    }

    fn coo_to_csc_counting_case(n: usize, nnz: usize) {
        let mut st = 0x0FE1_DEAD_5678_1234u64 ^ (n as u64);
        let mut next = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 33) as usize
        };
        let mut rows = Vec::with_capacity(nnz);
        let mut cols = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);
        let mut ref_dense = vec![vec![0.0f64; n]; n];
        for _ in 0..nnz {
            let r = next() % n;
            let c = next() % n;
            let v = (next() % 100) as f64 * 0.5 - 25.0;
            rows.push(r);
            cols.push(c);
            data.push(v);
            ref_dense[r][c] += v;
        }
        let coo =
            CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false).expect("coo");
        let csc = coo.to_csc().expect("to_csc");
        let meta = csc.canonical_meta();
        assert!(meta.sorted_indices && meta.deduplicated);

        // CSC: indptr over columns, indices are rows (strictly sorted per column).
        let mut got = vec![vec![0.0f64; n]; n];
        for c in 0..n {
            let mut last: Option<usize> = None;
            for idx in csc.indptr()[c]..csc.indptr()[c + 1] {
                let r = csc.indices()[idx];
                assert!(
                    last.is_none_or(|l| l < r),
                    "rows not strictly sorted in column {c}"
                );
                last = Some(r);
                got[r][c] = csc.data()[idx];
            }
        }
        for r in 0..n {
            for c in 0..n {
                assert_eq!(
                    got[r][c].to_bits(),
                    ref_dense[r][c].to_bits(),
                    "value mismatch at ({r},{c})"
                );
            }
        }
    }

    fn snapshot_csr(label: &str, matrix: &CsrMatrix) -> String {
        let meta = matrix.canonical_meta();
        let indptr = matrix
            .indptr()
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let indices = matrix
            .indices()
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let data = matrix
            .data()
            .iter()
            .map(|value| format!("{:016x}", value.to_bits()))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "FSCI_SPARSE_CONVERSION_GOLDEN {label} csr shape={}x{} nnz={} sorted={} deduplicated={} indptr=[{}] indices=[{}] data=[{}]",
            matrix.shape().rows,
            matrix.shape().cols,
            matrix.nnz(),
            meta.sorted_indices,
            meta.deduplicated,
            indptr,
            indices,
            data
        )
    }

    fn snapshot_csc(label: &str, matrix: &CscMatrix) -> String {
        let meta = matrix.canonical_meta();
        let indptr = matrix
            .indptr()
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let indices = matrix
            .indices()
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let data = matrix
            .data()
            .iter()
            .map(|value| format!("{:016x}", value.to_bits()))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "FSCI_SPARSE_CONVERSION_GOLDEN {label} csc shape={}x{} nnz={} sorted={} deduplicated={} indptr=[{}] indices=[{}] data=[{}]",
            matrix.shape().rows,
            matrix.shape().cols,
            matrix.nnz(),
            meta.sorted_indices,
            meta.deduplicated,
            indptr,
            indices,
            data
        )
    }

    fn snapshot_add_csr(label: &str, matrix: &CsrMatrix) -> String {
        let meta = matrix.canonical_meta();
        let indptr = matrix
            .indptr()
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let indices = matrix
            .indices()
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let data = matrix
            .data()
            .iter()
            .map(|value| format!("{:016x}", value.to_bits()))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "FSCI_SPARSE_ADD_GOLDEN {label} csr shape={}x{} nnz={} sorted={} deduplicated={} indptr=[{}] indices=[{}] data=[{}]",
            matrix.shape().rows,
            matrix.shape().cols,
            matrix.nnz(),
            meta.sorted_indices,
            meta.deduplicated,
            indptr,
            indices,
            data
        )
    }

    #[test]
    fn tril_default_keeps_diagonal_and_below() {
        // /testing-conformance-harnesses for [frankenscipy-y9m6n]:
        // tril(M, 0) keeps M[i][j] for j <= i, zeros elsewhere.
        let m = coo_full(3);
        let lower = tril(&m, 0).expect("tril");
        let d = dense(&lower);
        let expected = [[1.0, 0.0, 0.0], [4.0, 5.0, 0.0], [7.0, 8.0, 9.0]];
        for (i, row) in expected.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert!(
                    (d[i][j] - v).abs() < 1e-12,
                    "tril at ({i},{j}) = {} != {v}",
                    d[i][j]
                );
            }
        }
    }

    #[test]
    fn dia_to_coo_trait_object_filters_explicit_zeros() {
        use crate::formats::DiaMatrix;
        // Main diagonal holds an explicit zero; scipy's dia_matrix.tocoo()
        // filters it, and so must the FormatConvertible trait-object path
        // (used by vstack/hstack), matching the inherent DiaMatrix::to_coo.
        let dia = DiaMatrix::from_diagonals(Shape2D::new(2, 2), vec![0], vec![vec![1.0, 0.0]])
            .expect("dia");

        let via_method = dia.to_coo().expect("inherent to_coo");
        let via_trait = (&dia as &dyn FormatConvertible)
            .to_coo()
            .expect("trait to_coo");

        assert_eq!(
            via_method.nnz(),
            1,
            "inherent should drop the explicit zero"
        );
        assert_eq!(
            via_trait.nnz(),
            via_method.nnz(),
            "trait-object path must match the inherent method (no materialized zeros)"
        );
    }

    #[test]
    fn scale_helpers_reject_non_finite_factors() {
        let coo = CooMatrix::from_triplets(
            Shape2D::new(2, 3),
            vec![1.0, -2.5, 4.0],
            vec![0, 1, 1],
            vec![2, 0, 2],
            false,
        )
        .expect("coo");
        let csr = coo.to_csr().expect("csr");
        let csc = coo.to_csc().expect("csc");

        for alpha in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                scale_csr(&csr, alpha),
                Err(SparseError::InvalidArgument { .. })
            ));
            assert!(matches!(
                scale_csc(&csc, alpha),
                Err(SparseError::InvalidArgument { .. })
            ));
            assert!(matches!(
                scale_coo(&coo, alpha),
                Err(SparseError::InvalidArgument { .. })
            ));
        }
    }

    #[test]
    fn scale_coo_preserves_old_structure_and_value_bits() {
        fn old_scale_coo(matrix: &CooMatrix, alpha: f64) -> SparseResult<CooMatrix> {
            validate_scale_factor(alpha)?;
            let data = matrix.data().iter().map(|value| value * alpha).collect();
            CooMatrix::from_triplets(
                matrix.shape(),
                data,
                matrix.row_indices().to_vec(),
                matrix.col_indices().to_vec(),
                false,
            )
        }

        let matrix = CooMatrix::from_triplets(
            Shape2D::new(4, 5),
            vec![
                1.25,
                -0.0,
                0.0,
                -3.5,
                f64::INFINITY,
                f64::from_bits(0x7ff8_0000_0000_1234),
            ],
            vec![3, 0, 3, 1, 3, 1],
            vec![4, 2, 4, 0, 1, 0],
            false,
        )
        .expect("valid unsorted duplicate COO");
        let expected = old_scale_coo(&matrix, -2.0).expect("old scale");
        let actual = scale_coo(&matrix, -2.0).expect("scale");

        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.row_indices(), expected.row_indices());
        assert_eq!(actual.col_indices(), expected.col_indices());
        assert_eq!(
            actual
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn to_coo_direct_matches_from_triplets_path_bit_for_bit() {
        fn old_csr_to_coo(matrix: &CsrMatrix) -> CooMatrix {
            let mut rows = Vec::with_capacity(matrix.nnz());
            let mut cols = Vec::with_capacity(matrix.nnz());
            let mut data = Vec::with_capacity(matrix.nnz());
            for row in 0..matrix.shape().rows {
                for idx in matrix.indptr()[row]..matrix.indptr()[row + 1] {
                    rows.push(row);
                    cols.push(matrix.indices()[idx]);
                    data.push(matrix.data()[idx]);
                }
            }
            CooMatrix::from_triplets(matrix.shape(), data, rows, cols, false).expect("old csr->coo")
        }
        fn old_csc_to_coo(matrix: &CscMatrix) -> CooMatrix {
            let mut rows = Vec::with_capacity(matrix.nnz());
            let mut cols = Vec::with_capacity(matrix.nnz());
            let mut data = Vec::with_capacity(matrix.nnz());
            for col in 0..matrix.shape().cols {
                for idx in matrix.indptr()[col]..matrix.indptr()[col + 1] {
                    rows.push(matrix.indices()[idx]);
                    cols.push(col);
                    data.push(matrix.data()[idx]);
                }
            }
            CooMatrix::from_triplets(matrix.shape(), data, rows, cols, false).expect("old csc->coo")
        }
        fn assert_coo_bit_identical(actual: &CooMatrix, expected: &CooMatrix) {
            assert_eq!(actual.shape(), expected.shape());
            assert_eq!(actual.row_indices(), expected.row_indices());
            assert_eq!(actual.col_indices(), expected.col_indices());
            assert_eq!(
                actual
                    .data()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .data()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }

        // Include an empty row/col plus signed-zero, infinity and a NaN payload so
        // any reordering or value mangling would surface.
        let csr = CsrMatrix::from_components(
            Shape2D::new(4, 5),
            vec![
                -0.0,
                f64::INFINITY,
                f64::from_bits(0x7ff8_0000_0000_1234),
                -3.5,
                0.0,
            ],
            vec![0, 4, 2, 1, 3],
            vec![0, 2, 2, 4, 5],
            false,
        )
        .expect("valid csr");
        assert_coo_bit_identical(&csr.to_coo().expect("csr to_coo"), &old_csr_to_coo(&csr));

        let csc = csr.to_csc().expect("csr->csc");
        assert_coo_bit_identical(&csc.to_coo().expect("csc to_coo"), &old_csc_to_coo(&csc));
    }

    #[test]
    fn scale_csc_preserves_structure_bits_and_metadata() {
        let canonical = CscMatrix::from_components(
            Shape2D::new(4, 3),
            vec![1.25, -0.0, 0.0, -3.5],
            vec![0, 3, 1, 2],
            vec![0, 2, 3, 4],
            false,
        )
        .expect("canonical csc");
        let noncanonical = CscMatrix::from_components(
            Shape2D::new(4, 3),
            vec![1.25, -0.0, 2.0, 0.0, -3.5],
            vec![2, 0, 2, 1, 1],
            vec![0, 3, 5, 5],
            false,
        )
        .expect("noncanonical csc");

        assert_eq!(
            canonical.canonical_meta(),
            CanonicalMeta {
                sorted_indices: true,
                deduplicated: true,
            }
        );
        assert_eq!(
            noncanonical.canonical_meta(),
            CanonicalMeta {
                sorted_indices: false,
                deduplicated: false,
            }
        );

        for matrix in [&canonical, &noncanonical] {
            for alpha in [-2.5, 0.0] {
                let scaled = scale_csc(matrix, alpha).expect("scale csc");
                assert_eq!(scaled.shape(), matrix.shape());
                assert_eq!(scaled.indices(), matrix.indices());
                assert_eq!(scaled.indptr(), matrix.indptr());
                assert_eq!(scaled.canonical_meta(), matrix.canonical_meta());
                assert!(
                    scaled
                        .data()
                        .iter()
                        .zip(matrix.data())
                        .all(|(got, value)| got.to_bits() == (value * alpha).to_bits())
                );
            }
        }
    }

    #[test]
    fn add_csr_direct_canonical_merge_preserves_sorted_rows_and_zero_elision() {
        let lhs = CooMatrix::from_triplets(
            Shape2D::new(3, 4),
            vec![1.0, 2.0, -4.0, 5.0],
            vec![0, 1, 1, 2],
            vec![1, 0, 3, 2],
            false,
        )
        .expect("lhs")
        .to_csr()
        .expect("lhs csr");
        let rhs = CooMatrix::from_triplets(
            Shape2D::new(3, 4),
            vec![3.0, 4.0, -5.0, 6.0],
            vec![0, 1, 2, 2],
            vec![2, 3, 2, 3],
            false,
        )
        .expect("rhs")
        .to_csr()
        .expect("rhs csr");

        let sum = add_csr(&lhs, &rhs).expect("sum");

        assert_eq!(sum.indptr(), &[0, 2, 3, 4]);
        assert_eq!(sum.indices(), &[1, 2, 0, 3]);
        assert_eq!(sum.data(), &[1.0, 3.0, 2.0, 6.0]);
        assert!(sum.canonical_meta().sorted_indices);
        assert!(sum.canonical_meta().deduplicated);
    }

    #[test]
    fn add_sub_csc_canonical_merge_matches_csr_reference() {
        // The CSC add/sub fast path (combine_csc_cols_directly) treats a CSC(m,n)
        // as a CSR(n,m) over its colptr/row-index arrays, so it must produce the
        // same canonical matrix as the CSR add/sub round-tripped to CSC.
        let lhs_csr = CooMatrix::from_triplets(
            Shape2D::new(3, 4),
            vec![1.0, 2.0, -4.0, 5.0],
            vec![0, 1, 1, 2],
            vec![1, 0, 3, 2],
            false,
        )
        .unwrap()
        .to_csr()
        .unwrap();
        let rhs_csr = CooMatrix::from_triplets(
            Shape2D::new(3, 4),
            vec![3.0, 4.0, -5.0, 6.0],
            vec![0, 1, 2, 2],
            vec![2, 3, 2, 3],
            false,
        )
        .unwrap()
        .to_csr()
        .unwrap();
        let lhs = lhs_csr.to_csc().unwrap();
        let rhs = rhs_csr.to_csc().unwrap();

        let add_ref = add_csr(&lhs_csr, &rhs_csr).unwrap().to_csc().unwrap();
        let add_got = add_csc(&lhs, &rhs).unwrap();
        assert_eq!(add_got.indptr(), add_ref.indptr());
        assert_eq!(add_got.indices(), add_ref.indices());
        assert_eq!(add_got.data(), add_ref.data());
        assert!(add_got.canonical_meta().sorted_indices);
        assert!(add_got.canonical_meta().deduplicated);

        let sub_ref = sub_csr(&lhs_csr, &rhs_csr).unwrap().to_csc().unwrap();
        let sub_got = sub_csc(&lhs, &rhs).unwrap();
        assert_eq!(sub_got.indptr(), sub_ref.indptr());
        assert_eq!(sub_got.indices(), sub_ref.indices());
        assert_eq!(sub_got.data(), sub_ref.data());
    }

    #[test]
    fn add_csr_mislabelled_canonical_input_keeps_validating_path() {
        let lhs = CsrMatrix::from_components(
            Shape2D::new(1, 4),
            vec![1.0, 2.0],
            vec![3, 1],
            vec![0, 2],
            true,
        )
        .expect("mislabelled lhs");
        let rhs =
            CsrMatrix::from_components(Shape2D::new(1, 4), vec![4.0], vec![2], vec![0, 1], true)
                .expect("canonical rhs");

        let sum = add_csr(&lhs, &rhs).expect("sum");

        assert_eq!(sum.indices(), &[2, 3, 1]);
        assert_eq!(sum.data(), &[4.0, 1.0, 2.0]);
        assert!(!sum.canonical_meta().sorted_indices);
        assert!(sum.canonical_meta().deduplicated);
    }

    #[test]
    fn combine_rows_parallel_matches_serial_byte_for_byte() {
        let n = 768;
        let lhs = crate::random(Shape2D::new(n, n), 0.004, 0x51A5_E01D)
            .expect("lhs coo")
            .to_csr()
            .expect("lhs csr");
        let rhs = crate::random(Shape2D::new(n, n), 0.004, 0x51A5_E01D ^ 0xABCD)
            .expect("rhs coo")
            .to_csr()
            .expect("rhs csr");

        for &scale in &[1.0_f64, -1.0] {
            let (serial_data, serial_indices, serial_indptr, serial_meta) = combine_rows_serial(
                lhs.indices(),
                lhs.data(),
                lhs.indptr(),
                rhs.indices(),
                rhs.data(),
                rhs.indptr(),
                scale,
                n,
            );
            for &threads in &[2usize, 3, 7, 64] {
                let (parallel_data, parallel_indices, parallel_indptr, parallel_meta) =
                    combine_rows_parallel(
                        lhs.indices(),
                        lhs.data(),
                        lhs.indptr(),
                        rhs.indices(),
                        rhs.data(),
                        rhs.indptr(),
                        scale,
                        n,
                        threads,
                    );
                assert_eq!(
                    parallel_data, serial_data,
                    "data mismatch scale={scale} threads={threads}"
                );
                assert_eq!(
                    parallel_indices, serial_indices,
                    "indices mismatch scale={scale} threads={threads}"
                );
                assert_eq!(
                    parallel_indptr, serial_indptr,
                    "indptr mismatch scale={scale} threads={threads}"
                );
                assert_eq!(
                    parallel_meta, serial_meta,
                    "canonical mismatch scale={scale} threads={threads}"
                );

                let (direct_data, direct_indices, direct_indptr, direct_meta) =
                    combine_rows_parallel_direct(
                        lhs.indices(),
                        lhs.data(),
                        lhs.indptr(),
                        rhs.indices(),
                        rhs.data(),
                        rhs.indptr(),
                        scale,
                        n,
                        threads,
                    );
                assert_eq!(
                    direct_data, serial_data,
                    "direct data mismatch scale={scale} threads={threads}"
                );
                assert_eq!(
                    direct_indices, serial_indices,
                    "direct indices mismatch scale={scale} threads={threads}"
                );
                assert_eq!(
                    direct_indptr, serial_indptr,
                    "direct indptr mismatch scale={scale} threads={threads}"
                );
                assert_eq!(
                    direct_meta, serial_meta,
                    "direct canonical mismatch scale={scale} threads={threads}"
                );
            }
        }
    }

    #[test]
    fn csc_direct_output_matches_gather_for_zero_cancellation_and_nonfinite_values() {
        fn assert_bits_eq(actual: &[f64], expected: &[f64]) {
            assert_eq!(actual.len(), expected.len());
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(left, right)| left.to_bits() == right.to_bits())
            );
        }

        let lhs = CscMatrix::from_components(
            Shape2D::new(8, 4),
            vec![
                -0.0,
                1.0,
                f64::INFINITY,
                2.0,
                f64::from_bits(0x7ff8_0000_0000_0042),
                -3.0,
                4.0,
                5.0,
                f64::NEG_INFINITY,
            ],
            vec![0, 2, 4, 1, 3, 2, 5, 0, 7],
            vec![0, 3, 5, 7, 9],
            false,
        )
        .expect("canonical lhs CSC");
        let rhs = CscMatrix::from_components(
            Shape2D::new(8, 4),
            vec![
                0.0,
                -1.0,
                f64::NEG_INFINITY,
                -2.0,
                6.0,
                7.0,
                -4.0,
                -0.0,
                -5.0,
                8.0,
                f64::INFINITY,
            ],
            vec![1, 2, 4, 1, 4, 3, 5, 6, 0, 2, 7],
            vec![0, 3, 5, 8, 11],
            false,
        )
        .expect("canonical rhs CSC");

        for &rhs_scale in &[1.0, -1.0] {
            let gathered = combine_rows_parallel(
                lhs.indices(),
                lhs.data(),
                lhs.indptr(),
                rhs.indices(),
                rhs.data(),
                rhs.indptr(),
                rhs_scale,
                lhs.shape().cols,
                3,
            );
            let direct = combine_rows_parallel_direct(
                lhs.indices(),
                lhs.data(),
                lhs.indptr(),
                rhs.indices(),
                rhs.data(),
                rhs.indptr(),
                rhs_scale,
                lhs.shape().cols,
                3,
            );
            assert_bits_eq(&direct.0, &gathered.0);
            assert_eq!(direct.1, gathered.1);
            assert_eq!(direct.2, gathered.2);
            assert_eq!(direct.3, gathered.3);
        }
    }

    #[test]
    fn csc_direct_output_route_requires_full_worker_residency_gate() {
        assert!(!csc_combine_uses_direct_output(
            CSC_DIRECT_OUTPUT_MIN_INPUTS_PER_WORKER * 4 - 1,
            4
        ));
        assert!(csc_combine_uses_direct_output(
            CSC_DIRECT_OUTPUT_MIN_INPUTS_PER_WORKER * 4,
            4
        ));
        assert!(!csc_combine_uses_direct_output(
            CSC_DIRECT_OUTPUT_MIN_INPUTS_PER_WORKER * 4,
            1
        ));
    }

    #[test]
    fn add_csr_golden_snapshot() {
        let random_cases = [(8usize, 0.25, 0x1234_5678_u64), (256, 0.0025, 0xfeed_cafe)];
        for (n, density, seed) in random_cases {
            let lhs = crate::random(Shape2D::new(n, n), density, seed)
                .expect("lhs coo")
                .to_csr()
                .expect("lhs csr");
            let rhs = crate::random(Shape2D::new(n, n), density, seed ^ 0x5eed_1234)
                .expect("rhs coo")
                .to_csr()
                .expect("rhs csr");
            let sum = add_csr(&lhs, &rhs).expect("random add");
            println!(
                "{}",
                snapshot_add_csr(&format!("random-{n}-{density}"), &sum)
            );
        }

        let lhs = CooMatrix::from_triplets(
            Shape2D::new(3, 4),
            vec![1.0, 2.0, -4.0, f64::from_bits(0x7ff8_0000_0000_0042)],
            vec![0, 1, 1, 2],
            vec![1, 0, 3, 2],
            false,
        )
        .expect("lhs")
        .to_csr()
        .expect("lhs csr");
        let rhs = CooMatrix::from_triplets(
            Shape2D::new(3, 4),
            vec![3.0, 4.0, -5.0, 6.0],
            vec![0, 1, 2, 2],
            vec![2, 3, 2, 3],
            false,
        )
        .expect("rhs")
        .to_csr()
        .expect("rhs csr");
        let sum = add_csr(&lhs, &rhs).expect("edge add");
        println!("{}", snapshot_add_csr("cancellation-and-nan", &sum));
    }

    #[test]
    fn canonical_direct_conversion_matches_coo_fallback_bit_exact() {
        fn assert_csc_bits(actual: &CscMatrix, expected: &CscMatrix) {
            assert_eq!(actual.shape(), expected.shape());
            assert_eq!(actual.indices(), expected.indices());
            assert_eq!(actual.indptr(), expected.indptr());
            assert_eq!(actual.canonical_meta(), expected.canonical_meta());
            assert_eq!(
                actual
                    .data()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .data()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }

        fn assert_csr_bits(actual: &CsrMatrix, expected: &CsrMatrix) {
            assert_eq!(actual.shape(), expected.shape());
            assert_eq!(actual.indices(), expected.indices());
            assert_eq!(actual.indptr(), expected.indptr());
            assert_eq!(actual.canonical_meta(), expected.canonical_meta());
            assert_eq!(
                actual
                    .data()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .data()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }

        let matrices = [
            CsrMatrix::from_components(
                Shape2D::new(4, 5),
                vec![
                    -0.0,
                    f64::INFINITY,
                    f64::from_bits(0x7ff8_0000_0000_1234),
                    -3.5,
                    0.0,
                    f64::NEG_INFINITY,
                ],
                vec![0, 4, 1, 3, 2, 4],
                vec![0, 2, 2, 4, 6],
                false,
            )
            .expect("canonical rectangular csr"),
            CsrMatrix::from_components(Shape2D::new(3, 4), vec![], vec![], vec![0; 4], true)
                .expect("empty csr"),
        ];

        for csr in matrices {
            let direct_csc = csr.to_csc().expect("direct csr->csc");
            let fallback_csc = csr.to_coo().expect("csr->coo").to_csc().expect("coo->csc");
            assert_csc_bits(&direct_csc, &fallback_csc);

            let direct_csr = direct_csc.to_csr().expect("direct csc->csr");
            let fallback_csr = direct_csc
                .to_coo()
                .expect("csc->coo")
                .to_csr()
                .expect("coo->csr");
            assert_csr_bits(&direct_csr, &fallback_csr);
        }
    }

    #[test]
    fn conversion_golden_snapshot() {
        let canonical = CooMatrix::from_triplets(
            Shape2D::new(4, 5),
            vec![1.0, -0.0, 2.5, -3.25, 8.0, 13.0],
            vec![0, 0, 1, 2, 3, 3],
            vec![1, 4, 3, 0, 2, 4],
            false,
        )
        .expect("canonical coo")
        .to_csr()
        .expect("canonical csr");
        let canonical_csc = canonical.to_csc().expect("canonical csr->csc");
        println!("{}", snapshot_csc("canonical-csr-to-csc", &canonical_csc));
        println!(
            "{}",
            snapshot_csr(
                "canonical-csr-to-csc-to-csr",
                &canonical_csc.to_csr().expect("canonical csc->csr"),
            )
        );

        let duplicate_csr = CsrMatrix::from_components(
            Shape2D::new(2, 3),
            vec![1.0, 2.0, 4.0, 5.0],
            vec![1, 1, 2, 0],
            vec![0, 3, 4],
            false,
        )
        .expect("duplicate csr");
        println!(
            "{}",
            snapshot_csc(
                "duplicate-csr-to-csc",
                &duplicate_csr.to_csc().expect("duplicate csc")
            )
        );

        let unsorted_csc = CscMatrix::from_components(
            Shape2D::new(3, 2),
            vec![7.0, -1.0, 2.0, 11.0],
            vec![2, 0, 0, 1],
            vec![0, 2, 4],
            false,
        )
        .expect("unsorted csc");
        println!(
            "{}",
            snapshot_csr(
                "unsorted-csc-to-csr",
                &unsorted_csc.to_csr().expect("unsorted csr")
            )
        );

        let empty =
            CsrMatrix::from_components(Shape2D::new(3, 4), vec![], vec![], vec![0; 4], true)
                .expect("empty csr");
        println!(
            "{}",
            snapshot_csc("empty-csr-to-csc", &empty.to_csc().expect("empty csc"))
        );
    }

    #[test]
    fn triu_default_keeps_diagonal_and_above() {
        let m = coo_full(3);
        let upper = triu(&m, 0).expect("triu");
        let d = dense(&upper);
        let expected = [[1.0, 2.0, 3.0], [0.0, 5.0, 6.0], [0.0, 0.0, 9.0]];
        for (i, row) in expected.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert!(
                    (d[i][j] - v).abs() < 1e-12,
                    "triu at ({i},{j}) = {} != {v}",
                    d[i][j]
                );
            }
        }
    }

    #[test]
    fn tril_negative_k_excludes_diagonal() {
        // tril(M, -1) keeps strictly below the diagonal.
        let m = coo_full(3);
        let lower = tril(&m, -1).expect("tril");
        let d = dense(&lower);
        let expected = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [7.0, 8.0, 0.0]];
        for (i, row) in expected.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert!(
                    (d[i][j] - v).abs() < 1e-12,
                    "tril k=-1 at ({i},{j}) = {} != {v}",
                    d[i][j]
                );
            }
        }
    }

    #[test]
    fn triu_positive_k_excludes_diagonal() {
        let m = coo_full(3);
        let upper = triu(&m, 1).expect("triu");
        let d = dense(&upper);
        let expected = [[0.0, 2.0, 3.0], [0.0, 0.0, 6.0], [0.0, 0.0, 0.0]];
        for (i, row) in expected.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert!(
                    (d[i][j] - v).abs() < 1e-12,
                    "triu k=1 at ({i},{j}) = {} != {v}",
                    d[i][j]
                );
            }
        }
    }

    #[test]
    fn tril_plus_triu_minus_diag_recovers_original() {
        // /testing-metamorphic: for any square M,
        //   tril(M, 0) + triu(M, 0) − diag(M) = M
        let n = 4;
        let m = coo_full(n);
        let lower = dense(&tril(&m, 0).expect("tril"));
        let upper = dense(&triu(&m, 0).expect("triu"));
        for i in 0..n {
            for j in 0..n {
                let diag_val = if i == j { lower[i][j] } else { 0.0 };
                let recovered = lower[i][j] + upper[i][j] - diag_val;
                let original = (i * n + j + 1) as f64;
                assert!(
                    (recovered - original).abs() < 1e-12,
                    "tril+triu-diag at ({i},{j}) = {recovered} != {original}"
                );
            }
        }
    }
}
