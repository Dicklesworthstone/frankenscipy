use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;

use crate::formats::{
    BsrMatrix, CanonicalMeta, CooMatrix, CscMatrix, CsrMatrix, DiaMatrix, DokMatrix, LilMatrix,
    Shape2D, SparseError, SparseFormat, SparseResult,
};

pub trait FormatConvertible {
    fn to_csr(&self) -> SparseResult<CsrMatrix>;
    fn to_csc(&self) -> SparseResult<CscMatrix>;
    fn to_coo(&self) -> SparseResult<CooMatrix>;
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

        let triplets = canonical_triplets(self);
        let (data, indices, indptr) = compress_triplets(self.shape(), &triplets, false);
        CsrMatrix::from_components(self.shape(), data, indices, indptr, true)
    }

    fn to_csc(&self) -> SparseResult<CscMatrix> {
        let mut triplets = canonical_triplets(self);
        triplets.sort_unstable_by_key(|(r, c, _)| (*c, *r));
        let (data, indices, indptr) = compress_triplets(self.shape(), &triplets, true);
        CscMatrix::from_components(self.shape(), data, indices, indptr, true)
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
        CooMatrix::from_triplets(self.shape(), data, rows, cols, false)
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
        CooMatrix::from_triplets(self.shape(), data, rows, cols, false)
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
    let coo = matrix.to_coo()?;
    let canonical = CooMatrix::from_triplets(
        coo.shape(),
        coo.data().to_vec(),
        coo.row_indices().to_vec(),
        coo.col_indices().to_vec(),
        true,
    )?;

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

pub fn spmv_csr(matrix: &CsrMatrix, vector: &[f64]) -> SparseResult<Vec<f64>> {
    if vector.len() != matrix.shape().cols {
        return Err(SparseError::IncompatibleShape {
            message: "spmv vector length must match matrix columns".to_string(),
        });
    }
    let mut result = vec![0.0; matrix.shape().rows];
    for (row, output) in result.iter_mut().enumerate().take(matrix.shape().rows) {
        for idx in matrix.indptr()[row]..matrix.indptr()[row + 1] {
            *output += matrix.data()[idx] * vector[matrix.indices()[idx]];
        }
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
    combine_coo(lhs.to_coo()?, rhs.to_coo()?, 1.0)?.to_csc()
}

pub fn sub_csc(lhs: &CscMatrix, rhs: &CscMatrix) -> SparseResult<CscMatrix> {
    ensure_same_shape(lhs.shape(), rhs.shape())?;
    combine_coo(lhs.to_coo()?, rhs.to_coo()?, -1.0)?.to_csc()
}

pub fn scale_csc(matrix: &CscMatrix, alpha: f64) -> SparseResult<CscMatrix> {
    let data: Vec<f64> = matrix.data().iter().map(|v| v * alpha).collect();
    let mut scaled = CscMatrix::from_components(
        matrix.shape(),
        data,
        matrix.indices().to_vec(),
        matrix.indptr().to_vec(),
        false,
    )?;
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
    let data: Vec<f64> = matrix.data().iter().map(|v| v * alpha).collect();
    CooMatrix::from_triplets(
        matrix.shape(),
        data,
        matrix.row_indices().to_vec(),
        matrix.col_indices().to_vec(),
        false,
    )
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

    CooMatrix::from_triplets(coo.shape(), data, rows, cols, false)
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
    sorted_indices
        && deduplicated
        && indices.len() == nnz
        && indptr.len() == major_len + 1
        && indptr.first() == Some(&0)
        && indptr.last() == Some(&nnz)
        && indptr.windows(2).all(|window| window[0] <= window[1])
        && compressed_segments_are_strictly_sorted(minor_len, indptr, indices)
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

fn canonical_triplets(coo: &CooMatrix) -> Vec<(usize, usize, f64)> {
    let mut triplets: Vec<(usize, usize, f64)> = coo
        .row_indices()
        .iter()
        .copied()
        .zip(coo.col_indices().iter().copied())
        .zip(coo.data().iter().copied())
        .map(|((r, c), v)| (r, c, v))
        .collect();

    triplets.sort_unstable_by_key(|(r, c, _)| (*r, *c));

    let mut dedup = Vec::with_capacity(triplets.len());
    for (r, c, v) in triplets {
        let should_merge = match dedup.last_mut() {
            Some((lr, lc, lv)) if *lr == r && *lc == c => {
                *lv += v;
                true
            }
            _ => false,
        };
        if !should_merge {
            dedup.push((r, c, v));
        }
    }

    dedup
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

fn compress_triplets(
    shape: Shape2D,
    triplets: &[(usize, usize, f64)],
    by_col: bool,
) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    if by_col {
        let mut indptr = vec![0usize; shape.cols + 1];
        for (_, col, _) in triplets {
            indptr[*col + 1] += 1;
        }
        for col in 0..shape.cols {
            indptr[col + 1] += indptr[col];
        }
        let mut data = Vec::with_capacity(triplets.len());
        let mut indices = Vec::with_capacity(triplets.len());
        for (row, _, value) in triplets {
            indices.push(*row);
            data.push(*value);
        }
        (data, indices, indptr)
    } else {
        let mut indptr = vec![0usize; shape.rows + 1];
        for (row, _, _) in triplets {
            indptr[*row + 1] += 1;
        }
        for row in 0..shape.rows {
            indptr[row + 1] += indptr[row];
        }
        let mut data = Vec::with_capacity(triplets.len());
        let mut indices = Vec::with_capacity(triplets.len());
        for (_, col, value) in triplets {
            indices.push(*col);
            data.push(*value);
        }
        (data, indices, indptr)
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
            }
        }
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
