use fsci_runtime::RuntimeMode;
use nalgebra::sparse::CsMatrix as NalgebraCsMatrix;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Deref;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

pub type SparseResult<T> = Result<T, SparseError>;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum SparseError {
    #[error("invalid shape: {message}")]
    InvalidShape { message: String },
    #[error("invalid sparse structure: {message}")]
    InvalidSparseStructure { message: String },
    #[error("index {index} out of bounds for {axis} with bound {bound}")]
    IndexOutOfBounds {
        axis: &'static str,
        index: usize,
        bound: usize,
    },
    #[error("incompatible shape: {message}")]
    IncompatibleShape { message: String },
    #[error("invalid argument: {message}")]
    InvalidArgument { message: String },
    #[error("index overflow: {message}")]
    IndexOverflow { message: String },
    #[error("unsupported operation: {feature}")]
    Unsupported { feature: String },
    #[error("non-finite input: {message}")]
    NonFiniteInput { message: String },
    #[error("singular matrix: {message}")]
    SingularMatrix { message: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    Csr,
    Csc,
    Coo,
    Bsr,
    Dia,
    Dok,
    Lil,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape2D {
    pub rows: usize,
    pub cols: usize,
}

impl Shape2D {
    #[must_use]
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    #[must_use]
    pub const fn is_square(self) -> bool {
        self.rows == self.cols
    }
}

/// A two-dimensional sparse-array container backed by an existing matrix format.
///
/// SciPy distinguishes `*_array` from the legacy `*_matrix` classes even when
/// both use the same compressed or dictionary storage. This wrapper preserves
/// that distinction without duplicating the format implementation. The wrapped
/// matrix remains available through [`Self::as_matrix`] for explicit
/// interoperability.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseArray2D<M> {
    inner: M,
}

impl<M> SparseArray2D<M> {
    #[must_use]
    pub const fn new(inner: M) -> Self {
        Self { inner }
    }

    #[must_use]
    pub const fn as_matrix(&self) -> &M {
        &self.inner
    }

    #[must_use]
    pub fn into_matrix(self) -> M {
        self.inner
    }
}

impl<M> From<M> for SparseArray2D<M> {
    fn from(inner: M) -> Self {
        Self::new(inner)
    }
}

impl<M> Deref for SparseArray2D<M> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Integer width used by SciPy-compatible sparse index arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexDtype {
    Int32,
    Int64,
}

impl IndexDtype {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Int32 => "int32",
            Self::Int64 => "int64",
        }
    }
}

/// Borrowed integer array considered by [`get_index_dtype`].
///
/// SciPy receives NumPy arrays and can inspect both their declared dtype and,
/// optionally, their contents. Rust slices carry their element type
/// statically, so this enum makes that same distinction explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexArrayRef<'a> {
    I8(&'a [i8]),
    U8(&'a [u8]),
    I16(&'a [i16]),
    U16(&'a [u16]),
    I32(&'a [i32]),
    U32(&'a [u32]),
    I64(&'a [i64]),
    U64(&'a [u64]),
    Isize(&'a [isize]),
    Usize(&'a [usize]),
}

impl IndexArrayRef<'_> {
    const fn declared_type_fits_i32(self) -> bool {
        matches!(
            self,
            Self::I8(_) | Self::U8(_) | Self::I16(_) | Self::U16(_) | Self::I32(_)
        )
    }

    fn contents_fit_i32(self) -> bool {
        match self {
            Self::I8(_) | Self::U8(_) | Self::I16(_) | Self::U16(_) | Self::I32(_) => true,
            Self::U32(values) => values.iter().all(|&value| i32::try_from(value).is_ok()),
            Self::I64(values) => values.iter().all(|&value| i32::try_from(value).is_ok()),
            Self::U64(values) => values.iter().all(|&value| i32::try_from(value).is_ok()),
            Self::Isize(values) => values.iter().all(|&value| i32::try_from(value).is_ok()),
            Self::Usize(values) => values.iter().all(|&value| i32::try_from(value).is_ok()),
        }
    }
}

/// Determine the narrowest SciPy-compatible sparse index dtype.
///
/// This mirrors `scipy.sparse.get_index_dtype`: `maxval` is considered first;
/// declared element widths are used by default, while `check_contents` permits
/// wider integer slices to select `int32` when every value actually fits.
///
/// `maxval` is unsigned because sparse shapes and indices are non-negative.
/// Values above `i64::MAX` are rejected instead of claiming that `int64` could
/// represent them.
pub fn get_index_dtype(
    arrays: &[IndexArrayRef<'_>],
    maxval: Option<u64>,
    check_contents: bool,
) -> SparseResult<IndexDtype> {
    if let Some(maxval) = maxval {
        if maxval > i64::MAX.unsigned_abs() {
            return Err(SparseError::IndexOverflow {
                message: format!("maximum sparse index {maxval} does not fit in int64"),
            });
        }
        if maxval > u64::from(i32::MAX.unsigned_abs()) {
            return Ok(IndexDtype::Int64);
        }
    }

    for &array in arrays {
        if !array.declared_type_fits_i32() && (!check_contents || !array.contents_fit_i32()) {
            return Ok(IndexDtype::Int64);
        }
    }
    Ok(IndexDtype::Int32)
}

/// Owned integer array returned by [`safely_cast_index_arrays`].
///
/// FrankenSciPy stores native sparse indices as `usize`; explicit owned arrays
/// avoid implying NumPy-style object identity or in-place dtype mutation at the
/// Rust API boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexArray {
    Int32(Vec<i32>),
    Int64(Vec<i64>),
}

impl IndexArray {
    #[must_use]
    pub const fn dtype(&self) -> IndexDtype {
        match self {
            Self::Int32(_) => IndexDtype::Int32,
            Self::Int64(_) => IndexDtype::Int64,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Int32(values) => values.len(),
            Self::Int64(values) => values.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn as_i32(&self) -> Option<&[i32]> {
        match self {
            Self::Int32(values) => Some(values),
            Self::Int64(_) => None,
        }
    }

    #[must_use]
    pub fn as_i64(&self) -> Option<&[i64]> {
        match self {
            Self::Int32(_) => None,
            Self::Int64(values) => Some(values),
        }
    }
}

/// Format-shaped result from [`safely_cast_index_arrays`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseIndexArrays {
    Compressed {
        indices: IndexArray,
        indptr: IndexArray,
    },
    Coordinates(Vec<IndexArray>),
    Offsets(IndexArray),
}

/// Borrowed sparse value accepted by [`safely_cast_index_arrays`].
#[derive(Debug, Clone, Copy)]
pub enum SparseIndexSource<'a> {
    Csr(&'a CsrMatrix),
    Csc(&'a CscMatrix),
    Coo(&'a CooMatrix),
    Bsr(&'a BsrMatrix),
    Dia(&'a DiaMatrix),
    Dok(&'a DokMatrix),
    Lil(&'a LilMatrix),
}

impl<'a> From<&'a CsrMatrix> for SparseIndexSource<'a> {
    fn from(value: &'a CsrMatrix) -> Self {
        Self::Csr(value)
    }
}

impl<'a> From<&'a CscMatrix> for SparseIndexSource<'a> {
    fn from(value: &'a CscMatrix) -> Self {
        Self::Csc(value)
    }
}

impl<'a> From<&'a CooMatrix> for SparseIndexSource<'a> {
    fn from(value: &'a CooMatrix) -> Self {
        Self::Coo(value)
    }
}

impl<'a> From<&'a BsrMatrix> for SparseIndexSource<'a> {
    fn from(value: &'a BsrMatrix) -> Self {
        Self::Bsr(value)
    }
}

impl<'a> From<&'a DiaMatrix> for SparseIndexSource<'a> {
    fn from(value: &'a DiaMatrix) -> Self {
        Self::Dia(value)
    }
}

impl<'a> From<&'a DokMatrix> for SparseIndexSource<'a> {
    fn from(value: &'a DokMatrix) -> Self {
        Self::Dok(value)
    }
}

impl<'a> From<&'a LilMatrix> for SparseIndexSource<'a> {
    fn from(value: &'a LilMatrix) -> Self {
        Self::Lil(value)
    }
}

/// Safely cast the structural arrays of a sparse value to `int32` or `int64`.
///
/// CSR, CSC, and BSR return `Compressed`; COO returns one coordinate array per
/// axis; DIA returns `Offsets`. DOK and LIL are rejected because, as in SciPy,
/// they use map/list storage rather than structural index arrays.
pub fn safely_cast_index_arrays<'a>(
    source: impl Into<SparseIndexSource<'a>>,
    dtype: IndexDtype,
    message: &str,
) -> SparseResult<SparseIndexArrays> {
    let context = if message.is_empty() {
        format!("dtype {}", dtype.label())
    } else {
        message.to_string()
    };

    match source.into() {
        SparseIndexSource::Csr(matrix) => {
            cast_compressed_indices(matrix.indices(), matrix.indptr(), dtype, &context)
        }
        SparseIndexSource::Csc(matrix) => {
            cast_compressed_indices(matrix.indices(), matrix.indptr(), dtype, &context)
        }
        SparseIndexSource::Coo(matrix) => Ok(SparseIndexArrays::Coordinates(vec![
            cast_usize_indices(matrix.row_indices(), dtype, "coords", &context)?,
            cast_usize_indices(matrix.col_indices(), dtype, "coords", &context)?,
        ])),
        SparseIndexSource::Bsr(matrix) => {
            validate_scaled_indices(
                matrix.indptr(),
                matrix.block_shape().rows,
                dtype,
                "indptr",
                &context,
            )?;
            validate_scaled_indices(
                matrix.indices(),
                matrix.block_shape().cols,
                dtype,
                "indices",
                &context,
            )?;
            cast_compressed_indices(matrix.indices(), matrix.indptr(), dtype, &context)
        }
        SparseIndexSource::Dia(matrix) => Ok(SparseIndexArrays::Offsets(cast_isize_indices(
            matrix.offsets(),
            dtype,
            "offsets",
            &context,
        )?)),
        SparseIndexSource::Dok(_) => Err(SparseError::Unsupported {
            feature: "DOK format is not associated with index arrays".to_string(),
        }),
        SparseIndexSource::Lil(_) => Err(SparseError::Unsupported {
            feature: "LIL format is not associated with index arrays".to_string(),
        }),
    }
}

fn cast_compressed_indices(
    indices: &[usize],
    indptr: &[usize],
    dtype: IndexDtype,
    context: &str,
) -> SparseResult<SparseIndexArrays> {
    Ok(SparseIndexArrays::Compressed {
        indices: cast_usize_indices(indices, dtype, "indices", context)?,
        indptr: cast_usize_indices(indptr, dtype, "indptr", context)?,
    })
}

fn cast_usize_indices(
    values: &[usize],
    dtype: IndexDtype,
    kind: &str,
    context: &str,
) -> SparseResult<IndexArray> {
    match dtype {
        IndexDtype::Int32 => values
            .iter()
            .map(|&value| i32::try_from(value).map_err(|_| index_cast_overflow(kind, context)))
            .collect::<SparseResult<Vec<_>>>()
            .map(IndexArray::Int32),
        IndexDtype::Int64 => values
            .iter()
            .map(|&value| i64::try_from(value).map_err(|_| index_cast_overflow(kind, context)))
            .collect::<SparseResult<Vec<_>>>()
            .map(IndexArray::Int64),
    }
}

fn cast_isize_indices(
    values: &[isize],
    dtype: IndexDtype,
    kind: &str,
    context: &str,
) -> SparseResult<IndexArray> {
    match dtype {
        IndexDtype::Int32 => values
            .iter()
            .map(|&value| i32::try_from(value).map_err(|_| index_cast_overflow(kind, context)))
            .collect::<SparseResult<Vec<_>>>()
            .map(IndexArray::Int32),
        IndexDtype::Int64 => values
            .iter()
            .map(|&value| i64::try_from(value).map_err(|_| index_cast_overflow(kind, context)))
            .collect::<SparseResult<Vec<_>>>()
            .map(IndexArray::Int64),
    }
}

fn validate_scaled_indices(
    values: &[usize],
    scale: usize,
    dtype: IndexDtype,
    kind: &str,
    context: &str,
) -> SparseResult<()> {
    let max_value = match dtype {
        IndexDtype::Int32 => u128::from(i32::MAX.unsigned_abs()),
        IndexDtype::Int64 => u128::from(i64::MAX.unsigned_abs()),
    };
    if values
        .iter()
        .any(|&value| (value as u128) * (scale as u128) > max_value)
    {
        return Err(index_cast_overflow(kind, context));
    }
    Ok(())
}

fn index_cast_overflow(kind: &str, context: &str) -> SparseError {
    SparseError::IndexOverflow {
        message: format!("{kind} values too large for {context}"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseSliceSpec {
    pub start: usize,
    pub end: usize,
    pub step: usize,
}

impl SparseSliceSpec {
    #[must_use]
    pub const fn new(start: usize, end: usize, step: usize) -> Self {
        Self { start, end, step }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalMeta {
    pub sorted_indices: bool,
    pub deduplicated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstructionLogEntry {
    pub timestamp: String,
    pub operation_id: String,
    pub format: SparseFormat,
    pub shape: Shape2D,
    pub nnz: usize,
    pub mode: RuntimeMode,
    pub validation_result: String,
}

impl ConstructionLogEntry {
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            "{{\"timestamp\":\"{}\",\"operation_id\":\"{}\",\"format\":\"{}\",\"shape\":[{},{}],\"nnz\":{},\"mode\":\"{}\",\"validation_result\":\"{}\"}}",
            self.timestamp,
            self.operation_id,
            format_label(self.format),
            self.shape.rows,
            self.shape.cols,
            self.nnz,
            mode_label(self.mode),
            self.validation_result
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    pub(crate) shape: Shape2D,
    pub(crate) data: Vec<f64>,
    pub(crate) indices: Vec<usize>,
    pub(crate) indptr: Vec<usize>,
    pub(crate) canonical: CanonicalMeta,
}

impl CsrMatrix {
    pub fn from_components(
        shape: Shape2D,
        data: Vec<f64>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        canonicalize: bool,
    ) -> SparseResult<Self> {
        let canonical = validate_compressed(
            "CSR",
            shape.rows,
            shape.cols,
            &data,
            &indices,
            &indptr,
            canonicalize,
        )?;
        Ok(Self {
            shape,
            data,
            indices,
            indptr,
            canonical,
        })
    }

    /// Construct a CSR matrix without running `validate_compressed`, bypassing checks.
    pub(crate) fn from_components_unchecked(
        shape: Shape2D,
        data: Vec<f64>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
    ) -> Self {
        Self {
            shape,
            data,
            indices,
            indptr,
            canonical: CanonicalMeta {
                sorted_indices: false,
                deduplicated: false,
            },
        }
    }

    /// Construct a CSR matrix the caller has already proven valid AND canonical (sorted,
    /// deduplicated), skipping both the O(nnz) `validate_compressed` bounds scan and the
    /// `detect_canonical` pass. Same result AND metadata as `from_components(.., true)` when
    /// the inputs genuinely are canonical — the caller owns that invariant.
    pub(crate) fn from_components_trusted_canonical(
        shape: Shape2D,
        data: Vec<f64>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
    ) -> Self {
        Self {
            shape,
            data,
            indices,
            indptr,
            canonical: CanonicalMeta {
                sorted_indices: true,
                deduplicated: true,
            },
        }
    }

    #[must_use]
    pub const fn shape(&self) -> Shape2D {
        self.shape
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Sparse matrix–vector product `A·x` (each output row is the dot product of
    /// that CSR row with `x`). Matches `scipy.sparse.csr_matrix.dot` on a 1-D rhs.
    pub fn matvec(&self, x: &[f64]) -> SparseResult<Vec<f64>> {
        let (rows, cols) = (self.shape.rows, self.shape.cols);
        if x.len() != cols {
            return Err(SparseError::IncompatibleShape {
                message: format!("x length {} != cols {}", x.len(), cols),
            });
        }
        let mut y = vec![0.0; rows];
        for (i, yi) in y.iter_mut().enumerate() {
            let mut sum = 0.0;
            for idx in self.indptr[i]..self.indptr[i + 1] {
                sum += self.data[idx] * x[self.indices[idx]];
            }
            *yi = sum;
        }
        Ok(y)
    }

    #[must_use]
    pub fn canonical_meta(&self) -> CanonicalMeta {
        self.canonical
    }

    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    #[must_use]
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    /// Borrow this CSR matrix as the CSC representation of its transpose.
    ///
    /// CSR storage for `A` is exactly CSC storage for `Aᵀ`: each CSR row
    /// becomes one CSC column, with the same values, compressed indices, and
    /// pointer offsets. The view therefore swaps only the shape and performs
    /// no allocation, validation pass, or data movement.
    #[must_use]
    pub fn transpose_view(&self) -> CscMatrixView<'_> {
        CscMatrixView {
            shape: Shape2D::new(self.shape.cols, self.shape.rows),
            data: &self.data,
            indices: &self.indices,
            indptr: &self.indptr,
            canonical: self.canonical,
        }
    }

    #[must_use]
    pub fn construction_log(
        &self,
        mode: RuntimeMode,
        operation_id: impl Into<String>,
        validation_result: impl Into<String>,
    ) -> ConstructionLogEntry {
        ConstructionLogEntry {
            timestamp: now_timestamp(),
            operation_id: operation_id.into(),
            format: SparseFormat::Csr,
            shape: self.shape,
            nnz: self.nnz(),
            mode,
            validation_result: validation_result.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CscMatrix {
    pub(crate) shape: Shape2D,
    pub(crate) data: Vec<f64>,
    pub(crate) indices: Vec<usize>,
    pub(crate) indptr: Vec<usize>,
    pub(crate) canonical: CanonicalMeta,
}

impl CscMatrix {
    pub fn from_components(
        shape: Shape2D,
        data: Vec<f64>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        canonicalize: bool,
    ) -> SparseResult<Self> {
        let canonical = validate_compressed(
            "CSC",
            shape.cols,
            shape.rows,
            &data,
            &indices,
            &indptr,
            canonicalize,
        )?;
        Ok(Self {
            shape,
            data,
            indices,
            indptr,
            canonical,
        })
    }

    /// Construct a CSC matrix without running `validate_compressed`, bypassing checks.
    pub(crate) fn from_components_unchecked(
        shape: Shape2D,
        data: Vec<f64>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
    ) -> Self {
        Self {
            shape,
            data,
            indices,
            indptr,
            canonical: CanonicalMeta {
                sorted_indices: false,
                deduplicated: false,
            },
        }
    }

    #[must_use]
    pub const fn shape(&self) -> Shape2D {
        self.shape
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn canonical_meta(&self) -> CanonicalMeta {
        self.canonical
    }

    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    #[must_use]
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    /// Borrow this CSC matrix as the CSR representation of its transpose.
    ///
    /// The compressed buffers already have the required layout, so creating
    /// the view is constant-time and allocation-free.
    #[must_use]
    pub fn transpose_view(&self) -> CsrMatrixView<'_> {
        CsrMatrixView {
            shape: Shape2D::new(self.shape.cols, self.shape.rows),
            data: &self.data,
            indices: &self.indices,
            indptr: &self.indptr,
            canonical: self.canonical,
        }
    }

    pub fn get(&self, row: usize, col: usize) -> SparseResult<f64> {
        validate_coordinate(self.shape, row, col)?;
        let start = self.indptr[col];
        let end = self.indptr[col + 1];
        let rows = &self.indices[start..end];
        Ok(match rows.binary_search(&row) {
            Ok(offset) => self.data[start + offset],
            Err(_) => 0.0,
        })
    }

    pub fn boolean_row_index(&self, row_mask: &[bool]) -> SparseResult<Self> {
        validate_boolean_mask(row_mask, self.shape.rows, "row")?;
        let row_map = boolean_mask_index_map(row_mask);
        let kept_rows = row_mask.iter().filter(|&&keep| keep).count();

        let mut data = Vec::with_capacity(self.data.len());
        let mut indices = Vec::with_capacity(self.indices.len());
        let mut indptr = Vec::with_capacity(self.shape.cols + 1);
        indptr.push(0);

        for col in 0..self.shape.cols {
            for idx in self.indptr[col]..self.indptr[col + 1] {
                if let Some(new_row) = row_map[self.indices[idx]] {
                    data.push(self.data[idx]);
                    indices.push(new_row);
                }
            }
            indptr.push(data.len());
        }

        Self::from_components(
            Shape2D::new(kept_rows, self.shape.cols),
            data,
            indices,
            indptr,
            false,
        )
    }

    pub fn boolean_col_index(&self, col_mask: &[bool]) -> SparseResult<Self> {
        validate_boolean_mask(col_mask, self.shape.cols, "col")?;
        let kept_cols = col_mask.iter().filter(|&&keep| keep).count();

        let mut data = Vec::with_capacity(self.data.len());
        let mut indices = Vec::with_capacity(self.indices.len());
        let mut indptr = Vec::with_capacity(kept_cols + 1);
        indptr.push(0);

        for (col, &keep) in col_mask.iter().enumerate() {
            if !keep {
                continue;
            }
            let start = self.indptr[col];
            let end = self.indptr[col + 1];
            data.extend_from_slice(&self.data[start..end]);
            indices.extend_from_slice(&self.indices[start..end]);
            indptr.push(data.len());
        }

        Self::from_components(
            Shape2D::new(self.shape.rows, kept_cols),
            data,
            indices,
            indptr,
            false,
        )
    }

    pub fn boolean_index(&self, row_mask: &[bool], col_mask: &[bool]) -> SparseResult<Self> {
        self.boolean_row_index(row_mask)?
            .boolean_col_index(col_mask)
    }

    pub fn boolean_mask_values(&self, mask: &[Vec<bool>]) -> SparseResult<Vec<f64>> {
        if mask.len() != self.shape.rows {
            return Err(SparseError::InvalidArgument {
                message: format!(
                    "bool index has shape ({}, {}) instead of ({}, {})",
                    mask.len(),
                    mask.first().map_or(0, Vec::len),
                    self.shape.rows,
                    self.shape.cols
                ),
            });
        }

        let mut values = Vec::new();
        for (row, row_mask) in mask.iter().enumerate() {
            if row_mask.len() != self.shape.cols {
                return Err(SparseError::InvalidArgument {
                    message: format!(
                        "bool index has shape ({}, {}) instead of ({}, {})",
                        mask.len(),
                        row_mask.len(),
                        self.shape.rows,
                        self.shape.cols
                    ),
                });
            }
            for (col, &keep) in row_mask.iter().enumerate() {
                if keep {
                    values.push(self.get(row, col)?);
                }
            }
        }

        Ok(values)
    }

    #[must_use]
    pub fn construction_log(
        &self,
        mode: RuntimeMode,
        operation_id: impl Into<String>,
        validation_result: impl Into<String>,
    ) -> ConstructionLogEntry {
        ConstructionLogEntry {
            timestamp: now_timestamp(),
            operation_id: operation_id.into(),
            format: SparseFormat::Csc,
            shape: self.shape,
            nnz: self.nnz(),
            mode,
            validation_result: validation_result.into(),
        }
    }
}

/// Borrowed CSR matrix storage.
///
/// This view owns no buffers. It is primarily returned by
/// [`CscMatrix::transpose_view`], where CSC storage can be reinterpreted as the
/// CSR representation of the transpose without moving any stored entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CsrMatrixView<'a> {
    shape: Shape2D,
    data: &'a [f64],
    indices: &'a [usize],
    indptr: &'a [usize],
    canonical: CanonicalMeta,
}

impl<'a> CsrMatrixView<'a> {
    #[must_use]
    pub const fn shape(self) -> Shape2D {
        self.shape
    }

    #[must_use]
    pub const fn nnz(self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub const fn canonical_meta(self) -> CanonicalMeta {
        self.canonical
    }

    #[must_use]
    pub const fn data(self) -> &'a [f64] {
        self.data
    }

    #[must_use]
    pub const fn indices(self) -> &'a [usize] {
        self.indices
    }

    #[must_use]
    pub const fn indptr(self) -> &'a [usize] {
        self.indptr
    }

    pub fn get(self, row: usize, col: usize) -> SparseResult<f64> {
        validate_coordinate(self.shape, row, col)?;
        let start = self.indptr[row];
        let end = self.indptr[row + 1];
        let columns = &self.indices[start..end];
        Ok(match columns.binary_search(&col) {
            Ok(offset) => self.data[start + offset],
            Err(_) => 0.0,
        })
    }

    /// Reinterpret this borrowed CSR storage as CSC storage for its transpose.
    #[must_use]
    pub const fn transpose_view(self) -> CscMatrixView<'a> {
        CscMatrixView {
            shape: Shape2D::new(self.shape.cols, self.shape.rows),
            data: self.data,
            indices: self.indices,
            indptr: self.indptr,
            canonical: self.canonical,
        }
    }
}

/// Borrowed CSC matrix storage.
///
/// This view owns no buffers. It is primarily returned by
/// [`CsrMatrix::transpose_view`], where CSR storage can be reinterpreted as the
/// CSC representation of the transpose without moving any stored entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CscMatrixView<'a> {
    shape: Shape2D,
    data: &'a [f64],
    indices: &'a [usize],
    indptr: &'a [usize],
    canonical: CanonicalMeta,
}

impl<'a> CscMatrixView<'a> {
    #[must_use]
    pub const fn shape(self) -> Shape2D {
        self.shape
    }

    #[must_use]
    pub const fn nnz(self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub const fn canonical_meta(self) -> CanonicalMeta {
        self.canonical
    }

    #[must_use]
    pub const fn data(self) -> &'a [f64] {
        self.data
    }

    #[must_use]
    pub const fn indices(self) -> &'a [usize] {
        self.indices
    }

    #[must_use]
    pub const fn indptr(self) -> &'a [usize] {
        self.indptr
    }

    pub fn get(self, row: usize, col: usize) -> SparseResult<f64> {
        validate_coordinate(self.shape, row, col)?;
        let start = self.indptr[col];
        let end = self.indptr[col + 1];
        let rows = &self.indices[start..end];
        Ok(match rows.binary_search(&row) {
            Ok(offset) => self.data[start + offset],
            Err(_) => 0.0,
        })
    }

    /// Reinterpret this borrowed CSC storage as CSR storage for its transpose.
    #[must_use]
    pub const fn transpose_view(self) -> CsrMatrixView<'a> {
        CsrMatrixView {
            shape: Shape2D::new(self.shape.cols, self.shape.rows),
            data: self.data,
            indices: self.indices,
            indptr: self.indptr,
            canonical: self.canonical,
        }
    }
}

/// An N-dimensional sparse array in coordinate (COO) format.
///
/// Coordinates are stored axis-major: `coords[axis][entry]` is the coordinate
/// of one stored value on that axis. Unlike [`CooMatrix`], the shape may have
/// any positive number of axes, including one-dimensional arrays and
/// zero-length dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct CooArray {
    shape: Vec<usize>,
    data: Vec<f64>,
    coords: Vec<Vec<usize>>,
    canonical: bool,
}

impl CooArray {
    pub(crate) fn from_validated_parts(
        shape: Vec<usize>,
        data: Vec<f64>,
        coords: Vec<Vec<usize>>,
        canonical: bool,
    ) -> Self {
        Self {
            shape,
            data,
            coords,
            canonical,
        }
    }

    /// Construct an N-dimensional COO array from axis-major coordinates.
    ///
    /// When `sum_duplicates` is true, equal coordinate tuples are coalesced in
    /// encounter order and emitted in lexicographic coordinate order.
    pub fn from_coords(
        shape: Vec<usize>,
        data: Vec<f64>,
        coords: Vec<Vec<usize>>,
        sum_duplicates: bool,
    ) -> SparseResult<Self> {
        if shape.is_empty() {
            return Err(SparseError::InvalidShape {
                message: "sparse arrays must have at least one axis".to_string(),
            });
        }
        if coords.len() != shape.len() {
            return Err(SparseError::IncompatibleShape {
                message: format!(
                    "COO coordinate-axis count {} does not match ndim {}",
                    coords.len(),
                    shape.len()
                ),
            });
        }
        for (axis, axis_coords) in coords.iter().enumerate() {
            if axis_coords.len() != data.len() {
                return Err(SparseError::IncompatibleShape {
                    message: format!(
                        "COO coordinate length {} on axis {axis} does not match data length {}",
                        axis_coords.len(),
                        data.len()
                    ),
                });
            }
            for &index in axis_coords {
                if index >= shape[axis] {
                    return Err(SparseError::InvalidSparseStructure {
                        message: format!(
                            "COO coordinate {index} on axis {axis} is out of bounds for extent {}",
                            shape[axis]
                        ),
                    });
                }
            }
        }

        if !sum_duplicates {
            return Ok(Self {
                shape,
                data,
                coords,
                canonical: false,
            });
        }

        let mut entries: BTreeMap<Vec<usize>, f64> = BTreeMap::new();
        for (entry, &value) in data.iter().enumerate() {
            let coordinate: Vec<usize> = coords
                .iter()
                .map(|axis_coords| axis_coords[entry])
                .collect();
            if let Some(total) = entries.get_mut(&coordinate) {
                *total += value;
            } else {
                entries.insert(coordinate, value);
            }
        }

        let mut canonical_data = Vec::with_capacity(entries.len());
        let mut canonical_coords = vec![Vec::with_capacity(entries.len()); shape.len()];
        for (coordinate, value) in entries {
            for (axis, index) in coordinate.into_iter().enumerate() {
                canonical_coords[axis].push(index);
            }
            canonical_data.push(value);
        }

        Ok(Self {
            shape,
            data: canonical_data,
            coords: canonical_coords,
            canonical: true,
        })
    }

    /// Construct an N-dimensional COO array from row-major dense values.
    pub fn from_dense(shape: Vec<usize>, dense: &[f64]) -> SparseResult<Self> {
        let dense_len = checked_dense_len(&shape)?;
        if dense.len() != dense_len {
            return Err(SparseError::IncompatibleShape {
                message: format!(
                    "dense data length {} does not match shape product {dense_len}",
                    dense.len()
                ),
            });
        }

        let mut data = Vec::new();
        let mut coords = vec![Vec::new(); shape.len()];
        for (flat_index, &value) in dense.iter().enumerate() {
            if value == 0.0 {
                continue;
            }
            let mut remainder = flat_index;
            let mut coordinate = vec![0usize; shape.len()];
            for axis in (0..shape.len()).rev() {
                coordinate[axis] = remainder % shape[axis];
                remainder /= shape[axis];
            }
            for (axis, index) in coordinate.into_iter().enumerate() {
                coords[axis].push(index);
            }
            data.push(value);
        }

        Ok(Self {
            shape,
            data,
            coords,
            canonical: true,
        })
    }

    /// Convert a legacy two-dimensional COO matrix into a sparse array.
    #[must_use]
    pub fn from_coo_matrix(matrix: &CooMatrix) -> Self {
        Self {
            shape: vec![matrix.shape.rows, matrix.shape.cols],
            data: matrix.data.clone(),
            coords: vec![matrix.row_indices.clone(), matrix.col_indices.clone()],
            canonical: false,
        }
    }

    /// Convert a two-dimensional sparse array back to the legacy COO matrix API.
    pub fn to_coo_matrix(&self) -> SparseResult<CooMatrix> {
        if self.shape.len() != 2 {
            return Err(SparseError::InvalidShape {
                message: format!(
                    "legacy sparse matrices require exactly 2 axes, got {}",
                    self.shape.len()
                ),
            });
        }
        CooMatrix::from_triplets(
            Shape2D::new(self.shape[0], self.shape[1]),
            self.data.clone(),
            self.coords[0].clone(),
            self.coords[1].clone(),
            false,
        )
    }

    /// Materialize row-major dense values, summing duplicate coordinates.
    pub fn to_dense(&self) -> SparseResult<Vec<f64>> {
        let mut dense = vec![0.0; checked_dense_len(&self.shape)?];
        for entry in 0..self.data.len() {
            let mut flat_index = 0usize;
            for (axis, &extent) in self.shape.iter().enumerate() {
                flat_index = flat_index
                    .checked_mul(extent)
                    .and_then(|prefix| prefix.checked_add(self.coords[axis][entry]))
                    .ok_or_else(|| SparseError::IndexOverflow {
                        message: "COO dense offset overflows usize".to_string(),
                    })?;
            }
            dense[flat_index] += self.data[entry];
        }
        Ok(dense)
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    #[must_use]
    pub fn coords(&self) -> &[Vec<usize>] {
        &self.coords
    }

    #[must_use]
    pub const fn has_canonical_format(&self) -> bool {
        self.canonical
    }
}

fn checked_dense_len(shape: &[usize]) -> SparseResult<usize> {
    if shape.is_empty() {
        return Err(SparseError::InvalidShape {
            message: "sparse arrays must have at least one axis".to_string(),
        });
    }
    shape.iter().try_fold(1usize, |size, &extent| {
        size.checked_mul(extent)
            .ok_or_else(|| SparseError::IndexOverflow {
                message: "sparse array shape product overflows usize".to_string(),
            })
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct CooMatrix {
    pub(crate) shape: Shape2D,
    pub(crate) data: Vec<f64>,
    pub(crate) row_indices: Vec<usize>,
    pub(crate) col_indices: Vec<usize>,
}

#[doc(hidden)]
pub static COO_SUM_DUPLICATES_RADIX_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

const COO_SUM_DUPLICATES_RADIX_MIN_NNZ: usize = 65_536;
const COO_SUM_DUPLICATES_RADIX_MIN_AVG_ROW_NNZ: usize = 64;

#[derive(Clone, Copy)]
struct PackedCooEntry {
    key: u64,
    value: f64,
}

impl CooMatrix {
    pub fn from_triplets(
        shape: Shape2D,
        data: Vec<f64>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        sum_duplicates: bool,
    ) -> SparseResult<Self> {
        if data.len() != row_indices.len() || data.len() != col_indices.len() {
            return Err(SparseError::IncompatibleShape {
                message: "COO data/row/col lengths must match".to_string(),
            });
        }
        for &row in &row_indices {
            if row >= shape.rows {
                return Err(SparseError::IndexOutOfBounds {
                    axis: "row",
                    index: row,
                    bound: shape.rows,
                });
            }
        }
        for &col in &col_indices {
            if col >= shape.cols {
                return Err(SparseError::IndexOutOfBounds {
                    axis: "col",
                    index: col,
                    bound: shape.cols,
                });
            }
        }

        if !sum_duplicates {
            return Ok(Self {
                shape,
                data,
                row_indices,
                col_indices,
            });
        }

        let nnz = data.len();
        if !COO_SUM_DUPLICATES_RADIX_DISABLE.load(std::sync::atomic::Ordering::Relaxed)
            && let Some(coalesced) =
                coo_sum_duplicates_radix(shape, &data, &row_indices, &col_indices)
        {
            return Ok(coalesced);
        }

        // Counting sort by row (O(nnz + rows)) beats the global O(nnz·log nnz)
        // `sort_unstable_by_key((row,col))` — SciPy's `coo.sum_duplicates()` uses a
        // (slow) `np.lexsort`, and this path feeds `sparse.random` + graph builds.
        // GATE: only when rows are bounded relative to nnz, since `sparse.random`
        // supports enormous shapes with ~0 nnz (e.g. 1e9×1e9) where the O(rows)
        // histogram would blow memory — those keep the comparison sort (nnz tiny).
        if nnz >= 4096 && shape.rows <= nnz.saturating_mul(8).max(1 << 16) {
            let nrow = shape.rows;
            let mut row_ptr = vec![0usize; nrow + 1];
            for &r in &row_indices {
                row_ptr[r + 1] += 1;
            }
            for r in 0..nrow {
                row_ptr[r + 1] += row_ptr[r];
            }
            let mut pairs: Vec<(usize, f64)> = vec![(0usize, 0.0f64); nnz];
            let mut next = row_ptr[..nrow].to_vec();
            for i in 0..nnz {
                let r = row_indices[i];
                let p = next[r];
                pairs[p] = (col_indices[i], data[i]);
                next[r] = p + 1;
            }
            let mut merged_rows = Vec::with_capacity(nnz);
            let mut merged_cols = Vec::with_capacity(nnz);
            let mut merged_data = Vec::with_capacity(nnz);
            for r in 0..nrow {
                let seg = &mut pairs[row_ptr[r]..row_ptr[r + 1]];
                // Stable so duplicate sums land in per-row encounter order (matches
                // the CSR/CSC counting-sort paths); short rows use insertion sort.
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
                    merged_rows.push(r);
                    merged_cols.push(c);
                    merged_data.push(v);
                    j = k;
                }
            }
            return Ok(Self {
                shape,
                data: merged_data,
                row_indices: merged_rows,
                col_indices: merged_cols,
            });
        }

        let mut triplets: Vec<(usize, usize, f64)> = row_indices
            .into_iter()
            .zip(col_indices)
            .zip(data)
            .map(|((r, c), v)| (r, c, v))
            .collect();
        triplets.sort_unstable_by_key(|(r, c, _)| (*r, *c));

        let mut merged_rows = Vec::with_capacity(triplets.len());
        let mut merged_cols = Vec::with_capacity(triplets.len());
        let mut merged_data = Vec::with_capacity(triplets.len());

        for (row, col, value) in triplets {
            let is_same_as_last = match (merged_rows.last(), merged_cols.last()) {
                (Some(&last_row), Some(&last_col)) => last_row == row && last_col == col,
                _ => false,
            };

            if is_same_as_last {
                if let Some(last) = merged_data.last_mut() {
                    *last += value;
                }
            } else {
                merged_rows.push(row);
                merged_cols.push(col);
                merged_data.push(value);
            }
        }

        Ok(Self {
            shape,
            data: merged_data,
            row_indices: merged_rows,
            col_indices: merged_cols,
        })
    }

    #[must_use]
    pub const fn shape(&self) -> Shape2D {
        self.shape
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    #[must_use]
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    #[must_use]
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    pub fn setdiag(&mut self, values: &[f64], k: isize) -> SparseResult<()> {
        validate_setdiag_offset(self.shape, k)?;
        if values.is_empty() {
            return Ok(());
        }

        let count = diagonal_len_for_setdiag(self.shape, k).min(values.len());
        if count == 0 {
            return Ok(());
        }

        let coordinates = diagonal_coordinates_for_setdiag(self.shape, k, count);
        self.replace_coordinates(&coordinates, values.iter().copied());
        Ok(())
    }

    pub fn setdiag_scalar(&mut self, value: f64, k: isize) -> SparseResult<()> {
        validate_setdiag_offset(self.shape, k)?;
        let count = diagonal_len_for_setdiag(self.shape, k);
        if count == 0 {
            return Ok(());
        }

        let coordinates = diagonal_coordinates_for_setdiag(self.shape, k, count);
        self.replace_coordinates(&coordinates, std::iter::repeat_n(value, count));
        Ok(())
    }

    #[must_use]
    pub fn construction_log(
        &self,
        mode: RuntimeMode,
        operation_id: impl Into<String>,
        validation_result: impl Into<String>,
    ) -> ConstructionLogEntry {
        ConstructionLogEntry {
            timestamp: now_timestamp(),
            operation_id: operation_id.into(),
            format: SparseFormat::Coo,
            shape: self.shape,
            nnz: self.nnz(),
            mode,
            validation_result: validation_result.into(),
        }
    }

    fn replace_coordinates<I>(&mut self, coordinates: &[(usize, usize)], values: I)
    where
        I: IntoIterator<Item = f64>,
    {
        let targets: BTreeSet<(usize, usize)> = coordinates.iter().copied().collect();
        let mut kept_rows = Vec::with_capacity(self.row_indices.len());
        let mut kept_cols = Vec::with_capacity(self.col_indices.len());
        let mut kept_data = Vec::with_capacity(self.data.len());

        for ((&row, &col), &value) in self
            .row_indices
            .iter()
            .zip(self.col_indices.iter())
            .zip(self.data.iter())
        {
            if targets.contains(&(row, col)) {
                continue;
            }
            kept_rows.push(row);
            kept_cols.push(col);
            kept_data.push(value);
        }

        for ((row, col), value) in coordinates.iter().copied().zip(values) {
            kept_rows.push(row);
            kept_cols.push(col);
            kept_data.push(value);
        }

        self.row_indices = kept_rows;
        self.col_indices = kept_cols;
        self.data = kept_data;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BsrMatrix {
    pub(crate) shape: Shape2D,
    pub(crate) block_shape: Shape2D,
    pub(crate) data: Vec<Vec<f64>>,
    pub(crate) indices: Vec<usize>,
    pub(crate) indptr: Vec<usize>,
    pub(crate) canonical: CanonicalMeta,
}

impl BsrMatrix {
    pub fn from_components(
        shape: Shape2D,
        block_shape: Shape2D,
        data: Vec<Vec<f64>>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        canonicalize: bool,
    ) -> SparseResult<Self> {
        let canonical =
            validate_bsr_components(shape, block_shape, &data, &indices, &indptr, canonicalize)?;
        Ok(Self {
            shape,
            block_shape,
            data,
            indices,
            indptr,
            canonical,
        })
    }

    pub fn from_triplets(
        shape: Shape2D,
        block_shape: Shape2D,
        data: Vec<f64>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
    ) -> SparseResult<Self> {
        if data.len() != row_indices.len() || data.len() != col_indices.len() {
            return Err(SparseError::IncompatibleShape {
                message: "BSR data/row/col lengths must match".to_string(),
            });
        }

        let (block_rows, _, block_area) = validate_bsr_block_shape(shape, block_shape)?;
        let mut block_row_maps = vec![BTreeMap::new(); block_rows];

        for ((row, col), value) in row_indices.into_iter().zip(col_indices).zip(data) {
            validate_coordinate(shape, row, col)?;
            let block_row = row / block_shape.rows;
            let block_col = col / block_shape.cols;
            let offset = (row % block_shape.rows) * block_shape.cols + (col % block_shape.cols);
            let block = block_row_maps[block_row]
                .entry(block_col)
                .or_insert_with(|| vec![0.0; block_area]);
            block[offset] += value;
        }

        let mut data_blocks = Vec::new();
        let mut block_indices = Vec::new();
        let mut indptr = Vec::with_capacity(block_rows + 1);
        indptr.push(0);

        for row_map in block_row_maps {
            for (block_col, block) in row_map {
                block_indices.push(block_col);
                data_blocks.push(block);
            }
            indptr.push(block_indices.len());
        }

        Self::from_components(shape, block_shape, data_blocks, block_indices, indptr, true)
    }

    #[must_use]
    pub const fn shape(&self) -> Shape2D {
        self.shape
    }

    #[must_use]
    pub const fn block_shape(&self) -> Shape2D {
        self.block_shape
    }

    #[must_use]
    pub fn nnz_blocks(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        self.nnz_blocks()
            .saturating_mul(self.block_shape.rows.saturating_mul(self.block_shape.cols))
    }

    #[must_use]
    pub fn canonical_meta(&self) -> CanonicalMeta {
        self.canonical
    }

    #[must_use]
    pub fn data(&self) -> &[Vec<f64>] {
        &self.data
    }

    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    #[must_use]
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    pub fn get(&self, row: usize, col: usize) -> SparseResult<f64> {
        validate_coordinate(self.shape, row, col)?;
        let block_row = row / self.block_shape.rows;
        let block_col = col / self.block_shape.cols;
        let offset =
            (row % self.block_shape.rows) * self.block_shape.cols + (col % self.block_shape.cols);

        let mut value = 0.0;
        for idx in self.indptr[block_row]..self.indptr[block_row + 1] {
            if self.indices[idx] == block_col {
                value += self.data[idx][offset];
            }
        }
        Ok(value)
    }

    #[must_use]
    pub fn construction_log(
        &self,
        mode: RuntimeMode,
        operation_id: impl Into<String>,
        validation_result: impl Into<String>,
    ) -> ConstructionLogEntry {
        ConstructionLogEntry {
            timestamp: now_timestamp(),
            operation_id: operation_id.into(),
            format: SparseFormat::Bsr,
            shape: self.shape,
            nnz: self.nnz(),
            mode,
            validation_result: validation_result.into(),
        }
    }

    pub fn to_coo(&self) -> SparseResult<CooMatrix> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut data = Vec::with_capacity(self.nnz());

        for block_row in 0..self.indptr.len().saturating_sub(1) {
            for idx in self.indptr[block_row]..self.indptr[block_row + 1] {
                let block_col = self.indices[idx];
                let block = &self.data[idx];

                for local_row in 0..self.block_shape.rows {
                    for local_col in 0..self.block_shape.cols {
                        rows.push(block_row * self.block_shape.rows + local_row);
                        cols.push(block_col * self.block_shape.cols + local_col);
                        data.push(block[local_row * self.block_shape.cols + local_col]);
                    }
                }
            }
        }

        CooMatrix::from_triplets(self.shape, data, rows, cols, false)
    }

    pub fn to_csr(&self) -> SparseResult<CsrMatrix> {
        use crate::ops::FormatConvertible;
        self.to_coo()?.to_csr()
    }

    pub fn to_csc(&self) -> SparseResult<CscMatrix> {
        use crate::ops::FormatConvertible;
        self.to_coo()?.to_csc()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LilMatrix {
    pub(crate) shape: Shape2D,
    pub(crate) row_indices: Vec<Vec<usize>>,
    pub(crate) row_data: Vec<Vec<f64>>,
}

impl LilMatrix {
    #[must_use]
    pub fn new(shape: Shape2D) -> Self {
        Self {
            shape,
            row_indices: vec![Vec::new(); shape.rows],
            row_data: vec![Vec::new(); shape.rows],
        }
    }

    pub fn from_rows(
        shape: Shape2D,
        row_indices: Vec<Vec<usize>>,
        row_data: Vec<Vec<f64>>,
    ) -> SparseResult<Self> {
        if row_indices.len() != shape.rows || row_data.len() != shape.rows {
            return Err(SparseError::InvalidShape {
                message: "LIL outer row containers must match shape.rows".to_string(),
            });
        }

        let mut canonical_indices = Vec::with_capacity(shape.rows);
        let mut canonical_data = Vec::with_capacity(shape.rows);

        for (row_cols, row_values) in row_indices.into_iter().zip(row_data) {
            let (cols, values) = canonicalize_lil_row(row_cols, row_values, shape.cols)?;
            canonical_indices.push(cols);
            canonical_data.push(values);
        }

        Ok(Self {
            shape,
            row_indices: canonical_indices,
            row_data: canonical_data,
        })
    }

    pub fn from_triplets(
        shape: Shape2D,
        data: Vec<f64>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
    ) -> SparseResult<Self> {
        if data.len() != row_indices.len() || data.len() != col_indices.len() {
            return Err(SparseError::IncompatibleShape {
                message: "LIL data/row/col lengths must match".to_string(),
            });
        }

        let mut row_maps = vec![BTreeMap::new(); shape.rows];
        for ((row, col), value) in row_indices.into_iter().zip(col_indices).zip(data) {
            validate_coordinate(shape, row, col)?;
            *row_maps[row].entry(col).or_insert(0.0) += value;
        }

        let mut canonical_indices = Vec::with_capacity(shape.rows);
        let mut canonical_data = Vec::with_capacity(shape.rows);
        for row_map in row_maps {
            let mut cols = Vec::with_capacity(row_map.len());
            let mut values = Vec::with_capacity(row_map.len());
            for (col, value) in row_map {
                cols.push(col);
                values.push(value);
            }
            canonical_indices.push(cols);
            canonical_data.push(values);
        }

        Ok(Self {
            shape,
            row_indices: canonical_indices,
            row_data: canonical_data,
        })
    }

    #[must_use]
    pub const fn shape(&self) -> Shape2D {
        self.shape
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        self.row_data.iter().map(Vec::len).sum()
    }

    #[must_use]
    pub fn row_indices(&self) -> &[Vec<usize>] {
        &self.row_indices
    }

    #[must_use]
    pub fn row_data(&self) -> &[Vec<f64>] {
        &self.row_data
    }

    pub fn get(&self, row: usize, col: usize) -> SparseResult<f64> {
        validate_coordinate(self.shape, row, col)?;
        let row_cols = &self.row_indices[row];
        Ok(match row_cols.binary_search(&col) {
            Ok(idx) => self.row_data[row][idx],
            Err(_) => 0.0,
        })
    }

    pub fn contains(&self, row: usize, col: usize) -> SparseResult<bool> {
        validate_coordinate(self.shape, row, col)?;
        Ok(self.row_indices[row].binary_search(&col).is_ok())
    }

    pub fn insert(&mut self, row: usize, col: usize, value: f64) -> SparseResult<Option<f64>> {
        validate_coordinate(self.shape, row, col)?;
        match self.row_indices[row].binary_search(&col) {
            Ok(idx) => {
                let previous = self.row_data[row][idx];
                self.row_data[row][idx] = value;
                Ok(Some(previous))
            }
            Err(idx) => {
                self.row_indices[row].insert(idx, col);
                self.row_data[row].insert(idx, value);
                Ok(None)
            }
        }
    }

    pub fn remove(&mut self, row: usize, col: usize) -> SparseResult<Option<f64>> {
        validate_coordinate(self.shape, row, col)?;
        Ok(match self.row_indices[row].binary_search(&col) {
            Ok(idx) => {
                self.row_indices[row].remove(idx);
                Some(self.row_data[row].remove(idx))
            }
            Err(_) => None,
        })
    }

    #[must_use]
    pub fn construction_log(
        &self,
        mode: RuntimeMode,
        operation_id: impl Into<String>,
        validation_result: impl Into<String>,
    ) -> ConstructionLogEntry {
        ConstructionLogEntry {
            timestamp: now_timestamp(),
            operation_id: operation_id.into(),
            format: SparseFormat::Lil,
            shape: self.shape,
            nnz: self.nnz(),
            mode,
            validation_result: validation_result.into(),
        }
    }

    pub fn to_coo(&self) -> SparseResult<CooMatrix> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut data = Vec::with_capacity(self.nnz());

        for (row, (row_cols, row_values)) in self
            .row_indices
            .iter()
            .zip(self.row_data.iter())
            .enumerate()
        {
            for (&col, &value) in row_cols.iter().zip(row_values.iter()) {
                rows.push(row);
                cols.push(col);
                data.push(value);
            }
        }

        CooMatrix::from_triplets(self.shape, data, rows, cols, false)
    }

    pub fn to_csr(&self) -> SparseResult<CsrMatrix> {
        let indptr_capacity =
            self.shape
                .rows
                .checked_add(1)
                .ok_or_else(|| SparseError::IndexOverflow {
                    message: "LIL-to-CSR row-pointer length exceeds usize".to_string(),
                })?;
        let mut indptr = Vec::with_capacity(indptr_capacity);
        indptr.push(0);

        let mut nnz = 0usize;
        for row in &self.row_indices {
            nnz = nnz
                .checked_add(row.len())
                .ok_or_else(|| SparseError::IndexOverflow {
                    message: "LIL-to-CSR row offsets exceed usize".to_string(),
                })?;
            indptr.push(nnz);
        }

        let mut indices = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);
        for (row_indices, row_data) in self.row_indices.iter().zip(&self.row_data) {
            indices.extend_from_slice(row_indices);
            data.extend_from_slice(row_data);
        }

        Ok(CsrMatrix::from_components_trusted_canonical(
            self.shape, data, indices, indptr,
        ))
    }

    pub fn to_csc(&self) -> SparseResult<CscMatrix> {
        use crate::ops::FormatConvertible;
        self.to_coo()?.to_csc()
    }

    pub fn slice(
        &self,
        row_slice: SparseSliceSpec,
        col_slice: SparseSliceSpec,
    ) -> SparseResult<Self> {
        validate_slice_spec(row_slice, self.shape.rows, "row")?;
        validate_slice_spec(col_slice, self.shape.cols, "col")?;

        let mut sliced_row_indices = Vec::with_capacity(slice_len(row_slice));
        let mut sliced_row_data = Vec::with_capacity(slice_len(row_slice));

        for row in (row_slice.start..row_slice.end).step_by(row_slice.step) {
            let mut cols = Vec::new();
            let mut values = Vec::new();
            for (&col, &value) in self.row_indices[row].iter().zip(self.row_data[row].iter()) {
                if col < col_slice.start || col >= col_slice.end {
                    continue;
                }
                let offset = col - col_slice.start;
                if offset.is_multiple_of(col_slice.step) {
                    cols.push(offset / col_slice.step);
                    values.push(value);
                }
            }
            sliced_row_indices.push(cols);
            sliced_row_data.push(values);
        }

        Ok(Self {
            shape: Shape2D::new(slice_len(row_slice), slice_len(col_slice)),
            row_indices: sliced_row_indices,
            row_data: sliced_row_data,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DokMatrix {
    pub(crate) shape: Shape2D,
    pub(crate) entries: BTreeMap<(usize, usize), f64>,
}

impl DokMatrix {
    #[must_use]
    pub fn new(shape: Shape2D) -> Self {
        Self {
            shape,
            entries: BTreeMap::new(),
        }
    }

    pub fn from_triplets(
        shape: Shape2D,
        data: Vec<f64>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
    ) -> SparseResult<Self> {
        if data.len() != row_indices.len() || data.len() != col_indices.len() {
            return Err(SparseError::IncompatibleShape {
                message: "DOK data/row/col lengths must match".to_string(),
            });
        }

        let mut entries = BTreeMap::new();
        for ((row, col), value) in row_indices.into_iter().zip(col_indices).zip(data) {
            validate_coordinate(shape, row, col)?;
            let key = (row, col);
            let updated = entries.get(&key).copied().unwrap_or(0.0) + value;
            if updated == 0.0 {
                entries.remove(&key);
            } else {
                entries.insert(key, updated);
            }
        }

        Ok(Self { shape, entries })
    }

    #[must_use]
    pub const fn shape(&self) -> Shape2D {
        self.shape
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn entries(&self) -> &BTreeMap<(usize, usize), f64> {
        &self.entries
    }

    pub fn get(&self, row: usize, col: usize) -> SparseResult<f64> {
        validate_coordinate(self.shape, row, col)?;
        Ok(self.entries.get(&(row, col)).copied().unwrap_or(0.0))
    }

    pub fn contains(&self, row: usize, col: usize) -> SparseResult<bool> {
        validate_coordinate(self.shape, row, col)?;
        Ok(self.entries.contains_key(&(row, col)))
    }

    pub fn insert(&mut self, row: usize, col: usize, value: f64) -> SparseResult<Option<f64>> {
        validate_coordinate(self.shape, row, col)?;
        let key = (row, col);
        let previous = self.entries.get(&key).copied();
        if value == 0.0 {
            self.entries.remove(&key);
        } else {
            self.entries.insert(key, value);
        }
        Ok(previous)
    }

    pub fn remove(&mut self, row: usize, col: usize) -> SparseResult<Option<f64>> {
        validate_coordinate(self.shape, row, col)?;
        Ok(self.entries.remove(&(row, col)))
    }

    #[must_use]
    pub fn construction_log(
        &self,
        mode: RuntimeMode,
        operation_id: impl Into<String>,
        validation_result: impl Into<String>,
    ) -> ConstructionLogEntry {
        ConstructionLogEntry {
            timestamp: now_timestamp(),
            operation_id: operation_id.into(),
            format: SparseFormat::Dok,
            shape: self.shape,
            nnz: self.nnz(),
            mode,
            validation_result: validation_result.into(),
        }
    }

    pub fn to_coo(&self) -> SparseResult<CooMatrix> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut data = Vec::with_capacity(self.nnz());

        for (&(row, col), &value) in &self.entries {
            rows.push(row);
            cols.push(col);
            data.push(value);
        }

        CooMatrix::from_triplets(self.shape, data, rows, cols, false)
    }

    pub fn to_csr(&self) -> SparseResult<CsrMatrix> {
        use crate::ops::FormatConvertible;
        self.to_coo()?.to_csr()
    }

    pub fn to_csc(&self) -> SparseResult<CscMatrix> {
        use crate::ops::FormatConvertible;
        self.to_coo()?.to_csc()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiaMatrix {
    pub(crate) shape: Shape2D,
    pub(crate) offsets: Vec<isize>,
    pub(crate) data: Vec<Vec<f64>>,
}

impl DiaMatrix {
    pub fn from_triplets(
        shape: Shape2D,
        data: Vec<f64>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
    ) -> SparseResult<Self> {
        if data.len() != row_indices.len() || data.len() != col_indices.len() {
            return Err(SparseError::IncompatibleShape {
                message: "DIA data/row/col lengths must match".to_string(),
            });
        }

        let mut diagonals = BTreeMap::<isize, Vec<f64>>::new();
        for ((row, col), value) in row_indices.into_iter().zip(col_indices).zip(data) {
            validate_coordinate(shape, row, col)?;
            let offset = col as isize - row as isize;
            let diagonal = diagonals
                .entry(offset)
                .or_insert_with(|| vec![0.0; diagonal_len(shape, offset)]);
            let diagonal_index = if offset >= 0 { row } else { col };
            diagonal[diagonal_index] += value;
        }

        let (offsets, data): (Vec<_>, Vec<_>) = diagonals.into_iter().unzip();
        Self::from_diagonals(shape, offsets, data)
    }

    pub fn from_diagonals(
        shape: Shape2D,
        offsets: Vec<isize>,
        data: Vec<Vec<f64>>,
    ) -> SparseResult<Self> {
        if offsets.len() != data.len() {
            return Err(SparseError::IncompatibleShape {
                message: "DIA offsets and data lengths must match".to_string(),
            });
        }

        let mut seen = BTreeSet::new();
        for (&offset, _diagonal) in offsets.iter().zip(data.iter()) {
            if !seen.insert(offset) {
                return Err(SparseError::InvalidArgument {
                    message: "DIA offsets must be unique".to_string(),
                });
            }

            let expected_len = diagonal_len(shape, offset);
            if expected_len == 0 {
                return Err(SparseError::InvalidArgument {
                    message: format!("DIA offset {offset} is out of bounds for matrix shape"),
                });
            }
            if _diagonal.len() != expected_len {
                return Err(SparseError::IncompatibleShape {
                    message: format!(
                        "DIA diagonal for offset {offset} has length {}, expected {expected_len}",
                        _diagonal.len()
                    ),
                });
            }
        }

        Ok(Self {
            shape,
            offsets,
            data,
        })
    }

    #[must_use]
    pub const fn shape(&self) -> Shape2D {
        self.shape
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        let mut count = 0;
        for diag in &self.data {
            for &val in diag {
                if val != 0.0 {
                    count += 1;
                }
            }
        }
        count
    }

    #[must_use]
    pub fn offsets(&self) -> &[isize] {
        &self.offsets
    }

    #[must_use]
    pub fn data(&self) -> &[Vec<f64>] {
        &self.data
    }

    #[must_use]
    pub fn construction_log(
        &self,
        mode: RuntimeMode,
        operation_id: impl Into<String>,
        validation_result: impl Into<String>,
    ) -> ConstructionLogEntry {
        ConstructionLogEntry {
            timestamp: now_timestamp(),
            operation_id: operation_id.into(),
            format: SparseFormat::Dia,
            shape: self.shape,
            nnz: self.nnz(),
            mode,
            validation_result: validation_result.into(),
        }
    }

    pub fn new(shape: Shape2D, data: Vec<Vec<f64>>, offsets: Vec<isize>) -> SparseResult<Self> {
        if data.len() != offsets.len() {
            return Err(SparseError::InvalidArgument {
                message: "data and offsets must have same length".to_string(),
            });
        }
        let mut exact_data = Vec::with_capacity(data.len());
        for (k, &off) in offsets.iter().enumerate() {
            let expected_len = diagonal_len(shape, off);
            let mut diag = Vec::with_capacity(expected_len);
            let mut r = if off < 0 { (-off) as usize } else { 0 };
            let mut c = if off > 0 { off as usize } else { 0 };
            while r < shape.rows && c < shape.cols {
                let val = if c < data[k].len() { data[k][c] } else { 0.0 };
                diag.push(val);
                r += 1;
                c += 1;
            }
            exact_data.push(diag);
        }
        Self::from_diagonals(shape, offsets, exact_data)
    }

    pub fn num_diags(&self) -> usize {
        self.offsets.len()
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        let offset_needed = col as isize - row as isize;
        for (k, &off) in self.offsets.iter().enumerate() {
            if off == offset_needed {
                let idx = if off >= 0 { row } else { col };
                if idx < self.data[k].len() {
                    return self.data[k][idx];
                }
            }
        }
        0.0
    }

    pub fn matvec(&self, x: &[f64]) -> SparseResult<Vec<f64>> {
        let (rows, cols) = (self.shape.rows, self.shape.cols);
        if x.len() != cols {
            return Err(SparseError::IncompatibleShape {
                message: format!("x length {} != cols {}", x.len(), cols),
            });
        }

        let mut y = vec![0.0; rows];
        for (k, &off) in self.offsets.iter().enumerate() {
            let diag = &self.data[k];
            for (i, yi) in y.iter_mut().enumerate() {
                let j = i as isize + off;
                if j >= 0 && (j as usize) < cols {
                    let j = j as usize;
                    let idx = if off >= 0 { i } else { j };
                    if idx < diag.len() {
                        *yi += diag[idx] * x[j];
                    }
                }
            }
        }
        Ok(y)
    }

    pub fn to_coo(&self) -> SparseResult<CooMatrix> {
        let (rows, cols) = (self.shape.rows, self.shape.cols);
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut data = Vec::new();

        for (k, &off) in self.offsets.iter().enumerate() {
            let diag = &self.data[k];
            for i in 0..rows {
                let j = i as isize + off;
                if j >= 0 && (j as usize) < cols {
                    let j = j as usize;
                    let idx = if off >= 0 { i } else { j };
                    if idx < diag.len() {
                        let val = diag[idx];
                        if val != 0.0 {
                            row_indices.push(i);
                            col_indices.push(j);
                            data.push(val);
                        }
                    }
                }
            }
        }

        CooMatrix::from_triplets(self.shape, data, row_indices, col_indices, true)
    }

    pub fn to_csr(&self) -> SparseResult<CsrMatrix> {
        use crate::ops::FormatConvertible;
        self.to_coo()?.to_csr()
    }

    pub fn to_csc(&self) -> SparseResult<CscMatrix> {
        use crate::ops::FormatConvertible;
        self.to_coo()?.to_csc()
    }
}

fn coo_sum_duplicates_radix(
    shape: Shape2D,
    data: &[f64],
    row_indices: &[usize],
    col_indices: &[usize],
) -> Option<CooMatrix> {
    let nnz = data.len();
    if nnz < COO_SUM_DUPLICATES_RADIX_MIN_NNZ
        || shape.cols == 0
        || shape
            .rows
            .saturating_mul(COO_SUM_DUPLICATES_RADIX_MIN_AVG_ROW_NNZ)
            > nnz
    {
        return None;
    }

    let rows = u64::try_from(shape.rows).ok()?;
    let cols = u64::try_from(shape.cols).ok()?;
    rows.checked_mul(cols)?;

    let mut entries = Vec::with_capacity(nnz);
    let mut max_key = 0_u64;
    for ((&row_index, &col_index), &value) in row_indices.iter().zip(col_indices).zip(data) {
        let row = u64::try_from(row_index).ok()?;
        let col = u64::try_from(col_index).ok()?;
        let key = row.checked_mul(cols)?.checked_add(col)?;
        max_key = max_key.max(key);
        entries.push(PackedCooEntry { key, value });
    }

    stable_radix_sort_packed_coo(&mut entries, max_key);

    let mut merged_rows = Vec::with_capacity(nnz);
    let mut merged_cols = Vec::with_capacity(nnz);
    let mut merged_data = Vec::with_capacity(nnz);
    let mut i = 0;
    while i < entries.len() {
        let key = entries[i].key;
        let mut value = entries[i].value;
        let mut j = i + 1;
        while j < entries.len() && entries[j].key == key {
            value += entries[j].value;
            j += 1;
        }
        merged_rows.push(usize::try_from(key / cols).ok()?);
        merged_cols.push(usize::try_from(key % cols).ok()?);
        merged_data.push(value);
        i = j;
    }

    Some(CooMatrix {
        shape,
        data: merged_data,
        row_indices: merged_rows,
        col_indices: merged_cols,
    })
}

fn stable_radix_sort_packed_coo(entries: &mut Vec<PackedCooEntry>, max_key: u64) {
    let bit_width = 64usize - max_key.leading_zeros() as usize;
    let pass_count = bit_width.max(1).div_ceil(8);
    let mut scratch = vec![PackedCooEntry { key: 0, value: 0.0 }; entries.len()];

    for pass in 0..pass_count {
        let shift = pass * 8;
        let mut offsets = [0usize; 256];
        for entry in entries.iter() {
            offsets[((entry.key >> shift) & 0xff) as usize] += 1;
        }

        let mut total = 0usize;
        for offset in &mut offsets {
            let count = *offset;
            *offset = total;
            total += count;
        }

        for &entry in entries.iter() {
            let bucket = ((entry.key >> shift) & 0xff) as usize;
            scratch[offsets[bucket]] = entry;
            offsets[bucket] += 1;
        }

        std::mem::swap(entries, &mut scratch);
    }
}

pub trait NalgebraBridge {
    fn to_nalgebra_cs(&self) -> NalgebraCsMatrix<f64>;
    fn from_nalgebra_cs(matrix: &NalgebraCsMatrix<f64>) -> SparseResult<Self>
    where
        Self: Sized;
}

impl NalgebraBridge for CooMatrix {
    fn to_nalgebra_cs(&self) -> NalgebraCsMatrix<f64> {
        NalgebraCsMatrix::from_triplet(
            self.shape.rows,
            self.shape.cols,
            &self.row_indices,
            &self.col_indices,
            &self.data,
        )
    }

    fn from_nalgebra_cs(matrix: &NalgebraCsMatrix<f64>) -> SparseResult<Self> {
        let (rows, cols) = matrix.shape();
        let dense: nalgebra::DMatrix<f64> = matrix.clone().into();
        let mut triplet_rows = Vec::new();
        let mut triplet_cols = Vec::new();
        let mut triplet_data = Vec::new();

        for row in 0..rows {
            for col in 0..cols {
                let value = dense[(row, col)];
                if value != 0.0 {
                    triplet_rows.push(row);
                    triplet_cols.push(col);
                    triplet_data.push(value);
                }
            }
        }

        Self::from_triplets(
            Shape2D::new(rows, cols),
            triplet_data,
            triplet_rows,
            triplet_cols,
            false,
        )
    }
}

impl NalgebraBridge for DokMatrix {
    fn to_nalgebra_cs(&self) -> NalgebraCsMatrix<f64> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut vals = Vec::with_capacity(self.nnz());

        for (&(row, col), &value) in &self.entries {
            rows.push(row);
            cols.push(col);
            vals.push(value);
        }

        NalgebraCsMatrix::from_triplet(self.shape.rows, self.shape.cols, &rows, &cols, &vals)
    }

    fn from_nalgebra_cs(matrix: &NalgebraCsMatrix<f64>) -> SparseResult<Self> {
        let coo = CooMatrix::from_nalgebra_cs(matrix)?;
        Self::from_triplets(coo.shape, coo.data, coo.row_indices, coo.col_indices)
    }
}

impl NalgebraBridge for BsrMatrix {
    fn to_nalgebra_cs(&self) -> NalgebraCsMatrix<f64> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut vals = Vec::with_capacity(self.nnz());

        for block_row in 0..self.indptr.len().saturating_sub(1) {
            for idx in self.indptr[block_row]..self.indptr[block_row + 1] {
                let block_col = self.indices[idx];
                let block = &self.data[idx];
                for local_row in 0..self.block_shape.rows {
                    for local_col in 0..self.block_shape.cols {
                        rows.push(block_row * self.block_shape.rows + local_row);
                        cols.push(block_col * self.block_shape.cols + local_col);
                        vals.push(block[local_row * self.block_shape.cols + local_col]);
                    }
                }
            }
        }

        NalgebraCsMatrix::from_triplet(self.shape.rows, self.shape.cols, &rows, &cols, &vals)
    }

    fn from_nalgebra_cs(matrix: &NalgebraCsMatrix<f64>) -> SparseResult<Self> {
        let (rows, cols) = matrix.shape();
        Err(SparseError::Unsupported {
            feature: format!(
                "BSR conversion from nalgebra requires an explicit block size for {rows}x{cols} matrices"
            ),
        })
    }
}

impl NalgebraBridge for LilMatrix {
    fn to_nalgebra_cs(&self) -> NalgebraCsMatrix<f64> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut vals = Vec::with_capacity(self.nnz());

        for (row, (row_cols, row_values)) in self
            .row_indices
            .iter()
            .zip(self.row_data.iter())
            .enumerate()
        {
            for (&col, &value) in row_cols.iter().zip(row_values.iter()) {
                rows.push(row);
                cols.push(col);
                vals.push(value);
            }
        }

        NalgebraCsMatrix::from_triplet(self.shape.rows, self.shape.cols, &rows, &cols, &vals)
    }

    fn from_nalgebra_cs(matrix: &NalgebraCsMatrix<f64>) -> SparseResult<Self> {
        let coo = CooMatrix::from_nalgebra_cs(matrix)?;
        Self::from_triplets(coo.shape, coo.data, coo.row_indices, coo.col_indices)
    }
}

impl NalgebraBridge for CsrMatrix {
    fn to_nalgebra_cs(&self) -> NalgebraCsMatrix<f64> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut vals = Vec::with_capacity(self.nnz());

        for row in 0..self.shape.rows {
            for idx in self.indptr[row]..self.indptr[row + 1] {
                rows.push(row);
                cols.push(self.indices[idx]);
                vals.push(self.data[idx]);
            }
        }

        NalgebraCsMatrix::from_triplet(self.shape.rows, self.shape.cols, &rows, &cols, &vals)
    }

    fn from_nalgebra_cs(matrix: &NalgebraCsMatrix<f64>) -> SparseResult<Self> {
        let coo = CooMatrix::from_nalgebra_cs(matrix)?;
        let mut triplets: Vec<(usize, usize, f64)> = coo
            .row_indices
            .iter()
            .copied()
            .zip(coo.col_indices.iter().copied())
            .zip(coo.data.iter().copied())
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

        let shape = coo.shape;
        let mut indptr = vec![0usize; shape.rows + 1];
        for (r, _, _) in &dedup {
            indptr[r + 1] += 1;
        }
        for row in 0..shape.rows {
            indptr[row + 1] += indptr[row];
        }

        let mut indices = Vec::with_capacity(dedup.len());
        let mut data = Vec::with_capacity(dedup.len());
        for (_, c, v) in dedup {
            indices.push(c);
            data.push(v);
        }

        Self::from_components(shape, data, indices, indptr, true)
    }
}

impl NalgebraBridge for CscMatrix {
    fn to_nalgebra_cs(&self) -> NalgebraCsMatrix<f64> {
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());
        let mut vals = Vec::with_capacity(self.nnz());

        for col in 0..self.shape.cols {
            for idx in self.indptr[col]..self.indptr[col + 1] {
                rows.push(self.indices[idx]);
                cols.push(col);
                vals.push(self.data[idx]);
            }
        }

        NalgebraCsMatrix::from_triplet(self.shape.rows, self.shape.cols, &rows, &cols, &vals)
    }

    fn from_nalgebra_cs(matrix: &NalgebraCsMatrix<f64>) -> SparseResult<Self> {
        let coo = CooMatrix::from_nalgebra_cs(matrix)?;
        let mut triplets: Vec<(usize, usize, f64)> = coo
            .row_indices
            .iter()
            .copied()
            .zip(coo.col_indices.iter().copied())
            .zip(coo.data.iter().copied())
            .map(|((r, c), v)| (r, c, v))
            .collect();
        triplets.sort_unstable_by_key(|(r, c, _)| (*c, *r));

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

        let shape = coo.shape;
        let mut indptr = vec![0usize; shape.cols + 1];
        for (_, c, _) in &dedup {
            indptr[c + 1] += 1;
        }
        for col in 0..shape.cols {
            indptr[col + 1] += indptr[col];
        }

        let mut indices = Vec::with_capacity(dedup.len());
        let mut data = Vec::with_capacity(dedup.len());
        for (r, _, v) in dedup {
            indices.push(r);
            data.push(v);
        }

        Self::from_components(shape, data, indices, indptr, true)
    }
}

fn validate_compressed(
    label: &str,
    major_len: usize,
    minor_len: usize,
    data: &[f64],
    indices: &[usize],
    indptr: &[usize],
    canonicalize: bool,
) -> SparseResult<CanonicalMeta> {
    if data.len() != indices.len() {
        return Err(SparseError::IncompatibleShape {
            message: format!("{label} data and indices lengths differ"),
        });
    }
    if indptr.len() != major_len + 1 {
        return Err(SparseError::InvalidShape {
            message: format!("{label} indptr length must be major_len + 1"),
        });
    }
    if !indptr.windows(2).all(|w| w[0] <= w[1]) {
        return Err(SparseError::InvalidSparseStructure {
            message: format!("{label} indptr must be monotone non-decreasing"),
        });
    }
    if indptr.first().copied().unwrap_or_default() != 0 || indptr[major_len] != data.len() {
        return Err(SparseError::InvalidSparseStructure {
            message: format!(
                "{label} pointer endpoints must satisfy indptr[0]=0 and indptr[last]=nnz"
            ),
        });
    }
    for &idx in indices {
        if idx >= minor_len {
            return Err(SparseError::IndexOutOfBounds {
                axis: "minor",
                index: idx,
                bound: minor_len,
            });
        }
    }

    let (sorted, deduplicated) = detect_canonical(indptr, indices);
    Ok(CanonicalMeta {
        sorted_indices: canonicalize || sorted,
        deduplicated: canonicalize || deduplicated,
    })
}

fn validate_coordinate(shape: Shape2D, row: usize, col: usize) -> SparseResult<()> {
    if row >= shape.rows {
        return Err(SparseError::IndexOutOfBounds {
            axis: "row",
            index: row,
            bound: shape.rows,
        });
    }
    if col >= shape.cols {
        return Err(SparseError::IndexOutOfBounds {
            axis: "col",
            index: col,
            bound: shape.cols,
        });
    }
    Ok(())
}

fn validate_slice_spec(
    spec: SparseSliceSpec,
    bound: usize,
    axis: &'static str,
) -> SparseResult<()> {
    if spec.step == 0 {
        return Err(SparseError::InvalidArgument {
            message: "slice step must be >= 1".to_string(),
        });
    }
    if spec.start > bound {
        return Err(SparseError::IndexOutOfBounds {
            axis,
            index: spec.start,
            bound,
        });
    }
    if spec.end > bound {
        return Err(SparseError::IndexOutOfBounds {
            axis,
            index: spec.end,
            bound,
        });
    }
    if spec.start > spec.end {
        return Err(SparseError::InvalidArgument {
            message: "slice start must be <= end".to_string(),
        });
    }
    Ok(())
}

fn validate_boolean_mask(
    mask: &[bool],
    expected_len: usize,
    axis: &'static str,
) -> SparseResult<()> {
    if mask.len() != expected_len {
        return Err(SparseError::InvalidArgument {
            message: format!(
                "bool index for {axis} has shape ({},) instead of ({expected_len},)",
                mask.len()
            ),
        });
    }
    Ok(())
}

fn boolean_mask_index_map(mask: &[bool]) -> Vec<Option<usize>> {
    let mut mapping = Vec::with_capacity(mask.len());
    let mut next_index = 0usize;
    for &keep in mask {
        if keep {
            mapping.push(Some(next_index));
            next_index += 1;
        } else {
            mapping.push(None);
        }
    }
    mapping
}

fn slice_len(spec: SparseSliceSpec) -> usize {
    if spec.start >= spec.end {
        0
    } else {
        1 + (spec.end - spec.start - 1) / spec.step
    }
}

fn validate_bsr_block_shape(
    shape: Shape2D,
    block_shape: Shape2D,
) -> SparseResult<(usize, usize, usize)> {
    if block_shape.rows == 0 || block_shape.cols == 0 {
        return Err(SparseError::InvalidShape {
            message: "BSR block dimensions must be non-zero".to_string(),
        });
    }
    if !shape.rows.is_multiple_of(block_shape.rows) || !shape.cols.is_multiple_of(block_shape.cols)
    {
        return Err(SparseError::InvalidShape {
            message: "BSR block dimensions must divide the matrix shape".to_string(),
        });
    }

    let block_area = block_shape
        .rows
        .checked_mul(block_shape.cols)
        .ok_or_else(|| SparseError::IndexOverflow {
            message: "BSR block area overflows usize".to_string(),
        })?;
    Ok((
        shape.rows / block_shape.rows,
        shape.cols / block_shape.cols,
        block_area,
    ))
}

fn validate_bsr_components(
    shape: Shape2D,
    block_shape: Shape2D,
    data: &[Vec<f64>],
    indices: &[usize],
    indptr: &[usize],
    canonicalize: bool,
) -> SparseResult<CanonicalMeta> {
    let (block_rows, block_cols, block_area) = validate_bsr_block_shape(shape, block_shape)?;
    if data.len() != indices.len() {
        return Err(SparseError::IncompatibleShape {
            message: "BSR data and indices lengths differ".to_string(),
        });
    }
    for block in data {
        if block.len() != block_area {
            return Err(SparseError::IncompatibleShape {
                message: format!("BSR blocks must each store exactly {block_area} scalar values"),
            });
        }
    }
    if indptr.len() != block_rows + 1 {
        return Err(SparseError::InvalidShape {
            message: "BSR indptr length must be block_rows + 1".to_string(),
        });
    }
    if !indptr.windows(2).all(|w| w[0] <= w[1]) {
        return Err(SparseError::InvalidSparseStructure {
            message: "BSR indptr must be monotone non-decreasing".to_string(),
        });
    }
    if indptr.first().copied().unwrap_or_default() != 0 || indptr[block_rows] != data.len() {
        return Err(SparseError::InvalidSparseStructure {
            message: "BSR pointer endpoints must satisfy indptr[0]=0 and indptr[last]=nnz_blocks"
                .to_string(),
        });
    }
    for &idx in indices {
        if idx >= block_cols {
            return Err(SparseError::IndexOutOfBounds {
                axis: "minor",
                index: idx,
                bound: block_cols,
            });
        }
    }

    let (sorted, deduplicated) = detect_canonical(indptr, indices);
    Ok(CanonicalMeta {
        sorted_indices: canonicalize || sorted,
        deduplicated: canonicalize || deduplicated,
    })
}

fn canonicalize_lil_row(
    row_indices: Vec<usize>,
    row_data: Vec<f64>,
    col_bound: usize,
) -> SparseResult<(Vec<usize>, Vec<f64>)> {
    if row_indices.len() != row_data.len() {
        return Err(SparseError::IncompatibleShape {
            message: "LIL row indices/data lengths must match".to_string(),
        });
    }

    let mut pairs = Vec::with_capacity(row_indices.len());
    for (col, value) in row_indices.into_iter().zip(row_data) {
        if col >= col_bound {
            return Err(SparseError::IndexOutOfBounds {
                axis: "col",
                index: col,
                bound: col_bound,
            });
        }
        pairs.push((col, value));
    }

    pairs.sort_unstable_by_key(|(col, _)| *col);

    let mut canonical_indices = Vec::with_capacity(pairs.len());
    let mut canonical_data = Vec::with_capacity(pairs.len());
    for (col, value) in pairs {
        match canonical_indices.last().copied() {
            Some(last_col) if last_col == col => {
                if let Some(last_value) = canonical_data.last_mut() {
                    *last_value += value;
                }
            }
            _ => {
                canonical_indices.push(col);
                canonical_data.push(value);
            }
        }
    }

    Ok((canonical_indices, canonical_data))
}

fn diagonal_len(shape: Shape2D, offset: isize) -> usize {
    let start_row = if offset < 0 { (-offset) as usize } else { 0 };
    let start_col = if offset > 0 { offset as usize } else { 0 };
    shape
        .rows
        .saturating_sub(start_row)
        .min(shape.cols.saturating_sub(start_col))
}

fn validate_setdiag_offset(shape: Shape2D, k: isize) -> SparseResult<()> {
    if (k > 0 && k.unsigned_abs() >= shape.cols) || (k < 0 && k.unsigned_abs() >= shape.rows) {
        return Err(SparseError::InvalidArgument {
            message: "k exceeds array dimensions".to_string(),
        });
    }
    Ok(())
}

fn diagonal_len_for_setdiag(shape: Shape2D, k: isize) -> usize {
    if k >= 0 {
        shape.rows.min(shape.cols.saturating_sub(k.unsigned_abs()))
    } else {
        shape.rows.saturating_sub(k.unsigned_abs()).min(shape.cols)
    }
}

fn diagonal_coordinates_for_setdiag(shape: Shape2D, k: isize, count: usize) -> Vec<(usize, usize)> {
    let max_count = diagonal_len_for_setdiag(shape, k).min(count);
    let mut coordinates = Vec::with_capacity(max_count);
    if k >= 0 {
        let offset = k.unsigned_abs();
        for index in 0..max_count {
            coordinates.push((index, index + offset));
        }
    } else {
        let offset = k.unsigned_abs();
        for index in 0..max_count {
            coordinates.push((index + offset, index));
        }
    }
    coordinates
}

fn detect_canonical(indptr: &[usize], indices: &[usize]) -> (bool, bool) {
    let mut sorted = true;
    let mut deduplicated = true;

    for window in indptr.windows(2) {
        let start = window[0];
        let end = window[1];
        let mut prev: Option<usize> = None;
        for &idx in &indices[start..end] {
            if let Some(prev_idx) = prev {
                if idx < prev_idx {
                    sorted = false;
                }
                if idx == prev_idx {
                    deduplicated = false;
                }
            }
            prev = Some(idx);
        }
    }

    (sorted, deduplicated)
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

fn mode_label(mode: RuntimeMode) -> &'static str {
    match mode {
        RuntimeMode::Strict => "strict",
        RuntimeMode::Hardened => "hardened",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn coo_sum_duplicates_radix_matches_legacy_bits() {
        let n = 2_048usize;
        let nnz = 131_072usize;
        let mut state = 0x86d2_2b9f_cafe_2017u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };

        let mut rows = Vec::with_capacity(nnz);
        let mut cols = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);
        for i in 0..nnz {
            rows.push(next() % n);
            cols.push(next() % 257);
            data.push((i % 17) as f64 - 8.0);
        }

        COO_SUM_DUPLICATES_RADIX_DISABLE.store(true, Ordering::Relaxed);
        let legacy = CooMatrix::from_triplets(
            Shape2D::new(n, n),
            data.clone(),
            rows.clone(),
            cols.clone(),
            true,
        )
        .expect("legacy coo");

        COO_SUM_DUPLICATES_RADIX_DISABLE.store(false, Ordering::Relaxed);
        let radix = CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, true)
            .expect("radix coo");

        assert_eq!(radix.row_indices(), legacy.row_indices());
        assert_eq!(radix.col_indices(), legacy.col_indices());
        assert_eq!(
            radix
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            legacy
                .data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );

        COO_SUM_DUPLICATES_RADIX_DISABLE.store(false, Ordering::Relaxed);
    }

    #[test]
    fn coo_array_roundtrips_three_dimensional_dense_values() {
        let dense = vec![0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 5.0];
        let array =
            CooArray::from_dense(vec![2, 2, 3], &dense).expect("valid three-dimensional array");

        assert_eq!(array.shape(), &[2, 2, 3]);
        assert_eq!(array.ndim(), 3);
        assert_eq!(array.nnz(), 5);
        assert_eq!(
            array.coords(),
            &[
                vec![0, 0, 1, 1, 1],
                vec![0, 1, 0, 0, 1],
                vec![1, 0, 0, 2, 2],
            ]
        );
        assert_eq!(array.to_dense().expect("dense materialization"), dense);
        assert!(array.has_canonical_format());
    }

    #[test]
    fn coo_array_supports_one_dimensional_and_zero_extent_shapes() {
        let vector = CooArray::from_coords(vec![4], vec![2.0, 7.0], vec![vec![0, 3]], false)
            .expect("valid one-dimensional array");
        assert_eq!(
            vector.to_dense().expect("dense vector"),
            vec![2.0, 0.0, 0.0, 7.0]
        );

        let empty = CooArray::from_coords(
            vec![2, 0, 3],
            Vec::new(),
            vec![vec![], vec![], vec![]],
            false,
        )
        .expect("zero-extent sparse array");
        assert_eq!(empty.shape(), &[2, 0, 3]);
        assert!(empty.to_dense().expect("empty dense array").is_empty());
    }

    #[test]
    fn coo_array_duplicate_sum_is_lexicographic_and_bit_stable() {
        let array = CooArray::from_coords(
            vec![2, 3, 2],
            vec![1.0, 4.0, 2.0, -1.0],
            vec![vec![1, 0, 1, 1], vec![2, 1, 2, 2], vec![0, 1, 0, 0]],
            true,
        )
        .expect("duplicate coordinates");

        assert_eq!(array.coords(), &[vec![0, 1], vec![1, 2], vec![1, 0]]);
        assert_eq!(array.data(), &[4.0, 2.0]);
        assert!(array.has_canonical_format());
        let dense = array.to_dense().expect("dense materialization");
        assert_eq!(dense[3], 4.0);
        assert_eq!(dense[10], 2.0);
    }

    #[test]
    fn coo_array_rejects_invalid_coordinate_metadata() {
        assert!(matches!(
            CooArray::from_coords(Vec::new(), Vec::new(), Vec::new(), false),
            Err(SparseError::InvalidShape { .. })
        ));
        assert!(matches!(
            CooArray::from_coords(vec![2, 3], vec![1.0], vec![vec![0]], false),
            Err(SparseError::IncompatibleShape { .. })
        ));
        assert!(matches!(
            CooArray::from_coords(vec![2, 3], vec![1.0], vec![vec![0], vec![3]], false),
            Err(SparseError::InvalidSparseStructure { .. })
        ));
    }

    #[test]
    fn coo_array_matrix_conversion_requires_exactly_two_axes() {
        let matrix =
            CooMatrix::from_triplets(Shape2D::new(2, 3), vec![5.0], vec![1], vec![2], false)
                .expect("valid COO matrix");
        let array = CooArray::from_coo_matrix(&matrix);
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(
            array.to_coo_matrix().expect("two-dimensional conversion"),
            matrix
        );

        let tensor = CooArray::from_coords(
            vec![1, 2, 3],
            vec![5.0],
            vec![vec![0], vec![1], vec![2]],
            false,
        )
        .expect("valid tensor");
        assert!(matches!(
            tensor.to_coo_matrix(),
            Err(SparseError::InvalidShape { .. })
        ));
    }

    #[test]
    fn get_index_dtype_matches_scipy_width_and_content_rules() {
        assert_eq!(
            get_index_dtype(&[], None, false).expect("empty index selection"),
            IndexDtype::Int32
        );
        assert_eq!(
            get_index_dtype(
                &[IndexArrayRef::I32(&[0, 1]), IndexArrayRef::U16(&[2])],
                Some(u64::from(i32::MAX.unsigned_abs())),
                false,
            )
            .expect("narrow declared types"),
            IndexDtype::Int32
        );
        assert_eq!(
            get_index_dtype(&[IndexArrayRef::I64(&[0, 1])], None, false)
                .expect("wide declared type"),
            IndexDtype::Int64
        );
        assert_eq!(
            get_index_dtype(&[IndexArrayRef::I64(&[i32::MIN.into(), 7])], None, true)
                .expect("wide values fit int32"),
            IndexDtype::Int32
        );
        assert_eq!(
            get_index_dtype(
                &[IndexArrayRef::I64(&[i64::from(i32::MIN) - 1])],
                None,
                true
            )
            .expect("wide value needs int64"),
            IndexDtype::Int64
        );
        assert_eq!(
            get_index_dtype(&[IndexArrayRef::U64(&[])], None, true).expect("empty wide array"),
            IndexDtype::Int32
        );
        assert_eq!(
            get_index_dtype(&[], Some(u64::from(i32::MAX.unsigned_abs()) + 1), false,)
                .expect("large shape needs int64"),
            IndexDtype::Int64
        );
        assert!(matches!(
            get_index_dtype(&[], Some(i64::MAX.unsigned_abs() + 1), false),
            Err(SparseError::IndexOverflow { .. })
        ));
    }

    #[test]
    fn safely_cast_compressed_indices_preserves_values_and_shape() {
        let csr = CsrMatrix::from_components(
            Shape2D::new(2, 3),
            vec![1.0, 2.0, 3.0],
            vec![0, 2, 1],
            vec![0, 2, 3],
            true,
        )
        .expect("valid CSR fixture");

        assert_eq!(
            safely_cast_index_arrays(&csr, IndexDtype::Int32, "").expect("CSR indices fit int32"),
            SparseIndexArrays::Compressed {
                indices: IndexArray::Int32(vec![0, 2, 1]),
                indptr: IndexArray::Int32(vec![0, 2, 3]),
            }
        );
        assert_eq!(
            safely_cast_index_arrays(&csr, IndexDtype::Int64, "").expect("CSR indices fit int64"),
            SparseIndexArrays::Compressed {
                indices: IndexArray::Int64(vec![0, 2, 1]),
                indptr: IndexArray::Int64(vec![0, 2, 3]),
            }
        );
    }

    #[test]
    fn safely_cast_coo_checks_values_not_only_large_shape() {
        let int32_max = usize::try_from(i32::MAX).expect("i32 maximum fits usize");
        let sparse_large_shape = CooMatrix::from_triplets(
            Shape2D::new(int32_max + 2, 2),
            vec![4.0],
            vec![1],
            vec![0],
            false,
        )
        .expect("large COO shape with small coordinate");
        assert_eq!(
            safely_cast_index_arrays(&sparse_large_shape, IndexDtype::Int32, "")
                .expect("actual coordinates fit int32"),
            SparseIndexArrays::Coordinates(vec![
                IndexArray::Int32(vec![1]),
                IndexArray::Int32(vec![0]),
            ])
        );

        let large_coordinate = CooMatrix::from_triplets(
            Shape2D::new(int32_max + 2, 1),
            vec![4.0],
            vec![int32_max + 1],
            vec![0],
            false,
        )
        .expect("large COO coordinate");
        let error = safely_cast_index_arrays(&large_coordinate, IndexDtype::Int32, "SuperLU")
            .expect_err("coordinate does not fit int32");
        assert_eq!(
            error,
            SparseError::IndexOverflow {
                message: "coords values too large for SuperLU".to_string(),
            }
        );
        assert_eq!(
            safely_cast_index_arrays(&large_coordinate, IndexDtype::Int64, "")
                .expect("coordinate fits int64"),
            SparseIndexArrays::Coordinates(vec![
                IndexArray::Int64(vec![i64::from(i32::MAX) + 1]),
                IndexArray::Int64(vec![0]),
            ])
        );
    }

    #[test]
    fn safely_cast_dia_preserves_signed_offsets() {
        let dia = DiaMatrix::from_diagonals(
            Shape2D::new(3, 3),
            vec![-1, 1],
            vec![vec![2.0, 3.0], vec![4.0, 5.0]],
        )
        .expect("valid DIA fixture");
        assert_eq!(
            safely_cast_index_arrays(&dia, IndexDtype::Int32, "").expect("DIA offsets fit int32"),
            SparseIndexArrays::Offsets(IndexArray::Int32(vec![-1, 1]))
        );
    }

    #[test]
    fn safely_cast_bsr_checks_scalar_scaled_indices() {
        let int32_max = usize::try_from(i32::MAX).expect("i32 maximum fits usize");
        let block_col = int32_max / 2 + 1;
        let bsr = BsrMatrix::from_components(
            Shape2D::new(1, (block_col + 1) * 2),
            Shape2D::new(1, 2),
            vec![vec![1.0, 0.0]],
            vec![block_col],
            vec![0, 1],
            true,
        )
        .expect("valid large-column BSR fixture");
        assert!(matches!(
            safely_cast_index_arrays(&bsr, IndexDtype::Int32, ""),
            Err(SparseError::IndexOverflow { message })
                if message == "indices values too large for dtype int32"
        ));
    }

    #[test]
    fn safely_cast_rejects_non_array_index_formats() {
        let dok = DokMatrix::new(Shape2D::new(2, 2));
        let lil = LilMatrix::new(Shape2D::new(2, 2));
        assert!(matches!(
            safely_cast_index_arrays(&dok, IndexDtype::Int32, ""),
            Err(SparseError::Unsupported { feature })
                if feature.contains("DOK")
        ));
        assert!(matches!(
            safely_cast_index_arrays(&lil, IndexDtype::Int32, ""),
            Err(SparseError::Unsupported { feature })
                if feature.contains("LIL")
        ));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn safely_cast_rejects_usize_values_above_int64() {
        let row = usize::try_from(i64::MAX).expect("i64 maximum fits 64-bit usize") + 1;
        let coo = CooMatrix::from_triplets(
            Shape2D::new(row + 1, 1),
            vec![1.0],
            vec![row],
            vec![0],
            false,
        )
        .expect("COO can represent a native index above int64");
        assert!(matches!(
            safely_cast_index_arrays(&coo, IndexDtype::Int64, ""),
            Err(SparseError::IndexOverflow { .. })
        ));
    }
}
