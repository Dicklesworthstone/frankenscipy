// Numeric kernels: index walks over parallel slices, wide signatures the algorithm
// requires, and literals carried at full written precision.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::neg_cmp_op_on_partial_ord)]
#![allow(clippy::type_complexity)]
#![forbid(unsafe_code)]
// Numeric kernels: fixture vectors and deliberate min/max comparisons.
#![allow(clippy::useless_vec)]
#![allow(clippy::min_max)]
#![allow(clippy::absurd_extreme_comparisons)]

//! Input/Output routines for FrankenSciPy.
//!
//! Matches `scipy.io` core functions:
//! - `loadmat` / `savemat` — MATLAB MAT-files, Level 5 (v6/v7, compressed or not) and Level 4:
//!   numeric (real, complex, logical, N-D), char, cell, struct, object, sparse, function handles
//! - `whosmat` / `matfile_version` / `varmats_from_mat` — MAT-file inventory and splitting
//! - `mmread` / `mmwrite` — Matrix Market format read/write
//! - `wavfile.read` / `wavfile.write` — WAV audio file read/write
//! - `netcdf_file` — NetCDF (simplified) read/write
//! - `FortranFile` — sequential unformatted record read/write
//! - `hb_read` / `hb_write` — real assembled Harwell-Boeing sparse matrices
//! - `readsav` — IDL SAVE scalar and primitive array read support

// `write!` into the output String avoids the temporary String that
// `push_str(&format!(...))` allocates per cell/entry on the hot write paths.
use std::fmt::Write as _;

pub use fsci_runtime::{
    AuditAction, AuditEvent, AuditLedger, HARDENED_MAX_DIM, RuntimeMode, SyncSharedAuditLedger,
};
use fsci_runtime::{AuditScope, Fingerprinter, audit_finish, audit_reject, casp_now_unix_ms};

/// Create a new shared audit ledger for synchronous contexts.
#[must_use]
pub fn sync_audit_ledger() -> SyncSharedAuditLedger {
    AuditLedger::shared()
}

fn lock_or_recover(ledger: &SyncSharedAuditLedger) -> std::sync::MutexGuard<'_, AuditLedger> {
    match ledger.lock() {
        Ok(g) => g,
        Err(poisoned) => {
            ledger.clear_poison();
            poisoned.into_inner()
        }
    }
}

/// Record a fail-closed audit event when Hardened mode rejects input. `fingerprint` is the
/// call's `fsci_runtime::Fingerprinter` digest over every input and option.
pub fn record_fail_closed(
    ledger: &SyncSharedAuditLedger,
    fingerprint: &str,
    reason: &str,
    outcome: &str,
) {
    let event = AuditEvent::new(
        casp_now_unix_ms(),
        fingerprint,
        AuditAction::FailClosed {
            reason: reason.to_string(),
        },
        outcome.to_string(),
    );
    lock_or_recover(ledger).record(event);
}

/// Record a bounded-recovery audit event when Hardened mode falls back. `fingerprint` is the
/// call's `fsci_runtime::Fingerprinter` digest over every input and option.
pub fn record_bounded_recovery(
    ledger: &SyncSharedAuditLedger,
    fingerprint: &str,
    recovery_action: &str,
    outcome: &str,
) {
    let event = AuditEvent::new(
        casp_now_unix_ms(),
        fingerprint,
        AuditAction::BoundedRecovery {
            recovery_action: recovery_action.to_string(),
        },
        outcome.to_string(),
    );
    lock_or_recover(ledger).record(event);
}

/// Error type for I/O operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IoError {
    InvalidFormat(String),
    IoFailed(String),
    UnsupportedFeature(String),
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFormat(msg) => write!(f, "invalid format: {msg}"),
            Self::IoFailed(msg) => write!(f, "I/O failed: {msg}"),
            Self::UnsupportedFeature(msg) => write!(f, "unsupported: {msg}"),
        }
    }
}

impl std::error::Error for IoError {}

impl IoError {
    /// The audit reason code of this error (frankenscipy-3cu8u.2).
    const fn reason_code(&self) -> &'static str {
        match self {
            Self::InvalidFormat(_) => "invalid_format",
            Self::IoFailed(_) => "io_failed",
            Self::UnsupportedFeature(_) => "unsupported_feature",
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Matrix Market Format
// ══════════════════════════════════════════════════════════════════════

/// Matrix Market object type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmObject {
    Matrix,
    Vector,
}

/// Matrix Market format type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmFormat {
    Coordinate,
    Array,
}

/// Matrix Market field type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmField {
    Real,
    Integer,
    Complex,
    Pattern,
}

/// Matrix Market symmetry type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmSymmetry {
    General,
    Symmetric,
    SkewSymmetric,
    Hermitian,
}

/// Matrix Market header information.
#[derive(Debug, Clone)]
pub struct MmInfo {
    pub object: MmObject,
    pub format: MmFormat,
    pub field: MmField,
    pub symmetry: MmSymmetry,
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
}

/// Dense matrix result from Matrix Market.
#[derive(Debug, Clone)]
pub struct MmMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
    pub complex_data: Option<Vec<(f64, f64)>>,
    pub info: MmInfo,
}

/// Sparse (COO triplet) result from a Matrix Market **coordinate** matrix — the
/// stored nonzeros only, with NO dense `rows*cols` materialization. Matches
/// `scipy.io.mmread` returning a sparse COO matrix for coordinate-format files
/// (the format is designed for sparse data). Symmetric/skew/hermitian files have
/// their stored triangle expanded to both off-diagonal positions, so scattering
/// `(row_indices, col_indices, values)` with `+=` reproduces [`mmread`]'s dense
/// `data` exactly. Duplicate coordinates are preserved (COO semantics; they sum
/// on the dense scatter, matching `mmread`).
#[derive(Debug, Clone)]
pub struct MmSparse {
    pub rows: usize,
    pub cols: usize,
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f64>,
    pub info: MmInfo,
}

const MAX_MM_DENSE_ELEMENTS: usize = 128 * 1024 * 1024;

fn checked_matrix_len(rows: usize, cols: usize, context: &str) -> Result<usize, IoError> {
    rows.checked_mul(cols).ok_or_else(|| {
        IoError::InvalidFormat(format!(
            "{context} dimensions {rows}x{cols} overflowed usize"
        ))
    })
}

fn checked_mm_dense_read_len(rows: usize, cols: usize) -> Result<usize, IoError> {
    let dense_len = checked_matrix_len(rows, cols, "Matrix Market matrix")?;
    if dense_len > MAX_MM_DENSE_ELEMENTS {
        return Err(IoError::InvalidFormat(format!(
            "Matrix Market matrix dimensions {rows}x{cols} exceed dense read safety bound of {MAX_MM_DENSE_ELEMENTS} elements"
        )));
    }
    Ok(dense_len)
}

#[inline]
fn mm_token_eq(token: &str, expected: &str) -> bool {
    // ubs:ignore — Matrix Market grammar metadata is public input, not secret material.
    token.eq_ignore_ascii_case(expected) || (!token.is_ascii() && token.to_lowercase() == expected)
}

fn parse_mm_info(lines: &mut std::str::Lines<'_>) -> Result<MmInfo, IoError> {
    let header = lines
        .next()
        .ok_or_else(|| IoError::InvalidFormat("empty file".to_string()))?;
    if !header.starts_with("%%MatrixMarket") {
        return Err(IoError::InvalidFormat(
            "missing %%MatrixMarket header".to_string(),
        ));
    }

    // Matrix Market metadata has four fixed tokens after the banner. Pull them
    // directly from the iterator so the success path does not allocate a token
    // vector or four lowercase Strings. The non-ASCII fallback preserves the
    // former Unicode lowercase behavior outside the format's ASCII grammar.
    let mut parts = header.split_whitespace();
    let _banner = parts.next();
    let (Some(object_token), Some(format_token), Some(field_token), Some(symmetry_token)) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(IoError::InvalidFormat("incomplete header line".to_string()));
    };

    let object = if mm_token_eq(object_token, "matrix") {
        MmObject::Matrix
    } else if mm_token_eq(object_token, "vector") {
        MmObject::Vector
    } else {
        let other = object_token.to_lowercase();
        return Err(IoError::InvalidFormat(format!(
            "unknown object type: {other}"
        )));
    };

    let format = if mm_token_eq(format_token, "coordinate") {
        MmFormat::Coordinate
    } else if mm_token_eq(format_token, "array") {
        MmFormat::Array
    } else {
        let other = format_token.to_lowercase();
        return Err(IoError::InvalidFormat(format!("unknown format: {other}")));
    };

    let field = if mm_token_eq(field_token, "real") {
        MmField::Real
    } else if mm_token_eq(field_token, "integer") {
        MmField::Integer
    } else if mm_token_eq(field_token, "complex") {
        MmField::Complex
    } else if mm_token_eq(field_token, "pattern") {
        MmField::Pattern
    } else {
        let other = field_token.to_lowercase();
        return Err(IoError::InvalidFormat(format!(
            "unknown field type: {other}"
        )));
    };

    let symmetry = if mm_token_eq(symmetry_token, "general") {
        MmSymmetry::General
    } else if mm_token_eq(symmetry_token, "symmetric") {
        MmSymmetry::Symmetric
    } else if mm_token_eq(symmetry_token, "skew-symmetric") {
        MmSymmetry::SkewSymmetric
    } else if mm_token_eq(symmetry_token, "hermitian") {
        MmSymmetry::Hermitian
    } else {
        let other = symmetry_token.to_lowercase();
        return Err(IoError::InvalidFormat(format!("unknown symmetry: {other}")));
    };

    let size_str = lines
        .by_ref()
        .find_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('%') {
                None
            } else {
                Some(trimmed)
            }
        })
        .ok_or_else(|| IoError::InvalidFormat("missing size line".to_string()))?;
    let mut size_parts = size_str.split_whitespace();

    match format {
        MmFormat::Coordinate => {
            let (Some(rows_token), Some(cols_token), Some(nnz_token)) =
                (size_parts.next(), size_parts.next(), size_parts.next())
            else {
                return Err(IoError::InvalidFormat(
                    "coordinate format requires rows cols nnz".to_string(),
                ));
            };
            let rows: usize = rows_token
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("bad rows: {e}")))?;
            let cols: usize = cols_token
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("bad cols: {e}")))?;
            let nnz: usize = nnz_token
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("bad nnz: {e}")))?;

            Ok(MmInfo {
                object,
                format,
                field,
                symmetry,
                rows,
                cols,
                nnz,
            })
        }
        MmFormat::Array => {
            let (Some(rows_token), Some(cols_token)) = (size_parts.next(), size_parts.next())
            else {
                return Err(IoError::InvalidFormat(
                    "array format requires rows cols".to_string(),
                ));
            };
            let rows: usize = rows_token
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("bad rows: {e}")))?;
            let cols: usize = cols_token
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("bad cols: {e}")))?;
            let nnz = rows.checked_mul(cols).ok_or_else(|| {
                IoError::InvalidFormat("array dimensions overflowed nnz computation".to_string())
            })?;

            Ok(MmInfo {
                object,
                format,
                field,
                symmetry,
                rows,
                cols,
                nnz,
            })
        }
    }
}

/// Read a Matrix Market file.
///
/// Matches `scipy.io.mmread`.
pub fn mmread(content: &str) -> Result<MmMatrix, IoError> {
    mmread_with_mode(content, RuntimeMode::Strict, None)
}

/// Read a Matrix Market file under an explicit runtime policy with optional audit ledger.
///
/// With a ledger, every error it returns, in either mode, is recorded as one `FailClosed`
/// event (frankenscipy-3cu8u.2).
pub fn mmread_with_mode(
    content: &str,
    mode: RuntimeMode,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<MmMatrix, IoError> {
    // The whole request: every byte of `content`, then `mode` (Debug). Computed only when an
    // event is recorded.
    let fingerprint = || {
        Fingerprinter::new("fsci_io::mmread_with_mode")
            .bytes(content.as_bytes())
            .str(&format!("{mode:?}"))
            .finish()
    };
    let audit = audit_ledger.map(|ledger| AuditScope::new(ledger, &fingerprint));
    let result = mmread_audited(content, mode, audit.as_ref());
    audit_finish(audit.as_ref(), result, IoError::reason_code)
}

/// [`mmread_with_mode`]'s read, recording its rejections under `audit`.
fn mmread_audited(
    content: &str,
    mode: RuntimeMode,
    audit: Option<&AuditScope<'_>>,
) -> Result<MmMatrix, IoError> {
    let mut lines = content.lines();
    let info = parse_mm_info(&mut lines)?;

    if matches!(mode, RuntimeMode::Hardened)
        && (info.rows > HARDENED_MAX_DIM || info.cols > HARDENED_MAX_DIM)
    {
        audit_reject(
            audit,
            "resource_exhausted",
            "rejected: dimension exceeds hardened limit",
        );
        return Err(IoError::InvalidFormat(format!(
            "Matrix Market dimensions {}x{} exceed hardened limit ({HARDENED_MAX_DIM})",
            info.rows, info.cols
        )));
    }

    if info.symmetry != MmSymmetry::General && info.rows != info.cols {
        let symmetry = match info.symmetry {
            MmSymmetry::General => "general",
            MmSymmetry::Symmetric => "symmetric",
            MmSymmetry::SkewSymmetric => "skew-symmetric",
            MmSymmetry::Hermitian => "hermitian",
        };
        return Err(IoError::InvalidFormat(format!(
            "Matrix Market {symmetry} symmetry requires a square matrix, got {}x{}",
            info.rows, info.cols
        )));
    }

    match info.format {
        MmFormat::Coordinate => {
            let rows = info.rows;
            let cols = info.cols;
            let nnz = info.nnz;
            let dense_len = checked_mm_dense_read_len(rows, cols)?;
            let mut data = vec![0.0; dense_len];
            let mut complex_data =
                (info.field == MmField::Complex).then(|| vec![(0.0, 0.0); dense_len]);
            let mut seen_nnz = 0usize;

            for line in lines {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('%') {
                    continue;
                }
                // Pull the row/col indices (and optional value) straight off the
                // split iterator instead of materializing a Vec<&str> per entry —
                // byte-identical (same fields, same skip-if-<2-tokens, same errors).
                let mut fields = trimmed.split_whitespace();
                let (Some(f_row), Some(f_col)) = (fields.next(), fields.next()) else {
                    continue;
                };
                let r_one_based = f_row
                    .parse::<usize>()
                    .map_err(|e| IoError::InvalidFormat(format!("bad row index: {e}")))?;
                let c_one_based = f_col
                    .parse::<usize>()
                    .map_err(|e| IoError::InvalidFormat(format!("bad col index: {e}")))?;
                let r = r_one_based.checked_sub(1).ok_or_else(|| {
                    IoError::InvalidFormat(
                        "Matrix Market row indices must be 1-based and >= 1".to_string(),
                    )
                })?;
                let c = c_one_based.checked_sub(1).ok_or_else(|| {
                    IoError::InvalidFormat(
                        "Matrix Market col indices must be 1-based and >= 1".to_string(),
                    )
                })?;
                let v: f64;
                let v_im: f64;
                if info.field == MmField::Pattern {
                    v = 1.0;
                    v_im = 0.0;
                } else if let Some(f_val) = fields.next() {
                    v = f_val
                        .parse()
                        .map_err(|e| IoError::InvalidFormat(format!("bad value: {e}")))?;
                    v_im = if info.field == MmField::Complex {
                        fields
                            .next()
                            .ok_or_else(|| {
                                IoError::InvalidFormat(
                                    "complex coordinate entry missing imaginary value".to_string(),
                                )
                            })?
                            .parse()
                            .map_err(|e| {
                                IoError::InvalidFormat(format!("bad imaginary value: {e}"))
                            })?
                    } else {
                        0.0
                    };
                } else {
                    return Err(IoError::InvalidFormat(
                        "coordinate entry missing value for non-pattern field".to_string(),
                    ));
                }
                if r >= rows || c >= cols {
                    return Err(IoError::InvalidFormat(format!(
                        "coordinate entry ({r}, {c}) out of bounds for {rows}x{cols}"
                    )));
                }

                let add_val = |cd: &mut Option<Vec<(f64, f64)>>,
                               d: &mut Vec<f64>,
                               r: usize,
                               c: usize,
                               vr: f64,
                               vi: f64| {
                    let i = r * cols + c;
                    if let Some(cdata) = cd {
                        cdata[i].0 += vr;
                        cdata[i].1 += vi;
                    }
                    d[i] += vr;
                };
                let sub_val = |cd: &mut Option<Vec<(f64, f64)>>,
                               d: &mut Vec<f64>,
                               r: usize,
                               c: usize,
                               vr: f64,
                               vi: f64| {
                    let i = r * cols + c;
                    if let Some(cdata) = cd {
                        cdata[i].0 -= vr;
                        cdata[i].1 -= vi;
                    }
                    d[i] -= vr;
                };

                match info.symmetry {
                    MmSymmetry::General => {
                        add_val(&mut complex_data, &mut data, r, c, v, v_im);
                    }
                    MmSymmetry::Symmetric | MmSymmetry::Hermitian => {
                        add_val(&mut complex_data, &mut data, r, c, v, v_im);
                        if r != c {
                            if info.symmetry == MmSymmetry::Hermitian {
                                add_val(&mut complex_data, &mut data, c, r, v, -v_im);
                            } else {
                                add_val(&mut complex_data, &mut data, c, r, v, v_im);
                            }
                        }
                    }
                    MmSymmetry::SkewSymmetric => {
                        if r == c {
                            if v != 0.0 || v_im != 0.0 {
                                return Err(IoError::InvalidFormat(
                                    "skew-symmetric diagonal entries must be zero".to_string(),
                                ));
                            }
                        } else {
                            add_val(&mut complex_data, &mut data, r, c, v, v_im);
                            sub_val(&mut complex_data, &mut data, c, r, v, v_im);
                        }
                    }
                }
                seen_nnz += 1;
            }

            if seen_nnz != nnz {
                return Err(IoError::InvalidFormat(format!(
                    "coordinate format expected {nnz} entries but found {seen_nnz}"
                )));
            }

            Ok(MmMatrix {
                rows,
                cols,
                data,
                complex_data,
                info,
            })
        }
        MmFormat::Array => {
            let rows = info.rows;
            let cols = info.cols;
            let dense_len = checked_mm_dense_read_len(rows, cols)?;
            let mut data = vec![0.0; dense_len];
            let mut complex_data =
                (info.field == MmField::Complex).then(|| vec![(0.0, 0.0); dense_len]);

            // General arrays store every value in column-major order. Stream
            // them straight into the row-major destination instead of staging
            // one `(row, col)` pair and one parsed value per matrix element.
            if info.symmetry == MmSymmetry::General {
                let mut values_seen = 0usize;
                for line in lines {
                    let trimmed = line.trim();
                    if trimmed.is_empty() || trimmed.starts_with('%') {
                        continue;
                    }
                    if values_seen >= dense_len {
                        return Err(IoError::InvalidFormat(format!(
                            "array format has more than the declared {dense_len} values"
                        )));
                    }
                    let value = trimmed
                        .parse()
                        .map_err(|e| IoError::InvalidFormat(format!("bad value: {e}")))?;
                    let row = values_seen % rows;
                    let col = values_seen / rows;
                    data[row * cols + col] = value;
                    values_seen += 1;
                }
                if values_seen != dense_len {
                    return Err(IoError::InvalidFormat(format!(
                        "array format expected {dense_len} values but found {values_seen}"
                    )));
                }

                return Ok(MmMatrix {
                    rows,
                    cols,
                    data,
                    complex_data: None,
                    info,
                });
            }

            // Stored (row, col) positions in column-major file order. For
            // symmetric/hermitian only the lower triangle (incl. diagonal) is
            // stored; skew-symmetric stores the strictly-lower triangle; the
            // upper triangle is reconstructed by mirroring (negating for skew).
            // This matches `scipy.io.mmread` of `scipy.io.mmwrite(..., symmetry=)`.
            let positions: Vec<(usize, usize)> = match info.symmetry {
                MmSymmetry::General => Vec::new(),
                MmSymmetry::Symmetric | MmSymmetry::Hermitian => (0..cols)
                    .flat_map(|c| (c..rows).map(move |r| (r, c)))
                    .collect(),
                MmSymmetry::SkewSymmetric => (0..cols)
                    .flat_map(|c| (c + 1..rows).map(move |r| (r, c)))
                    .collect(),
            };

            let mut values: Vec<(f64, f64)> = Vec::with_capacity(positions.len());
            for line in lines {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('%') {
                    continue;
                }
                if values.len() >= positions.len() {
                    return Err(IoError::InvalidFormat(format!(
                        "array format has more than the declared {} values",
                        positions.len()
                    )));
                }
                let mut fields = trimmed.split_whitespace();
                let real = fields
                    .next()
                    .ok_or_else(|| IoError::InvalidFormat("array entry missing value".to_string()))?
                    .parse()
                    .map_err(|e| IoError::InvalidFormat(format!("bad value: {e}")))?;
                let imag = if info.field == MmField::Complex {
                    fields
                        .next()
                        .ok_or_else(|| {
                            IoError::InvalidFormat(
                                "complex array entry missing imaginary value".to_string(),
                            )
                        })?
                        .parse()
                        .map_err(|e| IoError::InvalidFormat(format!("bad imaginary value: {e}")))?
                } else {
                    0.0
                };
                values.push((real, imag));
            }
            if values.len() != positions.len() {
                return Err(IoError::InvalidFormat(format!(
                    "array format expected {} values but found {}",
                    positions.len(),
                    values.len()
                )));
            }

            for (&(row, col), &(real, imag)) in positions.iter().zip(values.iter()) {
                let index = row * cols + col;
                data[index] = real;
                if let Some(values) = &mut complex_data {
                    values[index] = (real, imag);
                }
                // Mirror the stored triangle for the symmetric families (each is
                // guaranteed square above, so the transposed index is in bounds).
                // General stores the full matrix, so it is never mirrored.
                if info.symmetry != MmSymmetry::General && row != col {
                    let (mirror_real, mirror_imag) = if info.symmetry == MmSymmetry::SkewSymmetric {
                        (-real, -imag)
                    } else if info.symmetry == MmSymmetry::Hermitian {
                        (real, -imag)
                    } else {
                        (real, imag)
                    };
                    let mirror_index = col * cols + row;
                    data[mirror_index] = mirror_real;
                    if let Some(values) = &mut complex_data {
                        values[mirror_index] = (mirror_real, mirror_imag);
                    }
                }
            }

            Ok(MmMatrix {
                rows,
                cols,
                data,
                complex_data,
                info,
            })
        }
    }
}

/// Read a Matrix Market **coordinate** (sparse) matrix as COO triplets, skipping
/// the dense `rows*cols` buffer that [`mmread`] allocates.
///
/// For a mostly-zero matrix that dense buffer dominates both runtime and memory:
/// [`mmread`] of a 4000×4000 @ 1% file spends ~120 ms (almost all in first-touch
/// page faults across the 128 MB dense array) and holds 128 MB for ~160 k
/// nonzeros; `mmread_sparse` parses the same file to COO in ~13 ms (≈10× faster,
/// ~SciPy parity) and holds only the triplets. Matches `scipy.io.mmread`, which
/// returns a sparse COO matrix for coordinate-format files.
///
/// Symmetric/skew-symmetric/hermitian files store one triangle; the mirrored
/// off-diagonal entry is emitted too (negated for skew), so scattering the
/// triplets into a dense array with `+=` reproduces [`mmread`]'s `data` exactly.
/// Errors on `array` (dense) format — use [`mmread`] there.
pub fn mmread_sparse(content: &str) -> Result<MmSparse, IoError> {
    let mut lines = content.lines();
    let info = parse_mm_info(&mut lines)?;

    if info.field == MmField::Complex {
        return Err(IoError::UnsupportedFeature(
            "Matrix Market complex field is not supported".to_string(),
        ));
    }
    if info.format != MmFormat::Coordinate {
        return Err(IoError::UnsupportedFeature(
            "mmread_sparse requires coordinate (sparse) format; use mmread for array format"
                .to_string(),
        ));
    }
    if info.symmetry != MmSymmetry::General && info.rows != info.cols {
        let symmetry = match info.symmetry {
            MmSymmetry::General => "general",
            MmSymmetry::Symmetric => "symmetric",
            MmSymmetry::SkewSymmetric => "skew-symmetric",
            MmSymmetry::Hermitian => "hermitian",
        };
        return Err(IoError::InvalidFormat(format!(
            "Matrix Market {symmetry} symmetry requires a square matrix, got {}x{}",
            info.rows, info.cols
        )));
    }

    let rows = info.rows;
    let cols = info.cols;
    // Off-diagonal entries of the symmetric families expand to two triplets.
    let cap = info
        .nnz
        .saturating_mul(if info.symmetry == MmSymmetry::General {
            1
        } else {
            2
        });
    let mut row_indices: Vec<usize> = Vec::with_capacity(cap);
    let mut col_indices: Vec<usize> = Vec::with_capacity(cap);
    let mut values: Vec<f64> = Vec::with_capacity(cap);
    let mut seen_nnz = 0usize;

    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }
        let mut fields = trimmed.split_whitespace();
        let (Some(f_row), Some(f_col)) = (fields.next(), fields.next()) else {
            continue;
        };
        let r = f_row
            .parse::<usize>()
            .map_err(|e| IoError::InvalidFormat(format!("bad row index: {e}")))?
            .checked_sub(1)
            .ok_or_else(|| {
                IoError::InvalidFormat(
                    "Matrix Market row indices must be 1-based and >= 1".to_string(),
                )
            })?;
        let c = f_col
            .parse::<usize>()
            .map_err(|e| IoError::InvalidFormat(format!("bad col index: {e}")))?
            .checked_sub(1)
            .ok_or_else(|| {
                IoError::InvalidFormat(
                    "Matrix Market col indices must be 1-based and >= 1".to_string(),
                )
            })?;
        let v: f64 = if info.field == MmField::Pattern {
            1.0
        } else if let Some(f_val) = fields.next() {
            f_val
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("bad value: {e}")))?
        } else {
            return Err(IoError::InvalidFormat(
                "coordinate entry missing value for non-pattern field".to_string(),
            ));
        };

        if r >= rows || c >= cols {
            return Err(IoError::InvalidFormat(format!(
                "coordinate entry ({r}, {c}) out of bounds for {rows}x{cols}"
            )));
        }

        match info.symmetry {
            MmSymmetry::General => {
                row_indices.push(r);
                col_indices.push(c);
                values.push(v);
            }
            MmSymmetry::Symmetric | MmSymmetry::Hermitian => {
                row_indices.push(r);
                col_indices.push(c);
                values.push(v);
                if r != c {
                    row_indices.push(c);
                    col_indices.push(r);
                    values.push(v);
                }
            }
            MmSymmetry::SkewSymmetric => {
                if r == c {
                    if v != 0.0 {
                        return Err(IoError::InvalidFormat(
                            "skew-symmetric diagonal entries must be zero".to_string(),
                        ));
                    }
                } else {
                    row_indices.push(r);
                    col_indices.push(c);
                    values.push(v);
                    row_indices.push(c);
                    col_indices.push(r);
                    values.push(-v);
                }
            }
        }
        seen_nnz += 1;
    }

    if seen_nnz != info.nnz {
        return Err(IoError::InvalidFormat(format!(
            "coordinate format expected {} entries but found {seen_nnz}",
            info.nnz
        )));
    }

    Ok(MmSparse {
        rows,
        cols,
        row_indices,
        col_indices,
        values,
        info,
    })
}

/// Write a dense matrix in Matrix Market format.
///
/// Matches `scipy.io.mmwrite`.
pub fn mmwrite(rows: usize, cols: usize, data: &[f64]) -> Result<String, IoError> {
    let expected_len = checked_matrix_len(rows, cols, "Matrix Market matrix")?;
    if data.len() != expected_len {
        return Err(IoError::InvalidFormat(format!(
            "data length {} doesn't match {}x{}",
            data.len(),
            rows,
            cols
        )));
    }

    const HEADER: &str = "%%MatrixMarket matrix array real general\n";
    let n = expected_len;

    // Serial gate FIRST (before the available_parallelism syscall — see the
    // per-call syscall-tax lesson): small matrices format in one pass.
    const MM_PAR_GATE: usize = 1 << 16;
    if n < MM_PAR_GATE {
        let mut out = String::new();
        out.push_str(HEADER);
        out.push_str(&format!("{rows} {cols}\n"));
        // Column-major order (Matrix Market convention)
        for c in 0..cols {
            for r in 0..rows {
                let v = data[r * cols + c];
                let _ = writeln!(out, "{v}");
            }
        }
        return Ok(out);
    }

    // The f64 Display formatting (not allocation) dominates mmwrite and is
    // embarrassingly parallel; SciPy's mmwrite is single-threaded. Each worker
    // formats a contiguous slice of the column-major value stream (value k maps
    // to col k/rows, row k%rows → data[row*cols+col]) into a private String;
    // concatenating the parts in order reproduces the serial output BIT-FOR-BIT.
    let nthreads = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
        .min(n / 16384)
        .max(1);
    if nthreads <= 1 {
        let mut out = String::new();
        out.push_str(HEADER);
        out.push_str(&format!("{rows} {cols}\n"));
        for c in 0..cols {
            for r in 0..rows {
                let v = data[r * cols + c];
                let _ = writeln!(out, "{v}");
            }
        }
        return Ok(out);
    }

    let chunk = n.div_ceil(nthreads);
    let mut parts: Vec<String> = (0..nthreads).map(|_| String::new()).collect();
    std::thread::scope(|scope| {
        for (t, slot) in parts.iter_mut().enumerate() {
            let k0 = t * chunk;
            let k1 = ((t + 1) * chunk).min(n);
            scope.spawn(move || {
                if k0 >= k1 {
                    return;
                }
                let mut local = String::with_capacity((k1 - k0) * 20);
                for k in k0..k1 {
                    let v = data[(k % rows) * cols + (k / rows)];
                    let _ = writeln!(local, "{v}");
                }
                *slot = local;
            });
        }
    });

    let total: usize = parts.iter().map(String::len).sum();
    let mut out = String::with_capacity(total + HEADER.len() + 32);
    out.push_str(HEADER);
    out.push_str(&format!("{rows} {cols}\n"));
    for p in &parts {
        out.push_str(p);
    }
    Ok(out)
}

/// Write a sparse matrix in coordinate Matrix Market format.
pub fn mmwrite_sparse(
    rows: usize,
    cols: usize,
    entries: &[(usize, usize, f64)],
) -> Result<String, IoError> {
    let mut out = String::new();
    out.push_str("%%MatrixMarket matrix coordinate real general\n");
    out.push_str(&format!("{rows} {cols} {}\n", entries.len()));

    for &(r, c, v) in entries {
        if r >= rows || c >= cols {
            return Err(IoError::InvalidFormat(format!(
                "sparse entry ({r}, {c}) out of bounds for {rows}x{cols}"
            )));
        }
        let row = r.checked_add(1).ok_or_else(|| {
            IoError::InvalidFormat("sparse row index overflowed Matrix Market encoding".to_string())
        })?;
        let col = c.checked_add(1).ok_or_else(|| {
            IoError::InvalidFormat("sparse col index overflowed Matrix Market encoding".to_string())
        })?;
        let _ = writeln!(out, "{row} {col} {v}");
    }

    Ok(out)
}

/// Write a complex dense matrix in Matrix Market format.
pub fn mmwrite_complex(rows: usize, cols: usize, data: &[(f64, f64)]) -> Result<String, IoError> {
    let expected_len = checked_matrix_len(rows, cols, "Matrix Market matrix")?;
    if data.len() != expected_len {
        return Err(IoError::InvalidFormat(format!(
            "data length {} doesn't match {}x{}",
            data.len(),
            rows,
            cols
        )));
    }

    let mut out = String::new();
    out.push_str("%%MatrixMarket matrix array complex general\n");
    out.push_str(&format!("{rows} {cols}\n"));

    // Column-major order (Matrix Market convention)
    for c in 0..cols {
        for r in 0..rows {
            let (vr, vi) = data[r * cols + c];
            let _ = writeln!(out, "{vr} {vi}");
        }
    }

    Ok(out)
}

/// Write a complex sparse matrix in coordinate Matrix Market format.
pub fn mmwrite_sparse_complex(
    rows: usize,
    cols: usize,
    entries: &[(usize, usize, (f64, f64))],
) -> Result<String, IoError> {
    let mut out = String::new();
    out.push_str("%%MatrixMarket matrix coordinate complex general\n");
    out.push_str(&format!("{rows} {cols} {}\n", entries.len()));

    for &(r, c, (vr, vi)) in entries {
        if r >= rows || c >= cols {
            return Err(IoError::InvalidFormat(format!(
                "sparse entry ({r}, {c}) out of bounds for {rows}x{cols}"
            )));
        }
        let row = r.checked_add(1).ok_or_else(|| {
            IoError::InvalidFormat("sparse row index overflowed Matrix Market encoding".to_string())
        })?;
        let col = c.checked_add(1).ok_or_else(|| {
            IoError::InvalidFormat("sparse col index overflowed Matrix Market encoding".to_string())
        })?;
        let _ = writeln!(out, "{row} {col} {vr} {vi}");
    }

    Ok(out)
}

/// Read Matrix Market info (header only).
///
/// Matches `scipy.io.mminfo`.
pub fn mminfo(content: &str) -> Result<MmInfo, IoError> {
    let mut lines = content.lines();
    parse_mm_info(&mut lines)
}

// ══════════════════════════════════════════════════════════════════════
// WAV File Format
// ══════════════════════════════════════════════════════════════════════

/// WAV file data.
///
/// `data` holds interleaved samples normalised to the floating-point range
/// `[-1.0, 1.0]` (the soundfile/librosa convention), **not** the raw integer
/// samples that `scipy.io.wavfile.read` returns. See [`wav_read`].
#[derive(Debug, Clone)]
pub struct WavData {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub data: Vec<f64>,
}

/// Read a WAV file from bytes.
///
/// Parses the RIFF/`fmt`/`data` chunks and returns the sample rate, channel
/// count, bit depth, and interleaved samples. Supported encodings: 8/16/24/32-bit
/// PCM and 32-bit IEEE float.
///
/// Note on values: samples are **normalised to `[-1.0, 1.0]`** by dividing each
/// PCM sample by its full-scale magnitude (8-bit is also recentred from
/// `[0, 255]`). `scipy.io.wavfile.read` instead returns the *raw* integer
/// samples in their native dtype, so the parsed sample rate / channels / bit
/// depth match SciPy exactly while `data` equals SciPy's samples divided by
/// full scale. This `f64` API cannot reproduce SciPy's dtype-driven raw output
/// losslessly; the chosen convention is tracked by bead frankenscipy-8hj9z.
/// At/above this sample count, wav_read's per-sample decode fans across threads.
const WAV_DECODE_PAR_GATE: usize = 1 << 18;

/// Decode `data_bytes` into one normalized `f64` per `stride`-byte sample via
/// `conv`. The decode is compute-bound (measured ~4 ns/sample scalar, and the
/// i16→f64 widen does not auto-vectorize) and per-sample independent, so above
/// WAV_DECODE_PAR_GATE it fans across threads. BIT-IDENTICAL to the serial
/// `chunks_exact(stride).map(conv)`: each worker runs the same `conv` on a
/// disjoint contiguous sample range. The serial gate is checked before the
/// available_parallelism syscall (per-call syscall-tax lesson).
fn decode_wav_samples(data_bytes: &[u8], stride: usize, conv: fn(&[u8]) -> f64) -> Vec<f64> {
    let ns = data_bytes.len() / stride;
    if ns < WAV_DECODE_PAR_GATE {
        return data_bytes.chunks_exact(stride).map(conv).collect();
    }
    let nthreads = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
        .min(ns / (1 << 17))
        .max(1);
    if nthreads <= 1 {
        return data_bytes.chunks_exact(stride).map(conv).collect();
    }
    let mut samples = vec![0.0f64; ns];
    let chunk = ns.div_ceil(nthreads);
    std::thread::scope(|scope| {
        for (t, out) in samples.chunks_mut(chunk).enumerate() {
            let base = t * chunk;
            scope.spawn(move || {
                for (k, o) in out.iter_mut().enumerate() {
                    let i = (base + k) * stride;
                    *o = conv(&data_bytes[i..i + stride]);
                }
            });
        }
    });
    samples
}

pub fn wav_read(bytes: &[u8]) -> Result<WavData, IoError> {
    if bytes.len() < 44 {
        return Err(IoError::InvalidFormat("WAV file too short".to_string()));
    }

    // RIFF header
    if &bytes[0..4] != b"RIFF" {
        return Err(IoError::InvalidFormat("missing RIFF header".to_string()));
    }
    if &bytes[8..12] != b"WAVE" {
        return Err(IoError::InvalidFormat(
            "missing WAVE identifier".to_string(),
        ));
    }

    // Find fmt chunk
    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut audio_format = 0u16;

    while pos + 8 <= bytes.len() {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;

        if chunk_id == b"fmt " {
            if chunk_size < 16 || pos + 8 + chunk_size > bytes.len() {
                return Err(IoError::InvalidFormat("fmt chunk too small".to_string()));
            }
            let fmt = &bytes[pos + 8..];
            audio_format = u16::from_le_bytes([fmt[0], fmt[1]]);
            channels = u16::from_le_bytes([fmt[2], fmt[3]]);
            if channels == 0 {
                return Err(IoError::InvalidFormat(
                    "fmt chunk declares zero channels".to_string(),
                ));
            }
            sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
            bits_per_sample = u16::from_le_bytes([fmt[14], fmt[15]]);
        } else if chunk_id == b"data" {
            if pos + 8 + chunk_size > bytes.len() {
                return Err(IoError::InvalidFormat(
                    "data chunk extends past file".to_string(),
                ));
            }
            let data_bytes = &bytes[pos + 8..pos + 8 + chunk_size];
            if sample_rate == 0 || channels == 0 || bits_per_sample == 0 {
                return Err(IoError::InvalidFormat(
                    "encountered data chunk before a valid fmt chunk".to_string(),
                ));
            }

            if audio_format != 1 && audio_format != 3 {
                return Err(IoError::UnsupportedFeature(format!(
                    "unsupported audio format: {audio_format} (only PCM=1 and IEEE_FLOAT=3 supported)"
                )));
            }
            if audio_format == 3 && bits_per_sample != 32 {
                return Err(IoError::UnsupportedFeature(format!(
                    "unsupported IEEE float bits per sample: {bits_per_sample}"
                )));
            }

            let bytes_per_sample = match bits_per_sample {
                8 => 1usize,
                16 => 2,
                24 => 3,
                32 => 4,
                _ => {
                    return Err(IoError::UnsupportedFeature(format!(
                        "unsupported bits per sample: {bits_per_sample}"
                    )));
                }
            };
            if !data_bytes.len().is_multiple_of(bytes_per_sample) {
                return Err(IoError::InvalidFormat(format!(
                    "data chunk size {} is not aligned to {}-byte samples",
                    data_bytes.len(),
                    bytes_per_sample
                )));
            }
            let frame_bytes = bytes_per_sample
                .checked_mul(channels as usize)
                .ok_or_else(|| {
                    IoError::InvalidFormat("WAV frame size overflowed usize".to_string())
                })?;
            if !data_bytes.len().is_multiple_of(frame_bytes) {
                return Err(IoError::InvalidFormat(format!(
                    "data chunk size {} does not contain whole {}-channel frames",
                    data_bytes.len(),
                    channels
                )));
            }

            // Per-sample decode, parallelized above the gate (byte-identical).
            let samples = match (bits_per_sample, audio_format) {
                (8, _) => decode_wav_samples(data_bytes, 1, |b| (b[0] as f64 - 128.0) / 128.0),
                (16, _) => decode_wav_samples(data_bytes, 2, |c| {
                    i16::from_le_bytes([c[0], c[1]]) as f64 / 32768.0
                }),
                (24, _) => decode_wav_samples(data_bytes, 3, |c| {
                    let sign = if c[2] & 0x80 != 0 { 0xFF } else { 0x00 };
                    let raw = i32::from_le_bytes([c[0], c[1], c[2], sign]);
                    raw as f64 / 8_388_608.0
                }),
                (32, 3) => decode_wav_samples(data_bytes, 4, |c| {
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64
                }),
                (32, _) => decode_wav_samples(data_bytes, 4, |c| {
                    i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64 / 2_147_483_648.0
                }),
                _ => {
                    return Err(IoError::UnsupportedFeature(format!(
                        "unsupported bits per sample: {bits_per_sample}"
                    )));
                }
            };

            return Ok(WavData {
                sample_rate,
                channels,
                bits_per_sample,
                data: samples,
            });
        }

        pos += 8 + chunk_size;
        // Chunks are word-aligned
        if !chunk_size.is_multiple_of(2) {
            pos += 1;
        }
    }

    Err(IoError::InvalidFormat("no data chunk found".to_string()))
}

/// Write WAV file data as bytes (16-bit PCM).
///
/// `data` is interpreted as interleaved samples in `[-1.0, 1.0]` (the inverse of
/// [`wav_read`]'s normalisation): each value is clamped to `[-1, 1]` and scaled
/// by `32767` into a little-endian `i16`. This differs from
/// `scipy.io.wavfile.write`, which writes the array's raw integer/float samples
/// in their native dtype without scaling. Tracked by bead frankenscipy-8hj9z.
pub fn wav_write(sample_rate: u32, channels: u16, data: &[f64]) -> Result<Vec<u8>, IoError> {
    if sample_rate == 0 {
        return Err(IoError::InvalidFormat(
            "WAV sample rate must be nonzero".to_string(),
        ));
    }
    if channels == 0 {
        return Err(IoError::InvalidFormat(
            "WAV channel count must be nonzero".to_string(),
        ));
    }
    if !data.len().is_multiple_of(channels as usize) {
        return Err(IoError::InvalidFormat(format!(
            "data length {} does not contain whole frames for {channels} channels",
            data.len()
        )));
    }
    let bits_per_sample: u16 = 16;
    let bytes_per_sample = bits_per_sample / 8;
    let data_size = data
        .len()
        .checked_mul(bytes_per_sample as usize)
        .and_then(|size| u32::try_from(size).ok())
        .ok_or_else(|| IoError::InvalidFormat("WAV data chunk too large".to_string()))?;
    let file_size = 36u32
        .checked_add(data_size)
        .ok_or_else(|| IoError::InvalidFormat("WAV file too large".to_string()))?;

    let mut buf = Vec::with_capacity(file_size as usize + 8);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate
        .checked_mul(channels as u32)
        .and_then(|rate| rate.checked_mul(bytes_per_sample as u32))
        .ok_or_else(|| IoError::InvalidFormat("WAV byte rate overflowed u32".to_string()))?;
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    let block_align = channels
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| IoError::InvalidFormat("WAV block align overflowed u16".to_string()))?;
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    for &sample in data {
        let clamped = sample.clamp(-1.0, 1.0);
        let val = (clamped * 32767.0) as i16;
        buf.extend_from_slice(&val.to_le_bytes());
    }

    Ok(buf)
}

// ══════════════════════════════════════════════════════════════════════
// MATLAB MAT-files (Level 4 and Level 5): loadmat / savemat / whosmat
// ══════════════════════════════════════════════════════════════════════
//
// A port of SciPy 1.17.1's `scipy.io.matlab` (`_mio.py`, `_mio4.py`, `_mio5.py` and the
// compiled `_mio5_utils`, `_mio_utils` and `_streams` modules, whose behaviour was pinned
// against the live interpreter). The reader keeps SciPy's stream semantics: sub-elements are
// read one after another without trusting the enclosing element's byte count, element padding
// may be missing at the end of a stream, a zlib stream without its end marker is decoded as far
// as it goes, and a trailing partial tag or a zero-length top-level element is an error. Where
// SciPy would build an inconsistent object from a malformed file (a sparse row index outside the
// matrix, a decreasing column pointer, a negative dimension) or crash on it (an unknown data type
// code in a numeric element segfaults `read_numeric`), fsci fails closed.
//
// Errors: a file SciPy rejects with ValueError, TypeError, OSError, zlib.error,
// UnicodeDecodeError or MatReadError is `IoError::InvalidFormat` here; the v7.3 (HDF5) format,
// NotImplementedError in SciPy, is `IoError::UnsupportedFeature`.

/// A MATLAB array class (`mx*_CLASS`), as the array flags of a Level 5 variable store it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatClass {
    /// `mxCELL_CLASS` (1).
    Cell,
    /// `mxSTRUCT_CLASS` (2).
    Struct,
    /// `mxOBJECT_CLASS` (3): a struct with a class name.
    Object,
    /// `mxCHAR_CLASS` (4).
    Char,
    /// `mxSPARSE_CLASS` (5).
    Sparse,
    /// `mxDOUBLE_CLASS` (6).
    Double,
    /// `mxSINGLE_CLASS` (7).
    Single,
    /// `mxINT8_CLASS` (8).
    Int8,
    /// `mxUINT8_CLASS` (9).
    Uint8,
    /// `mxINT16_CLASS` (10).
    Int16,
    /// `mxUINT16_CLASS` (11).
    Uint16,
    /// `mxINT32_CLASS` (12).
    Int32,
    /// `mxUINT32_CLASS` (13).
    Uint32,
    /// `mxINT64_CLASS` (14).
    Int64,
    /// `mxUINT64_CLASS` (15).
    Uint64,
    /// `mxFUNCTION_CLASS` (16): a function handle.
    Function,
    /// `mxOPAQUE_CLASS` (17): the workspace of an anonymous function.
    Opaque,
}

const MAT_CLASSES: [MatClass; 17] = [
    MatClass::Cell,
    MatClass::Struct,
    MatClass::Object,
    MatClass::Char,
    MatClass::Sparse,
    MatClass::Double,
    MatClass::Single,
    MatClass::Int8,
    MatClass::Uint8,
    MatClass::Int16,
    MatClass::Uint16,
    MatClass::Int32,
    MatClass::Uint32,
    MatClass::Int64,
    MatClass::Uint64,
    MatClass::Function,
    MatClass::Opaque,
];

impl MatClass {
    /// The class code stored in the array flags (`mxCELL_CLASS` = 1 … `mxOPAQUE_CLASS` = 17).
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::Cell => 1,
            Self::Struct => 2,
            Self::Object => 3,
            Self::Char => 4,
            Self::Sparse => 5,
            Self::Double => 6,
            Self::Single => 7,
            Self::Int8 => 8,
            Self::Uint8 => 9,
            Self::Int16 => 10,
            Self::Uint16 => 11,
            Self::Int32 => 12,
            Self::Uint32 => 13,
            Self::Int64 => 14,
            Self::Uint64 => 15,
            Self::Function => 16,
            Self::Opaque => 17,
        }
    }

    /// The class of a stored code; `None` for the codes SciPy has no reader for (0 and 18 up).
    #[must_use]
    pub fn from_code(code: u8) -> Option<Self> {
        usize::from(code)
            .checked_sub(1)
            .and_then(|index| MAT_CLASSES.get(index).copied())
    }

    /// SciPy's `mclass_info` name, as [`whosmat`] reports it (`"double"`, `"cell"`, …).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cell => "cell",
            Self::Struct => "struct",
            Self::Object => "object",
            Self::Char => "char",
            Self::Sparse => "sparse",
            Self::Double => "double",
            Self::Single => "single",
            Self::Int8 => "int8",
            Self::Uint8 => "uint8",
            Self::Int16 => "int16",
            Self::Uint16 => "uint16",
            Self::Int32 => "int32",
            Self::Uint32 => "uint32",
            Self::Int64 => "int64",
            Self::Uint64 => "uint64",
            Self::Function => "function",
            Self::Opaque => "opaque",
        }
    }

    /// The dtype of a numeric class, which [`LoadmatOptions::mat_dtype`] casts to.
    #[must_use]
    pub const fn numeric_dtype(self) -> Option<MatDtype> {
        match self {
            Self::Double => Some(MatDtype::F64),
            Self::Single => Some(MatDtype::F32),
            Self::Int8 => Some(MatDtype::I8),
            Self::Uint8 => Some(MatDtype::U8),
            Self::Int16 => Some(MatDtype::I16),
            Self::Uint16 => Some(MatDtype::U16),
            Self::Int32 => Some(MatDtype::I32),
            Self::Uint32 => Some(MatDtype::U32),
            Self::Int64 => Some(MatDtype::I64),
            Self::Uint64 => Some(MatDtype::U64),
            _ => None,
        }
    }
}

/// Element type of a [`MatData`] vector; [`MatDtype::name`] is the NumPy dtype SciPy returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatDtype {
    F64,
    F32,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    Bool,
}

impl MatDtype {
    /// NumPy's name for the dtype (`"float64"`, …, `"bool"`).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F64 => "float64",
            Self::F32 => "float32",
            Self::I8 => "int8",
            Self::U8 => "uint8",
            Self::I16 => "int16",
            Self::U16 => "uint16",
            Self::I32 => "int32",
            Self::U32 => "uint32",
            Self::I64 => "int64",
            Self::U64 => "uint64",
            Self::Bool => "bool",
        }
    }

    /// Bytes per element.
    #[must_use]
    pub const fn item_size(self) -> usize {
        match self {
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::I16 | Self::U16 => 2,
            Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }

    /// The class SciPy writes for data of this dtype: `bool` is a `uint8` array with the
    /// logical flag set.
    #[must_use]
    pub const fn class(self) -> MatClass {
        match self {
            Self::F64 => MatClass::Double,
            Self::F32 => MatClass::Single,
            Self::I8 => MatClass::Int8,
            Self::U8 | Self::Bool => MatClass::Uint8,
            Self::I16 => MatClass::Int16,
            Self::U16 => MatClass::Uint16,
            Self::I32 => MatClass::Int32,
            Self::U32 => MatClass::Uint32,
            Self::I64 => MatClass::Int64,
            Self::U64 => MatClass::Uint64,
        }
    }

    /// The `mi*` data type a Level 5 writer stores this dtype as (`bool` is stored as bytes).
    const fn mi_type(self) -> u32 {
        match self {
            Self::F64 => MI_DOUBLE,
            Self::F32 => MI_SINGLE,
            Self::I8 => MI_INT8,
            Self::U8 | Self::Bool => MI_UINT8,
            Self::I16 => MI_INT16,
            Self::U16 => MI_UINT16,
            Self::I32 => MI_INT32,
            Self::U32 => MI_UINT32,
            Self::I64 => MI_INT64,
            Self::U64 => MI_UINT64,
        }
    }

    /// SciPy's `mdtypes_template`: the dtype a numeric data element of type `mdtype` reads as.
    /// The three Unicode storage types read as unsigned integers of their code-unit width.
    const fn from_mi_type(mdtype: u32) -> Option<Self> {
        match mdtype {
            MI_INT8 => Some(Self::I8),
            MI_UINT8 | MI_UTF8 => Some(Self::U8),
            MI_INT16 => Some(Self::I16),
            MI_UINT16 | MI_UTF16 => Some(Self::U16),
            MI_INT32 => Some(Self::I32),
            MI_UINT32 | MI_UTF32 => Some(Self::U32),
            MI_SINGLE => Some(Self::F32),
            MI_DOUBLE => Some(Self::F64),
            MI_INT64 => Some(Self::I64),
            MI_UINT64 => Some(Self::U64),
            _ => None,
        }
    }
}

/// The elements of a MATLAB array, in MATLAB's column-major (Fortran) order.
///
/// The variant is the dtype SciPy returns. By default that is the type the file STORED the data
/// in, not the array's class: MATLAB narrows a `double` array of small integers to `uint8` on
/// disk and SciPy hands back `uint8` (the class stays in [`MatNumeric::class`]). With
/// [`LoadmatOptions::mat_dtype`] it is the class's dtype. `Bool` appears for logical arrays read
/// with `mat_dtype` and for logical sparse data that MATLAB stored as one byte per value.
#[derive(Debug, Clone, PartialEq)]
pub enum MatData {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I8(Vec<i8>),
    U8(Vec<u8>),
    I16(Vec<i16>),
    U16(Vec<u16>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    I64(Vec<i64>),
    U64(Vec<u64>),
    Bool(Vec<bool>),
}

/// Evaluate `$body` with `$v` bound to the vector inside any [`MatData`] variant.
macro_rules! mat_data_each {
    ($data:expr, $v:ident => $body:expr) => {
        match $data {
            MatData::F64($v) => $body,
            MatData::F32($v) => $body,
            MatData::I8($v) => $body,
            MatData::U8($v) => $body,
            MatData::I16($v) => $body,
            MatData::U16($v) => $body,
            MatData::I32($v) => $body,
            MatData::U32($v) => $body,
            MatData::I64($v) => $body,
            MatData::U64($v) => $body,
            MatData::Bool($v) => $body,
        }
    };
}

/// Build a [`MatData`] of the same variant from `$body`, a vector computed from `$v`.
macro_rules! mat_data_map {
    ($data:expr, $v:ident => $body:expr) => {
        match $data {
            MatData::F64($v) => MatData::F64($body),
            MatData::F32($v) => MatData::F32($body),
            MatData::I8($v) => MatData::I8($body),
            MatData::U8($v) => MatData::U8($body),
            MatData::I16($v) => MatData::I16($body),
            MatData::U16($v) => MatData::U16($body),
            MatData::I32($v) => MatData::I32($body),
            MatData::U32($v) => MatData::U32($body),
            MatData::I64($v) => MatData::I64($body),
            MatData::U64($v) => MatData::U64($body),
            MatData::Bool($v) => MatData::Bool($body),
        }
    };
}

/// Every element of `$data` converted to `$t`: integers and bools with `as` (NumPy's wrapping
/// integer casts), floats through `$from_float`.
macro_rules! mat_data_cast_vec {
    ($data:expr, $t:ty, $from_float:path) => {
        match $data {
            MatData::F64(v) => v.iter().map(|&x| $from_float(x)).collect::<Vec<$t>>(),
            MatData::F32(v) => v.iter().map(|&x| $from_float(f64::from(x))).collect(),
            MatData::I8(v) => v.iter().map(|&x| x as $t).collect(),
            MatData::U8(v) => v.iter().map(|&x| x as $t).collect(),
            MatData::I16(v) => v.iter().map(|&x| x as $t).collect(),
            MatData::U16(v) => v.iter().map(|&x| x as $t).collect(),
            MatData::I32(v) => v.iter().map(|&x| x as $t).collect(),
            MatData::U32(v) => v.iter().map(|&x| x as $t).collect(),
            MatData::I64(v) => v.iter().map(|&x| x as $t).collect(),
            MatData::U64(v) => v.iter().map(|&x| x as $t).collect(),
            MatData::Bool(v) => v.iter().map(|&x| u8::from(x) as $t).collect(),
        }
    };
}

// Float → integer conversions as NumPy 2.4 performs them on x86-64 for in-range values
// (truncation toward zero). Out of range and NaN, NumPy's result depends on which of its SIMD
// or scalar loops ran (the same value gave different integers depending on the array's length),
// so no single answer is SciPy's; these give the scalar `cvttsd2si` "integer indefinite" value.
// MATLAB only narrows a class's storage when every value fits, so real files never reach it.
fn numpy_f64_to_i32(x: f64) -> i32 {
    if x > -2_147_483_649.0 && x < 2_147_483_648.0 {
        x as i32
    } else {
        i32::MIN
    }
}

fn numpy_f64_to_i64(x: f64) -> i64 {
    if (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&x) {
        x as i64
    } else {
        i64::MIN
    }
}

fn numpy_f64_to_u64(x: f64) -> u64 {
    if x >= 9_223_372_036_854_775_808.0 {
        if x < 18_446_744_073_709_551_616.0 {
            x as u64
        } else {
            0
        }
    } else {
        numpy_f64_to_i64(x) as u64
    }
}

fn f64_to_i8(x: f64) -> i8 {
    numpy_f64_to_i32(x) as i8
}

fn f64_to_u8(x: f64) -> u8 {
    numpy_f64_to_i32(x) as u8
}

fn f64_to_i16(x: f64) -> i16 {
    numpy_f64_to_i32(x) as i16
}

fn f64_to_u16(x: f64) -> u16 {
    numpy_f64_to_i32(x) as u16
}

fn f64_to_u32(x: f64) -> u32 {
    numpy_f64_to_i64(x) as u32
}

fn f64_to_f32(x: f64) -> f32 {
    x as f32
}

const fn f64_to_f64(x: f64) -> f64 {
    x
}

impl MatData {
    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        mat_data_each!(self, v => v.len())
    }

    /// Whether there are no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The element type.
    #[must_use]
    pub const fn dtype(&self) -> MatDtype {
        match self {
            Self::F64(_) => MatDtype::F64,
            Self::F32(_) => MatDtype::F32,
            Self::I8(_) => MatDtype::I8,
            Self::U8(_) => MatDtype::U8,
            Self::I16(_) => MatDtype::I16,
            Self::U16(_) => MatDtype::U16,
            Self::I32(_) => MatDtype::I32,
            Self::U32(_) => MatDtype::U32,
            Self::I64(_) => MatDtype::I64,
            Self::U64(_) => MatDtype::U64,
            Self::Bool(_) => MatDtype::Bool,
        }
    }

    /// Element `index` as `f64` (64-bit integers round to nearest, `bool` is 0 or 1).
    #[must_use]
    pub fn get_f64(&self, index: usize) -> Option<f64> {
        match self {
            Self::F64(v) => v.get(index).copied(),
            Self::F32(v) => v.get(index).map(|&x| f64::from(x)),
            Self::I8(v) => v.get(index).map(|&x| f64::from(x)),
            Self::U8(v) => v.get(index).map(|&x| f64::from(x)),
            Self::I16(v) => v.get(index).map(|&x| f64::from(x)),
            Self::U16(v) => v.get(index).map(|&x| f64::from(x)),
            Self::I32(v) => v.get(index).map(|&x| f64::from(x)),
            Self::U32(v) => v.get(index).map(|&x| f64::from(x)),
            Self::I64(v) => v.get(index).map(|&x| x as f64),
            Self::U64(v) => v.get(index).map(|&x| x as f64),
            Self::Bool(v) => v.get(index).map(|&x| f64::from(u8::from(x))),
        }
    }

    /// A copy converted to `dtype` as NumPy's `astype` converts: integers wrap, floats truncate
    /// toward zero, integer → float rounds to nearest, and anything → `bool` is "nonzero".
    /// A float outside the target's range or NaN has no single NumPy answer (see
    /// `numpy_f64_to_i32`); MATLAB never stores one where a cast would meet it.
    #[must_use]
    #[allow(clippy::unnecessary_cast)] // the macro casts every variant, including the target's own
    pub fn cast(&self, dtype: MatDtype) -> Self {
        match dtype {
            MatDtype::F64 => Self::F64(mat_data_cast_vec!(self, f64, f64_to_f64)),
            MatDtype::F32 => Self::F32(mat_data_cast_vec!(self, f32, f64_to_f32)),
            MatDtype::I8 => Self::I8(mat_data_cast_vec!(self, i8, f64_to_i8)),
            MatDtype::U8 => Self::U8(mat_data_cast_vec!(self, u8, f64_to_u8)),
            MatDtype::I16 => Self::I16(mat_data_cast_vec!(self, i16, f64_to_i16)),
            MatDtype::U16 => Self::U16(mat_data_cast_vec!(self, u16, f64_to_u16)),
            MatDtype::I32 => Self::I32(mat_data_cast_vec!(self, i32, numpy_f64_to_i32)),
            MatDtype::U32 => Self::U32(mat_data_cast_vec!(self, u32, f64_to_u32)),
            MatDtype::I64 => Self::I64(mat_data_cast_vec!(self, i64, numpy_f64_to_i64)),
            MatDtype::U64 => Self::U64(mat_data_cast_vec!(self, u64, numpy_f64_to_u64)),
            MatDtype::Bool => Self::Bool(match self {
                Self::F64(v) => v.iter().map(|&x| x != 0.0).collect(),
                Self::F32(v) => v.iter().map(|&x| x != 0.0).collect(),
                Self::I8(v) => v.iter().map(|&x| x != 0).collect(),
                Self::U8(v) => v.iter().map(|&x| x != 0).collect(),
                Self::I16(v) => v.iter().map(|&x| x != 0).collect(),
                Self::U16(v) => v.iter().map(|&x| x != 0).collect(),
                Self::I32(v) => v.iter().map(|&x| x != 0).collect(),
                Self::U32(v) => v.iter().map(|&x| x != 0).collect(),
                Self::I64(v) => v.iter().map(|&x| x != 0).collect(),
                Self::U64(v) => v.iter().map(|&x| x != 0).collect(),
                Self::Bool(v) => v.clone(),
            }),
        }
    }

    #[allow(clippy::unnecessary_cast)]
    fn to_f64_vec(&self) -> Vec<f64> {
        mat_data_cast_vec!(self, f64, f64_to_f64)
    }

    /// The first `n` elements (all of them when there are fewer).
    fn truncated(&self, n: usize) -> Self {
        mat_data_map!(self, v => v[..n.min(v.len())].to_vec())
    }

    /// NumPy broadcasting of a one-element vector to `n` elements; any other length is kept.
    fn broadcast(&self, n: usize) -> Self {
        mat_data_map!(self, v => if v.len() == 1 { vec![v[0]; n] } else { v.clone() })
    }

    /// The elements at `order`, in that order.
    fn gather(&self, order: &[usize]) -> Self {
        mat_data_map!(self, v => order.iter().map(|&k| v[k]).collect())
    }

    /// Little-endian bytes as a writer stores them (`bool` as one byte, 0 or 1).
    fn le_bytes(&self) -> Vec<u8> {
        match self {
            Self::F64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::I8(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::U8(v) => v.clone(),
            Self::I16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::U16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::U32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::U64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::Bool(v) => v.iter().map(|&x| u8::from(x)).collect(),
        }
    }
}

/// Decode a stored data element: `bytes.len() / item_size` values (a partial trailing element is
/// dropped, as SciPy's `byte_count // itemsize` drops it).
fn decode_mat_data(dtype: MatDtype, bytes: &[u8], big_endian: bool) -> MatData {
    macro_rules! decode {
        ($variant:ident, $t:ty, $n:literal) => {
            MatData::$variant(
                bytes
                    .as_chunks::<$n>()
                    .0
                    .iter()
                    .map(|&chunk| {
                        if big_endian {
                            <$t>::from_be_bytes(chunk)
                        } else {
                            <$t>::from_le_bytes(chunk)
                        }
                    })
                    .collect(),
            )
        };
    }
    match dtype {
        MatDtype::F64 => decode!(F64, f64, 8),
        MatDtype::F32 => decode!(F32, f32, 4),
        MatDtype::I8 => decode!(I8, i8, 1),
        MatDtype::U8 => MatData::U8(bytes.to_vec()),
        MatDtype::I16 => decode!(I16, i16, 2),
        MatDtype::U16 => decode!(U16, u16, 2),
        MatDtype::I32 => decode!(I32, i32, 4),
        MatDtype::U32 => decode!(U32, u32, 4),
        MatDtype::I64 => decode!(I64, i64, 8),
        MatDtype::U64 => decode!(U64, u64, 8),
        MatDtype::Bool => MatData::Bool(bytes.iter().map(|&b| b != 0).collect()),
    }
}

/// A dense numeric (or logical) MATLAB array.
#[derive(Debug, Clone, PartialEq)]
pub struct MatNumeric {
    /// MATLAB dimensions (at least two in a file; fewer after [`LoadmatOptions::squeeze_me`]).
    pub dims: Vec<usize>,
    /// The array's class; the data may be stored in a narrower type (see [`MatData`]).
    pub class: MatClass,
    /// MATLAB's logical flag: SciPy's `bool` dtype under `mat_dtype`, `whosmat`'s `"logical"`.
    pub logical: bool,
    /// Real part, column-major, `product(dims)` elements.
    pub real: MatData,
    /// Imaginary part for complex arrays. SciPy returns `complex64` when the stored real part
    /// was 4 bytes wide and `complex128` otherwise, so both parts are `F32` or both are `F64`.
    pub imag: Option<MatData>,
}

impl MatNumeric {
    /// A real array of `real`'s dtype, with the class SciPy writes for that dtype (`bool` data
    /// is a logical `uint8` array).
    #[must_use]
    pub fn new(dims: Vec<usize>, real: MatData) -> Self {
        let dtype = real.dtype();
        Self {
            dims,
            class: dtype.class(),
            logical: dtype == MatDtype::Bool,
            real,
            imag: None,
        }
    }

    /// A complex array with the class of `real`'s dtype.
    #[must_use]
    pub fn complex(dims: Vec<usize>, real: MatData, imag: MatData) -> Self {
        Self {
            imag: Some(imag),
            ..Self::new(dims, real)
        }
    }

    /// A 2-D `double` array from row-major values, the layout the rest of fsci-io uses.
    ///
    /// # Errors
    /// `IoError::InvalidFormat` when `values.len() != rows * cols`.
    pub fn from_row_major(rows: usize, cols: usize, values: &[f64]) -> Result<Self, IoError> {
        let count = checked_matrix_len(rows, cols, "MAT array")?;
        if values.len() != count {
            return Err(IoError::InvalidFormat(format!(
                "a {rows}x{cols} MAT array needs {count} values, got {}",
                values.len()
            )));
        }
        let mut column_major = vec![0.0; count];
        for r in 0..rows {
            for c in 0..cols {
                column_major[c * rows + r] = values[r * cols + c];
            }
        }
        Ok(Self::new(vec![rows, cols], MatData::F64(column_major)))
    }

    /// A real 2-D array's `(rows, cols, values)` widened to `f64`, in row-major order.
    ///
    /// # Errors
    /// `IoError::UnsupportedFeature` for a complex array or one that is not 2-D.
    pub fn to_row_major_f64(&self) -> Result<(usize, usize, Vec<f64>), IoError> {
        if self.imag.is_some() {
            return Err(IoError::UnsupportedFeature(
                "a complex MAT array has no real row-major form".to_string(),
            ));
        }
        let [rows, cols] = self.dims[..] else {
            return Err(IoError::UnsupportedFeature(format!(
                "a MAT array with dimensions {:?} is not a 2-D matrix",
                self.dims
            )));
        };
        let column_major = self.real.to_f64_vec();
        if column_major.len() != rows * cols {
            return Err(IoError::InvalidFormat(format!(
                "a {rows}x{cols} MAT array holds {} values",
                column_major.len()
            )));
        }
        let mut row_major = vec![0.0; rows * cols];
        for c in 0..cols {
            for r in 0..rows {
                row_major[r * cols + c] = column_major[c * rows + r];
            }
        }
        Ok((rows, cols, row_major))
    }

    /// SciPy's `mat_dtype` cast: to the class's dtype, or `bool` for a logical array. A complex
    /// array keeps only its real part (NumPy's `astype` to a real dtype drops the imaginary part,
    /// and SciPy does exactly that), except that `bool` tests both parts for nonzero.
    fn into_class_dtype(self) -> Self {
        let target = if self.logical {
            Some(MatDtype::Bool)
        } else {
            self.class.numeric_dtype()
        };
        let Some(target) = target else {
            return self;
        };
        let real = match (&self.imag, target) {
            (Some(imag), MatDtype::Bool) => MatData::Bool(
                (0..self.real.len())
                    .map(|i| {
                        self.real.get_f64(i).is_some_and(|x| x != 0.0)
                            || imag.get_f64(i).is_some_and(|x| x != 0.0)
                    })
                    .collect(),
            ),
            _ => self.real.cast(target),
        };
        Self {
            dims: self.dims,
            class: self.class,
            logical: self.logical,
            real,
            imag: None,
        }
    }
}

/// A MATLAB char array, one Unicode character per element (SciPy's `U1` array, as returned
/// with `chars_as_strings = false`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatChar {
    /// MATLAB dimensions.
    pub dims: Vec<usize>,
    /// Characters in column-major order.
    pub chars: Vec<char>,
}

impl MatChar {
    /// A 1×n char row holding `text`.
    #[must_use]
    pub fn row(text: &str) -> Self {
        let chars: Vec<char> = text.chars().collect();
        Self {
            dims: vec![1, chars.len()],
            chars,
        }
    }

    /// SciPy's `chars_to_strings`: the last dimension becomes the characters of each string.
    fn into_strings(self) -> MatStrings {
        let Some((&width, prefix)) = self.dims.split_last() else {
            // A char array with no dimensions at all holds one character.
            return MatStrings {
                dims: Vec::new(),
                width: 1,
                strings: self
                    .chars
                    .iter()
                    .map(|c| c.to_string().trim_end_matches('\0').to_string())
                    .collect(),
            };
        };
        if width == 0 {
            let mut dims = prefix[..prefix.len().saturating_sub(1)].to_vec();
            dims.push(0);
            return MatStrings {
                dims,
                width: 1,
                strings: Vec::new(),
            };
        }
        let count = self.chars.len() / width;
        let strings = (0..count)
            .map(|i| {
                let text: String = (0..width).map(|j| self.chars[i + count * j]).collect();
                text.trim_end_matches('\0').to_string()
            })
            .collect();
        MatStrings {
            dims: prefix.to_vec(),
            width,
            strings,
        }
    }
}

/// A char array read with `chars_as_strings` (SciPy's default): an array of strings over all
/// but the last MATLAB dimension, each holding the characters along the last one. Trailing NUL
/// characters are not part of a string, as in NumPy's `U` dtype.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatStrings {
    /// Dimensions of the string array (the char array's without its last).
    pub dims: Vec<usize>,
    /// Characters per string: the char array's last dimension (`U<width>` in NumPy; 1 when that
    /// dimension is 0).
    pub width: usize,
    /// Strings in column-major order.
    pub strings: Vec<String>,
}

/// A MATLAB cell array.
#[derive(Debug, Clone, PartialEq)]
pub struct MatCell {
    /// MATLAB dimensions.
    pub dims: Vec<usize>,
    /// Elements in column-major order.
    pub items: Vec<MatValue>,
}

/// A MATLAB struct array (SciPy's `mat_struct`, or a record array under the default
/// `struct_as_record`: both hold these dimensions, field names and values).
#[derive(Debug, Clone, PartialEq)]
pub struct MatStruct {
    /// MATLAB dimensions.
    pub dims: Vec<usize>,
    /// Field names in file order. Repeated names are renamed as SciPy renames them: the second
    /// `x` is `_1_x`, the third `_2_x`.
    pub field_names: Vec<String>,
    /// Field values element by element (column-major), fields in `field_names` order within an
    /// element: element `e`'s field `f` is `values[e * field_names.len() + f]`.
    pub values: Vec<MatValue>,
}

impl MatStruct {
    /// Field `name` of element `element` (column-major index).
    #[must_use]
    pub fn field(&self, element: usize, name: &str) -> Option<&MatValue> {
        let index = self.field_names.iter().position(|f| f == name)?;
        self.values.get(
            element
                .checked_mul(self.field_names.len())?
                .checked_add(index)?,
        )
    }
}

/// A MATLAB object (SciPy's `MatlabObject`): a struct array with a class name.
#[derive(Debug, Clone, PartialEq)]
pub struct MatObject {
    pub class_name: String,
    pub fields: MatStruct,
}

/// A MATLAB sparse matrix in compressed-sparse-column form (SciPy's `csc_matrix`; a Level 4
/// sparse matrix, which SciPy returns as COO, is given in canonical CSC: row indices sorted
/// within each column and duplicate entries summed).
#[derive(Debug, Clone, PartialEq)]
pub struct MatSparse {
    pub rows: usize,
    pub cols: usize,
    /// MATLAB's logical flag.
    pub logical: bool,
    /// Column pointers, `cols + 1` of them.
    pub indptr: Vec<usize>,
    /// Row index of each stored entry.
    pub indices: Vec<usize>,
    /// Stored values (real part), in the dtype SciPy returns.
    pub data: MatData,
    /// Imaginary parts of a complex matrix.
    pub imag: Option<MatData>,
}

/// An anonymous function's workspace (SciPy's `MatlabOpaque`): three `int8` strings and an array.
#[derive(Debug, Clone, PartialEq)]
pub struct MatOpaque {
    pub s0: Vec<u8>,
    pub s1: Vec<u8>,
    pub s2: Vec<u8>,
    pub arr: Box<MatValue>,
}

/// One MATLAB value, as [`loadmat`] returns it and [`savemat`] writes it.
#[derive(Debug, Clone, PartialEq)]
pub enum MatValue {
    Numeric(MatNumeric),
    /// A char array read with `chars_as_strings = false`, or one to write as MATLAB chars.
    Char(MatChar),
    /// A char array read with `chars_as_strings = true` (the default), or a NumPy-style string
    /// array to write as SciPy writes one.
    Strings(MatStrings),
    Cell(MatCell),
    Struct(MatStruct),
    Object(MatObject),
    Sparse(MatSparse),
    /// A function handle (SciPy's `MatlabFunction`) wrapping the struct MATLAB stores for it.
    Function(Box<MatValue>),
    Opaque(MatOpaque),
}

impl MatValue {
    /// SciPy's class name for the value (`whosmat`'s names, `"logical"` for logical arrays).
    #[must_use]
    pub const fn class_name(&self) -> &'static str {
        match self {
            Self::Numeric(v) if v.logical => "logical",
            Self::Numeric(v) => v.class.name(),
            Self::Char(_) | Self::Strings(_) => "char",
            Self::Cell(_) => "cell",
            Self::Struct(_) => "struct",
            Self::Object(_) => "object",
            Self::Sparse(v) if v.logical => "logical",
            Self::Sparse(_) => "sparse",
            Self::Function(_) => "function",
            Self::Opaque(_) => "opaque",
        }
    }

    /// SciPy's `squeeze_element`: an empty array becomes 1-D of length 0, other arrays lose
    /// their unit dimensions, and a one-element cell becomes its element (NumPy's `.item()` of
    /// a 0-d object array).
    fn squeezed(self) -> Self {
        fn squeeze(dims: &mut Vec<usize>) {
            if dims.contains(&0) {
                *dims = vec![0];
            } else {
                dims.retain(|&d| d != 1);
            }
        }
        match self {
            Self::Numeric(mut v) => {
                squeeze(&mut v.dims);
                Self::Numeric(v)
            }
            Self::Char(mut v) => {
                squeeze(&mut v.dims);
                Self::Char(v)
            }
            Self::Strings(mut v) => {
                squeeze(&mut v.dims);
                Self::Strings(v)
            }
            Self::Struct(mut v) => {
                squeeze(&mut v.dims);
                Self::Struct(v)
            }
            Self::Object(mut v) => {
                squeeze(&mut v.fields.dims);
                Self::Object(v)
            }
            Self::Cell(mut v) => {
                squeeze(&mut v.dims);
                if v.dims.is_empty()
                    && v.items.len() == 1
                    && let Some(item) = v.items.pop()
                {
                    item
                } else {
                    Self::Cell(v)
                }
            }
            other => other,
        }
    }
}

/// SciPy's `__header__`, `__version__` and `__globals__` entries of a Level 5 file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatHeader {
    /// The 116-byte description with leading and trailing spaces, tabs, newlines and NULs removed.
    pub text: Vec<u8>,
    /// The version word as `"major.minor"` (`"1.0"`).
    pub version: String,
    /// Names of the variables stored with the global flag, in read order.
    pub globals: Vec<String>,
}

/// What [`loadmat`] returns: SciPy's result dict, in file order.
#[derive(Debug, Clone, PartialEq)]
pub struct MatFile {
    /// [`matfile_version`] of the file: `(0, 0)` for Level 4, `(1, 0)` for Level 5.
    pub version: (u8, u8),
    /// The file is big-endian. SciPy then returns `>`-ordered dtypes; the values are the same.
    pub big_endian: bool,
    /// Level 5 files only.
    pub header: Option<MatHeader>,
    /// Variables in file order. A name stored twice keeps its first position and its last value,
    /// as SciPy's dict does. An unnamed Level 5 variable is `"__function_workspace__"`.
    pub variables: Vec<(String, MatValue)>,
}

impl MatFile {
    /// The variable called `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&MatValue> {
        self.variables
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, value)| value)
    }
}

/// One entry of [`whosmat`]: SciPy's `(name, shape, class)` tuple.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatInfo {
    pub name: String,
    pub shape: Vec<usize>,
    /// `"double"`, `"logical"`, `"cell"`, … (SciPy's `mclass_info`), or `"unknown"`.
    pub class_name: String,
}

/// Options of [`loadmat`] and [`whosmat`], with SciPy's defaults.
///
/// Not carried over from SciPy: `struct_as_record` (fsci has one struct representation,
/// [`MatStruct`], holding what either SciPy layout holds: the dimensions, the field names in file
/// order and every element's values; `simplify_cells` still applies SciPy's non-record
/// conversions), `matlab_compatible` (it is `squeeze_me = false, chars_as_strings = false,
/// mat_dtype = true`), `byte_order` (the file's own byte order is used, as SciPy does by
/// default), `uint16_codec` (SciPy's default, UTF-8, is the codec), and `appendmat` /
/// `spmatrix` (file-name and Python-type concerns).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadmatOptions {
    /// Return numeric arrays in the dtype of their MATLAB class (`bool` for logical arrays)
    /// instead of the type the file stored them in. Default `false`.
    pub mat_dtype: bool,
    /// Remove unit dimensions: a 1×1 array becomes 0-d and a 1×1 cell becomes its element.
    /// Default `false`.
    pub squeeze_me: bool,
    /// Return char arrays as string arrays ([`MatValue::Strings`]). Default `true`.
    pub chars_as_strings: bool,
    /// SciPy's `simplify_cells`: implies `squeeze_me`, turns a 1×1 object into a plain struct,
    /// and turns 1-D struct arrays and 1-D cells whose first element is a struct into 1-D cells
    /// of 0-d structs (SciPy's lists of dicts). Level 5 files only, as in SciPy. Default `false`.
    pub simplify_cells: bool,
    /// Reject a compressed variable whose decompressed stream holds more than the variable, or
    /// whose compressed bytes the file does not fully contain. Default `true`.
    pub verify_compressed_data_integrity: bool,
    /// Read only these variables (all when `None`).
    pub variable_names: Option<Vec<String>>,
}

impl Default for LoadmatOptions {
    fn default() -> Self {
        Self {
            mat_dtype: false,
            squeeze_me: false,
            chars_as_strings: true,
            simplify_cells: false,
            verify_compressed_data_integrity: true,
            variable_names: None,
        }
    }
}

/// The file format [`savemat`] writes (SciPy's `format`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatFormat {
    /// MATLAB 4 (`format='4'`): 2-D numeric, char and sparse arrays only.
    V4,
    /// MATLAB 5 and up to 7.2 (`format='5'`, SciPy's default).
    #[default]
    V5,
}

/// How [`savemat`] writes an array with fewer than two dimensions (SciPy's `oned_as`): a 0-d
/// value is 1×1, an empty 1-D array 0×0 and a 1-D array of n elements 1×n (`Row`, SciPy's
/// default) or n×1 (`Column`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnedAs {
    #[default]
    Row,
    Column,
}

/// Options of [`savemat`], with SciPy's defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SavematOptions {
    pub format: MatFormat,
    /// Allow struct field names of up to 63 characters instead of 31 (MATLAB 7.6+).
    pub long_field_names: bool,
    /// Compress each Level 5 variable into an `miCOMPRESSED` element.
    pub do_compression: bool,
    pub oned_as: OnedAs,
}

// Level 5 data-element types (`mi*`).
const MI_INT8: u32 = 1;
const MI_UINT8: u32 = 2;
const MI_INT16: u32 = 3;
const MI_UINT16: u32 = 4;
const MI_INT32: u32 = 5;
const MI_UINT32: u32 = 6;
const MI_SINGLE: u32 = 7;
const MI_DOUBLE: u32 = 9;
const MI_INT64: u32 = 12;
const MI_UINT64: u32 = 13;
const MI_MATRIX: u32 = 14;
const MI_COMPRESSED: u32 = 15;
const MI_UTF8: u32 = 16;
const MI_UTF16: u32 = 17;
const MI_UTF32: u32 = 18;

/// Largest dimensions element SciPy reads: `_MAT_MAXDIMS` = 32 `int32` values.
const MAT5_MAX_DIMS_BYTES: usize = 128;
/// Bound on arrays whose size comes from a header alone, so a forged header cannot demand an
/// allocation the file does not pay for (a char array stored with zero bytes is that many spaces,
/// and a Level 4 sparse matrix's column pointers are sized by its stated column count).
const MAT_MAX_HEADER_ELEMENTS: usize = 1 << 28;
/// Bound on one decompressed variable: its `miMATRIX` tag counts bytes in a `u32`.
const MAT5_MAX_INFLATED_BYTES: usize = (u32::MAX as usize).saturating_add(16);
/// Deepest nesting of cells, structs and function handles read or written. Every level is a
/// recursive call, so this keeps a forged file from exhausting the stack; MATLAB data does not
/// nest anywhere near this deep.
const MAT_MAX_NESTING: usize = 100;

fn mat_nesting_error() -> IoError {
    IoError::InvalidFormat(format!(
        "MAT arrays nest deeper than {MAT_MAX_NESTING} levels"
    ))
}

const fn padding8(len: usize) -> usize {
    (8 - len % 8) % 8
}

fn element_count(dims: &[usize]) -> Result<usize, IoError> {
    dims.iter().try_fold(1usize, |count, &d| {
        count.checked_mul(d).ok_or_else(|| {
            IoError::InvalidFormat(format!(
                "MAT dimensions {dims:?} overflow the element count"
            ))
        })
    })
}

fn latin1_string(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| char::from(b)).collect()
}

fn latin1_bytes(text: &str) -> Result<Vec<u8>, IoError> {
    text.chars()
        .map(|ch| {
            u8::try_from(u32::from(ch)).map_err(|_| {
                IoError::InvalidFormat(format!(
                    "'latin-1' codec can't encode character {ch:?} in MAT name {text:?}"
                ))
            })
        })
        .collect()
}

fn utf8_name(bytes: &[u8], what: &str) -> Result<String, IoError> {
    String::from_utf8(bytes.to_vec())
        .map_err(|e| IoError::InvalidFormat(format!("MAT {what} is not valid UTF-8: {e}")))
}

fn mat73_unsupported() -> IoError {
    IoError::UnsupportedFeature(
        "Please use HDF reader for matlab v7.3 files, e.g. h5py".to_string(),
    )
}

/// A read position in a MAT byte stream (the file, or one decompressed variable) in the file's
/// byte order. Reading past the end fails like SciPy's `could not read bytes`; forward skips
/// (element padding) stop at the end silently, as SciPy's seeks do.
struct MatStream<'a> {
    data: &'a [u8],
    pos: usize,
    big_endian: bool,
}

/// The header of a Level 5 matrix (SciPy's `VarHeader5`).
struct Mat5Header<'a> {
    class_code: u8,
    logical: bool,
    global: bool,
    complex: bool,
    /// `None` for an opaque object, which stores neither dimensions nor a name.
    dims: Option<Vec<i32>>,
    name: Option<&'a [u8]>,
}

impl<'a> MatStream<'a> {
    const fn new(data: &'a [u8], big_endian: bool) -> Self {
        Self {
            data,
            pos: 0,
            big_endian,
        }
    }

    const fn at_end(&self) -> bool {
        self.pos >= self.data.len()
    }

    fn read(&mut self, n: usize) -> Result<&'a [u8], IoError> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.data.len())
            .ok_or_else(|| {
                IoError::InvalidFormat(format!(
                    "could not read bytes: {n} needed at offset {} of a {}-byte MAT stream",
                    self.pos,
                    self.data.len()
                ))
            })?;
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn skip(&mut self, n: usize) {
        self.pos = self.pos.saturating_add(n).min(self.data.len());
    }

    const fn word(&self, bytes: [u8; 4]) -> u32 {
        if self.big_endian {
            u32::from_be_bytes(bytes)
        } else {
            u32::from_le_bytes(bytes)
        }
    }

    const fn half(&self, bytes: [u8; 2]) -> u16 {
        if self.big_endian {
            u16::from_be_bytes(bytes)
        } else {
            u16::from_le_bytes(bytes)
        }
    }

    fn read_u32(&mut self) -> Result<u32, IoError> {
        let b = self.read(4)?;
        Ok(self.word([b[0], b[1], b[2], b[3]]))
    }

    /// Two words, as SciPy's `read_full_tag` reads them (no small-element decoding).
    fn read_full_tag(&mut self) -> Result<(u32, u32), IoError> {
        Ok((self.read_u32()?, self.read_u32()?))
    }

    /// SciPy's `read_element`: a data element's type and payload. A small data element (size in
    /// the upper half of the first word, at most 4) carries its payload in the tag; a full
    /// element's payload is followed by padding to 8 bytes, which may be cut off by the end of
    /// the stream.
    fn read_element(&mut self) -> Result<(u32, &'a [u8]), IoError> {
        let tag = self.read(8)?;
        let word = self.word([tag[0], tag[1], tag[2], tag[3]]);
        let small = (word >> 16) as usize;
        if small != 0 {
            if small > 4 {
                return Err(IoError::InvalidFormat(format!(
                    "Error in SDE format data: a small data element claims {small} bytes"
                )));
            }
            return Ok((word & 0xffff, &tag[4..4 + small]));
        }
        let count = self.word([tag[4], tag[5], tag[6], tag[7]]) as usize;
        let payload = self.read(count)?;
        self.skip(padding8(count));
        Ok((word, payload))
    }

    /// SciPy's `read_element_into`: a full element may hold at most `max` bytes.
    fn read_element_bounded(&mut self, max: usize) -> Result<(u32, &'a [u8]), IoError> {
        let start = self.pos;
        let tag = self.read(8)?;
        let word = self.word([tag[0], tag[1], tag[2], tag[3]]);
        if word >> 16 == 0 && self.word([tag[4], tag[5], tag[6], tag[7]]) as usize > max {
            return Err(IoError::InvalidFormat(
                "Unexpected amount of data to read (malformed input file?)".to_string(),
            ));
        }
        self.pos = start;
        self.read_element()
    }

    /// SciPy's `read_into_int32s`: `miINT32` values, or `miUINT32` values that are all below 2³¹.
    fn read_int32s(&mut self, max: usize) -> Result<Vec<i32>, IoError> {
        let (mdtype, bytes) = self.read_element_bounded(max)?;
        let unsigned = match mdtype {
            MI_INT32 => false,
            MI_UINT32 => true,
            other => {
                return Err(IoError::InvalidFormat(format!(
                    "Expecting miINT32 as data type, got {other}"
                )));
            }
        };
        let values: Vec<i32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| self.word(chunk) as i32)
            .collect();
        if unsigned && values.iter().any(|&v| v < 0) {
            return Err(IoError::InvalidFormat(
                "Expecting miINT32, got miUINT32 with negative values".to_string(),
            ));
        }
        Ok(values)
    }

    /// SciPy's `read_int8_string`: names are `miINT8`, or ASCII-only `miUTF8`.
    fn read_int8_string(&mut self) -> Result<&'a [u8], IoError> {
        let (mdtype, bytes) = self.read_element()?;
        match mdtype {
            MI_INT8 => Ok(bytes),
            MI_UTF8 if bytes.is_ascii() => Ok(bytes),
            MI_UTF8 => Err(IoError::InvalidFormat("Non ascii int8 string".to_string())),
            other => Err(IoError::InvalidFormat(format!(
                "Expecting miINT8 as data type, got {other}"
            ))),
        }
    }

    /// SciPy's `read_numeric`: an element in the dtype of its stored type. With `nnz`, a
    /// multi-byte element whose byte count equals `nnz` is `nnz` bytes of logical values: that is
    /// how MATLAB stores logical sparse data under an `miDOUBLE` tag. The flag reports that case.
    fn read_numeric(&mut self, nnz: Option<usize>) -> Result<(MatData, bool), IoError> {
        let (mdtype, bytes) = self.read_element()?;
        let dtype = MatDtype::from_mi_type(mdtype).ok_or_else(|| {
            IoError::InvalidFormat(format!(
                "MAT data element type {mdtype} is not a numeric type"
            ))
        })?;
        if dtype.item_size() != 1 && nnz == Some(bytes.len()) {
            return Ok((MatData::Bool(bytes.iter().map(|&b| b != 0).collect()), true));
        }
        Ok((decode_mat_data(dtype, bytes, self.big_endian), false))
    }

    /// SciPy's `read_header`: the array-flags element (whose tag SciPy does not inspect), then
    /// dimensions and name for every class except opaque.
    fn read_matrix_header(&mut self) -> Result<Mat5Header<'a>, IoError> {
        self.read(8)?;
        let flags = self.read_u32()?;
        self.read_u32()?; // nzmax: sparse readers size from the column pointers instead
        let class_code = (flags & 0xff) as u8;
        let mut header = Mat5Header {
            class_code,
            logical: flags >> 9 & 1 == 1,
            global: flags >> 10 & 1 == 1,
            complex: flags >> 11 & 1 == 1,
            dims: None,
            name: None,
        };
        if class_code != MatClass::Opaque.code() {
            header.dims = Some(self.read_int32s(MAT5_MAX_DIMS_BYTES)?);
            header.name = Some(self.read_int8_string()?);
        }
        Ok(header)
    }
}

fn header_dims(header: &Mat5Header<'_>) -> Result<Vec<usize>, IoError> {
    header
        .dims
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|&d| {
            usize::try_from(d).map_err(|_| {
                IoError::InvalidFormat(format!("MAT array has a negative dimension {d}"))
            })
        })
        .collect()
}

fn decode_utf16_lossy(bytes: &[u8], big_endian: bool) -> Vec<char> {
    let (units, rest) = bytes.as_chunks::<2>();
    let mut chars: Vec<char> = char::decode_utf16(units.iter().map(|&unit| {
        if big_endian {
            u16::from_be_bytes(unit)
        } else {
            u16::from_le_bytes(unit)
        }
    }))
    .map(|c| c.unwrap_or(char::REPLACEMENT_CHARACTER))
    .collect();
    if !rest.is_empty() {
        chars.push(char::REPLACEMENT_CHARACTER);
    }
    chars
}

fn decode_utf32_lossy(bytes: &[u8], big_endian: bool) -> Vec<char> {
    let (units, rest) = bytes.as_chunks::<4>();
    let mut chars: Vec<char> = units
        .iter()
        .map(|&unit| {
            let code = if big_endian {
                u32::from_be_bytes(unit)
            } else {
                u32::from_le_bytes(unit)
            };
            char::from_u32(code).unwrap_or(char::REPLACEMENT_CHARACTER)
        })
        .collect();
    if !rest.is_empty() {
        chars.push(char::REPLACEMENT_CHARACTER);
    }
    chars
}

/// SciPy's `read_char`. The stored type picks the decoding, always with replacement of invalid
/// sequences: `miUINT16` code units are narrowed to bytes and decoded as UTF-8 (SciPy's
/// `uint16_codec`, the system default), `miINT8`/`miUINT8` as ASCII, `miUTF8/16/32` as that
/// encoding. An element with no bytes is `product(dims)` spaces; otherwise the decoded text must
/// hold at least that many characters, and the rest is ignored.
fn read_mat5_char(s: &mut MatStream<'_>, header: &Mat5Header<'_>) -> Result<MatChar, IoError> {
    let dims = header_dims(header)?;
    let length = element_count(&dims)?;
    let (mdtype, bytes) = s.read_element()?;
    if bytes.is_empty() {
        if length > MAT_MAX_HEADER_ELEMENTS {
            return Err(IoError::InvalidFormat(format!(
                "empty char data stands for {length} characters, above the {MAT_MAX_HEADER_ELEMENTS}-element bound"
            )));
        }
        return Ok(MatChar {
            dims,
            chars: vec![' '; length],
        });
    }
    let mut chars: Vec<char> = match mdtype {
        MI_UINT16 => {
            let need = length
                .checked_mul(2)
                .filter(|&need| need <= bytes.len())
                .ok_or_else(|| {
                    IoError::InvalidFormat(format!(
                        "buffer is too small: {} bytes of uint16 char data for {length} characters",
                        bytes.len()
                    ))
                })?;
            let narrowed: Vec<u8> = bytes[..need]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&unit| s.half(unit) as u8)
                .collect();
            String::from_utf8_lossy(&narrowed).chars().collect()
        }
        MI_INT8 | MI_UINT8 => bytes
            .iter()
            .map(|&b| {
                if b.is_ascii() {
                    char::from(b)
                } else {
                    char::REPLACEMENT_CHARACTER
                }
            })
            .collect(),
        MI_UTF8 => String::from_utf8_lossy(bytes).chars().collect(),
        MI_UTF16 => decode_utf16_lossy(bytes, s.big_endian),
        MI_UTF32 => decode_utf32_lossy(bytes, s.big_endian),
        other => {
            return Err(IoError::InvalidFormat(format!(
                "Type {other} does not appear to be char type"
            )));
        }
    };
    if chars.len() < length {
        return Err(IoError::InvalidFormat(format!(
            "buffer is too small: char data decodes to {} characters, fewer than the {length} its dimensions hold",
            chars.len()
        )));
    }
    chars.truncate(length);
    Ok(MatChar { dims, chars })
}

/// SciPy's `read_real_complex`. A complex array's parts are both `float32` when the stored real
/// part is 4 bytes wide and both `float64` otherwise; a one-element imaginary part broadcasts.
fn read_mat5_numeric(
    s: &mut MatStream<'_>,
    header: &Mat5Header<'_>,
    class: MatClass,
) -> Result<MatNumeric, IoError> {
    let dims = header_dims(header)?;
    let count = element_count(&dims)?;
    let (real, imag) = if header.complex {
        let (real, _) = s.read_numeric(None)?;
        let (imag, _) = s.read_numeric(None)?;
        let parts = if real.dtype().item_size() == 4 {
            MatDtype::F32
        } else {
            MatDtype::F64
        };
        if imag.len() != real.len() && imag.len() != 1 {
            return Err(IoError::InvalidFormat(format!(
                "could not broadcast an imaginary part of {} values into {} real values",
                imag.len(),
                real.len()
            )));
        }
        let n = real.len();
        (real.cast(parts), Some(imag.cast(parts).broadcast(n)))
    } else {
        (s.read_numeric(None)?.0, None)
    };
    if real.len() != count {
        return Err(IoError::InvalidFormat(format!(
            "cannot reshape array of size {} into shape {dims:?}",
            real.len()
        )));
    }
    Ok(MatNumeric {
        dims,
        class,
        logical: header.logical,
        real,
        imag,
    })
}

/// Stored sparse indices as `usize`; SciPy's constructor casts float indices by truncation.
fn sparse_index_values(data: &MatData, what: &str) -> Result<Vec<usize>, IoError> {
    (0..data.len())
        .map(|i| {
            let value = data.get_f64(i).unwrap_or(f64::NAN).trunc();
            if (0.0..9_007_199_254_740_992.0).contains(&value) {
                Ok(value as usize)
            } else {
                Err(IoError::InvalidFormat(format!(
                    "MAT sparse {what} {value} is not a valid index"
                )))
            }
        })
        .collect()
}

/// SciPy's `read_sparse` and the `csc_array` checks it relies on. A complex matrix's parts are
/// promoted as NumPy promotes `real + imag * 1j`. fsci also rejects what SciPy would turn into
/// an inconsistent matrix: decreasing column pointers and row indices outside the matrix.
fn read_mat5_sparse(s: &mut MatStream<'_>, header: &Mat5Header<'_>) -> Result<MatSparse, IoError> {
    let dims = header_dims(header)?;
    let (row_data, _) = s.read_numeric(None)?;
    let (ptr_data, _) = s.read_numeric(None)?;
    let [rows, cols, ..] = dims[..] else {
        return Err(IoError::InvalidFormat(format!(
            "a sparse array needs 2 dimensions, this one has {}",
            dims.len()
        )));
    };
    let ptr_len = cols
        .checked_add(1)
        .ok_or_else(|| IoError::InvalidFormat("sparse column count overflows".to_string()))?;
    let indptr = sparse_index_values(&ptr_data.truncated(ptr_len), "column pointer")?;
    if indptr.len() != ptr_len {
        return Err(IoError::InvalidFormat(format!(
            "index pointer size {} should be {ptr_len}",
            indptr.len()
        )));
    }
    let nnz = indptr[cols];
    let (data, imag) = if header.complex {
        let (real, real_bytes) = s.read_numeric(Some(nnz))?;
        let (imag, imag_bytes) = s.read_numeric(Some(nnz))?;
        if (real_bytes || imag_bytes) && nnz > 0 {
            return Err(IoError::InvalidFormat(format!(
                "complex sparse data stored as {nnz} bytes of logical values cannot be combined"
            )));
        }
        let parts = if imag.dtype() == MatDtype::F32
            && matches!(
                real.dtype(),
                MatDtype::F32
                    | MatDtype::I8
                    | MatDtype::U8
                    | MatDtype::I16
                    | MatDtype::U16
                    | MatDtype::Bool
            ) {
            MatDtype::F32
        } else {
            MatDtype::F64
        };
        let len = match (real.len(), imag.len()) {
            (a, b) if a == b => a,
            (1, b) => b,
            (a, 1) => a,
            (a, b) => {
                return Err(IoError::InvalidFormat(format!(
                    "operands could not be broadcast together with shapes ({a},) ({b},)"
                )));
            }
        };
        (
            real.cast(parts).broadcast(len),
            Some(imag.cast(parts).broadcast(len)),
        )
    } else if header.logical {
        (s.read_numeric(Some(nnz))?.0, None)
    } else {
        (s.read_numeric(None)?.0, None)
    };
    let stored = row_data.len().min(nnz);
    if data.len().min(nnz) != stored {
        return Err(IoError::InvalidFormat(
            "indices and data should have the same size".to_string(),
        ));
    }
    if stored < nnz {
        return Err(IoError::InvalidFormat(
            "Last value of index pointer should be less than the size of index and data arrays"
                .to_string(),
        ));
    }
    if indptr[0] != 0 {
        return Err(IoError::InvalidFormat(
            "index pointer should start with 0".to_string(),
        ));
    }
    if indptr.windows(2).any(|w| w[0] > w[1]) {
        return Err(IoError::InvalidFormat(
            "sparse column pointers decrease".to_string(),
        ));
    }
    let indices = sparse_index_values(&row_data.truncated(nnz), "row index")?;
    if let Some(&row) = indices.iter().find(|&&row| row >= rows) {
        return Err(IoError::InvalidFormat(format!(
            "sparse row index {row} is outside the {rows}-row matrix"
        )));
    }
    Ok(MatSparse {
        rows,
        cols,
        logical: header.logical,
        indptr,
        indices,
        data: data.truncated(nnz),
        imag: imag.map(|m| m.truncated(nnz)),
    })
}

/// SciPy's `cread_fieldnames`: one name length, then names in fixed-width slots. A name runs to
/// its first NUL even past its slot, as `PyBytes_FromString` reads it; repeated names are
/// renamed `_1_name`, `_2_name`, … after the first occurrence.
fn read_mat5_field_names(s: &mut MatStream<'_>) -> Result<Vec<String>, IoError> {
    let lengths = s.read_int32s(4)?;
    let [name_length] = lengths[..] else {
        return Err(IoError::InvalidFormat(
            "Only one value for namelength".to_string(),
        ));
    };
    let names = s.read_int8_string()?;
    let width = usize::try_from(name_length)
        .ok()
        .filter(|&w| w > 0)
        .ok_or_else(|| {
            IoError::InvalidFormat(format!(
                "struct field name length {name_length} is not positive"
            ))
        })?;
    let count = names.len() / width;
    let mut fields: Vec<String> = Vec::with_capacity(count);
    let mut repeats = vec![0usize; count];
    for i in 0..count {
        let rest = &names[i * width..];
        let raw = &rest[..rest.iter().position(|&b| b == 0).unwrap_or(rest.len())];
        let mut name = utf8_name(raw, "struct field name")?;
        if let Some(first) = fields.iter().position(|f| *f == name) {
            repeats[first] += 1;
            name = format!("_{}_{name}", repeats[first]);
        }
        fields.push(name);
    }
    Ok(fields)
}

/// SciPy's per-file read options, as `VarReader5` holds them.
struct Mat5Reader {
    mat_dtype: bool,
    squeeze: bool,
    chars_as_strings: bool,
}

impl Mat5Reader {
    /// SciPy's `array_from_header`. `process` applies `mat_dtype`, `chars_as_strings` and
    /// `squeeze_me`; sparse matrices, function handles and opaque objects are never processed,
    /// and the unnamed function workspace is read unprocessed.
    fn read_array(
        &self,
        s: &mut MatStream<'_>,
        header: &Mat5Header<'_>,
        process: bool,
        depth: usize,
    ) -> Result<MatValue, IoError> {
        let class = MatClass::from_code(header.class_code).ok_or_else(|| {
            IoError::InvalidFormat(format!(
                "MAT array class code {} has no reader",
                header.class_code
            ))
        })?;
        let mut process = process;
        let value = match class {
            MatClass::Double
            | MatClass::Single
            | MatClass::Int8
            | MatClass::Uint8
            | MatClass::Int16
            | MatClass::Uint16
            | MatClass::Int32
            | MatClass::Uint32
            | MatClass::Int64
            | MatClass::Uint64 => {
                let numeric = read_mat5_numeric(s, header, class)?;
                MatValue::Numeric(if process && self.mat_dtype {
                    numeric.into_class_dtype()
                } else {
                    numeric
                })
            }
            MatClass::Sparse => {
                process = false;
                MatValue::Sparse(read_mat5_sparse(s, header)?)
            }
            MatClass::Char => {
                let chars = read_mat5_char(s, header)?;
                if process && self.chars_as_strings {
                    MatValue::Strings(chars.into_strings())
                } else {
                    MatValue::Char(chars)
                }
            }
            MatClass::Cell => {
                let dims = header_dims(header)?;
                let count = element_count(&dims)?;
                let mut items = Vec::new();
                for _ in 0..count {
                    items.push(self.read_mi_matrix(s, true, depth + 1)?);
                }
                MatValue::Cell(MatCell { dims, items })
            }
            MatClass::Struct => MatValue::Struct(self.read_struct(s, header, depth)?),
            MatClass::Object => {
                let class_name = utf8_name(s.read_int8_string()?, "object class name")?;
                MatValue::Object(MatObject {
                    class_name,
                    fields: self.read_struct(s, header, depth)?,
                })
            }
            MatClass::Function => {
                process = false;
                MatValue::Function(Box::new(self.read_mi_matrix(s, true, depth + 1)?))
            }
            MatClass::Opaque => {
                process = false;
                let s0 = s.read_int8_string()?.to_vec();
                let s1 = s.read_int8_string()?.to_vec();
                let s2 = s.read_int8_string()?.to_vec();
                let arr = Box::new(self.read_mi_matrix(s, true, depth + 1)?);
                MatValue::Opaque(MatOpaque { s0, s1, s2, arr })
            }
        };
        Ok(if process && self.squeeze {
            value.squeezed()
        } else {
            value
        })
    }

    /// SciPy's `read_mi_matrix`: a nested matrix. A zero-length one is an empty `double` array.
    fn read_mi_matrix(
        &self,
        s: &mut MatStream<'_>,
        process: bool,
        depth: usize,
    ) -> Result<MatValue, IoError> {
        if depth > MAT_MAX_NESTING {
            return Err(mat_nesting_error());
        }
        let (mdtype, count) = s.read_full_tag()?;
        if mdtype != MI_MATRIX {
            return Err(IoError::InvalidFormat(format!(
                "Expecting matrix here, got data element type {mdtype}"
            )));
        }
        if count == 0 {
            let dims = if process && self.squeeze {
                vec![0]
            } else {
                vec![1, 0]
            };
            return Ok(MatValue::Numeric(MatNumeric::new(
                dims,
                MatData::F64(Vec::new()),
            )));
        }
        let header = s.read_matrix_header()?;
        self.read_array(s, &header, process, depth)
    }

    fn read_struct(
        &self,
        s: &mut MatStream<'_>,
        header: &Mat5Header<'_>,
        depth: usize,
    ) -> Result<MatStruct, IoError> {
        let field_names = read_mat5_field_names(s)?;
        let dims = header_dims(header)?;
        let count = element_count(&dims)?;
        let mut values = Vec::new();
        if !field_names.is_empty() {
            for _ in 0..count {
                for _ in 0..field_names.len() {
                    values.push(self.read_mi_matrix(s, true, depth + 1)?);
                }
            }
        }
        Ok(MatStruct {
            dims,
            field_names,
            values,
        })
    }
}

/// How a zlib stream ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InflateEnd {
    /// The stream's end marker and checksum were read.
    Complete,
    /// The input ran out first; SciPy's `decompressobj` keeps what it decoded.
    Truncated,
    /// Decoding stopped at the output bound.
    Limited,
}

/// SciPy's `ZlibInputStream`: inflate a zlib stream, keeping what a truncated stream yields
/// (some MATLAB files lack a valid end-of-stream marker) and failing on a checksum mismatch or
/// a corrupt stream as `zlib.error` does.
fn inflate_zlib(input: &[u8], limit: usize) -> Result<(Vec<u8>, InflateEnd), IoError> {
    use miniz_oxide::inflate::TINFLStatus;
    use miniz_oxide::inflate::core::{DecompressorOxide, decompress, inflate_flags};

    let flags = inflate_flags::TINFL_FLAG_PARSE_ZLIB_HEADER
        | inflate_flags::TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF;
    let mut out = vec![0u8; input.len().saturating_mul(4).clamp(64, limit.max(64))];
    let mut decompressor = Box::<DecompressorOxide>::default();
    let (mut in_pos, mut out_pos) = (0usize, 0usize);
    loop {
        let (status, consumed, written) = decompress(
            &mut decompressor,
            &input[in_pos..],
            &mut out,
            out_pos,
            flags,
        );
        in_pos += consumed;
        out_pos += written;
        match status {
            TINFLStatus::Done => {
                out.truncate(out_pos);
                return Ok((out, InflateEnd::Complete));
            }
            TINFLStatus::HasMoreOutput => {
                if out.len() >= limit {
                    out.truncate(out_pos.min(limit));
                    return Ok((out, InflateEnd::Limited));
                }
                let grown = out.len().saturating_mul(2).min(limit);
                out.resize(grown, 0);
            }
            TINFLStatus::FailedCannotMakeProgress | TINFLStatus::NeedsMoreInput => {
                out.truncate(out_pos);
                return Ok((out, InflateEnd::Truncated));
            }
            TINFLStatus::Adler32Mismatch => {
                return Err(IoError::InvalidFormat(
                    "Error -3 while decompressing data: incorrect data check".to_string(),
                ));
            }
            TINFLStatus::Failed | TINFLStatus::BadParam => {
                return Err(IoError::InvalidFormat(format!(
                    "Error while decompressing MAT data: invalid zlib stream ({status:?})"
                )));
            }
        }
    }
}

/// Where a top-level Level 5 variable came from.
struct Mat5Top {
    /// Byte offset of the variable's top-level tag.
    start: usize,
    /// Byte offset where SciPy seeks next: the tag's end plus its byte count, unpadded.
    next: usize,
    /// `Some` for an `miCOMPRESSED` element: how its zlib stream ended, and whether the file held
    /// all the compressed bytes the tag counts.
    compressed: Option<(InflateEnd, bool)>,
}

impl Mat5Top {
    /// SciPy's `verify_compressed_data_integrity` check (`ZlibInputStream.all_data_read`): the
    /// variable consumed its whole decompressed stream, and every compressed byte was there.
    fn check_consumed(&self, s: &MatStream<'_>) -> Result<(), IoError> {
        match self.compressed {
            Some((end, whole_input))
                if !whole_input || end == InflateEnd::Limited || !s.at_end() =>
            {
                Err(IoError::InvalidFormat(
                    "Did not fully consume compressed contents of an miCOMPRESSED element. This \
                     can indicate that the .mat file is corrupted."
                        .to_string(),
                ))
            }
            _ => Ok(()),
        }
    }
}

/// The 128-byte Level 5 file header: byte order, `__header__` text and `__version__`.
fn read_mat5_file_header(bytes: &[u8]) -> Result<(bool, Vec<u8>, String), IoError> {
    let head = bytes.get(..128).ok_or_else(|| {
        IoError::InvalidFormat(format!(
            "MAT 5 file header is truncated: {} of 128 bytes",
            bytes.len()
        ))
    })?;
    // SciPy's `guess_byte_order`: anything but "IM" is big-endian.
    let big_endian = &head[126..128] != b"IM";
    let is_space = |b: &u8| matches!(b, b' ' | b'\t' | b'\n' | 0);
    let description = &head[..116];
    let first = description.iter().position(|b| !is_space(b)).unwrap_or(116);
    let last = description
        .iter()
        .rposition(|b| !is_space(b))
        .map_or(first, |i| i + 1);
    let word = if big_endian {
        u16::from_be_bytes([head[124], head[125]])
    } else {
        u16::from_le_bytes([head[124], head[125]])
    };
    Ok((
        big_endian,
        description[first..last].to_vec(),
        format!("{}.{}", word >> 8, word & 0xff),
    ))
}

/// SciPy's `read_var_header` loop over a Level 5 file: for each top-level element, open its
/// stream (inflating an `miCOMPRESSED` one), read the matrix header, and hand both to `visit`,
/// which returns whether to go on. An uncompressed variable is read straight from the file
/// stream, past its own byte count if its elements say so, as SciPy reads it.
fn for_each_mat5_variable<F>(bytes: &[u8], big_endian: bool, mut visit: F) -> Result<(), IoError>
where
    F: for<'s> FnMut(&mut MatStream<'s>, Mat5Header<'s>, &Mat5Top) -> Result<bool, IoError>,
{
    let mut pos = 128;
    while pos < bytes.len() {
        let mut file = MatStream::new(bytes, big_endian);
        file.pos = pos;
        let (mdtype, count) = file.read_full_tag()?;
        if count == 0 {
            return Err(IoError::InvalidFormat("Did not read any bytes".to_string()));
        }
        let body = file.pos;
        let next = body.saturating_add(count as usize);
        let keep_going = if mdtype == MI_COMPRESSED {
            let end = next.min(bytes.len());
            let (inflated, how) = inflate_zlib(&bytes[body..end], MAT5_MAX_INFLATED_BYTES)?;
            let top = Mat5Top {
                start: pos,
                next,
                compressed: Some((how, next <= bytes.len())),
            };
            let mut stream = MatStream::new(&inflated, big_endian);
            let (inner, _) = stream.read_full_tag()?;
            if inner != MI_MATRIX {
                return Err(IoError::InvalidFormat(format!(
                    "Expecting miMATRIX type here, got {inner}"
                )));
            }
            let header = stream.read_matrix_header()?;
            visit(&mut stream, header, &top)?
        } else {
            if mdtype != MI_MATRIX {
                return Err(IoError::InvalidFormat(format!(
                    "Expecting miMATRIX type here, got {mdtype}"
                )));
            }
            let top = Mat5Top {
                start: pos,
                next,
                compressed: None,
            };
            let header = file.read_matrix_header()?;
            visit(&mut file, header, &top)?
        };
        if !keep_going {
            break;
        }
        pos = next;
    }
    Ok(())
}

/// A top-level variable's name as SciPy keys it: Latin-1, `"None"` without a name.
fn mat5_variable_name(header: &Mat5Header<'_>) -> String {
    header
        .name
        .map_or_else(|| "None".to_string(), latin1_string)
}

fn loadmat_v5(
    bytes: &[u8],
    version: (u8, u8),
    options: &LoadmatOptions,
) -> Result<MatFile, IoError> {
    let (big_endian, text, version_text) = read_mat5_file_header(bytes)?;
    let reader = Mat5Reader {
        mat_dtype: options.mat_dtype,
        squeeze: options.squeeze_me || options.simplify_cells,
        chars_as_strings: options.chars_as_strings,
    };
    let mut wanted = options.variable_names.clone();
    let mut variables: Vec<(String, MatValue)> = Vec::new();
    let mut globals = Vec::new();
    for_each_mat5_variable(bytes, big_endian, |stream, header, top| {
        let mut name = mat5_variable_name(&header);
        // An unnamed matrix can only be MATLAB 7's function workspace; SciPy keeps it raw.
        let process = if name.is_empty() {
            name = "__function_workspace__".to_string();
            false
        } else {
            true
        };
        if let Some(list) = &wanted
            && !list.contains(&name)
        {
            return Ok(true);
        }
        let value = reader.read_array(stream, &header, process, 0)?;
        if options.verify_compressed_data_integrity {
            top.check_consumed(stream)?;
        }
        match variables.iter_mut().find(|(n, _)| *n == name) {
            Some(slot) => slot.1 = value,
            None => variables.push((name.clone(), value)),
        }
        if header.global {
            globals.push(name.clone());
        }
        if let Some(list) = &mut wanted
            && let Some(i) = list.iter().position(|n| *n == name)
        {
            list.remove(i);
            if list.is_empty() {
                return Ok(false);
            }
        }
        Ok(true)
    })?;
    if options.simplify_cells {
        variables = variables
            .into_iter()
            .map(|(name, value)| (name, simplify_mat_value(demote_scalar_objects(value))))
            .collect();
    }
    Ok(MatFile {
        version,
        big_endian,
        header: Some(MatHeader {
            text,
            version: version_text,
            globals,
        }),
        variables,
    })
}

fn whosmat_v5(bytes: &[u8], options: &LoadmatOptions) -> Result<Vec<MatInfo>, IoError> {
    let (big_endian, _, _) = read_mat5_file_header(bytes)?;
    let squeeze = options.squeeze_me || options.simplify_cells;
    let mut out = Vec::new();
    for_each_mat5_variable(bytes, big_endian, |_, header, _| {
        let mut name = mat5_variable_name(&header);
        if name.is_empty() {
            name = "__function_workspace__".to_string();
        }
        if header.dims.is_none() {
            return Err(IoError::InvalidFormat(format!(
                "variable '{name}' is an opaque object, which has no dimensions"
            )));
        }
        let mut shape = header_dims(&header)?;
        if header.class_code == MatClass::Char.code() && options.chars_as_strings {
            shape.pop();
        }
        if squeeze {
            shape.retain(|&d| d != 1);
        }
        let class_name = if header.logical {
            "logical"
        } else {
            MatClass::from_code(header.class_code).map_or("unknown", MatClass::name)
        };
        out.push(MatInfo {
            name,
            shape,
            class_name: class_name.to_string(),
        });
        Ok(true)
    })?;
    Ok(out)
}

// `simplify_cells`. SciPy reads structs as `mat_struct` objects when it is set (it implies
// `struct_as_record=False` and `squeeze_me=True`), then converts the result dict.

/// Squeezing a 1×1 object array read with `struct_as_record=False` yields the bare `mat_struct`
/// (NumPy's `.item()` of a 0-d object array): its class name is gone. That happens at every
/// level, so it is applied to the whole tree.
fn demote_scalar_objects(value: MatValue) -> MatValue {
    fn demote_values(mut fields: MatStruct) -> MatStruct {
        fields.values = fields
            .values
            .into_iter()
            .map(demote_scalar_objects)
            .collect();
        fields
    }
    match value {
        MatValue::Object(object) => {
            let fields = demote_values(object.fields);
            if fields.dims.is_empty() {
                MatValue::Struct(fields)
            } else {
                MatValue::Object(MatObject {
                    class_name: object.class_name,
                    fields,
                })
            }
        }
        MatValue::Struct(fields) => MatValue::Struct(demote_values(fields)),
        MatValue::Cell(mut cell) => {
            cell.items = cell.items.into_iter().map(demote_scalar_objects).collect();
            MatValue::Cell(cell)
        }
        MatValue::Function(inner) => MatValue::Function(Box::new(demote_scalar_objects(*inner))),
        MatValue::Opaque(mut opaque) => {
            opaque.arr = Box::new(demote_scalar_objects(*opaque.arr));
            MatValue::Opaque(opaque)
        }
        other => other,
    }
}

/// SciPy's `mat_struct`: a single struct (0-d after squeezing). `_matstruct_to_dict` makes it a
/// dict, which here stays a 0-d [`MatStruct`].
fn is_mat_struct(value: &MatValue) -> bool {
    matches!(value, MatValue::Struct(s) if s.dims.is_empty())
}

/// SciPy's `_has_struct`: a non-empty 1-D array whose first element is a `mat_struct`.
fn has_mat_struct(value: &MatValue) -> bool {
    match value {
        MatValue::Struct(s) => s.dims.len() == 1 && s.dims[0] > 0,
        MatValue::Object(o) => o.fields.dims.len() == 1 && o.fields.dims[0] > 0,
        MatValue::Cell(c) => c.dims.len() == 1 && c.items.first().is_some_and(is_mat_struct),
        _ => false,
    }
}

/// The per-value rule shared by `_simplify_cells`, `_matstruct_to_dict` and
/// `_inspect_cell_array`: a struct becomes a dict (its fields simplified), an array holding
/// structs becomes a list, anything else is kept as it is.
fn simplify_mat_value(value: MatValue) -> MatValue {
    if is_mat_struct(&value) {
        match value {
            MatValue::Struct(mut s) => {
                s.values = s.values.into_iter().map(simplify_mat_value).collect();
                MatValue::Struct(s)
            }
            other => other,
        }
    } else if has_mat_struct(&value) {
        let items: Vec<MatValue> = match value {
            MatValue::Struct(s) => split_struct_elements(s),
            MatValue::Object(o) => split_struct_elements(o.fields),
            MatValue::Cell(c) => c.items,
            other => return other,
        };
        let items: Vec<MatValue> = items.into_iter().map(simplify_mat_value).collect();
        MatValue::Cell(MatCell {
            dims: vec![items.len()],
            items,
        })
    } else {
        value
    }
}

/// Each element of a struct array as its own 0-d struct.
fn split_struct_elements(s: MatStruct) -> Vec<MatValue> {
    let count = s.dims.iter().product::<usize>();
    let width = s.field_names.len();
    let mut values = s.values.into_iter();
    (0..count)
        .map(|_| {
            MatValue::Struct(MatStruct {
                dims: Vec::new(),
                field_names: s.field_names.clone(),
                values: values.by_ref().take(width).collect(),
            })
        })
        .collect()
}

// Level 4.

/// A Level 4 variable header (SciPy's `VarHeader4`) and where its data lie.
struct Mat4Header<'a> {
    name: &'a [u8],
    dtype: MatDtype,
    class: i32,
    rows: usize,
    cols: usize,
    complex: bool,
    data: usize,
    next: usize,
}

fn read_mat4_header(bytes: &[u8], pos: usize, big_endian: bool) -> Result<Mat4Header<'_>, IoError> {
    let head = bytes.get(pos..pos.saturating_add(20)).ok_or_else(|| {
        IoError::InvalidFormat(format!(
            "MAT 4 variable header at offset {pos} is truncated"
        ))
    })?;
    let int = |i: usize| {
        let b = [
            head[4 * i],
            head[4 * i + 1],
            head[4 * i + 2],
            head[4 * i + 3],
        ];
        if big_endian {
            i32::from_be_bytes(b)
        } else {
            i32::from_le_bytes(b)
        }
    };
    let (mopt, mrows, ncols, imagf, namlen) = (int(0), int(1), int(2), int(3), int(4));
    let namlen = usize::try_from(namlen)
        .map_err(|_| IoError::InvalidFormat(format!("MAT 4 name length {namlen} is negative")))?;
    let name_start = pos + 20;
    let name_end = name_start.saturating_add(namlen).min(bytes.len());
    let raw = &bytes[name_start..name_end];
    let first = raw.iter().position(|&b| b != 0).unwrap_or(raw.len());
    let last = raw.iter().rposition(|&b| b != 0).map_or(first, |i| i + 1);
    if !(0..=5000).contains(&mopt) {
        return Err(IoError::InvalidFormat(
            "Mat 4 mopt wrong format, byteswapping problem?".to_string(),
        ));
    }
    // The thousands digit is the variable's byte order; SciPy warns about the VAX and Cray
    // codes and reads every variable in the order it detected for the file.
    let rest = mopt % 1000;
    if rest / 100 != 0 {
        return Err(IoError::InvalidFormat(
            "O in MOPT integer should be 0, wrong format?".to_string(),
        ));
    }
    let dtype = match rest % 100 / 10 {
        0 => MatDtype::F64,
        1 => MatDtype::F32,
        2 => MatDtype::I32,
        3 => MatDtype::I16,
        4 => MatDtype::U16,
        5 => MatDtype::U8,
        p => {
            return Err(IoError::InvalidFormat(format!(
                "MAT 4 data type code {p} is not one SciPy reads"
            )));
        }
    };
    let class = rest % 10;
    let dim = |value: i32, what: &str| {
        usize::try_from(value)
            .map_err(|_| IoError::InvalidFormat(format!("MAT 4 {what} {value} is negative")))
    };
    let rows = dim(mrows, "row count")?;
    let cols = dim(ncols, "column count")?;
    let complex = imagf == 1;
    let size = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(dtype.item_size()))
        .and_then(|n| n.checked_mul(if complex && class != 2 { 2 } else { 1 }))
        .ok_or_else(|| IoError::InvalidFormat("MAT 4 matrix size overflows".to_string()))?;
    Ok(Mat4Header {
        name: &raw[first..last],
        dtype,
        class,
        rows,
        cols,
        complex,
        data: name_end,
        next: name_end.saturating_add(size),
    })
}

/// SciPy's `read_sub_array`: `rows × cols` values of the stored type, starting at `offset`.
fn mat4_sub_array(
    bytes: &[u8],
    header: &Mat4Header<'_>,
    offset: usize,
    big_endian: bool,
) -> Result<MatData, IoError> {
    let len = header.rows * header.cols * header.dtype.item_size();
    let raw = offset
        .checked_add(len)
        .and_then(|end| bytes.get(offset..end))
        .ok_or_else(|| {
            IoError::InvalidFormat(format!(
                "Not enough bytes to read matrix '{}'; is this a badly-formed file? Consider \
                 listing matrices with `whosmat` and loading named matrices with \
                 `variable_names` kwarg to `loadmat`",
                latin1_string(header.name)
            ))
        })?;
    Ok(decode_mat_data(header.dtype, raw, big_endian))
}

/// SciPy's Level 4 `read_sparse_array`: an (nnz + 1) × 3 (or × 4, complex) table of 1-based
/// row and column indices and values, the last row holding the matrix shape. SciPy builds a COO
/// matrix; this returns its canonical CSC form (what SciPy's `.tocsc()` gives).
fn read_mat4_sparse(
    bytes: &[u8],
    header: &Mat4Header<'_>,
    big_endian: bool,
) -> Result<MatSparse, IoError> {
    let table = mat4_sub_array(bytes, header, header.data, big_endian)?.to_f64_vec();
    let (height, width) = (header.rows, header.cols);
    if height == 0 || width < 3 {
        return Err(IoError::InvalidFormat(format!(
            "MAT 4 sparse data must be (nnz + 1) x 3 or x 4, found {height} x {width}"
        )));
    }
    let at = |row: usize, col: usize| table[row + height * col];
    let nnz = height - 1;
    // The column count sizes the column pointers, so it is bounded like a header-sized array.
    let dimension = |value: f64, bound: f64| {
        let t = value.trunc();
        if (0.0..=bound).contains(&t) {
            Ok(t as usize)
        } else {
            Err(IoError::InvalidFormat(format!(
                "MAT 4 sparse dimension {value} is not a supported size"
            )))
        }
    };
    let rows = dimension(at(nnz, 0), 9_007_199_254_740_992.0)?;
    let cols = dimension(at(nnz, 1), MAT_MAX_HEADER_ELEMENTS as f64)?;
    let mut entries = Vec::with_capacity(nnz);
    for k in 0..nnz {
        let i = i64::from(numpy_f64_to_i32(at(k, 0))) - 1;
        let j = i64::from(numpy_f64_to_i32(at(k, 1))) - 1;
        if i < 0 || j < 0 {
            return Err(IoError::InvalidFormat(
                "negative row index found".to_string(),
            ));
        }
        let (i, j) = (i as usize, j as usize);
        if i >= rows || j >= cols {
            return Err(IoError::InvalidFormat(
                "row index exceeds matrix dimensions".to_string(),
            ));
        }
        entries.push((j, i, at(k, 2), if width > 3 { at(k, 3) } else { 0.0 }));
    }
    entries.sort_by_key(|&(col, row, _, _)| (col, row));
    let mut indptr = vec![0usize; cols + 1];
    let (mut indices, mut real, mut imag) = (Vec::new(), Vec::new(), Vec::new());
    let mut previous = None;
    for (col, row, re, im) in entries {
        if previous == Some((col, row))
            && let (Some(r), Some(m)) = (real.last_mut(), imag.last_mut())
        {
            *r += re;
            *m += im;
            continue;
        }
        previous = Some((col, row));
        indices.push(row);
        real.push(re);
        imag.push(im);
        indptr[col + 1] += 1;
    }
    for c in 0..cols {
        indptr[c + 1] += indptr[c];
    }
    Ok(MatSparse {
        rows,
        cols,
        logical: false,
        indptr,
        indices,
        data: MatData::F64(real),
        imag: (width > 3).then_some(MatData::F64(imag)),
    })
}

fn read_mat4_array(
    bytes: &[u8],
    header: &Mat4Header<'_>,
    big_endian: bool,
    options: &LoadmatOptions,
) -> Result<MatValue, IoError> {
    let dims = vec![header.rows, header.cols];
    let value = match header.class {
        0 => {
            let real = mat4_sub_array(bytes, header, header.data, big_endian)?;
            let numeric = if header.complex {
                let offset = header.data + real.len() * header.dtype.item_size();
                let imag = mat4_sub_array(bytes, header, offset, big_endian)?;
                // NumPy's `real + imag * 1j`: complex64 for float32 parts, complex128 otherwise.
                let parts = if header.dtype == MatDtype::F32 {
                    MatDtype::F32
                } else {
                    MatDtype::F64
                };
                MatNumeric {
                    dims,
                    class: MatClass::Double,
                    logical: false,
                    real: real.cast(parts),
                    imag: Some(imag.cast(parts)),
                }
            } else {
                MatNumeric {
                    dims,
                    class: MatClass::Double,
                    logical: false,
                    real,
                    imag: None,
                }
            };
            MatValue::Numeric(numeric)
        }
        1 => {
            let codes = mat4_sub_array(bytes, header, header.data, big_endian)?.cast(MatDtype::U8);
            let MatData::U8(codes) = codes else {
                return Err(IoError::InvalidFormat(
                    "MAT 4 char codes did not convert to bytes".to_string(),
                ));
            };
            let chars = MatChar {
                dims,
                chars: codes.iter().map(|&b| char::from(b)).collect(),
            };
            if options.chars_as_strings {
                MatValue::Strings(chars.into_strings())
            } else {
                MatValue::Char(chars)
            }
        }
        2 => {
            return Ok(MatValue::Sparse(read_mat4_sparse(
                bytes, header, big_endian,
            )?));
        }
        other => {
            return Err(IoError::InvalidFormat(format!(
                "No reader for class code {other}"
            )));
        }
    };
    Ok(if options.squeeze_me || options.simplify_cells {
        value.squeezed()
    } else {
        value
    })
}

/// SciPy's `MatFile4Reader.guess_byte_order`: a first `mopt` outside 0..=5000 read
/// little-endian means the file is big-endian.
fn mat4_big_endian(bytes: &[u8]) -> bool {
    let mopt = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    !(0..=5000).contains(&mopt)
}

fn loadmat_v4(bytes: &[u8], options: &LoadmatOptions) -> Result<MatFile, IoError> {
    let big_endian = mat4_big_endian(bytes);
    let mut wanted = options.variable_names.clone();
    let mut variables: Vec<(String, MatValue)> = Vec::new();
    let mut pos = 0;
    while pos < bytes.len() {
        let header = read_mat4_header(bytes, pos, big_endian)?;
        pos = header.next;
        let name = latin1_string(header.name);
        if let Some(list) = &wanted
            && !list.contains(&name)
        {
            continue;
        }
        let value = read_mat4_array(bytes, &header, big_endian, options)?;
        match variables.iter_mut().find(|(n, _)| *n == name) {
            Some(slot) => slot.1 = value,
            None => variables.push((name.clone(), value)),
        }
        if let Some(list) = &mut wanted
            && let Some(i) = list.iter().position(|n| *n == name)
        {
            list.remove(i);
            if list.is_empty() {
                break;
            }
        }
    }
    Ok(MatFile {
        version: (0, 0),
        big_endian,
        header: None,
        variables,
    })
}

fn whosmat_v4(bytes: &[u8], options: &LoadmatOptions) -> Result<Vec<MatInfo>, IoError> {
    let big_endian = mat4_big_endian(bytes);
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < bytes.len() {
        let header = read_mat4_header(bytes, pos, big_endian)?;
        pos = header.next;
        let mut shape = match header.class {
            0 => vec![header.rows, header.cols],
            1 => {
                if options.chars_as_strings {
                    vec![header.rows]
                } else {
                    vec![header.rows, header.cols]
                }
            }
            2 if header.rows >= 1 && header.cols >= 1 => {
                // SciPy reads just the shape from the table's last row.
                let size = header.dtype.item_size();
                let value_at = |index: usize| {
                    let offset = header.data + index * size;
                    let raw = bytes.get(offset..offset + size).ok_or_else(|| {
                        IoError::InvalidFormat(format!(
                            "buffer is too small for the shape of sparse matrix '{}'",
                            latin1_string(header.name)
                        ))
                    })?;
                    let value = decode_mat_data(header.dtype, raw, big_endian)
                        .get_f64(0)
                        .unwrap_or(f64::NAN)
                        .trunc();
                    if value >= 0.0 && value.is_finite() {
                        Ok(value as usize)
                    } else {
                        Err(IoError::InvalidFormat(format!(
                            "sparse matrix '{}' has shape entry {value}",
                            latin1_string(header.name)
                        )))
                    }
                };
                vec![value_at(header.rows - 1)?, value_at(2 * header.rows - 1)?]
            }
            2 => Vec::new(),
            other => {
                return Err(IoError::InvalidFormat(format!(
                    "No reader for class code {other}"
                )));
            }
        };
        if options.squeeze_me || options.simplify_cells {
            shape.retain(|&d| d != 1);
        }
        let class_name = match header.class {
            0 => "double",
            1 => "char",
            _ => "sparse",
        };
        out.push(MatInfo {
            name: latin1_string(header.name),
            shape,
            class_name: class_name.to_string(),
        });
    }
    Ok(out)
}

// Writing.

/// SciPy's `matdims`: the MATLAB dimensions of an array with `dims`.
fn matlab_dims(dims: &[usize], oned_as: OnedAs) -> Vec<usize> {
    match *dims {
        [] => vec![1, 1],
        [0] => vec![0, 0],
        [n] => match oned_as {
            OnedAs::Row => vec![1, n],
            OnedAs::Column => vec![n, 1],
        },
        _ => dims.to_vec(),
    }
}

fn check_element_count(len: usize, dims: &[usize], what: &str) -> Result<(), IoError> {
    let expected = element_count(dims)?;
    if len == expected {
        Ok(())
    } else {
        Err(IoError::InvalidFormat(format!(
            "a {what} array with dimensions {dims:?} needs {expected} elements, got {len}"
        )))
    }
}

/// A sparse matrix's structure: `cols + 1` column pointers from 0 to the stored count, never
/// decreasing, row indices inside the matrix, and data (and imaginary parts) for every entry.
fn validate_mat_sparse(v: &MatSparse) -> Result<(), IoError> {
    let nnz = v.indices.len();
    let pointers_ok = v.cols.checked_add(1) == Some(v.indptr.len())
        && v.indptr.first() == Some(&0)
        && v.indptr.last() == Some(&nnz)
        && v.indptr.windows(2).all(|w| w[0] <= w[1]);
    if !pointers_ok {
        return Err(IoError::InvalidFormat(format!(
            "sparse column pointers must be {} values rising from 0 to the {nnz} stored entries",
            v.cols.saturating_add(1)
        )));
    }
    if v.data.len() != nnz || v.imag.as_ref().is_some_and(|m| m.len() != nnz) {
        return Err(IoError::InvalidFormat(format!(
            "a sparse matrix with {nnz} stored entries has {} values",
            v.data.len()
        )));
    }
    if let Some(&row) = v.indices.iter().find(|&&row| row >= v.rows) {
        return Err(IoError::InvalidFormat(format!(
            "sparse row index {row} is outside the {}-row matrix",
            v.rows
        )));
    }
    Ok(())
}

fn index_bytes(values: &[usize]) -> Result<Vec<u8>, IoError> {
    let mut out = Vec::with_capacity(4 * values.len());
    for &value in values {
        let value = i32::try_from(value).map_err(|_| {
            IoError::InvalidFormat(format!("sparse index {value} does not fit a MAT int32"))
        })?;
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

/// SciPy's `write_element`: a payload of at most 4 bytes goes in a small data element, a larger
/// one in a full element padded to 8 bytes.
fn write_mat5_element(out: &mut Vec<u8>, mdtype: u32, payload: &[u8]) -> Result<(), IoError> {
    let count = u32::try_from(payload.len()).map_err(|_| {
        IoError::InvalidFormat(format!(
            "a MAT data element of {} bytes exceeds the format's 4 GiB limit",
            payload.len()
        ))
    })?;
    if count <= 4 {
        out.extend_from_slice(&((count << 16) | mdtype).to_le_bytes());
        out.extend_from_slice(payload);
        out.resize(out.len() + 4 - payload.len(), 0);
    } else {
        out.extend_from_slice(&mdtype.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(payload);
        out.resize(out.len() + padding8(payload.len()), 0);
    }
    Ok(())
}

fn write_mat5_data(out: &mut Vec<u8>, data: &MatData) -> Result<(), IoError> {
    write_mat5_element(out, data.dtype().mi_type(), &data.le_bytes())
}

/// SciPy's `write_header`: array flags, dimensions and name.
fn write_mat5_array_header(
    out: &mut Vec<u8>,
    dims: &[usize],
    class: MatClass,
    complex: bool,
    logical: bool,
    nzmax: u32,
    name: &[u8],
) -> Result<(), IoError> {
    out.extend_from_slice(&MI_UINT32.to_le_bytes());
    out.extend_from_slice(&8u32.to_le_bytes());
    let flags = u32::from(class.code()) | (u32::from(complex) << 11) | (u32::from(logical) << 9);
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&nzmax.to_le_bytes());
    let mut dim_bytes = Vec::with_capacity(4 * dims.len());
    for &d in dims {
        let d = i32::try_from(d).map_err(|_| {
            IoError::InvalidFormat(format!("MAT dimension {d} does not fit an int32"))
        })?;
        dim_bytes.extend_from_slice(&d.to_le_bytes());
    }
    write_mat5_element(out, MI_INT32, &dim_bytes)?;
    write_mat5_element(out, MI_INT8, name)
}

/// SciPy's `arr_to_chars` for a string array: the char dimensions (the array's, or `[1]` for a
/// 0-d one, plus the width) and the characters in column-major order, padded with spaces. NUL
/// characters are written as spaces, as NumPy's empty `U1` elements are. An array of only empty
/// strings (or none) is SciPy's special case: all-zero dimensions and no characters.
fn strings_char_matrix(v: &MatStrings) -> Result<(Vec<usize>, Vec<char>), IoError> {
    check_element_count(v.strings.len(), &v.dims, "string")?;
    if let Some(long) = v.strings.iter().find(|s| s.chars().count() > v.width) {
        return Err(IoError::InvalidFormat(format!(
            "string {long:?} is longer than the string array's width {}",
            v.width
        )));
    }
    if v.strings.iter().all(String::is_empty) {
        return Ok((vec![0; v.dims.len().max(2)], Vec::new()));
    }
    let mut dims = if v.dims.is_empty() {
        vec![1]
    } else {
        v.dims.clone()
    };
    dims.push(v.width);
    let rows: Vec<Vec<char>> = v.strings.iter().map(|s| s.chars().collect()).collect();
    let mut chars = Vec::with_capacity(rows.len() * v.width);
    for k in 0..v.width {
        for row in &rows {
            let ch = row.get(k).copied().unwrap_or(' ');
            chars.push(if ch == '\0' { ' ' } else { ch });
        }
    }
    Ok((dims, chars))
}

/// SciPy's `VarWriter5`, writing little-endian as SciPy does on this platform.
struct Mat5Writer {
    oned_as: OnedAs,
    long_field_names: bool,
}

impl Mat5Writer {
    /// One `miMATRIX` element: tag (byte count patched in at the end), header, contents.
    fn write_matrix(
        &self,
        out: &mut Vec<u8>,
        value: &MatValue,
        name: &[u8],
        depth: usize,
    ) -> Result<(), IoError> {
        if depth > MAT_MAX_NESTING {
            return Err(mat_nesting_error());
        }
        let start = out.len();
        out.extend_from_slice(&[0u8; 8]);
        match value {
            MatValue::Numeric(v) => {
                check_element_count(v.real.len(), &v.dims, "numeric")?;
                if v.class.numeric_dtype().is_none() {
                    return Err(IoError::InvalidFormat(format!(
                        "a numeric array cannot have class {}",
                        v.class.name()
                    )));
                }
                if let Some(imag) = &v.imag
                    && imag.len() != v.real.len()
                {
                    return Err(IoError::InvalidFormat(format!(
                        "a complex array has {} real and {} imaginary values",
                        v.real.len(),
                        imag.len()
                    )));
                }
                let dims = matlab_dims(&v.dims, self.oned_as);
                write_mat5_array_header(out, &dims, v.class, v.imag.is_some(), v.logical, 0, name)?;
                write_mat5_data(out, &v.real)?;
                if let Some(imag) = &v.imag {
                    write_mat5_data(out, imag)?;
                }
            }
            MatValue::Char(v) => {
                check_element_count(v.chars.len(), &v.dims, "char")?;
                let dims = matlab_dims(&v.dims, self.oned_as);
                write_mat5_array_header(out, &dims, MatClass::Char, false, false, 0, name)?;
                let text: String = v.chars.iter().collect();
                write_mat5_element(out, MI_UTF8, text.as_bytes())?;
            }
            MatValue::Strings(v) => {
                let (dims, chars) = strings_char_matrix(v)?;
                write_mat5_array_header(out, &dims, MatClass::Char, false, false, 0, name)?;
                let text: String = chars.into_iter().collect();
                write_mat5_element(out, MI_UTF8, text.as_bytes())?;
            }
            MatValue::Cell(v) => {
                check_element_count(v.items.len(), &v.dims, "cell")?;
                let dims = matlab_dims(&v.dims, self.oned_as);
                write_mat5_array_header(out, &dims, MatClass::Cell, false, false, 0, name)?;
                for item in &v.items {
                    self.write_matrix(out, item, &[], depth + 1)?;
                }
            }
            MatValue::Struct(v) => {
                let dims = matlab_dims(&v.dims, self.oned_as);
                write_mat5_array_header(out, &dims, MatClass::Struct, false, false, 0, name)?;
                self.write_fields(out, v, depth)?;
            }
            MatValue::Object(v) => {
                let dims = matlab_dims(&v.fields.dims, self.oned_as);
                write_mat5_array_header(out, &dims, MatClass::Object, false, false, 0, name)?;
                if !v.class_name.is_ascii() {
                    return Err(IoError::InvalidFormat(format!(
                        "object class name {:?} is not ASCII",
                        v.class_name
                    )));
                }
                write_mat5_element(out, MI_INT8, v.class_name.as_bytes())?;
                self.write_fields(out, &v.fields, depth)?;
            }
            MatValue::Sparse(v) => {
                validate_mat_sparse(v)?;
                // MATLAB expects row indices sorted within each column (SciPy's `sort_indices`).
                let mut order: Vec<usize> = (0..v.indices.len()).collect();
                for c in 0..v.cols {
                    order[v.indptr[c]..v.indptr[c + 1]].sort_by_key(|&k| v.indices[k]);
                }
                let indices: Vec<usize> = order.iter().map(|&k| v.indices[k]).collect();
                let nzmax = u32::try_from(indices.len().max(1)).map_err(|_| {
                    IoError::InvalidFormat("sparse matrix has too many entries".to_string())
                })?;
                write_mat5_array_header(
                    out,
                    &[v.rows, v.cols],
                    MatClass::Sparse,
                    v.imag.is_some(),
                    v.logical,
                    nzmax,
                    name,
                )?;
                write_mat5_element(out, MI_INT32, &index_bytes(&indices)?)?;
                write_mat5_element(out, MI_INT32, &index_bytes(&v.indptr)?)?;
                write_mat5_data(out, &v.data.gather(&order))?;
                if let Some(imag) = &v.imag {
                    write_mat5_data(out, &imag.gather(&order))?;
                }
            }
            MatValue::Function(_) => {
                return Err(IoError::UnsupportedFeature(
                    "Cannot write matlab functions".to_string(),
                ));
            }
            MatValue::Opaque(_) => {
                return Err(IoError::UnsupportedFeature(
                    "Cannot write MATLAB opaque objects".to_string(),
                ));
            }
        }
        let count = u32::try_from(out.len() - start - 8).map_err(|_| {
            IoError::InvalidFormat("Matrix too large to save with Matlab 5 format".to_string())
        })?;
        out[start..start + 4].copy_from_slice(&MI_MATRIX.to_le_bytes());
        out[start + 4..start + 8].copy_from_slice(&count.to_le_bytes());
        Ok(())
    }

    /// SciPy's `_write_items`: name length, fixed-width names, then every element's fields. A
    /// struct with no fields is written as SciPy writes an empty one: name length 1, no names.
    fn write_fields(&self, out: &mut Vec<u8>, v: &MatStruct, depth: usize) -> Result<(), IoError> {
        let count = element_count(&v.dims)?;
        let expected = count.checked_mul(v.field_names.len());
        if expected != Some(v.values.len()) {
            return Err(IoError::InvalidFormat(format!(
                "a struct array with dimensions {:?} and {} fields has {} values",
                v.dims,
                v.field_names.len(),
                v.values.len()
            )));
        }
        if v.field_names.is_empty() {
            write_mat5_element(out, MI_INT32, &1i32.to_le_bytes())?;
            return write_mat5_element(out, MI_INT8, &[]);
        }
        if let Some(bad) = v
            .field_names
            .iter()
            .find(|f| f.is_empty() || !f.is_ascii() || f.contains('\0'))
        {
            return Err(IoError::InvalidFormat(format!(
                "struct field name {bad:?} is not a non-empty ASCII name"
            )));
        }
        let length = v.field_names.iter().map(String::len).max().unwrap_or(0) + 1;
        let max_length = if self.long_field_names { 64 } else { 32 };
        if length > max_length {
            return Err(IoError::InvalidFormat(format!(
                "Field names are restricted to {} characters",
                max_length - 1
            )));
        }
        write_mat5_element(out, MI_INT32, &(length as i32).to_le_bytes())?;
        let mut names = vec![0u8; length * v.field_names.len()];
        for (i, field) in v.field_names.iter().enumerate() {
            names[i * length..i * length + field.len()].copy_from_slice(field.as_bytes());
        }
        write_mat5_element(out, MI_INT8, &names)?;
        for value in &v.values {
            self.write_matrix(out, value, &[], depth + 1)?;
        }
        Ok(())
    }
}

/// Days since 1970-01-01 as a proleptic Gregorian (year, month, day).
fn civil_from_days(days: i64) -> (i64, usize, u64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month as usize, day as u64)
}

/// `time.asctime()` for a UTC instant (SciPy stamps local time; fsci has no time-zone data).
fn asctime_utc(seconds: u64) -> String {
    const WEEKDAYS: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let days = seconds / 86_400;
    let clock = seconds % 86_400;
    let (year, month, day) = civil_from_days(i64::try_from(days).unwrap_or(i64::MAX / 2));
    format!(
        "{} {} {day:2} {:02}:{:02}:{:02} {year}",
        WEEKDAYS[(days % 7) as usize],
        MONTHS[month - 1],
        clock / 3_600,
        clock % 3_600 / 60,
        clock % 60
    )
}

/// SciPy's `write_file_header`: description, zero subsystem offset, version 0x0100, "IM".
fn mat5_file_header_bytes() -> [u8; 128] {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    let platform = if cfg!(windows) { "nt" } else { "posix" };
    let text = format!(
        "MATLAB 5.0 MAT-file Platform: {platform}, Created on: {}",
        asctime_utc(seconds)
    );
    let mut header = [0u8; 128];
    let n = text.len().min(116);
    header[..n].copy_from_slice(&text.as_bytes()[..n]);
    header[124..126].copy_from_slice(&0x0100u16.to_le_bytes());
    header[126..128].copy_from_slice(b"IM");
    header
}

fn savemat_v5(
    variables: &[(String, MatValue)],
    options: &SavematOptions,
) -> Result<Vec<u8>, IoError> {
    let writer = Mat5Writer {
        oned_as: options.oned_as,
        long_field_names: options.long_field_names,
    };
    let mut out = mat5_file_header_bytes().to_vec();
    for (name, value) in variables {
        if name.is_empty() {
            return Err(IoError::InvalidFormat(
                "MAT variable name cannot be empty".to_string(),
            ));
        }
        // SciPy skips these with a MatWriteWarning: MATLAB cannot load them.
        if name.starts_with('_') {
            continue;
        }
        let name = latin1_bytes(name)?;
        if options.do_compression {
            let mut raw = Vec::new();
            writer.write_matrix(&mut raw, value, &name, 0)?;
            let packed = miniz_oxide::deflate::compress_to_vec_zlib(&raw, 6);
            let count = u32::try_from(packed.len()).map_err(|_| {
                IoError::InvalidFormat("compressed MAT variable exceeds 4 GiB".to_string())
            })?;
            out.extend_from_slice(&MI_COMPRESSED.to_le_bytes());
            out.extend_from_slice(&count.to_le_bytes());
            out.extend_from_slice(&packed);
        } else {
            writer.write_matrix(&mut out, value, &name, 0)?;
        }
    }
    Ok(out)
}

/// One Level 4 variable: SciPy's `VarWriter4.write`.
fn write_mat4_variable(
    out: &mut Vec<u8>,
    name: &str,
    value: &MatValue,
    oned_as: OnedAs,
) -> Result<(), IoError> {
    let mut name_bytes = latin1_bytes(name)?;
    name_bytes.push(0);
    // (dims, P data type, T class, imagf, payload)
    let (dims, data_type, class, imagf, payload): (Vec<usize>, i32, i32, i32, Vec<u8>) = match value
    {
        MatValue::Numeric(v) => {
            check_element_count(v.real.len(), &v.dims, "numeric")?;
            if let Some(imag) = &v.imag
                && imag.len() != v.real.len()
            {
                return Err(IoError::InvalidFormat(format!(
                    "a complex array has {} real and {} imaginary values",
                    v.real.len(),
                    imag.len()
                )));
            }
            // SciPy's `np_to_mtypes`; any other dtype (and mixed complex parts) is written as
            // double, as SciPy casts it to float64 / complex128.
            let native = match (v.real.dtype(), v.imag.as_ref().map(MatData::dtype)) {
                (MatDtype::F64, None | Some(MatDtype::F64)) => Some(0),
                (MatDtype::F32, None | Some(MatDtype::F32)) => Some(1),
                (MatDtype::I32, None) => Some(2),
                (MatDtype::I16, None) => Some(3),
                (MatDtype::U16, None) => Some(4),
                (MatDtype::U8, None) => Some(5),
                _ => None,
            };
            let mut payload = Vec::new();
            let data_type = if let Some(p) = native {
                payload.extend(v.real.le_bytes());
                if let Some(imag) = &v.imag {
                    payload.extend(imag.le_bytes());
                }
                p
            } else {
                payload.extend(v.real.cast(MatDtype::F64).le_bytes());
                if let Some(imag) = &v.imag {
                    payload.extend(imag.cast(MatDtype::F64).le_bytes());
                }
                0
            };
            (
                matlab_dims(&v.dims, oned_as),
                data_type,
                0,
                i32::from(v.imag.is_some()),
                payload,
            )
        }
        MatValue::Char(v) => {
            check_element_count(v.chars.len(), &v.dims, "char")?;
            let text: String = v.chars.iter().collect();
            (matlab_dims(&v.dims, oned_as), 5, 1, 0, latin1_bytes(&text)?)
        }
        MatValue::Strings(v) => {
            // SciPy converts a string array to chars unless it is already one character wide.
            let (dims, chars) = if v.width == 1 {
                check_element_count(v.strings.len(), &v.dims, "string")?;
                let chars = v
                    .strings
                    .iter()
                    .map(|s| s.chars().next().unwrap_or('\0'))
                    .collect();
                (matlab_dims(&v.dims, oned_as), chars)
            } else {
                let (mut dims, chars) = strings_char_matrix(v)?;
                if chars.is_empty() {
                    // SciPy's Level 4 writer has no empty-string special case.
                    dims = if v.dims.is_empty() {
                        vec![1]
                    } else {
                        v.dims.clone()
                    };
                    dims.push(v.width);
                    let blank = vec![' '; element_count(&dims)?];
                    let text: String = blank.into_iter().collect();
                    return write_mat4_record(
                        out,
                        &name_bytes,
                        &dims,
                        5,
                        1,
                        0,
                        &latin1_bytes(&text)?,
                    );
                }
                (dims, chars)
            };
            let text: String = chars.into_iter().collect();
            (dims, 5, 1, 0, latin1_bytes(&text)?)
        }
        MatValue::Sparse(v) => {
            validate_mat_sparse(v)?;
            let nnz = v.indices.len();
            let width = if v.imag.is_some() { 4 } else { 3 };
            let height = nnz + 1;
            let real = v.data.to_f64_vec();
            let imag = v.imag.as_ref().map(MatData::to_f64_vec);
            let mut table = vec![0.0f64; height * width];
            for c in 0..v.cols {
                for k in v.indptr[c]..v.indptr[c + 1] {
                    table[k] = (v.indices[k] + 1) as f64;
                    table[height + k] = (c + 1) as f64;
                    table[2 * height + k] = real[k];
                    if let Some(imag) = &imag {
                        table[3 * height + k] = imag[k];
                    }
                }
            }
            table[nnz] = v.rows as f64;
            table[height + nnz] = v.cols as f64;
            let payload = table.iter().flat_map(|x| x.to_le_bytes()).collect();
            (vec![height, width], 0, 2, 0, payload)
        }
        MatValue::Cell(_)
        | MatValue::Struct(_)
        | MatValue::Object(_)
        | MatValue::Function(_)
        | MatValue::Opaque(_) => {
            return Err(IoError::UnsupportedFeature(format!(
                "Cannot save object arrays in Mat4 ('{name}' is a {})",
                value.class_name()
            )));
        }
    };
    write_mat4_record(out, &name_bytes, &dims, data_type, class, imagf, &payload)
}

fn write_mat4_record(
    out: &mut Vec<u8>,
    name: &[u8],
    dims: &[usize],
    data_type: i32,
    class: i32,
    imagf: i32,
    payload: &[u8],
) -> Result<(), IoError> {
    let [rows, cols] = *dims else {
        return Err(IoError::InvalidFormat(
            "Matlab 4 files cannot save arrays with more than 2 dimensions".to_string(),
        ));
    };
    let field = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| {
            IoError::InvalidFormat(format!("MAT 4 {what} {value} does not fit an int32"))
        })
    };
    // mopt = M*1000 + O*100 + P*10 + T with M = 0 (little-endian) and O = 0.
    for word in [
        data_type * 10 + class,
        field(rows, "row count")?,
        field(cols, "column count")?,
        imagf,
        field(name.len(), "name length")?,
    ] {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out.extend_from_slice(name);
    out.extend_from_slice(payload);
    Ok(())
}

fn savemat_v4(
    variables: &[(String, MatValue)],
    options: &SavematOptions,
) -> Result<Vec<u8>, IoError> {
    if options.long_field_names {
        return Err(IoError::InvalidFormat(
            "Long field names are not available for version 4 files".to_string(),
        ));
    }
    let mut out = Vec::new();
    for (name, value) in variables {
        write_mat4_variable(&mut out, name, value, options.oned_as)?;
    }
    Ok(out)
}

/// SciPy's `matfile_version`: `(0, 0)` for a Level 4 file, `(1, minor)` for Level 5 and
/// `(2, minor)` for a v7.3 (HDF5) file.
///
/// # Errors
/// `IoError::InvalidFormat` for fewer than SciPy's 20 probe bytes ("Mat file appears to be
/// truncated"), 20 zero bytes ("corrupt"), and a Level 5-style header whose version byte is not
/// 1 or 2 ("Unknown mat file type").
pub fn matfile_version(bytes: &[u8]) -> Result<(u8, u8), IoError> {
    if bytes.len() < 20 {
        return Err(IoError::InvalidFormat(
            "Mat file appears to be truncated".to_string(),
        ));
    }
    if bytes[..20].iter().all(|&b| b == 0) {
        return Err(IoError::InvalidFormat(
            "Mat file appears to be corrupt (first 20 bytes == 0)".to_string(),
        ));
    }
    // A Level 4 file starts with its first variable's `mopt`, which has a zero byte.
    if bytes[..4].contains(&0) {
        return Ok((0, 0));
    }
    // Bytes 124..128 hold the version word and the endian indicator; SciPy tests the third.
    if bytes.len() < 127 {
        return Err(IoError::InvalidFormat(format!(
            "MAT file header is truncated: {} bytes, the version word needs 127",
            bytes.len()
        )));
    }
    let major_at = usize::from(bytes[126] == b'I');
    let major = bytes[124 + major_at];
    let minor = bytes[125 - major_at];
    if matches!(major, 1 | 2) {
        Ok((major, minor))
    } else {
        Err(IoError::InvalidFormat(format!(
            "Unknown mat file type, version {major}, {minor}"
        )))
    }
}

/// Read a MATLAB MAT-file: `scipy.io.loadmat` for Level 5 (MATLAB v5 through v7.2, compressed or
/// not, either byte order) and Level 4 files.
///
/// Values come back as SciPy returns them under `options` (see [`MatValue`], [`MatData`] and
/// [`LoadmatOptions`]), with the file's `__header__`, `__version__` and `__globals__` in
/// [`MatFile::header`].
///
/// A compressed variable is inflated whole before its header is read, so a damaged zlib
/// checksum fails the read even for a variable `variable_names` skips; SciPy inflates such a
/// variable in 128 KiB blocks and only notices when it reaches the damaged block.
///
/// # Errors
/// `IoError::UnsupportedFeature` for a v7.3 (HDF5) file; `IoError::InvalidFormat` for every
/// file SciPy rejects, and for the malformed files SciPy would turn into inconsistent arrays.
pub fn loadmat(bytes: &[u8], options: &LoadmatOptions) -> Result<MatFile, IoError> {
    match matfile_version(bytes)? {
        (0, _) => loadmat_v4(bytes, options),
        version @ (1, _) => loadmat_v5(bytes, version, options),
        _ => Err(mat73_unsupported()),
    }
}

/// List a MAT-file's variables without reading their data: `scipy.io.whosmat`'s
/// `(name, shape, class)`. Only `squeeze_me`, `simplify_cells` (which implies it) and
/// `chars_as_strings` (which drops a char array's last dimension) affect the result.
///
/// # Errors
/// As [`loadmat`], for the headers.
pub fn whosmat(bytes: &[u8], options: &LoadmatOptions) -> Result<Vec<MatInfo>, IoError> {
    match matfile_version(bytes)? {
        (0, _) => whosmat_v4(bytes, options),
        (1, _) => whosmat_v5(bytes, options),
        _ => Err(mat73_unsupported()),
    }
}

/// Split a Level 5 MAT-file into one file per variable (`scipy.io.matlab.varmats_from_mat`):
/// each is the original 128-byte header followed by the variable's element, unread. Duplicate
/// names are all kept.
///
/// # Errors
/// `IoError::UnsupportedFeature` for a Level 4 or v7.3 file; otherwise as [`loadmat`], for the
/// headers.
pub fn varmats_from_mat(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>, IoError> {
    match matfile_version(bytes)? {
        (1, _) => {}
        (0, _) => {
            return Err(IoError::UnsupportedFeature(
                "varmats_from_mat splits Level 5 files; this is a Level 4 file".to_string(),
            ));
        }
        _ => return Err(mat73_unsupported()),
    }
    let (big_endian, _, _) = read_mat5_file_header(bytes)?;
    let mut spans = Vec::new();
    for_each_mat5_variable(bytes, big_endian, |_, header, top| {
        spans.push((mat5_variable_name(&header), top.start, top.next));
        Ok(true)
    })?;
    Ok(spans
        .into_iter()
        .map(|(name, start, next)| {
            let mut file = bytes[..128].to_vec();
            file.extend_from_slice(&bytes[start..next.min(bytes.len())]);
            (name, file)
        })
        .collect())
}

/// Write a MATLAB MAT-file: `scipy.io.savemat`.
///
/// Level 5 (the default) writes every [`MatValue`] but function handles and opaque objects,
/// which SciPy cannot write either; Level 4 writes 2-D numeric, char and sparse arrays. As in
/// SciPy, a Level 5 variable whose name starts with `_` is skipped (SciPy warns), a 0-d or 1-D
/// array is written by [`SavematOptions::oned_as`], numeric data is written in its own dtype
/// with the value's class and logical flag, and a [`MatStrings`] array is written as SciPy
/// writes a NumPy string array (padded with spaces; all-empty becomes an empty char array). A
/// [`MatChar`] is written with its own dimensions. The header's creation time is UTC.
///
/// # Errors
/// `IoError::UnsupportedFeature` for values the format cannot hold; `IoError::InvalidFormat` for
/// inconsistent values (element counts that do not match the dimensions, a malformed sparse
/// structure, over-long field names, names that are not Latin-1).
pub fn savemat(
    variables: &[(String, MatValue)],
    options: &SavematOptions,
) -> Result<Vec<u8>, IoError> {
    match options.format {
        MatFormat::V4 => savemat_v4(variables, options),
        MatFormat::V5 => savemat_v5(variables, options),
    }
}

// ══════════════════════════════════════════════════════════════════════
// Plain-text matrices (fsci's own `savemat_text` / `loadmat_text` format)
// ══════════════════════════════════════════════════════════════════════

/// A named 2-D `f64` matrix, row-major, in the plain-text format of [`savemat_text`] and
/// [`loadmat_text`] (not a MATLAB MAT-file; see [`loadmat`] for those).
#[derive(Debug, Clone)]
pub struct MatArray {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

/// Runtime switch to force the serial `savemat_text` formatter for same-binary A/B
/// benchmarks. Defaults off. `#[doc(hidden)]` — internal.
/// CONTRACT: BYTE-IDENTICAL output either way. Each worker formats a
/// contiguous range of ROWS into a private `String` and the chunks are joined
/// in row order, so the emitted text is the same sequence of bytes the serial
/// loop produces. `f64` `Display` is deterministic for a given value, and no
/// value is combined with any other -- there is no arithmetic here at all, so
/// nothing can reassociate.
///
/// The row boundary is what makes this safe: a chunk split MID-ROW would move
/// a delimiter or a newline, so the split is by whole rows and the per-row
/// formatter is identical in both arms.
#[doc(hidden)]
pub static SAVEMAT_TEXT_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Save arrays to a simple text-based format (similar to MATLAB ASCII).
pub fn savemat_text(arrays: &[MatArray]) -> Result<String, IoError> {
    let mut out = String::new();
    for arr in arrays {
        if arr.name.contains(['\n', '\r']) {
            return Err(IoError::InvalidFormat(format!(
                "array name '{}' contains a newline and cannot be encoded safely",
                arr.name.escape_debug()
            )));
        }
        let expected_len = arr.rows.checked_mul(arr.cols).ok_or_else(|| {
            IoError::InvalidFormat(format!(
                "array '{}' dimensions {}x{} overflow usize",
                arr.name, arr.rows, arr.cols
            ))
        })?;
        if arr.data.len() != expected_len {
            return Err(IoError::InvalidFormat(format!(
                "array '{}' expected {} values but found {}",
                arr.name,
                expected_len,
                arr.data.len()
            )));
        }
        out.push_str(&format!(
            "# name: {}\n# type: matrix\n# rows: {}\n# columns: {}\n",
            arr.name, arr.rows, arr.cols
        ));
        savemat_append_matrix_body(&mut out, &arr.data, arr.rows, arr.cols);
        out.push('\n');
    }
    Ok(out)
}

/// Append a space-delimited row-major matrix body to `out` — the f64 Display
/// formatting dominates and each row is independent, so a large body is formatted in
/// parallel per-row-range into private Strings and joined in row order (BIT-FOR-BIT
/// the serial loop). Serial gate BEFORE the available_parallelism syscall.
fn savemat_append_matrix_body(out: &mut String, data: &[f64], rows: usize, cols: usize) {
    const SAVEMAT_PAR_GATE: usize = 1 << 16;
    let n = rows.saturating_mul(cols);
    let serial = |out: &mut String| {
        for r in 0..rows {
            for c in 0..cols {
                if c > 0 {
                    out.push(' ');
                }
                let _ = write!(out, "{}", data[r * cols + c]);
            }
            out.push('\n');
        }
    };
    if n < SAVEMAT_PAR_GATE || SAVEMAT_TEXT_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
    {
        serial(out);
        return;
    }
    let nthreads = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
        .min(n / 16384)
        .min(rows)
        .max(1);
    if nthreads <= 1 {
        serial(out);
        return;
    }
    let chunk = rows.div_ceil(nthreads);
    let mut parts: Vec<String> = (0..nthreads).map(|_| String::new()).collect();
    std::thread::scope(|scope| {
        for (t, slot) in parts.iter_mut().enumerate() {
            let r0 = t * chunk;
            let r1 = ((t + 1) * chunk).min(rows);
            scope.spawn(move || {
                if r0 >= r1 {
                    return;
                }
                let mut local = String::with_capacity((r1 - r0) * cols * 12);
                for r in r0..r1 {
                    for c in 0..cols {
                        if c > 0 {
                            local.push(' ');
                        }
                        let _ = write!(local, "{}", data[r * cols + c]);
                    }
                    local.push('\n');
                }
                *slot = local;
            });
        }
    });
    let extra: usize = parts.iter().map(String::len).sum();
    out.reserve(extra);
    for p in &parts {
        out.push_str(p);
    }
}

/// Load arrays from the text-based format.
pub fn loadmat_text(content: &str) -> Result<Vec<MatArray>, IoError> {
    let mut arrays = Vec::new();
    let mut lines = content.lines().peekable();

    while lines.peek().is_some() {
        // Find "# name:" line
        let mut name = None;
        let mut rows = 0usize;
        let mut cols = 0usize;

        loop {
            match lines.next() {
                None => {
                    if name.is_some() || rows != 0 || cols != 0 {
                        return Err(IoError::InvalidFormat(
                            "incomplete MAT text block at end of file".to_string(),
                        ));
                    }
                    return Ok(arrays);
                }
                Some(line) => {
                    let trimmed = line.trim();
                    if let Some(stripped) = trimmed.strip_prefix("# name:") {
                        name = Some(stripped.trim().to_string());
                    } else if let Some(stripped) = trimmed.strip_prefix("# rows:") {
                        rows = stripped
                            .trim()
                            .parse()
                            .map_err(|e| IoError::InvalidFormat(format!("bad rows: {e}")))?;
                    } else if let Some(stripped) = trimmed.strip_prefix("# columns:") {
                        cols = stripped
                            .trim()
                            .parse()
                            .map_err(|e| IoError::InvalidFormat(format!("bad cols: {e}")))?;
                    } else if trimmed.starts_with("# type:") {
                        // Skip type line
                    } else if !trimmed.is_empty() && !trimmed.starts_with('#') {
                        // Data line — we've hit the matrix data
                        let n = name.as_ref().ok_or_else(|| {
                            IoError::InvalidFormat(
                                "encountered matrix data before '# name:' header".to_string(),
                            )
                        })?;
                        if rows == 0 || cols == 0 {
                            return Err(IoError::InvalidFormat(format!(
                                "array '{n}' is missing nonzero '# rows:' and '# columns:' headers before data"
                            )));
                        }
                        let expected_len = checked_matrix_len(rows, cols, "MAT text matrix")?;
                        let mut data = Vec::with_capacity(expected_len);
                        let parse_row = |line: &str| -> Result<Vec<f64>, IoError> {
                            line.split_whitespace()
                                .map(|val_str| {
                                    val_str.parse::<f64>().map_err(|e| {
                                        IoError::InvalidFormat(format!("bad value: {e}"))
                                    })
                                })
                                .collect()
                        };
                        let first_vals = parse_row(trimmed)?;
                        if first_vals.len() != cols {
                            return Err(IoError::InvalidFormat(format!(
                                "array '{n}' row 0 has {} columns, expected {cols}",
                                first_vals.len()
                            )));
                        }
                        data.extend_from_slice(&first_vals);
                        // Read remaining rows
                        for row_idx in 1..rows {
                            let line = lines.next().ok_or_else(|| {
                                IoError::InvalidFormat(format!(
                                    "array '{n}' expected {rows} rows but found {row_idx}"
                                ))
                            })?;
                            let row_vals = parse_row(line)?;
                            if row_vals.len() != cols {
                                return Err(IoError::InvalidFormat(format!(
                                    "array '{n}' row {row_idx} has {} columns, expected {cols}",
                                    row_vals.len()
                                )));
                            }
                            data.extend_from_slice(&row_vals);
                        }
                        if data.len() != expected_len {
                            return Err(IoError::InvalidFormat(format!(
                                "array '{n}' expected {} values but found {}",
                                expected_len,
                                data.len()
                            )));
                        }
                        arrays.push(MatArray {
                            name: n.clone(),
                            rows,
                            cols,
                            data,
                        });
                        break;
                    }
                }
            }
        }
    }

    Ok(arrays)
}

// ══════════════════════════════════════════════════════════════════════
// IDL SAVE files
// ══════════════════════════════════════════════════════════════════════

const IDL_MAX_ARRAY_ELEMENTS: usize = 64 * 1024 * 1024;

/// Primitive IDL SAVE type codes supported by `read_idl_save`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdlType {
    Byte,
    Int16,
    Int32,
    Float32,
    Float64,
    Complex32,
    String,
    Complex64,
    UInt16,
    UInt32,
    Int64,
    UInt64,
}

impl IdlType {
    fn from_code(code: i32) -> Result<Self, IoError> {
        match code {
            1 => Ok(Self::Byte),
            2 => Ok(Self::Int16),
            3 => Ok(Self::Int32),
            4 => Ok(Self::Float32),
            5 => Ok(Self::Float64),
            6 => Ok(Self::Complex32),
            7 => Ok(Self::String),
            9 => Ok(Self::Complex64),
            12 => Ok(Self::UInt16),
            13 => Ok(Self::UInt32),
            14 => Ok(Self::Int64),
            15 => Ok(Self::UInt64),
            8 => Err(IoError::UnsupportedFeature(
                "IDL SAVE structure type code 8 is not supported".to_string(),
            )),
            10 => Err(IoError::UnsupportedFeature(
                "IDL SAVE heap pointer type code 10 is not supported".to_string(),
            )),
            11 => Err(IoError::UnsupportedFeature(
                "IDL SAVE object pointer type code 11 is not supported".to_string(),
            )),
            other => Err(IoError::InvalidFormat(format!(
                "IDL SAVE unknown type code {other}"
            ))),
        }
    }
}

/// Scalar value read from an IDL SAVE file.
#[derive(Debug, Clone, PartialEq)]
pub enum IdlScalar {
    Byte(u8),
    Int16(i16),
    Int32(i32),
    Float32(f32),
    Float64(f64),
    Complex32 { real: f32, imag: f32 },
    String(Vec<u8>),
    Complex64 { real: f64, imag: f64 },
    UInt16(u16),
    UInt32(u32),
    Int64(i64),
    UInt64(u64),
}

/// Primitive IDL array. Dimensions follow SciPy's observable order: the IDL
/// descriptor dimensions are reversed before being exposed.
#[derive(Debug, Clone, PartialEq)]
pub struct IdlArray {
    pub element_type: IdlType,
    pub dims: Vec<usize>,
    pub values: Vec<IdlScalar>,
}

/// Variable value read from an IDL SAVE file.
#[derive(Debug, Clone, PartialEq)]
pub enum IdlValue {
    Null,
    Scalar(IdlScalar),
    Array(IdlArray),
}

/// Named IDL SAVE variable.
#[derive(Debug, Clone, PartialEq)]
pub struct IdlVariable {
    pub name: String,
    pub value: IdlValue,
}

/// Parsed IDL SAVE file.
#[derive(Debug, Clone, PartialEq)]
pub struct IdlSaveFile {
    pub variables: Vec<IdlVariable>,
}

impl IdlSaveFile {
    /// Case-insensitive variable lookup, matching SciPy's `readsav` access
    /// behavior for variable names.
    pub fn get(&self, name: &str) -> Option<&IdlValue> {
        self.variables
            .iter()
            .find(|variable| variable.name.eq_ignore_ascii_case(name))
            .map(|variable| &variable.value)
    }
}

#[derive(Debug, Clone)]
struct IdlTypeDesc {
    type_code: i32,
    is_array: bool,
    is_structure: bool,
    array_desc: Option<IdlArrayDesc>,
}

#[derive(Debug, Clone)]
struct IdlArrayDesc {
    nbytes: usize,
    nelements: usize,
    dims: Vec<usize>,
}

struct IdlReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> IdlReader<'a> {
    fn new(bytes: &'a [u8], offset: usize) -> Self {
        Self { bytes, offset }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn read_exact(&mut self, len: usize, context: &str) -> Result<&'a [u8], IoError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| IoError::InvalidFormat(format!("IDL SAVE {context} offset overflow")))?;
        if end > self.bytes.len() {
            return Err(IoError::InvalidFormat(format!(
                "IDL SAVE {context} truncated: need {len} bytes, have {}",
                self.remaining()
            )));
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn skip(&mut self, len: usize, context: &str) -> Result<(), IoError> {
        self.read_exact(len, context).map(|_| ())
    }

    fn seek(&mut self, offset: usize, context: &str) -> Result<(), IoError> {
        if offset > self.bytes.len() {
            return Err(IoError::InvalidFormat(format!(
                "IDL SAVE {context} seeks past end: {offset} > {}",
                self.bytes.len()
            )));
        }
        self.offset = offset;
        Ok(())
    }

    fn align_32(&mut self, context: &str) -> Result<(), IoError> {
        let aligned = match self.offset % 4 {
            0 => self.offset,
            rem => self.offset + 4 - rem,
        };
        self.seek(aligned, context)
    }

    fn read_i32(&mut self, context: &str) -> Result<i32, IoError> {
        let bytes = self.read_exact(4, context)?;
        Ok(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u32(&mut self, context: &str) -> Result<u32, IoError> {
        let bytes = self.read_exact(4, context)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i64(&mut self, context: &str) -> Result<i64, IoError> {
        let bytes = self.read_exact(8, context)?;
        Ok(i64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_u64(&mut self, context: &str) -> Result<u64, IoError> {
        let bytes = self.read_exact(8, context)?;
        Ok(u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_f32(&mut self, context: &str) -> Result<f32, IoError> {
        Ok(f32::from_bits(self.read_u32(context)?))
    }

    fn read_f64(&mut self, context: &str) -> Result<f64, IoError> {
        Ok(f64::from_bits(self.read_u64(context)?))
    }

    fn read_padded_u8(&mut self, context: &str) -> Result<u8, IoError> {
        Ok(self.read_exact(4, context)?[0])
    }

    fn read_padded_i16(&mut self, context: &str) -> Result<i16, IoError> {
        let bytes = self.read_exact(4, context)?;
        Ok(i16::from_be_bytes([bytes[2], bytes[3]]))
    }

    fn read_padded_u16(&mut self, context: &str) -> Result<u16, IoError> {
        let bytes = self.read_exact(4, context)?;
        Ok(u16::from_be_bytes([bytes[2], bytes[3]]))
    }
}

/// Read an IDL SAVE (`.sav`) byte stream.
///
/// This covers the uncompressed `scipy.io.readsav` scalar and primitive-array
/// surface: numeric scalars, byte strings, complex numbers, and arrays of those
/// primitive values. Structures, heap/object pointers, and compressed SAVE
/// files fail closed with `UnsupportedFeature`.
pub fn read_idl_save(bytes: &[u8]) -> Result<IdlSaveFile, IoError> {
    if bytes.len() < 4 {
        return Err(IoError::InvalidFormat(
            "IDL SAVE header truncated".to_string(),
        ));
    }
    if &bytes[..2] != b"SR" {
        return Err(IoError::InvalidFormat(format!(
            "IDL SAVE invalid signature: {:02x?}",
            &bytes[..2]
        )));
    }
    match &bytes[2..4] {
        b"\x00\x04" => {}
        b"\x00\x06" => {
            return Err(IoError::UnsupportedFeature(
                "compressed IDL SAVE files are not supported".to_string(),
            ));
        }
        recfmt => {
            return Err(IoError::InvalidFormat(format!(
                "IDL SAVE invalid record format: {recfmt:02x?}"
            )));
        }
    }

    let mut reader = IdlReader::new(bytes, 4);
    let mut variables = Vec::new();
    let mut saw_end = false;

    while reader.offset < bytes.len() {
        let record_start = reader.offset;
        let rectype = reader.read_i32("record type")?;
        let next_low = u64::from(reader.read_u32("next record low word")?);
        let next_high = u64::from(reader.read_u32("next record high word")?);
        reader.skip(4, "record header padding")?;

        if rectype == 6 {
            saw_end = true;
            break;
        }

        let next_record = checked_idl_next_record(next_low, next_high, reader.offset)?;
        if next_record > bytes.len() {
            return Err(IoError::InvalidFormat(format!(
                "IDL SAVE record at offset {record_start} points past end: {next_record} > {}",
                bytes.len()
            )));
        }

        match rectype {
            0 | 1 | 3 | 10 | 12 | 13 | 14 | 15 | 17 | 19 | 20 => {}
            2 => {
                let variable = read_idl_variable_record(&mut reader, next_record)?;
                variables.push(variable);
            }
            16 => {
                return Err(IoError::UnsupportedFeature(
                    "IDL SAVE heap data records are not supported".to_string(),
                ));
            }
            other => {
                return Err(IoError::InvalidFormat(format!(
                    "IDL SAVE unknown record type {other} at offset {record_start}"
                )));
            }
        }

        if reader.offset > next_record {
            return Err(IoError::InvalidFormat(format!(
                "IDL SAVE record type {rectype} over-read next record boundary"
            )));
        }
        reader.seek(next_record, "record boundary")?;
    }

    if !saw_end {
        return Err(IoError::InvalidFormat(
            "IDL SAVE missing END_MARKER record".to_string(),
        ));
    }

    Ok(IdlSaveFile { variables })
}

/// Alias matching `scipy.io.readsav`.
pub fn readsav(bytes: &[u8]) -> Result<IdlSaveFile, IoError> {
    read_idl_save(bytes)
}

fn checked_idl_next_record(low: u64, high: u64, current: usize) -> Result<usize, IoError> {
    let next = high
        .checked_mul(1u64 << 32)
        .and_then(|base| base.checked_add(low))
        .ok_or_else(|| IoError::InvalidFormat("IDL SAVE next record overflow".to_string()))?;
    let next = usize::try_from(next).map_err(|_| {
        IoError::InvalidFormat("IDL SAVE next record does not fit usize".to_string())
    })?;
    if next < current {
        return Err(IoError::InvalidFormat(format!(
            "IDL SAVE next record {next} precedes current offset {current}"
        )));
    }
    Ok(next)
}

fn read_idl_variable_record(
    reader: &mut IdlReader<'_>,
    next_record: usize,
) -> Result<IdlVariable, IoError> {
    let name = read_idl_string(reader, "variable name")?;
    let typedesc = read_idl_typedesc(reader)?;
    let value = if typedesc.type_code == 0 {
        if reader.offset != next_record {
            return Err(IoError::InvalidFormat(
                "IDL SAVE null typedesc has trailing payload".to_string(),
            ));
        }
        IdlValue::Null
    } else {
        let varstart = reader.read_i32("VARSTART")?;
        if varstart != 7 {
            return Err(IoError::InvalidFormat(format!(
                "IDL SAVE VARSTART must be 7, got {varstart}"
            )));
        }
        read_idl_value(reader, &typedesc)?
    };
    Ok(IdlVariable { name, value })
}

fn read_idl_typedesc(reader: &mut IdlReader<'_>) -> Result<IdlTypeDesc, IoError> {
    let type_code = reader.read_i32("type descriptor type code")?;
    let varflags = reader.read_i32("type descriptor flags")?;

    if varflags & 2 == 2 {
        return Err(IoError::UnsupportedFeature(
            "IDL SAVE system variables are not supported".to_string(),
        ));
    }
    let is_array = varflags & 4 == 4;
    let is_structure = varflags & 32 == 32;
    if is_structure {
        return Err(IoError::UnsupportedFeature(
            "IDL SAVE structure variables are not supported".to_string(),
        ));
    }
    let array_desc = if is_array {
        Some(read_idl_arraydesc(reader)?)
    } else {
        None
    };

    Ok(IdlTypeDesc {
        type_code,
        is_array,
        is_structure,
        array_desc,
    })
}

fn read_idl_arraydesc(reader: &mut IdlReader<'_>) -> Result<IdlArrayDesc, IoError> {
    let arrstart = reader.read_i32("array descriptor start")?;
    if arrstart == 18 {
        return Err(IoError::UnsupportedFeature(
            "IDL SAVE 64-bit array descriptors are not supported".to_string(),
        ));
    }
    if arrstart != 8 {
        return Err(IoError::InvalidFormat(format!(
            "IDL SAVE unknown array descriptor start {arrstart}"
        )));
    }

    reader.skip(4, "array descriptor padding")?;
    let nbytes = read_idl_nonnegative_usize(reader, "array byte count")?;
    let nelements = read_idl_nonnegative_usize(reader, "array element count")?;
    let ndims = read_idl_nonnegative_usize(reader, "array dimension count")?;
    reader.skip(8, "array descriptor reserved fields")?;
    let nmax = read_idl_nonnegative_usize(reader, "array max dimension count")?;
    if ndims > nmax {
        return Err(IoError::InvalidFormat(format!(
            "IDL SAVE array ndims {ndims} exceeds nmax {nmax}"
        )));
    }
    if nelements > IDL_MAX_ARRAY_ELEMENTS {
        return Err(IoError::InvalidFormat(format!(
            "IDL SAVE array element count {nelements} exceeds safety bound {IDL_MAX_ARRAY_ELEMENTS}"
        )));
    }

    let mut raw_dims = Vec::with_capacity(nmax);
    for _ in 0..nmax {
        raw_dims.push(read_idl_nonnegative_usize(reader, "array dimension")?);
    }
    validate_idl_array_shape(&raw_dims, ndims, nelements)?;
    let dims = raw_dims
        .iter()
        .take(ndims)
        .rev()
        .copied()
        .collect::<Vec<_>>();

    Ok(IdlArrayDesc {
        nbytes,
        nelements,
        dims,
    })
}

fn read_idl_nonnegative_usize(reader: &mut IdlReader<'_>, context: &str) -> Result<usize, IoError> {
    let value = reader.read_i32(context)?;
    if value < 0 {
        return Err(IoError::InvalidFormat(format!(
            "IDL SAVE {context} is negative: {value}"
        )));
    }
    usize::try_from(value).map_err(|_| {
        IoError::InvalidFormat(format!("IDL SAVE {context} does not fit usize: {value}"))
    })
}

fn validate_idl_array_shape(
    raw_dims: &[usize],
    ndims: usize,
    nelements: usize,
) -> Result<(), IoError> {
    let shape_product = raw_dims.iter().take(ndims).try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| IoError::InvalidFormat("IDL SAVE array shape overflow".to_string()))
    })?;
    if shape_product != nelements {
        return Err(IoError::InvalidFormat(format!(
            "IDL SAVE array shape product {shape_product} does not match element count {nelements}"
        )));
    }
    Ok(())
}

fn read_idl_value(reader: &mut IdlReader<'_>, typedesc: &IdlTypeDesc) -> Result<IdlValue, IoError> {
    if typedesc.is_structure {
        return Err(IoError::UnsupportedFeature(
            "IDL SAVE structure values are not supported".to_string(),
        ));
    }
    let idl_type = IdlType::from_code(typedesc.type_code)?;
    if typedesc.is_array {
        let array_desc = typedesc.array_desc.as_ref().ok_or_else(|| {
            IoError::InvalidFormat("IDL SAVE array flag without descriptor".to_string())
        })?;
        Ok(IdlValue::Array(read_idl_array(
            reader, idl_type, array_desc,
        )?))
    } else {
        Ok(IdlValue::Scalar(read_idl_scalar(reader, idl_type)?))
    }
}

fn read_idl_scalar(reader: &mut IdlReader<'_>, idl_type: IdlType) -> Result<IdlScalar, IoError> {
    match idl_type {
        IdlType::Byte => {
            let byte_count = reader.read_i32("byte scalar marker")?;
            if byte_count != 1 {
                return Err(IoError::InvalidFormat(format!(
                    "IDL SAVE byte scalar marker must be 1, got {byte_count}"
                )));
            }
            Ok(IdlScalar::Byte(reader.read_padded_u8("byte scalar")?))
        }
        IdlType::Int16 => Ok(IdlScalar::Int16(reader.read_padded_i16("int16 scalar")?)),
        IdlType::Int32 => Ok(IdlScalar::Int32(reader.read_i32("int32 scalar")?)),
        IdlType::Float32 => Ok(IdlScalar::Float32(reader.read_f32("float32 scalar")?)),
        IdlType::Float64 => Ok(IdlScalar::Float64(reader.read_f64("float64 scalar")?)),
        IdlType::Complex32 => {
            let real = reader.read_f32("complex32 real")?;
            let imag = reader.read_f32("complex32 imag")?;
            Ok(IdlScalar::Complex32 { real, imag })
        }
        IdlType::String => Ok(IdlScalar::String(read_idl_string_data(
            reader,
            "string scalar",
        )?)),
        IdlType::Complex64 => {
            let real = reader.read_f64("complex64 real")?;
            let imag = reader.read_f64("complex64 imag")?;
            Ok(IdlScalar::Complex64 { real, imag })
        }
        IdlType::UInt16 => Ok(IdlScalar::UInt16(reader.read_padded_u16("uint16 scalar")?)),
        IdlType::UInt32 => Ok(IdlScalar::UInt32(reader.read_u32("uint32 scalar")?)),
        IdlType::Int64 => Ok(IdlScalar::Int64(reader.read_i64("int64 scalar")?)),
        IdlType::UInt64 => Ok(IdlScalar::UInt64(reader.read_u64("uint64 scalar")?)),
    }
}

fn read_idl_array(
    reader: &mut IdlReader<'_>,
    idl_type: IdlType,
    desc: &IdlArrayDesc,
) -> Result<IdlArray, IoError> {
    let values = match idl_type {
        IdlType::Byte => read_idl_byte_array(reader, desc)?,
        IdlType::Int16 => read_idl_repeated(reader, desc.nelements, |reader| {
            Ok(IdlScalar::Int16(
                reader.read_padded_i16("int16 array element")?,
            ))
        })?,
        IdlType::Int32 => read_idl_repeated(reader, desc.nelements, |reader| {
            Ok(IdlScalar::Int32(reader.read_i32("int32 array element")?))
        })?,
        IdlType::Float32 => read_idl_repeated(reader, desc.nelements, |reader| {
            Ok(IdlScalar::Float32(
                reader.read_f32("float32 array element")?,
            ))
        })?,
        IdlType::Float64 => read_idl_repeated(reader, desc.nelements, |reader| {
            Ok(IdlScalar::Float64(
                reader.read_f64("float64 array element")?,
            ))
        })?,
        IdlType::Complex32 => read_idl_repeated(reader, desc.nelements, |reader| {
            let real = reader.read_f32("complex32 array real")?;
            let imag = reader.read_f32("complex32 array imag")?;
            Ok(IdlScalar::Complex32 { real, imag })
        })?,
        IdlType::String => read_idl_repeated(reader, desc.nelements, |reader| {
            Ok(IdlScalar::String(read_idl_string_data(
                reader,
                "string array element",
            )?))
        })?,
        IdlType::Complex64 => read_idl_repeated(reader, desc.nelements, |reader| {
            let real = reader.read_f64("complex64 array real")?;
            let imag = reader.read_f64("complex64 array imag")?;
            Ok(IdlScalar::Complex64 { real, imag })
        })?,
        IdlType::UInt16 => read_idl_repeated(reader, desc.nelements, |reader| {
            Ok(IdlScalar::UInt16(
                reader.read_padded_u16("uint16 array element")?,
            ))
        })?,
        IdlType::UInt32 => read_idl_repeated(reader, desc.nelements, |reader| {
            Ok(IdlScalar::UInt32(reader.read_u32("uint32 array element")?))
        })?,
        IdlType::Int64 => read_idl_repeated(reader, desc.nelements, |reader| {
            Ok(IdlScalar::Int64(reader.read_i64("int64 array element")?))
        })?,
        IdlType::UInt64 => read_idl_repeated(reader, desc.nelements, |reader| {
            Ok(IdlScalar::UInt64(reader.read_u64("uint64 array element")?))
        })?,
    };
    reader.align_32("array payload alignment")?;
    Ok(IdlArray {
        element_type: idl_type,
        dims: desc.dims.clone(),
        values,
    })
}

fn read_idl_repeated<F>(
    reader: &mut IdlReader<'_>,
    count: usize,
    mut read_one: F,
) -> Result<Vec<IdlScalar>, IoError>
where
    F: FnMut(&mut IdlReader<'_>) -> Result<IdlScalar, IoError>,
{
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(read_one(reader)?);
    }
    Ok(values)
}

fn read_idl_byte_array(
    reader: &mut IdlReader<'_>,
    desc: &IdlArrayDesc,
) -> Result<Vec<IdlScalar>, IoError> {
    let byte_count = read_idl_nonnegative_usize(reader, "byte array payload byte count")?;
    if byte_count != desc.nbytes || byte_count != desc.nelements {
        return Err(IoError::InvalidFormat(format!(
            "IDL SAVE byte array count {byte_count} does not match descriptor nbytes={} nelements={}",
            desc.nbytes, desc.nelements
        )));
    }
    reader
        .read_exact(byte_count, "byte array payload")?
        .iter()
        .map(|&value| Ok(IdlScalar::Byte(value)))
        .collect()
}

fn read_idl_string(reader: &mut IdlReader<'_>, context: &str) -> Result<String, IoError> {
    let len = read_idl_nonnegative_usize(reader, context)?;
    if len == 0 {
        return Ok(String::new());
    }
    let bytes = reader.read_exact(len, context)?;
    reader.align_32(context)?;
    Ok(bytes.iter().map(|&byte| char::from(byte)).collect())
}

fn read_idl_string_data(reader: &mut IdlReader<'_>, context: &str) -> Result<Vec<u8>, IoError> {
    let len = read_idl_nonnegative_usize(reader, context)?;
    if len == 0 {
        return Ok(Vec::new());
    }
    let repeated_len = read_idl_nonnegative_usize(reader, context)?;
    if repeated_len != len {
        return Err(IoError::InvalidFormat(format!(
            "IDL SAVE string length marker mismatch: {len} != {repeated_len}"
        )));
    }
    let bytes = reader.read_exact(len, context)?.to_vec();
    reader.align_32(context)?;
    Ok(bytes)
}

// ══════════════════════════════════════════════════════════════════════
// Text matrix utility
// ══════════════════════════════════════════════════════════════════════

/// Load a whitespace-delimited text file as a matrix.
///
/// Like `numpy.loadtxt`.
/// Reduce a raw text line to its numeric payload the way `numpy.loadtxt` does:
/// everything from the first `#` to end-of-line is a comment and is dropped,
/// then the remainder is trimmed. A line that is only a comment (or blank)
/// becomes empty. The legacy `%` leading-comment convention is handled by the
/// callers (an empty payload or a `%`-led payload is skipped).
fn loadtxt_line_payload(line: &str) -> &str {
    line.split('#').next().unwrap_or("").trim()
}

pub fn loadtxt(content: &str) -> Result<(usize, usize, Vec<f64>), IoError> {
    // Parsing `split_whitespace().parse::<f64>()` for every field dominates loadtxt on
    // large numeric files. Each data line maps to its own contiguous output row with no
    // cross-line state (the only shared value is `cols`, fixed by the first data row), so
    // the lines parse independently. We split the line list into contiguous chunks, parse
    // each on its own core into a local buffer, and concatenate the buffers in chunk order
    // — byte-identical to the serial loop (same deterministic f64 parse, same row order,
    // same `cols`). Any malformed input (parse error or column mismatch) falls back to the
    // exact serial loop so the returned error — message and order — is unchanged.
    let lines: Vec<&str> = content.lines().collect();

    // `cols` is set by the first non-comment line, exactly as the serial loop does.
    let mut cols = 0usize;
    let mut have_data = false;
    for line in &lines {
        let trimmed = loadtxt_line_payload(line);
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }
        cols = trimmed.split_whitespace().count();
        have_data = true;
        break;
    }
    if !have_data {
        return Ok((0, 0, Vec::new()));
    }

    let nthreads = if lines.len() < 4096 {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(lines.len() / 2048)
            .max(1)
    };
    if nthreads <= 1 {
        return loadtxt_serial(content);
    }

    // Parse one contiguous chunk of lines into (data_rows, values), or signal that the
    // input is malformed (so the caller replays the serial path for the exact error).
    let parse_chunk = |chunk: &[&str]| -> Option<(usize, Vec<f64>)> {
        let mut local = Vec::new();
        let mut r = 0usize;
        for line in chunk {
            let trimmed = loadtxt_line_payload(line);
            if trimmed.is_empty() || trimmed.starts_with('%') {
                continue;
            }
            let mut count = 0usize;
            for s in trimmed.split_whitespace() {
                match s.parse::<f64>() {
                    Ok(v) => local.push(v),
                    Err(_) => return None,
                }
                count += 1;
            }
            if count != cols {
                return None;
            }
            r += 1;
        }
        Some((r, local))
    };

    let chunk = lines.len().div_ceil(nthreads);
    let parse_chunk = &parse_chunk;
    let lines_ref = &lines;
    let chunk_results: Vec<Option<(usize, Vec<f64>)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..nthreads)
            .filter_map(|t| {
                let i0 = t * chunk;
                if i0 >= lines_ref.len() {
                    return None;
                }
                let i1 = (i0 + chunk).min(lines_ref.len());
                Some(scope.spawn(move || parse_chunk(&lines_ref[i0..i1])))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("loadtxt worker panicked"))
            .collect()
    });

    if chunk_results.iter().any(Option::is_none) {
        // Malformed input — replay the serial loop to reproduce the exact error.
        return loadtxt_serial(content);
    }

    let mut rows = 0usize;
    let total: usize = chunk_results
        .iter()
        .map(|c| c.as_ref().map_or(0, |(_, v)| v.len()))
        .sum();
    let mut data = Vec::with_capacity(total);
    for c in chunk_results {
        let (r, v) = c.expect("none handled above");
        rows += r;
        data.extend_from_slice(&v);
    }
    Ok((rows, cols, data))
}

/// Exact serial loadtxt — the byte-identical reference and the malformed-input fallback.
fn loadtxt_serial(content: &str) -> Result<(usize, usize, Vec<f64>), IoError> {
    let mut data = Vec::new();
    let mut cols = 0usize;
    let mut rows = 0usize;

    for line in content.lines() {
        let trimmed = loadtxt_line_payload(line);
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }

        // Parse fields straight into `data` (no per-line Vec<f64>); the row's column
        // count is the length delta. Matches the parallel loadtxt parse_chunk path —
        // byte-identical output (same deterministic parse order, same first-error
        // message, same column-consistency check).
        let start = data.len();
        for s in trimmed.split_whitespace() {
            let v = s
                .parse::<f64>()
                .map_err(|e| IoError::InvalidFormat(format!("parse error: {e}")))?;
            data.push(v);
        }
        let row_cols = data.len() - start;

        if rows == 0 {
            cols = row_cols;
        } else if row_cols != cols {
            return Err(IoError::InvalidFormat(format!(
                "row {rows} has {row_cols} columns, expected {cols}"
            )));
        }

        rows += 1;
    }

    Ok((rows, cols, data))
}

// Runtime switch to force the serial `savetxt` formatter for same-binary A/B
// benchmarks. Defaults off.
// CONTRACT: BYTE-IDENTICAL output either way. Workers format contiguous ROW
// ranges into private `String`s, joined in row order. The inline comment at
// the parallel branch already said "reproduces serial output byte-for-byte";
// this states it where a reader -- or the drqu7 ratchet -- looks first.
//
// The delimiter is emitted only BETWEEN columns within a row, so splitting by
// whole rows cannot move one. A mid-row split would.
#[doc(hidden)]
pub static SAVETXT_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Save a matrix as whitespace-delimited text.
///
/// Like `numpy.savetxt`.
pub fn savetxt(rows: usize, cols: usize, data: &[f64], delimiter: &str) -> Result<String, IoError> {
    if delimiter.contains(['\n', '\r']) {
        return Err(IoError::InvalidFormat(format!(
            "delimiter {:?} contains a newline and cannot be encoded safely",
            delimiter
        )));
    }
    let expected_len = checked_matrix_len(rows, cols, "text matrix")?;
    if data.len() != expected_len {
        return Err(IoError::InvalidFormat(format!(
            "data length {} doesn't match {}x{}",
            data.len(),
            rows,
            cols
        )));
    }
    // The f64 Display formatting dominates savetxt and is embarrassingly
    // parallel across rows. Each worker formats a contiguous range into a
    // private String; joining in row order reproduces serial output byte-for-byte.
    const SAVETXT_PAR_GATE: usize = 1 << 16;
    let n = expected_len;
    if n < SAVETXT_PAR_GATE || SAVETXT_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        let mut out = String::new();
        for r in 0..rows {
            for c in 0..cols {
                if c > 0 {
                    out.push_str(delimiter);
                }
                let _ = write!(out, "{}", data[r * cols + c]);
            }
            out.push('\n');
        }
        return Ok(out);
    }

    let nthreads = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
        .min(n / 16384)
        .min(rows)
        .max(1);
    if nthreads <= 1 {
        let mut out = String::new();
        for r in 0..rows {
            for c in 0..cols {
                if c > 0 {
                    out.push_str(delimiter);
                }
                let _ = write!(out, "{}", data[r * cols + c]);
            }
            out.push('\n');
        }
        return Ok(out);
    }

    let chunk = rows.div_ceil(nthreads);
    let mut parts: Vec<String> = (0..nthreads).map(|_| String::new()).collect();
    std::thread::scope(|scope| {
        for (t, slot) in parts.iter_mut().enumerate() {
            let r0 = t * chunk;
            let r1 = ((t + 1) * chunk).min(rows);
            scope.spawn(move || {
                if r0 >= r1 {
                    return;
                }
                let mut local = String::with_capacity((r1 - r0) * cols * 12);
                for r in r0..r1 {
                    for c in 0..cols {
                        if c > 0 {
                            local.push_str(delimiter);
                        }
                        let _ = write!(local, "{}", data[r * cols + c]);
                    }
                    local.push('\n');
                }
                *slot = local;
            });
        }
    });

    let total: usize = parts.iter().map(String::len).sum();
    let mut out = String::with_capacity(total);
    for p in &parts {
        out.push_str(p);
    }
    Ok(out)
}

/// Read a CSV file into rows of f64 values.
///
/// Simple CSV reader for numerical data.
pub type CsvResult = Result<(Option<Vec<String>>, Vec<Vec<f64>>), IoError>;

pub fn read_csv(content: &str, delimiter: char, has_header: bool) -> CsvResult {
    // Field parsing (`split(delimiter).trim().parse::<f64>()`) dominates read_csv on large
    // files. Each data row is an independent `Vec<f64>` with no cross-row state beyond the
    // fixed column count, so the rows parse on disjoint cores and concatenate in order —
    // byte-identical to the serial loop (same deterministic f64 parse, same row order).
    // Any anomaly (empty-with-header, header/row column mismatch, parse error) replays the
    // verbatim serial loop so the returned header/data and every Err are unchanged.
    let lines: Vec<&str> = content.lines().collect();

    let header: Option<Vec<String>> = if has_header {
        match lines.first() {
            Some(h) => Some(h.split(delimiter).map(|s| s.trim().to_string()).collect()),
            None => return read_csv_serial(content, delimiter, has_header),
        }
    } else {
        None
    };
    let header_cols = header.as_ref().map(std::vec::Vec::len);
    let data_start = usize::from(has_header);

    // Column count is fixed by the first non-comment data row, exactly as the serial loop.
    let mut expected_cols = None;
    for line in &lines[data_start..] {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        expected_cols = Some(trimmed.split(delimiter).count());
        break;
    }
    let expected_cols = match expected_cols {
        Some(c) => c,
        None => return Ok((header, Vec::new())),
    };
    // The serial loop reports the header/first-row mismatch specifically; defer to it.
    if let Some(hc) = header_cols
        && hc != expected_cols
    {
        return read_csv_serial(content, delimiter, has_header);
    }

    let data_lines = &lines[data_start..];
    let nthreads = if data_lines.len() < 4096 {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(data_lines.len() / 2048)
            .max(1)
    };
    if nthreads <= 1 {
        return read_csv_serial(content, delimiter, has_header);
    }

    let parse_chunk = |chunk: &[&str]| -> Option<Vec<Vec<f64>>> {
        let mut out = Vec::new();
        for line in chunk {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let mut row = Vec::new();
            for s in trimmed.split(delimiter) {
                match s.trim().parse::<f64>() {
                    Ok(v) => row.push(v),
                    Err(_) => return None,
                }
            }
            if row.len() != expected_cols {
                return None;
            }
            out.push(row);
        }
        Some(out)
    };

    let chunk = data_lines.len().div_ceil(nthreads);
    let parse_chunk = &parse_chunk;
    let chunk_results: Vec<Option<Vec<Vec<f64>>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..nthreads)
            .filter_map(|t| {
                let i0 = t * chunk;
                if i0 >= data_lines.len() {
                    return None;
                }
                let i1 = (i0 + chunk).min(data_lines.len());
                Some(scope.spawn(move || parse_chunk(&data_lines[i0..i1])))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("read_csv worker panicked"))
            .collect()
    });

    if chunk_results.iter().any(Option::is_none) {
        return read_csv_serial(content, delimiter, has_header);
    }

    let total: usize = chunk_results
        .iter()
        .map(|c| c.as_ref().map_or(0, Vec::len))
        .sum();
    let mut data = Vec::with_capacity(total);
    for c in chunk_results {
        data.extend(c.expect("none handled above"));
    }
    Ok((header, data))
}

/// Exact serial read_csv — the byte-identical reference and the malformed-input fallback.
fn read_csv_serial(content: &str, delimiter: char, has_header: bool) -> CsvResult {
    let mut lines = content.lines();
    let header = if has_header {
        Some(
            lines
                .next()
                .ok_or_else(|| {
                    IoError::InvalidFormat(
                        "CSV header row is required but the input is empty".to_string(),
                    )
                })?
                .split(delimiter)
                .map(|s| s.trim().to_string())
                .collect(),
        )
    } else {
        None
    };
    let header_cols = header.as_ref().map(std::vec::Vec::len);

    let mut data = Vec::new();
    let mut expected_cols = None;
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let row: Result<Vec<f64>, _> = trimmed
            .split(delimiter)
            .map(|s| s.trim().parse::<f64>())
            .collect();
        match row {
            Ok(r) => {
                if let Some(cols) = expected_cols {
                    if r.len() != cols {
                        return Err(IoError::InvalidFormat(format!(
                            "CSV row has {} columns, expected {cols}",
                            r.len()
                        )));
                    }
                } else {
                    if let Some(header_cols) = header_cols
                        && r.len() != header_cols
                    {
                        return Err(IoError::InvalidFormat(format!(
                            "CSV header has {header_cols} columns but first data row has {}",
                            r.len()
                        )));
                    }
                    expected_cols = Some(r.len());
                }
                data.push(r);
            }
            Err(e) => {
                return Err(IoError::InvalidFormat(format!("CSV parse error: {e}")));
            }
        }
    }

    Ok((header, data))
}

/// Write data to CSV format.
/// Runtime switch to force the serial `write_csv` formatter for same-binary A/B
/// benchmarks. Defaults off. `#[doc(hidden)]` — internal.
/// CONTRACT: BYTE-IDENTICAL output either way. The same per-row formatter runs
/// in both arms; the parallel one only chooses which core formats which row,
/// and chunks are concatenated in row order. No arithmetic, so nothing to
/// reassociate.
#[doc(hidden)]
pub static WRITE_CSV_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn write_csv(
    header: Option<&[&str]>,
    data: &[Vec<f64>],
    delimiter: char,
) -> Result<String, IoError> {
    let mut out = String::new();
    let header_cols = header.map(<[&str]>::len);
    if let Some(h) = header {
        for cell in h {
            if cell.contains(['\n', '\r']) {
                return Err(IoError::InvalidFormat(format!(
                    "CSV header cell {:?} contains a newline and cannot be encoded safely",
                    cell
                )));
            }
            if cell.contains(delimiter) {
                return Err(IoError::InvalidFormat(format!(
                    "CSV header cell {:?} contains the delimiter {:?} and cannot be encoded safely",
                    cell, delimiter
                )));
            }
        }
        out.push_str(&h.join(&delimiter.to_string()));
        out.push('\n');
    }
    // Validate row shapes FIRST (returning the first error in row order, identical to
    // the fused loop) so the subsequent formatting has no early-exit and can be split
    // across threads. Pure O(rows) length checks — no f64 formatting.
    let mut expected_cols = None;
    for row in data {
        if let Some(cols) = expected_cols {
            if row.len() != cols {
                return Err(IoError::InvalidFormat(format!(
                    "CSV row has {} columns, expected {cols}",
                    row.len()
                )));
            }
        } else {
            if let Some(header_cols) = header_cols
                && row.len() != header_cols
            {
                return Err(IoError::InvalidFormat(format!(
                    "CSV header has {header_cols} columns but first data row has {}",
                    row.len()
                )));
            }
            expected_cols = Some(row.len());
        }
    }

    // The f64 Display formatting dominates; each row is independent, so format contiguous
    // row ranges into private Strings and join in row order (BIT-FOR-BIT the serial
    // output). scipy/pandas CSV writers are single-threaded. Serial gate BEFORE the
    // available_parallelism syscall (per-call syscall-tax lesson).
    let fmt_row = |out: &mut String, row: &[f64]| {
        for (idx, value) in row.iter().enumerate() {
            if idx > 0 {
                out.push(delimiter);
            }
            let _ = write!(out, "{value}");
        }
        out.push('\n');
    };
    const CSV_PAR_GATE: usize = 1 << 16;
    let total = data.len().saturating_mul(expected_cols.unwrap_or(0));
    if total < CSV_PAR_GATE || WRITE_CSV_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        for row in data {
            fmt_row(&mut out, row);
        }
        return Ok(out);
    }

    let nthreads = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
        .min(total / 16384)
        .min(data.len())
        .max(1);
    if nthreads <= 1 {
        for row in data {
            fmt_row(&mut out, row);
        }
        return Ok(out);
    }
    let chunk = data.len().div_ceil(nthreads);
    let mut parts: Vec<String> = (0..nthreads).map(|_| String::new()).collect();
    let fmt_row_ref = &fmt_row;
    std::thread::scope(|scope| {
        for (t, slot) in parts.iter_mut().enumerate() {
            let r0 = t * chunk;
            let r1 = ((t + 1) * chunk).min(data.len());
            scope.spawn(move || {
                if r0 >= r1 {
                    return;
                }
                let mut local = String::with_capacity((r1 - r0) * expected_cols.unwrap_or(0) * 12);
                for row in &data[r0..r1] {
                    fmt_row_ref(&mut local, row);
                }
                *slot = local;
            });
        }
    });
    let parts_len: usize = parts.iter().map(String::len).sum();
    out.reserve(parts_len);
    for p in &parts {
        out.push_str(p);
    }
    Ok(out)
}

/// Read a simple JSON array of numbers.
pub fn read_json_array(content: &str) -> Result<Vec<f64>, IoError> {
    let trimmed = content.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Err(IoError::InvalidFormat("expected JSON array".to_string()));
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f64>()
                .map_err(|e| IoError::InvalidFormat(format!("JSON parse error: {e}")))
                .and_then(|v| {
                    if v.is_finite() {
                        Ok(v)
                    } else {
                        Err(IoError::InvalidFormat(format!(
                            "JSON parse error: non-finite value {v}"
                        )))
                    }
                })
        })
        .collect()
}

/// Runtime switch to force the serial `write_json_array` formatter for same-binary A/B
/// benchmarks. Defaults off.
/// CONTRACT: BYTE-IDENTICAL output either way, and this one has a genuine
/// boundary hazard worth naming. The `", "` separator is emitted before every
/// element EXCEPT the first, so a chunked writer has to get the seam right:
/// element 0 of chunk 2 is not element 0 of the array and still needs its
/// separator. Values themselves are independent `f64` `Display` calls with no
/// arithmetic between them, so the ONLY way this arm can differ is by
/// mishandling that seam -- which is exactly what its A/B test exercises.
#[doc(hidden)]
pub static WRITE_JSON_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Write a vector as a JSON array.
pub fn write_json_array(data: &[f64]) -> Result<String, IoError> {
    if let Some((idx, value)) = data
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| !v.is_finite())
    {
        return Err(IoError::InvalidFormat(format!(
            "JSON array value at index {idx} is not finite: {value}"
        )));
    }

    // f64 Display dominates here. Each value is independent, so format
    // contiguous ranges into private Strings and join chunks in order.
    const JSON_PAR_GATE: usize = 1 << 16;
    let n = data.len();
    if n < JSON_PAR_GATE || WRITE_JSON_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        let mut out = String::with_capacity(n * 8 + 2);
        out.push('[');
        for (idx, value) in data.iter().enumerate() {
            if idx > 0 {
                out.push_str(", ");
            }
            let _ = write!(out, "{value}");
        }
        out.push(']');
        return Ok(out);
    }

    let nthreads = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
        .min(n / 16384)
        .min(n)
        .max(1);
    if nthreads <= 1 {
        let mut out = String::with_capacity(n * 8 + 2);
        out.push('[');
        for (idx, value) in data.iter().enumerate() {
            if idx > 0 {
                out.push_str(", ");
            }
            let _ = write!(out, "{value}");
        }
        out.push(']');
        return Ok(out);
    }
    let chunk = n.div_ceil(nthreads);
    let mut parts: Vec<String> = (0..nthreads).map(|_| String::new()).collect();
    std::thread::scope(|scope| {
        for (t, slot) in parts.iter_mut().enumerate() {
            let i0 = t * chunk;
            let i1 = ((t + 1) * chunk).min(n);
            scope.spawn(move || {
                if i0 >= i1 {
                    return;
                }
                let mut local = String::with_capacity((i1 - i0) * 8);
                for (k, value) in data[i0..i1].iter().enumerate() {
                    if k > 0 {
                        local.push_str(", ");
                    }
                    let _ = write!(local, "{value}");
                }
                *slot = local;
            });
        }
    });
    let total: usize = parts.iter().map(String::len).sum();
    let mut out = String::with_capacity(total + parts.len() * 2 + 2);
    out.push('[');
    let mut first = true;
    for p in &parts {
        if p.is_empty() {
            continue;
        }
        if !first {
            out.push_str(", ");
        }
        out.push_str(p);
        first = false;
    }
    out.push(']');
    Ok(out)
}

/// Read a simple NPY-like header (shape + dtype) from text representation.
///
/// Returns (shape, data).
pub fn read_npy_text(content: &str) -> Result<(Vec<usize>, Vec<f64>), IoError> {
    let mut lines = content.lines();

    // Read shape line
    let shape_line = lines
        .next()
        .ok_or_else(|| IoError::InvalidFormat("missing shape line".to_string()))?;
    let trimmed_shape = shape_line.trim();
    if trimmed_shape.is_empty() {
        return Err(IoError::InvalidFormat(
            "shape declaration must contain at least one dimension".to_string(),
        ));
    }
    if trimmed_shape
        .split(',')
        .any(|segment| segment.trim().is_empty())
    {
        return Err(IoError::InvalidFormat(
            "shape declaration contains an empty dimension".to_string(),
        ));
    }
    let shape: Result<Vec<usize>, _> = shape_line
        .trim()
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect();
    let shape = shape.map_err(|e| IoError::InvalidFormat(format!("bad shape: {e}")))?;
    // Read data
    let mut data = Vec::new();
    for line in lines {
        for val in line.split_whitespace() {
            let v: f64 = val
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("bad value: {e}")))?;
            data.push(v);
        }
    }

    let expected_len = shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| IoError::InvalidFormat("shape product overflowed usize".to_string()))
    })?;
    if data.len() != expected_len {
        return Err(IoError::InvalidFormat(format!(
            "shape {:?} expects {expected_len} values but found {}",
            shape,
            data.len()
        )));
    }

    Ok((shape, data))
}

// ══════════════════════════════════════════════════════════════════════
// NetCDF classic v3
// ══════════════════════════════════════════════════════════════════════

const NC_DIMENSION: u32 = 10;
const NC_VARIABLE: u32 = 11;
const NC_ATTRIBUTE: u32 = 12;
const NC_BYTE: u32 = 1;
const NC_CHAR: u32 = 2;
const NC_SHORT: u32 = 3;
const NC_INT: u32 = 4;
const NC_FLOAT: u32 = 5;
const NC_DOUBLE: u32 = 6;

/// NetCDF classic scalar type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetcdfType {
    Byte,
    Char,
    Short,
    Int,
    Float,
    Double,
}

/// NetCDF dimension. `len == None` represents the classic unlimited dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetcdfDimension {
    pub name: String,
    pub len: Option<usize>,
}

/// NetCDF typed payload.
#[derive(Debug, Clone, PartialEq)]
pub enum NetcdfValue {
    Byte(Vec<i8>),
    Char(String),
    Short(Vec<i16>),
    Int(Vec<i32>),
    Float(Vec<f32>),
    Double(Vec<f64>),
}

impl NetcdfValue {
    fn value_type(&self) -> NetcdfType {
        match self {
            Self::Byte(_) => NetcdfType::Byte,
            Self::Char(_) => NetcdfType::Char,
            Self::Short(_) => NetcdfType::Short,
            Self::Int(_) => NetcdfType::Int,
            Self::Float(_) => NetcdfType::Float,
            Self::Double(_) => NetcdfType::Double,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Byte(v) => v.len(),
            Self::Char(s) => s.len(),
            Self::Short(v) => v.len(),
            Self::Int(v) => v.len(),
            Self::Float(v) => v.len(),
            Self::Double(v) => v.len(),
        }
    }
}

/// NetCDF attribute.
#[derive(Debug, Clone, PartialEq)]
pub struct NetcdfAttribute {
    pub name: String,
    pub value: NetcdfValue,
}

/// NetCDF variable. `dim_ids` indexes into `NetcdfFile::dimensions`.
#[derive(Debug, Clone, PartialEq)]
pub struct NetcdfVariable {
    pub name: String,
    pub dim_ids: Vec<usize>,
    pub attributes: Vec<NetcdfAttribute>,
    pub data: NetcdfValue,
}

/// Parsed NetCDF classic file.
#[derive(Debug, Clone, PartialEq)]
pub struct NetcdfFile {
    pub dimensions: Vec<NetcdfDimension>,
    pub attributes: Vec<NetcdfAttribute>,
    pub variables: Vec<NetcdfVariable>,
}

/// SciPy-compatible spelling for [`NetcdfFile`].
#[allow(non_camel_case_types)]
pub type netcdf_file = NetcdfFile;

/// SciPy-compatible spelling for [`NetcdfVariable`].
#[allow(non_camel_case_types)]
pub type netcdf_variable = NetcdfVariable;

#[derive(Debug, Clone)]
struct NetcdfVariableHeader {
    name: String,
    dim_ids: Vec<usize>,
    attributes: Vec<NetcdfAttribute>,
    value_type: NetcdfType,
    begin: usize,
}

struct NetcdfReader<'a> {
    bytes: &'a [u8],
    offset: usize,
    version: u8,
}

/// Read a NetCDF classic or 64-bit-offset file from bytes.
///
/// This covers the fixed-size NetCDF v3 subset exposed by
/// `scipy.io.netcdf_file`: dimensions, global attributes, variables, and
/// primitive numeric/character arrays. Unlimited record variables are rejected
/// until the record-interleaving path is implemented.
pub fn read_netcdf_classic(bytes: &[u8]) -> Result<NetcdfFile, IoError> {
    if bytes.len() < 4 {
        return Err(IoError::InvalidFormat("NetCDF file too short".to_string()));
    }
    if &bytes[0..3] != b"CDF" {
        return Err(IoError::InvalidFormat(
            "NetCDF missing CDF magic".to_string(),
        ));
    }
    let version = bytes[3];
    if version != 1 && version != 2 {
        return Err(IoError::UnsupportedFeature(format!(
            "NetCDF version byte {version} is not supported"
        )));
    }

    let mut reader = NetcdfReader {
        bytes,
        offset: 4,
        version,
    };
    let numrecs = read_netcdf_u32(&mut reader)? as usize;
    let dimensions = read_netcdf_dimensions(&mut reader)?;
    if dimensions.iter().filter(|dim| dim.len.is_none()).count() > 1 {
        return Err(IoError::InvalidFormat(
            "NetCDF classic permits at most one unlimited dimension".to_string(),
        ));
    }
    let attributes = read_netcdf_attributes(&mut reader)?;
    let variable_headers = read_netcdf_variable_headers(&mut reader)?;

    let mut variables = Vec::with_capacity(variable_headers.len());
    for header in variable_headers {
        let element_count = netcdf_element_count_for_dims(&dimensions, &header.dim_ids, numrecs)?;
        if header
            .dim_ids
            .iter()
            .any(|&dim_id| dimensions[dim_id].len.is_none())
        {
            return Err(IoError::UnsupportedFeature(
                "NetCDF unlimited record variables are not supported".to_string(),
            ));
        }
        let raw_len = netcdf_value_raw_len(header.value_type, element_count)?;
        let end = header.begin.checked_add(raw_len).ok_or_else(|| {
            IoError::InvalidFormat("NetCDF variable payload offset overflowed usize".to_string())
        })?;
        let payload = bytes.get(header.begin..end).ok_or_else(|| {
            IoError::InvalidFormat(format!(
                "NetCDF variable '{}' payload extends past file",
                header.name
            ))
        })?;
        let data = decode_netcdf_values(header.value_type, payload, element_count, &header.name)?;
        variables.push(NetcdfVariable {
            name: header.name,
            dim_ids: header.dim_ids,
            attributes: header.attributes,
            data,
        });
    }

    Ok(NetcdfFile {
        dimensions,
        attributes,
        variables,
    })
}

/// Runtime switch to retain the former payload-encoding header-size path for
/// same-binary A/B benchmarks. Defaults off.
/// CONTRACT: BYTE-IDENTICAL WHOLE FILE either way, and note that "same length"
/// would be the weaker claim. The legacy arm fully encodes each variable's
/// padded payload just to measure its length; the default computes that length
/// directly with `netcdf_padded_value_len`. Because the length is then WRITTEN
/// INTO the header, a disagreement of even one byte changes the file rather
/// than merely the work done to produce it -- so the two paths must agree
/// exactly, and the test compares whole files.
///
/// This is a work-elimination lever, not a parallel one: no threads, no float
/// arithmetic, and the padding rule is the same function in both arms.
#[doc(hidden)]
pub static WRITE_NETCDF_FORCE_REDUNDANT_HEADER_ENCODING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Write a fixed-size NetCDF classic file.
///
/// The writer emits NetCDF classic v1 files and fails closed for unlimited
/// dimensions or payload offsets that require the 64-bit-offset variant.
pub fn write_netcdf_classic(file: &NetcdfFile) -> Result<Vec<u8>, IoError> {
    validate_netcdf_fixed_file(file)?;

    let placeholder_begins = vec![0usize; file.variables.len()];
    let header = encode_netcdf_header(file, &placeholder_begins)?;
    let mut cursor = align4(header.len());
    let mut begins = Vec::with_capacity(file.variables.len());
    let mut payloads = Vec::with_capacity(file.variables.len());
    for variable in &file.variables {
        let payload = encode_netcdf_padded_values(&variable.data)?;
        begins.push(cursor);
        cursor = cursor.checked_add(payload.len()).ok_or_else(|| {
            IoError::InvalidFormat("NetCDF output size overflowed usize".to_string())
        })?;
        payloads.push(payload);
    }
    if begins.iter().any(|&begin| u32::try_from(begin).is_err()) {
        return Err(IoError::UnsupportedFeature(
            "NetCDF output requires 64-bit variable offsets".to_string(),
        ));
    }

    let mut out = encode_netcdf_header(file, &begins)?;
    while out.len() < align4(out.len()) {
        out.push(0);
    }
    for payload in payloads {
        out.extend_from_slice(&payload);
    }
    Ok(out)
}

/// Alias matching the SciPy surface name.
pub fn netcdf_file_read(bytes: &[u8]) -> Result<NetcdfFile, IoError> {
    read_netcdf_classic(bytes)
}

/// Alias matching the SciPy surface name.
pub fn netcdf_file_write(file: &NetcdfFile) -> Result<Vec<u8>, IoError> {
    write_netcdf_classic(file)
}

fn netcdf_type_code(value_type: NetcdfType) -> u32 {
    match value_type {
        NetcdfType::Byte => NC_BYTE,
        NetcdfType::Char => NC_CHAR,
        NetcdfType::Short => NC_SHORT,
        NetcdfType::Int => NC_INT,
        NetcdfType::Float => NC_FLOAT,
        NetcdfType::Double => NC_DOUBLE,
    }
}

fn netcdf_type_from_code(code: u32) -> Result<NetcdfType, IoError> {
    match code {
        NC_BYTE => Ok(NetcdfType::Byte),
        NC_CHAR => Ok(NetcdfType::Char),
        NC_SHORT => Ok(NetcdfType::Short),
        NC_INT => Ok(NetcdfType::Int),
        NC_FLOAT => Ok(NetcdfType::Float),
        NC_DOUBLE => Ok(NetcdfType::Double),
        other => Err(IoError::UnsupportedFeature(format!(
            "NetCDF type code {other} is not supported"
        ))),
    }
}

fn netcdf_type_size(value_type: NetcdfType) -> usize {
    match value_type {
        NetcdfType::Byte | NetcdfType::Char => 1,
        NetcdfType::Short => 2,
        NetcdfType::Int | NetcdfType::Float => 4,
        NetcdfType::Double => 8,
    }
}

fn align4(value: usize) -> usize {
    value + ((4 - (value % 4)) % 4)
}

fn netcdf_value_raw_len(value_type: NetcdfType, count: usize) -> Result<usize, IoError> {
    count
        .checked_mul(netcdf_type_size(value_type))
        .ok_or_else(|| {
            IoError::InvalidFormat("NetCDF value byte length overflowed usize".to_string())
        })
}

fn netcdf_padded_value_len(value: &NetcdfValue) -> Result<usize, IoError> {
    let raw_len = netcdf_value_raw_len(value.value_type(), value.len())?;
    raw_len
        .checked_add((4 - (raw_len % 4)) % 4)
        .ok_or_else(|| IoError::InvalidFormat("NetCDF padded value length overflowed usize".into()))
}

fn read_netcdf_u32(reader: &mut NetcdfReader<'_>) -> Result<u32, IoError> {
    let end = reader
        .offset
        .checked_add(4)
        .ok_or_else(|| IoError::InvalidFormat("NetCDF u32 offset overflowed usize".to_string()))?;
    let slice = reader
        .bytes
        .get(reader.offset..end)
        .ok_or_else(|| IoError::InvalidFormat("truncated NetCDF u32".to_string()))?;
    reader.offset = end;
    Ok(u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_netcdf_begin(reader: &mut NetcdfReader<'_>) -> Result<usize, IoError> {
    if reader.version == 1 {
        Ok(read_netcdf_u32(reader)? as usize)
    } else {
        let hi = read_netcdf_u32(reader)? as u64;
        let lo = read_netcdf_u32(reader)? as u64;
        let value = (hi << 32) | lo;
        usize::try_from(value).map_err(|_| {
            IoError::InvalidFormat(format!(
                "NetCDF 64-bit offset {value} cannot be represented as usize"
            ))
        })
    }
}

fn read_netcdf_name(reader: &mut NetcdfReader<'_>, context: &str) -> Result<String, IoError> {
    let len = read_netcdf_u32(reader)? as usize;
    let end = reader.offset.checked_add(len).ok_or_else(|| {
        IoError::InvalidFormat(format!("NetCDF {context} name offset overflowed usize"))
    })?;
    let name_bytes = reader
        .bytes
        .get(reader.offset..end)
        .ok_or_else(|| IoError::InvalidFormat(format!("truncated NetCDF {context} name")))?;
    reader.offset = align4(end);
    if reader.offset > reader.bytes.len() {
        return Err(IoError::InvalidFormat(format!(
            "truncated NetCDF {context} name padding"
        )));
    }
    let name = String::from_utf8(name_bytes.to_vec())
        .map_err(|e| IoError::InvalidFormat(format!("NetCDF {context} name is not UTF-8: {e}")))?;
    validate_netcdf_name(&name, context)?;
    Ok(name)
}

fn read_netcdf_list_count(
    reader: &mut NetcdfReader<'_>,
    expected_tag: u32,
    context: &str,
) -> Result<usize, IoError> {
    let tag = read_netcdf_u32(reader)?;
    let count = read_netcdf_u32(reader)? as usize;
    if tag == 0 && count == 0 {
        return Ok(0);
    }
    if tag != expected_tag {
        return Err(IoError::InvalidFormat(format!(
            "NetCDF {context} list tag {tag} does not match expected {expected_tag}"
        )));
    }
    Ok(count)
}

fn read_netcdf_dimensions(reader: &mut NetcdfReader<'_>) -> Result<Vec<NetcdfDimension>, IoError> {
    let count = read_netcdf_list_count(reader, NC_DIMENSION, "dimension")?;
    let mut dimensions = Vec::with_capacity(count);
    for _ in 0..count {
        let name = read_netcdf_name(reader, "dimension")?;
        let raw_len = read_netcdf_u32(reader)?;
        let len = if raw_len == 0 {
            None
        } else {
            Some(raw_len as usize)
        };
        dimensions.push(NetcdfDimension { name, len });
    }
    Ok(dimensions)
}

fn read_netcdf_attributes(reader: &mut NetcdfReader<'_>) -> Result<Vec<NetcdfAttribute>, IoError> {
    let count = read_netcdf_list_count(reader, NC_ATTRIBUTE, "attribute")?;
    let mut attributes = Vec::with_capacity(count);
    for _ in 0..count {
        let name = read_netcdf_name(reader, "attribute")?;
        let value_type = netcdf_type_from_code(read_netcdf_u32(reader)?)?;
        let value_count = read_netcdf_u32(reader)? as usize;
        let raw_len = netcdf_value_raw_len(value_type, value_count)?;
        let value_end = reader.offset.checked_add(raw_len).ok_or_else(|| {
            IoError::InvalidFormat("NetCDF attribute payload offset overflowed usize".to_string())
        })?;
        let payload = reader.bytes.get(reader.offset..value_end).ok_or_else(|| {
            IoError::InvalidFormat(format!("truncated NetCDF attribute '{name}'"))
        })?;
        let value = decode_netcdf_values(value_type, payload, value_count, &name)?;
        reader.offset = align4(value_end);
        if reader.offset > reader.bytes.len() {
            return Err(IoError::InvalidFormat(format!(
                "truncated NetCDF attribute '{name}' padding"
            )));
        }
        attributes.push(NetcdfAttribute { name, value });
    }
    Ok(attributes)
}

fn read_netcdf_variable_headers(
    reader: &mut NetcdfReader<'_>,
) -> Result<Vec<NetcdfVariableHeader>, IoError> {
    let count = read_netcdf_list_count(reader, NC_VARIABLE, "variable")?;
    let mut variables = Vec::with_capacity(count);
    for _ in 0..count {
        let name = read_netcdf_name(reader, "variable")?;
        let dim_count = read_netcdf_u32(reader)? as usize;
        let mut dim_ids = Vec::with_capacity(dim_count);
        for _ in 0..dim_count {
            dim_ids.push(read_netcdf_u32(reader)? as usize);
        }
        let attributes = read_netcdf_attributes(reader)?;
        let value_type = netcdf_type_from_code(read_netcdf_u32(reader)?)?;
        let _vsize = read_netcdf_u32(reader)?;
        let begin = read_netcdf_begin(reader)?;
        variables.push(NetcdfVariableHeader {
            name,
            dim_ids,
            attributes,
            value_type,
            begin,
        });
    }
    Ok(variables)
}

fn netcdf_element_count_for_dims(
    dimensions: &[NetcdfDimension],
    dim_ids: &[usize],
    numrecs: usize,
) -> Result<usize, IoError> {
    if dim_ids.is_empty() {
        return Ok(1);
    }
    let mut count = 1usize;
    for &dim_id in dim_ids {
        let dimension = dimensions.get(dim_id).ok_or_else(|| {
            IoError::InvalidFormat(format!(
                "NetCDF variable references bad dimension id {dim_id}"
            ))
        })?;
        let dim_len = dimension.len.unwrap_or(numrecs);
        if dim_len == 0 {
            return Err(IoError::InvalidFormat(format!(
                "NetCDF dimension '{}' has zero length",
                dimension.name
            )));
        }
        count = count.checked_mul(dim_len).ok_or_else(|| {
            IoError::InvalidFormat("NetCDF variable shape overflowed usize".to_string())
        })?;
    }
    Ok(count)
}

fn decode_netcdf_values(
    value_type: NetcdfType,
    payload: &[u8],
    count: usize,
    name: &str,
) -> Result<NetcdfValue, IoError> {
    let expected_len = netcdf_value_raw_len(value_type, count)?;
    if payload.len() < expected_len {
        return Err(IoError::InvalidFormat(format!(
            "NetCDF value '{name}' expected {expected_len} bytes but found {}",
            payload.len()
        )));
    }
    match value_type {
        NetcdfType::Byte => Ok(NetcdfValue::Byte(
            payload[..count].iter().map(|&byte| byte as i8).collect(),
        )),
        NetcdfType::Char => {
            let s = String::from_utf8(payload[..count].to_vec()).map_err(|e| {
                IoError::InvalidFormat(format!("NetCDF char value '{name}' is not UTF-8: {e}"))
            })?;
            Ok(NetcdfValue::Char(s))
        }
        NetcdfType::Short => {
            let mut values = Vec::with_capacity(count);
            for &chunk in payload[..expected_len].as_chunks::<2>().0 {
                values.push(i16::from_be_bytes(chunk));
            }
            Ok(NetcdfValue::Short(values))
        }
        NetcdfType::Int => {
            let mut values = Vec::with_capacity(count);
            for &chunk in payload[..expected_len].as_chunks::<4>().0 {
                values.push(i32::from_be_bytes(chunk));
            }
            Ok(NetcdfValue::Int(values))
        }
        NetcdfType::Float => {
            let mut values = Vec::with_capacity(count);
            for &chunk in payload[..expected_len].as_chunks::<4>().0 {
                values.push(f32::from_be_bytes(chunk));
            }
            Ok(NetcdfValue::Float(values))
        }
        NetcdfType::Double => {
            let mut values = Vec::with_capacity(count);
            for &chunk in payload[..expected_len].as_chunks::<8>().0 {
                values.push(f64::from_be_bytes(chunk));
            }
            Ok(NetcdfValue::Double(values))
        }
    }
}

fn validate_netcdf_name(name: &str, context: &str) -> Result<(), IoError> {
    if name.is_empty() {
        return Err(IoError::InvalidFormat(format!(
            "NetCDF {context} name cannot be empty"
        )));
    }
    if name.contains('\0') {
        return Err(IoError::InvalidFormat(format!(
            "NetCDF {context} name '{}' contains NUL",
            name.escape_debug()
        )));
    }
    Ok(())
}

fn validate_netcdf_fixed_file(file: &NetcdfFile) -> Result<(), IoError> {
    let unlimited_count = file
        .dimensions
        .iter()
        .filter(|dimension| dimension.len.is_none())
        .count();
    if unlimited_count > 0 {
        return Err(IoError::UnsupportedFeature(
            "NetCDF writer does not yet support unlimited record dimensions".to_string(),
        ));
    }
    for dimension in &file.dimensions {
        validate_netcdf_name(&dimension.name, "dimension")?;
        if dimension.len == Some(0) {
            return Err(IoError::InvalidFormat(format!(
                "NetCDF dimension '{}' has zero length",
                dimension.name
            )));
        }
    }
    for attribute in &file.attributes {
        validate_netcdf_name(&attribute.name, "attribute")?;
    }
    for variable in &file.variables {
        validate_netcdf_name(&variable.name, "variable")?;
        for attribute in &variable.attributes {
            validate_netcdf_name(&attribute.name, "attribute")?;
        }
        let expected = netcdf_element_count_for_dims(&file.dimensions, &variable.dim_ids, 0)?;
        if variable.data.len() != expected {
            return Err(IoError::InvalidFormat(format!(
                "NetCDF variable '{}' expected {expected} values from its dimensions but found {}",
                variable.name,
                variable.data.len()
            )));
        }
    }
    Ok(())
}

fn write_netcdf_u32(out: &mut Vec<u8>, value: usize, context: &str) -> Result<(), IoError> {
    let value = u32::try_from(value).map_err(|_| {
        IoError::InvalidFormat(format!("NetCDF {context} {value} exceeds u32 range"))
    })?;
    out.extend_from_slice(&value.to_be_bytes());
    Ok(())
}

fn write_netcdf_name(out: &mut Vec<u8>, name: &str, context: &str) -> Result<(), IoError> {
    validate_netcdf_name(name, context)?;
    write_netcdf_u32(out, name.len(), "name length")?;
    out.extend_from_slice(name.as_bytes());
    while !out.len().is_multiple_of(4) {
        out.push(0);
    }
    Ok(())
}

fn encode_netcdf_header(file: &NetcdfFile, begins: &[usize]) -> Result<Vec<u8>, IoError> {
    if begins.len() != file.variables.len() {
        return Err(IoError::InvalidFormat(
            "NetCDF begin-offset count does not match variable count".to_string(),
        ));
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"CDF");
    out.push(1);
    out.extend_from_slice(&0u32.to_be_bytes());

    if file.dimensions.is_empty() {
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
    } else {
        out.extend_from_slice(&NC_DIMENSION.to_be_bytes());
        write_netcdf_u32(&mut out, file.dimensions.len(), "dimension count")?;
        for dimension in &file.dimensions {
            write_netcdf_name(&mut out, &dimension.name, "dimension")?;
            let len = dimension.len.ok_or_else(|| {
                IoError::UnsupportedFeature(
                    "NetCDF classic writer does not support unlimited dimensions".to_string(),
                )
            })?;
            write_netcdf_u32(&mut out, len, "dimension length")?;
        }
    }

    encode_netcdf_attribute_list(&mut out, &file.attributes)?;

    if file.variables.is_empty() {
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
    } else {
        out.extend_from_slice(&NC_VARIABLE.to_be_bytes());
        write_netcdf_u32(&mut out, file.variables.len(), "variable count")?;
        for (idx, variable) in file.variables.iter().enumerate() {
            write_netcdf_name(&mut out, &variable.name, "variable")?;
            write_netcdf_u32(&mut out, variable.dim_ids.len(), "variable dimension count")?;
            for &dim_id in &variable.dim_ids {
                if dim_id >= file.dimensions.len() {
                    return Err(IoError::InvalidFormat(format!(
                        "NetCDF variable '{}' references bad dimension id {dim_id}",
                        variable.name
                    )));
                }
                write_netcdf_u32(&mut out, dim_id, "variable dimension id")?;
            }
            encode_netcdf_attribute_list(&mut out, &variable.attributes)?;
            out.extend_from_slice(&netcdf_type_code(variable.data.value_type()).to_be_bytes());
            let value_size = if WRITE_NETCDF_FORCE_REDUNDANT_HEADER_ENCODING
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                encode_netcdf_padded_values(&variable.data)?.len()
            } else {
                netcdf_padded_value_len(&variable.data)?
            };
            write_netcdf_u32(&mut out, value_size, "variable byte size")?;
            write_netcdf_u32(&mut out, begins[idx], "variable begin offset")?;
        }
    }
    Ok(out)
}

fn encode_netcdf_attribute_list(
    out: &mut Vec<u8>,
    attributes: &[NetcdfAttribute],
) -> Result<(), IoError> {
    if attributes.is_empty() {
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        return Ok(());
    }
    out.extend_from_slice(&NC_ATTRIBUTE.to_be_bytes());
    write_netcdf_u32(out, attributes.len(), "attribute count")?;
    for attribute in attributes {
        write_netcdf_name(out, &attribute.name, "attribute")?;
        out.extend_from_slice(&netcdf_type_code(attribute.value.value_type()).to_be_bytes());
        write_netcdf_u32(out, attribute.value.len(), "attribute value count")?;
        out.extend_from_slice(&encode_netcdf_padded_values(&attribute.value)?);
    }
    Ok(())
}

fn encode_netcdf_padded_values(value: &NetcdfValue) -> Result<Vec<u8>, IoError> {
    let mut out = Vec::new();
    match value {
        NetcdfValue::Byte(values) => {
            out.reserve(values.len());
            for &value in values {
                out.push(value as u8);
            }
        }
        NetcdfValue::Char(value) => out.extend_from_slice(value.as_bytes()),
        NetcdfValue::Short(values) => {
            out.reserve(values.len() * 2);
            for &value in values {
                out.extend_from_slice(&value.to_be_bytes());
            }
        }
        NetcdfValue::Int(values) => {
            out.reserve(values.len() * 4);
            for &value in values {
                out.extend_from_slice(&value.to_be_bytes());
            }
        }
        NetcdfValue::Float(values) => {
            out.reserve(values.len() * 4);
            for &value in values {
                out.extend_from_slice(&value.to_be_bytes());
            }
        }
        NetcdfValue::Double(values) => {
            out.reserve(values.len() * 8);
            for &value in values {
                out.extend_from_slice(&value.to_be_bytes());
            }
        }
    }
    while !out.len().is_multiple_of(4) {
        out.push(0);
    }
    Ok(out)
}

// ══════════════════════════════════════════════════════════════════════
// Harwell-Boeing sparse matrix format
// ══════════════════════════════════════════════════════════════════════

/// Harwell-Boeing matrix data type from the title-line "Type" code (RUA/RSA/etc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HbType {
    /// "R" — real values, "U" — unsymmetric, "A" — assembled.
    RealUnsymmetricAssembled,
    /// "R" — real values, "S" — symmetric, "A" — assembled. Only the lower
    /// triangle is stored; callers wanting the full matrix must symmetrize.
    RealSymmetricAssembled,
}

/// Result of reading a Harwell-Boeing sparse matrix.
///
/// Returns the matrix in CSC (compressed sparse column) form: the canonical
/// on-disk Harwell-Boeing layout. `col_ptr` has `cols + 1` entries; row
/// indices and values are 0-indexed (the on-disk format is 1-indexed).
#[derive(Debug, Clone, PartialEq)]
pub struct HbMatrix {
    pub title: String,
    pub key: String,
    pub matrix_type: HbType,
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
    pub values: Vec<f64>,
}

/// Read a real, assembled Harwell-Boeing sparse matrix file.
///
/// Supports the dominant subset of `scipy.io.hb_read`: real (R), assembled (A),
/// either unsymmetric (U) or symmetric (S). The unassembled (E), pattern-only
/// (P), and complex (C) variants intentionally return an UnsupportedFeature
/// error and are tracked under the parent bead `frankenscipy-vsas0`.
///
/// The on-disk format uses 1-based indexing per the spec; the returned
/// `col_ptr` and `row_idx` are converted to 0-based for parity with the
/// rest of `fsci-io` and `fsci-sparse`.
pub fn read_harwell_boeing(content: &str) -> Result<HbMatrix, IoError> {
    let mut lines = content.lines();
    let title_line = lines.next().ok_or_else(|| {
        IoError::InvalidFormat("Harwell-Boeing file missing title line".to_string())
    })?;
    if !title_line.is_ascii() {
        return Err(IoError::InvalidFormat(
            "Harwell-Boeing title/key card must be ASCII".to_string(),
        ));
    }
    if title_line.len() < 72 {
        return Err(IoError::InvalidFormat(format!(
            "Harwell-Boeing title line must be ≥72 chars, got {}",
            title_line.len()
        )));
    }
    let title = title_line[..72].trim_end().to_string();
    let key = title_line[72..].trim().to_string();

    let totcrd_line = lines.next().ok_or_else(|| {
        IoError::InvalidFormat("Harwell-Boeing file missing totcrd line".to_string())
    })?;
    let counts = parse_hb_int_fields(totcrd_line, 5, "totcrd")?;
    let _totcrd = hb_usize_field(counts[0], "TOTCRD")?;
    let ptrcrd = hb_usize_field(counts[1], "PTRCRD")?;
    let indcrd = hb_usize_field(counts[2], "INDCRD")?;
    let valcrd = hb_usize_field(counts[3], "VALCRD")?;
    let rhscrd = hb_usize_field(counts[4], "RHSCRD")?;
    if rhscrd != 0 {
        return Err(IoError::UnsupportedFeature(
            "Harwell-Boeing right-hand-side cards are not supported".to_string(),
        ));
    }

    let mxtype_line = lines.next().ok_or_else(|| {
        IoError::InvalidFormat("Harwell-Boeing file missing mxtype line".to_string())
    })?;
    if !mxtype_line.is_ascii() {
        return Err(IoError::InvalidFormat(
            "Harwell-Boeing matrix-type card must be ASCII".to_string(),
        ));
    }
    let mxtype_field = mxtype_line.get(..3).map(str::trim_start).unwrap_or("");
    let dims = parse_hb_int_fields(&mxtype_line[3.min(mxtype_line.len())..], 4, "mxtype")?;
    let rows = hb_usize_field(dims[0], "NROW")?;
    let cols = hb_usize_field(dims[1], "NCOL")?;
    let nnz = hb_usize_field(dims[2], "NNZERO")?;
    let neltvl = dims[3];

    let matrix_type = match mxtype_field {
        "RUA" => HbType::RealUnsymmetricAssembled,
        "RSA" => HbType::RealSymmetricAssembled,
        other => {
            return Err(IoError::UnsupportedFeature(format!(
                "Harwell-Boeing type '{other}' not supported (only RUA and RSA today; \
                 see frankenscipy-vsas0 for follow-on)"
            )));
        }
    };

    if neltvl != 0 {
        return Err(IoError::UnsupportedFeature(
            "Harwell-Boeing unassembled (NELTVL != 0) is not supported".to_string(),
        ));
    }

    // Standard Harwell-Boeing files put PTRFMT, INDFMT, VALFMT, and RHSFMT in
    // fixed-width fields on one line. Early FrankenSciPy fixtures used one line
    // per format, so accept both layouts while always writing the standard one.
    let format_line = lines
        .next()
        .ok_or_else(|| IoError::InvalidFormat("Harwell-Boeing missing ptrfmt line".to_string()))?;
    if format_line.bytes().filter(|&byte| byte == b'(').count() < 3 {
        let _indfmt = lines.next().ok_or_else(|| {
            IoError::InvalidFormat("Harwell-Boeing missing indfmt line".to_string())
        })?;
        let _valfmt = lines.next().ok_or_else(|| {
            IoError::InvalidFormat("Harwell-Boeing missing valfmt line".to_string())
        })?;
    }
    if valcrd == 0 && nnz != 0 {
        return Err(IoError::UnsupportedFeature(
            "Harwell-Boeing pattern-only (valcrd == 0) is not supported".to_string(),
        ));
    }

    let remaining: Vec<&str> = lines.collect();

    // ptrcrd lines hold col_ptr (cols+1 ints), then indcrd lines row_idx (nnz ints),
    // then valcrd lines hold the values (nnz reals). All whitespace-separated within
    // each card group.
    let payload_cards = ptrcrd
        .checked_add(indcrd)
        .and_then(|count| count.checked_add(valcrd))
        .ok_or_else(|| {
            IoError::InvalidFormat("Harwell-Boeing payload card count overflowed usize".to_string())
        })?;
    if remaining.len() < payload_cards {
        return Err(IoError::InvalidFormat(format!(
            "Harwell-Boeing payload truncated: need {} cards, got {}",
            payload_cards,
            remaining.len()
        )));
    }
    let ptr_lines = &remaining[..ptrcrd];
    let ind_lines = &remaining[ptrcrd..ptrcrd + indcrd];
    let val_lines = &remaining[ptrcrd + indcrd..ptrcrd + indcrd + valcrd];

    let raw_col_ptr = parse_hb_int_stream(ptr_lines, cols + 1, "col_ptr")?;
    let raw_row_idx = parse_hb_int_stream(ind_lines, nnz, "row_idx")?;
    let raw_values = parse_hb_real_stream(val_lines, nnz, "values")?;

    // Convert 1-based → 0-based and sanity-check.
    let mut col_ptr = Vec::with_capacity(cols + 1);
    for (i, &p) in raw_col_ptr.iter().enumerate() {
        if p < 1 {
            return Err(IoError::InvalidFormat(format!(
                "Harwell-Boeing col_ptr[{i}] = {p} is below 1 (1-based input expected)"
            )));
        }
        col_ptr.push((p - 1) as usize);
    }
    if col_ptr[0] != 0 {
        return Err(IoError::InvalidFormat(format!(
            "Harwell-Boeing col_ptr[0] = {} (after 0-based shift) but must be 0",
            col_ptr[0]
        )));
    }
    if col_ptr.last().copied() != Some(nnz) {
        return Err(IoError::InvalidFormat(format!(
            "Harwell-Boeing col_ptr[last] = {} but nnz = {nnz}",
            col_ptr.last().copied().unwrap_or_default()
        )));
    }
    for (index, pair) in col_ptr.windows(2).enumerate() {
        if pair[0] > pair[1] || pair[1] > nnz {
            return Err(IoError::InvalidFormat(format!(
                "Harwell-Boeing col_ptr is invalid at columns {index}/{}",
                index + 1
            )));
        }
    }
    let mut row_idx = Vec::with_capacity(nnz);
    for (i, &r) in raw_row_idx.iter().enumerate() {
        if r < 1 || (r as usize) > rows {
            return Err(IoError::InvalidFormat(format!(
                "Harwell-Boeing row_idx[{i}] = {r} out of range [1, {rows}]"
            )));
        }
        row_idx.push((r - 1) as usize);
    }

    Ok(HbMatrix {
        title,
        key,
        matrix_type,
        rows,
        cols,
        nnz,
        col_ptr,
        row_idx,
        values: raw_values,
    })
}

/// Read a real, assembled Harwell-Boeing matrix.
///
/// This is the SciPy-compatible public spelling for
/// [`read_harwell_boeing`].
pub fn hb_read(content: &str) -> Result<HbMatrix, IoError> {
    read_harwell_boeing(content)
}

/// Write a real, assembled CSC matrix in standard Harwell-Boeing form.
///
/// `col_ptr` and `row_idx` are zero-based in memory and are converted to the
/// format's one-based representation. The writer accepts the same RUA/RSA
/// subset as [`hb_read`] and uses enough decimal digits for exact `f64`
/// round-trips through FrankenSciPy's reader.
pub fn hb_write(matrix: &HbMatrix) -> Result<String, IoError> {
    validate_hb_matrix_for_write(matrix)?;

    const INTS_PER_CARD: usize = 8;
    const VALUES_PER_CARD: usize = 3;

    let ptrcrd = matrix.col_ptr.len().div_ceil(INTS_PER_CARD);
    let indcrd = matrix.row_idx.len().div_ceil(INTS_PER_CARD);
    let valcrd = matrix.values.len().div_ceil(VALUES_PER_CARD);
    let totcrd = ptrcrd
        .checked_add(indcrd)
        .and_then(|count| count.checked_add(valcrd))
        .ok_or_else(|| {
            IoError::InvalidFormat("Harwell-Boeing card count overflowed usize".to_string())
        })?;
    let matrix_type = match matrix.matrix_type {
        HbType::RealUnsymmetricAssembled => "RUA",
        HbType::RealSymmetricAssembled => "RSA",
    };

    let mut out = String::new();
    let _ = writeln!(out, "{:<72}{:<8}", matrix.title, matrix.key);
    let _ = writeln!(
        out,
        "{totcrd:>14}{ptrcrd:>14}{indcrd:>14}{valcrd:>14}{:>14}",
        0
    );
    let _ = writeln!(
        out,
        "{matrix_type:<14}{:>14}{:>14}{:>14}{:>14}",
        matrix.rows, matrix.cols, matrix.nnz, 0
    );
    let _ = writeln!(
        out,
        "{:<16}{:<16}{:<20}{:<20}",
        "(8I10)", "(8I10)", "(3E26.18)", ""
    );
    write_hb_index_cards(&mut out, &matrix.col_ptr, INTS_PER_CARD, "col_ptr")?;
    write_hb_index_cards(&mut out, &matrix.row_idx, INTS_PER_CARD, "row_idx")?;
    for values in matrix.values.chunks(VALUES_PER_CARD) {
        for value in values {
            let _ = write!(out, "{value:>26.18E}");
        }
        out.push('\n');
    }
    Ok(out)
}

fn validate_hb_matrix_for_write(matrix: &HbMatrix) -> Result<(), IoError> {
    if !matrix.title.is_ascii() || matrix.title.len() > 72 {
        return Err(IoError::InvalidFormat(
            "Harwell-Boeing title must be ASCII and at most 72 bytes".to_string(),
        ));
    }
    if !matrix.key.is_ascii() || matrix.key.len() > 8 {
        return Err(IoError::InvalidFormat(
            "Harwell-Boeing key must be ASCII and at most 8 bytes".to_string(),
        ));
    }
    if matrix.rows > i64::MAX as usize
        || matrix.cols > i64::MAX as usize
        || matrix.nnz > i64::MAX as usize
    {
        return Err(IoError::InvalidFormat(
            "Harwell-Boeing dimensions exceed signed 64-bit fields".to_string(),
        ));
    }
    let expected_ptr_len = matrix.cols.checked_add(1).ok_or_else(|| {
        IoError::InvalidFormat("Harwell-Boeing column count overflowed usize".to_string())
    })?;
    if matrix.col_ptr.len() != expected_ptr_len {
        return Err(IoError::InvalidFormat(format!(
            "Harwell-Boeing col_ptr has {} entries, expected {expected_ptr_len}",
            matrix.col_ptr.len()
        )));
    }
    if matrix.row_idx.len() != matrix.nnz || matrix.values.len() != matrix.nnz {
        return Err(IoError::InvalidFormat(format!(
            "Harwell-Boeing nnz is {}, but row_idx/value lengths are {}/{}",
            matrix.nnz,
            matrix.row_idx.len(),
            matrix.values.len()
        )));
    }
    if matrix.col_ptr.first() != Some(&0) || matrix.col_ptr.last() != Some(&matrix.nnz) {
        return Err(IoError::InvalidFormat(
            "Harwell-Boeing col_ptr must start at 0 and end at nnz".to_string(),
        ));
    }
    for (index, pair) in matrix.col_ptr.windows(2).enumerate() {
        if pair[0] > pair[1] || pair[1] > matrix.nnz {
            return Err(IoError::InvalidFormat(format!(
                "Harwell-Boeing col_ptr is invalid at columns {index}/{}",
                index + 1
            )));
        }
    }
    for (index, &row) in matrix.row_idx.iter().enumerate() {
        if row >= matrix.rows {
            return Err(IoError::InvalidFormat(format!(
                "Harwell-Boeing row_idx[{index}] = {row} out of range for {} rows",
                matrix.rows
            )));
        }
    }
    for (index, &value) in matrix.values.iter().enumerate() {
        if !value.is_finite() {
            return Err(IoError::InvalidFormat(format!(
                "Harwell-Boeing value[{index}] is not finite"
            )));
        }
    }
    Ok(())
}

fn write_hb_index_cards(
    out: &mut String,
    indices: &[usize],
    per_card: usize,
    field: &str,
) -> Result<(), IoError> {
    for card in indices.chunks(per_card) {
        for &index in card {
            let one_based = index.checked_add(1).ok_or_else(|| {
                IoError::InvalidFormat(format!(
                    "Harwell-Boeing {field} index overflowed one-based representation"
                ))
            })?;
            if one_based > i64::MAX as usize {
                return Err(IoError::InvalidFormat(format!(
                    "Harwell-Boeing {field} index exceeds signed 64-bit fields"
                )));
            }
            let _ = write!(out, "{one_based:>10}");
        }
        out.push('\n');
    }
    Ok(())
}

fn parse_hb_int_fields(line: &str, expected: usize, ctx: &str) -> Result<Vec<i64>, IoError> {
    let toks: Vec<i64> = line
        .split_whitespace()
        .map(|s| s.parse::<i64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| IoError::InvalidFormat(format!("Harwell-Boeing {ctx}: {e}")))?;
    if toks.len() < expected {
        return Err(IoError::InvalidFormat(format!(
            "Harwell-Boeing {ctx} line expected {expected} integers, got {}",
            toks.len()
        )));
    }
    Ok(toks.into_iter().take(expected).collect())
}

fn hb_usize_field(value: i64, field: &str) -> Result<usize, IoError> {
    usize::try_from(value).map_err(|_| {
        IoError::InvalidFormat(format!(
            "Harwell-Boeing {field} must be non-negative, got {value}"
        ))
    })
}

fn parse_hb_int_stream(lines: &[&str], expected: usize, ctx: &str) -> Result<Vec<i64>, IoError> {
    let mut out = Vec::with_capacity(expected);
    for line in lines {
        for tok in line.split_whitespace() {
            let v: i64 = tok
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("Harwell-Boeing {ctx}: {e}")))?;
            out.push(v);
        }
    }
    if out.len() < expected {
        return Err(IoError::InvalidFormat(format!(
            "Harwell-Boeing {ctx} expected {expected} integers, got {}",
            out.len()
        )));
    }
    out.truncate(expected);
    Ok(out)
}

fn parse_hb_real_stream(lines: &[&str], expected: usize, ctx: &str) -> Result<Vec<f64>, IoError> {
    // Harwell-Boeing values are written in Fortran D-format (e.g. "1.0D+00").
    // Tokenize on whitespace and rewrite the Fortran exponent marker before parsing.
    let mut out = Vec::with_capacity(expected);
    for line in lines {
        for tok in line.split_whitespace() {
            let normalized = tok.replace('D', "E").replace('d', "e");
            let v: f64 = normalized
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("Harwell-Boeing {ctx}: {e}")))?;
            out.push(v);
        }
    }
    if out.len() < expected {
        return Err(IoError::InvalidFormat(format!(
            "Harwell-Boeing {ctx} expected {expected} reals, got {}",
            out.len()
        )));
    }
    out.truncate(expected);
    Ok(out)
}

// ══════════════════════════════════════════════════════════════════════
// Fortran sequential unformatted binary records
// ══════════════════════════════════════════════════════════════════════

/// Endianness of the i32 length-marker words framing each record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FortranEndian {
    Little,
    Big,
}

/// Clean end-of-file while requesting the next Fortran record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FortranEOFError {
    pub offset: usize,
}

impl std::fmt::Display for FortranEOFError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "end of Fortran file while reading record at offset {}",
            self.offset
        )
    }
}

impl std::error::Error for FortranEOFError {}

/// Malformed or truncated Fortran sequential record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FortranFormattingError {
    pub offset: usize,
    pub message: String,
}

impl std::fmt::Display for FortranFormattingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Fortran record at offset {}: {}",
            self.offset, self.message
        )
    }
}

impl std::error::Error for FortranFormattingError {}

/// Error returned by [`FortranFile`] record reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FortranFileError {
    EndOfFile(FortranEOFError),
    Formatting(FortranFormattingError),
}

impl std::fmt::Display for FortranFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EndOfFile(error) => error.fmt(f),
            Self::Formatting(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for FortranFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::EndOfFile(error) => Some(error),
            Self::Formatting(error) => Some(error),
        }
    }
}

/// In-memory sequential unformatted Fortran file.
///
/// Records use matching signed 32-bit leading and trailing size words. This
/// deliberately rejects compiler-specific chained subrecords, matching the
/// portable subset supported by SciPy's `FortranFile`. Numeric convenience
/// methods encode `i32` and `f64` values using the file endianness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FortranFile {
    bytes: Vec<u8>,
    cursor: usize,
    endian: FortranEndian,
}

impl FortranFile {
    /// Create an empty file for writing.
    #[must_use]
    pub const fn new(endian: FortranEndian) -> Self {
        Self {
            bytes: Vec::new(),
            cursor: 0,
            endian,
        }
    }

    /// Open existing bytes for reading from the first record.
    #[must_use]
    pub const fn from_bytes(bytes: Vec<u8>, endian: FortranEndian) -> Self {
        Self {
            bytes,
            cursor: 0,
            endian,
        }
    }

    /// Return the encoded file without copying it.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Borrow the complete encoded file.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Current byte offset of the record cursor.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.cursor
    }

    /// Whether the record cursor is at clean end-of-file.
    #[must_use]
    pub fn is_eof(&self) -> bool {
        self.cursor == self.bytes.len()
    }

    /// Return the record cursor to the beginning.
    pub const fn rewind(&mut self) {
        self.cursor = 0;
    }

    /// Read the next raw record.
    pub fn read_record(&mut self) -> Result<Vec<u8>, FortranFileError> {
        if self.cursor == self.bytes.len() {
            return Err(FortranFileError::EndOfFile(FortranEOFError {
                offset: self.cursor,
            }));
        }
        let (payload, next_cursor) =
            decode_fortran_record_at(&self.bytes, self.cursor, self.endian)
                .map_err(FortranFileError::Formatting)?;
        self.cursor = next_cursor;
        Ok(payload)
    }

    /// Append one raw record and advance the cursor to the new end.
    pub fn write_record(&mut self, payload: &[u8]) -> Result<(), FortranFormattingError> {
        let record = encode_fortran_record_checked(payload, self.endian)?;
        self.bytes.extend_from_slice(&record);
        self.cursor = self.bytes.len();
        Ok(())
    }

    /// Read the next record as endian-aware signed 32-bit integers.
    pub fn read_ints(&mut self) -> Result<Vec<i32>, FortranFileError> {
        let record_offset = self.cursor;
        let payload = self.read_record()?;
        if !payload.len().is_multiple_of(4) {
            return Err(FortranFileError::Formatting(FortranFormattingError {
                offset: record_offset,
                message: format!(
                    "integer record has {} payload bytes, not a multiple of 4",
                    payload.len()
                ),
            }));
        }
        Ok(payload
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&bytes| match self.endian {
                FortranEndian::Little => i32::from_le_bytes(bytes),
                FortranEndian::Big => i32::from_be_bytes(bytes),
            })
            .collect())
    }

    /// Read the next record as endian-aware 64-bit floating-point values.
    pub fn read_reals(&mut self) -> Result<Vec<f64>, FortranFileError> {
        let record_offset = self.cursor;
        let payload = self.read_record()?;
        if !payload.len().is_multiple_of(8) {
            return Err(FortranFileError::Formatting(FortranFormattingError {
                offset: record_offset,
                message: format!(
                    "real record has {} payload bytes, not a multiple of 8",
                    payload.len()
                ),
            }));
        }
        Ok(payload
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&bytes| {
                let bits = match self.endian {
                    FortranEndian::Little => u64::from_le_bytes(bytes),
                    FortranEndian::Big => u64::from_be_bytes(bytes),
                };
                f64::from_bits(bits)
            })
            .collect())
    }

    /// Append one record of endian-aware signed 32-bit integers.
    pub fn write_ints(&mut self, values: &[i32]) -> Result<(), FortranFormattingError> {
        let mut payload = Vec::with_capacity(values.len().saturating_mul(4));
        for &value in values {
            let bytes = match self.endian {
                FortranEndian::Little => value.to_le_bytes(),
                FortranEndian::Big => value.to_be_bytes(),
            };
            payload.extend_from_slice(&bytes);
        }
        self.write_record(&payload)
    }

    /// Append one record of endian-aware 64-bit floating-point values.
    pub fn write_reals(&mut self, values: &[f64]) -> Result<(), FortranFormattingError> {
        let mut payload = Vec::with_capacity(values.len().saturating_mul(8));
        for &value in values {
            let bytes = match self.endian {
                FortranEndian::Little => value.to_bits().to_le_bytes(),
                FortranEndian::Big => value.to_bits().to_be_bytes(),
            };
            payload.extend_from_slice(&bytes);
        }
        self.write_record(&payload)
    }
}

fn decode_fortran_record_at(
    bytes: &[u8],
    cursor: usize,
    endian: FortranEndian,
) -> Result<(Vec<u8>, usize), FortranFormattingError> {
    let fail = |message: String| FortranFormattingError {
        offset: cursor,
        message,
    };
    let header_end = cursor
        .checked_add(4)
        .ok_or_else(|| fail("header offset overflow".to_string()))?;
    if header_end > bytes.len() {
        return Err(fail(format!(
            "header truncated ({} bytes remaining)",
            bytes.len().saturating_sub(cursor)
        )));
    }
    let header_bytes = [
        bytes[cursor],
        bytes[cursor + 1],
        bytes[cursor + 2],
        bytes[cursor + 3],
    ];
    let length = match endian {
        FortranEndian::Little => i32::from_le_bytes(header_bytes),
        FortranEndian::Big => i32::from_be_bytes(header_bytes),
    };
    if length < 0 {
        return Err(fail(format!("negative length {length}")));
    }
    let length = length as usize;
    let payload_end = header_end
        .checked_add(length)
        .ok_or_else(|| fail("payload offset overflow".to_string()))?;
    let trailer_end = payload_end
        .checked_add(4)
        .ok_or_else(|| fail("trailer offset overflow".to_string()))?;
    if trailer_end > bytes.len() {
        return Err(fail(format!(
            "payload and trailer truncated (need {trailer_end}, have {})",
            bytes.len()
        )));
    }
    let trailer_bytes = [
        bytes[payload_end],
        bytes[payload_end + 1],
        bytes[payload_end + 2],
        bytes[payload_end + 3],
    ];
    let trailer = match endian {
        FortranEndian::Little => i32::from_le_bytes(trailer_bytes),
        FortranEndian::Big => i32::from_be_bytes(trailer_bytes),
    };
    if trailer != length as i32 {
        return Err(fail(format!(
            "header length {length} does not match trailer {trailer}"
        )));
    }
    Ok((bytes[header_end..payload_end].to_vec(), trailer_end))
}

fn encode_fortran_record_checked(
    payload: &[u8],
    endian: FortranEndian,
) -> Result<Vec<u8>, FortranFormattingError> {
    let length = i32::try_from(payload.len()).map_err(|_| FortranFormattingError {
        offset: 0,
        message: format!(
            "payload length {} exceeds the signed 32-bit record limit",
            payload.len()
        ),
    })?;
    let length_bytes = match endian {
        FortranEndian::Little => length.to_le_bytes(),
        FortranEndian::Big => length.to_be_bytes(),
    };
    let capacity = payload
        .len()
        .checked_add(8)
        .ok_or_else(|| FortranFormattingError {
            offset: 0,
            message: "encoded record length overflowed usize".to_string(),
        })?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(&length_bytes);
    out.extend_from_slice(payload);
    out.extend_from_slice(&length_bytes);
    Ok(out)
}

/// Read a Fortran sequential unformatted file.
///
/// Each record on disk is framed as `<len:i32><payload:len><len:i32>` with
/// the leading and trailing length words matching. Returns the concatenated
/// list of payloads in order. Matches the most common subset of
/// `scipy.io.FortranFile.read_record()` driven across the whole file.
///
/// Errors:
/// - `InvalidFormat` if a record header is truncated or the trailing length
///   does not match the leading length.
/// - `InvalidFormat` if the input contains trailing bytes that do not form
///   a complete record.
pub fn read_fortran_unformatted(
    bytes: &[u8],
    endian: FortranEndian,
) -> Result<Vec<Vec<u8>>, IoError> {
    let mut records = Vec::new();
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let (payload, next_cursor) = decode_fortran_record_at(bytes, cursor, endian)
            .map_err(|error| IoError::InvalidFormat(error.to_string()))?;
        records.push(payload);
        cursor = next_cursor;
    }
    Ok(records)
}

/// Frame a payload with the Fortran sequential unformatted header+trailer.
/// Useful for building fixtures and for round-trip tests.
pub fn write_fortran_record(payload: &[u8], endian: FortranEndian) -> Vec<u8> {
    let length = payload.len() as i32;
    let length_bytes = match endian {
        FortranEndian::Little => length.to_le_bytes(),
        FortranEndian::Big => length.to_be_bytes(),
    };
    let mut out = Vec::with_capacity(payload.len() + 8);
    out.extend_from_slice(&length_bytes);
    out.extend_from_slice(payload);
    out.extend_from_slice(&length_bytes);
    out
}

// ══════════════════════════════════════════════════════════════════════
// ARFF (Weka attribute-relation file format)
// ══════════════════════════════════════════════════════════════════════

/// ARFF attribute type. Relational attributes are intentionally not supported
/// and return `UnsupportedFeature`.
#[derive(Debug, Clone, PartialEq)]
pub enum ArffAttribute {
    /// `@attribute name numeric` / `real` / `integer`.
    Numeric { name: String },
    /// `@attribute name {a, b, c}`. Domain is the ordered list of allowed
    /// nominal values (case-sensitive).
    Nominal { name: String, domain: Vec<String> },
    /// `@attribute name string`. Free-form text.
    String { name: String },
    /// `@attribute name date "yyyy-MM-dd"`. Format uses Weka's Java-style
    /// `SimpleDateFormat` subset, matching `scipy.io.arff` for simple fields.
    Date {
        name: String,
        format: String,
        unit: ArffDateUnit,
    },
}

impl ArffAttribute {
    pub fn name(&self) -> &str {
        match self {
            Self::Numeric { name }
            | Self::Nominal { name, .. }
            | Self::String { name }
            | Self::Date { name, .. } => name,
        }
    }
}

/// Precision selected for an ARFF date attribute. This mirrors the numpy
/// `datetime64` unit SciPy chooses from the declared format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ArffDateUnit {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
}

/// Parsed ARFF date value. `raw` preserves the input token after unquoting;
/// `normalized` matches numpy `datetime64` display at the selected unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArffDateTime {
    pub raw: String,
    pub normalized: String,
    pub unit: ArffDateUnit,
}

/// One ARFF data cell. Missing values are encoded as `?` in the file and
/// surface as `Missing` here.
#[derive(Debug, Clone, PartialEq)]
pub enum ArffValue {
    Numeric(f64),
    Nominal(String),
    String(String),
    Date(ArffDateTime),
    Missing,
}

/// Parsed ARFF file. Sparse rows are expanded to their dense equivalent
/// (missing positions filled with the type-appropriate zero/empty value)
/// for parity with `scipy.io.arff.loadarff`.
#[derive(Debug, Clone, PartialEq)]
pub struct ArffData {
    pub relation: String,
    pub attributes: Vec<ArffAttribute>,
    pub rows: Vec<Vec<ArffValue>>,
}

/// Read an ARFF (Weka attribute-relation file format) string.
///
/// Supports numeric, nominal, string, and simple date attributes; comments
/// (`%`); sparse rows (`{ idx val, idx val }`); single- and double-quoted
/// values. Relational attributes return `UnsupportedFeature`.
pub fn read_arff(content: &str) -> Result<ArffData, IoError> {
    let mut relation: Option<String> = None;
    let mut attributes: Vec<ArffAttribute> = Vec::new();
    let mut rows: Vec<Vec<ArffValue>> = Vec::new();
    let mut in_data_section = false;

    for raw_line in content.lines() {
        let line = strip_arff_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        if !in_data_section {
            let lower = line.to_ascii_lowercase();
            if lower.starts_with("@relation") {
                relation = Some(unquote_arff(line["@relation".len()..].trim()));
            } else if lower.starts_with("@attribute") {
                attributes.push(parse_arff_attribute(line)?);
            } else if lower == "@data" {
                in_data_section = true;
            } else {
                return Err(IoError::InvalidFormat(format!(
                    "ARFF: unexpected directive in header: {line}"
                )));
            }
        } else {
            rows.push(parse_arff_data_row(line, &attributes)?);
        }
    }

    let relation = relation
        .ok_or_else(|| IoError::InvalidFormat("ARFF: missing @relation header".to_string()))?;
    if attributes.is_empty() {
        return Err(IoError::InvalidFormat(
            "ARFF: at least one @attribute is required".to_string(),
        ));
    }
    Ok(ArffData {
        relation,
        attributes,
        rows,
    })
}

fn strip_arff_comment(line: &str) -> &str {
    // Comments start with '%' but not inside quotes. ARFF doesn't escape
    // '%' inside quotes by anything other than scanning past the quote.
    let bytes = line.as_bytes();
    let mut in_single = false;
    let mut in_double = false;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'\'' if !in_double => in_single = !in_single,
            b'"' if !in_single => in_double = !in_double,
            b'%' if !in_single && !in_double => return &line[..i],
            _ => {}
        }
    }
    line
}

fn unquote_arff(raw: &str) -> String {
    let trimmed = raw.trim();
    if (trimmed.starts_with('\'') && trimmed.ends_with('\'') && trimmed.len() >= 2)
        || (trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2)
    {
        trimmed[1..trimmed.len() - 1].to_string()
    } else {
        trimmed.to_string()
    }
}

fn parse_arff_attribute(line: &str) -> Result<ArffAttribute, IoError> {
    // After '@attribute', tokens: name TYPE_OR_DOMAIN
    let body = &line[10.min(line.len())..].trim_start();
    let (name, rest) = split_arff_first_token(body)?;
    let name = unquote_arff(&name);
    let rest = rest.trim();
    if rest.starts_with('{') {
        let close = rest.find('}').ok_or_else(|| {
            IoError::InvalidFormat(format!("ARFF nominal attribute '{name}': missing '}}'"))
        })?;
        let domain: Vec<String> = rest[1..close]
            .split(',')
            .map(|s| unquote_arff(s.trim()))
            .filter(|s| !s.is_empty())
            .collect();
        if domain.is_empty() {
            return Err(IoError::InvalidFormat(format!(
                "ARFF nominal attribute '{name}': empty domain"
            )));
        }
        Ok(ArffAttribute::Nominal { name, domain })
    } else {
        let type_part = rest
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        match type_part.as_str() {
            "numeric" | "real" | "integer" => Ok(ArffAttribute::Numeric { name }),
            "string" => Ok(ArffAttribute::String { name }),
            "date" => {
                let (format, unit) = parse_arff_date_format(&name, rest)?;
                Ok(ArffAttribute::Date { name, format, unit })
            }
            "relational" => Err(IoError::UnsupportedFeature(format!(
                "ARFF relational attribute '{name}' (frankenscipy-vsas0 follow-on)"
            ))),
            other => Err(IoError::InvalidFormat(format!(
                "ARFF attribute '{name}': unknown type '{other}'"
            ))),
        }
    }
}

fn split_arff_first_token(input: &str) -> Result<(String, &str), IoError> {
    let bytes = input.as_bytes();
    if bytes.is_empty() {
        return Err(IoError::InvalidFormat(
            "ARFF: missing attribute name".into(),
        ));
    }
    let (end, _quote) = match bytes[0] {
        b'\'' | b'"' => {
            let quote = bytes[0];
            let mut i = 1usize;
            while i < bytes.len() && bytes[i] != quote {
                i += 1;
            }
            if i >= bytes.len() {
                return Err(IoError::InvalidFormat(
                    "ARFF: unterminated quoted attribute name".into(),
                ));
            }
            (i + 1, Some(quote))
        }
        _ => (
            bytes
                .iter()
                .position(|b| b.is_ascii_whitespace())
                .unwrap_or(bytes.len()),
            None,
        ),
    };
    Ok((input[..end].to_string(), &input[end..]))
}

fn parse_arff_data_row(
    line: &str,
    attributes: &[ArffAttribute],
) -> Result<Vec<ArffValue>, IoError> {
    let trimmed = line.trim();
    if trimmed.starts_with('{') {
        let close = trimmed
            .find('}')
            .ok_or_else(|| IoError::InvalidFormat("ARFF sparse row: missing '}'".into()))?;
        let mut row = vec![ArffValue::Missing; attributes.len()];
        // Initialize numeric cells to 0.0 per ARFF sparse semantics.
        for (i, attr) in attributes.iter().enumerate() {
            if matches!(attr, ArffAttribute::Numeric { .. }) {
                row[i] = ArffValue::Numeric(0.0);
            }
        }
        for entry in trimmed[1..close].split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            let mut parts = entry.splitn(2, char::is_whitespace);
            let idx_part = parts.next().unwrap_or("");
            let value_part = parts.next().unwrap_or("").trim();
            let idx: usize = idx_part.parse().map_err(|e| {
                IoError::InvalidFormat(format!("ARFF sparse index '{idx_part}': {e}"))
            })?;
            if idx >= attributes.len() {
                return Err(IoError::InvalidFormat(format!(
                    "ARFF sparse index {idx} out of range for {} attributes",
                    attributes.len()
                )));
            }
            row[idx] = parse_arff_cell(value_part, &attributes[idx])?;
        }
        Ok(row)
    } else {
        let cells = split_arff_csv_cells(trimmed);
        if cells.len() != attributes.len() {
            return Err(IoError::InvalidFormat(format!(
                "ARFF dense row: got {} cells but {} attributes declared",
                cells.len(),
                attributes.len()
            )));
        }
        cells
            .iter()
            .zip(attributes.iter())
            .map(|(cell, attr)| parse_arff_cell(cell.trim(), attr))
            .collect()
    }
}

fn split_arff_csv_cells(line: &str) -> Vec<String> {
    let mut cells = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    for ch in line.chars() {
        match ch {
            '\'' if !in_double => {
                in_single = !in_single;
                current.push(ch);
            }
            '"' if !in_single => {
                in_double = !in_double;
                current.push(ch);
            }
            ',' if !in_single && !in_double => {
                cells.push(std::mem::take(&mut current));
            }
            _ => current.push(ch),
        }
    }
    cells.push(current);
    cells
}

fn parse_arff_cell(token: &str, attribute: &ArffAttribute) -> Result<ArffValue, IoError> {
    if matches!(token.as_bytes(), [b'?']) {
        return Ok(ArffValue::Missing);
    }
    match attribute {
        ArffAttribute::Numeric { name } => {
            let v: f64 = token.parse().map_err(|e| {
                IoError::InvalidFormat(format!("ARFF numeric '{name}' value '{token}': {e}"))
            })?;
            Ok(ArffValue::Numeric(v))
        }
        ArffAttribute::Nominal { name, domain } => {
            let unquoted = unquote_arff(token);
            if !domain.contains(&unquoted) {
                return Err(IoError::InvalidFormat(format!(
                    "ARFF nominal '{name}': value '{unquoted}' not in domain"
                )));
            }
            Ok(ArffValue::Nominal(unquoted))
        }
        ArffAttribute::String { .. } => Ok(ArffValue::String(unquote_arff(token))),
        ArffAttribute::Date { name, format, unit } => {
            parse_arff_date_cell(token, name, format, *unit)
        }
    }
}

fn parse_arff_date_format(
    name: &str,
    attribute_spec: &str,
) -> Result<(String, ArffDateUnit), IoError> {
    let raw_format = attribute_spec["date".len()..].trim();
    let format = unquote_arff(raw_format);
    if format.is_empty() {
        return Err(IoError::InvalidFormat(format!(
            "ARFF date attribute '{name}': missing date format"
        )));
    }
    if format.contains('z') || format.contains('Z') {
        return Err(IoError::UnsupportedFeature(format!(
            "ARFF date attribute '{name}': timezone formats are not supported"
        )));
    }

    let mut unit = None;
    let mut i = 0usize;
    while i < format.len() {
        let rest = &format[i..];
        if rest.starts_with("yyyy") {
            unit = Some(max_arff_date_unit(unit, ArffDateUnit::Year));
            i += 4;
        } else if rest.starts_with("yy") {
            unit = Some(max_arff_date_unit(unit, ArffDateUnit::Year));
            i += 2;
        } else if rest.starts_with("MM") {
            unit = Some(max_arff_date_unit(unit, ArffDateUnit::Month));
            i += 2;
        } else if rest.starts_with("dd") {
            unit = Some(max_arff_date_unit(unit, ArffDateUnit::Day));
            i += 2;
        } else if rest.starts_with("HH") {
            unit = Some(max_arff_date_unit(unit, ArffDateUnit::Hour));
            i += 2;
        } else if rest.starts_with("mm") {
            unit = Some(max_arff_date_unit(unit, ArffDateUnit::Minute));
            i += 2;
        } else if rest.starts_with("ss") {
            unit = Some(max_arff_date_unit(unit, ArffDateUnit::Second));
            i += 2;
        } else {
            let ch = rest.chars().next().expect("nonempty format tail");
            i += ch.len_utf8();
        }
    }

    let unit = unit.ok_or_else(|| {
        IoError::InvalidFormat(format!(
            "ARFF date attribute '{name}': invalid or unsupported date format '{format}'"
        ))
    })?;
    Ok((format, unit))
}

fn max_arff_date_unit(current: Option<ArffDateUnit>, candidate: ArffDateUnit) -> ArffDateUnit {
    current.map_or(candidate, |unit| unit.max(candidate))
}

fn parse_arff_date_cell(
    token: &str,
    name: &str,
    format: &str,
    unit: ArffDateUnit,
) -> Result<ArffValue, IoError> {
    let raw = unquote_arff(token);
    let components = parse_arff_date_components(&raw, format).map_err(|message| {
        IoError::InvalidFormat(format!("ARFF date '{name}' value '{raw}': {message}"))
    })?;
    let normalized = normalize_arff_date(&components, unit).map_err(|message| {
        IoError::InvalidFormat(format!("ARFF date '{name}' value '{raw}': {message}"))
    })?;
    Ok(ArffValue::Date(ArffDateTime {
        raw,
        normalized,
        unit,
    }))
}

#[derive(Debug, Default)]
struct ArffDateComponents {
    year: Option<i32>,
    month: Option<u8>,
    day: Option<u8>,
    hour: Option<u8>,
    minute: Option<u8>,
    second: Option<u8>,
}

fn parse_arff_date_components(value: &str, format: &str) -> Result<ArffDateComponents, String> {
    let mut components = ArffDateComponents::default();
    let mut format_pos = 0usize;
    let mut value_pos = 0usize;

    while format_pos < format.len() {
        let rest = &format[format_pos..];
        if rest.starts_with("yyyy") {
            components.year = Some(read_arff_date_number(value, &mut value_pos, 4)?);
            format_pos += 4;
        } else if rest.starts_with("yy") {
            let year: i32 = read_arff_date_number(value, &mut value_pos, 2)?;
            components.year = Some(if year <= 68 { 2000 + year } else { 1900 + year });
            format_pos += 2;
        } else if rest.starts_with("MM") {
            components.month = Some(read_arff_date_number(value, &mut value_pos, 2)?);
            format_pos += 2;
        } else if rest.starts_with("dd") {
            components.day = Some(read_arff_date_number(value, &mut value_pos, 2)?);
            format_pos += 2;
        } else if rest.starts_with("HH") {
            components.hour = Some(read_arff_date_number(value, &mut value_pos, 2)?);
            format_pos += 2;
        } else if rest.starts_with("mm") {
            components.minute = Some(read_arff_date_number(value, &mut value_pos, 2)?);
            format_pos += 2;
        } else if rest.starts_with("ss") {
            components.second = Some(read_arff_date_number(value, &mut value_pos, 2)?);
            format_pos += 2;
        } else if rest.starts_with('\'') {
            format_pos += 1;
            while format_pos < format.len() && !format[format_pos..].starts_with('\'') {
                let ch = format[format_pos..]
                    .chars()
                    .next()
                    .expect("nonempty quoted format literal");
                consume_arff_date_literal(value, &mut value_pos, ch)?;
                format_pos += ch.len_utf8();
            }
            if format_pos >= format.len() {
                return Err("unterminated quoted literal in date format".to_string());
            }
            format_pos += 1;
        } else {
            let ch = rest.chars().next().expect("nonempty date format literal");
            consume_arff_date_literal(value, &mut value_pos, ch)?;
            format_pos += ch.len_utf8();
        }
    }

    if value_pos != value.len() {
        return Err(format!("trailing input '{}'", &value[value_pos..]));
    }

    Ok(components)
}

fn read_arff_date_number<T>(value: &str, value_pos: &mut usize, width: usize) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    if *value_pos + width > value.len() {
        return Err(format!("expected {width} digits"));
    }
    let field = &value[*value_pos..*value_pos + width];
    if !field.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("expected {width} digits, got '{field}'"));
    }
    *value_pos += width;
    field
        .parse()
        .map_err(|e| format!("invalid number '{field}': {e}"))
}

fn consume_arff_date_literal(
    value: &str,
    value_pos: &mut usize,
    expected: char,
) -> Result<(), String> {
    let actual = value[*value_pos..]
        .chars()
        .next()
        .ok_or_else(|| format!("expected literal '{expected}'"))?;
    if actual != expected {
        return Err(format!("expected literal '{expected}', got '{actual}'"));
    }
    *value_pos += actual.len_utf8();
    Ok(())
}

fn normalize_arff_date(
    components: &ArffDateComponents,
    unit: ArffDateUnit,
) -> Result<String, String> {
    let year = components
        .year
        .ok_or_else(|| "date format must include a year".to_string())?;
    let month = components.month.unwrap_or(1);
    let day = components.day.unwrap_or(1);
    let hour = components.hour.unwrap_or(0);
    let minute = components.minute.unwrap_or(0);
    let second = components.second.unwrap_or(0);

    validate_arff_date_components(year, month, day, hour, minute, second)?;

    match unit {
        ArffDateUnit::Year => Ok(format!("{year:04}")),
        ArffDateUnit::Month => {
            require_arff_date_component(components.month, "month")?;
            Ok(format!("{year:04}-{month:02}"))
        }
        ArffDateUnit::Day => {
            require_arff_date_component(components.month, "month")?;
            require_arff_date_component(components.day, "day")?;
            Ok(format!("{year:04}-{month:02}-{day:02}"))
        }
        ArffDateUnit::Hour => {
            require_arff_date_component(components.month, "month")?;
            require_arff_date_component(components.day, "day")?;
            require_arff_date_component(components.hour, "hour")?;
            Ok(format!("{year:04}-{month:02}-{day:02}T{hour:02}"))
        }
        ArffDateUnit::Minute => {
            require_arff_date_component(components.month, "month")?;
            require_arff_date_component(components.day, "day")?;
            require_arff_date_component(components.hour, "hour")?;
            require_arff_date_component(components.minute, "minute")?;
            Ok(format!(
                "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}"
            ))
        }
        ArffDateUnit::Second => {
            require_arff_date_component(components.month, "month")?;
            require_arff_date_component(components.day, "day")?;
            require_arff_date_component(components.hour, "hour")?;
            require_arff_date_component(components.minute, "minute")?;
            require_arff_date_component(components.second, "second")?;
            Ok(format!(
                "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}"
            ))
        }
    }
}

fn require_arff_date_component<T>(component: Option<T>, name: &str) -> Result<(), String> {
    if component.is_some() {
        Ok(())
    } else {
        Err(format!("date format is missing required {name} field"))
    }
}

fn validate_arff_date_components(
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,
) -> Result<(), String> {
    if !(1..=12).contains(&month) {
        return Err(format!("month {month} out of range"));
    }
    let max_day = days_in_arff_month(year, month);
    if !(1..=max_day).contains(&day) {
        return Err(format!("day {day} out of range for month {month}"));
    }
    if hour > 23 {
        return Err(format!("hour {hour} out of range"));
    }
    if minute > 59 {
        return Err(format!("minute {minute} out of range"));
    }
    if second > 59 {
        return Err(format!("second {second} out of range"));
    }
    Ok(())
}

fn days_in_arff_month(year: i32, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_arff_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

fn is_arff_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat_data_elements_decode_typed_chunks_in_either_byte_order() {
        // frankenscipy-3xbva pinned the typed chunk decode; frankenscipy-1ksfv.12 moved it to
        // SciPy's rule for a payload that is not a whole number of elements: `byte_count //
        // itemsize` values, the partial one dropped (live SciPy 1.17.1 reads a 1x2 double from
        // 19 payload bytes), with the array's reshape rejecting a count its dims do not match.
        assert_eq!(
            decode_mat_data(MatDtype::I16, &[0xfe, 0xff, 0x34, 0x12], false),
            MatData::I16(vec![-2, 4660])
        );
        assert_eq!(
            decode_mat_data(MatDtype::I16, &[0xff, 0xfe, 0x12, 0x34], true),
            MatData::I16(vec![-2, 4660])
        );
        assert_eq!(
            decode_mat_data(MatDtype::F64, &1.0f64.to_le_bytes(), false),
            MatData::F64(vec![1.0])
        );
        assert_eq!(
            decode_mat_data(MatDtype::I32, &[0, 0, 0], false),
            MatData::I32(Vec::new())
        );
        let mut nineteen = [1.0f64, 2.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>();
        nineteen.extend_from_slice(&[1, 2, 3]);
        let file = mat5_file(&[mat5_matrix(
            MatClass::Double.code(),
            0,
            &[1, 2],
            b"x",
            &[mat5_element(MI_DOUBLE, &nineteen)],
        )]);
        let loaded = loadmat(&file, &LoadmatOptions::default()).expect("partial element");
        assert_eq!(
            loaded.get("x"),
            Some(&MatValue::Numeric(MatNumeric::new(
                vec![1, 2],
                MatData::F64(vec![1.0, 2.0])
            )))
        );
        let short = mat5_file(&[mat5_matrix(
            MatClass::Double.code(),
            0,
            &[1, 3],
            b"x",
            &[mat5_element(MI_DOUBLE, &nineteen)],
        )]);
        let err = loadmat(&short, &LoadmatOptions::default()).expect_err("2 values for 1x3");
        assert!(matches!(err, IoError::InvalidFormat(m) if m.contains("cannot reshape")));
    }

    // ── MAT-file test builders: the byte layout SciPy's probes used, little-endian unless a
    //    helper says otherwise. ──

    fn mat5_header_bytes(big_endian: bool) -> Vec<u8> {
        let mut head = b"MATLAB 5.0 MAT-file, fsci-io unit test".to_vec();
        head.resize(116, b' ');
        head.extend_from_slice(&[0; 8]);
        if big_endian {
            head.extend_from_slice(&0x0100u16.to_be_bytes());
            head.extend_from_slice(b"MI");
        } else {
            head.extend_from_slice(&0x0100u16.to_le_bytes());
            head.extend_from_slice(b"IM");
        }
        head
    }

    fn mat5_file(elements: &[Vec<u8>]) -> Vec<u8> {
        let mut file = mat5_header_bytes(false);
        for element in elements {
            file.extend_from_slice(element);
        }
        file
    }

    /// A full (never small) data element, padded to 8 bytes.
    fn mat5_element(mdtype: u32, payload: &[u8]) -> Vec<u8> {
        mat5_element_in(mdtype, payload, false)
    }

    fn mat5_element_in(mdtype: u32, payload: &[u8], big_endian: bool) -> Vec<u8> {
        let word = |w: u32| {
            if big_endian {
                w.to_be_bytes()
            } else {
                w.to_le_bytes()
            }
        };
        let mut out = word(mdtype).to_vec();
        out.extend_from_slice(&word(u32::try_from(payload.len()).expect("small payload")));
        out.extend_from_slice(payload);
        out.resize(out.len() + padding8(payload.len()), 0);
        out
    }

    fn mat5_matrix(class: u8, flags: u32, dims: &[i32], name: &[u8], subs: &[Vec<u8>]) -> Vec<u8> {
        mat5_matrix_in(class, flags, dims, name, subs, false)
    }

    fn mat5_matrix_in(
        class: u8,
        flags: u32,
        dims: &[i32],
        name: &[u8],
        subs: &[Vec<u8>],
        big_endian: bool,
    ) -> Vec<u8> {
        let word = |w: u32| {
            if big_endian {
                w.to_be_bytes()
            } else {
                w.to_le_bytes()
            }
        };
        let mut flag_bytes = word(u32::from(class) | (flags << 8)).to_vec();
        flag_bytes.extend_from_slice(&[0; 4]);
        let dim_bytes: Vec<u8> = dims
            .iter()
            .flat_map(|&d| {
                if big_endian {
                    d.to_be_bytes()
                } else {
                    d.to_le_bytes()
                }
            })
            .collect();
        let mut body = mat5_element_in(MI_UINT32, &flag_bytes, big_endian);
        body.extend(mat5_element_in(MI_INT32, &dim_bytes, big_endian));
        body.extend(mat5_element_in(MI_INT8, name, big_endian));
        for sub in subs {
            body.extend_from_slice(sub);
        }
        let mut out = word(MI_MATRIX).to_vec();
        out.extend_from_slice(&word(u32::try_from(body.len()).expect("small matrix")));
        out.extend(body);
        out
    }

    fn f64_bytes(values: &[f64]) -> Vec<u8> {
        values.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn i32_bytes(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn scalar_double(name: &[u8], value: f64) -> Vec<u8> {
        mat5_matrix(
            MatClass::Double.code(),
            0,
            &[1, 1],
            name,
            &[mat5_element(MI_DOUBLE, &value.to_le_bytes())],
        )
    }

    fn zlib_element(raw: &[u8]) -> Vec<u8> {
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(raw, 6);
        let mut out = MI_COMPRESSED.to_le_bytes().to_vec();
        out.extend_from_slice(&u32::try_from(packed.len()).expect("small").to_le_bytes());
        out.extend(packed);
        out
    }

    fn numeric(dims: &[usize], data: MatData) -> MatValue {
        MatValue::Numeric(MatNumeric::new(dims.to_vec(), data))
    }

    fn strings(dims: &[usize], width: usize, values: &[&str]) -> MatValue {
        MatValue::Strings(MatStrings {
            dims: dims.to_vec(),
            width,
            strings: values.iter().map(|s| (*s).to_string()).collect(),
        })
    }

    fn load_default(bytes: &[u8]) -> MatFile {
        loadmat(bytes, &LoadmatOptions::default()).expect("MAT file loads")
    }

    fn hex_bytes(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex digit pair"))
            .collect()
    }

    /// The content of the SciPy reference files below, as fsci values.
    fn scipy_reference_content() -> Vec<(String, MatValue)> {
        vec![
            (
                "s".to_string(),
                MatValue::Struct(MatStruct {
                    dims: vec![1, 1],
                    field_names: vec!["ab".to_string(), "c".to_string()],
                    values: vec![
                        numeric(&[2, 2], MatData::F64(vec![1.0, 3.0, 2.0, 4.0])),
                        MatValue::Char(MatChar::row("hi")),
                    ],
                }),
            ),
            (
                "c".to_string(),
                MatValue::Cell(MatCell {
                    dims: vec![1, 2],
                    items: vec![
                        numeric(&[1, 1], MatData::F64(vec![1.0])),
                        MatValue::Char(MatChar::row("x")),
                    ],
                }),
            ),
            (
                "sp".to_string(),
                MatValue::Sparse(MatSparse {
                    rows: 2,
                    cols: 2,
                    logical: false,
                    indptr: vec![0, 1, 2],
                    indices: vec![1, 0],
                    data: MatData::F64(vec![2.0, 1.5]),
                    imag: None,
                }),
            ),
            (
                "b".to_string(),
                numeric(&[1, 2], MatData::Bool(vec![true, false])),
            ),
            (
                "z".to_string(),
                MatValue::Numeric(MatNumeric::complex(
                    vec![1, 2],
                    MatData::F32(vec![1.0, 3.0]),
                    MatData::F32(vec![2.0, -4.0]),
                )),
            ),
            ("e".to_string(), numeric(&[0, 0], MatData::F64(Vec::new()))),
            ("st".to_string(), strings(&[2], 3, &["a", "bcd"])),
            (
                "i".to_string(),
                numeric(&[2, 3, 4], MatData::I8((0..24).collect())),
            ),
            (
                "v".to_string(),
                numeric(&[3], MatData::F64(vec![0.0, 1.0, 2.0])),
            ),
        ]
    }

    /// `scipy.io.savemat(f, content)` (SciPy 1.17.1, uncompressed, oned_as='row') from byte 116
    /// on; the description before it holds a timestamp. Content: see `scipy_reference_content`
    /// (the generator used NumPy equivalents: a dict for `s`, an object array for `c`, a
    /// `csc_array`, a bool array, complex64, `np.zeros((0, 0))`, `np.array(['a', 'bcd'])`, an
    /// int8 2x3x4 array and `np.arange(3.0)`).
    const SCIPY_V5_REFERENCE_FROM_116: &str = concat!(
        "00000000000000000001494d0e000000d000000006000000080000000200000000000000050000000800000001000000",
        "0100000001000100730000000500040003000000010000000600000061620063000000000e0000005000000006000000",
        "080000000600000000000000050000000800000002000000020000000100000000000000090000002000000000000000",
        "0000f03f0000000000000840000000000000004000000000000010400e00000030000000060000000800000004000000",
        "0000000005000000080000000100000002000000010000000000000010000200686900000e000000a000000006000000",
        "0800000001000000000000000500000008000000010000000200000001000100630000000e0000003800000006000000",
        "080000000600000000000000050000000800000001000000010000000100000000000000090000000800000000000000",
        "0000f03f0e00000030000000060000000800000004000000000000000500000008000000010000000100000001000000",
        "0000000010000100780000000e0000006800000006000000080000000500000002000000050000000800000002000000",
        "02000000010002007370000005000000080000000100000000000000050000000c000000000000000100000002000000",
        "0000000009000000100000000000000000000040000000000000f83f0e00000030000000060000000800000009020000",
        "0000000005000000080000000100000002000000010001006200000002000200010000000e0000004800000006000000",
        "08000000070800000000000005000000080000000100000002000000010001007a00000007000000080000000000803f",
        "00004040070000000800000000000040000080c00e000000300000000600000008000000060000000000000005000000",
        "080000000000000000000000010001006500000009000000000000000e00000038000000060000000800000004000000",
        "00000000050000000800000002000000030000000100020073740000100000000600000061622063206400000e000000",
        "5000000006000000080000000800000000000000050000000c0000000200000003000000040000000000000001000100",
        "690000000100000018000000000102030405060708090a0b0c0d0e0f10111213141516170e0000004800000006000000",
        "080000000600000000000000050000000800000001000000030000000100010076000000090000001800000000000000",
        "00000000000000000000f03f0000000000000040",
    );

    /// `scipy.io.savemat(f, {'x': [[1., 2., 3.], [4., 5., 6.]], 's': 'hi', 'z': [[1+2j]],
    /// 'sp': csc_array([[0, 1.5], [2., 0]]), 'i': np.array([[7, 8]], dtype=np.int16)},
    /// format='4')`, SciPy 1.17.1.
    const SCIPY_V4_REFERENCE: &str = concat!(
        "00000000020000000300000000000000020000007800000000000000f03f000000000000104000000000000000400000",
        "000000001440000000000000084000000000000018403300000001000000020000000000000002000000730068690000",
        "0000010000000100000001000000020000007a00000000000000f03f0000000000000040020000000300000003000000",
        "00000000030000007370000000000000000040000000000000f03f0000000000000040000000000000f03f0000000000",
        "00004000000000000000400000000000000040000000000000f83f00000000000000001e000000010000000200000000",
        "00000002000000690007000800",
    );

    #[test]
    fn savemat_v5_writes_scipys_bytes() {
        let written = savemat(&scipy_reference_content(), &SavematOptions::default())
            .expect("reference content writes");
        assert_eq!(&written[..10], b"MATLAB 5.0");
        assert_eq!(written[116..], hex_bytes(SCIPY_V5_REFERENCE_FROM_116));
        // Must-miss: a column-oriented 1-D write differs exactly in `v`'s dimensions.
        let column = savemat(
            &scipy_reference_content(),
            &SavematOptions {
                oned_as: OnedAs::Column,
                ..SavematOptions::default()
            },
        )
        .expect("column write");
        assert_ne!(column[116..], hex_bytes(SCIPY_V5_REFERENCE_FROM_116));
        assert_eq!(
            whosmat(&column, &LoadmatOptions::default())
                .expect("inventory")
                .iter()
                .find(|info| info.name == "v")
                .map(|info| info.shape.clone()),
            Some(vec![3, 1])
        );
    }

    #[test]
    fn savemat_v4_writes_scipys_bytes() {
        let content = vec![
            (
                "x".to_string(),
                MatValue::Numeric(
                    MatNumeric::from_row_major(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("2x3"),
                ),
            ),
            ("s".to_string(), MatValue::Char(MatChar::row("hi"))),
            (
                "z".to_string(),
                MatValue::Numeric(MatNumeric::complex(
                    vec![1, 1],
                    MatData::F64(vec![1.0]),
                    MatData::F64(vec![2.0]),
                )),
            ),
            (
                "sp".to_string(),
                MatValue::Sparse(MatSparse {
                    rows: 2,
                    cols: 2,
                    logical: false,
                    indptr: vec![0, 1, 2],
                    indices: vec![1, 0],
                    data: MatData::F64(vec![2.0, 1.5]),
                    imag: None,
                }),
            ),
            ("i".to_string(), numeric(&[1, 2], MatData::I16(vec![7, 8]))),
        ];
        let options = SavematOptions {
            format: MatFormat::V4,
            ..SavematOptions::default()
        };
        let written = savemat(&content, &options).expect("v4 write");
        assert_eq!(written, hex_bytes(SCIPY_V4_REFERENCE));
        let loaded = load_default(&written);
        assert_eq!(loaded.version, (0, 0));
        assert_eq!(
            loaded.get("x"),
            Some(&numeric(
                &[2, 3],
                MatData::F64(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            ))
        );
        assert_eq!(loaded.get("s"), Some(&strings(&[1], 2, &["hi"])));
        assert_eq!(
            loaded.get("sp"),
            content.iter().find(|(n, _)| n == "sp").map(|(_, v)| v)
        );
        // A cell cannot be written to Level 4 (SciPy: "Cannot save object arrays in Mat4").
        let cell = vec![(
            "c".to_string(),
            MatValue::Cell(MatCell {
                dims: vec![1, 1],
                items: vec![numeric(&[1, 1], MatData::F64(vec![1.0]))],
            }),
        )];
        assert!(matches!(
            savemat(&cell, &options),
            Err(IoError::UnsupportedFeature(_))
        ));
        let cube = vec![(
            "c".to_string(),
            numeric(&[1, 1, 2], MatData::F64(vec![1.0, 2.0])),
        )];
        assert!(
            matches!(savemat(&cube, &options), Err(IoError::InvalidFormat(m)) if m.contains("more than 2"))
        );
    }

    #[test]
    fn savemat_v5_round_trips_every_value_kind_compressed_and_not() {
        let mut content = scipy_reference_content();
        content.push((
            "o".to_string(),
            MatValue::Object(MatObject {
                class_name: "inline".to_string(),
                fields: MatStruct {
                    dims: vec![1, 1],
                    field_names: vec!["expr".to_string()],
                    values: vec![MatValue::Char(MatChar::row("x^2"))],
                },
            }),
        ));
        content.push((
            "spc".to_string(),
            MatValue::Sparse(MatSparse {
                rows: 3,
                cols: 2,
                logical: false,
                indptr: vec![0, 2, 3],
                indices: vec![2, 0, 1],
                data: MatData::F64(vec![1.0, 2.0, 3.0]),
                imag: Some(MatData::F64(vec![-1.0, 0.5, 0.0])),
            }),
        ));
        content.push(("u".to_string(), MatValue::Char(MatChar::row("Grüße, 世界"))));
        content.push((
            "_hidden".to_string(),
            numeric(&[1, 1], MatData::F64(vec![9.0])),
        ));
        for do_compression in [false, true] {
            let options = SavematOptions {
                do_compression,
                ..SavematOptions::default()
            };
            let written = savemat(&content, &options).expect("write");
            let file = loadmat(
                &written,
                &LoadmatOptions {
                    chars_as_strings: false,
                    ..LoadmatOptions::default()
                },
            )
            .expect("read back");
            assert_eq!(
                file.variables.len(),
                content.len() - 1,
                "_hidden is skipped"
            );
            assert!(file.get("_hidden").is_none());
            // The row-sorted sparse matrix and the object come back as written.
            for name in ["s", "sp", "o", "u", "i", "z", "c", "e"] {
                assert_eq!(
                    file.get(name),
                    content.iter().find(|(n, _)| n == name).map(|(_, v)| v),
                    "{name} (compressed: {do_compression})"
                );
            }
            // Row indices are sorted within columns on write, as MATLAB requires.
            assert_eq!(
                file.get("spc"),
                Some(&MatValue::Sparse(MatSparse {
                    rows: 3,
                    cols: 2,
                    logical: false,
                    indptr: vec![0, 2, 3],
                    indices: vec![0, 2, 1],
                    data: MatData::F64(vec![2.0, 1.0, 3.0]),
                    imag: Some(MatData::F64(vec![0.5, -1.0, 0.0])),
                }))
            );
            // Logical data is stored as bytes: it reads back as uint8 unless mat_dtype asks for
            // the class dtype.
            assert_eq!(
                file.get("b"),
                Some(&MatValue::Numeric(MatNumeric {
                    dims: vec![1, 2],
                    class: MatClass::Uint8,
                    logical: true,
                    real: MatData::U8(vec![1, 0]),
                    imag: None,
                }))
            );
        }
        // Function handles cannot be written (SciPy raises MatWriteError).
        let function = vec![(
            "f".to_string(),
            MatValue::Function(Box::new(numeric(&[1, 1], MatData::F64(vec![1.0])))),
        )];
        assert!(matches!(
            savemat(&function, &SavematOptions::default()),
            Err(IoError::UnsupportedFeature(_))
        ));
        // Field names: 31 characters by default, 63 with long_field_names.
        let long = vec![(
            "s".to_string(),
            MatValue::Struct(MatStruct {
                dims: vec![1, 1],
                field_names: vec!["f".repeat(40)],
                values: vec![numeric(&[1, 1], MatData::F64(vec![1.0]))],
            }),
        )];
        assert!(matches!(
            savemat(&long, &SavematOptions::default()),
            Err(IoError::InvalidFormat(m)) if m == "Field names are restricted to 31 characters"
        ));
        let long_ok = savemat(
            &long,
            &SavematOptions {
                long_field_names: true,
                ..SavematOptions::default()
            },
        )
        .expect("63-character field names");
        assert_eq!(load_default(&long_ok).get("s"), Some(&long[0].1));
    }

    #[test]
    fn mmread_coordinate() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 3 3\n\
                        1 1 1.0\n\
                        2 2 2.0\n\
                        3 3 3.0\n";
        let mat = mmread(content).unwrap();
        assert_eq!(mat.rows, 3);
        assert_eq!(mat.cols, 3);
        assert_eq!(mat.data[0], 1.0); // (0,0)
        assert_eq!(mat.data[4], 2.0); // (1,1)
        assert_eq!(mat.data[8], 3.0); // (2,2)
        assert_eq!(mat.data[1], 0.0); // off-diagonal is zero
    }

    // frankenscipy-1f4yh. The coordinate parser pulls fields straight off the
    // split_whitespace iterator, and a line with fewer than two tokens hits a
    // bare `continue`. Read on its own that looks like silent data loss: a
    // truncated file would load with entries quietly missing.
    //
    // It is not, and this pins why. The declared-nnz count backstops the skip,
    // so a short entry line still fails the read:
    //     InvalidFormat("coordinate format expected 3 entries but found 2")
    // SciPy fails the same file too, with a different message
    // (ValueError "Line 4: Invalid integer value."), so both fail closed and
    // there is no parity gap here.
    //
    // The test exists because the safety is NON-LOCAL: it lives in the nnz
    // check, not at the `continue`. Anyone relaxing or removing that count
    // check would turn the skip into real silent data loss, and nothing else in
    // the suite would notice.
    #[test]
    fn mmread_rejects_truncated_coordinate_entry_via_nnz_backstop() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 3 3\n\
                        1 1 1.0\n\
                        2\n\
                        3 3 3.0\n";
        let err = mmread(content).expect_err("a truncated entry line must not load silently");
        match err {
            IoError::InvalidFormat(detail) => assert!(
                detail.contains("expected 3 entries but found 2"),
                "expected the nnz-shortfall diagnostic, got: {detail}"
            ),
            other => panic!("expected InvalidFormat from the nnz backstop, got {other:?}"),
        }

        // Control: the same file with the entry intact loads fine, so the
        // rejection above is caused by the truncation and not by the fixture.
        let intact = "%%MatrixMarket matrix coordinate real general\n\
                        3 3 3\n\
                        1 1 1.0\n\
                        2 2 2.0\n\
                        3 3 3.0\n";
        let m = mmread(intact).expect("intact file must load");
        assert_eq!(m.data[4], 2.0);
    }

    #[test]
    fn mmread_sparse_matches_dense_mmread() {
        // The COO triplets from mmread_sparse, scattered into a dense array with
        // `+=`, must reproduce mmread's dense `data` BIT-FOR-BIT across every
        // coordinate symmetry/field (general, symmetric, skew, duplicates,
        // pattern) — mmread_sparse is the no-dense-materialization sibling.
        let cases = [
            "%%MatrixMarket matrix coordinate real general\n3 3 3\n1 1 1.0\n2 2 2.0\n3 3 3.0\n",
            "%%MatrixMarket matrix coordinate real symmetric\n3 3 2\n1 1 5.0\n2 1 3.0\n",
            "%%MatrixMarket matrix coordinate real skew-symmetric\n3 3 1\n1 3 2.0\n",
            // duplicate (r,c) entries: COO keeps both, dense scatter sums them.
            "%%MatrixMarket matrix coordinate real general\n2 2 3\n1 1 1.0\n1 1 2.5\n2 1 -1.0\n",
            // pattern field: every stored entry is value 1.0.
            "%%MatrixMarket matrix coordinate pattern general\n3 3 2\n1 2\n3 1\n",
        ];
        for content in cases {
            let dense = mmread(content).unwrap();
            let sp = mmread_sparse(content).unwrap();
            assert_eq!((sp.rows, sp.cols), (dense.rows, dense.cols));
            assert_eq!(sp.row_indices.len(), sp.values.len());
            assert_eq!(sp.col_indices.len(), sp.values.len());
            let mut recon = vec![0.0f64; dense.rows * dense.cols];
            for k in 0..sp.values.len() {
                recon[sp.row_indices[k] * sp.cols + sp.col_indices[k]] += sp.values[k];
            }
            for (i, (&r, &d)) in recon.iter().zip(dense.data.iter()).enumerate() {
                assert_eq!(r.to_bits(), d.to_bits(), "mismatch at flat index {i}");
            }
        }
        // Array (dense) format is not a coordinate matrix → explicit error.
        let arr = "%%MatrixMarket matrix array real general\n%\n2 2\n1\n2\n3\n4\n";
        assert!(mmread_sparse(arr).is_err());
        // nnz mismatch is rejected just like mmread.
        let bad = "%%MatrixMarket matrix coordinate real general\n3 3 5\n1 1 1.0\n";
        assert!(mmread_sparse(bad).is_err());
    }

    #[test]
    fn mmwrite_parallel_path_matches_serial_and_roundtrips() {
        // A matrix above MM_PAR_GATE (65536 elements) exercises mmwrite's
        // parallel formatting path. It must (a) reproduce the exact same text a
        // single serial pass would and (b) round-trip through mmread bit-for-bit.
        let rows = 300usize;
        let cols = 256usize; // 76800 > 65536 → parallel path
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| ((i as f64) * 0.013).cos() - 0.5 * (i as f64).sqrt())
            .collect();
        let text = mmwrite(rows, cols, &data).unwrap();

        // Reference serial output (column-major, same Display formatting).
        let mut expected = String::from("%%MatrixMarket matrix array real general\n");
        expected.push_str(&format!("{rows} {cols}\n"));
        for c in 0..cols {
            for r in 0..rows {
                expected.push_str(&format!("{}\n", data[r * cols + c]));
            }
        }
        assert_eq!(text, expected, "parallel mmwrite must equal serial output");

        // Round-trip: mmread reconstructs the dense matrix bit-for-bit.
        let back = mmread(&text).unwrap();
        assert_eq!((back.rows, back.cols), (rows, cols));
        for (i, (&orig, &got)) in data.iter().zip(back.data.iter()).enumerate() {
            assert_eq!(orig.to_bits(), got.to_bits(), "roundtrip mismatch at {i}");
        }
    }

    #[test]
    fn mmread_symmetric() {
        let content = "%%MatrixMarket matrix coordinate real symmetric\n\
                        3 3 2\n\
                        1 1 5.0\n\
                        2 1 3.0\n";
        let mat = mmread(content).unwrap();
        assert_eq!(mat.data[0], 5.0); // (0,0)
        assert_eq!(mat.data[3], 3.0); // (1,0)
        assert_eq!(mat.data[1], 3.0); // (0,1) = symmetric
    }

    #[test]
    fn mmread_array_symmetry_matches_scipy() {
        // Exact strings from scipy.io.mmwrite(A, symmetry=...) (SciPy 1.17.1).
        // Symmetric/hermitian array files store only the lower triangle incl.
        // diagonal; skew-symmetric the strictly-lower triangle. The Array branch
        // must reconstruct the full matrix (mirroring, negating for skew) — it
        // previously demanded rows*cols values and failed on these files.
        let sym = "%%MatrixMarket matrix array real symmetric\n%\n3 3\n1\n2\n3\n4\n5\n6\n";
        let m = mmread(sym).expect("symmetric array");
        assert_eq!((m.rows, m.cols), (3, 3));
        assert_eq!(m.data, vec![1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0]);

        let skew = "%%MatrixMarket matrix array real skew-symmetric\n%\n3 3\n-2\n-3\n-5\n";
        let m = mmread(skew).expect("skew array");
        assert_eq!(m.data, vec![0.0, 2.0, 3.0, -2.0, 0.0, 5.0, -3.0, -5.0, 0.0]);

        // General (rectangular) must remain a plain column-major read.
        let rect = "%%MatrixMarket matrix array real general\n%\n2 3\n1\n4\n2\n5.5\n3\n-6\n";
        let m = mmread(rect).expect("general array");
        assert_eq!((m.rows, m.cols), (2, 3));
        assert_eq!(m.data, vec![1.0, 2.0, 3.0, 4.0, 5.5, -6.0]);
    }

    #[test]
    fn mmread_coordinate_sums_duplicates() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        2 2 3\n\
                        1 1 1.0\n\
                        1 1 2.5\n\
                        2 1 -1.0\n";
        let mat = mmread(content).unwrap();
        assert_eq!(mat.data[0], 3.5);
        assert_eq!(mat.data[2], -1.0);
        assert_eq!(mat.data[1], 0.0);
    }

    #[test]
    fn mmread_skew_symmetric() {
        let content = "%%MatrixMarket matrix coordinate real skew-symmetric\n\
                        3 3 1\n\
                        1 3 2.0\n";
        let mat = mmread(content).unwrap();
        assert_eq!(mat.data[2], 2.0); // (0,2)
        assert_eq!(mat.data[6], -2.0); // (2,0)
    }

    #[test]
    fn mmread_skew_symmetric_rejects_nonzero_diagonal() {
        let content = "%%MatrixMarket matrix coordinate real skew-symmetric\n\
                        2 2 1\n\
                        1 1 1.0\n";
        let err = mmread(content).expect_err("skew-symmetric diagonal must be zero");
        assert_eq!(
            err,
            IoError::InvalidFormat("skew-symmetric diagonal entries must be zero".to_string())
        );
    }

    #[test]
    fn mmread_rejects_rectangular_symmetric_coordinate_matrix() {
        let content = "%%MatrixMarket matrix coordinate real symmetric\n\
                        2 3 1\n\
                        1 3 2.0\n";
        let err = mmread(content).expect_err("rectangular symmetric Matrix Market should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "Matrix Market symmetric symmetry requires a square matrix, got 2x3".to_string()
            )
        );
    }

    #[test]
    fn mmread_rejects_zero_based_coordinate_indices() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 3 1\n\
                        0 1 5.0\n";
        let err = mmread(content).expect_err("zero-based row index should be rejected");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "Matrix Market row indices must be 1-based and >= 1".to_string()
            )
        );
    }

    #[test]
    fn mmread_rejects_out_of_bounds_coordinate_indices() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 3 1\n\
                        4 1 5.0\n";
        let err = mmread(content).expect_err("out-of-bounds row index should be rejected");
        assert_eq!(
            err,
            IoError::InvalidFormat("coordinate entry (3, 0) out of bounds for 3x3".to_string())
        );
    }

    #[test]
    fn mmread_rejects_coordinate_nnz_mismatch() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 3 2\n\
                        1 1 1.0\n";
        let err = mmread(content).expect_err("declared nnz mismatch should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("coordinate format expected 2 entries but found 1".to_string())
        );
    }

    #[test]
    fn mmread_rejects_missing_coordinate_value_for_real_field() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 3 1\n\
                        1 1\n";
        let err = mmread(content).expect_err("real coordinate entries need explicit values");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "coordinate entry missing value for non-pattern field".to_string()
            )
        );
    }

    #[test]
    fn mmread_complex_coordinate_preserves_hermitian_conjugates() {
        let content = "%%MatrixMarket matrix coordinate complex hermitian\n\
                        2 2 1\n\
                        2 1 1.0 2.0\n";
        let mat = mmread(content).expect("complex coordinate matrix should parse");
        assert_eq!(mat.data, vec![0.0, 1.0, 1.0, 0.0]);
        assert_eq!(
            mat.complex_data,
            Some(vec![(0.0, 0.0), (1.0, -2.0), (1.0, 2.0), (0.0, 0.0)])
        );
    }

    #[test]
    fn mmread_array() {
        let content = "%%MatrixMarket matrix array real general\n\
                        2 3\n\
                        1.0\n2.0\n3.0\n4.0\n5.0\n6.0\n";
        let mat = mmread(content).unwrap();
        assert_eq!(mat.rows, 2);
        assert_eq!(mat.cols, 3);
        // Column-major: 1,2 are col 0, 3,4 are col 1, 5,6 are col 2
        assert_eq!(mat.data[0], 1.0); // (0,0)
        assert_eq!(mat.data[3], 2.0); // (1,0)
        assert_eq!(mat.data[1], 3.0); // (0,1)
    }

    #[test]
    fn mmread_general_array_preserves_value_bits() {
        let content = "%%MatrixMarket matrix array real general\n\
                        2 3\n\
                        -0.0\n1.25\n0.0\n-2.5\n3.75\n-4.0\n";
        let mat = mmread(content).expect("general array");
        let expected = [-0.0_f64, 0.0, 3.75, 1.25, -2.5, -4.0];
        assert!(
            mat.data
                .iter()
                .zip(expected)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
        );
    }

    #[test]
    fn mminfo_reads_coordinate_header_without_body() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 4 2\n";
        let info = mminfo(content).expect("header-only coordinate metadata should parse");
        assert_eq!(info.object, MmObject::Matrix);
        assert_eq!(info.format, MmFormat::Coordinate);
        assert_eq!(info.field, MmField::Real);
        assert_eq!(info.symmetry, MmSymmetry::General);
        assert_eq!(info.rows, 3);
        assert_eq!(info.cols, 4);
        assert_eq!(info.nnz, 2);
    }

    #[test]
    fn mminfo_reads_array_header_without_body() {
        let content = "%%MatrixMarket matrix array real general\n\
                        2 3\n";
        let info = mminfo(content).expect("header-only array metadata should parse");
        assert_eq!(info.object, MmObject::Matrix);
        assert_eq!(info.format, MmFormat::Array);
        assert_eq!(info.field, MmField::Real);
        assert_eq!(info.symmetry, MmSymmetry::General);
        assert_eq!(info.rows, 2);
        assert_eq!(info.cols, 3);
        assert_eq!(info.nnz, 6);
    }

    #[test]
    fn mminfo_streaming_tokens_preserve_case_and_diagnostics() {
        let content = "%%MatrixMarket VeCtOr CoOrDiNaTe InTeGeR SkEw-SyMmEtRiC ignored\n\
                       % metadata comment\n\
                       3 3 2 ignored\n";
        let info = mminfo(content).expect("mixed-case metadata should parse");
        assert_eq!(info.object, MmObject::Vector);
        assert_eq!(info.format, MmFormat::Coordinate);
        assert_eq!(info.field, MmField::Integer);
        assert_eq!(info.symmetry, MmSymmetry::SkewSymmetric);
        assert_eq!((info.rows, info.cols, info.nnz), (3, 3, 2));

        assert_eq!(
            mminfo("%%MatrixMarket NoPe coordinate real general\n1 1 0\n")
                .expect_err("unknown object should fail"),
            IoError::InvalidFormat("unknown object type: nope".to_string())
        );
        assert_eq!(
            mminfo("%%MatrixMarket matrix array real general\n1\n")
                .expect_err("short size line should fail"),
            IoError::InvalidFormat("array format requires rows cols".to_string())
        );
    }

    #[test]
    fn mmread_complex_array_preserves_skew_mirror() {
        let content = "%%MatrixMarket matrix array complex skew-symmetric\n\
                        2 2\n\
                        1.0 2.0\n";
        let mat = mmread(content).expect("complex array matrix should parse");
        assert_eq!(mat.data, vec![0.0, -1.0, 1.0, 0.0]);
        assert_eq!(
            mat.complex_data,
            Some(vec![(0.0, 0.0), (-1.0, -2.0), (1.0, 2.0), (0.0, 0.0)])
        );
    }

    #[test]
    fn mmread_array_rejects_too_few_values() {
        let content = "%%MatrixMarket matrix array real general\n\
                        2 3\n\
                        1.0\n2.0\n3.0\n4.0\n5.0\n";
        let err = mmread(content).expect_err("underfilled array payload should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("array format expected 6 values but found 5".to_string())
        );
    }

    #[test]
    fn mmread_array_rejects_too_many_values() {
        let content = "%%MatrixMarket matrix array real general\n\
                        2 3\n\
                        1.0\n2.0\n3.0\n4.0\n5.0\n6.0\n7.0\n";
        let err = mmread(content).expect_err("overfilled array payload should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("array format has more than the declared 6 values".to_string())
        );
    }

    #[test]
    fn mmread_rejects_dense_size_overflow() {
        let content = format!(
            "%%MatrixMarket matrix coordinate real general\n{} 2 0\n",
            usize::MAX
        );
        let err = mmread(&content).expect_err("overflowing dense dimensions should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(format!(
                "Matrix Market matrix dimensions {}x2 overflowed usize",
                usize::MAX
            ))
        );
    }

    #[test]
    fn mmread_rejects_coordinate_dense_allocation_dos() {
        let content = "%%MatrixMarket matrix coordinate real general\n1000000 1000000 1\n1 1 1.0\n";
        let err = mmread(content).expect_err("hostile dense allocation should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(format!(
                "Matrix Market matrix dimensions 1000000x1000000 exceed dense read safety bound of {MAX_MM_DENSE_ELEMENTS} elements"
            ))
        );
    }

    #[test]
    fn mmwrite_complex_output_format() {
        // mmwrite_complex/mmwrite_sparse_complex were previously untested. Assert the
        // exact MatrixMarket text they emit; `mmread` accepts the resulting complex
        // Matrix Market payloads as well.
        let s = mmwrite_complex(1, 1, &[(3.0, -4.0)]).unwrap();
        assert_eq!(
            s,
            "%%MatrixMarket matrix array complex general\n1 1\n3 -4\n"
        );
        // sparse coordinate complex: (0,0)=1+2i, (1,1)=5+6i.
        let ss = mmwrite_sparse_complex(2, 2, &[(0, 0, (1.0, 2.0)), (1, 1, (5.0, 6.0))]).unwrap();
        assert_eq!(
            ss,
            "%%MatrixMarket matrix coordinate complex general\n2 2 2\n1 1 1 2\n2 2 5 6\n"
        );
    }

    #[test]
    fn mmwrite_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let content = mmwrite(2, 3, &data).unwrap();
        let mat = mmread(&content).unwrap();
        assert_eq!(mat.rows, 2);
        assert_eq!(mat.cols, 3);
        for (i, (&orig, &read)) in data.iter().zip(mat.data.iter()).enumerate() {
            assert!(
                (orig - read).abs() < 1e-10,
                "mismatch at {i}: orig={orig}, read={read}"
            );
        }
    }

    #[test]
    fn metamorphic_text_roundtrips_preserve_payload_and_metadata() {
        let dense_cases = [
            (1, 1, vec![42.0]),
            (2, 3, vec![1.0, -2.5, 3.25, 4.0, 0.0, 9.5]),
            (3, 2, vec![0.125, 1.25, 2.5, 5.0, 10.0, 20.0]),
        ];
        for (rows, cols, data) in dense_cases {
            let encoded = mmwrite(rows, cols, &data).expect("Matrix Market encode succeeds");
            let info = mminfo(&encoded).expect("mminfo should parse writer output");
            let decoded = mmread(&encoded).expect("mmread should parse writer output");

            assert_eq!(info.rows, rows);
            assert_eq!(info.cols, cols);
            assert_eq!(info.nnz, rows * cols);
            assert_eq!(info.rows, decoded.info.rows);
            assert_eq!(info.cols, decoded.info.cols);
            assert_eq!(info.nnz, decoded.info.nnz);
            assert_eq!(decoded.data, data);
        }

        let csv_rows = vec![vec![0.0, 1.5], vec![1.0, 2.25], vec![2.0, 3.0]];
        let csv = write_csv(Some(&["time", "value"]), &csv_rows, ',').expect("CSV encode");
        let (header, decoded_csv) = read_csv(&csv, ',', true).expect("CSV decode");
        assert_eq!(header, Some(vec!["time".to_string(), "value".to_string()]));
        assert_eq!(decoded_csv, csv_rows);

        let json_values = vec![1.5, -2.25, 0.0, 3.75];
        let json = write_json_array(&json_values).expect("JSON array encode");
        let decoded_json = read_json_array(&json).expect("JSON array decode");
        assert_eq!(decoded_json, json_values);
    }

    #[test]
    fn mmwrite_rejects_dense_size_overflow() {
        let err = mmwrite(usize::MAX, 2, &[])
            .expect_err("overflowing dense dimensions should fail before length comparison");
        assert_eq!(
            err,
            IoError::InvalidFormat(format!(
                "Matrix Market matrix dimensions {}x2 overflowed usize",
                usize::MAX
            ))
        );
    }

    #[test]
    fn wav_read_parallel_decode_matches_serial() {
        // Re-derived by afba78e19 (frankenscipy-wum95) after 4b42292a4 deleted
        // the original. Kept over the parent's version because it names
        // WAV_DECODE_PAR_GATE instead of hardcoding 262144 and uses the
        // as_chunks/try_from forms that c40f0641d went on to enforce.
        let sample_count = WAV_DECODE_PAR_GATE + 1_234;
        let mut payload = Vec::with_capacity(sample_count * 2);
        for index in 0..sample_count {
            let sample = ((index as u32).wrapping_mul(2_654_435_761) >> 8) as i16;
            payload.extend_from_slice(&sample.to_le_bytes());
        }
        let (samples, remainder) = payload.as_chunks::<2>();
        assert!(remainder.is_empty());
        let expected: Vec<f64> = samples
            .iter()
            .map(|&chunk| i16::from_le_bytes(chunk) as f64 / 32768.0)
            .collect();

        let payload_len = u32::try_from(payload.len()).expect("test payload fits WAV u32 length");
        let mut wav = Vec::with_capacity(44 + payload.len());
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + payload_len).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&44_100u32.to_le_bytes());
        wav.extend_from_slice(&88_200u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&payload_len.to_le_bytes());
        wav.extend_from_slice(&payload);

        let decoded = wav_read(&wav).expect("parallel WAV decode should succeed");
        assert_eq!(decoded.data.len(), expected.len());
        for (index, (actual, serial)) in decoded.data.iter().zip(expected.iter()).enumerate() {
            assert_eq!(actual.to_bits(), serial.to_bits(), "sample {index}");
        }
    }

    #[test]
    fn wav_roundtrip() {
        let samples = vec![0.0, 0.5, 1.0, -1.0, -0.5, 0.0];
        let bytes = wav_write(44100, 1, &samples).expect("mono samples should encode");
        let wav = wav_read(&bytes).unwrap();
        assert_eq!(wav.sample_rate, 44100);
        assert_eq!(wav.channels, 1);
        assert_eq!(wav.data.len(), samples.len());
        // 16-bit quantization: ~1/32768 precision
        for (i, (&orig, &read)) in samples.iter().zip(wav.data.iter()).enumerate() {
            assert!(
                (orig - read).abs() < 0.001,
                "sample {i}: orig={orig}, read={read}"
            );
        }
    }

    #[test]
    fn wav_read_matches_scipy_wavfile_structure_and_scale() {
        // Exact bytes from scipy.io.wavfile.write(8000, np.array(
        // [0, 16384, -16384, 32767, -32768], dtype=int16)) (SciPy 1.17.1).
        // fsci must parse the same rate/channels/bit-depth SciPy reports, and
        // its normalised samples must equal SciPy's raw int16 values / 32768.
        let scipy_wav: &[u8] = &[
            82, 73, 70, 70, 46, 0, 0, 0, 87, 65, 86, 69, 102, 109, 116, 32, 16, 0, 0, 0, 1, 0, 1,
            0, 64, 31, 0, 0, 128, 62, 0, 0, 2, 0, 16, 0, 100, 97, 116, 97, 10, 0, 0, 0, 0, 0, 0,
            64, 0, 192, 255, 127, 0, 128,
        ];
        let wav = wav_read(scipy_wav).expect("fsci must read scipy WAV bytes");
        assert_eq!(wav.sample_rate, 8000);
        assert_eq!(wav.channels, 1);
        assert_eq!(wav.bits_per_sample, 16);
        // SciPy returns these raw int16 samples; fsci returns them / 32768.
        let scipy_raw = [0i32, 16384, -16384, 32767, -32768];
        assert_eq!(wav.data.len(), scipy_raw.len());
        for (&got, &raw) in wav.data.iter().zip(scipy_raw.iter()) {
            assert!(
                (got - raw as f64 / 32768.0).abs() < 1e-12,
                "sample {got} != scipy {raw}/32768"
            );
        }
    }

    #[test]
    fn wav_write_rejects_partial_frames() {
        let err = wav_write(44_100, 2, &[0.0, 0.5, 1.0])
            .expect_err("stereo data with odd sample count should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "data length 3 does not contain whole frames for 2 channels".to_string()
            )
        );
    }

    #[test]
    fn wav_write_rejects_zero_sample_rate() {
        let err =
            wav_write(0, 1, &[0.0]).expect_err("zero sample rate should fail before encoding");
        assert_eq!(
            err,
            IoError::InvalidFormat("WAV sample rate must be nonzero".to_string())
        );
    }

    #[test]
    fn wav_write_rejects_block_align_overflow() {
        let err = wav_write(1, 32_768, &[])
            .expect_err("oversized channel count should fail before header overflow");
        assert_eq!(
            err,
            IoError::InvalidFormat("WAV block align overflowed u16".to_string())
        );
    }

    #[test]
    fn wav_read_rejects_partial_sample_bytes() {
        let mut bytes = wav_write(44_100, 1, &[0.0, 0.5]).expect("mono samples should encode");
        bytes[40..44].copy_from_slice(&3u32.to_le_bytes());
        bytes.truncate(44 + 3);

        let err = wav_read(&bytes).expect_err("misaligned sample bytes should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "data chunk size 3 is not aligned to 2-byte samples".to_string()
            )
        );
    }

    #[test]
    fn wav_read_rejects_partial_frames() {
        let mut bytes =
            wav_write(44_100, 2, &[0.0, 0.5, 1.0, -1.0]).expect("stereo samples should encode");
        bytes[40..44].copy_from_slice(&6u32.to_le_bytes());
        bytes.truncate(44 + 6);

        let err = wav_read(&bytes).expect_err("partial stereo frame should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "data chunk size 6 does not contain whole 2-channel frames".to_string()
            )
        );
    }

    #[test]
    fn wav_read_24bit_pcm_sign_extends_negative_samples() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + 3u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&44100u32.to_le_bytes());
        bytes.extend_from_slice(&(44100u32 * 3).to_le_bytes());
        bytes.extend_from_slice(&3u16.to_le_bytes());
        bytes.extend_from_slice(&24u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&[0x00, 0x00, 0x80]); // -1.0 in signed 24-bit PCM
        bytes.push(0); // pad odd-sized chunk

        let wav = wav_read(&bytes).expect("24-bit wav should decode");
        assert_eq!(wav.bits_per_sample, 24);
        assert!(
            (wav.data[0] + 1.0).abs() < 1e-6,
            "expected -1.0 sample, got {}",
            wav.data[0]
        );
    }

    #[test]
    fn wav_read_rejects_zero_channel_fmt() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + 2u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&44_100u32.to_le_bytes());
        bytes.extend_from_slice(&(44_100u32 * 2).to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i16.to_le_bytes());

        let err = wav_read(&bytes).expect_err("zero-channel WAV should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("fmt chunk declares zero channels".to_string())
        );
    }

    #[test]
    fn wav_read_rejects_data_before_fmt_chunk() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + 2u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i16.to_le_bytes());
        bytes.resize(44, 0);

        let err = wav_read(&bytes).expect_err("data before fmt should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("encountered data chunk before a valid fmt chunk".to_string())
        );
    }

    #[test]
    fn wav_read_rejects_float_with_non_32bit_samples() {
        let mut bytes = wav_write(44_100, 1, &[0.0]).expect("mono samples should encode");
        bytes[20..22].copy_from_slice(&3u16.to_le_bytes());

        let err = wav_read(&bytes).expect_err("float WAV with 16-bit samples should fail");
        assert_eq!(
            err,
            IoError::UnsupportedFeature("unsupported IEEE float bits per sample: 16".to_string())
        );
    }

    #[test]
    fn savemat_loadmat_roundtrip() {
        let arrays = vec![
            MatArray {
                name: "A".to_string(),
                rows: 2,
                cols: 2,
                data: vec![1.0, 2.0, 3.0, 4.0],
            },
            MatArray {
                name: "b".to_string(),
                rows: 3,
                cols: 1,
                data: vec![10.0, 20.0, 30.0],
            },
        ];
        let text = savemat_text(&arrays).expect("well-formed arrays should serialize");
        let loaded = loadmat_text(&text).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "A");
        assert_eq!(loaded[0].data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded[1].name, "b");
        assert_eq!(loaded[1].data, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn whosmat_reports_name_shape_and_class() {
        let bytes =
            savemat(&scipy_reference_content(), &SavematOptions::default()).expect("MAT encode");
        let info = |name: &str, shape: &[usize], class_name: &str| MatInfo {
            name: name.to_string(),
            shape: shape.to_vec(),
            class_name: class_name.to_string(),
        };
        // SciPy 1.17.1's whosmat of the same content (see SCIPY_V5_REFERENCE_FROM_116).
        assert_eq!(
            whosmat(&bytes, &LoadmatOptions::default()).expect("MAT inventory"),
            vec![
                info("s", &[1, 1], "struct"),
                info("c", &[1, 2], "cell"),
                info("sp", &[2, 2], "sparse"),
                info("b", &[1, 2], "logical"),
                info("z", &[1, 2], "single"),
                info("e", &[0, 0], "double"),
                info("st", &[2], "char"),
                info("i", &[2, 3, 4], "int8"),
                info("v", &[1, 3], "double"),
            ]
        );
        let squeezed = whosmat(
            &bytes,
            &LoadmatOptions {
                squeeze_me: true,
                chars_as_strings: false,
                ..LoadmatOptions::default()
            },
        )
        .expect("squeezed inventory");
        assert_eq!(squeezed[0], info("s", &[], "struct"));
        assert_eq!(squeezed[6], info("st", &[2, 3], "char"));
        assert_eq!(squeezed[8], info("v", &[3], "double"));
        let v4 = savemat(
            &[(
                "m".to_string(),
                numeric(&[2, 3], MatData::F32(vec![0.0; 6])),
            )],
            &SavematOptions {
                format: MatFormat::V4,
                ..SavematOptions::default()
            },
        )
        .expect("v4");
        // A Level 4 full matrix is "double" whatever it was stored as, as in SciPy.
        assert_eq!(
            whosmat(&v4, &LoadmatOptions::default()).expect("v4 inventory"),
            vec![info("m", &[2, 3], "double")]
        );
    }

    #[test]
    fn savemat_text_parallel_is_byte_identical_to_serial_above_gate() {
        use std::sync::atomic::Ordering;

        let rows = 10_000usize;
        let cols = 20usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|idx| idx as f64 * 0.001 + 1.0)
            .collect();
        let arrays = vec![MatArray {
            name: "A".to_string(),
            rows,
            cols,
            data,
        }];

        SAVEMAT_TEXT_FORCE_SERIAL.store(true, Ordering::Relaxed);
        let serial = savemat_text(&arrays).expect("serial savemat text");
        SAVEMAT_TEXT_FORCE_SERIAL.store(false, Ordering::Relaxed);
        let parallel = savemat_text(&arrays).expect("parallel savemat text");

        assert_eq!(serial, parallel, "parallel savemat_text must equal serial");
    }

    #[test]
    fn savemat_loadmat_binary_matches_scipy_semantics() {
        // scipy.io.loadmat(savemat({'x': [[1., 2., 3.]], 'y': [[1., 2.], [3., 4.]]})) gives back
        // the same 1x3 and 2x2 float64 arrays, in either format.
        let content = vec![
            (
                "x".to_string(),
                MatValue::Numeric(MatNumeric::from_row_major(1, 3, &[1.0, 2.0, 3.0]).expect("1x3")),
            ),
            (
                "y".to_string(),
                MatValue::Numeric(
                    MatNumeric::from_row_major(2, 2, &[1.0, 2.0, 3.0, 4.0]).expect("2x2"),
                ),
            ),
        ];
        for format in [MatFormat::V5, MatFormat::V4] {
            let bytes = savemat(
                &content,
                &SavematOptions {
                    format,
                    ..SavematOptions::default()
                },
            )
            .expect("savemat");
            let loaded = load_default(&bytes);
            assert_eq!(loaded.variables, content, "{format:?}");
            let y = loaded.get("y").and_then(|value| match value {
                MatValue::Numeric(n) => n.to_row_major_f64().ok(),
                _ => None,
            });
            assert_eq!(y, Some((2, 2, vec![1.0, 2.0, 3.0, 4.0])), "{format:?}");
        }
        assert!(MatNumeric::from_row_major(2, 2, &[1.0]).is_err());
    }

    #[test]
    fn matfile_version_matches_scipys_probe() {
        assert_eq!(
            matfile_version(&[1; 19]),
            Err(IoError::InvalidFormat(
                "Mat file appears to be truncated".to_string()
            ))
        );
        assert_eq!(
            matfile_version(&[0; 20]),
            Err(IoError::InvalidFormat(
                "Mat file appears to be corrupt (first 20 bytes == 0)".to_string()
            ))
        );
        // A zero byte among the first four is Level 4.
        let mut v4 = vec![b'x'; 40];
        v4[3] = 0;
        assert_eq!(matfile_version(&v4), Ok((0, 0)));
        assert_eq!(matfile_version(&mat5_header_bytes(false)), Ok((1, 0)));
        assert_eq!(matfile_version(&mat5_header_bytes(true)), Ok((1, 0)));
        // SciPy: "Unknown mat file type, version 231, 173" for its japanese_utf8.txt.
        let mut text = mat5_header_bytes(false);
        text[124..128].copy_from_slice(&[0xe7, 0xad, 0x89, 0xe3]);
        assert_eq!(
            matfile_version(&text),
            Err(IoError::InvalidFormat(
                "Unknown mat file type, version 231, 173".to_string()
            ))
        );
    }

    #[test]
    fn loadmat_fails_closed_on_truncated_corrupt_and_hdf5_files() {
        let opts = LoadmatOptions::default();
        assert_eq!(
            loadmat(&[0, 0, 0], &opts),
            Err(IoError::InvalidFormat(
                "Mat file appears to be truncated".to_string()
            ))
        );
        // v7.3 (HDF5): SciPy raises NotImplementedError.
        let mut hdf5 = mat5_header_bytes(false);
        hdf5[124..126].copy_from_slice(&0x0200u16.to_le_bytes());
        hdf5.extend_from_slice(&[0x89, b'H', b'D', b'F', 0x0d, 0x0a, 0x1a, 0x0a]);
        assert_eq!(matfile_version(&hdf5), Ok((2, 0)));
        assert!(
            matches!(loadmat(&hdf5, &opts), Err(IoError::UnsupportedFeature(m)) if m.contains("v7.3"))
        );
        assert!(matches!(
            whosmat(&hdf5, &opts),
            Err(IoError::UnsupportedFeature(_))
        ));

        // A file cut inside a variable fails as SciPy's stream read does; the whole file loads.
        let whole = mat5_file(&[scalar_double(b"x", 1.5), scalar_double(b"y", 2.5)]);
        assert_eq!(load_default(&whole).variables.len(), 2);
        for cut in [whole.len() - 1, whole.len() - 12, 128 + 20] {
            assert!(
                matches!(
                    loadmat(&whole[..cut], &opts),
                    Err(IoError::InvalidFormat(_))
                ),
                "cut at {cut}"
            );
        }
        // Trailing zero padding is a zero-length element to SciPy, and a partial tag a short read.
        let mut padded = whole.clone();
        padded.extend_from_slice(&[0; 8]);
        assert_eq!(
            loadmat(&padded, &opts),
            Err(IoError::InvalidFormat("Did not read any bytes".to_string()))
        );
        let mut trailing = whole.clone();
        trailing.extend_from_slice(&[1, 2, 3]);
        assert!(matches!(
            loadmat(&trailing, &opts),
            Err(IoError::InvalidFormat(m)) if m.starts_with("could not read bytes")
        ));
        // A top-level element that is not a matrix.
        let loose = mat5_file(&[mat5_element(MI_DOUBLE, &f64_bytes(&[1.0]))]);
        assert_eq!(
            loadmat(&loose, &opts),
            Err(IoError::InvalidFormat(
                "Expecting miMATRIX type here, got 9".to_string()
            ))
        );
        // An unknown numeric element type (SciPy 1.17.1 segfaults on it).
        let unknown = mat5_file(&[mat5_matrix(
            6,
            0,
            &[1, 1],
            b"x",
            &[mat5_element(11, &[0; 8])],
        )]);
        assert!(matches!(
            loadmat(&unknown, &opts),
            Err(IoError::InvalidFormat(m)) if m.contains("not a numeric type")
        ));
        // A negative dimension (SciPy reshapes it into nonsense).
        let negative = mat5_file(&[mat5_matrix(
            6,
            0,
            &[-1, 1],
            b"x",
            &[mat5_element(MI_DOUBLE, &[])],
        )]);
        assert!(matches!(
            loadmat(&negative, &opts),
            Err(IoError::InvalidFormat(m)) if m.contains("negative dimension")
        ));
        // Dimensions longer than SciPy's 32-value buffer.
        let many_dims = mat5_file(&[mat5_matrix(6, 0, &[1; 33], b"x", &[])]);
        assert_eq!(
            loadmat(&many_dims, &opts),
            Err(IoError::InvalidFormat(
                "Unexpected amount of data to read (malformed input file?)".to_string()
            ))
        );
        // A sparse row index outside the matrix (SciPy builds an invalid csc_matrix), with the
        // in-range control loading.
        let sparse = |row: i32| {
            mat5_file(&[mat5_matrix(
                5,
                0,
                &[2, 2],
                b"s",
                &[
                    mat5_element(MI_INT32, &i32_bytes(&[0, row])),
                    mat5_element(MI_INT32, &i32_bytes(&[0, 1, 2])),
                    mat5_element(MI_DOUBLE, &f64_bytes(&[5.0, 6.0])),
                ],
            )])
        };
        assert!(loadmat(&sparse(1), &opts).is_ok());
        assert!(matches!(
            loadmat(&sparse(7), &opts),
            Err(IoError::InvalidFormat(m)) if m.contains("outside the 2-row matrix")
        ));
        // Nesting deeper than the stack guard, with a shallow control.
        let nested = |depth: usize| {
            let mut inner = scalar_double(b"", 1.0);
            for _ in 0..depth {
                inner = mat5_matrix(MatClass::Cell.code(), 0, &[1, 1], b"", &[inner]);
            }
            mat5_file(&[mat5_matrix(
                MatClass::Cell.code(),
                0,
                &[1, 1],
                b"deep",
                &[inner],
            )])
        };
        assert!(loadmat(&nested(20), &opts).is_ok());
        assert!(matches!(
            loadmat(&nested(150), &opts),
            Err(IoError::InvalidFormat(m)) if m.contains("nest deeper")
        ));
    }

    #[test]
    fn loadmat_handles_compressed_streams_like_scipys_zlib_input_stream() {
        let opts = LoadmatOptions::default();
        let raw = scalar_double(b"x", 1.5);
        let good = mat5_file(&[zlib_element(&raw)]);
        assert_eq!(
            load_default(&good).get("x"),
            Some(&numeric(&[1, 1], MatData::F64(vec![1.5])))
        );
        // Not a zlib stream: SciPy raises zlib.error.
        let mut bad_stream = mat5_header_bytes(false);
        bad_stream.extend_from_slice(&MI_COMPRESSED.to_le_bytes());
        bad_stream.extend_from_slice(&8u32.to_le_bytes());
        bad_stream.extend_from_slice(&[0; 8]);
        assert!(matches!(
            loadmat(&bad_stream, &opts),
            Err(IoError::InvalidFormat(m)) if m.contains("decompressing")
        ));
        // A damaged Adler-32 checksum.
        let mut bad_sum = good.clone();
        let last = bad_sum.len() - 1;
        bad_sum[last] ^= 0xff;
        assert_eq!(
            loadmat(&bad_sum, &opts),
            Err(IoError::InvalidFormat(
                "Error -3 while decompressing data: incorrect data check".to_string()
            ))
        );
        // A stream without its checksum decodes as far as it goes, as `decompressobj` does, and
        // passes the integrity check because the variable consumed all of it.
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&raw, 6);
        let mut no_checksum = mat5_header_bytes(false);
        no_checksum.extend_from_slice(&MI_COMPRESSED.to_le_bytes());
        no_checksum.extend_from_slice(
            &u32::try_from(packed.len() - 4)
                .expect("small")
                .to_le_bytes(),
        );
        no_checksum.extend_from_slice(&packed[..packed.len() - 4]);
        assert_eq!(
            load_default(&no_checksum).get("x"),
            Some(&numeric(&[1, 1], MatData::F64(vec![1.5])))
        );
        // Decompressed bytes left over after the variable: "Did not fully consume", unless the
        // check is turned off.
        let mut raw_plus = raw.clone();
        raw_plus.extend_from_slice(&[0; 8]);
        let overlong = mat5_file(&[zlib_element(&raw_plus)]);
        assert!(matches!(
            loadmat(&overlong, &opts),
            Err(IoError::InvalidFormat(m)) if m.starts_with("Did not fully consume")
        ));
        let lenient = LoadmatOptions {
            verify_compressed_data_integrity: false,
            ..LoadmatOptions::default()
        };
        assert_eq!(
            loadmat(&overlong, &lenient).map(|f| f.variables.len()),
            Ok(1)
        );
        // A compressed element the file does not fully hold fails the check too.
        let mut cut = good.clone();
        let count = u32::try_from(good.len() - 128 - 8 + 5).expect("small");
        cut[132..136].copy_from_slice(&count.to_le_bytes());
        assert!(matches!(
            loadmat(&cut, &opts),
            Err(IoError::InvalidFormat(m)) if m.starts_with("Did not fully consume")
        ));
    }

    #[test]
    fn loadmat_reads_elements_as_scipy_1_17_1_does() {
        // Each expectation is what live SciPy 1.17.1 returned for the same bytes.
        let char_file = |dims: &[i32], mdtype: u32, payload: &[u8]| {
            mat5_file(&[mat5_matrix(
                4,
                0,
                dims,
                b"c",
                &[mat5_element(mdtype, payload)],
            )])
        };
        let u16s = |units: &[u16]| {
            units
                .iter()
                .flat_map(|u| u.to_le_bytes())
                .collect::<Vec<u8>>()
        };
        let text = |file: Vec<u8>| load_default(&file).get("c").cloned();
        // miUINT16 units are narrowed to bytes and decoded as UTF-8 with replacement.
        assert_eq!(
            text(char_file(&[1, 3], MI_UINT16, &u16s(&[0x141, 0x42, 0x43]))),
            Some(strings(&[1], 3, &["ABC"]))
        );
        assert_eq!(
            text(char_file(&[1, 3], MI_UINT16, &u16s(&[0x80, 0x41, 0x42]))),
            Some(strings(&[1], 3, &["\u{fffd}AB"]))
        );
        assert!(
            loadmat(
                &char_file(&[1, 2], MI_UINT16, &u16s(&[0xc3, 0xa9, 0x41, 0x42])),
                &LoadmatOptions::default()
            )
            .is_err()
        );
        // miUTF16 with a lone surrogate, miUTF8 with a broken sequence, miINT8 above 127.
        assert_eq!(
            text(char_file(&[1, 3], MI_UTF16, &u16s(&[0x41, 0xd800, 0x42]))),
            Some(strings(&[1], 3, &["A\u{fffd}B"]))
        );
        assert_eq!(
            text(char_file(&[1, 4], MI_UTF8, b"\xe2\x82A\xffB")),
            Some(strings(&[1], 4, &["\u{fffd}A\u{fffd}B"]))
        );
        assert_eq!(
            text(char_file(&[1, 2], MI_INT8, &[0x41, 0x80])),
            Some(strings(&[1], 2, &["A\u{fffd}"]))
        );
        // Extra characters are ignored, too few are an error, no bytes at all are spaces.
        assert_eq!(
            text(char_file(&[1, 2], MI_UTF8, b"abcd")),
            Some(strings(&[1], 2, &["ab"]))
        );
        assert!(
            loadmat(
                &char_file(&[1, 4], MI_UTF8, b"ab"),
                &LoadmatOptions::default()
            )
            .is_err()
        );
        assert_eq!(
            text(char_file(&[1, 3], MI_UTF8, b"")),
            Some(strings(&[1], 3, &["   "]))
        );
        // chars_to_strings: trailing NULs end a string, the last axis is the string axis.
        assert_eq!(
            text(char_file(&[1, 4], MI_UTF8, b"ab\0\0")),
            Some(strings(&[1], 4, &["ab"]))
        );
        assert_eq!(
            text(char_file(&[2, 2], MI_UTF8, b"a\0b\0")),
            Some(strings(&[2], 2, &["ab", ""]))
        );
        assert_eq!(
            text(char_file(&[2, 1, 2], MI_UTF8, b"abcd")),
            Some(strings(&[2, 1], 2, &["ac", "bd"]))
        );
        assert_eq!(
            text(char_file(&[0, 5], MI_UTF8, b"")),
            Some(strings(&[0], 5, &[]))
        );
        assert_eq!(
            text(char_file(&[1, 0], MI_UTF8, b"")),
            Some(strings(&[0], 1, &[]))
        );
        assert_eq!(
            text(char_file(&[2, 3, 0], MI_UTF8, b"")),
            Some(strings(&[2, 0], 1, &[]))
        );
        // A numeric type is not char data.
        assert_eq!(
            loadmat(
                &char_file(&[1, 2], MI_DOUBLE, &f64_bytes(&[65.0, 66.0])),
                &LoadmatOptions::default()
            ),
            Err(IoError::InvalidFormat(
                "Type 9 does not appear to be char type".to_string()
            ))
        );

        // Complex parts: complex64 for a 4-byte real part (16777217 rounds in float32), and a
        // one-value imaginary part broadcasts.
        let complex = mat5_file(&[mat5_matrix(
            6,
            8,
            &[1, 2],
            b"z",
            &[
                mat5_element(MI_INT32, &i32_bytes(&[16_777_217, 2])),
                mat5_element(MI_UINT8, &[5, 6]),
            ],
        )]);
        assert_eq!(
            load_default(&complex).get("z"),
            Some(&MatValue::Numeric(MatNumeric {
                dims: vec![1, 2],
                class: MatClass::Double,
                logical: false,
                real: MatData::F32(vec![16_777_216.0, 2.0]),
                imag: Some(MatData::F32(vec![5.0, 6.0])),
            }))
        );
        let broadcast = mat5_file(&[mat5_matrix(
            6,
            8,
            &[1, 3],
            b"z",
            &[
                mat5_element(MI_DOUBLE, &f64_bytes(&[1.0, 2.0, 3.0])),
                mat5_element(MI_DOUBLE, &f64_bytes(&[5.0])),
            ],
        )]);
        let imag = match load_default(&broadcast).get("z") {
            Some(MatValue::Numeric(z)) => z.imag.clone(),
            _ => None,
        };
        assert_eq!(imag, Some(MatData::F64(vec![5.0; 3])));

        // Big-endian numeric data (SciPy returns '>f8' with the same values).
        let mut big = mat5_header_bytes(true);
        let payload: Vec<u8> = [1.5f64, -2.0]
            .iter()
            .flat_map(|x| x.to_be_bytes())
            .collect();
        big.extend(mat5_matrix_in(
            6,
            0,
            &[1, 2],
            b"x",
            &[mat5_element_in(MI_DOUBLE, &payload, true)],
            true,
        ));
        let file = load_default(&big);
        assert!(file.big_endian);
        assert_eq!(
            file.get("x"),
            Some(&numeric(&[1, 2], MatData::F64(vec![1.5, -2.0])))
        );

        // Struct field names: repeated names renamed as SciPy renames them, a name running past
        // its slot to the next NUL, and miUINT32 accepted for the name length.
        let field = |names: &[u8], width: i32| {
            mat5_file(&[mat5_matrix(
                2,
                0,
                &[1, 1],
                b"s",
                &[
                    mat5_element(MI_UINT32, &width.to_le_bytes()),
                    mat5_element(MI_INT8, names),
                    scalar_double(b"", 1.0),
                    scalar_double(b"", 2.0),
                    scalar_double(b"", 3.0),
                ],
            )])
        };
        let names = |file: Vec<u8>| match load_default(&file).get("s") {
            Some(MatValue::Struct(s)) => s.field_names.clone(),
            _ => Vec::new(),
        };
        assert_eq!(
            names(field(b"a\0\0a\0\0a\0\0", 3)),
            vec!["a", "_1_a", "_2_a"]
        );
        assert_eq!(names(field(b"abcd", 2)), vec!["abcd", "cd"]);
        assert!(loadmat(&field(b"ab", 0), &LoadmatOptions::default()).is_err());
    }

    #[test]
    fn loadmat_options_follow_scipy() {
        let logical = mat5_matrix(9, 2, &[2, 1], b"b", &[mat5_element(MI_UINT8, &[1, 0])]);
        let narrowed = mat5_matrix(
            8,
            0,
            &[1, 2],
            b"i8",
            &[mat5_element(MI_DOUBLE, &f64_bytes(&[1.0, -1.5]))],
        );
        let complex = mat5_matrix(
            6,
            8,
            &[1, 1],
            b"z",
            &[
                mat5_element(MI_DOUBLE, &f64_bytes(&[1.0])),
                mat5_element(MI_DOUBLE, &f64_bytes(&[2.0])),
            ],
        );
        let cell = mat5_matrix(1, 0, &[1, 1], b"c", &[scalar_double(b"", 3.5)]);
        let structs = mat5_matrix(
            2,
            0,
            &[1, 2],
            b"sa",
            &[
                mat5_element(MI_INT32, &2i32.to_le_bytes()),
                mat5_element(MI_INT8, b"f\0"),
                scalar_double(b"", 1.0),
                scalar_double(b"", 2.0),
            ],
        );
        let object = mat5_matrix(
            3,
            0,
            &[1, 1],
            b"o",
            &[
                mat5_element(MI_INT8, b"cls"),
                mat5_element(MI_INT32, &2i32.to_le_bytes()),
                mat5_element(MI_INT8, b"f\0"),
                scalar_double(b"", 4.0),
            ],
        );
        let workspace = mat5_matrix(9, 0, &[1, 3], b"", &[mat5_element(MI_UINT8, &[1, 2, 3])]);
        let empty = mat5_matrix(6, 0, &[0, 3], b"e", &[mat5_element(MI_DOUBLE, &[])]);
        let file = mat5_file(&[
            logical, narrowed, complex, cell, structs, object, workspace, empty,
        ]);

        // Defaults: stored types, 2-D shapes.
        let plain = load_default(&file);
        assert_eq!(
            plain.get("b"),
            Some(&MatValue::Numeric(MatNumeric {
                dims: vec![2, 1],
                class: MatClass::Uint8,
                logical: true,
                real: MatData::U8(vec![1, 0]),
                imag: None,
            }))
        );
        // mat_dtype: logical -> bool, the int8 class's dtype (truncation), complex -> real part.
        let typed = loadmat(
            &file,
            &LoadmatOptions {
                mat_dtype: true,
                ..LoadmatOptions::default()
            },
        )
        .expect("mat_dtype");
        let data = |file: &MatFile, name: &str| match file.get(name) {
            Some(MatValue::Numeric(n)) => Some((n.real.clone(), n.imag.clone())),
            _ => None,
        };
        assert_eq!(
            data(&typed, "b"),
            Some((MatData::Bool(vec![true, false]), None))
        );
        assert_eq!(data(&typed, "i8"), Some((MatData::I8(vec![1, -1]), None)));
        assert_eq!(data(&typed, "z"), Some((MatData::F64(vec![1.0]), None)));
        // The unnamed function workspace is read raw: no mat_dtype, no squeeze.
        assert_eq!(
            data(&typed, "__function_workspace__"),
            Some((MatData::U8(vec![1, 2, 3]), None))
        );

        // squeeze_me: unit dimensions go, a 1x1 cell is its element, empty is 1-D length 0.
        let squeezed = loadmat(
            &file,
            &LoadmatOptions {
                squeeze_me: true,
                ..LoadmatOptions::default()
            },
        )
        .expect("squeeze");
        assert_eq!(
            squeezed.get("c"),
            Some(&numeric(&[], MatData::F64(vec![3.5])))
        );
        assert_eq!(
            squeezed.get("e"),
            Some(&numeric(&[0], MatData::F64(Vec::new())))
        );
        let dims = |file: &MatFile, name: &str| match file.get(name) {
            Some(MatValue::Numeric(n)) => Some(n.dims.clone()),
            _ => None,
        };
        assert_eq!(dims(&squeezed, "b"), Some(vec![2]));
        assert_eq!(dims(&squeezed, "__function_workspace__"), Some(vec![1, 3]));

        // simplify_cells: a 1x2 struct array is a list of dicts, a 1x1 object a plain dict.
        let simple = loadmat(
            &file,
            &LoadmatOptions {
                simplify_cells: true,
                ..LoadmatOptions::default()
            },
        )
        .expect("simplify_cells");
        let dict = |value: f64| {
            MatValue::Struct(MatStruct {
                dims: Vec::new(),
                field_names: vec!["f".to_string()],
                values: vec![numeric(&[], MatData::F64(vec![value]))],
            })
        };
        assert_eq!(
            simple.get("sa"),
            Some(&MatValue::Cell(MatCell {
                dims: vec![2],
                items: vec![dict(1.0), dict(2.0)],
            }))
        );
        assert_eq!(simple.get("o"), Some(&dict(4.0)));
        // Without simplify_cells the object keeps its class.
        assert!(matches!(plain.get("o"), Some(MatValue::Object(o)) if o.class_name == "cls"));

        // variable_names: only those, in file order.
        let picked = loadmat(
            &file,
            &LoadmatOptions {
                variable_names: Some(vec!["z".to_string(), "b".to_string()]),
                ..LoadmatOptions::default()
            },
        )
        .expect("variable_names");
        let names: Vec<&str> = picked.variables.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["b", "z"]);
        let none = loadmat(
            &file,
            &LoadmatOptions {
                variable_names: Some(Vec::new()),
                ..LoadmatOptions::default()
            },
        )
        .expect("no variables");
        assert!(none.variables.is_empty());

        // A repeated name keeps its first position and its last value; the global flag is kept.
        let repeated = mat5_file(&[
            scalar_double(b"x", 1.0),
            scalar_double(b"y", 2.0),
            mat5_matrix(
                6,
                4,
                &[1, 1],
                b"x",
                &[mat5_element(MI_DOUBLE, &f64_bytes(&[3.0]))],
            ),
        ]);
        let file = load_default(&repeated);
        assert_eq!(
            file.variables[0],
            ("x".to_string(), numeric(&[1, 1], MatData::F64(vec![3.0])))
        );
        assert_eq!(file.header.map(|h| h.globals), Some(vec!["x".to_string()]));
    }

    #[test]
    fn mat_data_cast_follows_numpy_astype() {
        // NumPy 2.4.3 on x86-64, identical for 4- and 9-element arrays (`mat_dtype` casts).
        let cast = |values: &[f64], dtype| MatData::F64(values.to_vec()).cast(dtype);
        assert_eq!(
            cast(&[300.7, -1.5, 70000.0, 40000.0], MatDtype::I8),
            MatData::I8(vec![44, -1, 112, 64])
        );
        assert_eq!(
            cast(&[300.7, -1.5, 1e10], MatDtype::U8),
            MatData::U8(vec![44, 255, 0])
        );
        assert_eq!(
            cast(&[70000.0, 40000.0, -70000.0], MatDtype::I16),
            MatData::I16(vec![4464, -25536, -4464])
        );
        assert_eq!(
            cast(&[f64::NAN, f64::INFINITY, -3e9, 2.5], MatDtype::I32),
            MatData::I32(vec![i32::MIN, i32::MIN, i32::MIN, 2])
        );
        assert_eq!(
            cast(&[1e19, -2.5], MatDtype::I64),
            MatData::I64(vec![i64::MIN, -2])
        );
        assert_eq!(
            cast(
                &[
                    -1.0,
                    9_223_372_036_854_775_808.0,
                    f64::NAN,
                    18_446_744_073_709_551_616.0
                ],
                MatDtype::U64
            ),
            MatData::U64(vec![u64::MAX, 1 << 63, 1 << 63, 0])
        );
        assert_eq!(
            MatData::U64(vec![u64::MAX, 70000]).cast(MatDtype::I16),
            MatData::I16(vec![-1, 4464])
        );
        assert_eq!(
            MatData::I64(vec![(1 << 53) + 1]).cast(MatDtype::F32),
            MatData::F32(vec![9_007_199_254_740_992.0])
        );
        assert_eq!(
            cast(&[0.0, -0.0, f64::NAN, 2.0], MatDtype::Bool),
            MatData::Bool(vec![false, false, true, true])
        );
    }

    #[test]
    fn mat5_header_stamps_asctime_in_utc() {
        // Python's time.asctime(time.gmtime(s)).
        assert_eq!(asctime_utc(0), "Thu Jan  1 00:00:00 1970");
        assert_eq!(asctime_utc(951_782_400), "Tue Feb 29 00:00:00 2000");
        assert_eq!(asctime_utc(1_700_000_000), "Tue Nov 14 22:13:20 2023");
        assert_eq!(asctime_utc(4_102_444_800), "Fri Jan  1 00:00:00 2100");
        let header = mat5_file_header_bytes();
        assert!(header.starts_with(b"MATLAB 5.0 MAT-file Platform: "));
        assert_eq!(&header[124..128], &[0x00, 0x01, b'I', b'M']);
        assert_eq!(matfile_version(&header), Ok((1, 0)));
    }

    #[test]
    fn varmats_from_mat_splits_a_file_into_single_variable_files() {
        let written = savemat(
            &scipy_reference_content(),
            &SavematOptions {
                do_compression: true,
                ..SavematOptions::default()
            },
        )
        .expect("write");
        let parts = varmats_from_mat(&written).expect("split");
        assert_eq!(parts.len(), scipy_reference_content().len());
        let whole = load_default(&written);
        for ((name, part), (whole_name, value)) in parts.iter().zip(&whole.variables) {
            assert_eq!(name, whole_name);
            let single = load_default(part);
            assert_eq!(single.variables, vec![(name.clone(), value.clone())]);
        }
        assert!(matches!(
            varmats_from_mat(&hex_bytes(SCIPY_V4_REFERENCE)),
            Err(IoError::UnsupportedFeature(_))
        ));
    }

    #[test]
    fn loadmat_reads_real_scipy_mat4_bytes() {
        // No-mock differential coverage: these are the exact bytes produced by
        // `scipy.io.savemat(buf, {'A': [[1,2,3],[4,5,6]], 'v': [[10,20,30,40]]},
        // format='4')` (SciPy 1.17.1). fsci `loadmat` must read them back with
        // the same shapes and row-major values SciPy reports. This guards the
        // real MATLAB v4 wire format, not just an fsci→fsci round-trip.
        let scipy_mat4: &[u8] = &[
            0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 65, 0, 0, 0, 0, 0, 0, 0,
            240, 63, 0, 0, 0, 0, 0, 0, 16, 64, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 20, 64,
            0, 0, 0, 0, 0, 0, 8, 64, 0, 0, 0, 0, 0, 0, 24, 64, 0, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0,
            0, 0, 0, 0, 2, 0, 0, 0, 118, 0, 0, 0, 0, 0, 0, 0, 36, 64, 0, 0, 0, 0, 0, 0, 52, 64, 0,
            0, 0, 0, 0, 0, 62, 64, 0, 0, 0, 0, 0, 0, 68, 64,
        ];
        let loaded = load_default(scipy_mat4);
        assert_eq!(loaded.version, (0, 0));
        assert!(loaded.header.is_none());
        assert_eq!(loaded.variables.len(), 2);
        // Column-major, as MATLAB and SciPy (`order='F'`) hold it.
        assert_eq!(
            loaded.get("A"),
            Some(&numeric(
                &[2, 3],
                MatData::F64(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            ))
        );
        assert_eq!(
            loaded.get("v"),
            Some(&numeric(
                &[1, 4],
                MatData::F64(vec![10.0, 20.0, 30.0, 40.0])
            ))
        );
    }

    #[test]
    fn loadmat_reads_real_scipy_mat5_bytes() {
        // No-mock differential coverage for MATLAB Level-5: exact bytes from
        // `scipy.io.savemat(buf, {'A': [[1,2,3],[4,5,6]], 'v': [[10,20,30,40]]},
        // format='5')` (SciPy 1.17.1, uncompressed default). fsci loadmat must
        // detect the v5 header and recover SciPy's shapes/row-major values.
        let scipy_mat5: &[u8] = &[
            77, 65, 84, 76, 65, 66, 32, 53, 46, 48, 32, 77, 65, 84, 45, 102, 105, 108, 101, 32, 80,
            108, 97, 116, 102, 111, 114, 109, 58, 32, 112, 111, 115, 105, 120, 44, 32, 67, 114,
            101, 97, 116, 101, 100, 32, 111, 110, 58, 32, 87, 101, 100, 32, 74, 117, 110, 32, 49,
            55, 32, 50, 49, 58, 53, 51, 58, 50, 57, 32, 50, 48, 50, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 73, 77, 14, 0, 0, 0, 96, 0, 0, 0, 6, 0, 0,
            0, 8, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 8, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0,
            1, 0, 1, 0, 65, 0, 0, 0, 9, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0,
            0, 0, 0, 16, 64, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 20, 64, 0, 0, 0, 0, 0, 0,
            8, 64, 0, 0, 0, 0, 0, 0, 24, 64, 14, 0, 0, 0, 80, 0, 0, 0, 6, 0, 0, 0, 8, 0, 0, 0, 6,
            0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 8, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 1, 0, 1, 0, 118,
            0, 0, 0, 9, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 64, 0, 0, 0, 0, 0, 0, 52, 64,
            0, 0, 0, 0, 0, 0, 62, 64, 0, 0, 0, 0, 0, 0, 68, 64,
        ];
        let loaded = load_default(scipy_mat5);
        assert_eq!(loaded.version, (1, 0));
        let header = loaded.header.clone().expect("Level 5 header");
        assert!(
            header
                .text
                .starts_with(b"MATLAB 5.0 MAT-file Platform: posix")
        );
        assert_eq!(header.version, "1.0");
        assert!(header.globals.is_empty());
        assert_eq!(
            loaded.get("A"),
            Some(&numeric(
                &[2, 3],
                MatData::F64(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            ))
        );
        assert_eq!(
            loaded.get("v"),
            Some(&numeric(
                &[1, 4],
                MatData::F64(vec![10.0, 20.0, 30.0, 40.0])
            ))
        );
    }

    #[test]
    fn savemat_binary_uses_mat4_column_major_double_layout() {
        let bytes = savemat(
            &[(
                "A".to_string(),
                MatValue::Numeric(
                    MatNumeric::from_row_major(2, 2, &[1.0, 2.0, 3.0, 4.0]).expect("2x2"),
                ),
            )],
            &SavematOptions {
                format: MatFormat::V4,
                ..SavematOptions::default()
            },
        )
        .expect("MAT v4 real doubles should serialize");

        let expected = [
            0, 0, 0, 0, // mopt: little-endian double full matrix
            2, 0, 0, 0, // rows
            2, 0, 0, 0, // cols
            0, 0, 0, 0, // imagf
            2, 0, 0, 0, // namlen, including trailing NUL
            b'A', 0, // name
            0, 0, 0, 0, 0, 0, 240, 63, // 1.0
            0, 0, 0, 0, 0, 0, 8, 64, // 3.0
            0, 0, 0, 0, 0, 0, 0, 64, // 2.0
            0, 0, 0, 0, 0, 0, 16, 64, // 4.0
        ];
        assert_eq!(bytes, expected);
    }

    #[test]
    fn loadmat_reads_complex_char_and_sparse_mat4_payloads() {
        let header = |mopt: i32, rows: i32, cols: i32, imagf: i32, name: &[u8]| {
            let mut out: Vec<u8> = [
                mopt,
                rows,
                cols,
                imagf,
                i32::try_from(name.len()).expect("short"),
            ]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
            out.extend_from_slice(name);
            out
        };
        let mut bytes = header(0, 1, 1, 1, b"z\0");
        bytes.extend(f64_bytes(&[1.0, 2.0]));
        assert_eq!(
            load_default(&bytes).get("z"),
            Some(&MatValue::Numeric(MatNumeric::complex(
                vec![1, 1],
                MatData::F64(vec![1.0]),
                MatData::F64(vec![2.0]),
            )))
        );
        // Char codes stored as doubles (teststring_4.2c_SOL2.mat does this), read as Latin-1.
        let mut chars = header(1, 1, 3, 0, b"c\0");
        chars.extend(f64_bytes(&[104.0, 233.0, 121.0]));
        assert_eq!(
            load_default(&chars).get("c"),
            Some(&strings(&[1], 3, &["héy"]))
        );
        // Sparse: 1-based (row, col, value) rows and a closing (rows, cols, 0) row, out of order
        // and with a duplicate, become canonical CSC.
        let mut sparse = header(2, 4, 3, 0, b"s\0");
        sparse.extend(f64_bytes(&[
            3.0, 1.0, 3.0, 3.0, 2.0, 1.0, 2.0, 2.0, 5.0, 7.0, 1.0, 0.0,
        ]));
        assert_eq!(
            load_default(&sparse).get("s"),
            Some(&MatValue::Sparse(MatSparse {
                rows: 3,
                cols: 2,
                logical: false,
                indptr: vec![0, 1, 2],
                indices: vec![0, 2],
                data: MatData::F64(vec![7.0, 6.0]),
                imag: None,
            }))
        );
        // Big-endian (mopt 1000): the file's first mopt decides the byte order.
        let mut big: Vec<u8> = [1000i32, 1, 1, 0, 2]
            .iter()
            .flat_map(|w| w.to_be_bytes())
            .collect();
        big.extend_from_slice(b"b\0");
        big.extend_from_slice(&(-0.5f64).to_be_bytes());
        let file = load_default(&big);
        assert!(file.big_endian);
        assert_eq!(
            file.get("b"),
            Some(&numeric(&[1, 1], MatData::F64(vec![-0.5])))
        );
        // A truncated matrix: SciPy's "Not enough bytes to read matrix" (debigged_m4.mat).
        assert!(matches!(
            loadmat(&bytes[..bytes.len() - 1], &LoadmatOptions::default()),
            Err(IoError::InvalidFormat(m)) if m.starts_with("Not enough bytes to read matrix 'z'")
        ));
    }

    #[test]
    fn savemat_rejects_wrong_element_count() {
        let arrays = vec![MatArray {
            name: "A".to_string(),
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0],
        }];
        let err = savemat_text(&arrays).expect_err("truncated matrix payload should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("array 'A' expected 4 values but found 3".to_string())
        );
    }

    #[test]
    fn savemat_rejects_multiline_names() {
        let arrays = vec![MatArray {
            name: "bad\nname".to_string(),
            rows: 1,
            cols: 1,
            data: vec![1.0],
        }];
        let err = savemat_text(&arrays).expect_err("multiline names should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "array name 'bad\\nname' contains a newline and cannot be encoded safely"
                    .to_string()
            )
        );
    }

    #[test]
    fn loadmat_rejects_wrong_element_count() {
        let text = "# name: A\n# type: matrix\n# rows: 2\n# columns: 2\n1 2\n";
        let err = loadmat_text(text).expect_err("truncated matrix payload should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("array 'A' expected 2 rows but found 1".to_string())
        );
    }

    #[test]
    fn loadmat_rejects_ragged_rows() {
        let text = "# name: A\n# type: matrix\n# rows: 2\n# columns: 2\n1 2 3\n4\n";
        let err = loadmat_text(text).expect_err("ragged rows should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("array 'A' row 0 has 3 columns, expected 2".to_string())
        );
    }

    #[test]
    fn loadmat_rejects_data_without_name_header() {
        let err = loadmat_text("1 2\n3 4\n")
            .expect_err("data block without a name header should fail closed");
        assert_eq!(
            err,
            IoError::InvalidFormat("encountered matrix data before '# name:' header".to_string())
        );
    }

    #[test]
    fn loadmat_rejects_incomplete_trailing_header_block() {
        let err = loadmat_text("# name: A\n# rows: 2\n# columns: 2\n")
            .expect_err("trailing header-only block should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("incomplete MAT text block at end of file".to_string())
        );
    }

    #[test]
    fn loadmat_rejects_data_before_dimension_headers_are_complete() {
        let err = loadmat_text("# name: A\n# rows: 2\n1 2\n3 4\n")
            .expect_err("data before full dimension metadata should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "array 'A' is missing nonzero '# rows:' and '# columns:' headers before data"
                    .to_string()
            )
        );
    }

    #[test]
    fn loadmat_rejects_dimension_overflow() {
        let text = format!(
            "# name: A\n# type: matrix\n# rows: {}\n# columns: 2\n1 2\n",
            usize::MAX
        );
        let err = loadmat_text(&text).expect_err("overflowing MAT dimensions should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(format!(
                "MAT text matrix dimensions {}x2 overflowed usize",
                usize::MAX
            ))
        );
    }

    #[test]
    fn read_idl_save_scalar_int32_matches_readsav_case_lookup() {
        let mut bytes = idl_save_header();
        idl_push_variable_record(&mut bytes, "I32S", 3, 0, |out| {
            idl_push_i32(out, -1_234_567_890);
        });
        idl_push_end_record(&mut bytes);

        let parsed = readsav(&bytes).expect("IDL SAVE scalar int32 should parse");

        assert_eq!(
            parsed.get("i32s"),
            Some(&IdlValue::Scalar(IdlScalar::Int32(-1_234_567_890)))
        );
        assert_eq!(
            parsed.get("I32S"),
            Some(&IdlValue::Scalar(IdlScalar::Int32(-1_234_567_890)))
        );
    }

    #[test]
    fn read_idl_save_scalar_string_preserves_bytes() {
        let mut bytes = idl_save_header();
        idl_push_variable_record(&mut bytes, "S", 7, 0, |out| {
            idl_push_string_data(out, b"The quick brown fox");
        });
        idl_push_end_record(&mut bytes);

        let parsed = read_idl_save(&bytes).expect("IDL SAVE string should parse");

        assert_eq!(
            parsed.get("s"),
            Some(&IdlValue::Scalar(IdlScalar::String(
                b"The quick brown fox".to_vec()
            )))
        );
    }

    #[test]
    fn read_idl_save_float32_array_reverses_dimensions_like_scipy() {
        let mut bytes = idl_save_header();
        idl_push_array_variable_record(&mut bytes, "ARRAY2D", 4, 24, 6, &[3, 2], |out| {
            for value in [1.0_f32, 2.0, 3.5, 4.5, 5.25, 6.75] {
                out.extend_from_slice(&value.to_be_bytes());
            }
        });
        idl_push_end_record(&mut bytes);

        let parsed = read_idl_save(&bytes).expect("IDL SAVE float32 array should parse");
        let value = parsed.get("array2d").expect("array variable present");

        let IdlValue::Array(array) = value else {
            assert!(
                matches!(value, IdlValue::Array(_)),
                "expected IDL array, got {value:?}"
            );
            return;
        };
        assert_eq!(array.element_type, IdlType::Float32);
        assert_eq!(array.dims, vec![2, 3]);
        assert_eq!(
            array.values,
            vec![
                IdlScalar::Float32(1.0),
                IdlScalar::Float32(2.0),
                IdlScalar::Float32(3.5),
                IdlScalar::Float32(4.5),
                IdlScalar::Float32(5.25),
                IdlScalar::Float32(6.75),
            ]
        );
    }

    #[test]
    fn read_idl_save_rejects_bad_signature_and_compressed_streams() {
        let bad_signature =
            read_idl_save(b"NO\x00\x04").expect_err("bad signature should fail closed");
        assert!(matches!(bad_signature, IoError::InvalidFormat(_)));

        let compressed =
            read_idl_save(b"SR\x00\x06").expect_err("compressed IDL SAVE should be unsupported");
        assert!(matches!(compressed, IoError::UnsupportedFeature(_)));
    }

    #[test]
    fn loadtxt_basic() {
        let content = "# comment\n1 2 3\n4 5 6\n";
        let (rows, cols, data) = loadtxt(content).unwrap();
        assert_eq!(rows, 2);
        assert_eq!(cols, 3);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn loadtxt_strips_inline_comments_like_numpy() {
        // numpy.loadtxt drops everything from the first '#' to end of line, so
        // trailing comments and a full-line comment are both handled.
        let content = "1 2 3 # first row\n# skip me\n4 5 6  #trailing\n7 8 9\n";
        let (rows, cols, data) = loadtxt(content).unwrap();
        assert_eq!((rows, cols), (3, 3));
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        // A '#' immediately after a value (no space) still delimits the comment.
        let (r, c, d) = loadtxt("1.5 2.5#note\n3.5 4.5\n").unwrap();
        assert_eq!((r, c), (2, 2));
        assert_eq!(d, vec![1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn savetxt_loadtxt_roundtrip_is_exact() {
        // savetxt uses f64 Display (shortest round-trippable form), so loadtxt
        // must recover the data bit-for-bit across signs, decimals, scientific
        // magnitudes, and zero — for several delimiters.
        let data = vec![
            0.0,
            -1.5,
            3.25,
            1.0e-12,
            -2.5e8,
            123_456.789,
            -0.000_125,
            9.0,
        ];
        for delim in [" ", "\t", ","] {
            let text = savetxt(2, 4, &data, delim).expect("savetxt");
            // loadtxt is whitespace-delimited; only exercise it for space/tab.
            if delim == "," {
                continue;
            }
            let (rows, cols, back) = loadtxt(&text).expect("loadtxt");
            assert_eq!((rows, cols), (2, 4), "shape for delim {delim:?}");
            assert_eq!(back, data, "round-trip exact for delim {delim:?}");
        }
        // A trailing inline comment appended to savetxt output is ignored.
        let text = savetxt(1, 3, &[1.0, 2.0, 3.0], " ").unwrap();
        let annotated = format!("{}# generated\n", text.trim_end());
        let (r, c, back) = loadtxt(&annotated).unwrap();
        assert_eq!((r, c, back), (1, 3, vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn savetxt_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let text = savetxt(2, 2, &data, " ").expect("matching shape should succeed");
        assert_eq!(text, "1 2\n3 4\n");
    }

    #[test]
    fn savetxt_parallel_matches_serial_output() {
        let rows = 4_000;
        let cols = 20;
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| (i as f64 * 0.125) - 500.0)
            .collect();

        SAVETXT_FORCE_SERIAL.store(true, std::sync::atomic::Ordering::Relaxed);
        let serial = savetxt(rows, cols, &data, " ").expect("serial savetxt");
        SAVETXT_FORCE_SERIAL.store(false, std::sync::atomic::Ordering::Relaxed);
        let parallel = savetxt(rows, cols, &data, " ").expect("parallel savetxt");

        assert_eq!(parallel, serial);
    }

    #[test]
    fn savetxt_rejects_shape_length_mismatch() {
        let err = savetxt(2, 2, &[1.0, 2.0, 3.0], " ").expect_err("mismatched shape should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("data length 3 doesn't match 2x2".to_string())
        );
    }

    #[test]
    fn savetxt_rejects_dimension_overflow() {
        let err = savetxt(usize::MAX, 2, &[], " ")
            .expect_err("overflowing savetxt dimensions should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(format!(
                "text matrix dimensions {}x2 overflowed usize",
                usize::MAX
            ))
        );
    }

    #[test]
    fn savetxt_rejects_multiline_delimiter() {
        let err = savetxt(1, 2, &[1.0, 2.0], "\n").expect_err("multiline delimiters should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "delimiter \"\\n\" contains a newline and cannot be encoded safely".to_string()
            )
        );
    }

    #[test]
    fn read_npy_text_rejects_shape_payload_mismatch() {
        let err = read_npy_text("2,2\n1 2 3\n").expect_err("truncated payload should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("shape [2, 2] expects 4 values but found 3".to_string())
        );
    }

    #[test]
    fn read_npy_text_rejects_empty_shape() {
        let err = read_npy_text("\n1\n").expect_err("empty shape line should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "shape declaration must contain at least one dimension".to_string()
            )
        );
    }

    #[test]
    fn read_npy_text_rejects_empty_shape_dimension() {
        let err = read_npy_text("2,,2\n1 2 3 4\n")
            .expect_err("shape declarations with empty dimensions should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("shape declaration contains an empty dimension".to_string())
        );
    }

    #[test]
    fn read_csv_rejects_ragged_rows() {
        let err = read_csv("1,2,3\n4,5\n", ',', false).expect_err("ragged CSV should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("CSV row has 2 columns, expected 3".to_string())
        );
    }

    #[test]
    fn read_csv_rejects_empty_input_when_header_is_required() {
        let err =
            read_csv("", ',', true).expect_err("empty input with required header should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("CSV header row is required but the input is empty".to_string())
        );
    }

    #[test]
    fn read_csv_rejects_header_data_column_mismatch() {
        let err =
            read_csv("a,b,c\n1,2\n3,4\n", ',', true).expect_err("header/data mismatch should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("CSV header has 3 columns but first data row has 2".to_string())
        );
    }

    #[test]
    fn write_csv_basic() {
        let text = write_csv(None, &[vec![1.0, 2.0], vec![3.0, 4.0]], ',')
            .expect("rectangular CSV should serialize");
        assert_eq!(text, "1,2\n3,4\n");
    }

    #[test]
    fn write_csv_parallel_is_byte_identical_to_serial_above_gate() {
        use std::sync::atomic::Ordering;
        // Above the rows·cols ≥ 2^16 fan-out gate the parallel per-row formatter must be
        // BYTE-IDENTICAL to the serial loop (rows formatted independently, joined in order).
        let (rows, cols) = (5000usize, 20usize); // 100000 > 65536 gate
        let mut state = 0x2468u64;
        let data: Vec<Vec<f64>> = (0..rows)
            .map(|_| {
                (0..cols)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        (state >> 11) as f64 / (1u64 << 53) as f64 * 200.0 - 100.0
                    })
                    .collect()
            })
            .collect();
        for delim in [',', ' ', ';'] {
            WRITE_CSV_FORCE_SERIAL.store(true, Ordering::Relaxed);
            let serial = write_csv(None, &data, delim).expect("serial");
            WRITE_CSV_FORCE_SERIAL.store(false, Ordering::Relaxed);
            let parallel = write_csv(None, &data, delim).expect("parallel");
            assert_eq!(
                serial, parallel,
                "delim {delim:?}: parallel write_csv must equal serial"
            );
        }
    }

    #[test]
    fn write_csv_rejects_header_data_column_mismatch() {
        let err = write_csv(Some(&["a", "b", "c"]), &[vec![1.0, 2.0]], ',')
            .expect_err("header/data mismatch should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("CSV header has 3 columns but first data row has 2".to_string())
        );
    }

    #[test]
    fn write_csv_rejects_header_cells_with_delimiters() {
        let err = write_csv(Some(&["bad,header"]), &[vec![1.0]], ',')
            .expect_err("header cells containing the delimiter should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "CSV header cell \"bad,header\" contains the delimiter ',' and cannot be encoded safely"
                    .to_string()
            )
        );
    }

    #[test]
    fn write_csv_rejects_header_cells_with_newlines() {
        let err = write_csv(Some(&["bad\nheader"]), &[vec![1.0]], ',')
            .expect_err("header cells containing newlines should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "CSV header cell \"bad\\nheader\" contains a newline and cannot be encoded safely"
                    .to_string()
            )
        );
    }

    #[test]
    fn write_csv_rejects_ragged_rows() {
        let err = write_csv(None, &[vec![1.0, 2.0, 3.0], vec![4.0, 5.0]], ',')
            .expect_err("ragged CSV should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("CSV row has 2 columns, expected 3".to_string())
        );
    }

    #[test]
    fn read_json_array_accepts_empty_array() {
        let data = read_json_array("[]").expect("empty array should parse");
        assert!(data.is_empty());
    }

    #[test]
    fn read_json_array_rejects_non_finite_values() {
        let err = read_json_array("[1.0, NaN]").expect_err("NaN is not valid JSON");
        assert_eq!(
            err,
            IoError::InvalidFormat("JSON parse error: non-finite value NaN".to_string())
        );
    }

    #[test]
    fn write_json_array_rejects_non_finite_values() {
        let err = write_json_array(&[1.0, f64::NAN]).expect_err("NaN is not valid JSON");
        assert_eq!(
            err,
            IoError::InvalidFormat("JSON array value at index 1 is not finite: NaN".to_string())
        );
    }

    #[test]
    fn write_json_array_parallel_is_byte_identical_to_serial_above_gate() {
        use std::sync::atomic::Ordering;
        // Above the len ≥ 2^16 fan-out gate the parallel per-chunk formatter must be
        // BIT-FOR-BIT the serial `[v0, v1, …]` (each chunk `", "`-joined, chunks joined
        // with `", "`).
        let n = 100_000usize; // > 65536 gate
        let mut state = 0x13579u64;
        let data: Vec<f64> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 11) as f64 / (1u64 << 53) as f64 * 200.0 - 100.0
            })
            .collect();
        WRITE_JSON_FORCE_SERIAL.store(true, Ordering::Relaxed);
        let serial = write_json_array(&data).expect("serial");
        WRITE_JSON_FORCE_SERIAL.store(false, Ordering::Relaxed);
        let parallel = write_json_array(&data).expect("parallel");
        assert_eq!(
            serial, parallel,
            "parallel write_json_array must equal serial"
        );
    }

    #[test]
    fn mmwrite_sparse_format() {
        let entries = vec![(0, 0, 1.0), (1, 1, 2.0)];
        let content = mmwrite_sparse(3, 3, &entries).unwrap();
        assert!(content.contains("coordinate"));
        let mat = mmread(&content).unwrap();
        assert_eq!(mat.data[0], 1.0);
        assert_eq!(mat.data[4], 2.0);
    }

    #[test]
    fn mmwrite_sparse_rejects_out_of_bounds_entries() {
        let err = mmwrite_sparse(2, 2, &[(2, 0, 1.0)])
            .expect_err("out-of-bounds sparse entry should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("sparse entry (2, 0) out of bounds for 2x2".to_string())
        );
    }

    #[test]
    fn read_csv_single_column_no_header() {
        let (header, data) = read_csv("1\n2\n3\n", ',', false).expect("single column CSV");
        assert_eq!(data.len(), 3);
        assert!(header.is_none());
        assert_eq!(data[0], vec![1.0]);
        assert_eq!(data[2], vec![3.0]);
    }

    #[test]
    fn read_csv_with_tab_delimiter() {
        let (header, data) = read_csv("a\tb\n1\t2\n3\t4\n", '\t', true).expect("TSV");
        assert_eq!(header, Some(vec!["a".to_string(), "b".to_string()]));
        assert_eq!(data[0], vec![1.0, 2.0]);
    }

    #[test]
    fn read_csv_empty_no_header() {
        let (_header, data) = read_csv("", ',', false).expect("empty without header is ok");
        assert!(data.is_empty());
    }

    #[test]
    fn netcdf_classic_roundtrip_double_matrix_with_attributes() {
        let file = NetcdfFile {
            dimensions: vec![
                NetcdfDimension {
                    name: "time".to_string(),
                    len: Some(2),
                },
                NetcdfDimension {
                    name: "station".to_string(),
                    len: Some(3),
                },
            ],
            attributes: vec![NetcdfAttribute {
                name: "title".to_string(),
                value: NetcdfValue::Char("demo".to_string()),
            }],
            variables: vec![NetcdfVariable {
                name: "temperature".to_string(),
                dim_ids: vec![0, 1],
                attributes: vec![NetcdfAttribute {
                    name: "units".to_string(),
                    value: NetcdfValue::Char("K".to_string()),
                }],
                data: NetcdfValue::Double(vec![280.0, 281.5, 282.25, 283.0, 284.5, 285.25]),
            }],
        };

        let bytes = write_netcdf_classic(&file).expect("NetCDF classic encode");
        assert_eq!(&bytes[..4], b"CDF\x01");
        let parsed = read_netcdf_classic(&bytes).expect("NetCDF classic decode");
        assert_eq!(parsed, file);
    }

    #[test]
    fn netcdf_padded_value_len_matches_encoded_payloads() {
        let values = [
            NetcdfValue::Byte(vec![-1, 0, 1]),
            NetcdfValue::Char("hello".to_string()),
            NetcdfValue::Short(vec![-2, 3, 4]),
            NetcdfValue::Int(vec![-5, 6]),
            NetcdfValue::Float(vec![1.25, -2.5, 3.75]),
            NetcdfValue::Double(vec![1.25, -2.5, 3.75]),
        ];

        for value in &values {
            assert_eq!(
                netcdf_padded_value_len(value).expect("checked padded length"),
                encode_netcdf_padded_values(value)
                    .expect("encoded payload")
                    .len()
            );
        }
    }

    #[test]
    fn netcdf_file_aliases_roundtrip_int_scalar() {
        let file = NetcdfFile {
            dimensions: Vec::new(),
            attributes: Vec::new(),
            variables: vec![NetcdfVariable {
                name: "answer".to_string(),
                dim_ids: Vec::new(),
                attributes: Vec::new(),
                data: NetcdfValue::Int(vec![42]),
            }],
        };

        let bytes = netcdf_file_write(&file).expect("NetCDF alias encode");
        let parsed = netcdf_file_read(&bytes).expect("NetCDF alias decode");
        assert_eq!(parsed.variables.len(), 1);
        assert_eq!(parsed.variables[0].name, "answer");
        assert_eq!(parsed.variables[0].data, NetcdfValue::Int(vec![42]));
    }

    #[test]
    fn netcdf_classic_metamorphic_variable_shape_matches_dimension_product() {
        let file = NetcdfFile {
            dimensions: vec![
                NetcdfDimension {
                    name: "x".to_string(),
                    len: Some(4),
                },
                NetcdfDimension {
                    name: "y".to_string(),
                    len: Some(2),
                },
            ],
            attributes: Vec::new(),
            variables: vec![NetcdfVariable {
                name: "mask".to_string(),
                dim_ids: vec![0, 1],
                attributes: Vec::new(),
                data: NetcdfValue::Byte(vec![1, 0, 1, 0, 1, 0, 1, 0]),
            }],
        };
        let bytes = write_netcdf_classic(&file).expect("NetCDF encode");
        let parsed = read_netcdf_classic(&bytes).expect("NetCDF decode");
        let variable = &parsed.variables[0];
        let expected_len = variable
            .dim_ids
            .iter()
            .map(|&dim_id| parsed.dimensions[dim_id].len.unwrap_or(0))
            .product::<usize>();
        assert_eq!(variable.data.len(), expected_len);
    }

    #[test]
    fn netcdf_classic_rejects_bad_magic() {
        let err = read_netcdf_classic(b"BAD\x01\0\0\0\0").expect_err("bad magic should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("NetCDF missing CDF magic".to_string())
        );
    }

    #[test]
    fn netcdf_header_size_shortcut_is_byte_identical_to_redundant_encoding() {
        // frankenscipy-99ru7. WRITE_NETCDF_FORCE_REDUNDANT_HEADER_ENCODING is the
        // only one of this crate's four restored levers that came back without an
        // A/B test. It restores the old path that fully ENCODED each variable's
        // padded payload just to measure its length; the lever computes that
        // length directly with netcdf_padded_value_len. The two must agree
        // exactly, and since the length is written into the header the whole file
        // must come out byte-identical — not merely the same length.
        //
        // No size gate: the toggle is read once per variable on every call. The
        // fixture deliberately mixes value types and includes a Char attribute
        // and a Short variable, because the padding rule differs per type and a
        // single-type file could agree by accident.
        use std::sync::atomic::Ordering;
        let file = NetcdfFile {
            dimensions: vec![
                NetcdfDimension {
                    name: "time".to_string(),
                    len: Some(2),
                },
                NetcdfDimension {
                    name: "station".to_string(),
                    len: Some(3),
                },
            ],
            attributes: vec![NetcdfAttribute {
                name: "title".to_string(),
                value: NetcdfValue::Char("byte-identity".to_string()),
            }],
            variables: vec![
                NetcdfVariable {
                    name: "temperature".to_string(),
                    dim_ids: vec![0, 1],
                    attributes: vec![NetcdfAttribute {
                        name: "units".to_string(),
                        value: NetcdfValue::Char("K".to_string()),
                    }],
                    data: NetcdfValue::Double(vec![280.0, 281.5, 282.25, 283.0, 284.5, 285.25]),
                },
                NetcdfVariable {
                    name: "flags".to_string(),
                    dim_ids: vec![1],
                    attributes: Vec::new(),
                    // 3 shorts = 6 bytes, so this variable actually exercises the
                    // 4-byte padding path rather than landing on a boundary.
                    data: NetcdfValue::Short(vec![-1, 0, 7]),
                },
            ],
        };

        WRITE_NETCDF_FORCE_REDUNDANT_HEADER_ENCODING.store(true, Ordering::Relaxed);
        let redundant = write_netcdf_classic(&file).expect("redundant-encoding arm");
        WRITE_NETCDF_FORCE_REDUNDANT_HEADER_ENCODING.store(false, Ordering::Relaxed);
        let shortcut = write_netcdf_classic(&file).expect("length-shortcut arm");

        assert_eq!(
            redundant.len(),
            shortcut.len(),
            "header size shortcut changed the file length"
        );
        assert_eq!(
            redundant, shortcut,
            "header size shortcut must be byte-identical to the redundant encoding"
        );
        // And the result must still round-trip, so neither arm is trivially wrong.
        assert_eq!(read_netcdf_classic(&shortcut).expect("round-trip"), file);
    }

    #[test]
    fn netcdf_classic_writer_rejects_unlimited_dimension() {
        let file = NetcdfFile {
            dimensions: vec![NetcdfDimension {
                name: "time".to_string(),
                len: None,
            }],
            attributes: Vec::new(),
            variables: Vec::new(),
        };
        let err = write_netcdf_classic(&file).expect_err("unlimited dims are follow-on work");
        assert!(matches!(err, IoError::UnsupportedFeature(_)));
    }

    /// Tiny canonical HB file: 3x3 RUA, 4 nonzeros at (0,0)=1.0, (1,1)=2.0,
    /// (2,1)=3.0, (2,2)=4.0. Title is exactly 72 chars long (padded), key 8.
    fn sample_hb_rua_3x3_4nnz() -> String {
        let title = format!("{:<72}", "Test 3x3 RUA");
        format!(
            "{title}KEY00001\n\
             4 1 1 1 0\n\
             RUA            3            3            4            0\n\
             (4I20)\n\
             (4I20)\n\
             (4D20.13)\n\
             1 2 3 5\n\
             1 2 3 3\n\
             1.0D+00 2.0D+00 3.0D+00 4.0D+00\n"
        )
    }

    #[test]
    fn read_harwell_boeing_rua_basic_dimensions() {
        let content = sample_hb_rua_3x3_4nnz();
        let mat = read_harwell_boeing(&content).expect("RUA parse");
        assert_eq!(mat.rows, 3);
        assert_eq!(mat.cols, 3);
        assert_eq!(mat.nnz, 4);
        assert_eq!(mat.matrix_type, HbType::RealUnsymmetricAssembled);
        assert_eq!(mat.col_ptr, vec![0, 1, 2, 4]);
        assert_eq!(mat.row_idx, vec![0, 1, 2, 2]);
        assert_eq!(mat.values, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(mat.title, "Test 3x3 RUA");
        assert_eq!(mat.key, "KEY00001");
    }

    #[test]
    fn read_harwell_boeing_metamorphic_value_sum_invariant() {
        let mat = read_harwell_boeing(&sample_hb_rua_3x3_4nnz()).unwrap();
        let direct: f64 = mat.values.iter().sum();
        let by_column: f64 = (0..mat.cols)
            .map(|j| {
                mat.values[mat.col_ptr[j]..mat.col_ptr[j + 1]]
                    .iter()
                    .sum::<f64>()
            })
            .sum();
        assert!((direct - by_column).abs() < 1e-12);
    }

    #[test]
    fn read_harwell_boeing_metamorphic_col_ptr_monotone() {
        let mat = read_harwell_boeing(&sample_hb_rua_3x3_4nnz()).unwrap();
        for w in mat.col_ptr.windows(2) {
            assert!(w[0] <= w[1], "col_ptr must be monotone non-decreasing");
        }
        assert_eq!(*mat.col_ptr.last().unwrap(), mat.nnz);
    }

    #[test]
    fn read_harwell_boeing_rejects_unsupported_complex() {
        let title = format!("{:<72}", "Bad type");
        let content = format!(
            "{title}KEY00002\n\
             4 1 1 1 0\n\
             CUA            3            3            4            0\n\
             (4I20)\n(4I20)\n(4D20.13)\n\
             1 2 3 5\n1 2 3 3\n1.0D+00 2.0D+00 3.0D+00 4.0D+00\n"
        );
        let err = read_harwell_boeing(&content)
            .expect_err("complex unsupported variant must be rejected");
        assert!(matches!(err, IoError::UnsupportedFeature(ref m) if m.contains("CUA")));
    }

    #[test]
    fn read_harwell_boeing_rejects_truncated_payload() {
        let title = format!("{:<72}", "Truncated");
        let content = format!(
            "{title}KEY00003\n\
             4 1 1 1 0\n\
             RUA            3            3            4            0\n\
             (4I20)\n(4I20)\n(4D20.13)\n\
             1 2 3 5\n"
        );
        let err = read_harwell_boeing(&content).expect_err("truncated payload must be rejected");
        assert!(matches!(err, IoError::InvalidFormat(_)));
    }

    #[test]
    fn read_harwell_boeing_rejects_non_monotone_col_ptr() {
        let title = format!("{:<72}", "Bad pointers");
        let content = format!(
            "{title}KEY00005\n\
             4 1 1 1 0\n\
             RUA            3            3            4            0\n\
             (4I20)\n(4I20)\n(4D20.13)\n\
             1 4 3 5\n1 2 3 3\n1.0D+00 2.0D+00 3.0D+00 4.0D+00\n"
        );
        let err =
            read_harwell_boeing(&content).expect_err("non-monotone pointers must be rejected");
        assert!(matches!(err, IoError::InvalidFormat(ref message) if message.contains("col_ptr")));
    }

    #[test]
    fn read_harwell_boeing_handles_lowercase_d_exponent() {
        let title = format!("{:<72}", "Lowercase D");
        let content = format!(
            "{title}KEY00004\n\
             4 1 1 1 0\n\
             RUA            2            2            2            0\n\
             (4I20)\n(4I20)\n(4D20.13)\n\
             1 2 3\n1 2\n1.5d+00 -2.5d-01\n"
        );
        let mat = read_harwell_boeing(&content).expect("lowercase d parse");
        assert_eq!(mat.values, vec![1.5, -0.25]);
    }

    #[test]
    fn hb_write_read_roundtrip_preserves_csc_and_value_bits() {
        let matrix = HbMatrix {
            title: "round-trip".to_string(),
            key: "BITS".to_string(),
            matrix_type: HbType::RealUnsymmetricAssembled,
            rows: 4,
            cols: 3,
            nnz: 5,
            col_ptr: vec![0, 2, 3, 5],
            row_idx: vec![0, 3, 1, 0, 2],
            values: vec![
                -0.0,
                std::f64::consts::PI,
                f64::MIN_POSITIVE,
                -123_456.75,
                f64::MAX,
            ],
        };
        let encoded = hb_write(&matrix).expect("HB encode");
        assert_eq!(
            encoded
                .lines()
                .nth(3)
                .expect("standard format card")
                .bytes()
                .filter(|&byte| byte == b'(')
                .count(),
            3
        );
        let decoded = hb_read(&encoded).expect("HB decode");
        assert_eq!(decoded.title, matrix.title);
        assert_eq!(decoded.key, matrix.key);
        assert_eq!(decoded.matrix_type, matrix.matrix_type);
        assert_eq!(decoded.rows, matrix.rows);
        assert_eq!(decoded.cols, matrix.cols);
        assert_eq!(decoded.nnz, matrix.nnz);
        assert_eq!(decoded.col_ptr, matrix.col_ptr);
        assert_eq!(decoded.row_idx, matrix.row_idx);
        assert!(
            decoded
                .values
                .iter()
                .zip(&matrix.values)
                .all(|(left, right)| left.to_bits() == right.to_bits())
        );
    }

    #[test]
    fn hb_write_rejects_invalid_csc_structure() {
        let matrix = HbMatrix {
            title: "bad pointers".to_string(),
            key: "BAD".to_string(),
            matrix_type: HbType::RealUnsymmetricAssembled,
            rows: 2,
            cols: 2,
            nnz: 2,
            col_ptr: vec![0, 2, 1],
            row_idx: vec![0, 1],
            values: vec![1.0, 2.0],
        };
        let error = hb_write(&matrix).expect_err("non-monotone pointers must fail");
        assert!(matches!(error, IoError::InvalidFormat(_)));
    }

    #[test]
    fn read_arff_minimal_dense_numeric_and_nominal() {
        let arff = "@relation iris\n\
                    @attribute sepal_length numeric\n\
                    @attribute class {Iris-setosa, Iris-versicolor, Iris-virginica}\n\
                    @data\n\
                    5.1, Iris-setosa\n\
                    7.0, Iris-versicolor\n";
        let parsed = read_arff(arff).expect("ARFF parse");
        assert_eq!(parsed.relation, "iris");
        assert_eq!(parsed.attributes.len(), 2);
        assert_eq!(parsed.rows.len(), 2);
        assert!(matches!(&parsed.rows[0][0], ArffValue::Numeric(v) if (v - 5.1).abs() < 1e-12));
        assert_eq!(
            parsed.rows[0][1],
            ArffValue::Nominal("Iris-setosa".to_string())
        );
    }

    #[test]
    fn read_arff_metamorphic_attribute_count_invariant() {
        let arff = "@relation r\n\
                    @attribute a numeric\n\
                    @attribute b numeric\n\
                    @attribute c {x, y}\n\
                    @data\n\
                    1, 2, x\n\
                    3, 4, y\n";
        let parsed = read_arff(arff).unwrap();
        for (idx, row) in parsed.rows.iter().enumerate() {
            assert_eq!(row.len(), parsed.attributes.len(), "row {idx}");
        }
    }

    #[test]
    fn read_arff_metamorphic_nominal_domain_consistency() {
        // Every nominal cell in a row must match its column's declared domain.
        let arff = "@relation r\n\
                    @attribute c {a, b, c}\n\
                    @data\n\
                    a\n\
                    b\n\
                    c\n";
        let parsed = read_arff(arff).unwrap();
        assert!(matches!(
            &parsed.attributes[0],
            ArffAttribute::Nominal { .. }
        ));
        for row in &parsed.rows {
            assert!(matches!(
                (&parsed.attributes[0], &row[0]),
                (ArffAttribute::Nominal { domain, .. }, ArffValue::Nominal(v)) if domain.contains(v)
            ));
        }
    }

    #[test]
    fn read_arff_sparse_row_expansion() {
        let arff = "@relation r\n\
                    @attribute a numeric\n\
                    @attribute b numeric\n\
                    @attribute c {x, y}\n\
                    @data\n\
                    {0 5.5, 2 y}\n";
        let parsed = read_arff(arff).unwrap();
        assert_eq!(parsed.rows.len(), 1);
        let row = &parsed.rows[0];
        // Per ARFF sparse semantics: missing numeric cells default to 0.0.
        assert!(matches!(&row[0], ArffValue::Numeric(v) if (v - 5.5).abs() < 1e-12));
        assert_eq!(row[1], ArffValue::Numeric(0.0));
        assert_eq!(row[2], ArffValue::Nominal("y".to_string()));
    }

    #[test]
    fn read_arff_handles_comments_and_quotes() {
        let arff = "% top comment\n\
                    @relation 'fancy name'\n\
                    @attribute a numeric % inline numeric\n\
                    @attribute b string\n\
                    @data\n\
                    1.0, 'hello world'\n\
                    2.0, \"with comma, here\"\n\
                    ?, ?\n";
        let parsed = read_arff(arff).expect("ARFF parse with comments");
        assert_eq!(parsed.relation, "fancy name");
        assert_eq!(parsed.rows.len(), 3);
        assert_eq!(
            parsed.rows[1][1],
            ArffValue::String("with comma, here".to_string())
        );
        assert_eq!(parsed.rows[2][0], ArffValue::Missing);
        assert_eq!(parsed.rows[2][1], ArffValue::Missing);
    }

    #[test]
    fn read_arff_dense_date_attributes_match_scipy_units() {
        let arff = "@relation r\n\
                    @attribute day date \"yyyy-MM-dd\"\n\
                    @attribute instant date \"yyyy-MM-dd HH:mm:ss\"\n\
                    @data\n\
                    \"2026-05-03\", \"2026-05-03 14:05:09\"\n\
                    ?, ?\n";
        let parsed = read_arff(arff).expect("date attributes parse");
        assert_eq!(
            parsed.attributes[0],
            ArffAttribute::Date {
                name: "day".to_string(),
                format: "yyyy-MM-dd".to_string(),
                unit: ArffDateUnit::Day,
            }
        );
        assert_eq!(
            parsed.rows[0][0],
            ArffValue::Date(ArffDateTime {
                raw: "2026-05-03".to_string(),
                normalized: "2026-05-03".to_string(),
                unit: ArffDateUnit::Day,
            })
        );
        assert_eq!(
            parsed.rows[0][1],
            ArffValue::Date(ArffDateTime {
                raw: "2026-05-03 14:05:09".to_string(),
                normalized: "2026-05-03T14:05:09".to_string(),
                unit: ArffDateUnit::Second,
            })
        );
        assert_eq!(parsed.rows[1][0], ArffValue::Missing);
        assert_eq!(parsed.rows[1][1], ArffValue::Missing);
    }

    #[test]
    fn read_arff_sparse_date_rows_keep_missing_dates() {
        let arff = "@relation r\n\
                    @attribute score numeric\n\
                    @attribute when date \"yyyy-MM-dd\"\n\
                    @data\n\
                    {1 \"2026-05-03\"}\n\
                    {0 2.5}\n";
        let parsed = read_arff(arff).expect("sparse date row parse");
        assert_eq!(parsed.rows[0][0], ArffValue::Numeric(0.0));
        assert_eq!(
            parsed.rows[0][1],
            ArffValue::Date(ArffDateTime {
                raw: "2026-05-03".to_string(),
                normalized: "2026-05-03".to_string(),
                unit: ArffDateUnit::Day,
            })
        );
        assert_eq!(parsed.rows[1][0], ArffValue::Numeric(2.5));
        assert_eq!(parsed.rows[1][1], ArffValue::Missing);
    }

    #[test]
    fn read_arff_rejects_date_attribute_with_timezone() {
        let arff = "@relation r\n\
                    @attribute when date \"yyyy-MM-dd Z\"\n\
                    @data\n\
                    \"2026-05-03 +0000\"\n";
        let err = read_arff(arff).expect_err("timezone date should be unsupported");
        assert!(matches!(err, IoError::UnsupportedFeature(_)));
    }

    #[test]
    fn read_arff_rejects_relational_attribute_as_unsupported() {
        let arff = "@relation r\n\
                    @attribute nested relational\n\
                    @attribute child numeric\n\
                    @end nested\n\
                    @data\n\
                    \"1\"\n";
        let err = read_arff(arff).expect_err("relational should be unsupported");
        assert!(matches!(err, IoError::UnsupportedFeature(_)));
    }

    #[test]
    fn read_arff_rejects_value_outside_nominal_domain() {
        let arff = "@relation r\n\
                    @attribute c {a, b}\n\
                    @data\n\
                    z\n";
        let err = read_arff(arff).expect_err("z not in {a,b}");
        assert!(matches!(err, IoError::InvalidFormat(_)));
    }

    #[test]
    fn fortran_roundtrip_two_little_endian_records() {
        let r1 = b"hello".to_vec();
        let r2 = b"world!".to_vec();
        let mut bytes = write_fortran_record(&r1, FortranEndian::Little);
        bytes.extend(write_fortran_record(&r2, FortranEndian::Little));
        let parsed = read_fortran_unformatted(&bytes, FortranEndian::Little).expect("two records");
        assert_eq!(parsed, vec![r1, r2]);
    }

    #[test]
    fn fortran_roundtrip_big_endian() {
        let payload = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0xFF];
        let bytes = write_fortran_record(&payload, FortranEndian::Big);
        let parsed = read_fortran_unformatted(&bytes, FortranEndian::Big).expect("BE record");
        assert_eq!(parsed, vec![payload.clone()]);
        // Reading the same bytes with the wrong endian must fail at the
        // length-mismatch check (or report a header far larger than the
        // input).
        let err = read_fortran_unformatted(&bytes, FortranEndian::Little)
            .expect_err("wrong endian must fail");
        assert!(matches!(err, IoError::InvalidFormat(_)));
    }

    #[test]
    fn fortran_metamorphic_record_count_invariant() {
        // For any sequence of payloads, the number of decoded records equals
        // the number we framed.
        let payloads: Vec<Vec<u8>> = (1..=5).map(|i| vec![i as u8; i * 3]).collect();
        let mut bytes = Vec::new();
        for p in &payloads {
            bytes.extend(write_fortran_record(p, FortranEndian::Little));
        }
        let parsed = read_fortran_unformatted(&bytes, FortranEndian::Little).unwrap();
        assert_eq!(parsed.len(), payloads.len());
        for (orig, got) in payloads.iter().zip(parsed.iter()) {
            assert_eq!(orig, got);
        }
    }

    #[test]
    fn fortran_rejects_truncated_payload() {
        let mut bytes = write_fortran_record(b"abcdef", FortranEndian::Little);
        bytes.truncate(bytes.len() - 4); // drop the trailer + 1
        let err = read_fortran_unformatted(&bytes, FortranEndian::Little)
            .expect_err("truncated must fail");
        assert!(matches!(err, IoError::InvalidFormat(_)));
    }

    #[test]
    fn fortran_rejects_length_mismatch() {
        // Hand-craft a record whose trailer disagrees with the header.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&5_i32.to_le_bytes());
        bytes.extend_from_slice(b"hello");
        bytes.extend_from_slice(&7_i32.to_le_bytes()); // wrong trailer
        let err = read_fortran_unformatted(&bytes, FortranEndian::Little)
            .expect_err("trailer mismatch must fail");
        assert!(matches!(err, IoError::InvalidFormat(ref msg) if msg.contains("trailer")));
    }

    #[test]
    fn fortran_empty_input_produces_no_records() {
        let parsed = read_fortran_unformatted(&[], FortranEndian::Little).unwrap();
        assert!(parsed.is_empty());
    }

    #[test]
    fn fortran_zero_length_record_is_supported() {
        let bytes = write_fortran_record(&[], FortranEndian::Little);
        let parsed = read_fortran_unformatted(&bytes, FortranEndian::Little).unwrap();
        assert_eq!(parsed.len(), 1);
        assert!(parsed[0].is_empty());
    }

    #[test]
    fn fortran_file_typed_roundtrip_is_bit_identical() {
        let ints = [i32::MIN, -7, 0, 42, i32::MAX];
        let reals = [
            -0.0,
            std::f64::consts::PI,
            f64::MIN_POSITIVE,
            f64::INFINITY,
            f64::from_bits(0x7ff8_0000_0000_1234),
        ];
        let mut file = FortranFile::new(FortranEndian::Big);
        file.write_ints(&ints).expect("integer record");
        file.write_reals(&reals).expect("real record");
        assert!(file.is_eof());
        file.rewind();

        assert_eq!(file.read_ints().expect("integer decode"), ints);
        let decoded = file.read_reals().expect("real decode");
        assert!(
            decoded
                .iter()
                .zip(reals)
                .all(|(left, right)| left.to_bits() == right.to_bits())
        );
        assert!(file.is_eof());
        assert!(matches!(
            file.read_record(),
            Err(FortranFileError::EndOfFile(FortranEOFError { .. }))
        ));
    }

    #[test]
    fn fortran_file_distinguishes_formatting_error_from_clean_eof() {
        let mut bytes = write_fortran_record(b"abc", FortranEndian::Little);
        let trailer_start = bytes.len() - 4;
        bytes[trailer_start..].copy_from_slice(&9_i32.to_le_bytes());
        let mut file = FortranFile::from_bytes(bytes, FortranEndian::Little);
        assert!(matches!(
            file.read_record(),
            Err(FortranFileError::Formatting(FortranFormattingError {
                offset: 0,
                ..
            }))
        ));
        assert_eq!(file.position(), 0);
    }

    #[test]
    fn fortran_file_rejects_typed_record_with_partial_element() {
        let bytes = write_fortran_record(&[1, 2, 3, 4, 5], FortranEndian::Little);
        let mut file = FortranFile::from_bytes(bytes, FortranEndian::Little);
        let error = file
            .read_ints()
            .expect_err("five bytes cannot encode whole i32 values");
        assert!(matches!(
            error,
            FortranFileError::Formatting(FortranFormattingError { offset: 0, .. })
        ));
    }

    #[test]
    fn mmread_matches_scipy_reference_values() {
        // scipy.io.mmread on a Matrix Market coordinate format file
        // import scipy.io as spio; import io
        // mm = "%%MatrixMarket matrix coordinate real general\n3 3 3\n1 1 1.0\n2 2 2.0\n3 3 3.0\n"
        // spio.mmread(io.StringIO(mm)).toarray()
        // -> array([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        let mm_content =
            "%%MatrixMarket matrix coordinate real general\n3 3 3\n1 1 1.0\n2 2 2.0\n3 3 3.0\n";
        let result = mmread(mm_content).expect("mmread should succeed");
        assert_eq!(result.rows, 3);
        assert_eq!(result.cols, 3);
        // Expanded to dense row-major: data[i*cols+j]
        // (0,0)=1.0, (1,1)=2.0, (2,2)=3.0, all others 0.0
        assert!((result.data[0] - 1.0).abs() < 1e-10, "data[0,0] = 1.0");
        assert!((result.data[4] - 2.0).abs() < 1e-10, "data[1,1] = 2.0");
        assert!((result.data[8] - 3.0).abs() < 1e-10, "data[2,2] = 3.0");
        assert!((result.data[1] - 0.0).abs() < 1e-10, "off-diagonal is 0");
    }

    #[test]
    fn mmwrite_matches_scipy_output_format() {
        // scipy.io.mmwrite produces Matrix Market array format for dense matrices
        // import scipy.io as spio; import io
        // spio.mmwrite(io.StringIO(), [[1, 2], [3, 4]])
        // Result format: header + "1\n2\n3\n4\n" (column-major)
        let output = mmwrite(2, 2, &[1.0, 2.0, 3.0, 4.0]).expect("mmwrite should succeed");
        assert!(output.contains("%%MatrixMarket matrix array real general"));
        assert!(output.contains("2 2"));
        // Verify roundtrip
        let parsed = mmread(&output).expect("roundtrip should work");
        assert_eq!(parsed.rows, 2);
        assert_eq!(parsed.cols, 2);
        let expected = [1.0, 2.0, 3.0, 4.0];
        for (i, &want) in expected.iter().enumerate() {
            let got = parsed.data[i];
            assert!(
                (got - want).abs() < 1e-10,
                "data[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn loadtxt_matches_scipy_reference_values() {
        // np.loadtxt(StringIO("1 2\n3 4")) -> [[1, 2], [3, 4]]
        let content = "1 2\n3 4\n";
        let (rows, cols, data) = loadtxt(content).expect("loadtxt should succeed");
        assert_eq!(rows, 2);
        assert_eq!(cols, 2);
        let expected = [1.0, 2.0, 3.0, 4.0];
        for (i, (&got, &want)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "data[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn savetxt_matches_scipy_output_format() {
        // np.savetxt produces space-delimited text by default
        let output = savetxt(2, 2, &[1.0, 2.0, 3.0, 4.0], " ").expect("savetxt should succeed");
        // Verify roundtrip
        let (rows, cols, data) = loadtxt(&output).expect("roundtrip should work");
        assert_eq!(rows, 2);
        assert_eq!(cols, 2);
        let expected = [1.0, 2.0, 3.0, 4.0];
        for (i, (&got, &want)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "data[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn savetxt_parallel_is_byte_identical_to_serial_above_gate() {
        use std::sync::atomic::Ordering;
        // Above the rows·cols ≥ 2^16 fan-out gate the parallel per-row formatter must be
        // BYTE-IDENTICAL to the serial loop (each row is formatted independently and the
        // parts are joined in row order).
        let (rows, cols) = (5000usize, 20usize); // 100000 > 65536 gate
        let mut state = 0xABCDu64;
        let data: Vec<f64> = (0..rows * cols)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 11) as f64 / (1u64 << 53) as f64 * 200.0 - 100.0
            })
            .collect();
        for delim in [" ", ",", "\t"] {
            SAVETXT_FORCE_SERIAL.store(true, Ordering::Relaxed);
            let serial = savetxt(rows, cols, &data, delim).expect("serial");
            SAVETXT_FORCE_SERIAL.store(false, Ordering::Relaxed);
            let parallel = savetxt(rows, cols, &data, delim).expect("parallel");
            assert_eq!(
                serial, parallel,
                "delim {delim:?}: parallel savetxt must equal serial"
            );
        }
    }

    #[test]
    fn mminfo_matches_scipy_reference_values() {
        // scipy.io.mminfo returns (rows, cols, nnz, format, field, symmetry)
        let content = "%%MatrixMarket matrix coordinate real general\n2 3 4\n1 1 1.0\n1 2 2.0\n2 2 3.0\n2 3 4.0\n";
        let info = mminfo(content).expect("mminfo");
        assert_eq!(info.rows, 2, "mminfo rows");
        assert_eq!(info.cols, 3, "mminfo cols");
        assert_eq!(info.nnz, 4, "mminfo nnz");
    }

    #[test]
    fn wav_write_read_roundtrip_matches_scipy_semantics() {
        // scipy.io.wavfile roundtrip preserves data
        let original = vec![0.5, -0.5, 0.25, -0.25];
        let bytes = wav_write(44100, 1, &original).expect("wav_write");
        let result = wav_read(&bytes).expect("wav_read");
        assert_eq!(result.sample_rate, 44100, "sample rate preserved");
        assert_eq!(result.data.len(), original.len(), "data length preserved");
    }

    fn idl_save_header() -> Vec<u8> {
        b"SR\x00\x04".to_vec()
    }

    fn idl_push_variable_record<F>(
        out: &mut Vec<u8>,
        name: &str,
        type_code: i32,
        varflags: i32,
        write_payload: F,
    ) where
        F: FnOnce(&mut Vec<u8>),
    {
        let record_start = idl_begin_record(out, 2);
        idl_push_string(out, name.as_bytes());
        idl_push_i32(out, type_code);
        idl_push_i32(out, varflags);
        idl_push_i32(out, 7);
        write_payload(out);
        idl_finish_record(out, record_start);
    }

    fn idl_push_array_variable_record<F>(
        out: &mut Vec<u8>,
        name: &str,
        type_code: i32,
        nbytes: usize,
        nelements: usize,
        dims: &[usize],
        write_payload: F,
    ) where
        F: FnOnce(&mut Vec<u8>),
    {
        let record_start = idl_begin_record(out, 2);
        idl_push_string(out, name.as_bytes());
        idl_push_i32(out, type_code);
        idl_push_i32(out, 4);
        idl_push_array_desc(out, nbytes, nelements, dims);
        idl_push_i32(out, 7);
        write_payload(out);
        idl_finish_record(out, record_start);
    }

    fn idl_push_end_record(out: &mut Vec<u8>) {
        idl_push_i32(out, 6);
        out.extend_from_slice(&0_u32.to_be_bytes());
        out.extend_from_slice(&0_u32.to_be_bytes());
        out.extend_from_slice(&0_u32.to_be_bytes());
    }

    fn idl_begin_record(out: &mut Vec<u8>, rectype: i32) -> usize {
        let start = out.len();
        idl_push_i32(out, rectype);
        out.extend_from_slice(&0_u32.to_be_bytes());
        out.extend_from_slice(&0_u32.to_be_bytes());
        out.extend_from_slice(&0_u32.to_be_bytes());
        start
    }

    fn idl_finish_record(out: &mut [u8], record_start: usize) {
        let next = u64::try_from(out.len()).expect("test fixture length fits u64");
        let low = u32::try_from(next & 0xffff_ffff).expect("low word fits u32");
        let high = u32::try_from(next >> 32).expect("high word fits u32");
        out[record_start + 4..record_start + 8].copy_from_slice(&low.to_be_bytes());
        out[record_start + 8..record_start + 12].copy_from_slice(&high.to_be_bytes());
    }

    fn idl_push_array_desc(out: &mut Vec<u8>, nbytes: usize, nelements: usize, dims: &[usize]) {
        idl_push_i32(out, 8);
        idl_push_i32(out, 0);
        idl_push_usize_as_i32(out, nbytes);
        idl_push_usize_as_i32(out, nelements);
        idl_push_usize_as_i32(out, dims.len());
        idl_push_i32(out, 0);
        idl_push_i32(out, 0);
        idl_push_i32(out, 8);
        for idx in 0..8 {
            idl_push_usize_as_i32(out, dims.get(idx).copied().unwrap_or(0));
        }
    }

    fn idl_push_string(out: &mut Vec<u8>, bytes: &[u8]) {
        idl_push_usize_as_i32(out, bytes.len());
        out.extend_from_slice(bytes);
        idl_pad_32(out);
    }

    fn idl_push_string_data(out: &mut Vec<u8>, bytes: &[u8]) {
        idl_push_usize_as_i32(out, bytes.len());
        if !bytes.is_empty() {
            idl_push_usize_as_i32(out, bytes.len());
            out.extend_from_slice(bytes);
            idl_pad_32(out);
        }
    }

    fn idl_pad_32(out: &mut Vec<u8>) {
        while !out.len().is_multiple_of(4) {
            out.push(0);
        }
    }

    fn idl_push_usize_as_i32(out: &mut Vec<u8>, value: usize) {
        let value = i32::try_from(value).expect("IDL test fixture field fits i32");
        idl_push_i32(out, value);
    }

    fn idl_push_i32(out: &mut Vec<u8>, value: i32) {
        out.extend_from_slice(&value.to_be_bytes());
    }
}

/// frankenscipy-drqu7 — fsci-io's accuracy-contract ratchet.
///
/// This crate had FIVE toggles and ZERO contracts, the worst ratio in the fleet.
/// All five are now documented and all five already had A/B drivers, so the
/// claims here are checked rather than merely asserted -- which is the whole
/// distinction this bead turned out to be about.
///
/// The budget is 0 and may only ever stay 0: every toggle in this crate states
/// what its two arms preserve, so a new one without a contract fails immediately
/// and there is no headroom to absorb it quietly.
#[cfg(test)]
mod accuracy_contract_ratchet {
    /// Every toggle in this crate is contracted, so the budget is zero.
    const UNCONTRACTED_TOGGLE_BUDGET: usize = 0;

    /// The drqu7 predicate, matching the fsci-stats and fsci-linalg ratchets:
    /// the keyword list PLUS a bare numeric tolerance such as `1e-15`, which
    /// states a bound without using any keyword. Omitting that clause is what
    /// made the census script disagree with those ratchets by exactly one toggle
    /// in each crate.
    fn accuracy_contract_is_stated(doc: &str) -> bool {
        let d = doc.to_ascii_lowercase();
        [
            "byte-identical",
            "byte identical",
            "bit-identical",
            "bit identical",
            "identical output",
            "identical result",
            "ulp",
            "toleran",
            "reassoc",
            "agrees to",
            "not bit",
        ]
        .iter()
        .any(|k| d.contains(k))
            || d.split_whitespace().any(|w| {
                let w =
                    w.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '.' && c != '-');
                w.contains("e-") && w.chars().next().is_some_and(|c| c.is_ascii_digit())
            })
    }

    #[test]
    fn no_new_perf_toggle_may_ship_without_an_accuracy_contract() {
        let source = include_str!("lib.rs");
        let lines: Vec<&str> = source.lines().collect();
        let mut uncontracted: Vec<&str> = Vec::new();
        let mut total = 0usize;

        for (i, line) in lines.iter().enumerate() {
            let Some(rest) = line.trim_start().strip_prefix("pub static ") else {
                continue;
            };
            let name = rest.split(':').next().unwrap_or("").trim();
            if name.is_empty()
                || !name
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
            {
                continue;
            }
            total += 1;
            let mut doc = String::new();
            let mut j = i;
            while j > 0 {
                j -= 1;
                let t = lines[j].trim_start();
                if t.starts_with("//") {
                    doc.push(' ');
                    doc.push_str(t);
                } else if t.starts_with("#[") {
                    // Skip attributes. `#[doc(hidden)]` sits between the doc block
                    // and the declaration on every toggle in this crate, and a walk
                    // that stops here reads an EMPTY doc and calls a fully
                    // contracted lever uncontracted. That bug made the fsci-stats
                    // ratchet report 125 uncontracted where the truth was 12.
                    continue;
                } else {
                    break;
                }
            }
            if !accuracy_contract_is_stated(&doc) {
                uncontracted.push(name);
            }
        }

        // MUST-HIT on the predicate itself, so a scan that silently stopped
        // matching cannot report a clean zero.
        assert!(
            accuracy_contract_is_stated("CONTRACT: BYTE-IDENTICAL either way"),
            "the contract predicate no longer recognises a contract; its verdict on \
             this file is worthless"
        );
        // MUST-MISS: a bare description is not a contract.
        assert!(
            !accuracy_contract_is_stated("Runtime switch to force the serial path."),
            "the contract predicate accepts a bare description; it would pass every \
             undocumented toggle"
        );
        // MUST-MISS on vacuity: if the scan stops finding statics it would report
        // zero uncontracted because it saw nothing.
        assert!(
            total >= 5,
            "only {total} pub statics found in this file; this crate has five \
             toggles, so the scan is broken rather than the code"
        );

        eprintln!(
            "drqu7 fsci-io: {} perf statics, {} uncontracted, budget {}",
            total,
            uncontracted.len(),
            UNCONTRACTED_TOGGLE_BUDGET
        );
        assert!(
            uncontracted.len() <= UNCONTRACTED_TOGGLE_BUDGET,
            "{} perf toggles have no accuracy contract, budget is {}. Document what \
             the two arms preserve -- bit identity, a tolerance, or a deliberate \
             difference. Offenders: {:?}",
            uncontracted.len(),
            UNCONTRACTED_TOGGLE_BUDGET,
            uncontracted
        );
    }

    #[test]
    fn mmread_mode_audit_and_dimension_cap() {
        let header = format!(
            "%%MatrixMarket matrix coordinate real general\n{} 10 1\n1 1 1.0\n",
            crate::HARDENED_MAX_DIM + 1
        );
        let ledger = crate::sync_audit_ledger();

        // Hardened mode rejects dimension > HARDENED_MAX_DIM
        let err = crate::mmread_with_mode(&header, crate::RuntimeMode::Hardened, Some(&ledger));
        assert!(err.is_err());

        // Verify audit event
        let guard = ledger.lock().unwrap();
        assert_eq!(guard.entries().len(), 1);
        assert!(matches!(
            guard.entries()[0].action,
            crate::AuditAction::FailClosed { .. }
        ));
    }

    /// frankenscipy-3cu8u.1: the mmread audit fingerprint covers the whole file. It used to be
    /// the row count alone, so every pair of over-limit files below shared one fingerprint.
    #[test]
    fn audit_fingerprints_cover_every_input() {
        let fingerprint_of = |content: &str| {
            let ledger = crate::sync_audit_ledger();
            let result =
                crate::mmread_with_mode(content, crate::RuntimeMode::Hardened, Some(&ledger));
            assert!(result.is_err());
            let guard = ledger.lock().unwrap();
            assert_eq!(guard.entries().len(), 1);
            assert!(matches!(
                guard.entries()[0].action,
                crate::AuditAction::FailClosed { .. }
            ));
            guard.entries()[0].input_fingerprint.clone()
        };
        let rows = crate::HARDENED_MAX_DIM + 1;
        let file = |cols: usize, value: &str| {
            format!("%%MatrixMarket matrix coordinate real general\n{rows} {cols} 1\n1 1 {value}\n")
        };

        let base = fingerprint_of(&file(10, "1.0"));
        assert!(base.starts_with("blake3:"));
        assert_eq!(base, fingerprint_of(&file(10, "1.0")));
        // Same row count, different entry value (the tail of the file).
        assert_ne!(base, fingerprint_of(&file(10, "2.0")));
        // Same row count, different column count.
        assert_ne!(base, fingerprint_of(&file(11, "1.0")));
    }
}
