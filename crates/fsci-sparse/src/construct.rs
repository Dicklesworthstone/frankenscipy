use std::collections::HashSet;

use crate::formats::{
    BsrMatrix, CanonicalMeta, CooArray, CooMatrix, CscMatrix, CsrMatrix, DiaMatrix, DokMatrix,
    LilMatrix, Shape2D, SparseArray2D, SparseError, SparseFormat, SparseResult,
};
use crate::ops::{FormatConvertible, add_csr};
use crate::{SparseArray, SparseObject};

/// Add a length-one axis to a sparse array and return N-dimensional COO form.
///
/// Valid axes are the closed interval `[-N-1, N]`, matching
/// `scipy.sparse.expand_dims`. Negative axes are interpreted against the
/// expanded dimensionality.
pub fn expand_dims<A: SparseArray + ?Sized>(array: &A, axis: isize) -> SparseResult<CooArray> {
    let coo = array.to_coo_array()?;
    let ndim = coo.ndim();
    let ndim_signed = isize::try_from(ndim).map_err(|_| SparseError::IndexOverflow {
        message: "sparse array ndim does not fit in isize".to_string(),
    })?;
    let expanded_ndim = ndim_signed
        .checked_add(1)
        .ok_or_else(|| SparseError::IndexOverflow {
            message: "expanded sparse array ndim overflows isize".to_string(),
        })?;
    let normalized = if axis < 0 {
        axis.checked_add(expanded_ndim)
    } else {
        Some(axis)
    };
    let Some(normalized) = normalized.filter(|&value| value >= 0 && value <= ndim_signed) else {
        return Err(SparseError::InvalidArgument {
            message: format!("invalid axis {axis} for ndim {ndim}; expected [-N-1, N]"),
        });
    };
    let insertion = usize::try_from(normalized).map_err(|_| SparseError::IndexOverflow {
        message: "normalized sparse axis does not fit usize".to_string(),
    })?;

    let mut shape = coo.shape().to_vec();
    shape.insert(insertion, 1);
    let mut coords = coo.coords().to_vec();
    coords.insert(insertion, vec![0usize; coo.nnz()]);
    Ok(CooArray::from_validated_parts(
        shape,
        coo.data().to_vec(),
        coords,
        coo.has_canonical_format(),
    ))
}

/// Permute the axes of a sparse array and return N-dimensional COO form.
///
/// `None` reverses the axes. Explicit axes may be negative, must have exactly
/// `N` entries, and must form a permutation. Rust returns an owned value, so
/// SciPy's `copy=False` identity optimization has no aliasing effect here; the
/// argument is retained to keep the public contract explicit.
pub fn permute_dims<A: SparseArray + ?Sized>(
    array: &A,
    axes: Option<&[isize]>,
    _copy: bool,
) -> SparseResult<CooArray> {
    let coo = array.to_coo_array()?;
    let ndim = coo.ndim();
    let normalized_axes: Vec<usize> = match axes {
        None => (0..ndim).rev().collect(),
        Some(axes) => {
            if axes.len() != ndim {
                return Err(SparseError::InvalidArgument {
                    message: format!(
                        "incorrect number of axes: got {}, expected {ndim}",
                        axes.len()
                    ),
                });
            }
            let mut normalized = Vec::with_capacity(ndim);
            let mut seen = vec![false; ndim];
            for &axis in axes {
                let axis = normalize_sparse_axis(axis, ndim)?;
                if seen[axis] {
                    return Err(SparseError::InvalidArgument {
                        message: format!("duplicate value {axis} in axes"),
                    });
                }
                seen[axis] = true;
                normalized.push(axis);
            }
            normalized
        }
    };

    let is_identity = normalized_axes.iter().copied().eq(0..ndim);
    let shape = normalized_axes
        .iter()
        .map(|&axis| coo.shape()[axis])
        .collect();
    let coords = normalized_axes
        .iter()
        .map(|&axis| coo.coords()[axis].clone())
        .collect();
    Ok(CooArray::from_validated_parts(
        shape,
        coo.data().to_vec(),
        coords,
        is_identity && coo.has_canonical_format(),
    ))
}

/// Interchange two axes of a sparse array and return N-dimensional COO form.
pub fn swapaxes<A: SparseArray + ?Sized>(
    array: &A,
    axis1: isize,
    axis2: isize,
) -> SparseResult<CooArray> {
    let ndim = array.shape_nd().len();
    let axis1 = normalize_sparse_axis(axis1, ndim)?;
    let axis2 = normalize_sparse_axis(axis2, ndim)?;
    let mut axes: Vec<isize> = (0..ndim)
        .map(|axis| {
            isize::try_from(axis).map_err(|_| SparseError::IndexOverflow {
                message: "sparse array axis does not fit in isize".to_string(),
            })
        })
        .collect::<SparseResult<_>>()?;
    axes.swap(axis1, axis2);
    permute_dims(array, Some(&axes), true)
}

fn normalize_sparse_axis(axis: isize, ndim: usize) -> SparseResult<usize> {
    let ndim_signed = isize::try_from(ndim).map_err(|_| SparseError::IndexOverflow {
        message: "sparse array ndim does not fit in isize".to_string(),
    })?;
    let normalized = if axis < 0 {
        axis.checked_add(ndim_signed)
    } else {
        Some(axis)
    };
    let Some(normalized) = normalized.filter(|&value| value >= 0 && value < ndim_signed) else {
        return Err(SparseError::InvalidArgument {
            message: format!("axis {axis} is out of range for ndim {ndim}"),
        });
    };
    usize::try_from(normalized).map_err(|_| SparseError::IndexOverflow {
        message: "normalized sparse axis does not fit usize".to_string(),
    })
}

pub fn eye(size: usize) -> SparseResult<CsrMatrix> {
    let shape = Shape2D::new(size, size);
    let data = vec![1.0; size];
    let indices: Vec<usize> = (0..size).collect();
    let indptr: Vec<usize> = (0..=size).collect();
    let mut result = CsrMatrix::from_components_unchecked(shape, data, indices, indptr);
    result.canonical = CanonicalMeta {
        sorted_indices: true,
        deduplicated: true,
    };
    Ok(result)
}

/// Identity-like sparse array with ones on the `k`-th diagonal, matching
/// `scipy.sparse.eye_array(m, n, k)`.
///
/// Returns an `m × n` matrix with `1.0` wherever `column = row + k` (in range);
/// the array-API spelling of [`eye_rectangular`].
pub fn eye_array(m: usize, n: usize, k: isize) -> SparseResult<SparseArray2D<CsrMatrix>> {
    eye_rectangular(m, n, k).map(SparseArray2D::new)
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
    // The k-th diagonal places exactly one entry per row in `[row_start, row_start+length)`, at
    // column `col_start + i` for the i-th diagonal row (col_start = k for k>=0, else 0). That is
    // already the row-major, per-row-sorted, duplicate-free CSR layout, so build it directly:
    // `indptr[row]` = entries in rows before `row` = clamp(row - row_start, 0, length). This skips
    // the intermediate COO allocation, its `from_triplets` validation, and `to_csr`'s sort +
    // canonicalization — byte-identical to the old COO→to_csr path (same indices/indptr/values and
    // {sorted, deduplicated} metadata).
    let col_start = if k >= 0 { k as usize } else { 0 };
    let data = vec![1.0; length];
    let indices: Vec<usize> = (col_start..col_start + length).collect();
    let indptr: Vec<usize> = (0..=rows)
        .map(|row| row.saturating_sub(row_start).min(length))
        .collect();
    let mut result = CsrMatrix::from_components_unchecked(shape, data, indices, indptr);
    result.canonical = CanonicalMeta {
        sorted_indices: true,
        deduplicated: true,
    };
    Ok(result)
}

/// Emit rows `[base..end)` of a [`diags`] matrix into local `(counts, indices, data)` buffers by
/// walking the offset-sorted diagonals per row. `counts[k]` is the stored nnz of row `base+k`, and
/// entries are laid down in ascending-row, ascending-offset (hence ascending-column) order.
/// Factored so the serial path and each parallel worker run byte-identical emit code.
fn diags_row_block(
    diagonal_order: &[(isize, &[f64])],
    shape: Shape2D,
    base: usize,
    end: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut counts = Vec::with_capacity(end.saturating_sub(base));
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for row in base..end {
        let mut c = 0usize;
        for &(offset, diag) in diagonal_order {
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
            // Store only structural non-zeros. SciPy keeps every diagonal
            // element in the DIA intermediate but masks `data != 0` when it
            // converts to CSR, so `sparse.diags(..., format="csr")` prunes both
            // all-zero diagonals and interior zeros. [`diags`] returns CSR
            // directly and therefore matches the converted peer, not the DIA
            // one. `-0.0 == 0.0` and `NaN != 0.0` in Rust exactly as in numpy,
            // so those two edges agree without special-casing.
            if let Some((col, value)) = entry
                && value != 0.0
            {
                indices.push(col);
                data.push(value);
                c += 1;
            }
        }
        counts.push(c);
    }
    (counts, indices, data)
}

/// When `true`, [`diags`] emits its rows serially (the ORIG behaviour); default `false` fans the
/// independent per-row diagonal emit across contiguous row-blocks. Byte-identical.
#[doc(hidden)]
pub static SPARSE_DIAGS_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// When `true`, [`diags`] takes the ORIG validating path: bounds-check EVERY diagonal element
/// (O(nnz) pre-scan) then build the CSR through `from_components(.., true)` (another O(nnz) bounds
/// scan + `detect_canonical`). Default `false` checks only each diagonal's last (extremal) element
/// and, since the emitted rows are provably in-bounds and canonical, builds via the trusted
/// constructor — skipping both O(nnz) validation scans. Byte-identical result AND metadata. A/B gate.
#[doc(hidden)]
pub static DIAGS_VALIDATE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Construct a CSR matrix from diagonals, matching
/// `scipy.sparse.diags(diagonals, offsets, shape=shape, format="csr")`.
///
/// # Stored zeros
/// SciPy's `diags` first builds a DIA matrix, which stores every diagonal
/// element (`sp.diags([np.zeros(5)], [0], shape=(5, 5)).nnz == 5`), and prunes
/// on conversion — its dia→csr path masks `data != 0`, so the CSR peer has
/// `nnz == 0`, and interior zeros are dropped as well. This returns CSR
/// directly, so it matches the converted peer and stores no exact zeros. Use
/// [`spdiags`], which returns a [`DiaMatrix`], when the unpruned diagonal
/// layout is what you want. See [frankenscipy-mhxei].
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

    let validate = DIAGS_VALIDATE.load(std::sync::atomic::Ordering::Relaxed);
    for (diag, &offset) in diagonals.iter().zip(offsets.iter()) {
        if diag.is_empty() {
            continue;
        }
        let start_row = if offset < 0 { (-offset) as usize } else { 0 };
        let start_col = if offset > 0 { offset as usize } else { 0 };
        // `row` and `col` both increase by 1 per step `k`, so the LAST element (k = len-1)
        // carries the maximum row and column. If it fits the shape, every earlier element
        // does too; if any element would exceed the shape, the last one does (same axis).
        // Checking only that extremum yields the byte-identical valid/invalid verdict while
        // collapsing an O(nnz) bounds pre-scan to O(#diagonals) — meaningful because this
        // pass is serial while the row fill below fans across cores.
        let range = if validate {
            0..diag.len()
        } else {
            (diag.len() - 1)..diag.len()
        };
        for k in range {
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

    // Each output row is a pure function of `row` and the (shared, read-only) offset-sorted
    // diagonals — the rows are independent. The stored COUNT per row is data-dependent, so use
    // gather-then-concat: each worker emits a contiguous row-block into a local buffer, then the
    // blocks are concatenated in ascending row order and `indptr` is built from per-row counts.
    // Concatenating in row order reproduces the exact serial emission → BYTE-IDENTICAL. Gated on the
    // total entry count so small banded matrices stay serial.
    let n = shape.rows;
    let nthreads = if SPARSE_DIAGS_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
        || capacity < 65_536
        || n < 2
    {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(n)
    };

    let parts: Vec<(Vec<usize>, Vec<usize>, Vec<f64>)> = if nthreads <= 1 {
        vec![diags_row_block(&diagonal_order, shape, 0, n)]
    } else {
        let chunk = n.div_ceil(nthreads);
        let dord = &diagonal_order;
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..nthreads)
                .map(|t| {
                    let base = (t * chunk).min(n);
                    let end = ((t + 1) * chunk).min(n);
                    scope.spawn(move || diags_row_block(dord, shape, base, end))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    };

    let mut data = Vec::with_capacity(capacity);
    let mut indices = Vec::with_capacity(capacity);
    let mut indptr = vec![0usize; n + 1];
    let mut row_i = 0usize;
    for (counts, idx, dat) in &parts {
        for &c in counts {
            indptr[row_i + 1] = c;
            row_i += 1;
        }
        indices.extend_from_slice(idx);
        data.extend_from_slice(dat);
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }

    // Each output row emits its diagonals in offset-sorted order, and for a fixed row the column
    // of the entry from offset `o` is `row + o` — strictly ascending in `o` with unique offsets —
    // so every row is already sorted and deduplicated, and the bounds pre-check above proved the
    // indices in range. The trusted constructor therefore skips `from_components`' redundant O(nnz)
    // bounds scan + `detect_canonical`, yielding the identical matrix AND canonical metadata.
    if validate {
        CsrMatrix::from_components(shape, data, indices, indptr, true)
    } else {
        Ok(CsrMatrix::from_components_trusted_canonical(
            shape, data, indices, indptr,
        ))
    }
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

/// Construct a sparse array from diagonals, matching `scipy.sparse.diags_array`.
///
/// The array-API equivalent of [`diags`] (same result; scipy distinguishes only
/// the returned container type, which fsci unifies).
pub fn diags_array(
    diagonals: &[Vec<f64>],
    offsets: &[isize],
    shape: Option<Shape2D>,
) -> SparseResult<SparseArray2D<CsrMatrix>> {
    diags(diagonals, offsets, shape).map(SparseArray2D::new)
}

/// Random sparse array with SciPy's sampling contract, matching
/// `scipy.sparse.random_array(shape, density=..., format=..., rng=...)`.
///
/// # Why this is not a thin wrapper over [`random`]
///
/// [`random`] predates the SciPy-facing surface and has a DIFFERENT contract in two
/// ways that would be silently wrong under a SciPy name:
///
/// 1. **Value range.** `random` samples values in `[-1, 1)`; SciPy samples
///    `[0, 1)`. A caller porting NumPy/SciPy code and getting negative entries
///    would have no way to know.
/// 2. **Rounding of `nnz`.** SciPy computes `int(round(density * m * n))` with
///    NumPy's **round-half-to-even**; Rust's `f64::round` is half-away-from-zero.
///    They agree on `3.5 -> 4` and disagree on `2.5` (SciPy 2, Rust 3).
///    This uses [`f64::round_ties_even`] to match.
///
/// [`random`] is left exactly as it is — a large number of in-crate tests depend on
/// its stream — so the two coexist with distinct, documented contracts.
///
/// Determinism: the value stream is a function of `seed` alone and is stable across
/// platforms and toolchains. It does NOT reproduce NumPy's Generator stream; SciPy
/// does not specify particular values, only the structure (shape, `nnz`, dtype,
/// distinct coordinates), which this matches exactly.
pub fn random_array(
    shape: Shape2D,
    density: f64,
    format: Option<&str>,
    seed: u64,
) -> SparseResult<SparseArrayOutput> {
    scipy_random_matrix_output(shape, density, format, seed).map(SparseArrayOutput::from)
}

fn scipy_random_matrix_output(
    shape: Shape2D,
    density: f64,
    format: Option<&str>,
    seed: u64,
) -> SparseResult<HstackOutput> {
    let requested = parse_hstack_format(format)?;
    if !(0.0..=1.0).contains(&density) || !density.is_finite() {
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
    // SciPy: `size = int(round(density * prod(shape)))`, NumPy round-half-to-even.
    let nnz = if total == 0 {
        0
    } else {
        ((density * total as f64).round_ties_even()).clamp(0.0, total as f64) as usize
    };

    let mut rows = Vec::with_capacity(nnz);
    let mut cols = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    if nnz > 0 {
        // Sample DISTINCT flat positions, as SciPy does (it draws without
        // replacement), so the result has exactly `nnz` stored entries and needs no
        // duplicate summation.
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut seen: HashSet<usize> = HashSet::with_capacity(nnz);
        while seen.len() < nnz {
            state = xorshift64(state);
            let flat = (state as usize) % total;
            if !seen.insert(flat) {
                continue;
            }
            rows.push(flat / shape.cols);
            cols.push(flat % shape.cols);
            state = xorshift64(state);
            // Uniform [0, 1): 53 mantissa bits scaled by 2^-53, matching the
            // standard double-precision construction and SciPy's half-open range.
            data.push(((state >> 11) as f64) * (1.0f64 / 9_007_199_254_740_992.0));
        }
    }

    let coo = CooMatrix::from_triplets(shape, data, rows, cols, false)?;
    hstack_output_from_csr(coo.to_csr()?, requested)
}

/// Random sparse matrix, matching `scipy.sparse.rand(m, n, density, format, rng)`.
///
/// SciPy's `rand` is `random_array` with `(m, n)` spelled as two arguments and the
/// same sampling contract; see [`random_array`] for the value range and `nnz`
/// rounding rules.
pub fn rand(
    m: usize,
    n: usize,
    density: f64,
    format: Option<&str>,
    seed: u64,
) -> SparseResult<HstackOutput> {
    scipy_random_matrix_output(Shape2D::new(m, n), density, format, seed)
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
    if total == 0 || density == 0.0 {
        return CooMatrix::from_triplets(shape, Vec::new(), Vec::new(), Vec::new(), true);
    }

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
    if expected_nnz == 0 {
        return CooMatrix::from_triplets(shape, Vec::new(), Vec::new(), Vec::new(), true);
    }

    if expected_nnz <= total / 8 {
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

    // Fast path: when every block is already canonical (sorted + deduplicated), the
    // block-diagonal output is canonical by construction — block `b` occupies the
    // disjoint output rows `[row_offset_b .. +rows_b)` and columns
    // `[col_offset_b .. +cols_b)`, so each output row is exactly one block row with
    // its columns shifted right by `col_offset_b` (order preserved ⇒ still sorted).
    // Build the CSR directly (skipping the COO triplets + the O(nnz log nnz)/O(nnz)
    // `to_csr` pass) and fill it in parallel across blocks, which write disjoint
    // slices. Byte-identical to the COO path's `sorted_unique_coo_to_csr` result
    // (same values, same positions, same canonical meta). SciPy's `sparse.block_diag`
    // routes through COO stacking and is single-threaded, so this is a clean win.
    if matrices.iter().all(|m| {
        let c = m.canonical_meta();
        c.sorted_indices && c.deduplicated
    }) {
        return block_diag_canonical_csr(matrices, shape);
    }

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

/// Direct canonical-CSR construction for [`block_diag`] when all blocks are
/// canonical (see the fast-path comment there). `indptr` is built serially
/// (O(total_rows), cheap); the `indices`/`data` fill fans across blocks — each
/// block owns a disjoint, pre-sized nnz slice — gated by [`kron_fill_threads`].
fn block_diag_canonical_csr(matrices: &[&CsrMatrix], shape: Shape2D) -> SparseResult<CsrMatrix> {
    let total_rows = shape.rows;
    let nblocks = matrices.len();
    let mut col_offsets = Vec::with_capacity(nblocks);
    let mut nnz_offsets = Vec::with_capacity(nblocks);
    let mut col_acc = 0usize;
    let mut nnz_acc = 0usize;
    for &mat in matrices {
        col_offsets.push(col_acc);
        nnz_offsets.push(nnz_acc);
        col_acc += mat.shape().cols;
        nnz_acc += mat.nnz();
    }
    let total_nnz = nnz_acc;

    // Global indptr: each block contributes its per-row nnz in output-row order.
    let mut indptr = Vec::with_capacity(total_rows + 1);
    indptr.push(0usize);
    let mut acc = 0usize;
    for &mat in matrices {
        let bp = mat.indptr();
        for i in 0..mat.shape().rows {
            acc += bp[i + 1] - bp[i];
            indptr.push(acc);
        }
    }

    let nthreads = kron_fill_threads(total_rows, total_nnz);

    let (data, indices) = if nthreads <= 1 {
        let mut data = Vec::with_capacity(total_nnz);
        let mut indices = Vec::with_capacity(total_nnz);
        for (b, &mat) in matrices.iter().enumerate() {
            let co = col_offsets[b];
            data.extend_from_slice(mat.data());
            indices.extend(mat.indices().iter().map(|&ix| co + ix));
        }
        (data, indices)
    } else {
        let mut data = vec![0.0f64; total_nnz];
        let mut indices = vec![0usize; total_nnz];
        // Contiguous, nnz-balanced block bands so uneven block sizes still split
        // the fill work evenly.
        let target = total_nnz.div_ceil(nthreads).max(1);
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(nthreads);
        let mut start = 0usize;
        while start < nblocks {
            let base = nnz_offsets[start];
            let mut end = start + 1;
            while end < nblocks && nnz_offsets[end] - base < target {
                end += 1;
            }
            ranges.push((start, end));
            start = end;
        }
        let mut idx_rest: &mut [usize] = &mut indices;
        let mut dat_rest: &mut [f64] = &mut data;
        type FillJob<'a> = (usize, usize, &'a mut [usize], &'a mut [f64]);
        let mut jobs: Vec<FillJob<'_>> = Vec::with_capacity(ranges.len());
        for &(b0, b1) in &ranges {
            let band_nnz = if b1 < nblocks {
                nnz_offsets[b1]
            } else {
                total_nnz
            } - nnz_offsets[b0];
            let (ih, it) = idx_rest.split_at_mut(band_nnz);
            let (dh, dt) = dat_rest.split_at_mut(band_nnz);
            idx_rest = it;
            dat_rest = dt;
            jobs.push((b0, b1, ih, dh));
        }
        let col_offsets_s: &[usize] = &col_offsets;
        std::thread::scope(|scope| {
            for (b0, b1, mut idx_slice, mut dat_slice) in jobs {
                scope.spawn(move || {
                    for b in b0..b1 {
                        let mat = matrices[b];
                        let bn = mat.nnz();
                        let co = col_offsets_s[b];
                        let (this_idx, rest_idx) = idx_slice.split_at_mut(bn);
                        let (this_dat, rest_dat) = dat_slice.split_at_mut(bn);
                        this_dat.copy_from_slice(mat.data());
                        for (dst, &src) in this_idx.iter_mut().zip(mat.indices()) {
                            *dst = co + src;
                        }
                        idx_slice = rest_idx;
                        dat_slice = rest_dat;
                    }
                });
            }
        });
        (data, indices)
    };

    let mut result = CsrMatrix::from_components_unchecked(shape, data, indices, indptr);
    result.canonical = CanonicalMeta {
        sorted_indices: true,
        deduplicated: true,
    };
    Ok(result)
}

/// Construct a sparse matrix from a block layout of sparse matrices.
///
/// `blocks` is a row-major grid where `None` entries represent zero blocks.
/// All blocks in the same row must have the same number of rows,
/// and all blocks in the same column must have the same number of columns.
/// Matches `scipy.sparse.bmat(blocks)`.
/// When `true`, [`bmat`] takes the ORIG COO->to_csr path even for canonical CSR blocks; default
/// `false` builds the CSR directly by shifting each canonical block row into its column band.
/// Byte-identical result. A/B perf gate.
#[doc(hidden)]
pub static BMAT_FORCE_GENERIC: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

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

    let cap: usize = blocks
        .iter()
        .flatten()
        .filter_map(|block| block.map(CsrMatrix::nnz))
        .sum();

    // Fast path: when every present block is canonical CSR, the bmat output is canonical by
    // construction — output row (i,r) collects block (i,0)'s row r, then (i,1)'s, … with each
    // block's columns shifted into its disjoint, ascending column band [col_offset_j .. +width_j),
    // so entries come out strictly (row,col)-sorted and unique (columns also stay in range).
    // Build the CSR directly (indptr from per-row nnz, indices/data by shifting each block row's
    // slice), skipping the COO triplets, from_triplets, and to_csr. Byte-identical to the
    // COO->to_csr result (same entries, order, canonical meta). `BMAT_FORCE_GENERIC` restores the
    // original path; falls through when any block is non-canonical (there to_csr would re-sort).
    let all_canonical = blocks.iter().flatten().all(|block| match block {
        None => true,
        Some(mat) => {
            let c = mat.canonical_meta();
            c.sorted_indices && c.deduplicated
        }
    });
    if all_canonical && !BMAT_FORCE_GENERIC.load(std::sync::atomic::Ordering::Relaxed) {
        let mut col_offsets = vec![0usize; n_block_cols];
        let mut acc_c = 0usize;
        for (j, off) in col_offsets.iter_mut().enumerate() {
            *off = acc_c;
            acc_c += col_widths[j];
        }
        let mut indptr = Vec::with_capacity(total_rows + 1);
        indptr.push(0usize);
        let mut indices = Vec::with_capacity(cap);
        let mut data = Vec::with_capacity(cap);
        let mut acc = 0usize;
        for (i, block_row) in blocks.iter().enumerate() {
            for r in 0..row_heights[i] {
                for (j, block) in block_row.iter().enumerate() {
                    if let Some(mat) = block {
                        let bp = mat.indptr();
                        let co = col_offsets[j];
                        for idx in bp[r]..bp[r + 1] {
                            indices.push(co + mat.indices()[idx]);
                            data.push(mat.data()[idx]);
                        }
                        acc += bp[r + 1] - bp[r];
                    }
                }
                indptr.push(acc);
            }
        }
        let mut result = CsrMatrix::from_components_unchecked(shape, data, indices, indptr);
        result.canonical = CanonicalMeta {
            sorted_indices: true,
            deduplicated: true,
        };
        return Ok(result);
    }

    let mut all_rows = Vec::with_capacity(cap);
    let mut all_cols = Vec::with_capacity(cap);
    let mut all_data = Vec::with_capacity(cap);

    // Emit row-by-row ACROSS the blocks of each block-row (not block-by-block) so
    // the triplets come out strictly (row,col)-sorted: a fixed output row
    // `row_offset+r` collects its entries from block (i,0), then (i,1), … with
    // `col_offset` increasing and each block's row sorted, so columns strictly
    // increase; rows strictly increase across r and block-rows. Sorted+unique
    // triplets let `CooMatrix::to_csr` take its O(nnz) `sorted_unique_coo_to_csr`
    // fast path instead of sorting all the block-major (row-repeating) triplets —
    // 7.45x faster (a 6.26x SciPy loss → 1.19x faster), byte-identical. The old
    // block-major order re-emitted each output row once per block column, forcing
    // the O(nnz log nnz) sort. (Same lever as kron's sorted emission.)
    let mut row_offset = 0;
    for (i, block_row) in blocks.iter().enumerate() {
        for r in 0..row_heights[i] {
            let out_row = row_offset + r;
            let mut col_offset = 0;
            for (j, block) in block_row.iter().enumerate() {
                if let Some(mat) = block {
                    let indptr = mat.indptr();
                    let indices = mat.indices();
                    let mat_data = mat.data();
                    for idx in indptr[r]..indptr[r + 1] {
                        all_rows.push(out_row);
                        all_cols.push(col_offset + indices[idx]);
                        all_data.push(mat_data[idx]);
                    }
                }
                col_offset += col_widths[j];
            }
        }
        row_offset += row_heights[i];
    }

    let coo = CooMatrix::from_triplets(shape, all_data, all_rows, all_cols, false)?;
    coo.to_csr()
}

/// Build a sparse array from a 2-D grid of (optional) blocks, matching
/// `scipy.sparse.block_array`.
///
/// The array-API equivalent of [`bmat`] (same result; scipy distinguishes only
/// the returned container type, which fsci unifies). `None` entries are treated
/// as all-zero blocks.
pub fn block_array(blocks: &[Vec<Option<&CsrMatrix>>]) -> SparseResult<SparseArray2D<CsrMatrix>> {
    bmat(blocks).map(SparseArray2D::new)
}

/// Stack sparse matrices vertically (row wise).
///
/// Mirrors `scipy.sparse.vstack` for sparse inputs. The Rust API returns CSR
/// directly; call `.to_coo()` when coordinate output is desired.
/// When `true`, [`vstack`] takes the ORIG COO->to_csr path even for canonical CSR blocks; default
/// `false` concatenates canonical blocks directly into a CSR. Byte-identical result. A/B perf gate.
#[doc(hidden)]
pub static VSTACK_FORCE_GENERIC: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

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

/// Format-polymorphic result of [`random_array`].
///
/// Every variant is an array container, never a legacy sparse matrix.
#[derive(Debug, Clone, PartialEq)]
pub enum SparseArrayOutput {
    Csr(SparseArray2D<CsrMatrix>),
    Csc(SparseArray2D<CscMatrix>),
    Coo(CooArray),
    Bsr(SparseArray2D<BsrMatrix>),
    Dia(SparseArray2D<DiaMatrix>),
    Dok(SparseArray2D<DokMatrix>),
    Lil(SparseArray2D<LilMatrix>),
}

impl SparseArrayOutput {
    /// Return the SciPy format spelling for the contained sparse array.
    #[must_use]
    pub fn format_name(&self) -> &'static str {
        sparse_format_name(self.format())
    }
}

impl From<HstackOutput> for SparseArrayOutput {
    fn from(output: HstackOutput) -> Self {
        match output {
            HstackOutput::Csr(matrix) => Self::Csr(SparseArray2D::new(matrix)),
            HstackOutput::Csc(matrix) => Self::Csc(SparseArray2D::new(matrix)),
            HstackOutput::Coo(matrix) => Self::Coo(CooArray::from_coo_matrix(&matrix)),
            HstackOutput::Bsr(matrix) => Self::Bsr(SparseArray2D::new(matrix)),
            HstackOutput::Dia(matrix) => Self::Dia(SparseArray2D::new(matrix)),
            HstackOutput::Dok(matrix) => Self::Dok(SparseArray2D::new(matrix)),
            HstackOutput::Lil(matrix) => Self::Lil(SparseArray2D::new(matrix)),
        }
    }
}

impl SparseObject for SparseArrayOutput {
    fn format(&self) -> SparseFormat {
        match self {
            Self::Csr(array) => array.format(),
            Self::Csc(array) => array.format(),
            Self::Coo(array) => array.format(),
            Self::Bsr(array) => array.format(),
            Self::Dia(array) => array.format(),
            Self::Dok(array) => array.format(),
            Self::Lil(array) => array.format(),
        }
    }

    fn shape_nd(&self) -> Vec<usize> {
        match self {
            Self::Csr(array) => array.shape_nd(),
            Self::Csc(array) => array.shape_nd(),
            Self::Coo(array) => array.shape_nd(),
            Self::Bsr(array) => array.shape_nd(),
            Self::Dia(array) => array.shape_nd(),
            Self::Dok(array) => array.shape_nd(),
            Self::Lil(array) => array.shape_nd(),
        }
    }

    fn nnz(&self) -> usize {
        match self {
            Self::Csr(array) => array.nnz(),
            Self::Csc(array) => array.nnz(),
            Self::Coo(array) => array.nnz(),
            Self::Bsr(array) => array.nnz(),
            Self::Dia(array) => array.nnz(),
            Self::Dok(array) => array.nnz(),
            Self::Lil(array) => array.nnz(),
        }
    }

    fn is_matrix(&self) -> bool {
        false
    }
}

impl SparseArray for SparseArrayOutput {
    fn to_coo_array(&self) -> SparseResult<CooArray> {
        match self {
            Self::Csr(array) => array.to_coo_array(),
            Self::Csc(array) => array.to_coo_array(),
            Self::Coo(array) => Ok(array.clone()),
            Self::Bsr(array) => array.to_coo_array(),
            Self::Dia(array) => array.to_coo_array(),
            Self::Dok(array) => array.to_coo_array(),
            Self::Lil(array) => array.to_coo_array(),
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
/// When `true`, [`kron`]'s canonical fast path validates its (provably canonical) output through
/// `from_components(.., true)` — the ORIG O(nnz) bounds scan + detect_canonical. Default `false`
/// stamps {sorted, deduplicated} via the trusted constructor. Byte-identical result. A/B perf gate.
#[doc(hidden)]
pub static KRON_VALIDATE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

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

    if let Some(csr) = kron_canonical_csr(a, b, shape)? {
        return Ok(csr);
    }

    kron_via_coo(a, b, shape)
}

fn kron_canonical_csr(
    a: &CsrMatrix,
    b: &CsrMatrix,
    shape: Shape2D,
) -> SparseResult<Option<CsrMatrix>> {
    let a_meta = a.canonical_meta();
    let b_meta = b.canonical_meta();
    if !a_meta.sorted_indices
        || !a_meta.deduplicated
        || !b_meta.sorted_indices
        || !b_meta.deduplicated
    {
        return Ok(None);
    }

    let a_indptr = a.indptr();
    let a_indices = a.indices();
    let a_data = a.data();
    let b_indptr = b.indptr();
    let b_indices = b.indices();
    let b_data = b.data();
    let b_shape = b.shape();
    let total_nnz = a
        .nnz()
        .checked_mul(b.nnz())
        .ok_or_else(|| SparseError::IndexOverflow {
            message: "kron output nnz overflow".to_string(),
        })?;

    let mut indptr = Vec::with_capacity(shape.rows + 1);
    indptr.push(0usize);
    for ai in 0..a.shape().rows {
        let a_row_nnz = a_indptr[ai + 1] - a_indptr[ai];
        for bi in 0..b_shape.rows {
            let b_row_nnz = b_indptr[bi + 1] - b_indptr[bi];
            let row_nnz =
                a_row_nnz
                    .checked_mul(b_row_nnz)
                    .ok_or_else(|| SparseError::IndexOverflow {
                        message: "kron output row nnz overflow".to_string(),
                    })?;
            let next = indptr
                .last()
                .copied()
                .expect("kron indptr starts at zero")
                .checked_add(row_nnz)
                .ok_or_else(|| SparseError::IndexOverflow {
                    message: "kron output nnz overflow".to_string(),
                })?;
            indptr.push(next);
        }
    }

    // The output row `r = ai*mb + bi` has a fixed, already-known extent
    // `[indptr[r]..indptr[r+1]]`, and each output column `aj*nb + b_col` is bounded
    // by the top-level `out_cols` overflow check (aj < a_cols ⇒ aj*nb < a_cols*nb =
    // out_cols, a valid usize), so the per-entry `checked_mul` is redundant here.
    // Every output row writes a disjoint slice, so the fill fans across cores: each
    // worker owns a contiguous, nnz-balanced band of output rows and writes directly
    // into its pre-sized slice of `indices`/`data`. Byte-identical to the serial
    // push order (the same ai-outer/bi-inner/a_idx/b_idx nesting lands each entry at
    // the same position). SciPy's `sparse.kron` is single-threaded, so this is a
    // straight multicore win; gated below `total_nnz` so small products stay serial.
    let a_rows = a.shape().rows;
    let b_rows = b_shape.rows;
    let b_cols = b_shape.cols;
    let nthreads = kron_fill_threads(shape.rows, total_nnz);

    let (data, indices) = if nthreads <= 1 {
        let mut indices = Vec::with_capacity(total_nnz);
        let mut data = Vec::with_capacity(total_nnz);
        for ai in 0..a_rows {
            for bi in 0..b_rows {
                for a_idx in a_indptr[ai]..a_indptr[ai + 1] {
                    let aj = a_indices[a_idx];
                    let a_val = a_data[a_idx];
                    let col_base = aj * b_cols;
                    for b_idx in b_indptr[bi]..b_indptr[bi + 1] {
                        indices.push(col_base + b_indices[b_idx]);
                        data.push(a_val * b_data[b_idx]);
                    }
                }
            }
        }
        (data, indices)
    } else {
        let mut indices = vec![0usize; total_nnz];
        let mut data = vec![0.0f64; total_nnz];
        // Contiguous, nnz-balanced output-row bands (each ≈ total_nnz/nthreads
        // entries) so uneven rows still split the fill work evenly.
        let target = total_nnz.div_ceil(nthreads).max(1);
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(nthreads);
        let mut start = 0usize;
        while start < shape.rows {
            let base = indptr[start];
            let mut end = start + 1;
            while end < shape.rows && indptr[end] - base < target {
                end += 1;
            }
            ranges.push((start, end));
            start = end;
        }
        // Carve one disjoint mutable slice per band up front (indptr gives the exact
        // extent), then fill them concurrently.
        let indptr_s: &[usize] = &indptr;
        let mut idx_rest: &mut [usize] = &mut indices;
        let mut dat_rest: &mut [f64] = &mut data;
        type FillJob<'a> = (usize, usize, &'a mut [usize], &'a mut [f64]);
        let mut jobs: Vec<FillJob<'_>> = Vec::with_capacity(ranges.len());
        for &(r0, r1) in &ranges {
            let len = indptr_s[r1] - indptr_s[r0];
            let (ih, it) = idx_rest.split_at_mut(len);
            let (dh, dt) = dat_rest.split_at_mut(len);
            idx_rest = it;
            dat_rest = dt;
            jobs.push((r0, r1, ih, dh));
        }
        std::thread::scope(|scope| {
            for (r0, r1, mut idx_slice, mut dat_slice) in jobs {
                scope.spawn(move || {
                    for r in r0..r1 {
                        let ai = r / b_rows;
                        let bi = r % b_rows;
                        let row_len = indptr_s[r + 1] - indptr_s[r];
                        let (this_idx, rest_idx) = idx_slice.split_at_mut(row_len);
                        let (this_dat, rest_dat) = dat_slice.split_at_mut(row_len);
                        let mut pos = 0usize;
                        for a_idx in a_indptr[ai]..a_indptr[ai + 1] {
                            let aj = a_indices[a_idx];
                            let a_val = a_data[a_idx];
                            let col_base = aj * b_cols;
                            for b_idx in b_indptr[bi]..b_indptr[bi + 1] {
                                this_idx[pos] = col_base + b_indices[b_idx];
                                this_dat[pos] = a_val * b_data[b_idx];
                                pos += 1;
                            }
                        }
                        idx_slice = rest_idx;
                        dat_slice = rest_dat;
                    }
                });
            }
        });
        (data, indices)
    };

    // Both inputs are canonical (checked above), so each output row `r` emits columns
    // `aj*b_cols + b_col` with aj ascending (a's row sorted) and b_col ascending within each aj
    // block — strictly increasing and unique, i.e. already sorted + deduplicated. Columns are also
    // in range (aj*b_cols + b_col < a_cols*b_cols = out_cols). Route through the trusted constructor
    // to skip `from_components`' redundant O(nnz) bounds scan + detect_canonical; byte-identical
    // result AND canonical metadata.
    if KRON_VALIDATE.load(std::sync::atomic::Ordering::Relaxed) {
        Ok(Some(CsrMatrix::from_components(
            shape, data, indices, indptr, true,
        )?))
    } else {
        Ok(Some(CsrMatrix::from_components_trusted_canonical(
            shape, data, indices, indptr,
        )))
    }
}

/// Thread count for the [`kron_canonical_csr`] disjoint-slice fill. Serial below
/// 64K output entries or 1024 output rows (thread-spawn cost dominates); otherwise
/// caps at 16 cores with ≥32K entries per worker so each spawn is amortized.
fn kron_fill_threads(out_rows: usize, total_nnz: usize) -> usize {
    if total_nnz < 1 << 16 || out_rows < 1024 {
        return 1;
    }
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    cores.min(16).min(total_nnz / 32_768).max(1)
}

fn kron_via_coo(a: &CsrMatrix, b: &CsrMatrix, shape: Shape2D) -> SparseResult<CsrMatrix> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let a_indptr = a.indptr();
    let a_indices = a.indices();
    let a_data = a.data();
    let b_indptr = b.indptr();
    let b_indices = b.indices();
    let b_data = b.data();

    // Nest the block row (bi) OUTSIDE A's column scan so the emitted triplets are
    // strictly increasing in (row, col): the output row `ai*mb + bi` is constant
    // for a fixed (ai, bi) and monotonic across them, and within that row the
    // columns `aj*nb + bj` increase (A's row sorted by aj outer, B's row sorted by
    // bj inner, with bj < nb). Sorted+unique triplets let `CooMatrix::to_csr` take
    // its O(nnz) `sorted_unique_coo_to_csr` fast path instead of sorting all
    // nnz_a·nnz_b entries — 3.65x faster (a 4.09x SciPy loss → ~parity),
    // byte-identical (same entries, same canonical CSR). The old order (A-col scan
    // outside bi) produced unsorted rows and forced the O(nnz log nnz) sort.
    let cap = a.nnz().saturating_mul(b.nnz());
    let mut rows = Vec::with_capacity(cap);
    let mut cols = Vec::with_capacity(cap);
    let mut data = Vec::with_capacity(cap);

    for ai in 0..a_shape.rows {
        for bi in 0..b_shape.rows {
            let out_row = ai * b_shape.rows + bi;
            for a_idx in a_indptr[ai]..a_indptr[ai + 1] {
                let aj = a_indices[a_idx];
                let a_val = a_data[a_idx];
                let col_base = aj * b_shape.cols;
                for b_idx in b_indptr[bi]..b_indptr[bi + 1] {
                    rows.push(out_row);
                    cols.push(col_base + b_indices[b_idx]);
                    data.push(a_val * b_data[b_idx]);
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

    // hstack (Cols): all blocks share the row count and occupy disjoint COLUMN
    // ranges, so the old block-by-block append re-emitted rows 0..R once per
    // block → non-monotonic rows → `to_csr` sorted all nnz triplets
    // O(nnz log nnz). Emit ROW-BY-ROW across the blocks instead: row is
    // monotonic and within a row `col_offset` grows across blocks (each block's
    // row already sorted) → strictly (row,col)-sorted → the O(nnz)
    // `sorted_unique_coo_to_csr` fast path fires. 10.64x faster (an 11.1x SciPy
    // loss → ~parity), byte-identical. (vstack's Rows axis below already emits
    // disjoint, monotonic row ranges, so it stays on the original path.)
    if matches!(axis, StackAxis::Cols) {
        let csrs: Vec<CsrMatrix> = blocks
            .iter()
            .map(|block| block.to_csr())
            .collect::<SparseResult<Vec<_>>>()?;
        let shared_rows = csrs[0].shape().rows;
        let mut total_cols = 0usize;
        for csr in &csrs {
            if csr.shape().rows != shared_rows {
                return Err(SparseError::IncompatibleShape {
                    message: "hstack requires all blocks to have matching row counts".to_string(),
                });
            }
            total_cols += csr.shape().cols;
        }
        let nnz: usize = csrs.iter().map(CsrMatrix::nnz).sum();
        let mut rows = Vec::with_capacity(nnz);
        let mut cols = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);
        for r in 0..shared_rows {
            let mut col_offset = 0usize;
            for csr in &csrs {
                let indptr = csr.indptr();
                let indices = csr.indices();
                let csr_data = csr.data();
                for idx in indptr[r]..indptr[r + 1] {
                    rows.push(r);
                    cols.push(col_offset + indices[idx]);
                    data.push(csr_data[idx]);
                }
                col_offset += csr.shape().cols;
            }
        }
        let shape = Shape2D::new(shared_rows, total_cols);
        let coo = CooMatrix::from_triplets(shape, data, rows, cols, false)?;
        return coo.to_csr();
    }

    // vstack (Rows): blocks occupy disjoint, monotonic output-row ranges and SHARE the column
    // space (no column shift). When every block is canonical CSR, the stacked result is already
    // canonical — concatenate each block's indices/data verbatim and append its per-row nnz to
    // indptr, skipping the N to_coo conversions, the COO triplets, and to_csr's rebuild.
    // Byte-identical to the COO->to_csr result (same entries in row-major order, same canonical
    // meta). Falls through to the generic path when a block isn't canonical (there to_csr would
    // re-sort). `VSTACK_FORCE_GENERIC` restores the original COO path (same-binary A/B).
    if matches!(axis, StackAxis::Rows)
        && !VSTACK_FORCE_GENERIC.load(std::sync::atomic::Ordering::Relaxed)
    {
        let csrs: Vec<CsrMatrix> = blocks
            .iter()
            .map(|block| block.to_csr())
            .collect::<SparseResult<Vec<_>>>()?;
        let shared_cols = csrs[0].shape().cols;
        let mut all_canonical = true;
        let mut total_rows = 0usize;
        let mut total_nnz = 0usize;
        for csr in &csrs {
            if csr.shape().cols != shared_cols {
                return Err(SparseError::IncompatibleShape {
                    message: "vstack requires all blocks to have matching column counts"
                        .to_string(),
                });
            }
            let c = csr.canonical_meta();
            all_canonical &= c.sorted_indices && c.deduplicated;
            total_rows += csr.shape().rows;
            total_nnz += csr.nnz();
        }
        if all_canonical {
            let mut indices = Vec::with_capacity(total_nnz);
            let mut data = Vec::with_capacity(total_nnz);
            let mut indptr = Vec::with_capacity(total_rows + 1);
            indptr.push(0usize);
            let mut acc = 0usize;
            for csr in &csrs {
                data.extend_from_slice(csr.data());
                indices.extend_from_slice(csr.indices());
                let bp = csr.indptr();
                for i in 0..csr.shape().rows {
                    acc += bp[i + 1] - bp[i];
                    indptr.push(acc);
                }
            }
            let shape = Shape2D::new(total_rows, shared_cols);
            let mut result = CsrMatrix::from_components_unchecked(shape, data, indices, indptr);
            result.canonical = CanonicalMeta {
                sorted_indices: true,
                deduplicated: true,
            };
            return Ok(result);
        }
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
        assert_eq!(
            id.data()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![1.0_f64.to_bits(); 4]
        );
        assert_eq!(id.indices(), &[0, 1, 2, 3]);
        assert_eq!(id.indptr(), &[0, 1, 2, 3, 4]);
        assert_eq!(
            id.canonical_meta(),
            CanonicalMeta {
                sorted_indices: true,
                deduplicated: true,
            }
        );

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
        assert!(id.data().is_empty());
        assert!(id.indices().is_empty());
        assert_eq!(id.indptr(), &[0]);
        assert_eq!(
            id.canonical_meta(),
            CanonicalMeta {
                sorted_indices: true,
                deduplicated: true,
            }
        );
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

    /// `diags` returns CSR, so its peer is `scipy.sparse.diags(...,
    /// format="csr")` — which prunes exact zeros in the dia→csr conversion —
    /// not the unpruned DIA intermediate. Goldens measured live on scipy
    /// 1.17.1 / numpy 2.4.3. See [frankenscipy-mhxei].
    #[test]
    fn diags_prunes_exact_zeros_like_scipy_csr_peer() {
        // sp.diags([np.zeros(5)], [0], shape=(5,5), format="csr")
        //   -> nnz 0, data [], indptr [0 0 0 0 0 0]
        // (the DIA intermediate has nnz 5; that is `spdiags`' contract, not this one)
        let all_zero = diags(&[vec![0.0; 5]], &[0], Some(Shape2D::new(5, 5))).expect("diags");
        assert_eq!(all_zero.nnz(), 0, "all-zero diagonal stores nothing");
        assert!(all_zero.data().is_empty());
        assert_eq!(all_zero.indptr(), &[0, 0, 0, 0, 0, 0]);

        // sp.diags([[1,0,2,0,3]], [0], shape=(5,5), format="csr")
        //   -> nnz 3, data [1. 2. 3.], indices [0 2 4], indptr [0 1 1 2 2 3]
        let mixed = diags(
            &[vec![1.0, 0.0, 2.0, 0.0, 3.0]],
            &[0],
            Some(Shape2D::new(5, 5)),
        )
        .expect("diags");
        assert_eq!(mixed.nnz(), 3, "interior zeros pruned too");
        assert_eq!(mixed.data(), &[1.0, 2.0, 3.0]);
        assert_eq!(mixed.indices(), &[0, 2, 4]);
        assert_eq!(mixed.indptr(), &[0, 1, 1, 2, 2, 3]);

        // Pruning must not disturb the surviving entries of an off-diagonal
        // build: sp.diags([[0,7],[0,0,0]], [1,0], shape=(3,3), format="csr")
        //   -> nnz 1, data [7.], indices [2], indptr [0 0 1 1]
        let offset = diags(
            &[vec![0.0, 7.0], vec![0.0, 0.0, 0.0]],
            &[1, 0],
            Some(Shape2D::new(3, 3)),
        )
        .expect("diags");
        assert_eq!(offset.nnz(), 1);
        assert_eq!(offset.data(), &[7.0]);
        assert_eq!(offset.indices(), &[2]);
        assert_eq!(offset.indptr(), &[0, 0, 1, 1]);

        // NaN is not equal to zero, so it survives — as `data != 0` does in
        // numpy. -0.0 == 0.0 in both, so it is pruned.
        let edges =
            diags(&[vec![f64::NAN, -0.0, 1.0]], &[0], Some(Shape2D::new(3, 3))).expect("diags");
        assert_eq!(edges.nnz(), 2, "NaN kept, -0.0 pruned");
        assert!(edges.data()[0].is_nan());
        assert_eq!(edges.indices(), &[0, 2]);
    }

    #[test]
    fn diags_rejects_out_of_bounds_for_explicit_shape() {
        let err = diags(&[vec![1.0, 2.0, 3.0]], &[1], Some(Shape2D::new(3, 3)))
            .expect_err("bounds violation");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn eye_rectangular_direct_matches_coo_to_csr_path() {
        fn old_eye_rect(rows: usize, cols: usize, k: isize) -> CsrMatrix {
            let shape = Shape2D::new(rows, cols);
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
            CooMatrix::from_triplets(shape, data, r, c, false)
                .unwrap()
                .to_csr()
                .unwrap()
        }
        // Main diagonal, upper/lower shifts, non-square both ways, out-of-range k (empty), 1x1.
        for &(rows, cols, k) in &[
            (5usize, 5usize, 0isize),
            (5, 5, 2),
            (5, 5, -1),
            (4, 7, 3),
            (7, 4, -2),
            (6, 6, 10),
            (6, 6, -10),
            (1, 1, 0),
            (5, 3, -4),
        ] {
            let old = old_eye_rect(rows, cols, k);
            let new = eye_rectangular(rows, cols, k).expect("eye_rectangular");
            assert_eq!(new.shape(), old.shape(), "shape {rows}x{cols} k{k}");
            assert_eq!(new.indptr(), old.indptr(), "indptr {rows}x{cols} k{k}");
            assert_eq!(new.indices(), old.indices(), "indices {rows}x{cols} k{k}");
            assert_eq!(
                new.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                old.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "data {rows}x{cols} k{k}"
            );
            assert_eq!(
                new.canonical_meta().sorted_indices,
                old.canonical_meta().sorted_indices
            );
            assert_eq!(
                new.canonical_meta().deduplicated,
                old.canonical_meta().deduplicated
            );
        }
    }

    #[test]
    fn bmat_direct_matches_generic_bitwise() {
        use std::sync::atomic::Ordering;
        // 2x2 grid of canonical CSR blocks with a None (zero) block; heights/widths consistent.
        let a = random(Shape2D::new(6, 5), 0.25, 0xa1)
            .unwrap()
            .to_csr()
            .unwrap();
        let b = random(Shape2D::new(6, 7), 0.2, 0xb2)
            .unwrap()
            .to_csr()
            .unwrap();
        let d = random(Shape2D::new(4, 7), 0.3, 0xd4)
            .unwrap()
            .to_csr()
            .unwrap();
        // Row 0: [a (6x5), b (6x7)]; Row 1: [None (4x5), d (4x7)].
        let grid: Vec<Vec<Option<&CsrMatrix>>> =
            vec![vec![Some(&a), Some(&b)], vec![None, Some(&d)]];
        BMAT_FORCE_GENERIC.store(true, Ordering::Relaxed);
        let generic = bmat(&grid).unwrap();
        BMAT_FORCE_GENERIC.store(false, Ordering::Relaxed);
        let direct = bmat(&grid).unwrap();
        BMAT_FORCE_GENERIC.store(false, Ordering::Relaxed);
        assert_eq!(direct.shape(), generic.shape());
        assert_eq!(direct.indptr(), generic.indptr());
        assert_eq!(direct.indices(), generic.indices());
        assert_eq!(
            direct
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            generic
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            direct.canonical_meta().sorted_indices,
            generic.canonical_meta().sorted_indices
        );
        assert_eq!(
            direct.canonical_meta().deduplicated,
            generic.canonical_meta().deduplicated
        );
    }

    #[test]
    fn vstack_direct_matches_generic_bitwise() {
        use std::sync::atomic::Ordering;
        // Canonical CSR blocks sharing the column count (20).
        let a = random(Shape2D::new(10, 20), 0.2, 0x11)
            .unwrap()
            .to_csr()
            .unwrap();
        let b = random(Shape2D::new(15, 20), 0.15, 0x22)
            .unwrap()
            .to_csr()
            .unwrap();
        let c = random(Shape2D::new(5, 20), 0.3, 0x33)
            .unwrap()
            .to_csr()
            .unwrap();
        let blocks: Vec<&dyn FormatConvertible> = vec![&a, &b, &c];
        VSTACK_FORCE_GENERIC.store(true, Ordering::Relaxed);
        let generic = vstack(&blocks).unwrap();
        VSTACK_FORCE_GENERIC.store(false, Ordering::Relaxed);
        let direct = vstack(&blocks).unwrap();
        VSTACK_FORCE_GENERIC.store(false, Ordering::Relaxed);
        assert_eq!(direct.shape(), generic.shape());
        assert_eq!(direct.indptr(), generic.indptr());
        assert_eq!(direct.indices(), generic.indices());
        assert_eq!(
            direct
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            generic
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            direct.canonical_meta().sorted_indices,
            generic.canonical_meta().sorted_indices
        );
        assert_eq!(
            direct.canonical_meta().deduplicated,
            generic.canonical_meta().deduplicated
        );
    }

    #[test]
    fn kron_trusted_matches_validated_bitwise() {
        use std::sync::atomic::Ordering;
        // Canonical CSR inputs (random -> to_csr canonicalizes) so kron takes its fast path.
        let a = random(Shape2D::new(30, 40), 0.1, 0x1234)
            .unwrap()
            .to_csr()
            .unwrap();
        let b = random(Shape2D::new(20, 25), 0.15, 0x9abc)
            .unwrap()
            .to_csr()
            .unwrap();
        KRON_VALIDATE.store(true, Ordering::Relaxed);
        let validated = kron(&a, &b).unwrap();
        KRON_VALIDATE.store(false, Ordering::Relaxed);
        let trusted = kron(&a, &b).unwrap();
        KRON_VALIDATE.store(false, Ordering::Relaxed);
        assert_eq!(trusted.shape(), validated.shape());
        assert_eq!(trusted.indptr(), validated.indptr());
        assert_eq!(trusted.indices(), validated.indices());
        assert_eq!(
            trusted
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            validated
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            trusted.canonical_meta().sorted_indices,
            validated.canonical_meta().sorted_indices
        );
        assert_eq!(
            trusted.canonical_meta().deduplicated,
            validated.canonical_meta().deduplicated
        );
    }

    #[test]
    fn diags_precheck_last_only_matches_full_scan_verdict() {
        use std::sync::atomic::Ordering;
        // (diagonals, offsets, shape) cases: valid ones and ones that exceed the shape.
        // Named so `clippy::type_complexity` is satisfied by a definition
        // rather than an allow (frankenscipy-nwx8m).
        type DiagsCase<'a> = (&'a [&'a [f64]], &'a [isize], Shape2D);
        let cases: &[DiagsCase] = &[
            (&[&[1.0, 2.0, 3.0]], &[0], Shape2D::new(3, 3)),
            (
                &[&[1.0, 2.0], &[3.0, 4.0, 5.0]],
                &[1, 0],
                Shape2D::new(3, 3),
            ),
            (&[&[1.0, 2.0, 3.0]], &[1], Shape2D::new(3, 3)), // last col out of bounds
            (&[&[1.0, 2.0, 3.0, 4.0]], &[0], Shape2D::new(3, 3)), // overruns both axes
        ];
        for &(diags_in, offsets, shape) in cases {
            let owned: Vec<Vec<f64>> = diags_in.iter().map(|d| d.to_vec()).collect();
            DIAGS_VALIDATE.store(true, Ordering::Relaxed);
            let full = diags(&owned, offsets, Some(shape));
            DIAGS_VALIDATE.store(false, Ordering::Relaxed);
            let fast = diags(&owned, offsets, Some(shape));
            assert_eq!(
                full.is_ok(),
                fast.is_ok(),
                "verdict mismatch for {offsets:?}"
            );
            if let (Ok(a), Ok(b)) = (&full, &fast) {
                assert_eq!(a.indptr(), b.indptr());
                assert_eq!(a.indices(), b.indices());
                assert_eq!(
                    a.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                // Trusted path must also carry the same canonical metadata as the validated path.
                assert_eq!(
                    a.canonical_meta().sorted_indices,
                    b.canonical_meta().sorted_indices
                );
                assert_eq!(
                    a.canonical_meta().deduplicated,
                    b.canonical_meta().deduplicated
                );
            }
        }
        DIAGS_VALIDATE.store(false, Ordering::Relaxed);
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

    /// `random_array` must match SciPy's STRUCTURAL contract exactly: `nnz` from
    /// round-half-to-even, values in `[0, 1)`, distinct coordinates, requested format.
    /// The expected `nnz` values below were taken from
    /// `scipy.sparse.random_array((7,5), density=d)` on scipy 1.17.1.
    #[test]
    fn random_array_matches_scipy_structural_contract() {
        // (density, expected nnz) for a 7x5 = 35-element array, from scipy 1.17.1.
        for &(density, expected_nnz) in &[
            (0.0, 0usize),
            (0.1, 4),    // 3.5 -> 4
            (0.25, 9),   // 8.75 -> 9
            (0.333, 12), // 11.655 -> 12
            (1.0, 35),
        ] {
            let out = random_array(Shape2D::new(7, 5), density, Some("coo"), 0xC0FFEE)
                .expect("random_array");
            assert!(
                matches!(&out, SparseArrayOutput::Coo(_)),
                "format=coo must return a COO array, got {out:?}"
            );
            let coo = out.to_coo_array().expect("COO array conversion");
            assert_eq!(
                coo.data().len(),
                expected_nnz,
                "nnz for density={density} must match scipy"
            );
            for &v in coo.data() {
                assert!((0.0..1.0).contains(&v), "value {v} outside scipy's [0, 1)");
            }
            // Distinct coordinates: scipy samples without replacement.
            let mut seen: HashSet<(usize, usize)> = HashSet::new();
            for (&r, &c) in coo.coords()[0].iter().zip(&coo.coords()[1]) {
                assert!(r < 7 && c < 5, "index ({r},{c}) out of bounds");
                assert!(seen.insert((r, c)), "duplicate coordinate ({r},{c})");
            }
        }
    }

    /// Round-half-to-even is the specific rule, not "round". `2.5` must give 2.
    /// Rust's `f64::round` would give 3 here, and that divergence is the whole
    /// reason `random_array` does not delegate to `random`.
    #[test]
    fn random_array_nnz_uses_round_half_to_even() {
        // 10x1 = 10 elements, density 0.25 -> 2.5 -> ties-even -> 2 (scipy agrees).
        let out = random_array(Shape2D::new(10, 1), 0.25, Some("coo"), 7).expect("random_array");
        assert!(matches!(&out, SparseArrayOutput::Coo(_)));
        let coo = out.to_coo_array().expect("COO array conversion");
        assert_eq!(coo.data().len(), 2, "2.5 must round to 2, not 3");
    }

    /// Same seed must give the same array; different seeds must differ.
    #[test]
    fn random_array_is_deterministic_in_seed() {
        let get = |seed: u64| {
            let output =
                random_array(Shape2D::new(20, 20), 0.2, Some("coo"), seed).expect("random_array");
            assert!(matches!(&output, SparseArrayOutput::Coo(_)));
            let c = output.to_coo_array().expect("COO array conversion");
            (
                c.coords()[0].clone(),
                c.coords()[1].clone(),
                c.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            )
        };
        assert_eq!(get(42), get(42), "same seed must reproduce bit-for-bit");
        assert_ne!(get(42).0, get(43).0, "different seeds must differ");
    }

    #[test]
    fn random_array_returns_array_variants_for_every_format() {
        for format in ["csr", "csc", "coo", "bsr", "dia", "dok", "lil"] {
            let output =
                random_array(Shape2D::new(4, 3), 0.5, Some(format), 17).expect("random array");
            assert!(!output.is_matrix());
            assert_eq!(output.shape_nd(), vec![4, 3]);
            assert_eq!(output.format_name(), format);
        }
    }

    /// Every SciPy format string must be accepted and honoured, and a bad one
    /// rejected — `rand` is just `random_array` with the shape spelled out.
    #[test]
    fn rand_honours_every_format_and_rejects_bad_ones() {
        for fmt in ["csr", "csc", "coo", "bsr", "dia", "dok", "lil"] {
            let out = rand(6, 4, 0.5, Some(fmt), 0x5EED).expect("rand");
            let matches = matches!(
                (fmt, &out),
                ("csr", HstackOutput::Csr(_))
                    | ("csc", HstackOutput::Csc(_))
                    | ("coo", HstackOutput::Coo(_))
                    | ("bsr", HstackOutput::Bsr(_))
                    | ("dia", HstackOutput::Dia(_))
                    | ("dok", HstackOutput::Dok(_))
                    | ("lil", HstackOutput::Lil(_))
            );
            assert!(matches, "format {fmt} returned the wrong variant");
        }
        assert!(rand(4, 4, 0.5, Some("nope"), 1).is_err());
        assert!(
            rand(4, 4, 1.5, Some("coo"), 1).is_err(),
            "density > 1 rejected"
        );
        assert!(
            rand(4, 4, f64::NAN, Some("coo"), 1).is_err(),
            "NaN density rejected"
        );
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
    fn random_density_zero_does_not_scan_huge_shape() {
        let coo = random(Shape2D::new(usize::MAX, 1), 0.0, 11).expect("random");
        assert_eq!(coo.shape(), Shape2D::new(usize::MAX, 1));
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
    fn random_tiny_density_rounds_to_empty_without_dense_scan() {
        let shape = Shape2D::new(1_000_000_000, 1_000_000_000);
        let coo = random(shape, 1e-19, 42).expect("random");
        assert_eq!(coo.shape(), shape);
        assert_eq!(coo.nnz(), 0);
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
    fn block_diag_canonical_parallel_is_byte_identical_to_reference() {
        // Many canonical blocks, total_nnz past the parallel gate: the direct-CSR
        // fast path (parallel across blocks) must reproduce the block-diagonal
        // emission order exactly, bit-for-bit.
        fn canonical_csr(rows: usize, cols: usize, per_row: usize, seed: u64) -> CsrMatrix {
            let mut state = seed;
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as usize
            };
            let mut data = Vec::new();
            let mut indices = Vec::new();
            let mut indptr = vec![0usize];
            for _ in 0..rows {
                let mut cs: Vec<usize> = (0..per_row).map(|_| next() % cols).collect();
                cs.sort_unstable();
                cs.dedup();
                for &c in &cs {
                    indices.push(c);
                    data.push(((next() % 1000) as f64) + 1.0);
                }
                indptr.push(indices.len());
            }
            CsrMatrix::from_components(Shape2D::new(rows, cols), data, indices, indptr, false)
                .expect("canonical csr")
        }

        let blocks: Vec<CsrMatrix> = (0..1500)
            .map(|i| canonical_csr(40, 40, 6, 0xABCD_0001 ^ (i as u64).wrapping_mul(0x9E37)))
            .collect();
        let refs: Vec<&CsrMatrix> = blocks.iter().collect();
        let total_nnz: usize = blocks.iter().map(CsrMatrix::nnz).sum();
        assert!(total_nnz >= 1 << 16, "too small to hit the parallel gate");

        let result = block_diag(&refs).expect("block_diag");

        // Independent block-diagonal reference.
        let mut ref_indptr = vec![0usize];
        let mut ref_indices = Vec::new();
        let mut ref_data = Vec::new();
        let mut col_off = 0usize;
        let mut acc = 0usize;
        for blk in &blocks {
            let bp = blk.indptr();
            for i in 0..blk.shape().rows {
                for idx in bp[i]..bp[i + 1] {
                    ref_indices.push(col_off + blk.indices()[idx]);
                    ref_data.push(blk.data()[idx]);
                    acc += 1;
                }
                ref_indptr.push(acc);
            }
            col_off += blk.shape().cols;
        }

        assert_eq!(result.indptr(), ref_indptr.as_slice(), "indptr mismatch");
        assert_eq!(result.indices(), ref_indices.as_slice(), "indices mismatch");
        assert!(
            result
                .data()
                .iter()
                .zip(&ref_data)
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "data not bit-identical to reference"
        );
        let meta = result.canonical_meta();
        assert!(meta.sorted_indices && meta.deduplicated);
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
        let meta = result.canonical_meta();
        assert!(meta.sorted_indices);
        assert!(meta.deduplicated);
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
    fn kron_preserves_duplicate_csr_semantics_on_fallback() {
        let a = CsrMatrix::from_components(
            Shape2D::new(1, 2),
            vec![2.0, 3.0],
            vec![1, 1],
            vec![0, 2],
            false,
        )
        .expect("duplicate csr");
        assert!(!a.canonical_meta().deduplicated);

        let b = eye(2).expect("eye");
        let result = kron(&a, &b).expect("kron");
        assert_eq!(result.shape(), Shape2D::new(2, 4));
        assert_eq!(result.nnz(), 2);
        assert_eq!(
            dense_from_csr(&result),
            vec![vec![0.0, 0.0, 5.0, 0.0], vec![0.0, 0.0, 0.0, 5.0]]
        );
    }

    #[test]
    fn kron_parallel_fill_is_byte_identical_to_serial_reference() {
        // Build canonical CSR operands large enough that `kron_canonical_csr`
        // exceeds the parallel-fill gate (total_nnz ≥ 64K, out_rows ≥ 1024): the
        // fanned disjoint-slice fill must reproduce the serial push order exactly.
        fn canonical_csr(rows: usize, cols: usize, per_row: usize, seed: u64) -> CsrMatrix {
            let mut state = seed;
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as usize
            };
            let mut data = Vec::new();
            let mut indices = Vec::new();
            let mut indptr = vec![0usize];
            for _ in 0..rows {
                // Distinct, sorted columns for this row.
                let mut cs: Vec<usize> = (0..per_row).map(|_| next() % cols).collect();
                cs.sort_unstable();
                cs.dedup();
                for &c in &cs {
                    indices.push(c);
                    data.push(((next() % 1000) as f64) + 1.0);
                }
                indptr.push(indices.len());
            }
            CsrMatrix::from_components(Shape2D::new(rows, cols), data, indices, indptr, false)
                .expect("canonical csr")
        }

        let a = canonical_csr(220, 60, 6, 0x1234_5678);
        let b = canonical_csr(40, 40, 6, 0x9abc_def0);
        // out_rows = 220*40 = 8800 ≥ 1024; nnz_a*nnz_b comfortably ≥ 64K.
        assert!(
            a.nnz() * b.nnz() >= 1 << 16,
            "operands too small to hit gate"
        );

        let result = kron(&a, &b).expect("kron");

        // Independent serial reference (naive triple loop, canonical by construction).
        let (a_ip, a_ix, a_d) = (a.indptr(), a.indices(), a.data());
        let (b_ip, b_ix, b_d) = (b.indptr(), b.indices(), b.data());
        let (b_rows, b_cols) = (b.shape().rows, b.shape().cols);
        let mut ref_indptr = vec![0usize];
        let mut ref_indices = Vec::new();
        let mut ref_data = Vec::new();
        for ai in 0..a.shape().rows {
            for bi in 0..b_rows {
                for a_idx in a_ip[ai]..a_ip[ai + 1] {
                    let col_base = a_ix[a_idx] * b_cols;
                    let a_val = a_d[a_idx];
                    for b_idx in b_ip[bi]..b_ip[bi + 1] {
                        ref_indices.push(col_base + b_ix[b_idx]);
                        ref_data.push(a_val * b_d[b_idx]);
                    }
                }
                ref_indptr.push(ref_indices.len());
            }
        }

        assert_eq!(result.indptr(), ref_indptr.as_slice(), "indptr mismatch");
        assert_eq!(result.indices(), ref_indices.as_slice(), "indices mismatch");
        // Byte-identical values (same multiply/emit order), not just approx-equal.
        assert!(
            result
                .data()
                .iter()
                .zip(&ref_data)
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "data not bit-identical to serial reference"
        );
        let meta = result.canonical_meta();
        assert!(meta.sorted_indices && meta.deduplicated);
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
