//! Same-process A/B + parity harness for spsolve routing.
//!
//! Before: spsolve densified any sparse A (n<=32768) into an n×n dense matrix and
//! ran O(n³) nalgebra dense LU. After: genuinely-sparse A routes to the native
//! sparse LU (~O(n·fill)). On a diagonally-dominant pentadiagonal system the fill
//! is O(n), so the sparse path is orders of magnitude cheaper. The solution is
//! unique, so x matches the dense path to rounding (PARITY block prints max|Δx|).
//! Run: `cargo run --profile release-perf -p fsci-sparse --bin perf_spsolve`.

use std::hint::black_box;
#[cfg(feature = "sparse-incumbent-bench")]
use std::io::Write;
use std::time::Instant;

use fsci_sparse::{
    CooMatrix, CscMatrix, CsrMatrix, FormatConvertible, LuOptions, PermutationOrdering, Shape2D,
    SolveOptions, splu, splu_solve, spsolve,
};
// Only the feature-gated wavefront profile calls this, so an ungated import is
// dead under default features and fails `clippy -D warnings`
// (frankenscipy-nwx8m). Gate the import to match its only use site.
#[cfg(feature = "sparse-incumbent-bench")]
use fsci_sparse::spsolve_triangular;
// Structural fast-path toggles, read only by `splu_profile_options` when
// FSCI_DISABLE_STRUCTURAL_FASTPATHS is set. Gated to match that single use site.
#[cfg(feature = "sparse-incumbent-bench")]
use fsci_sparse::linalg::{
    SPLU_CUBIC_SPECTRAL_DISABLE, SPLU_SUPERNODAL_ENABLE, SPLU_SUPERNODAL_FACTOR_HITS,
    SPSOLVE_CUBIC_SPECTRAL_DISABLE, SPSOLVE_PERIODIC_CUBOID_SPECTRAL_DISABLE,
};
use nalgebra::{DMatrix, DVector};
#[cfg(feature = "sparse-incumbent-bench")]
use sha2::{Digest, Sha256};

// This is deliberately exposed through `--source-marker`: RCH validations run
// both marker values before accepting a green test or performance result.
const PERF_SPSOLVE_SOURCE_MARKER: &str = "perf-spsolve-freshness-a";

// Pentadiagonal whose row/col labels are scrambled by a fixed pseudo-random symmetric
// permutation: same nnz (~5/row) but huge bandwidth in natural order, so natural-order
// sparse LU fills toward dense — while a fill-reducing reorder (RCM) recovers the band.
fn scattered_pentadiagonal(n: usize, seed: u64) -> CsrMatrix {
    // Fisher-Yates shuffle of 0..n with an LCG.
    let mut q: Vec<usize> = (0..n).collect();
    let mut s = seed;
    for i in (1..n).rev() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (s >> 11) as usize % (i + 1);
        q.swap(i, j);
    }
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for i in 0..n {
        for off in [-2i64, -1, 0, 1, 2] {
            let j = i as i64 + off;
            if j >= 0 && (j as usize) < n {
                rows.push(q[i]);
                cols.push(q[j as usize]);
                data.push(if off == 0 { 6.0 } else { -1.0 });
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

fn opts_with(ordering: PermutationOrdering) -> SolveOptions {
    SolveOptions {
        ordering,
        ..SolveOptions::default()
    }
}

// Diagonally-dominant banded matrix, half-bandwidth `hb` (2·hb+1 nnz/row). For hb>8 this
// exceeds the old nnz<=16n gate and used to densify to O(n³); the bandwidth gate now
// routes it to the sparse LU (fill bounded by the band).
fn banded(n: usize, hb: usize) -> CsrMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for i in 0..n {
        let lo = i.saturating_sub(hb);
        let hi = (i + hb).min(n - 1);
        for j in lo..=hi {
            rows.push(i);
            cols.push(j);
            data.push(if i == j { 2.0 * hb as f64 + 2.0 } else { -1.0 });
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// 2D 5-point Laplacian on a k×k grid (n=k²): the canonical fill-reduction benchmark.
// RCM keeps bandwidth ~k -> fill O(n·k)=O(n^1.5); minimum-degree/nested-dissection
// achieve O(n log n) fill. Diagonally dominant (diag 4+eps, neighbors -1) -> stable.
fn laplacian_2d(k: usize) -> CsrMatrix {
    let n = k * k;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    let idx = |r: usize, c: usize| r * k + c;
    for r in 0..k {
        for c in 0..k {
            let i = idx(r, c);
            rows.push(i);
            cols.push(i);
            data.push(4.001);
            for (dr, dc) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                let (nr, nc) = (r as i64 + dr, c as i64 + dc);
                if nr >= 0 && nr < k as i64 && nc >= 0 && nc < k as i64 {
                    rows.push(i);
                    cols.push(idx(nr as usize, nc as usize));
                    data.push(-1.0);
                }
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

/// Nonsymmetric five-point convection–diffusion operator used by the generic
/// sparse-LU structural profile. Differing west/east weights prevent every
/// symmetric tensor recognizer while the positive diagonal margin keeps the
/// solve deterministic and well conditioned enough for a strict residual gate.
#[cfg(feature = "sparse-incumbent-bench")]
fn convection_diffusion_2d(side: usize) -> CsrMatrix {
    const DIAGONAL: f64 = 4.001;
    const WEST: f64 = -1.2;
    const EAST: f64 = -0.8;
    const VERTICAL: f64 = -1.0;

    let n = side * side;
    let expected_nnz = 5 * n - 4 * side;
    let mut data = Vec::with_capacity(expected_nnz);
    let mut indices = Vec::with_capacity(expected_nnz);
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0);
    for row in 0..side {
        for column in 0..side {
            let index = row * side + column;
            if row > 0 {
                indices.push(index - side);
                data.push(VERTICAL);
            }
            if column > 0 {
                indices.push(index - 1);
                data.push(WEST);
            }
            indices.push(index);
            data.push(DIAGONAL);
            if column + 1 < side {
                indices.push(index + 1);
                data.push(EAST);
            }
            if row + 1 < side {
                indices.push(index + side);
                data.push(VERTICAL);
            }
            indptr.push(data.len());
        }
    }
    assert_eq!(data.len(), expected_nnz);
    CsrMatrix::from_components(Shape2D::new(n, n), data, indices, indptr, false)
        .expect("canonical nonsymmetric convection–diffusion CSR")
}

// Rectangular counterpart used by the profile-first widening campaign. The
// insertion order matches the square fixture above and produces canonical CSR.
fn laplacian_2d_rectangular(rows_count: usize, cols_count: usize) -> CsrMatrix {
    let n = rows_count * cols_count;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    let idx = |row: usize, col: usize| row * cols_count + col;
    for row in 0..rows_count {
        for col in 0..cols_count {
            let i = idx(row, col);
            rows.push(i);
            cols.push(i);
            data.push(4.001);
            for (row_delta, col_delta) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                let neighbor_row = row as i64 + row_delta;
                let neighbor_col = col as i64 + col_delta;
                if neighbor_row >= 0
                    && neighbor_row < rows_count as i64
                    && neighbor_col >= 0
                    && neighbor_col < cols_count as i64
                {
                    rows.push(i);
                    cols.push(idx(neighbor_row as usize, neighbor_col as usize));
                    data.push(-1.0);
                }
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// Cubic 3D 7-point Dirichlet operator used only by the profile-first tensor
// campaign. Coordinates are flattened z-major, then y, then contiguous x.
//
// Deliberately RETAINED while no call site is active (frankenscipy-nwx8m): this
// is a fixture spec for the tensor campaign, and deleting it to silence
// dead_code would throw away the operator definition rather than fix the lint.
#[allow(dead_code)]
fn laplacian_3d_cubic(side: usize) -> CsrMatrix {
    let n = side * side * side;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    let idx = |z: usize, y: usize, x: usize| (z * side + y) * side + x;
    for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                let i = idx(z, y, x);
                rows.push(i);
                cols.push(i);
                data.push(6.001);
                for (dz, dy, dx) in [
                    (-1i64, 0i64, 0i64),
                    (1, 0, 0),
                    (0, -1, 0),
                    (0, 1, 0),
                    (0, 0, -1),
                    (0, 0, 1),
                ] {
                    let neighbor_z = z as i64 + dz;
                    let neighbor_y = y as i64 + dy;
                    let neighbor_x = x as i64 + dx;
                    if neighbor_z >= 0
                        && neighbor_z < side as i64
                        && neighbor_y >= 0
                        && neighbor_y < side as i64
                        && neighbor_x >= 0
                        && neighbor_x < side as i64
                    {
                        rows.push(i);
                        cols.push(idx(
                            neighbor_z as usize,
                            neighbor_y as usize,
                            neighbor_x as usize,
                        ));
                        data.push(-1.0);
                    }
                }
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// Rectangular 3D 7-point Dirichlet operator used only by the profile-first
// cuboid campaign. Coordinates are flattened z-major, then y, then x.
//
// Retained for the same reason as `laplacian_3d_cubic` above
// (frankenscipy-nwx8m): a fixture spec, not dead weight.
#[allow(dead_code)]
fn laplacian_3d_cuboid(x_extent: usize, y_extent: usize, z_extent: usize) -> CsrMatrix {
    let plane = x_extent * y_extent;
    let n = plane * z_extent;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    let idx = |z: usize, y: usize, x: usize| (z * y_extent + y) * x_extent + x;
    for z in 0..z_extent {
        for y in 0..y_extent {
            for x in 0..x_extent {
                let i = idx(z, y, x);
                rows.push(i);
                cols.push(i);
                data.push(6.001);
                for (dz, dy, dx) in [
                    (-1i64, 0i64, 0i64),
                    (1, 0, 0),
                    (0, -1, 0),
                    (0, 1, 0),
                    (0, 0, -1),
                    (0, 0, 1),
                ] {
                    let neighbor_z = z as i64 + dz;
                    let neighbor_y = y as i64 + dy;
                    let neighbor_x = x as i64 + dx;
                    if neighbor_z >= 0
                        && neighbor_z < z_extent as i64
                        && neighbor_y >= 0
                        && neighbor_y < y_extent as i64
                        && neighbor_x >= 0
                        && neighbor_x < x_extent as i64
                    {
                        rows.push(i);
                        cols.push(idx(
                            neighbor_z as usize,
                            neighbor_y as usize,
                            neighbor_x as usize,
                        ));
                        data.push(-1.0);
                    }
                }
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// Shifted 3D graph-Laplacian cube used only by the profile-first Neumann
// factor campaign. Boundary diagonals equal the number of incident edges plus
// the positive shift, so this topology is deliberately rejected by the kept
// constant-diagonal Dirichlet recognizer.
#[cfg(feature = "sparse-incumbent-bench")]
fn laplacian_3d_neumann_cubic(side: usize, shift: f64) -> CsrMatrix {
    let n = side * side * side;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    let idx = |z: usize, y: usize, x: usize| (z * side + y) * side + x;
    for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                let i = idx(z, y, x);
                let degree = usize::from(z > 0)
                    + usize::from(z + 1 < side)
                    + usize::from(y > 0)
                    + usize::from(y + 1 < side)
                    + usize::from(x > 0)
                    + usize::from(x + 1 < side);
                rows.push(i);
                cols.push(i);
                data.push(shift + degree as f64);
                for (dz, dy, dx) in [
                    (-1i64, 0i64, 0i64),
                    (1, 0, 0),
                    (0, -1, 0),
                    (0, 1, 0),
                    (0, 0, -1),
                    (0, 0, 1),
                ] {
                    let neighbor_z = z as i64 + dz;
                    let neighbor_y = y as i64 + dy;
                    let neighbor_x = x as i64 + dx;
                    if neighbor_z >= 0
                        && neighbor_z < side as i64
                        && neighbor_y >= 0
                        && neighbor_y < side as i64
                        && neighbor_x >= 0
                        && neighbor_x < side as i64
                    {
                        rows.push(i);
                        cols.push(idx(
                            neighbor_z as usize,
                            neighbor_y as usize,
                            neighbor_x as usize,
                        ));
                        data.push(-1.0);
                    }
                }
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// Shifted anisotropic graph-Laplacian cuboid used only by the profile-first
// widening campaign. Each boundary diagonal is the shift plus the magnitudes
// of its incident, axis-specific edge weights.
#[cfg(feature = "sparse-incumbent-bench")]
fn laplacian_3d_neumann_cuboid(
    x_extent: usize,
    y_extent: usize,
    z_extent: usize,
    shift: f64,
    x_weight: f64,
    y_weight: f64,
    z_weight: f64,
) -> CsrMatrix {
    let plane = x_extent * y_extent;
    let n = plane * z_extent;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    let idx = |z: usize, y: usize, x: usize| (z * y_extent + y) * x_extent + x;
    for z in 0..z_extent {
        for y in 0..y_extent {
            for x in 0..x_extent {
                let row = idx(z, y, x);
                let diagonal = shift
                    - x_weight * (usize::from(x > 0) + usize::from(x + 1 < x_extent)) as f64
                    - y_weight * (usize::from(y > 0) + usize::from(y + 1 < y_extent)) as f64
                    - z_weight * (usize::from(z > 0) + usize::from(z + 1 < z_extent)) as f64;
                rows.push(row);
                cols.push(row);
                data.push(diagonal);
                for (neighbor_z, neighbor_y, neighbor_x, weight) in [
                    (z.checked_sub(1), Some(y), Some(x), z_weight),
                    (
                        (z + 1 < z_extent).then_some(z + 1),
                        Some(y),
                        Some(x),
                        z_weight,
                    ),
                    (Some(z), y.checked_sub(1), Some(x), y_weight),
                    (
                        Some(z),
                        (y + 1 < y_extent).then_some(y + 1),
                        Some(x),
                        y_weight,
                    ),
                    (Some(z), Some(y), x.checked_sub(1), x_weight),
                    (
                        Some(z),
                        Some(y),
                        (x + 1 < x_extent).then_some(x + 1),
                        x_weight,
                    ),
                ] {
                    if let (Some(neighbor_z), Some(neighbor_y), Some(neighbor_x)) =
                        (neighbor_z, neighbor_y, neighbor_x)
                    {
                        rows.push(row);
                        cols.push(idx(neighbor_z, neighbor_y, neighbor_x));
                        data.push(weight);
                    }
                }
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// Shifted anisotropic graph-Laplacian on a 3-D torus, used only by the
// profile-first periodic factor campaign. Every row has two wrapped neighbors
// per axis and therefore the same weighted degree.
#[cfg(feature = "sparse-incumbent-bench")]
fn laplacian_3d_periodic_cuboid(
    x_extent: usize,
    y_extent: usize,
    z_extent: usize,
    shift: f64,
    x_weight: f64,
    y_weight: f64,
    z_weight: f64,
) -> CsrMatrix {
    let plane = x_extent * y_extent;
    let n = plane * z_extent;
    let mut rows = Vec::with_capacity(7 * n);
    let mut cols = Vec::with_capacity(7 * n);
    let mut data = Vec::with_capacity(7 * n);
    let idx = |z: usize, y: usize, x: usize| (z * y_extent + y) * x_extent + x;
    let diagonal = shift - 2.0 * (x_weight + y_weight + z_weight);
    for z in 0..z_extent {
        for y in 0..y_extent {
            for x in 0..x_extent {
                let row = idx(z, y, x);
                rows.push(row);
                cols.push(row);
                data.push(diagonal);
                for (neighbor_z, neighbor_y, neighbor_x, weight) in [
                    ((z + z_extent - 1) % z_extent, y, x, z_weight),
                    ((z + 1) % z_extent, y, x, z_weight),
                    (z, (y + y_extent - 1) % y_extent, x, y_weight),
                    (z, (y + 1) % y_extent, x, y_weight),
                    (z, y, (x + x_extent - 1) % x_extent, x_weight),
                    (z, y, (x + 1) % x_extent, x_weight),
                ] {
                    rows.push(row);
                    cols.push(idx(neighbor_z, neighbor_y, neighbor_x));
                    data.push(weight);
                }
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// Arrowhead: diagonal + a dense hub row/col through node 0. nnz ~= 3n. Eliminating the
// hub early (natural/RCM, which can't isolate it) fills the whole trailing block O(n²);
// minimum-degree eliminates the degree-1 spokes first (no fill) and the hub last (no
// fill) -> O(n). The showcase where min-degree crushes bandwidth ordering.
fn arrowhead(n: usize) -> CsrMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        data.push(n as f64 + 4.0); // strong diagonal -> diagonally dominant, stable
        if i != 0 {
            rows.push(0);
            cols.push(i);
            data.push(-1.0);
            rows.push(i);
            cols.push(0);
            data.push(-1.0);
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// Diagonally-dominant pentadiagonal A (bandwidth 2): diag 6, off-diagonals -1 at
// ±1, ±2. nnz/row ~= 5, so a.nnz() <= 16n -> routes to the native sparse LU.
fn pentadiagonal(n: usize) -> CsrMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for i in 0..n {
        for off in [-2i64, -1, 0, 1, 2] {
            let j = i as i64 + off;
            if j >= 0 && (j as usize) < n {
                rows.push(i);
                cols.push(j as usize);
                data.push(if off == 0 { 6.0 } else { -1.0 });
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// Verbatim of the OLD dense path: densify the CSR and solve with nalgebra LU.
fn dense_solve_baseline(a: &CsrMatrix, b: &[f64]) -> Vec<f64> {
    let n = a.shape().rows;
    let mut dense = vec![0.0f64; n * n];
    let indptr = a.indptr();
    let indices = a.indices();
    let data = a.data();
    for i in 0..n {
        for idx in indptr[i]..indptr[i + 1] {
            dense[i * n + indices[idx]] = data[idx];
        }
    }
    let matrix = DMatrix::from_row_slice(n, n, &dense);
    let rhs = DVector::from_column_slice(b);
    let x = matrix.lu().solve(&rhs).expect("dense lu");
    x.iter().copied().collect()
}

fn time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / reps as f64
}

fn profile_rectangular_rust(repetitions: usize) {
    let rows = 32usize;
    let cols = 128usize;
    let n = rows * cols;
    let matrix = laplacian_2d_rectangular(rows, cols);
    let rhs: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();
    let warm = spsolve(&matrix, &rhs, SolveOptions::default()).expect("rectangular warmup");
    let mut checksum = warm.solution.iter().sum::<f64>();
    let started = Instant::now();
    for _ in 0..repetitions {
        let solved = spsolve(black_box(&matrix), black_box(&rhs), SolveOptions::default())
            .expect("rectangular profile solve");
        checksum += black_box(solved.solution[n / 2]);
    }
    println!(
        "RECTANGULAR_PROFILE rows={rows} cols={cols} n={n} nnz={} repetitions={repetitions} elapsed_seconds={:.9} checksum={checksum:.17e}",
        matrix.nnz(),
        started.elapsed().as_secs_f64(),
    );
}

#[cfg(feature = "sparse-incumbent-bench")]
fn cubic_fixture_sha256(matrix: &CsrMatrix, rhs: &[f64]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((matrix.shape().rows as u64).to_le_bytes());
    hasher.update((matrix.nnz() as u64).to_le_bytes());
    for &value in matrix.data() {
        hasher.update(value.to_le_bytes());
    }
    for &index in matrix.indices() {
        hasher.update((index as u64).to_le_bytes());
    }
    for &pointer in matrix.indptr() {
        hasher.update((pointer as u64).to_le_bytes());
    }
    for &value in rhs {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(feature = "sparse-incumbent-bench")]
fn triangular_wavefront_fixture(levels: usize, width: usize) -> (CsrMatrix, Vec<f64>, Vec<f64>) {
    assert!(levels > 1, "wavefront must have multiple levels");
    assert!(width > 1, "wavefront levels must have multiple rows");
    let n = levels.checked_mul(width).expect("wavefront dimension");
    let previous_level_nnz = width
        .checked_mul(4)
        .and_then(|value| value.checked_sub(2))
        .expect("wavefront level nnz");
    let expected_nnz = width
        .checked_add(
            (levels - 1)
                .checked_mul(previous_level_nnz)
                .expect("wavefront nnz"),
        )
        .expect("wavefront nnz");
    let mut data = Vec::with_capacity(expected_nnz);
    let mut indices = Vec::with_capacity(expected_nnz);
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0);
    for level in 0..levels {
        let base = level * width;
        let previous = base.saturating_sub(width);
        for lane in 0..width {
            let row = base + lane;
            if level > 0 {
                if lane > 0 {
                    indices.push(previous + lane - 1);
                    data.push(-0.125);
                }
                indices.push(previous + lane);
                data.push(-0.5);
                if lane + 1 < width {
                    indices.push(previous + lane + 1);
                    data.push(-0.125);
                }
            }
            indices.push(row);
            data.push(2.0);
            indptr.push(data.len());
        }
    }
    assert_eq!(data.len(), expected_nnz);
    let matrix = CsrMatrix::from_components(Shape2D::new(n, n), data, indices, indptr, false)
        .expect("canonical triangular wavefront CSR");
    let expected = (0..n)
        .map(|index| 1.0 + 0.03125 * ((17 * index) % 29) as f64)
        .collect::<Vec<_>>();
    let mut rhs = vec![0.0; n];
    for (row, output) in rhs.iter_mut().enumerate() {
        let mut sum = 0.0;
        for entry in matrix.indptr()[row]..matrix.indptr()[row + 1] {
            sum += matrix.data()[entry] * expected[matrix.indices()[entry]];
        }
        *output = sum;
    }
    (matrix, expected, rhs)
}

#[cfg(feature = "sparse-incumbent-bench")]
fn relative_triangular_residual(matrix: &CsrMatrix, rhs: &[f64], solution: &[f64]) -> f64 {
    let mut residual_squared = 0.0;
    let mut rhs_squared = 0.0;
    for (row, &rhs_row) in rhs.iter().enumerate().take(matrix.shape().rows) {
        let mut product = 0.0;
        for entry in matrix.indptr()[row]..matrix.indptr()[row + 1] {
            product += matrix.data()[entry] * solution[matrix.indices()[entry]];
        }
        residual_squared += (rhs_row - product).powi(2);
        rhs_squared += rhs_row.powi(2);
    }
    residual_squared.sqrt() / rhs_squared.sqrt()
}

#[cfg(feature = "sparse-incumbent-bench")]
fn relative_triangular_l2(actual: &[f64], expected: &[f64]) -> f64 {
    let mut difference_squared = 0.0;
    let mut expected_squared = 0.0;
    for (&actual_value, &expected_value) in actual.iter().zip(expected) {
        difference_squared += (actual_value - expected_value).powi(2);
        expected_squared += expected_value * expected_value;
    }
    difference_squared.sqrt() / expected_squared.sqrt()
}

#[cfg(feature = "sparse-incumbent-bench")]
fn profile_observed_os_threads() -> usize {
    std::fs::read_dir("/proc/self/task")
        .expect("read /proc/self/task for profile provenance")
        .count()
}

#[cfg(feature = "sparse-incumbent-bench")]
fn profile_triangular_wavefront_rust(repetitions: usize, levels: usize, width: usize) {
    let (matrix, expected, rhs) = triangular_wavefront_fixture(levels, width);
    let n = matrix.shape().rows;
    let warm = spsolve_triangular(&matrix, &rhs, true).expect("triangular wavefront warmup");
    let max_abs_error = warm
        .iter()
        .zip(&expected)
        .map(|(actual, target)| (actual - target).abs())
        .fold(0.0_f64, f64::max);
    let residual = relative_triangular_residual(&matrix, &rhs, &warm);
    let relative_l2 = relative_triangular_l2(&warm, &expected);
    let mut maximum_threads = profile_observed_os_threads();
    let mut checksum = warm
        .iter()
        .fold(0_u64, |accumulator, value| accumulator ^ value.to_bits());
    let started = Instant::now();
    for _ in 0..repetitions {
        let solution = spsolve_triangular(black_box(&matrix), black_box(&rhs), true)
            .expect("triangular wavefront profile solve");
        checksum ^= black_box(
            solution
                .iter()
                .fold(0_u64, |accumulator, value| accumulator ^ value.to_bits()),
        );
        maximum_threads = maximum_threads.max(profile_observed_os_threads());
    }
    println!(
        "TRIANGULAR_WAVEFRONT_PROFILE levels={levels} width={width} n={n} nnz={} \
         repetitions={repetitions} elapsed_seconds={:.9} checksum={checksum} \
         max_abs_error={max_abs_error:.17e} residual={residual:.17e} \
         relative_l2={relative_l2:.17e} actual_observed_worker_threads={maximum_threads} \
         input_sha256={}",
        matrix.nnz(),
        started.elapsed().as_secs_f64(),
        cubic_fixture_sha256(&matrix, &rhs),
    );
}

#[cfg(feature = "sparse-incumbent-bench")]
fn profile_cubic_rust(repetitions: usize, side: usize) {
    let n = side * side * side;
    let matrix = laplacian_3d_cubic(side);
    let rhs: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();
    let warm = spsolve(&matrix, &rhs, SolveOptions::default()).expect("cubic warmup");
    let mut checksum = warm.solution.iter().sum::<f64>();
    let started = Instant::now();
    for _ in 0..repetitions {
        let solved = spsolve(black_box(&matrix), black_box(&rhs), SolveOptions::default())
            .expect("cubic profile solve");
        checksum += black_box(solved.solution[n / 2]);
    }
    println!(
        "CUBIC_PROFILE side={side} n={n} nnz={} repetitions={repetitions} elapsed_seconds={:.9} checksum={checksum:.17e} input_sha256={}",
        matrix.nnz(),
        started.elapsed().as_secs_f64(),
        cubic_fixture_sha256(&matrix, &rhs),
    );
}

#[cfg(feature = "sparse-incumbent-bench")]
fn profile_cuboid_rust(repetitions: usize, x_extent: usize, y_extent: usize, z_extent: usize) {
    let n = x_extent * y_extent * z_extent;
    let matrix = laplacian_3d_cuboid(x_extent, y_extent, z_extent);
    let rhs: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();
    let warm = spsolve(&matrix, &rhs, SolveOptions::default()).expect("cuboid warmup");
    let mut checksum = warm.solution.iter().sum::<f64>();
    let started = Instant::now();
    for _ in 0..repetitions {
        let solved = spsolve(black_box(&matrix), black_box(&rhs), SolveOptions::default())
            .expect("cuboid profile solve");
        checksum += black_box(solved.solution[n / 2]);
    }
    println!(
        "CUBOID_PROFILE x={x_extent} y={y_extent} z={z_extent} n={n} nnz={} repetitions={repetitions} elapsed_seconds={:.9} checksum={checksum:.17e} input_sha256={}",
        matrix.nnz(),
        started.elapsed().as_secs_f64(),
        cubic_fixture_sha256(&matrix, &rhs),
    );
}

#[cfg(feature = "sparse-incumbent-bench")]
fn cubic_splu_rhs(n: usize, rhs_index: usize) -> Vec<f64> {
    (0..n)
        .map(|index| 1.0 + 0.125 * ((17 * index + 23 * rhs_index) % 29) as f64)
        .collect()
}

#[cfg(feature = "sparse-incumbent-bench")]
fn cubic_splu_fixture_sha256(matrix: &CscMatrix, right_hand_sides: &[Vec<f64>]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((matrix.shape().rows as u64).to_le_bytes());
    hasher.update((matrix.nnz() as u64).to_le_bytes());
    for &value in matrix.data() {
        hasher.update(value.to_le_bytes());
    }
    for &index in matrix.indices() {
        hasher.update((index as u64).to_le_bytes());
    }
    for &pointer in matrix.indptr() {
        hasher.update((pointer as u64).to_le_bytes());
    }
    for rhs in right_hand_sides {
        for &value in rhs {
            hasher.update(value.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

/// One fsci-only factorization, one fixture, one ordering -- the sweep behind frankenscipy-run7d.1.
///
/// WHY IT EXISTS. AMD is the better arm on run7d's convection cell and the worse arm on llywn's
/// cubic cell, and the losing cell holds strictly LESS fill, so fill is refuted as the predictor.
/// Deciding whether AMD can ever be the default needs INSTRUCTIONS across a spread of shapes, and
/// `perf stat -e instructions` is load-independent and runs at native speed -- the only kind of
/// measurement this host has reliably supported.
///
/// IT PRINTS THE BACKEND THAT ACTUALLY RAN, which is not decoration. Since the structural fast
/// paths now accept `Amd` as well as `Colamd`, a fixture like the Dirichlet cubic Laplacian takes
/// the SPECTRAL route under `amd` and the general LU under `rcm`. Comparing those two would be
/// comparing different algorithms and would read as a spectacular AMD win. The driver must check
/// that both arms report `NativeSparseLu` before comparing them.
#[cfg(feature = "sparse-incumbent-bench")]
fn profile_ordering_sweep(fixture: &str, size: usize, repetitions: usize) {
    let matrix = match fixture {
        "convection" => convection_diffusion_2d(size),
        "cubic" => laplacian_3d_cubic(size),
        "lap2d" => laplacian_2d(size),
        "arrowhead" => arrowhead(size),
        // rows x 1 column IS a tridiagonal matrix: the MUST-MISS shape, where no ordering can
        // beat the natural one and an "AMD wins" reading would mean the sweep is broken.
        "tridiag" => laplacian_2d_rectangular(size, 1),
        "pentadiag" => scattered_pentadiagonal(size, 0x5eed_1234),
        other => panic!("unknown sweep fixture {other:?}"),
    };
    let n = matrix.shape().rows;
    let csc = matrix.to_csc().expect("sweep CSC");
    let rhs: Vec<f64> = (0..n)
        .map(|index| 1.0 + 0.125 * ((17 * index + 23) % 29) as f64)
        .collect();

    // Warm once outside the timed/counted region so allocator growth is not attributed.
    let warm = splu(&csc, splu_profile_options()).expect("sweep warmup");
    let solution = splu_solve(&warm, &rhs).expect("sweep warm solve");
    let residual = splu_max_relative_residual(
        &matrix,
        std::slice::from_ref(&rhs),
        std::slice::from_ref(&solution),
    );
    let fill = fsci_sparse::linalg::splu_factor_payload_bytes(&warm);

    let started = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..repetitions {
        let factor = splu(black_box(&csc), splu_profile_options()).expect("sweep factor");
        checksum += black_box(fsci_sparse::linalg::splu_factor_payload_bytes(&factor)) as f64;
    }
    let elapsed = started.elapsed().as_secs_f64();

    println!(
        "ORDERING_SWEEP fixture={fixture} size={size} n={n} nnz={} \
         ordering_used={:?} backend_used={:?} factor_payload_bytes={fill} \
         max_relative_residual={residual:.6e} repetitions={repetitions} \
         elapsed_seconds={elapsed:.9} checksum={checksum:.6e}",
        matrix.nnz(),
        warm.ordering_used,
        warm.backend_used,
    );
}

#[cfg(feature = "sparse-incumbent-bench")]
fn profile_cubic_splu_rust(repetitions: usize, side: usize, rhs_count: usize) {
    let n = side * side * side;
    let matrix = laplacian_3d_cubic(side).to_csc().expect("cubic CSC");
    let right_hand_sides = (0..rhs_count)
        .map(|rhs_index| cubic_splu_rhs(n, rhs_index))
        .collect::<Vec<_>>();

    let warm_factor = splu(&matrix, splu_profile_options()).expect("cubic splu warmup");
    let mut checksum = 0.0;
    for rhs in &right_hand_sides {
        let solution = splu_solve(&warm_factor, rhs).expect("cubic splu warmup solve");
        checksum += black_box(solution[n / 2]);
    }

    let started = Instant::now();
    for _ in 0..repetitions {
        let factor = splu(black_box(&matrix), splu_profile_options()).expect("cubic splu profile");
        for rhs in &right_hand_sides {
            let solution =
                splu_solve(black_box(&factor), black_box(rhs)).expect("cubic splu profile solve");
            checksum += black_box(solution[n / 2]);
        }
    }
    println!(
        "CUBIC_SPLU_PROFILE side={side} n={n} nnz={} rhs_count={rhs_count} repetitions={repetitions} elapsed_seconds={:.9} checksum={checksum:.17e} input_sha256={} {}",
        matrix.nnz(),
        started.elapsed().as_secs_f64(),
        cubic_splu_fixture_sha256(&matrix, &right_hand_sides),
        supernodal_arm_status(),
    );
}

/// Apply `FSCI_DISABLE_STRUCTURAL_FASTPATHS`, forcing the general sparse LU.
///
/// SEPARATE FROM `splu_profile_options` BECAUSE NOT EVERY PROFILE BUILDS `LuOptions`. The periodic
/// cuboid profile calls `spsolve` with `SolveOptions::default()` and never touches the ordering
/// builder, so while this logic lived inside that builder the variable was INERT there: both arms
/// of a supposed route A/B ran identical code and reported instruction counts within 0.03% of
/// each other. That is a dead A/B, and it reads as "the route makes no difference" rather than as
/// a broken switch. Every profile that can reach a structural fast path must call this, and every
/// such profile prints the backend it actually used so the failure is visible on the row.
///
/// Off by default, so every arm runs exactly as it ships unless the variable is set.
#[cfg(feature = "sparse-incumbent-bench")]
fn apply_structural_fastpath_env() {
    if matches!(
        std::env::var("FSCI_DISABLE_STRUCTURAL_FASTPATHS")
            .ok()
            .as_deref(),
        Some("1") | Some("true")
    ) {
        use std::sync::atomic::Ordering as AtomicOrdering;
        SPLU_CUBIC_SPECTRAL_DISABLE.store(true, AtomicOrdering::Relaxed);
        SPSOLVE_CUBIC_SPECTRAL_DISABLE.store(true, AtomicOrdering::Relaxed);
        SPSOLVE_PERIODIC_CUBOID_SPECTRAL_DISABLE.store(true, AtomicOrdering::Relaxed);
    }
}

/// `FSCI_SPLU_SUPERNODAL=1` routes the splu profiles through the supernodal arm
/// (frankenscipy-bfk5l).
///
/// WHY A PROFILE NEEDS TO REACH THIS. `SPLU_SUPERNODAL_ENABLE` was driven only by in-crate tests,
/// so the census counted it as exercised while no PROFILE could select it -- and the question it
/// answers is a profiling question. hpx50 left llywn's packing ceiling as a range, 1.20-fold to
/// 1.38-fold, and the entire spread is whether a supernodal symbolic pass absorbs
/// `matched_run_length` (19.26% of the cubic cell's factorization, run PER UPDATE) or whether that
/// compare survives per update. Profiling this path and reading that function's share off it
/// decides between the two.
///
/// THE ARM'S KNOWN SLOWNESS IS NOT THE POINT. It measured 5.77-fold worse overall on this cell.
/// The question here is STRUCTURAL -- which functions the path executes -- and a path can be
/// slower in total while still showing that the per-update pattern compare is unnecessary. Do not
/// read a total-cost number off a profile taken through this switch.
///
/// Off by default, so every arm runs exactly as it ships unless the variable is set. The profile
/// prints whether the arm actually planned or declined, because `factorize_csr_supernodal` returns
/// `None` and falls through whenever the plan cannot be trusted -- and a silent fall-through would
/// make this switch look inert exactly the way `FSCI_DISABLE_STRUCTURAL_FASTPATHS` once did.
#[cfg(feature = "sparse-incumbent-bench")]
fn apply_supernodal_env() {
    if matches!(
        std::env::var("FSCI_SPLU_SUPERNODAL").ok().as_deref(),
        Some("1") | Some("true")
    ) {
        SPLU_SUPERNODAL_ENABLE.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Did the supernodal arm actually run, or did it plan-and-decline? Printed by the profiles that
/// can reach it, so the switch cannot read as inert when it merely fell through.
#[cfg(feature = "sparse-incumbent-bench")]
fn supernodal_arm_status() -> String {
    let requested = matches!(
        std::env::var("FSCI_SPLU_SUPERNODAL").ok().as_deref(),
        Some("1") | Some("true")
    );
    let hits = SPLU_SUPERNODAL_FACTOR_HITS.load(std::sync::atomic::Ordering::Relaxed);
    format!("supernodal_requested={requested} supernodal_factor_hits={hits}")
}

/// Ordering for the fsci-only splu profiles (convection and cubic).
///
/// It used to be hardcoded to `LuOptions::default()` while the live arm read
/// `FSCI_SPLU_ORDERING`, so the cheap profile and the row it is supposed to pre-cost were
/// silently measuring different configurations. Reading the same variable is the whole fix.
#[cfg(feature = "sparse-incumbent-bench")]
fn splu_profile_options() -> LuOptions {
    apply_structural_fastpath_env();
    apply_supernodal_env();
    let ordering = match std::env::var("FSCI_SPLU_ORDERING").ok().as_deref() {
        None | Some("") | Some("default") => LuOptions::default().ordering,
        Some("colamd") => PermutationOrdering::Colamd,
        Some("rcm") => PermutationOrdering::ReverseCuthillMcKee,
        Some("mmd-ata") => PermutationOrdering::MmdAta,
        Some("mmd-at-plus-a") => PermutationOrdering::MmdAtPlusA,
        Some("amd") => PermutationOrdering::Amd,
        Some("natural") => PermutationOrdering::Natural,
        Some(other) => panic!("FSCI_SPLU_ORDERING={other:?} is not a known ordering"),
    };
    LuOptions {
        ordering,
        ..LuOptions::default()
    }
}

#[cfg(feature = "sparse-incumbent-bench")]
fn profile_convection_splu_rust(
    repetitions: usize,
    side: usize,
    rhs_count: usize,
    output_path: Option<&str>,
) {
    let n = side * side;
    let matrix_csr = convection_diffusion_2d(side);
    let matrix = matrix_csr.to_csc().expect("convection–diffusion CSC");
    let right_hand_sides = (0..rhs_count)
        .map(|rhs_index| cubic_splu_rhs(n, rhs_index))
        .collect::<Vec<_>>();

    let warm_factor =
        splu(&matrix, splu_profile_options()).expect("convection–diffusion splu warmup");
    let warm_solutions = right_hand_sides
        .iter()
        .map(|rhs| splu_solve(&warm_factor, rhs).expect("convection–diffusion splu warmup solve"))
        .collect::<Vec<_>>();
    let maximum_residual =
        splu_max_relative_residual(&matrix_csr, &right_hand_sides, &warm_solutions);
    let mut checksum = warm_solutions
        .iter()
        .map(|solution| solution[n / 2])
        .sum::<f64>();
    let mut maximum_threads = profile_observed_os_threads();

    let started = Instant::now();
    for _ in 0..repetitions {
        let factor = splu(black_box(&matrix), splu_profile_options())
            .expect("convection–diffusion splu profile");
        for rhs in &right_hand_sides {
            let solution = splu_solve(black_box(&factor), black_box(rhs))
                .expect("convection–diffusion splu profile solve");
            checksum += black_box(solution[n / 2]);
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    maximum_threads = maximum_threads.max(profile_observed_os_threads());

    let mut output_bytes = Vec::with_capacity(rhs_count * n * std::mem::size_of::<f64>());
    for solution in &warm_solutions {
        for &value in solution {
            output_bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let output_sha256 = format!("{:x}", Sha256::digest(&output_bytes));
    if let Some(path) = output_path {
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("create new convection–diffusion solution artifact");
        output
            .write_all(&output_bytes)
            .expect("write convection–diffusion solution artifact");
    }

    println!(
        "CONVECTION_SPLU_PROFILE side={side} diagonal=4.001 west=-1.2 east=-0.8 \
         vertical=-1.0 n={n} nnz={} rhs_count={rhs_count} repetitions={repetitions} \
         elapsed_seconds={elapsed:.9} checksum={checksum:.17e} \
         max_residual={maximum_residual:.17e} \
         actual_observed_worker_threads={maximum_threads} input_sha256={} \
         output_sha256={output_sha256}",
        matrix.nnz(),
        cubic_splu_fixture_sha256(&matrix, &right_hand_sides),
    );
}

#[cfg(feature = "sparse-incumbent-bench")]
fn profile_neumann_cubic_splu_rust(repetitions: usize, side: usize, rhs_count: usize) {
    let shift = 1.0e-3;
    let n = side * side * side;
    let matrix = laplacian_3d_neumann_cubic(side, shift)
        .to_csc()
        .expect("shifted-Neumann cubic CSC");
    let right_hand_sides = (0..rhs_count)
        .map(|rhs_index| cubic_splu_rhs(n, rhs_index))
        .collect::<Vec<_>>();

    let warm_factor = splu(&matrix, LuOptions::default()).expect("Neumann splu warmup");
    let mut checksum = 0.0;
    for rhs in &right_hand_sides {
        let solution = splu_solve(&warm_factor, rhs).expect("Neumann splu warmup solve");
        checksum += black_box(solution[n / 2]);
    }

    let started = Instant::now();
    for _ in 0..repetitions {
        let factor = splu(black_box(&matrix), LuOptions::default()).expect("Neumann splu profile");
        for rhs in &right_hand_sides {
            let solution =
                splu_solve(black_box(&factor), black_box(rhs)).expect("Neumann splu profile solve");
            checksum += black_box(solution[n / 2]);
        }
    }
    println!(
        "NEUMANN_CUBIC_SPLU_PROFILE side={side} shift={shift:.17e} n={n} nnz={} rhs_count={rhs_count} repetitions={repetitions} elapsed_seconds={:.9} checksum={checksum:.17e} input_sha256={}",
        matrix.nnz(),
        started.elapsed().as_secs_f64(),
        cubic_splu_fixture_sha256(&matrix, &right_hand_sides),
    );
}

#[cfg(feature = "sparse-incumbent-bench")]
fn splu_max_relative_residual(
    matrix: &CsrMatrix,
    right_hand_sides: &[Vec<f64>],
    solutions: &[Vec<f64>],
) -> f64 {
    right_hand_sides
        .iter()
        .zip(solutions)
        .map(|(rhs, solution)| {
            let mut residual_squared = 0.0;
            let mut rhs_squared = 0.0;
            for (row, &rhs_row) in rhs.iter().enumerate().take(matrix.shape().rows) {
                let mut product = 0.0;
                for entry in matrix.indptr()[row]..matrix.indptr()[row + 1] {
                    product += matrix.data()[entry] * solution[matrix.indices()[entry]];
                }
                residual_squared += (rhs_row - product).powi(2);
                rhs_squared += rhs_row * rhs_row;
            }
            residual_squared.sqrt() / rhs_squared.sqrt()
        })
        .fold(0.0_f64, f64::max)
}

#[cfg(feature = "sparse-incumbent-bench")]
fn profile_neumann_cuboid_splu_rust(
    repetitions: usize,
    x_extent: usize,
    y_extent: usize,
    z_extent: usize,
    rhs_count: usize,
    output_path: Option<&str>,
) {
    let shift = 1.0e-3;
    let x_weight = -0.75;
    let y_weight = -1.0;
    let z_weight = -1.25;
    let n = x_extent * y_extent * z_extent;
    let matrix_csr = laplacian_3d_neumann_cuboid(
        x_extent, y_extent, z_extent, shift, x_weight, y_weight, z_weight,
    );
    let matrix = matrix_csr.to_csc().expect("shifted-Neumann cuboid CSC");
    let right_hand_sides = (0..rhs_count)
        .map(|rhs_index| cubic_splu_rhs(n, rhs_index))
        .collect::<Vec<_>>();

    let warm_factor = splu(&matrix, LuOptions::default()).expect("Neumann cuboid splu warmup");
    let warm_solutions = right_hand_sides
        .iter()
        .map(|rhs| splu_solve(&warm_factor, rhs).expect("Neumann cuboid splu warmup solve"))
        .collect::<Vec<_>>();
    let maximum_residual =
        splu_max_relative_residual(&matrix_csr, &right_hand_sides, &warm_solutions);
    let mut checksum = warm_solutions
        .iter()
        .map(|solution| solution[n / 2])
        .sum::<f64>();
    let mut maximum_threads = profile_observed_os_threads();

    let started = Instant::now();
    for _ in 0..repetitions {
        let factor =
            splu(black_box(&matrix), LuOptions::default()).expect("Neumann cuboid splu profile");
        for rhs in &right_hand_sides {
            let solution = splu_solve(black_box(&factor), black_box(rhs))
                .expect("Neumann cuboid splu profile solve");
            checksum += black_box(solution[n / 2]);
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    maximum_threads = maximum_threads.max(profile_observed_os_threads());

    let mut output_bytes = Vec::with_capacity(rhs_count * n * std::mem::size_of::<f64>());
    for solution in &warm_solutions {
        for &value in solution {
            output_bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let output_sha256 = format!("{:x}", Sha256::digest(&output_bytes));
    if let Some(path) = output_path {
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("create new Neumann cuboid solution artifact");
        output
            .write_all(&output_bytes)
            .expect("write Neumann cuboid solution artifact");
    }

    println!(
        "NEUMANN_CUBOID_SPLU_PROFILE x={x_extent} y={y_extent} z={z_extent} \
         x_weight={x_weight:.17e} y_weight={y_weight:.17e} z_weight={z_weight:.17e} \
         shift={shift:.17e} n={n} nnz={} rhs_count={rhs_count} \
         repetitions={repetitions} elapsed_seconds={elapsed:.9} checksum={checksum:.17e} \
         max_residual={maximum_residual:.17e} \
         actual_observed_worker_threads={maximum_threads} input_sha256={} \
         output_sha256={output_sha256}",
        matrix.nnz(),
        cubic_splu_fixture_sha256(&matrix, &right_hand_sides),
    );
}

#[cfg(feature = "sparse-incumbent-bench")]
fn profile_periodic_cuboid_splu_rust(
    repetitions: usize,
    x_extent: usize,
    y_extent: usize,
    z_extent: usize,
    rhs_count: usize,
    output_path: Option<&str>,
) {
    let shift = 1.0e-3;
    let x_weight = -0.75;
    let y_weight = -1.0;
    let z_weight = -1.25;
    let n = x_extent * y_extent * z_extent;
    let matrix_csr = laplacian_3d_periodic_cuboid(
        x_extent, y_extent, z_extent, shift, x_weight, y_weight, z_weight,
    );
    let matrix = matrix_csr.to_csc().expect("shifted-periodic cuboid CSC");
    let right_hand_sides = (0..rhs_count)
        .map(|rhs_index| cubic_splu_rhs(n, rhs_index))
        .collect::<Vec<_>>();

    let warm_factor = splu(&matrix, LuOptions::default()).expect("periodic cuboid splu warmup");
    let warm_solutions = right_hand_sides
        .iter()
        .map(|rhs| splu_solve(&warm_factor, rhs).expect("periodic cuboid splu warmup solve"))
        .collect::<Vec<_>>();
    let maximum_residual =
        splu_max_relative_residual(&matrix_csr, &right_hand_sides, &warm_solutions);
    let mut checksum = warm_solutions
        .iter()
        .map(|solution| solution[n / 2])
        .sum::<f64>();
    let mut maximum_threads = profile_observed_os_threads();

    let started = Instant::now();
    for _ in 0..repetitions {
        let factor =
            splu(black_box(&matrix), LuOptions::default()).expect("periodic cuboid splu profile");
        for rhs in &right_hand_sides {
            let solution = splu_solve(black_box(&factor), black_box(rhs))
                .expect("periodic cuboid splu profile solve");
            checksum += black_box(solution[n / 2]);
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    maximum_threads = maximum_threads.max(profile_observed_os_threads());

    let mut output_bytes = Vec::with_capacity(rhs_count * n * std::mem::size_of::<f64>());
    for solution in &warm_solutions {
        for &value in solution {
            output_bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let output_sha256 = format!("{:x}", Sha256::digest(&output_bytes));
    if let Some(path) = output_path {
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("create new periodic cuboid solution artifact");
        output
            .write_all(&output_bytes)
            .expect("write periodic cuboid solution artifact");
    }

    println!(
        "PERIODIC_CUBOID_SPLU_PROFILE x={x_extent} y={y_extent} z={z_extent} \
         x_weight={x_weight:.17e} y_weight={y_weight:.17e} z_weight={z_weight:.17e} \
         shift={shift:.17e} n={n} nnz={} rhs_count={rhs_count} \
         repetitions={repetitions} elapsed_seconds={elapsed:.9} checksum={checksum:.17e} \
         max_residual={maximum_residual:.17e} \
         actual_observed_worker_threads={maximum_threads} input_sha256={} \
         output_sha256={output_sha256}",
        matrix.nnz(),
        cubic_splu_fixture_sha256(&matrix, &right_hand_sides),
    );
}

#[cfg(feature = "sparse-incumbent-bench")]
fn profile_periodic_cuboid_spsolve_rust(
    repetitions: usize,
    x_extent: usize,
    y_extent: usize,
    z_extent: usize,
    output_path: Option<&str>,
) {
    let shift = 1.0e-3;
    let x_weight = -0.75;
    let y_weight = -1.0;
    let z_weight = -1.25;
    let n = x_extent * y_extent * z_extent;
    let matrix = laplacian_3d_periodic_cuboid(
        x_extent, y_extent, z_extent, shift, x_weight, y_weight, z_weight,
    );
    let matrix_csc = matrix.to_csc().expect("shifted-periodic cuboid CSC");
    let rhs = cubic_splu_rhs(n, 1);

    apply_structural_fastpath_env();
    let warm =
        spsolve(&matrix, &rhs, SolveOptions::default()).expect("periodic cuboid spsolve warmup");
    println!("periodic_profile_backend_used={:?}", warm.backend_used);
    let maximum_residual = splu_max_relative_residual(
        &matrix,
        std::slice::from_ref(&rhs),
        std::slice::from_ref(&warm.solution),
    );
    let mut checksum = warm.solution[n / 2];
    let mut maximum_threads = profile_observed_os_threads();

    let started = Instant::now();
    for _ in 0..repetitions {
        let solved = spsolve(black_box(&matrix), black_box(&rhs), SolveOptions::default())
            .expect("periodic cuboid spsolve profile");
        checksum += black_box(solved.solution[n / 2]);
    }
    let elapsed = started.elapsed().as_secs_f64();
    maximum_threads = maximum_threads.max(profile_observed_os_threads());

    let mut output_bytes = Vec::with_capacity(n * std::mem::size_of::<f64>());
    for &value in &warm.solution {
        output_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let output_sha256 = format!("{:x}", Sha256::digest(&output_bytes));
    if let Some(path) = output_path {
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("create new periodic cuboid spsolve solution artifact");
        output
            .write_all(&output_bytes)
            .expect("write periodic cuboid spsolve solution artifact");
    }

    println!(
        "PERIODIC_CUBOID_SPSOLVE_PROFILE x={x_extent} y={y_extent} z={z_extent} \
         x_weight={x_weight:.17e} y_weight={y_weight:.17e} z_weight={z_weight:.17e} \
         shift={shift:.17e} n={n} nnz={} repetitions={repetitions} \
         elapsed_seconds={elapsed:.9} checksum={checksum:.17e} \
         max_residual={maximum_residual:.17e} \
         actual_observed_worker_threads={maximum_threads} input_sha256={} \
         output_sha256={output_sha256}",
        matrix.nnz(),
        cubic_splu_fixture_sha256(&matrix_csc, std::slice::from_ref(&rhs)),
    );
}

#[cfg(feature = "sparse-incumbent-bench")]
mod cubic_live {
    use super::{
        convection_diffusion_2d, laplacian_3d_cubic, laplacian_3d_neumann_cubic,
        laplacian_3d_periodic_cuboid,
    };
    use fsci_sparse::linalg::{
        SPLU_PERIODIC_CUBOID_SPECTRAL_DISABLE, SPLU_PERIODIC_CUBOID_SPECTRAL_FACTOR_HITS,
        SPLU_PERIODIC_CUBOID_SPECTRAL_SOLVE_HITS, SPSOLVE_PERIODIC_CUBOID_SPECTRAL_DISABLE,
        SPSOLVE_PERIODIC_CUBOID_SPECTRAL_HITS,
    };
    use fsci_sparse::{
        CscMatrix, CsrMatrix, FormatConvertible, LuOptions, PermutationOrdering,
        SPLU_CUBIC_NEUMANN_SPECTRAL_DISABLE, SPLU_CUBIC_NEUMANN_SPECTRAL_FACTOR_HITS,
        SPLU_CUBIC_NEUMANN_SPECTRAL_SOLVE_HITS, SPLU_CUBIC_SPECTRAL_DISABLE,
        SPLU_CUBIC_SPECTRAL_FACTOR_HITS, SPLU_CUBIC_SPECTRAL_SOLVE_HITS,
        SPLU_MERGE_FORCE_LEGACY_WALK, SPLU_SOLVE_FORCE_MATERIALIZED_RHS,
        SPSOLVE_CUBIC_SPECTRAL_DISABLE, SPSOLVE_CUBIC_SPECTRAL_HITS, SolveOptions,
        SparseLuFactorization, splu, splu_factor_payload_bytes, splu_solve, spsolve,
    };
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeMap, BTreeSet};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::path::{Path, PathBuf};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    const SIDES: [usize; 3] = [12, 14, 16];
    const EXPECTED_COMPONENTS: usize = 8_568;
    const SPLU_RHS_COUNT: usize = 16;
    const EXPECTED_SPLU_COMPONENTS: usize = 137_088;
    const CUBIC_EXTENTS: [[usize; 3]; 3] = [[12, 12, 12], [14, 14, 14], [16, 16, 16]];
    const PERIODIC_CUBOID_EXTENTS: [[usize; 3]; 3] = [[9, 11, 13], [11, 13, 15], [13, 15, 17]];
    const EXPECTED_PERIODIC_CUBOID_SPLU_COMPONENTS: usize = 107_952;
    const CONVECTION_EXTENTS: [[usize; 3]; 1] = [[64, 64, 1]];
    const EXPECTED_CONVECTION_SPLU_COMPONENTS: usize = 65_536;
    const PERIODIC_CUBOID_SPSOLVE_RHS_COUNT: usize = 32;
    const EXPECTED_PERIODIC_CUBOID_SPSOLVE_COMPONENTS: usize = 41_184;
    const RESIDUAL_LIMIT: f64 = 1.0e-8;
    const L2_LIMIT: f64 = 1.0e-10;
    const MINIMUM_ROUNDS: usize = 21;
    const NULL_MEDIAN_LIMIT: f64 = 0.02;
    const LOAD_BUSY_LIMIT: f64 = 0.20;
    const LOAD_SAMPLE: Duration = Duration::from_secs(1);
    const LINALG_SOURCE_BYTES: &[u8] = include_bytes!("../linalg.rs");
    const HARNESS_SOURCE_BYTES: &[u8] = include_bytes!("perf_spsolve.rs");

    struct Fixture {
        side: usize,
        matrix: CsrMatrix,
        rhs: Vec<f64>,
    }

    struct SpluFixture {
        matrix: CsrMatrix,
        csc: CscMatrix,
        right_hand_sides: Vec<Vec<f64>>,
    }

    #[derive(Clone, Copy)]
    enum SpluFamily {
        Dirichlet,
        Neumann,
        PeriodicCuboid,
        // The lazy-column A/B control this family once exercised was deleted from
        // production and is NOT coming back (frankenscipy-gkzq8 forbids reviving a
        // toggle to satisfy a harness). What that removed is the A/B, not the
        // fixture: this is the nonsymmetric side-64 factor-plus-16-solves cell
        // carrying the campaign's worst standing vs-incumbent ratio. Since
        // `frankenscipy-9nw95` its control arm drives `SPLU_MERGE_FORCE_LEGACY_WALK`,
        // so the family has a real A/B again — via the merge-walk kernel, not the
        // deleted lazy-column toggle (frankenscipy-run7d).
        Convection,
    }

    impl SpluFamily {
        fn extents(self) -> &'static [[usize; 3]] {
            match self {
                Self::Dirichlet | Self::Neumann => &CUBIC_EXTENTS,
                Self::PeriodicCuboid => &PERIODIC_CUBOID_EXTENTS,
                Self::Convection => &CONVECTION_EXTENTS,
            }
        }

        fn matrix(self, extents: [usize; 3]) -> CsrMatrix {
            let [x_extent, y_extent, z_extent] = extents;
            match self {
                Self::Dirichlet => laplacian_3d_cubic(x_extent),
                Self::Neumann => laplacian_3d_neumann_cubic(x_extent, 1.0e-3),
                Self::PeriodicCuboid => laplacian_3d_periodic_cuboid(
                    x_extent, y_extent, z_extent, 1.0e-3, -0.75, -1.0, -1.25,
                ),
                Self::Convection => {
                    debug_assert_eq!(x_extent, y_extent);
                    debug_assert_eq!(z_extent, 1);
                    convection_diffusion_2d(x_extent)
                }
            }
        }

        fn expected_components(self) -> usize {
            match self {
                Self::Dirichlet | Self::Neumann => EXPECTED_SPLU_COMPONENTS,
                Self::PeriodicCuboid => EXPECTED_PERIODIC_CUBOID_SPLU_COMPONENTS,
                Self::Convection => EXPECTED_CONVECTION_SPLU_COMPONENTS,
            }
        }

        fn set_disabled(self, disabled: bool) {
            match self {
                Self::Dirichlet => {
                    SPLU_CUBIC_SPECTRAL_DISABLE.store(disabled, Ordering::Relaxed);
                }
                Self::Neumann => {
                    SPLU_CUBIC_NEUMANN_SPECTRAL_DISABLE.store(disabled, Ordering::Relaxed);
                }
                Self::PeriodicCuboid => {
                    SPLU_PERIODIC_CUBOID_SPECTRAL_DISABLE.store(disabled, Ordering::Relaxed);
                }
                Self::Convection => {
                    // NO-OP AGAIN, and this is a REVERT of my own wiring. I briefly drove
                    // `SPLU_MERGE_FORCE_LEGACY_WALK` from here so the two merge kernels would
                    // pair inside one window. Then the dispatch guard below fired on its first
                    // run and proved WHY that cannot work: `apply_supernode_tails` is reached
                    // only through `factorize_csr_supernodal`, which is gated on
                    // `SPLU_SUPERNODAL_ENABLE` — default FALSE. The kernel never executes on
                    // this cell, so the toggle selects between two identical behaviours and the
                    // guard correctly refuses to let a ratio be computed from them.
                    //
                    // Leaving the wiring in place would block the live vs-SuperLU row for this
                    // cell permanently, which is a worse outcome than having no A/B here.
                }
            }
        }

        fn reset_hits(self) {
            match self {
                Self::Dirichlet => {
                    SPLU_CUBIC_SPECTRAL_FACTOR_HITS.store(0, Ordering::Relaxed);
                    SPLU_CUBIC_SPECTRAL_SOLVE_HITS.store(0, Ordering::Relaxed);
                }
                Self::Neumann => {
                    SPLU_CUBIC_NEUMANN_SPECTRAL_FACTOR_HITS.store(0, Ordering::Relaxed);
                    SPLU_CUBIC_NEUMANN_SPECTRAL_SOLVE_HITS.store(0, Ordering::Relaxed);
                }
                Self::PeriodicCuboid => {
                    SPLU_PERIODIC_CUBOID_SPECTRAL_FACTOR_HITS.store(0, Ordering::Relaxed);
                    SPLU_PERIODIC_CUBOID_SPECTRAL_SOLVE_HITS.store(0, Ordering::Relaxed);
                }
                Self::Convection => {}
            }
        }

        fn hits(self) -> (usize, usize) {
            match self {
                Self::Dirichlet => (
                    SPLU_CUBIC_SPECTRAL_FACTOR_HITS.load(Ordering::Relaxed),
                    SPLU_CUBIC_SPECTRAL_SOLVE_HITS.load(Ordering::Relaxed),
                ),
                Self::Neumann => (
                    SPLU_CUBIC_NEUMANN_SPECTRAL_FACTOR_HITS.load(Ordering::Relaxed),
                    SPLU_CUBIC_NEUMANN_SPECTRAL_SOLVE_HITS.load(Ordering::Relaxed),
                ),
                Self::PeriodicCuboid => (
                    SPLU_PERIODIC_CUBOID_SPECTRAL_FACTOR_HITS.load(Ordering::Relaxed),
                    SPLU_PERIODIC_CUBOID_SPECTRAL_SOLVE_HITS.load(Ordering::Relaxed),
                ),
                Self::Convection => (0, 0),
            }
        }

        fn expected_hits(self) -> (usize, usize) {
            let factor_hits = self.extents().len();
            let solve_hits = match self {
                Self::Convection => 0,
                Self::Dirichlet | Self::Neumann | Self::PeriodicCuboid => {
                    factor_hits * SPLU_RHS_COUNT
                }
            };
            if matches!(self, Self::Convection) {
                (0, 0)
            } else {
                (factor_hits, solve_hits)
            }
        }

        fn is_nonsymmetric(self) -> bool {
            matches!(self, Self::Convection)
        }

        /// Whether this family's "control" arm is a DIFFERENT code path from the candidate.
        ///
        /// For the three spectral families `set_disabled(true)` routes the control onto the
        /// generic path, so control/candidate is a maintenance ratio. For `Convection`
        /// `set_disabled` drives `SPLU_MERGE_FORCE_LEGACY_WALK` (frankenscipy-9nw95), so
        /// its control arm is the legacy merge kernel — genuinely different code, and the
        /// ratio is held to the same registered 1.20x minimum as every other family.
        fn ab_control(self) -> bool {
            // Convection has NO A/B control: `set_disabled` is a no-op for it, so the control
            // arm re-runs the candidate's own code and control/candidate is a fourth A/A null
            // rather than a maintenance ratio. I wired it to the merge-kernel toggle for one
            // commit and reverted: that kernel is default-unreachable, so the "two arms" were
            // the same code.
            !matches!(self, Self::Convection)
        }

        fn decision_label(self) -> &'static str {
            match self {
                Self::Dirichlet => "CUBIC_SPLU_DECISION",
                Self::Neumann => "NEUMANN_CUBIC_SPLU_DECISION",
                Self::PeriodicCuboid => "PERIODIC_CUBOID_SPLU_DECISION",
                // NOT `..._LAZY_COLUMNS_DECISION`: that named an A/B control which no longer
                // exists, and a label is what a reader greps for. This arm compares against
                // the live incumbent and claims nothing about a toggle.
                Self::Convection => "CONVECTION_SPLU_VS_INCUMBENT",
            }
        }

        fn name(self) -> &'static str {
            match self {
                Self::Dirichlet => "cubic",
                Self::Neumann => "shifted-Neumann cubic",
                Self::PeriodicCuboid => "shifted-periodic cuboid",
                Self::Convection => "nonsymmetric convection-diffusion",
            }
        }
    }

    struct Scipy {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
        components: usize,
        maximum_threads: usize,
    }

    impl Scipy {
        fn start(script: &Path) -> Result<(Self, String), String> {
            Self::start_method(script, "spsolve")
        }

        fn start_splu(script: &Path) -> Result<(Self, String), String> {
            Self::start_method(script, "splu")
        }

        fn start_spsolve_many(script: &Path) -> Result<(Self, String), String> {
            Self::start_method(script, "spsolve_many")
        }

        fn start_method(script: &Path, method: &str) -> Result<(Self, String), String> {
            let mut child = Command::new("python3")
                .arg("-u")
                .arg(script)
                .arg("--live")
                .arg(method)
                .env("OPENBLAS_NUM_THREADS", "1")
                .env("OMP_NUM_THREADS", "1")
                .env("MKL_NUM_THREADS", "1")
                .env("BLIS_NUM_THREADS", "1")
                .env("VECLIB_MAXIMUM_THREADS", "1")
                .env("NUMEXPR_NUM_THREADS", "1")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .map_err(|error| format!("spawn live SciPy arm: {error}"))?;
            let stdin = child
                .stdin
                .take()
                .ok_or_else(|| "live SciPy arm has no stdin".to_string())?;
            let mut stdout = BufReader::new(
                child
                    .stdout
                    .take()
                    .ok_or_else(|| "live SciPy arm has no stdout".to_string())?,
            );
            let mut identity = String::new();
            stdout
                .read_line(&mut identity)
                .map_err(|error| format!("read live SciPy identity: {error}"))?;
            if identity.is_empty() {
                return Err("live SciPy arm exited before reporting identity".to_string());
            }
            Ok((
                Self {
                    child,
                    stdin,
                    stdout,
                    components: 0,
                    maximum_threads: 0,
                },
                identity.trim().to_string(),
            ))
        }

        fn read_reply(&mut self, context: &str) -> Result<String, String> {
            let mut reply = String::new();
            self.stdout
                .read_line(&mut reply)
                .map_err(|error| format!("read {context}: {error}"))?;
            if reply.is_empty() {
                return Err(format!("live SciPy arm exited while reading {context}"));
            }
            Ok(reply.trim().to_string())
        }

        fn initialize(&mut self, fixture: &Fixture) -> Result<(), String> {
            let n = fixture.side * fixture.side * fixture.side;
            writeln!(self.stdin, "INIT {n} {} 0.0 1", fixture.matrix.nnz())
                .map_err(|error| format!("write SciPy INIT: {error}"))?;
            write_usize_vector(&mut self.stdin, "INDPTR", fixture.matrix.indptr())?;
            write_usize_vector(&mut self.stdin, "INDICES", fixture.matrix.indices())?;
            write_f64_vector(&mut self.stdin, "DATA", fixture.matrix.data())?;
            write_f64_vector(&mut self.stdin, "B", &fixture.rhs)?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush SciPy INIT: {error}"))?;
            let reply = self.read_reply("SciPy CASE")?;
            if !reply.starts_with("CASE method=spsolve ")
                || !reply.contains(&format!("n={n} "))
                || !reply.contains(&format!("nnz={} ", fixture.matrix.nnz()))
                || !reply.contains("sorted=True ")
                || !reply.contains("canonical=True ")
                || !reply.contains("finite=True ")
                || !reply.ends_with("nonsymmetric=False")
            {
                return Err(format!("inadmissible SciPy fixture: {reply}"));
            }
            writeln!(self.stdin, "INPUT_SHA256")
                .map_err(|error| format!("write SciPy INPUT_SHA256: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush SciPy INPUT_SHA256: {error}"))?;
            let reported_hash = self.read_reply("SciPy input SHA-256")?;
            let expected_hash = fixture_input_sha256(fixture);
            if reported_hash != format!("INPUT_SHA256 {expected_hash}") {
                return Err(format!(
                    "SciPy input SHA-256 mismatch: expected {expected_hash}, received {reported_hash}"
                ));
            }
            self.components = n;
            Ok(())
        }

        fn initialize_splu(
            &mut self,
            fixture: &SpluFixture,
            nonsymmetric: bool,
        ) -> Result<(), String> {
            self.initialize_csc_many(fixture, "splu", false, nonsymmetric)
        }

        fn initialize_spsolve_many(&mut self, fixture: &SpluFixture) -> Result<(), String> {
            self.initialize_csc_many(fixture, "spsolve_many", true, false)
        }

        fn initialize_csc_many(
            &mut self,
            fixture: &SpluFixture,
            method: &str,
            per_rhs_digest: bool,
            nonsymmetric: bool,
        ) -> Result<(), String> {
            let n = fixture.matrix.shape().rows;
            let rhs_count = fixture.right_hand_sides.len();
            writeln!(
                self.stdin,
                "INIT_SPLU {n} {} {rhs_count}",
                fixture.csc.nnz()
            )
            .map_err(|error| format!("write SciPy INIT_SPLU: {error}"))?;
            write_usize_vector(&mut self.stdin, "INDPTR", fixture.csc.indptr())?;
            write_usize_vector(&mut self.stdin, "INDICES", fixture.csc.indices())?;
            write_f64_vector(&mut self.stdin, "DATA", fixture.csc.data())?;
            let flattened_rhs = fixture
                .right_hand_sides
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            write_f64_vector(&mut self.stdin, "RHS", &flattened_rhs)?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush SciPy INIT_SPLU: {error}"))?;
            let reply = self.read_reply("SciPy splu CASE")?;
            let nonsymmetric_text = if nonsymmetric { "True" } else { "False" };
            if !reply.starts_with(&format!("CASE method={method} "))
                || !reply.contains(&format!("n={n} "))
                || !reply.contains(&format!("nnz={} ", fixture.csc.nnz()))
                || !reply.contains(&format!("rhs_count={rhs_count} "))
                || !reply.contains("sorted=True ")
                || !reply.contains("canonical=True ")
                || !reply.contains("finite=True ")
                || !reply.ends_with(&format!("nonsymmetric={nonsymmetric_text}"))
            {
                return Err(format!("inadmissible SciPy splu fixture: {reply}"));
            }
            writeln!(self.stdin, "INPUT_SHA256")
                .map_err(|error| format!("write SciPy splu INPUT_SHA256: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush SciPy splu INPUT_SHA256: {error}"))?;
            let reported_hash = self.read_reply("SciPy splu input SHA-256")?;
            let expected_hash = if per_rhs_digest {
                spsolve_many_fixture_input_sha256s(fixture).join(",")
            } else {
                splu_fixture_input_sha256(fixture)
            };
            if reported_hash != format!("INPUT_SHA256 {expected_hash}") {
                return Err(format!(
                    "SciPy splu input SHA-256 mismatch: expected {expected_hash}, received {reported_hash}"
                ));
            }
            self.components = n.saturating_mul(rhs_count);
            Ok(())
        }

        fn parity(&mut self) -> Result<(Vec<f64>, f64), String> {
            writeln!(self.stdin, "PARITY")
                .map_err(|error| format!("write SciPy PARITY: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush SciPy PARITY: {error}"))?;
            let result = self.read_reply("SciPy parity result")?;
            if !result.starts_with("RESULT info=0 iterations=0 ") {
                return Err(format!("inadmissible SciPy parity result: {result}"));
            }
            let residual = result
                .split_whitespace()
                .find_map(|field| field.strip_prefix("residual="))
                .ok_or_else(|| "SciPy parity omitted residual".to_string())?
                .parse::<f64>()
                .map_err(|error| format!("parse SciPy parity residual: {error}"))?;
            if !result.contains(&format!("components={}", self.components)) {
                return Err(format!("SciPy parity component mismatch: {result}"));
            }
            let output = self.read_reply("SciPy parity solution")?;
            let payload = output
                .strip_prefix("X ")
                .ok_or_else(|| format!("invalid SciPy parity solution: {output}"))?;
            let solution = payload
                .split(',')
                .map(|value| {
                    value
                        .parse::<f64>()
                        .map_err(|error| format!("parse SciPy solution: {error}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if solution.len() != self.components || solution.iter().any(|value| !value.is_finite())
            {
                return Err("SciPy parity solution is incomplete or non-finite".to_string());
            }
            Ok((solution, residual))
        }

        fn parity_splu(&mut self) -> Result<(Vec<f64>, f64, usize), String> {
            writeln!(self.stdin, "PARITY")
                .map_err(|error| format!("write SciPy splu PARITY: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush SciPy splu PARITY: {error}"))?;
            let result = self.read_reply("SciPy splu parity result")?;
            if !result.starts_with("RESULT info=0 iterations=0 ") {
                return Err(format!("inadmissible SciPy splu parity result: {result}"));
            }
            let residual = result
                .split_whitespace()
                .find_map(|field| field.strip_prefix("residual="))
                .ok_or_else(|| "SciPy splu parity omitted residual".to_string())?
                .parse::<f64>()
                .map_err(|error| format!("parse SciPy splu parity residual: {error}"))?;
            let payload_bytes = result
                .split_whitespace()
                .find_map(|field| field.strip_prefix("payload_bytes="))
                .ok_or_else(|| "SciPy splu parity omitted payload bytes".to_string())?
                .parse::<usize>()
                .map_err(|error| format!("parse SciPy splu payload bytes: {error}"))?;
            if !result.contains(&format!("components={} ", self.components)) {
                return Err(format!("SciPy splu parity component mismatch: {result}"));
            }
            let output = self.read_reply("SciPy splu parity solution")?;
            let payload = output
                .strip_prefix("X ")
                .ok_or_else(|| format!("invalid SciPy splu parity solution: {output}"))?;
            let solution = payload
                .split(',')
                .map(|value| {
                    value
                        .parse::<f64>()
                        .map_err(|error| format!("parse SciPy splu solution: {error}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if solution.len() != self.components || solution.iter().any(|value| !value.is_finite())
            {
                return Err("SciPy splu parity solution is incomplete or non-finite".to_string());
            }
            Ok((solution, residual, payload_bytes))
        }

        fn time_one(&mut self) -> Result<(f64, u64), String> {
            writeln!(self.stdin, "SOLVE 1")
                .map_err(|error| format!("write SciPy SOLVE: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush SciPy SOLVE: {error}"))?;
            let reply = self.read_reply("timed SciPy solve")?;
            let fields = reply.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 6 || fields[0] != "TIME" {
                return Err(format!("invalid timed SciPy reply: {reply}"));
            }
            let elapsed = parse::<f64>(fields[1], "SciPy elapsed")?;
            let info = parse::<i32>(fields[2], "SciPy status")?;
            let components = parse::<usize>(fields[3], "SciPy components")?;
            let threads = parse::<usize>(fields[4], "SciPy observed threads")?;
            let checksum = parse::<u64>(fields[5], "SciPy output-bit checksum")?;
            self.maximum_threads = self.maximum_threads.max(threads);
            if info != 0
                || components != self.components
                || threads != 1
                || !elapsed.is_finite()
                || elapsed <= 0.0
            {
                return Err(format!("inadmissible timed SciPy reply: {reply}"));
            }
            Ok((elapsed, checksum))
        }
    }

    impl Drop for Scipy {
        fn drop(&mut self) {
            let _ = writeln!(self.stdin, "QUIT");
            let _ = self.stdin.flush();
            let _ = self.child.wait();
        }
    }

    fn parse<T: std::str::FromStr>(value: &str, label: &str) -> Result<T, String>
    where
        T::Err: std::fmt::Display,
    {
        value
            .parse::<T>()
            .map_err(|error| format!("parse {label}: {error}"))
    }

    fn write_usize_vector(
        output: &mut ChildStdin,
        label: &str,
        values: &[usize],
    ) -> Result<(), String> {
        write!(output, "{label} ").map_err(|error| format!("write {label}: {error}"))?;
        for (index, value) in values.iter().enumerate() {
            if index != 0 {
                write!(output, ",").map_err(|error| format!("write {label}: {error}"))?;
            }
            write!(output, "{value}").map_err(|error| format!("write {label}: {error}"))?;
        }
        writeln!(output).map_err(|error| format!("write {label}: {error}"))
    }

    fn write_f64_vector(
        output: &mut ChildStdin,
        label: &str,
        values: &[f64],
    ) -> Result<(), String> {
        write!(output, "{label} ").map_err(|error| format!("write {label}: {error}"))?;
        for (index, value) in values.iter().enumerate() {
            if index != 0 {
                write!(output, ",").map_err(|error| format!("write {label}: {error}"))?;
            }
            write!(output, "{value:.17e}").map_err(|error| format!("write {label}: {error}"))?;
        }
        writeln!(output).map_err(|error| format!("write {label}: {error}"))
    }

    fn fixtures() -> Vec<Fixture> {
        SIDES
            .into_iter()
            .map(|side| {
                let n = side * side * side;
                Fixture {
                    side,
                    matrix: laplacian_3d_cubic(side),
                    rhs: (0..n).map(|i| 1.0 + 0.5 * (i % 13) as f64).collect(),
                }
            })
            .collect()
    }

    fn splu_fixtures(family: SpluFamily) -> Result<Vec<SpluFixture>, String> {
        family
            .extents()
            .iter()
            .copied()
            .map(|extents| {
                let matrix = family.matrix(extents);
                let n = matrix.shape().rows;
                let csc = matrix
                    .to_csc()
                    .map_err(|error| format!("construct {} CSC: {error}", family.name()))?;
                let right_hand_sides = (0..SPLU_RHS_COUNT)
                    .map(|rhs_index| {
                        (0..n)
                            .map(|index| 1.0 + 0.125 * ((17 * index + 23 * rhs_index) % 29) as f64)
                            .collect::<Vec<_>>()
                    })
                    .collect();
                Ok(SpluFixture {
                    matrix,
                    csc,
                    right_hand_sides,
                })
            })
            .collect()
    }

    /// Extents for the periodic-cuboid live row, selected by `FSCI_PERIODIC_CUBOID_EXTENTS`.
    ///
    /// EXISTS TO MEASURE A WIDENING, NOT TO TUNE. The default is the pre-registered 9x11x13 and
    /// is what every arm runs when the variable is unset. Until frankenscipy-g68jq the recognizer
    /// refused any grid that was not >= 9, odd and pairwise distinct, so the only fixture this
    /// harness could build was one it already recognized -- which is precisely the fixture that
    /// CANNOT show what widening the recognizer is worth. This knob lets a cubic grid be put in
    /// front of the same live SciPy arm, in one invocation, on the same ELF.
    fn periodic_cuboid_extents() -> Result<(usize, usize, usize), String> {
        let raw = match std::env::var("FSCI_PERIODIC_CUBOID_EXTENTS") {
            Ok(value) if !value.is_empty() && value != "default" => value,
            _ => return Ok((9, 11, 13)),
        };
        let parts: Vec<&str> = raw.split(['x', ',']).collect();
        if parts.len() != 3 {
            return Err(format!(
                "FSCI_PERIODIC_CUBOID_EXTENTS={raw:?} must be three extents, e.g. 11x11x11"
            ));
        }
        let mut extents = [0usize; 3];
        for (slot, part) in parts.iter().enumerate() {
            extents[slot] = parse::<usize>(part, "cuboid extent")?;
            if extents[slot] < 3 {
                return Err(format!(
                    "FSCI_PERIODIC_CUBOID_EXTENTS={raw:?}: every extent must be at least three"
                ));
            }
        }
        Ok((extents[0], extents[1], extents[2]))
    }

    fn periodic_cuboid_spsolve_fixtures() -> Result<Vec<SpluFixture>, String> {
        let (x_extent, y_extent, z_extent) = periodic_cuboid_extents()?;
        let matrix =
            laplacian_3d_periodic_cuboid(x_extent, y_extent, z_extent, 1.0e-3, -0.75, -1.0, -1.25);
        let n = matrix.shape().rows;
        let csc = matrix
            .to_csc()
            .map_err(|error| format!("construct periodic cuboid spsolve CSC: {error}"))?;
        let right_hand_sides = (0..PERIODIC_CUBOID_SPSOLVE_RHS_COUNT)
            .map(|rhs_index| {
                (0..n)
                    .map(|index| 1.0 + 0.125 * ((17 * index + 23 * rhs_index) % 29) as f64)
                    .collect::<Vec<_>>()
            })
            .collect();
        Ok(vec![SpluFixture {
            matrix,
            csc,
            right_hand_sides,
        }])
    }

    fn fixture_input_sha256(fixture: &Fixture) -> String {
        let n = fixture.side * fixture.side * fixture.side;
        let mut hasher = Sha256::new();
        hasher.update((n as u64).to_le_bytes());
        hasher.update((fixture.matrix.nnz() as u64).to_le_bytes());
        for &value in fixture.matrix.data() {
            hasher.update(value.to_le_bytes());
        }
        for &index in fixture.matrix.indices() {
            hasher.update((index as u64).to_le_bytes());
        }
        for &pointer in fixture.matrix.indptr() {
            hasher.update((pointer as u64).to_le_bytes());
        }
        for &value in &fixture.rhs {
            hasher.update(value.to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    fn combined_input_sha256(fixtures: &[Fixture]) -> String {
        let mut hasher = Sha256::new();
        for fixture in fixtures {
            hasher.update((fixture.side as u64).to_le_bytes());
            hasher.update(fixture_input_sha256(fixture).as_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    fn splu_fixture_input_sha256(fixture: &SpluFixture) -> String {
        let n = fixture.matrix.shape().rows;
        let mut hasher = Sha256::new();
        hasher.update((n as u64).to_le_bytes());
        hasher.update((fixture.csc.nnz() as u64).to_le_bytes());
        for &value in fixture.csc.data() {
            hasher.update(value.to_le_bytes());
        }
        for &index in fixture.csc.indices() {
            hasher.update((index as u64).to_le_bytes());
        }
        for &pointer in fixture.csc.indptr() {
            hasher.update((pointer as u64).to_le_bytes());
        }
        for right_hand_side in &fixture.right_hand_sides {
            for &value in right_hand_side {
                hasher.update(value.to_le_bytes());
            }
        }
        format!("{:x}", hasher.finalize())
    }

    fn spsolve_many_fixture_input_sha256s(fixture: &SpluFixture) -> Vec<String> {
        let n = fixture.matrix.shape().rows;
        fixture
            .right_hand_sides
            .iter()
            .map(|right_hand_side| {
                let mut hasher = Sha256::new();
                hasher.update((n as u64).to_le_bytes());
                hasher.update((fixture.csc.nnz() as u64).to_le_bytes());
                for &value in fixture.csc.data() {
                    hasher.update(value.to_le_bytes());
                }
                for &index in fixture.csc.indices() {
                    hasher.update((index as u64).to_le_bytes());
                }
                for &pointer in fixture.csc.indptr() {
                    hasher.update((pointer as u64).to_le_bytes());
                }
                for &value in right_hand_side {
                    hasher.update(value.to_le_bytes());
                }
                format!("{:x}", hasher.finalize())
            })
            .collect()
    }

    fn combined_splu_input_sha256(fixtures: &[SpluFixture]) -> String {
        let mut hasher = Sha256::new();
        for fixture in fixtures {
            hasher.update(splu_fixture_input_sha256(fixture).as_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    fn sha256_file(path: &Path) -> Result<String, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("read {} for SHA-256: {error}", path.display()))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    fn sha256_of_self() -> Result<String, String> {
        let executable =
            std::env::current_exe().map_err(|error| format!("current executable: {error}"))?;
        sha256_file(&executable)
    }

    fn source_is_fresh(compiled: &[u8], runtime: &[u8]) -> bool {
        Sha256::digest(compiled) == Sha256::digest(runtime)
    }

    /// Prove that this process contains the `linalg.rs` and harness bytes present on the worker.
    ///
    /// A green test cannot establish this by itself: RCH has previously served a stale test
    /// binary. `include_bytes!` makes the compiler's input observable from inside the process,
    /// while the runtime reads detect a stale target cache or failed source transfer.
    pub fn run_source_freshness(arguments: &[String]) -> Result<(), String> {
        let expected_marker = match arguments {
            [] => None,
            [flag, marker] if flag == "--expect-marker" => Some(marker.as_str()),
            _ => {
                return Err("usage: --source-freshness [--expect-marker <marker>]".to_string());
            }
        };
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let linalg_runtime = std::fs::read(manifest.join("src/linalg.rs"))
            .map_err(|error| format!("read worker linalg.rs: {error}"))?;
        let harness_runtime = std::fs::read(manifest.join("src/bin/perf_spsolve.rs"))
            .map_err(|error| format!("read worker perf_spsolve.rs: {error}"))?;
        let linalg_fresh = source_is_fresh(LINALG_SOURCE_BYTES, &linalg_runtime);
        let harness_fresh = source_is_fresh(HARNESS_SOURCE_BYTES, &harness_runtime);
        let marker_fresh =
            expected_marker.is_none_or(|marker| marker == super::PERF_SPSOLVE_SOURCE_MARKER);
        let elf_sha256 = sha256_of_self()?;
        println!("elf_sha256={elf_sha256}");
        println!(
            "linalg_source_fresh={linalg_fresh} compiled_sha256={:x} runtime_sha256={:x}",
            Sha256::digest(LINALG_SOURCE_BYTES),
            Sha256::digest(&linalg_runtime)
        );
        println!(
            "harness_source_fresh={harness_fresh} compiled_sha256={:x} runtime_sha256={:x}",
            Sha256::digest(HARNESS_SOURCE_BYTES),
            Sha256::digest(&harness_runtime)
        );
        println!(
            "marker_check expected={} observed={} fresh={marker_fresh}",
            expected_marker.unwrap_or("<none>"),
            super::PERF_SPSOLVE_SOURCE_MARKER,
        );
        if linalg_fresh && harness_fresh && marker_fresh {
            println!("SOURCE_FRESHNESS_VERDICT=FRESH");
            Ok(())
        } else {
            Err("SOURCE_FRESHNESS_VERDICT=STALE".to_string())
        }
    }

    fn relative_residual(fixture: &Fixture, solution: &[f64]) -> f64 {
        let mut residual_squared = 0.0;
        let mut rhs_squared = 0.0;
        for (row, &rhs) in fixture.rhs.iter().enumerate() {
            let mut product = 0.0;
            for index in fixture.matrix.indptr()[row]..fixture.matrix.indptr()[row + 1] {
                product += fixture.matrix.data()[index] * solution[fixture.matrix.indices()[index]];
            }
            residual_squared += (product - rhs).powi(2);
            rhs_squared += rhs * rhs;
        }
        residual_squared.sqrt() / rhs_squared.sqrt()
    }

    fn relative_l2(left: &[Vec<f64>], right: &[Vec<f64>]) -> f64 {
        let mut difference_squared = 0.0;
        let mut reference_squared = 0.0;
        for (left_solution, right_solution) in left.iter().zip(right) {
            for (&left_value, &right_value) in left_solution.iter().zip(right_solution) {
                difference_squared += (left_value - right_value).powi(2);
                reference_squared += right_value * right_value;
            }
        }
        difference_squared.sqrt() / reference_squared.sqrt()
    }

    fn component_mismatches(left: &[Vec<f64>], right: &[Vec<f64>]) -> usize {
        left.iter()
            .zip(right)
            .map(|(left_solution, right_solution)| {
                left_solution
                    .iter()
                    .zip(right_solution)
                    .filter(|(left_value, right_value)| {
                        (**left_value - **right_value).abs() > 1.0e-10 + 1.0e-10 * right_value.abs()
                    })
                    .count()
            })
            .sum()
    }

    fn splu_max_relative_residual(fixture: &SpluFixture, solutions: &[f64]) -> f64 {
        let n = fixture.matrix.shape().rows;
        fixture
            .right_hand_sides
            .iter()
            .zip(solutions.chunks_exact(n))
            .map(|(rhs, solution)| {
                let mut residual_squared = 0.0;
                let mut rhs_squared = 0.0;
                for (row, &rhs_value) in rhs.iter().enumerate() {
                    let mut product = 0.0;
                    for index in fixture.matrix.indptr()[row]..fixture.matrix.indptr()[row + 1] {
                        product += fixture.matrix.data()[index]
                            * solution[fixture.matrix.indices()[index]];
                    }
                    residual_squared += (product - rhs_value).powi(2);
                    rhs_squared += rhs_value * rhs_value;
                }
                residual_squared.sqrt() / rhs_squared.sqrt()
            })
            .fold(0.0f64, f64::max)
    }

    fn rust_solutions(fixtures: &[Fixture], disable: bool) -> Result<Vec<Vec<f64>>, String> {
        SPSOLVE_CUBIC_SPECTRAL_DISABLE.store(disable, Ordering::Relaxed);
        let result = fixtures
            .iter()
            .map(|fixture| {
                spsolve(&fixture.matrix, &fixture.rhs, SolveOptions::default())
                    .map(|result| result.solution)
                    .map_err(|error| format!("FrankenSciPy spsolve: {error}"))
            })
            .collect::<Result<Vec<_>, _>>();
        SPSOLVE_CUBIC_SPECTRAL_DISABLE.store(false, Ordering::Relaxed);
        result
    }

    /// Ordering used by the fsci arms of the splu family live row, selected by
    /// `FSCI_SPLU_ORDERING` and defaulting to the library default.
    ///
    /// EXISTS TO CERTIFY, NOT TO TUNE. The ordering sweep (df7a1fc52) showed min-degree cuts
    /// this cell's fill 2.61x and its SOLVE 2.31x while making the whole job 1.96x WORSE,
    /// because the exact min-degree is O(V^2) and `splu` pays it inside every factor call. A
    /// solve-only speedup is a self-speedup; the only way to know what it is worth is to put
    /// the SAME ordering in front of the live SuperLU arm in one invocation. This knob does
    /// that and changes no default: unset, every arm runs exactly as it ships.
    fn splu_arm_ordering() -> Result<PermutationOrdering, String> {
        match std::env::var("FSCI_SPLU_ORDERING").ok().as_deref() {
            None | Some("") | Some("default") => Ok(LuOptions::default().ordering),
            Some("colamd") => Ok(PermutationOrdering::Colamd),
            Some("rcm") => Ok(PermutationOrdering::ReverseCuthillMcKee),
            Some("mmd-ata") => Ok(PermutationOrdering::MmdAta),
            Some("mmd-at-plus-a") => Ok(PermutationOrdering::MmdAtPlusA),
            Some("amd") => Ok(PermutationOrdering::Amd),
            Some("natural") => Ok(PermutationOrdering::Natural),
            Some(other) => Err(format!(
                "FSCI_SPLU_ORDERING={other:?} is not one of default|colamd|rcm|mmd-ata|mmd-at-plus-a|amd|natural"
            )),
        }
    }

    fn splu_arm_options() -> Result<LuOptions, String> {
        Ok(LuOptions {
            ordering: splu_arm_ordering()?,
            ..LuOptions::default()
        })
    }

    fn rust_splu_solutions(
        fixtures: &[SpluFixture],
        disable: bool,
        family: SpluFamily,
    ) -> Result<(Vec<Vec<f64>>, usize), String> {
        family.set_disabled(disable);
        let result = (|| {
            let mut all_solutions = Vec::with_capacity(fixtures.len());
            let mut payload_bytes = 0usize;
            for fixture in fixtures {
                let factor = splu(&fixture.csc, splu_arm_options()?)
                    .map_err(|error| format!("FrankenSciPy splu: {error}"))?;
                payload_bytes = payload_bytes.saturating_add(splu_factor_payload_bytes(&factor));
                let mut flattened = Vec::with_capacity(
                    fixture
                        .matrix
                        .shape()
                        .rows
                        .saturating_mul(fixture.right_hand_sides.len()),
                );
                for right_hand_side in &fixture.right_hand_sides {
                    let solution = splu_solve(&factor, right_hand_side)
                        .map_err(|error| format!("FrankenSciPy splu_solve: {error}"))?;
                    flattened.extend(solution);
                }
                all_solutions.push(flattened);
            }
            Ok((all_solutions, payload_bytes))
        })();
        family.set_disabled(false);
        result
    }

    fn rust_periodic_cuboid_spsolve_solutions(
        fixtures: &[SpluFixture],
        disable: bool,
    ) -> Result<Vec<Vec<f64>>, String> {
        SPSOLVE_PERIODIC_CUBOID_SPECTRAL_DISABLE.store(disable, Ordering::Relaxed);
        let result = (|| {
            let mut all_solutions = Vec::with_capacity(fixtures.len());
            for fixture in fixtures {
                let mut flattened = Vec::with_capacity(
                    fixture
                        .matrix
                        .shape()
                        .rows
                        .saturating_mul(fixture.right_hand_sides.len()),
                );
                for right_hand_side in &fixture.right_hand_sides {
                    let solution =
                        spsolve(&fixture.matrix, right_hand_side, SolveOptions::default())
                            .map_err(|error| format!("FrankenSciPy periodic spsolve: {error}"))?
                            .solution;
                    flattened.extend(solution);
                }
                all_solutions.push(flattened);
            }
            Ok(all_solutions)
        })();
        SPSOLVE_PERIODIC_CUBOID_SPECTRAL_DISABLE.store(false, Ordering::Relaxed);
        result
    }

    fn time_rust_job(fixtures: &[Fixture], disable: bool) -> Result<f64, String> {
        SPSOLVE_CUBIC_SPECTRAL_DISABLE.store(disable, Ordering::Relaxed);
        let result = (|| {
            let started = Instant::now();
            let mut checksum = 0u64;
            for fixture in fixtures {
                let solution = spsolve(
                    black_box(&fixture.matrix),
                    black_box(&fixture.rhs),
                    SolveOptions::default(),
                )
                .map_err(|error| format!("timed FrankenSciPy spsolve: {error}"))?
                .solution;
                for value in solution {
                    checksum = checksum.rotate_left(1) ^ value.to_bits();
                }
            }
            black_box(checksum);
            Ok(started.elapsed().as_secs_f64())
        })();
        SPSOLVE_CUBIC_SPECTRAL_DISABLE.store(false, Ordering::Relaxed);
        result
    }

    fn time_rust_splu_job(
        fixtures: &[SpluFixture],
        disable: bool,
        family: SpluFamily,
    ) -> Result<f64, String> {
        family.set_disabled(disable);
        let result = (|| {
            let started = Instant::now();
            let mut checksum = 0u64;
            for fixture in fixtures {
                let factor = splu(black_box(&fixture.csc), splu_arm_options()?)
                    .map_err(|error| format!("timed FrankenSciPy splu: {error}"))?;
                for right_hand_side in &fixture.right_hand_sides {
                    let solution = splu_solve(&factor, black_box(right_hand_side))
                        .map_err(|error| format!("timed FrankenSciPy splu_solve: {error}"))?;
                    for value in solution {
                        checksum = checksum.rotate_left(1) ^ value.to_bits();
                    }
                }
            }
            black_box(checksum);
            Ok(started.elapsed().as_secs_f64())
        })();
        family.set_disabled(false);
        result
    }

    fn time_rust_periodic_cuboid_spsolve_job(
        fixtures: &[SpluFixture],
        disable: bool,
    ) -> Result<f64, String> {
        SPSOLVE_PERIODIC_CUBOID_SPECTRAL_DISABLE.store(disable, Ordering::Relaxed);
        let result = (|| {
            let started = Instant::now();
            let mut checksum = 0u64;
            for fixture in fixtures {
                for right_hand_side in &fixture.right_hand_sides {
                    let solution = spsolve(
                        black_box(&fixture.matrix),
                        black_box(right_hand_side),
                        SolveOptions::default(),
                    )
                    .map_err(|error| format!("timed FrankenSciPy periodic spsolve: {error}"))?
                    .solution;
                    for value in solution {
                        checksum = checksum.rotate_left(1) ^ value.to_bits();
                    }
                }
            }
            black_box(checksum);
            Ok(started.elapsed().as_secs_f64())
        })();
        SPSOLVE_PERIODIC_CUBOID_SPECTRAL_DISABLE.store(false, Ordering::Relaxed);
        result
    }

    fn time_scipy_job(oracles: &mut [Scipy]) -> Result<f64, String> {
        let mut elapsed = 0.0;
        let mut checksum = 0u64;
        for oracle in oracles {
            let (fixture_elapsed, fixture_checksum) = oracle.time_one()?;
            elapsed += fixture_elapsed;
            checksum = checksum.rotate_left(1) ^ fixture_checksum;
        }
        black_box(checksum);
        Ok(elapsed)
    }

    fn median(mut values: Vec<f64>) -> f64 {
        values.sort_by(f64::total_cmp);
        if values.len().is_multiple_of(2) {
            0.5 * (values[values.len() / 2 - 1] + values[values.len() / 2])
        } else {
            values[values.len() / 2]
        }
    }

    fn percentile(mut values: Vec<f64>, quantile: f64) -> f64 {
        values.sort_by(f64::total_cmp);
        let index = ((values.len() - 1) as f64 * quantile).ceil() as usize;
        values[index.min(values.len() - 1)]
    }

    fn bootstrap_median_ci(values: &[f64]) -> (f64, f64) {
        let mut state = 0x6a09_e667_f3bc_c909u64;
        let mut medians = Vec::with_capacity(10_000);
        for _ in 0..10_000 {
            let mut sample = Vec::with_capacity(values.len());
            for _ in 0..values.len() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                sample.push(values[(state as usize) % values.len()]);
            }
            medians.push(median(sample));
        }
        medians.sort_by(f64::total_cmp);
        (medians[250], medians[9_750])
    }

    fn cv(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len().saturating_sub(1).max(1) as f64;
        variance.sqrt() / mean
    }

    fn ratios(numerator: &[f64], denominator: &[f64]) -> Vec<f64> {
        numerator
            .iter()
            .zip(denominator)
            .map(|(left, right)| left / right)
            .collect()
    }

    fn csv(values: &[f64]) -> String {
        values
            .iter()
            .map(|value| format!("{value:.9}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    #[derive(Clone, Copy)]
    enum Arm {
        Candidate,
        Control,
        Live,
    }

    // The two-arm schedule is deliberately copied from the fleet's sanctioned
    // busy-host substrate. Its mirrored halves make each arm's own A/A ratio a
    // direct drift detector, rather than requiring an impossible quiet-host
    // admission on a shared machine.
    const BALANCED_SQUARE: [bool; 8] = [true, false, false, true, true, false, false, true];

    #[derive(Clone, Copy)]
    enum Sample {
        Headline(Arm),
        NullLeft(Arm),
        NullRight(Arm),
    }

    #[derive(Default)]
    struct Measurement {
        candidate: Vec<f64>,
        candidate_live: Vec<f64>,
        control: Vec<f64>,
        live: Vec<f64>,
        candidate_null_left: Vec<f64>,
        candidate_null_right: Vec<f64>,
        candidate_live_null_left: Vec<f64>,
        candidate_live_null_right: Vec<f64>,
        control_null_left: Vec<f64>,
        control_null_right: Vec<f64>,
        live_null_left: Vec<f64>,
        live_null_right: Vec<f64>,
    }

    impl Measurement {
        fn push(&mut self, sample: Sample, elapsed: f64) {
            match sample {
                Sample::Headline(Arm::Candidate) => self.candidate.push(elapsed),
                Sample::Headline(Arm::Control) => self.control.push(elapsed),
                Sample::Headline(Arm::Live) => self.live.push(elapsed),
                Sample::NullLeft(Arm::Candidate) => self.candidate_null_left.push(elapsed),
                Sample::NullRight(Arm::Candidate) => self.candidate_null_right.push(elapsed),
                Sample::NullLeft(Arm::Control) => self.control_null_left.push(elapsed),
                Sample::NullRight(Arm::Control) => self.control_null_right.push(elapsed),
                Sample::NullLeft(Arm::Live) => self.live_null_left.push(elapsed),
                Sample::NullRight(Arm::Live) => self.live_null_right.push(elapsed),
            }
        }
    }

    struct BalancedSquareTiming {
        first: f64,
        second: f64,
        first_null_left: f64,
        first_null_right: f64,
        second_null_left: f64,
        second_null_right: f64,
    }

    fn summarize_balanced_square(left: Vec<f64>, right: Vec<f64>) -> BalancedSquareTiming {
        debug_assert_eq!(left.len(), 4);
        debug_assert_eq!(right.len(), 4);
        BalancedSquareTiming {
            first: median(left.clone()),
            second: median(right.clone()),
            first_null_left: median(left[..2].to_vec()),
            first_null_right: median(left[2..].to_vec()),
            second_null_left: median(right[..2].to_vec()),
            second_null_right: median(right[2..].to_vec()),
        }
    }

    fn time_balanced_square<F>(
        first: Arm,
        second: Arm,
        mut time: F,
    ) -> Result<BalancedSquareTiming, String>
    where
        F: FnMut(Arm) -> Result<f64, String>,
    {
        let mut first_slots = Vec::with_capacity(4);
        let mut second_slots = Vec::with_capacity(4);
        for first_slot in BALANCED_SQUARE {
            let arm = if first_slot { first } else { second };
            let elapsed = time(arm)?;
            if first_slot {
                first_slots.push(elapsed);
            } else {
                second_slots.push(elapsed);
            }
        }
        Ok(summarize_balanced_square(first_slots, second_slots))
    }

    fn time_arm(arm: Arm, fixtures: &[Fixture], oracles: &mut [Scipy]) -> Result<f64, String> {
        match arm {
            Arm::Candidate => time_rust_job(fixtures, false),
            Arm::Control => time_rust_job(fixtures, true),
            Arm::Live => time_scipy_job(oracles),
        }
    }

    fn measure(
        fixtures: &[Fixture],
        oracles: &mut [Scipy],
        rounds: usize,
    ) -> Result<Measurement, String> {
        const ORDER: [Sample; 9] = [
            Sample::Headline(Arm::Candidate),
            Sample::NullLeft(Arm::Control),
            Sample::NullRight(Arm::Live),
            Sample::Headline(Arm::Control),
            Sample::NullLeft(Arm::Live),
            Sample::NullRight(Arm::Candidate),
            Sample::Headline(Arm::Live),
            Sample::NullLeft(Arm::Candidate),
            Sample::NullRight(Arm::Control),
        ];
        let mut measurement = Measurement::default();
        for round in 0..rounds {
            println!("measurement_round={} total_rounds={rounds}", round + 1);
            for offset in 0..ORDER.len() {
                let sample = ORDER[(offset + round) % ORDER.len()];
                let arm = match sample {
                    Sample::Headline(arm) | Sample::NullLeft(arm) | Sample::NullRight(arm) => arm,
                };
                measurement.push(sample, time_arm(arm, fixtures, oracles)?);
            }
        }
        Ok(measurement)
    }

    fn time_splu_arm(
        arm: Arm,
        fixtures: &[SpluFixture],
        oracles: &mut [Scipy],
        family: SpluFamily,
    ) -> Result<f64, String> {
        match arm {
            Arm::Candidate => time_rust_splu_job(fixtures, false, family),
            Arm::Control => time_rust_splu_job(fixtures, true, family),
            Arm::Live => time_scipy_job(oracles),
        }
    }

    fn measure_splu(
        fixtures: &[SpluFixture],
        oracles: &mut [Scipy],
        rounds: usize,
        family: SpluFamily,
    ) -> Result<Measurement, String> {
        let mut measurement = Measurement::default();
        for round in 0..rounds {
            println!("measurement_round={} total_rounds={rounds}", round + 1);
            let mut candidate_control = None;
            let mut candidate_live = None;
            let pairs = if round % 2 == 0 {
                [(Arm::Candidate, Arm::Control), (Arm::Candidate, Arm::Live)]
            } else {
                [(Arm::Candidate, Arm::Live), (Arm::Candidate, Arm::Control)]
            };
            for pair in pairs {
                let timed = time_balanced_square(pair.0, pair.1, |arm| {
                    time_splu_arm(arm, fixtures, oracles, family)
                })?;
                if matches!(pair.1, Arm::Control) {
                    candidate_control = Some(timed);
                } else {
                    candidate_live = Some(timed);
                }
            }
            let candidate_control = candidate_control.expect("candidate/control square is timed");
            let candidate_live = candidate_live.expect("candidate/live square is timed");
            measurement.candidate.push(candidate_control.first);
            measurement.control.push(candidate_control.second);
            measurement
                .candidate_null_left
                .push(candidate_control.first_null_left);
            measurement
                .candidate_null_right
                .push(candidate_control.first_null_right);
            measurement
                .control_null_left
                .push(candidate_control.second_null_left);
            measurement
                .control_null_right
                .push(candidate_control.second_null_right);
            measurement.candidate_live.push(candidate_live.first);
            measurement.live.push(candidate_live.second);
            measurement
                .candidate_live_null_left
                .push(candidate_live.first_null_left);
            measurement
                .candidate_live_null_right
                .push(candidate_live.first_null_right);
            measurement
                .live_null_left
                .push(candidate_live.second_null_left);
            measurement
                .live_null_right
                .push(candidate_live.second_null_right);
        }
        Ok(measurement)
    }

    fn time_periodic_cuboid_spsolve_arm(
        arm: Arm,
        fixtures: &[SpluFixture],
        oracles: &mut [Scipy],
    ) -> Result<f64, String> {
        match arm {
            Arm::Candidate => time_rust_periodic_cuboid_spsolve_job(fixtures, false),
            Arm::Control => time_rust_periodic_cuboid_spsolve_job(fixtures, true),
            Arm::Live => time_scipy_job(oracles),
        }
    }

    fn measure_periodic_cuboid_spsolve(
        fixtures: &[SpluFixture],
        oracles: &mut [Scipy],
        rounds: usize,
    ) -> Result<Measurement, String> {
        const ORDER: [Sample; 9] = [
            Sample::Headline(Arm::Candidate),
            Sample::NullLeft(Arm::Control),
            Sample::NullRight(Arm::Live),
            Sample::Headline(Arm::Control),
            Sample::NullLeft(Arm::Live),
            Sample::NullRight(Arm::Candidate),
            Sample::Headline(Arm::Live),
            Sample::NullLeft(Arm::Candidate),
            Sample::NullRight(Arm::Control),
        ];
        let mut measurement = Measurement::default();
        for round in 0..rounds {
            println!("measurement_round={} total_rounds={rounds}", round + 1);
            for offset in 0..ORDER.len() {
                let sample = ORDER[(offset + round) % ORDER.len()];
                let arm = match sample {
                    Sample::Headline(arm) | Sample::NullLeft(arm) | Sample::NullRight(arm) => arm,
                };
                measurement.push(
                    sample,
                    time_periodic_cuboid_spsolve_arm(arm, fixtures, oracles)?,
                );
            }
        }
        Ok(measurement)
    }

    fn print_distribution(label: &str, values: &[f64]) {
        println!(
            "{label}: p50_ms={:.6} p95_ms={:.6} p99_ms={:.6} cv_percent={:.3}",
            median(values.to_vec()) * 1e3,
            percentile(values.to_vec(), 0.95) * 1e3,
            percentile(values.to_vec(), 0.99) * 1e3,
            cv(values) * 100.0
        );
    }

    /// Does this null's bootstrap CI contain unity — i.e. is it consistent with NO arm-order
    /// bias at all?
    ///
    /// Split out of the printing so both arms are testable: a predicate that answered `true`
    /// for everything would make the diagnostic worthless in exactly the direction that
    /// flatters a failing run, so `null_ci_unity_predicate_separates_biased_from_imprecise`
    /// pins a CI that spans unity AND one that does not.
    fn null_ci_spans_unity(low: f64, high: f64) -> bool {
        low <= 1.0 && 1.0 <= high
    }

    /// Decision word for the labelled verdict line.
    ///
    /// Split out of the printing so the "this arm claims no A/B" path is testable without
    /// capturing stdout: `KEEP`/`REVERT` is a claim about a candidate beating a DIFFERENT
    /// control, and an arm whose control re-runs the candidate's own code must not print
    /// either word. Both arms of that distinction are pinned by
    /// `convection_reports_no_ab_decision_while_the_spectral_families_still_do`.
    fn decision_word(keep: bool, ab_control: bool) -> &'static str {
        if !ab_control {
            "NO_AB_DECISION"
        } else if keep {
            "KEEP"
        } else {
            "REVERT"
        }
    }

    fn print_measurement_named(
        measurement: &Measurement,
        decision_label: &str,
        minimum_candidate_seconds: f64,
        ab_control: bool,
    ) -> bool {
        let live_candidate = if measurement.candidate_live.is_empty() {
            &measurement.candidate
        } else {
            &measurement.candidate_live
        };
        let live_candidate_null_left = if measurement.candidate_live_null_left.is_empty() {
            &measurement.candidate_null_left
        } else {
            &measurement.candidate_live_null_left
        };
        let live_candidate_null_right = if measurement.candidate_live_null_right.is_empty() {
            &measurement.candidate_null_right
        } else {
            &measurement.candidate_live_null_right
        };
        let control_ratios = ratios(&measurement.control, &measurement.candidate);
        let live_ratios = ratios(&measurement.live, live_candidate);
        let candidate_nulls = ratios(
            &measurement.candidate_null_left,
            &measurement.candidate_null_right,
        );
        let control_nulls = ratios(
            &measurement.control_null_left,
            &measurement.control_null_right,
        );
        let candidate_live_nulls = ratios(live_candidate_null_left, live_candidate_null_right);
        let live_nulls = ratios(&measurement.live_null_left, &measurement.live_null_right);
        let (control_low, control_high) = bootstrap_median_ci(&control_ratios);
        let (live_low, live_high) = bootstrap_median_ci(&live_ratios);
        let (candidate_null_low, candidate_null_high) = bootstrap_median_ci(&candidate_nulls);
        let (control_null_low, control_null_high) = bootstrap_median_ci(&control_nulls);
        let (candidate_live_null_low, candidate_live_null_high) =
            bootstrap_median_ci(&candidate_live_nulls);
        let (live_null_low, live_null_high) = bootstrap_median_ci(&live_nulls);

        print_distribution("candidate_whole_job", &measurement.candidate);
        if !measurement.candidate_live.is_empty() {
            print_distribution("candidate_live_pair_whole_job", live_candidate);
        }
        print_distribution("same_elf_control_whole_job", &measurement.control);
        print_distribution("live_scipy_whole_job", &measurement.live);
        println!(
            "raw_samples_seconds: candidate={} candidate_live_pair={} control={} live={} \
             candidate_null_left={} candidate_null_right={} \
             candidate_live_null_left={} candidate_live_null_right={} \
             control_null_left={} control_null_right={} \
             live_null_left={} live_null_right={}",
            csv(&measurement.candidate),
            csv(live_candidate),
            csv(&measurement.control),
            csv(&measurement.live),
            csv(&measurement.candidate_null_left),
            csv(&measurement.candidate_null_right),
            csv(live_candidate_null_left),
            csv(live_candidate_null_right),
            csv(&measurement.control_null_left),
            csv(&measurement.control_null_right),
            csv(&measurement.live_null_left),
            csv(&measurement.live_null_right),
        );
        println!("control_over_candidate_ratios={}", csv(&control_ratios));
        println!("live_over_candidate_ratios={}", csv(&live_ratios));

        let candidate_null_median = median(candidate_nulls.clone());
        let control_null_median = median(control_nulls.clone());
        let candidate_live_null_median = median(candidate_live_nulls.clone());
        let live_null_median = median(live_nulls.clone());
        println!(
            "candidate_A/A: median={candidate_null_median:.6} \
             ci95=[{candidate_null_low:.6},{candidate_null_high:.6}] raw={}",
            csv(&candidate_nulls)
        );
        println!(
            "control_A/A: median={control_null_median:.6} \
             ci95=[{control_null_low:.6},{control_null_high:.6}] raw={}",
            csv(&control_nulls)
        );
        println!(
            "candidate_live_pair_A/A: median={candidate_live_null_median:.6} \
             ci95=[{candidate_live_null_low:.6},{candidate_live_null_high:.6}] raw={}",
            csv(&candidate_live_nulls)
        );
        println!(
            "live_A/A: median={live_null_median:.6} \
             ci95=[{live_null_low:.6},{live_null_high:.6}] raw={}",
            csv(&live_nulls)
        );

        let widest_null_edge = candidate_null_high
            .max(control_null_high)
            .max(candidate_live_null_high)
            .max(live_null_high)
            .max(1.0 / candidate_null_low.max(1.0e-12))
            .max(1.0 / control_null_low.max(1.0e-12))
            .max(1.0 / candidate_live_null_low.max(1.0e-12))
            .max(1.0 / live_null_low.max(1.0e-12))
            .max(1.0);
        let twice_null_threshold = 1.0 + 2.0 * (widest_null_edge - 1.0);
        let null_medians_pass = [
            candidate_null_median,
            control_null_median,
            candidate_live_null_median,
            live_null_median,
        ]
        .into_iter()
        .all(|value| (value - 1.0).abs() <= NULL_MEDIAN_LIMIT);
        // IS A FAILING NULL BIASED, OR MERELY IMPRECISE? The gate below tests each null's
        // MEDIAN against a 2% band, which is the right thing to gate on and is not changed
        // here. But a median 5% off unity means two completely different things depending on
        // that null's own precision, and the printed line did not distinguish them:
        //
        //   biased    - the null's CI EXCLUDES 1.0. Arm order is really doing something and
        //               the ratio above it is contaminated.
        //   imprecise - the null's CI INCLUDES 1.0. The null is consistent with no bias at
        //               all; there is nothing to fix in the schedule, and the remedy is more
        //               replicates, because the median simply is not pinned down yet.
        //
        // This distinction cost real work: on the 2026-08-23 pinned run of this cell the
        // candidate and control nulls read 0.949 and 0.939 and were reported as a cold/warm
        // ORDERING effect. Their CIs were [0.907,1.025] and [0.890,1.023] — both spanning
        // unity, as did all four nulls — so no bias had been shown at all, and a schedule fix
        // aimed at one would have been aimed at nothing.
        //
        // DIAGNOSTIC ONLY. This flag is printed, never consulted: `keep`, `maintenance_pass`,
        // `competitive_pass` and the verdict word are computed exactly as before. Widening a
        // gate to admit a result is not on the table (/data/projects/AGENTS.md); telling the
        // reader which KIND of failure they are looking at is.
        let all_null_cis_span_unity = null_ci_spans_unity(candidate_null_low, candidate_null_high)
            && null_ci_spans_unity(control_null_low, control_null_high)
            && null_ci_spans_unity(candidate_live_null_low, candidate_live_null_high)
            && null_ci_spans_unity(live_null_low, live_null_high);

        let candidate_p50 = median(measurement.candidate.clone());
        let candidate_duration_pass = candidate_p50 >= minimum_candidate_seconds;
        let maintenance_pass =
            ab_control && control_low >= 1.20 && control_low > twice_null_threshold;
        let competitive_pass = live_low > twice_null_threshold;
        if ab_control {
            println!(
                "maintenance_ratio: control/candidate median={:.6} \
                 bootstrap_median_ci95=[{control_low:.6},{control_high:.6}] \
                 registered_minimum=1.200000 \
                 twice_widest_null_threshold={twice_null_threshold:.6}",
                median(control_ratios)
            );
        } else {
            // Same code in both arms, so this is a third A/A null rather than a maintenance
            // ratio. Printed under a name that says so: calling it `maintenance_ratio` is how
            // a 1.0 gets read as "the lever bought nothing" instead of "there is no lever".
            println!(
                "same_elf_replica_ratio: control/candidate median={:.6} \
                 bootstrap_median_ci95=[{control_low:.6},{control_high:.6}] \
                 twice_widest_null_threshold={twice_null_threshold:.6} \
                 maintenance_claim=none_this_arm_has_no_ab_control",
                median(control_ratios)
            );
        }
        println!(
            "competitive_ratio: live_scipy/candidate median={:.6} \
             bootstrap_median_ci95=[{live_low:.6},{live_high:.6}] registered_minimum=1.000000",
            median(live_ratios)
        );
        println!(
            "decision_gate: null_medians_within_2pct={null_medians_pass} \
             all_null_ci95_span_unity={all_null_cis_span_unity} \
             candidate_p50_at_least_registered_minimum={candidate_duration_pass} \
             candidate_p50_ms={:.6} registered_candidate_minimum_ms={:.6} \
             maintenance_ci_low_at_least_1_20_and_beyond_2x_null={maintenance_pass} \
             competitive_ci_low_beyond_2x_null={competitive_pass} cv_used_for_decision=false",
            candidate_p50 * 1.0e3,
            minimum_candidate_seconds * 1.0e3,
        );
        let keep = null_medians_pass && candidate_duration_pass && maintenance_pass;
        // The competitive claim is about the LIVE incumbent and does not depend on there
        // being an A/B control, so an arm with no control can still pass or fail it — it
        // just needs the nulls and the sample duration behind it, not a maintenance ratio.
        let competitive_claim = competitive_pass
            && null_medians_pass
            && candidate_duration_pass
            && (keep || !ab_control);
        println!(
            "{decision_label}={} competitive_claim={}",
            decision_word(keep, ab_control),
            if competitive_claim { "PASS" } else { "FAIL" }
        );
        keep
    }

    fn print_measurement(measurement: &Measurement) -> bool {
        print_measurement_named(measurement, "CUBIC_SPSOLVE_DECISION", 0.0, true)
    }

    #[derive(Clone, Copy)]
    struct CpuTicks {
        total: u64,
        idle: u64,
    }

    fn read_cpu_ticks() -> Result<BTreeMap<usize, CpuTicks>, String> {
        let stat = std::fs::read_to_string("/proc/stat")
            .map_err(|error| format!("read /proc/stat: {error}"))?;
        let mut cpus = BTreeMap::new();
        for line in stat.lines() {
            let mut fields = line.split_whitespace();
            let Some(name) = fields.next() else {
                continue;
            };
            let Some(cpu_text) = name.strip_prefix("cpu") else {
                continue;
            };
            if cpu_text.is_empty() || !cpu_text.bytes().all(|byte| byte.is_ascii_digit()) {
                continue;
            }
            let cpu = parse::<usize>(cpu_text, "CPU index")?;
            let ticks = fields
                .map(|field| parse::<u64>(field, "CPU tick"))
                .collect::<Result<Vec<_>, _>>()?;
            if ticks.len() < 5 {
                return Err(format!("CPU {cpu} has an incomplete /proc/stat row"));
            }
            cpus.insert(
                cpu,
                CpuTicks {
                    total: ticks.iter().sum(),
                    idle: ticks[3].saturating_add(ticks[4]),
                },
            );
        }
        if cpus.is_empty() {
            return Err("/proc/stat exposed no per-CPU rows".to_string());
        }
        Ok(cpus)
    }

    fn parse_cpu_set(value: &str) -> Result<BTreeSet<usize>, String> {
        let mut cpus = BTreeSet::new();
        for segment in value.trim().split(',') {
            if let Some((start, end)) = segment.split_once('-') {
                let start = parse::<usize>(start, "CPU range start")?;
                let end = parse::<usize>(end, "CPU range end")?;
                if start > end {
                    return Err(format!("invalid CPU range {segment}"));
                }
                cpus.extend(start..=end);
            } else {
                cpus.insert(parse::<usize>(segment, "CPU")?);
            }
        }
        if cpus.is_empty() {
            return Err("CPU set is empty".to_string());
        }
        Ok(cpus)
    }

    fn cpu_affinity() -> Result<String, String> {
        let status = std::fs::read_to_string("/proc/self/status")
            .map_err(|error| format!("read /proc/self/status: {error}"))?;
        status
            .lines()
            .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
            .map(str::trim)
            .map(str::to_string)
            .ok_or_else(|| "Cpus_allowed_list missing from /proc/self/status".to_string())
    }

    fn sibling_cpus(cpu: usize) -> Result<BTreeSet<usize>, String> {
        let value = std::fs::read_to_string(format!(
            "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        ))
        .map_err(|error| format!("read SMT siblings for CPU {cpu}: {error}"))?;
        parse_cpu_set(&value)
    }

    fn host_load_sample(
        label: &str,
        attempt: usize,
        pinned_cpu: usize,
        siblings: &BTreeSet<usize>,
    ) -> Result<bool, String> {
        let before = read_cpu_ticks()?;
        std::thread::sleep(LOAD_SAMPLE);
        let after = read_cpu_ticks()?;
        if before.len() != after.len() {
            return Err("CPU topology changed during load sample".to_string());
        }
        let mut fractions = BTreeMap::new();
        for (cpu, first) in &before {
            let second = after
                .get(cpu)
                .ok_or_else(|| format!("CPU {cpu} disappeared during load sample"))?;
            let total = second.total.saturating_sub(first.total);
            let idle = second.idle.saturating_sub(first.idle);
            if total == 0 {
                return Err(format!("CPU {cpu} accumulated no ticks during load sample"));
            }
            fractions.insert(*cpu, 1.0 - idle as f64 / total as f64);
        }
        let pinned_busy = *fractions
            .get(&pinned_cpu)
            .ok_or_else(|| format!("pinned CPU {pinned_cpu} missing from load sample"))?;
        let sibling_busy = siblings
            .iter()
            .map(|cpu| {
                fractions
                    .get(cpu)
                    .copied()
                    .map(|fraction| (*cpu, fraction))
                    .ok_or_else(|| format!("sibling CPU {cpu} missing from load sample"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let host_mean = fractions.values().sum::<f64>() / fractions.len() as f64;
        let busiest_unrelated = fractions
            .iter()
            .filter(|(cpu, _)| !siblings.contains(cpu))
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(cpu, fraction)| (*cpu, *fraction));
        let admitted = pinned_busy <= LOAD_BUSY_LIMIT
            && sibling_busy
                .iter()
                .all(|(_, fraction)| *fraction <= LOAD_BUSY_LIMIT)
            && host_mean <= LOAD_BUSY_LIMIT;
        println!(
            "load_sample: phase={label} attempt={attempt} pinned_cpu={pinned_cpu} \
             pinned_busy_fraction={pinned_busy:.3} sibling_busy={} host_mean_busy_fraction={host_mean:.3} \
             busiest_unrelated={} limit={LOAD_BUSY_LIMIT:.3} admitted={admitted}",
            sibling_busy
                .iter()
                .map(|(cpu, fraction)| format!("{cpu}:{fraction:.3}"))
                .collect::<Vec<_>>()
                .join(","),
            busiest_unrelated
                .map(|(cpu, fraction)| format!("{cpu}:{fraction:.3}"))
                .unwrap_or_else(|| "none".to_string())
        );
        Ok(admitted)
    }

    fn bounded_preflight(pinned_cpu: usize, siblings: &BTreeSet<usize>) -> Result<(), String> {
        for attempt in 1..=12 {
            if host_load_sample("preflight", attempt, pinned_cpu, siblings)? {
                println!("preflight=ADMITTED attempt={attempt} maximum_attempts=12");
                return Ok(());
            }
        }
        Err("preflight exhausted twelve one-second samples".to_string())
    }

    fn require_load_gate(
        label: &str,
        pinned_cpu: usize,
        siblings: &BTreeSet<usize>,
    ) -> Result<(), String> {
        if host_load_sample(label, 1, pinned_cpu, siblings)? {
            Ok(())
        } else {
            Err(format!("single-shot {label} load gate failed"))
        }
    }

    fn observed_os_threads() -> Result<usize, String> {
        std::fs::read_dir("/proc/self/task")
            .map_err(|error| format!("read /proc/self/task: {error}"))
            .map(Iterator::count)
    }

    fn required_env(name: &str) -> Result<String, String> {
        std::env::var(name).map_err(|_| format!("required provenance variable {name} is absent"))
    }

    fn ready_value<'a>(identity: &'a str, prefix: &str) -> Option<&'a str> {
        identity
            .split_whitespace()
            .find_map(|field| field.strip_prefix(prefix))
    }

    fn is_sha256(value: &str) -> bool {
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
    }

    fn oracle_script(argument: Option<&String>) -> Result<PathBuf, String> {
        if let Some(argument) = argument {
            let path = PathBuf::from(argument);
            if path.is_file() {
                return Ok(path);
            }
            return Err(format!(
                "explicit SciPy oracle is unavailable: {}",
                path.display()
            ));
        }
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/scipy_sparse_arm.py");
        if path.is_file() {
            Ok(path)
        } else {
            Err(format!("SciPy oracle is unavailable: {}", path.display()))
        }
    }

    fn print_hardware_provenance(cpu: usize) -> Result<(), String> {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo")
            .map_err(|error| format!("read /proc/cpuinfo: {error}"))?;
        let model = cpuinfo
            .lines()
            .find_map(|line| line.strip_prefix("model name\t: "))
            .unwrap_or("unknown");
        let flags = cpuinfo
            .lines()
            .find_map(|line| line.strip_prefix("flags\t\t: "))
            .unwrap_or("");
        let meminfo = std::fs::read_to_string("/proc/meminfo")
            .map_err(|error| format!("read /proc/meminfo: {error}"))?;
        let memory_kib = meminfo
            .lines()
            .find_map(|line| line.strip_prefix("MemTotal:"))
            .and_then(|line| line.split_whitespace().next())
            .ok_or_else(|| "MemTotal missing from /proc/meminfo".to_string())?;
        let memory_bytes = parse::<u64>(memory_kib, "MemTotal KiB")?.saturating_mul(1024);
        let numa_nodes = std::fs::read_dir("/sys/devices/system/node")
            .map_err(|error| format!("read NUMA topology: {error}"))?
            .filter_map(Result::ok)
            .filter(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.strip_prefix("node")
                    .is_some_and(|suffix| suffix.bytes().all(|byte| byte.is_ascii_digit()))
            })
            .count();
        let frequency_base = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq");
        let read_frequency = |name: &str| {
            std::fs::read_to_string(format!("{frequency_base}/{name}"))
                .map(|value| value.trim().to_string())
                .unwrap_or_else(|_| "unavailable".to_string())
        };
        println!(
            "hardware_provenance: cpu_model={model:?} memory_bytes={memory_bytes} \
             numa_nodes={numa_nodes} avx2={} fma={} rust_observed_os_threads={}",
            flags.split_whitespace().any(|flag| flag == "avx2"),
            flags.split_whitespace().any(|flag| flag == "fma"),
            observed_os_threads()?
        );
        println!(
            "cpu_frequency_policy: cpu={cpu} scaling_driver={} scaling_governor={} \
             energy_performance_preference={} scaling_min_freq_khz={} scaling_max_freq_khz={}",
            read_frequency("scaling_driver"),
            read_frequency("scaling_governor"),
            read_frequency("energy_performance_preference"),
            read_frequency("scaling_min_freq"),
            read_frequency("scaling_max_freq"),
        );
        Ok(())
    }

    pub fn run(arguments: &[String]) -> Result<(), String> {
        let rounds = arguments
            .first()
            .map(|value| parse::<usize>(value, "rounds"))
            .transpose()?
            .unwrap_or(MINIMUM_ROUNDS);
        if rounds < MINIMUM_ROUNDS {
            return Err(format!(
                "cubic live gate requires at least {MINIMUM_ROUNDS} rounds"
            ));
        }

        let elf_sha256 = sha256_of_self()?;
        let source_commit = required_env("BINARY_SOURCE_COMMIT")?;
        let builder_identity = required_env("BINARY_BUILDER_IDENTITY")?;
        let build_route = required_env("BINARY_BUILD_ROUTE")?;
        let booking_claim = required_env("TRJ_BOOKING_CLAIM_MESSAGE_ID")?;
        println!("elf_sha256={elf_sha256}");
        println!("frankenscipy_engine_sha256={elf_sha256}");
        println!(
            "binary_provenance: source_commit={source_commit} builder_identity={builder_identity} \
             build_route={build_route}"
        );
        println!("trj_booking_claim_message_id={booking_claim}");
        println!(
            "linalg_source_sha256={:x}",
            Sha256::digest(LINALG_SOURCE_BYTES)
        );
        println!(
            "harness_source_sha256={:x}",
            Sha256::digest(HARNESS_SOURCE_BYTES)
        );

        let affinity = cpu_affinity()?;
        let cpus = parse_cpu_set(&affinity)?;
        if cpus.len() != 1 {
            return Err(format!(
                "all benchmark arms require one pinned physical CPU, observed affinity {affinity}"
            ));
        }
        let cpu = *cpus.first().expect("one affinity CPU");
        let siblings = sibling_cpus(cpu)?;
        if observed_os_threads()? != 1 {
            return Err("FrankenSciPy harness started with more than one OS thread".to_string());
        }
        println!(
            "thread_provenance: cpu_affinity={affinity} smt_siblings={} \
             requested_frankenscipy_threads=1 actual_observed_frankenscipy_threads=1 \
             requested_scipy_threads=1",
            siblings
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        print_hardware_provenance(cpu)?;
        bounded_preflight(cpu, &siblings)?;

        let fixtures = fixtures();
        let total_components = fixtures
            .iter()
            .map(|fixture| fixture.side.pow(3))
            .sum::<usize>();
        if total_components != EXPECTED_COMPONENTS {
            return Err(format!(
                "fixture components {total_components} != {EXPECTED_COMPONENTS}"
            ));
        }
        let shared_input_sha256 = combined_input_sha256(&fixtures);
        println!(
            "fixture: cubic_sides=12,14,16 diagonal=6.001 x=-1 y=-1 z=-1 \
             rhs=1+0.5*(i_mod_13) matrices=3 materialized_components={total_components} \
             rounds={rounds}"
        );
        println!(
            "whole_job_boundary: INCLUDED=3_public_spsolve_calls,8568_materialized_outputs,\
             folded_all_output_bits; EXCLUDED=matrix_rhs_construction,python_startup,\
             scipy_import,csr_transport,warmup,parity,provenance,bootstrap"
        );
        println!("shared_matrix_rhs_sha256={shared_input_sha256}");
        println!(
            "live_verified_fixture_sha256={}",
            fixtures
                .iter()
                .map(fixture_input_sha256)
                .collect::<Vec<_>>()
                .join(",")
        );

        SPSOLVE_CUBIC_SPECTRAL_HITS.store(0, Ordering::Relaxed);
        let candidate = rust_solutions(&fixtures, false)?;
        let candidate_hits = SPSOLVE_CUBIC_SPECTRAL_HITS.load(Ordering::Relaxed);
        SPSOLVE_CUBIC_SPECTRAL_HITS.store(0, Ordering::Relaxed);
        let control = rust_solutions(&fixtures, true)?;
        let control_hits = SPSOLVE_CUBIC_SPECTRAL_HITS.load(Ordering::Relaxed);
        if candidate_hits != 3 || control_hits != 0 {
            return Err(format!(
                "dispatch proof failed: candidate_hits={candidate_hits} control_hits={control_hits}"
            ));
        }
        let candidate_residual = fixtures
            .iter()
            .zip(&candidate)
            .map(|(fixture, solution)| relative_residual(fixture, solution))
            .fold(0.0f64, f64::max);
        let control_residual = fixtures
            .iter()
            .zip(&control)
            .map(|(fixture, solution)| relative_residual(fixture, solution))
            .fold(0.0f64, f64::max);
        let candidate_control_l2 = relative_l2(&candidate, &control);
        if candidate_residual > RESIDUAL_LIMIT
            || control_residual > RESIDUAL_LIMIT
            || candidate_control_l2 > L2_LIMIT
        {
            return Err(format!(
                "candidate/control conformance failed: candidate_residual={candidate_residual:.3e} \
                 control_residual={control_residual:.3e} relative_l2={candidate_control_l2:.3e}"
            ));
        }
        println!(
            "candidate_control_proof: candidate_hits={candidate_hits} control_hits={control_hits} \
             candidate_max_relative_residual={candidate_residual:.3e} \
             control_max_relative_residual={control_residual:.3e} \
             relative_l2={candidate_control_l2:.3e}"
        );

        let script = oracle_script(arguments.get(1))?;
        println!("scipy_oracle_script={}", script.display());
        println!("scipy_oracle_script_sha256={}", sha256_file(&script)?);
        let mut oracles = Vec::with_capacity(fixtures.len());
        let mut engine_sha256 = None;
        for (index, fixture) in fixtures.iter().enumerate() {
            let (mut oracle, identity) = Scipy::start(&script)?;
            println!("scipy_arm_{index}: {identity}");
            if !identity.starts_with("READY scipy=1.17.1 ")
                || !identity.contains("method=spsolve ")
                || !identity.contains("solver_mod=scipy.sparse.linalg._dsolve")
                || !identity.contains("actual_observed_worker_threads=1")
                || !identity.contains("fsci_loaded=False")
                || !identity.ends_with("genuine=True")
            {
                return Err(format!("live SciPy arm failed identity gate: {identity}"));
            }
            let reported_engine = ready_value(&identity, "scipy_engine_sha256=")
                .ok_or_else(|| "SciPy identity omitted engine SHA-256".to_string())?;
            if !is_sha256(reported_engine) {
                return Err("SciPy identity reported an invalid engine SHA-256".to_string());
            }
            if engine_sha256
                .as_deref()
                .is_some_and(|expected| expected != reported_engine)
            {
                return Err("SciPy oracle processes reported different engines".to_string());
            }
            engine_sha256 = Some(reported_engine.to_string());
            oracle.initialize(fixture)?;
            oracles.push(oracle);
        }
        println!(
            "scipy_engine_sha256={}",
            engine_sha256.expect("three SciPy engine identities")
        );

        let mut live = Vec::with_capacity(fixtures.len());
        let mut live_reported_residual = 0.0f64;
        for oracle in &mut oracles {
            let (solution, residual) = oracle.parity()?;
            live_reported_residual = live_reported_residual.max(residual);
            live.push(solution);
        }
        let live_recomputed_residual = fixtures
            .iter()
            .zip(&live)
            .map(|(fixture, solution)| relative_residual(fixture, solution))
            .fold(0.0f64, f64::max);
        let candidate_live_l2 = relative_l2(&candidate, &live);
        if live_reported_residual > RESIDUAL_LIMIT
            || live_recomputed_residual > RESIDUAL_LIMIT
            || candidate_live_l2 > L2_LIMIT
        {
            return Err(format!(
                "candidate/live conformance failed: reported_residual={live_reported_residual:.3e} \
                 recomputed_residual={live_recomputed_residual:.3e} relative_l2={candidate_live_l2:.3e}"
            ));
        }
        println!(
            "candidate_live_proof: genuine_scipy=1.17.1 input_sha_match=true \
             live_reported_max_relative_residual={live_reported_residual:.3e} \
             live_recomputed_max_relative_residual={live_recomputed_residual:.3e} \
             relative_l2={candidate_live_l2:.3e}"
        );

        black_box(time_rust_job(&fixtures, false)?);
        black_box(time_rust_job(&fixtures, true)?);
        black_box(time_scipy_job(&mut oracles)?);
        require_load_gate("measurement", cpu, &siblings)?;
        let measurement = measure(&fixtures, &mut oracles, rounds)?;
        require_load_gate("post", cpu, &siblings)?;
        if observed_os_threads()? != 1 || oracles.iter().any(|oracle| oracle.maximum_threads != 1) {
            return Err("observed worker count changed during measurement".to_string());
        }
        println!(
            "observed_workers: candidate=1 control=1 live_scipy=1 \
             matrix_rhs_sha256={shared_input_sha256}"
        );
        let _keep = print_measurement(&measurement);
        Ok(())
    }

    pub fn run_periodic_cuboid_spsolve(arguments: &[String]) -> Result<(), String> {
        let rounds = arguments
            .first()
            .map(|value| parse::<usize>(value, "rounds"))
            .transpose()?
            .unwrap_or(MINIMUM_ROUNDS);
        if rounds < MINIMUM_ROUNDS {
            return Err(format!(
                "periodic cuboid spsolve live gate requires at least {MINIMUM_ROUNDS} rounds"
            ));
        }

        let elf_sha256 = sha256_of_self()?;
        let source_commit = required_env("BINARY_SOURCE_COMMIT")?;
        let builder_identity = required_env("BINARY_BUILDER_IDENTITY")?;
        let build_route = required_env("BINARY_BUILD_ROUTE")?;
        let booking_claim = required_env("TRJ_BOOKING_CLAIM_MESSAGE_ID")?;
        println!("elf_sha256={elf_sha256}");
        println!("frankenscipy_engine_sha256={elf_sha256}");
        println!(
            "binary_provenance: source_commit={source_commit} builder_identity={builder_identity} \
             build_route={build_route}"
        );
        println!("trj_booking_claim_message_id={booking_claim}");
        println!(
            "linalg_source_sha256={:x}",
            Sha256::digest(LINALG_SOURCE_BYTES)
        );
        println!(
            "harness_source_sha256={:x}",
            Sha256::digest(HARNESS_SOURCE_BYTES)
        );

        let affinity = cpu_affinity()?;
        let cpus = parse_cpu_set(&affinity)?;
        if cpus.len() != 1 {
            return Err(format!(
                "all benchmark arms require one pinned physical CPU, observed affinity {affinity}"
            ));
        }
        let cpu = *cpus.first().expect("one affinity CPU");
        let siblings = sibling_cpus(cpu)?;
        if observed_os_threads()? != 1 {
            return Err("FrankenSciPy harness started with more than one OS thread".to_string());
        }
        println!(
            "thread_provenance: cpu_affinity={affinity} smt_siblings={} \
             requested_frankenscipy_threads=1 actual_observed_frankenscipy_threads=1 \
             requested_scipy_threads=1",
            siblings
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        print_hardware_provenance(cpu)?;
        bounded_preflight(cpu, &siblings)?;

        let fixtures = periodic_cuboid_spsolve_fixtures()?;
        let total_components = fixtures
            .iter()
            .map(|fixture| {
                fixture
                    .matrix
                    .shape()
                    .rows
                    .saturating_mul(fixture.right_hand_sides.len())
            })
            .sum::<usize>();
        // The guard against a silently-shrunken fixture now tracks the SELECTED extents instead
        // of a single baked-in count, so it still refuses a fixture that lost rows or RHSs while
        // allowing the extents to be chosen. At the default 9x11x13 it is the same 41,184 the
        // constant names, and that equality is asserted rather than assumed.
        let (guard_x, guard_y, guard_z) = periodic_cuboid_extents()?;
        let expected_components = guard_x
            .saturating_mul(guard_y)
            .saturating_mul(guard_z)
            .saturating_mul(PERIODIC_CUBOID_SPSOLVE_RHS_COUNT);
        if (guard_x, guard_y, guard_z) == (9, 11, 13)
            && expected_components != EXPECTED_PERIODIC_CUBOID_SPSOLVE_COMPONENTS
        {
            return Err(format!(
                "default periodic fixture must still be \
                 {EXPECTED_PERIODIC_CUBOID_SPSOLVE_COMPONENTS} components, computed \
                 {expected_components}"
            ));
        }
        if total_components != expected_components
            || fixtures.len() != 1
            || fixtures[0].right_hand_sides.len() != PERIODIC_CUBOID_SPSOLVE_RHS_COUNT
        {
            return Err(format!(
                "periodic cuboid spsolve components {total_components} != \
                 {expected_components}"
            ));
        }
        let live_input_digests = fixtures
            .iter()
            .flat_map(spsolve_many_fixture_input_sha256s)
            .collect::<Vec<_>>();
        let mut combined_hasher = Sha256::new();
        for digest in &live_input_digests {
            combined_hasher.update(digest.as_bytes());
        }
        let shared_input_sha256 = format!("{:x}", combined_hasher.finalize());
        let (row_x, row_y, row_z) = periodic_cuboid_extents()?;
        println!(
            "fixture: cuboid_extents={row_x}x{row_y}x{row_z} boundary=periodic shift=0.001 \
             diagonal=6.001 x=-0.75 y=-1 z=-1.25 \
             rhs_count={PERIODIC_CUBOID_SPSOLVE_RHS_COUNT} \
             rhs=1+0.125*((17*i+23*rhs_index)_mod_29) matrices=1 \
             materialized_components={total_components} rounds={rounds}"
        );
        println!("fsci_arm_cuboid_extents={row_x}x{row_y}x{row_z}");
        println!(
            "whole_job_boundary: INCLUDED=32_independent_public_spsolve_calls,\
             fresh_solver_state_per_call,{total_components}_materialized_outputs,\
             folded_all_output_bits; \
             EXCLUDED=matrix_rhs_construction,csc_transport,python_startup,scipy_import,\
             warmup,parity,provenance,bootstrap"
        );
        println!("shared_matrix_rhs_sha256={shared_input_sha256}");
        println!(
            "live_verified_fixture_sha256={}",
            live_input_digests.join(",")
        );

        SPSOLVE_PERIODIC_CUBOID_SPECTRAL_HITS.store(0, Ordering::Relaxed);
        let candidate = rust_periodic_cuboid_spsolve_solutions(&fixtures, false)?;
        let candidate_hits = SPSOLVE_PERIODIC_CUBOID_SPECTRAL_HITS.load(Ordering::Relaxed);
        SPSOLVE_PERIODIC_CUBOID_SPECTRAL_HITS.store(0, Ordering::Relaxed);
        let control = rust_periodic_cuboid_spsolve_solutions(&fixtures, true)?;
        let control_hits = SPSOLVE_PERIODIC_CUBOID_SPECTRAL_HITS.load(Ordering::Relaxed);
        if candidate_hits != PERIODIC_CUBOID_SPSOLVE_RHS_COUNT || control_hits != 0 {
            return Err(format!(
                "periodic spsolve dispatch proof failed: candidate_hits={candidate_hits} \
                 control_hits={control_hits}"
            ));
        }
        let candidate_residual = fixtures
            .iter()
            .zip(&candidate)
            .map(|(fixture, solutions)| splu_max_relative_residual(fixture, solutions))
            .fold(0.0f64, f64::max);
        let control_residual = fixtures
            .iter()
            .zip(&control)
            .map(|(fixture, solutions)| splu_max_relative_residual(fixture, solutions))
            .fold(0.0f64, f64::max);
        let candidate_control_l2 = relative_l2(&candidate, &control);
        if candidate_residual > RESIDUAL_LIMIT
            || control_residual > RESIDUAL_LIMIT
            || candidate_control_l2 > L2_LIMIT
        {
            return Err(format!(
                "periodic spsolve candidate/control conformance failed: \
                 candidate_residual={candidate_residual:.3e} \
                 control_residual={control_residual:.3e} relative_l2={candidate_control_l2:.3e}"
            ));
        }
        println!(
            "candidate_control_proof: candidate_hits={candidate_hits} \
             control_hits={control_hits} candidate_max_relative_residual={candidate_residual:.3e} \
             control_max_relative_residual={control_residual:.3e} \
             relative_l2={candidate_control_l2:.3e}"
        );

        let script = oracle_script(arguments.get(1))?;
        println!("scipy_oracle_script={}", script.display());
        println!("scipy_oracle_script_sha256={}", sha256_file(&script)?);
        let mut oracles = Vec::with_capacity(fixtures.len());
        let mut engine_sha256 = None;
        for (index, fixture) in fixtures.iter().enumerate() {
            let (mut oracle, identity) = Scipy::start_spsolve_many(&script)?;
            println!("scipy_arm_{index}: {identity}");
            if !identity.starts_with("READY scipy=1.17.1 ")
                || !identity.contains("method=spsolve_many ")
                || !identity.contains("solver_mod=scipy.sparse.linalg._dsolve")
                || !identity.contains("actual_observed_worker_threads=1")
                || !identity.contains("fsci_loaded=False")
                || !identity.ends_with("genuine=True")
            {
                return Err(format!(
                    "live SciPy periodic spsolve arm failed identity gate: {identity}"
                ));
            }
            let reported_engine = ready_value(&identity, "scipy_engine_sha256=")
                .ok_or_else(|| "SciPy periodic identity omitted engine SHA-256".to_string())?;
            if !is_sha256(reported_engine) {
                return Err(
                    "SciPy periodic identity reported an invalid engine SHA-256".to_string()
                );
            }
            if engine_sha256
                .as_deref()
                .is_some_and(|expected| expected != reported_engine)
            {
                return Err(
                    "SciPy periodic oracle processes reported different engines".to_string()
                );
            }
            engine_sha256 = Some(reported_engine.to_string());
            oracle.initialize_spsolve_many(fixture)?;
            oracles.push(oracle);
        }
        println!(
            "scipy_engine_sha256={}",
            engine_sha256.expect("periodic SciPy engine identity")
        );

        let mut live = Vec::with_capacity(fixtures.len());
        let mut live_reported_residual = 0.0f64;
        for oracle in &mut oracles {
            let (solution, residual, payload_bytes) = oracle.parity_splu()?;
            if payload_bytes != 0 {
                return Err(format!(
                    "one-shot live SciPy unexpectedly reported factor payload {payload_bytes}"
                ));
            }
            live_reported_residual = live_reported_residual.max(residual);
            live.push(solution);
        }
        let live_recomputed_residual = fixtures
            .iter()
            .zip(&live)
            .map(|(fixture, solutions)| splu_max_relative_residual(fixture, solutions))
            .fold(0.0f64, f64::max);
        let candidate_live_l2 = relative_l2(&candidate, &live);
        let candidate_live_component_mismatches = component_mismatches(&candidate, &live);
        if live_reported_residual > RESIDUAL_LIMIT
            || live_recomputed_residual > RESIDUAL_LIMIT
            || candidate_live_l2 > L2_LIMIT
            || candidate_live_component_mismatches != 0
        {
            return Err(format!(
                "periodic spsolve candidate/live conformance failed: \
                 reported_residual={live_reported_residual:.3e} \
                 recomputed_residual={live_recomputed_residual:.3e} \
                 relative_l2={candidate_live_l2:.3e} \
                 component_mismatches={candidate_live_component_mismatches}"
            ));
        }
        println!(
            "candidate_live_proof: genuine_scipy=1.17.1 input_sha_match=true \
             live_reported_max_relative_residual={live_reported_residual:.3e} \
             live_recomputed_max_relative_residual={live_recomputed_residual:.3e} \
             relative_l2={candidate_live_l2:.3e} \
             component_mismatches={candidate_live_component_mismatches} \
             component_tolerance=1e-10+1e-10*abs(live)"
        );

        black_box(time_rust_periodic_cuboid_spsolve_job(&fixtures, false)?);
        black_box(time_rust_periodic_cuboid_spsolve_job(&fixtures, true)?);
        black_box(time_scipy_job(&mut oracles)?);
        require_load_gate("measurement", cpu, &siblings)?;
        let measurement = measure_periodic_cuboid_spsolve(&fixtures, &mut oracles, rounds)?;
        require_load_gate("post", cpu, &siblings)?;
        if observed_os_threads()? != 1 || oracles.iter().any(|oracle| oracle.maximum_threads != 1) {
            return Err(
                "observed worker count changed during periodic spsolve measurement".to_string(),
            );
        }
        println!(
            "observed_workers: candidate=1 control=1 live_scipy=1 \
             matrix_rhs_sha256={shared_input_sha256}"
        );
        let _keep = print_measurement_named(
            &measurement,
            "PERIODIC_CUBOID_SPSOLVE_DECISION",
            0.005,
            true,
        );
        Ok(())
    }

    pub fn run_splu(arguments: &[String]) -> Result<(), String> {
        run_splu_family(arguments, SpluFamily::Dirichlet)
    }

    pub fn run_neumann_splu(arguments: &[String]) -> Result<(), String> {
        run_splu_family(arguments, SpluFamily::Neumann)
    }

    pub fn run_periodic_cuboid_splu(arguments: &[String]) -> Result<(), String> {
        run_splu_family(arguments, SpluFamily::PeriodicCuboid)
    }

    pub fn run_convection_splu(arguments: &[String]) -> Result<(), String> {
        run_splu_family(arguments, SpluFamily::Convection)
    }

    fn loadavg() -> String {
        std::fs::read_to_string("/proc/loadavg")
            .ok()
            .and_then(|text| {
                let fields = text.split_whitespace().take(3).collect::<Vec<_>>();
                (fields.len() == 3).then(|| fields.join("/"))
            })
            .unwrap_or_else(|| "unavailable".to_string())
    }

    /// Fill and whole-job cost per FILL-REDUCING ORDERING, on the run7d cell.
    ///
    /// WHY THIS AND NOT A NEW ORDERING ALGORITHM. The ledger carries an explicit REJECT on
    /// attacking the ordering ("do not attack the fill-reducing ordering -- it is already at
    /// SuperLU parity", frankenscipy-llywn) and a measured row showing that adopting COLAMD
    /// ALONE would be a PESSIMIZATION for us: it hands SuperLU 1.65x MORE element-updates
    /// (21.8M against 13.2M) and is still faster only because its blocked kernel then has
    /// supernodes to exploit, at 2.21 instructions per update against 13.45 under RCM. Our
    /// merge kernel cannot exploit them, so more fill-reduction is not automatically less work
    /// for us.
    ///
    /// That row's retry predicate is the reason this exists: "measure instructions per update
    /// for a candidate ordering FIRST -- ordering is now known to move that number by 6x".
    /// `minimum_degree_ordering` is ALREADY IMPLEMENTED and wired to `MmdAta`/`MmdAtPlusA`, so
    /// the candidate needs measuring, not writing.
    ///
    /// Reports, per ordering: the retained factor payload (fill), and the whole-job
    /// factor-plus-sixteen-solves median. No incumbent arm and no A/A null -- this decides
    /// which ordering is worth taking to the live harness, it does not claim a ratio.
    pub fn run_ordering_sweep(arguments: &[String]) -> Result<(), String> {
        let rounds = arguments
            .first()
            .map(|value| parse::<usize>(value, "rounds"))
            .transpose()?
            .unwrap_or(7);
        if rounds < 3 {
            return Err("the ordering sweep needs at least 3 rounds".to_string());
        }
        println!("# probe=ordering_fill_sweep bead=frankenscipy-run7d claim=SELF_ATTRIBUTION_ONLY");
        println!("elf_sha256={}", sha256_of_self()?);
        println!("loadavg_before={}", loadavg());

        let fixtures = splu_fixtures(SpluFamily::Convection)?;
        let fixture = fixtures
            .first()
            .ok_or_else(|| "the convection family has no fixture".to_string())?;
        let n = fixture.matrix.shape().rows;

        for ordering in [
            PermutationOrdering::Colamd,
            PermutationOrdering::ReverseCuthillMcKee,
            PermutationOrdering::MmdAtPlusA,
            PermutationOrdering::MmdAta,
            PermutationOrdering::Natural,
        ] {
            let options = LuOptions {
                ordering,
                ..LuOptions::default()
            };
            let mut factor_ms = Vec::with_capacity(rounds);
            let mut solve_ms = Vec::with_capacity(rounds);
            let mut payload = 0usize;
            let mut checksum = 0u64;
            let mut failed = None;
            for round in 0..=rounds {
                let started = Instant::now();
                let factor = match splu(black_box(&fixture.csc), options) {
                    Ok(factor) => factor,
                    Err(error) => {
                        failed = Some(format!("{error}"));
                        break;
                    }
                };
                let factored = started.elapsed().as_secs_f64() * 1.0e3;
                payload = splu_factor_payload_bytes(&factor);
                let started = Instant::now();
                for right_hand_side in &fixture.right_hand_sides {
                    match splu_solve(&factor, black_box(right_hand_side)) {
                        Ok(solution) => {
                            for value in solution {
                                checksum = checksum.rotate_left(1) ^ value.to_bits();
                            }
                        }
                        Err(error) => {
                            failed = Some(format!("{error}"));
                            break;
                        }
                    }
                }
                let solved = started.elapsed().as_secs_f64() * 1.0e3;
                if failed.is_some() {
                    break;
                }
                if round == 0 {
                    continue;
                }
                factor_ms.push(factored);
                solve_ms.push(solved);
            }
            black_box(checksum);
            if let Some(error) = failed {
                println!("ordering={ordering:?} REFUSED: {error}");
                continue;
            }
            // Entries back out of the packed payload: 12 bytes per entry (u32 column +
            // f64 value) plus one usize offset per row on each triangle, plus row_perm.
            let overhead = 2 * 8 * (n + 1) + 8 * n;
            let entries = payload.saturating_sub(overhead) / 12;
            let factor_median = median(factor_ms.clone());
            let solve_median = median(solve_ms.clone());
            println!(
                "ordering={ordering:?} lu_entries={entries} payload_bytes={payload} \
                 factor_p50_ms={factor_median:.6} sixteen_solves_p50_ms={solve_median:.6} \
                 job_p50_ms={:.6}",
                factor_median + solve_median
            );
        }
        println!("loadavg_after={}", loadavg());
        Ok(())
    }

    /// GATE (a) for `frankenscipy-run7d`: what fraction of this cell is the SOLVE?
    ///
    /// The cell is one factorization plus SIXTEEN solves, so "optimize the solve path" is worth
    /// doing only in proportion to what those solves actually cost HERE. The profile banked in
    /// `docs/NEGATIVE_EVIDENCE.md` put the solve at ~31% with about twofold headroom — but it
    /// was taken on a DIFFERENT fixture (n=1,728), and a share does not transfer across sizes:
    /// factorization grows superlinearly in fill while the solve is linear in factor nonzeros,
    /// so the share should FALL as n grows. This measures it on the fixture the ratio is
    /// actually quoted on (n=4,096), because a lever sized off the wrong fixture is how a 4%
    /// change gets budgeted as a 16% one.
    ///
    /// The solve arm is timed TWICE per round from the SAME factor, and the ratio of those two
    /// passes is reported as an A/A null. Without it a share is just a number: the second pass
    /// bounds how much of the spread is the host rather than the code.
    ///
    /// This claims NOTHING against the incumbent — it is a self-attribution probe whose only
    /// output is a proportion, and it must not be quoted as a ratio.
    pub fn run_convection_split(arguments: &[String]) -> Result<(), String> {
        let rounds = arguments
            .first()
            .map(|value| parse::<usize>(value, "rounds"))
            .transpose()?
            .unwrap_or(MINIMUM_ROUNDS);
        if rounds < 3 {
            return Err("the split probe needs at least 3 rounds".to_string());
        }

        println!("# probe=convection_factor_solve_split bead=frankenscipy-run7d");
        println!("elf_sha256={}", sha256_of_self()?);
        println!(
            "# host={} observed_os_threads={} rounds={rounds} claim=SELF_ATTRIBUTION_ONLY",
            std::fs::read_to_string("/proc/sys/kernel/hostname")
                .map(|value| value.trim().to_string())
                .unwrap_or_else(|_| "unknown".to_string()),
            observed_os_threads()?,
        );
        println!("loadavg_before={}", loadavg());

        let fixtures = splu_fixtures(SpluFamily::Convection)?;
        let fixture = fixtures
            .first()
            .ok_or_else(|| "the convection family has no fixture".to_string())?;
        let n = fixture.matrix.shape().rows;

        let mut factor_ms = Vec::with_capacity(rounds);
        let mut solve_ms = Vec::with_capacity(rounds);
        let mut nulls = Vec::with_capacity(rounds);
        let mut orig_solve_ms = Vec::with_capacity(rounds);
        let mut ab = Vec::with_capacity(rounds);
        let mut factor_payload = 0usize;
        let mut dispatch_observed = false;

        // Round 0 is discarded, so the allocator and the page cache are warm in both phases
        // before anything is recorded.
        for round in 0..=rounds {
            let started = Instant::now();
            let factor = splu(black_box(&fixture.csc), LuOptions::default())
                .map_err(|error| format!("split probe splu: {error}"))?;
            let factored = started.elapsed().as_secs_f64() * 1.0e3;
            factor_payload = splu_factor_payload_bytes(&factor);

            let mut checksum = 0u64;
            let mut sixteen_solves = |factor: &SparseLuFactorization| -> Result<f64, String> {
                let started = Instant::now();
                for right_hand_side in &fixture.right_hand_sides {
                    let solution = splu_solve(factor, black_box(right_hand_side))
                        .map_err(|error| format!("split probe splu_solve: {error}"))?;
                    for value in solution {
                        checksum = checksum.rotate_left(1) ^ value.to_bits();
                    }
                }
                Ok(started.elapsed().as_secs_f64() * 1.0e3)
            };
            // ABBA over the composed-index lever, inside ONE window. A cross-window
            // before/after on this change is not readable: between two such windows the
            // UNTOUCHED factorization moved 21% on this host, which is larger than the
            // lever. A1 and A2 bracket B1 and B2, so a monotone drift over the quartet
            // cancels in the A/B ratio and surfaces in the A/A null.
            let arm = |materialize: bool,
                       solves: &mut dyn FnMut(&SparseLuFactorization) -> Result<f64, String>|
             -> Result<f64, String> {
                SPLU_SOLVE_FORCE_MATERIALIZED_RHS.store(materialize, Ordering::Relaxed);
                let elapsed = solves(&factor);
                SPLU_SOLVE_FORCE_MATERIALIZED_RHS.store(false, Ordering::Relaxed);
                elapsed
            };
            if round == 0 {
                // TWO-ARM PROBE CONTROL. Before any timing is believed, prove the library
                // actually CONSULTS the toggle: if it does not, both "arms" are the same code
                // and every ratio below is an A/A comparison wearing an A/B label.
                dispatch_observed = SPLU_SOLVE_FORCE_MATERIALIZED_RHS.dispatch_observed(|| {
                    let _ = splu_solve(&factor, &fixture.right_hand_sides[0]);
                });
            }
            // POSITION-BALANCED, not merely ABBA. ABBA cancels a monotone DRIFT, and the
            // effect here is not drift: the first sixteen-solve pass after a fresh
            // factorization is systematically slower than an immediately repeated one
            // (measured: an A/A null of 1.185, 20 of 21 replicates above unity, on a probe
            // that always ran the ORIG arm in the cold outer slot). With a fixed schedule
            // that bias lands entirely on whichever arm owns slot 1, and it is LARGER than
            // the lever — the first version of this probe reported the candidate at 1.126
            // against a null of 1.185, i.e. the schedule, not the code.
            //
            // So the quartet flips with the round: ABBA on even rounds, BAAB on odd. Each
            // arm then owns the cold slot in half the replicates and the position effect
            // cancels in the aggregate ratio instead of being attributed to the candidate.
            // The A/A null is still same-arm-over-same-arm and still SEES the effect, which
            // is what makes the candidate's separation from it meaningful.
            let orig_first = round % 2 == 0;
            let (orig_1, head_1, head_2, orig_2) = if orig_first {
                let a1 = arm(true, &mut sixteen_solves)?;
                let b1 = arm(false, &mut sixteen_solves)?;
                let b2 = arm(false, &mut sixteen_solves)?;
                let a2 = arm(true, &mut sixteen_solves)?;
                (a1, b1, b2, a2)
            } else {
                let b1 = arm(false, &mut sixteen_solves)?;
                let a1 = arm(true, &mut sixteen_solves)?;
                let a2 = arm(true, &mut sixteen_solves)?;
                let b2 = arm(false, &mut sixteen_solves)?;
                (a1, b1, b2, a2)
            };
            black_box(checksum);

            if round == 0 {
                continue;
            }
            factor_ms.push(factored);
            let head = (head_1 + head_2) / 2.0;
            let orig = (orig_1 + orig_2) / 2.0;
            solve_ms.push(head);
            orig_solve_ms.push(orig);
            ab.push(orig / head);
            // The null is taken from whichever arm occupied the two OUTER slots this round,
            // so it carries the same cold/warm exposure the ratio has to clear.
            nulls.push(if orig_first {
                orig_1 / orig_2
            } else {
                head_1 / head_2
            });
        }

        let factor_median = median(factor_ms.clone());
        let solve_median = median(solve_ms.clone());
        let job = factor_median + solve_median;
        println!(
            "split: n={n} rhs_count={SPLU_RHS_COUNT} factor_p50_ms={factor_median:.6} \
             sixteen_solves_p50_ms={solve_median:.6} factor_plus_solves_p50_ms={job:.6} \
             solve_share={:.4} per_solve_p50_ms={:.6} factor_payload_bytes={factor_payload}",
            solve_median / job,
            solve_median / SPLU_RHS_COUNT as f64,
        );
        // A/B DECIDED only when the candidate ratios and the A/A nulls SEPARATE at the
        // replicate level: the candidate's p10 must clear the null's p90 or vice versa. A
        // median that merely sits outside a null's extremes is one outlier away from either
        // verdict.
        let ab_median = median(ab.clone());
        let null_median = median(nulls.clone());
        let (ab_low, ab_high) = (percentile(ab.clone(), 0.10), percentile(ab.clone(), 0.90));
        let (null_low, null_high) = (
            percentile(nulls.clone(), 0.10),
            percentile(nulls.clone(), 0.90),
        );
        let decided = ab_low > null_high || ab_high < null_low;
        println!(
            "AB(materialized/composed): median={ab_median:.6} p10_p90=[{ab_low:.6},{ab_high:.6}] \
             orig_sixteen_solves_p50_ms={:.6} verdict={} dispatch_observed={dispatch_observed}",
            median(orig_solve_ms.clone()),
            if !dispatch_observed {
                "VOID_NO_DISPATCH"
            } else if decided {
                "DECIDED"
            } else {
                "IN-FLOOR"
            },
        );
        println!(
            "solve_A/A: median={null_median:.6} p10_p90=[{null_low:.6},{null_high:.6}] raw={}",
            csv(&nulls)
        );
        println!("factor_raw_ms={}", csv(&factor_ms));
        println!("composed_solve_raw_ms={}", csv(&solve_ms));
        println!("materialized_solve_raw_ms={}", csv(&orig_solve_ms));
        println!("loadavg_after={}", loadavg());
        Ok(())
    }

    fn run_splu_family(arguments: &[String], family: SpluFamily) -> Result<(), String> {
        let rounds = arguments
            .first()
            .map(|value| parse::<usize>(value, "rounds"))
            .transpose()?
            .unwrap_or(MINIMUM_ROUNDS);
        if rounds < MINIMUM_ROUNDS {
            return Err(format!(
                "{} splu live gate requires at least {MINIMUM_ROUNDS} rounds",
                family.name()
            ));
        }

        let elf_sha256 = sha256_of_self()?;
        let source_commit = required_env("BINARY_SOURCE_COMMIT")?;
        let builder_identity = required_env("BINARY_BUILDER_IDENTITY")?;
        let build_route = required_env("BINARY_BUILD_ROUTE")?;
        let booking_claim = required_env("TRJ_BOOKING_CLAIM_MESSAGE_ID")?;
        println!("elf_sha256={elf_sha256}");
        println!("frankenscipy_engine_sha256={elf_sha256}");
        println!(
            "binary_provenance: source_commit={source_commit} builder_identity={builder_identity} \
             build_route={build_route}"
        );
        println!("trj_booking_claim_message_id={booking_claim}");
        println!(
            "linalg_source_sha256={:x}",
            Sha256::digest(LINALG_SOURCE_BYTES)
        );
        println!(
            "harness_source_sha256={:x}",
            Sha256::digest(HARNESS_SOURCE_BYTES)
        );
        // SOURCE MARKER ON THE ROW ITSELF, not only behind `--source-marker`.
        //
        // A source sha proves which bytes the compiler read; it does not help a reader who is
        // holding a printed row and asking "was this binary built from the source I think it
        // was". The marker is a value I can FLIP in one edit, so the freshness check has two
        // observable arms: build with `-a` and the row says `-a`, change the const and the row
        // must say the new value. This is the check that catches the failure actually seen in
        // this session — `rch exec -- cargo test` returned exit 0 and 5/5 green while running
        // a test that had already been deleted from the working tree (frankenscipy-ozg54),
        // because the remote arm built the COMMITTED tree. A stale binary now contradicts its
        // own row instead of passing silently.
        println!(
            "perf_spsolve_source_marker={}",
            super::PERF_SPSOLVE_SOURCE_MARKER
        );

        if observed_os_threads()? != 1 {
            return Err("FrankenSciPy harness started with more than one OS thread".to_string());
        }
        println!(
            "thread_provenance: cpu_affinity={} \
             requested_frankenscipy_threads=1 actual_observed_frankenscipy_threads=1 \
             requested_scipy_threads=1",
            cpu_affinity()?
        );
        println!(
            "balanced_square_provenance: schedule=ABBAABBA \
             host_quiescence_required=false null_gate=per_arm_first_half_over_second_half \
             null_median_limit={NULL_MEDIAN_LIMIT:.3}"
        );
        // The ordering is part of what was measured, so it goes on the row. A row that
        // does not name it cannot be compared with one taken under a different one.
        println!("fsci_arm_ordering={:?}", splu_arm_ordering()?);

        let fixtures = splu_fixtures(family)?;
        let total_components = fixtures
            .iter()
            .map(|fixture| {
                fixture
                    .matrix
                    .shape()
                    .rows
                    .saturating_mul(fixture.right_hand_sides.len())
            })
            .sum::<usize>();
        let expected_components = family.expected_components();
        if total_components != expected_components
            || fixtures
                .iter()
                .any(|fixture| fixture.right_hand_sides.len() != SPLU_RHS_COUNT)
        {
            return Err(format!(
                "splu fixture components {total_components} != {expected_components}"
            ));
        }
        let shared_input_sha256 = combined_splu_input_sha256(&fixtures);
        match family {
            SpluFamily::Dirichlet => println!(
                "fixture: cubic_sides=12,14,16 boundary=Dirichlet diagonal=6.001 \
                 x=-1 y=-1 z=-1 rhs_count_per_factor={SPLU_RHS_COUNT} \
                 rhs=1+0.125*((17*i+23*rhs_index)_mod_29) matrices=3 \
                 materialized_components={total_components} rounds={rounds}"
            ),
            SpluFamily::Neumann => println!(
                "fixture: cubic_sides=12,14,16 boundary=Neumann shift=0.001 \
                 diagonal=shift+vertex_degree x=-1 y=-1 z=-1 \
                 rhs_count_per_factor={SPLU_RHS_COUNT} \
                 rhs=1+0.125*((17*i+23*rhs_index)_mod_29) matrices=3 \
                 materialized_components={total_components} rounds={rounds}"
            ),
            SpluFamily::PeriodicCuboid => println!(
                "fixture: cuboid_extents=9x11x13,11x13x15,13x15x17 boundary=periodic \
                 shift=0.001 diagonal=6.001 x=-0.75 y=-1 z=-1.25 \
                 rhs_count_per_factor={SPLU_RHS_COUNT} \
                 rhs=1+0.125*((17*i+23*rhs_index)_mod_29) matrices=3 \
                 materialized_components={total_components} rounds={rounds}"
            ),
            SpluFamily::Convection => println!(
                "fixture: grid_side=64 boundary=Dirichlet convection_diffusion=true \
                 diagonal=4.001 west=-1.2 east=-0.8 vertical=-1 \
                 rhs_count_per_factor={SPLU_RHS_COUNT} \
                 rhs=1+0.125*((17*i+23*rhs_index)_mod_29) matrices=1 \
                 materialized_components={total_components} rounds={rounds}"
            ),
        }
        let factor_count = fixtures.len();
        let solve_count = fixtures.len().saturating_mul(SPLU_RHS_COUNT);
        println!(
            "whole_job_boundary: INCLUDED={factor_count}_public_splu_calls,\
             {solve_count}_public_splu_solve_calls,\
             {total_components}_materialized_outputs,folded_all_output_bits; \
             EXCLUDED=matrix_rhs_construction,csc_transport,python_startup,scipy_import,\
             warmup,parity,provenance,bootstrap"
        );
        println!("shared_matrix_rhs_sha256={shared_input_sha256}");
        println!(
            "live_verified_fixture_sha256={}",
            fixtures
                .iter()
                .map(splu_fixture_input_sha256)
                .collect::<Vec<_>>()
                .join(",")
        );

        // TWO-ARM PROBE CONTROL, applied only where an A/B is actually claimed. A family
        // whose control arm is a deliberate A/A replica has no toggle to consult, so demanding
        // a dispatch there would refuse a perfectly good vs-incumbent row -- which is exactly
        // what happened when Convection was briefly wired to a default-unreachable kernel.
        if family.ab_control() && matches!(family, SpluFamily::Convection) {
            let probe_fixture = fixtures
                .first()
                .ok_or_else(|| "convection family has no fixture".to_string())?;
            let dispatch_observed = SPLU_MERGE_FORCE_LEGACY_WALK.dispatch_observed(|| {
                if let Ok(factor) = splu(&probe_fixture.csc, LuOptions::default()) {
                    let _ = splu_solve(&factor, &probe_fixture.right_hand_sides[0]);
                }
            });
            println!("merge_ab_dispatch_observed={dispatch_observed}");
            if !dispatch_observed {
                return Err(
                    "the merge-kernel A/B toggle was never consulted: the two arms are the \
                     same code and no ratio from them is reportable"
                        .to_string(),
                );
            }
        }

        family.reset_hits();
        let (candidate, candidate_payload_bytes) = rust_splu_solutions(&fixtures, false, family)?;
        let (candidate_factor_hits, candidate_solve_hits) = family.hits();
        family.reset_hits();
        let (control, control_payload_bytes) = rust_splu_solutions(&fixtures, true, family)?;
        let (control_factor_hits, control_solve_hits) = family.hits();
        let (expected_factor_hits, expected_solve_hits) = family.expected_hits();
        if candidate_factor_hits != expected_factor_hits
            || candidate_solve_hits != expected_solve_hits
            || control_factor_hits != 0
            || control_solve_hits != 0
        {
            return Err(format!(
                "splu dispatch proof failed: candidate_factor_hits={candidate_factor_hits} \
                 candidate_solve_hits={candidate_solve_hits} \
                 expected_factor_hits={expected_factor_hits} \
                 expected_solve_hits={expected_solve_hits} \
                 control_factor_hits={control_factor_hits} control_solve_hits={control_solve_hits}"
            ));
        }
        let candidate_residual = fixtures
            .iter()
            .zip(&candidate)
            .map(|(fixture, solutions)| splu_max_relative_residual(fixture, solutions))
            .fold(0.0f64, f64::max);
        let control_residual = fixtures
            .iter()
            .zip(&control)
            .map(|(fixture, solutions)| splu_max_relative_residual(fixture, solutions))
            .fold(0.0f64, f64::max);
        let candidate_control_l2 = relative_l2(&candidate, &control);
        if candidate_residual > RESIDUAL_LIMIT
            || control_residual > RESIDUAL_LIMIT
            || candidate_control_l2 > L2_LIMIT
        {
            return Err(format!(
                "splu candidate/control conformance failed: \
                 candidate_residual={candidate_residual:.3e} \
                 control_residual={control_residual:.3e} relative_l2={candidate_control_l2:.3e}"
            ));
        }
        println!(
            "candidate_control_proof: candidate_factor_hits={candidate_factor_hits} \
             candidate_solve_hits={candidate_solve_hits} control_factor_hits={control_factor_hits} \
             control_solve_hits={control_solve_hits} \
             candidate_max_relative_residual={candidate_residual:.3e} \
             control_max_relative_residual={control_residual:.3e} \
             relative_l2={candidate_control_l2:.3e} \
             candidate_factor_vector_payload_bytes={candidate_payload_bytes} \
             control_factor_vector_payload_bytes={control_payload_bytes} memory_claim=false"
        );

        let script = oracle_script(arguments.get(1))?;
        println!("scipy_oracle_script={}", script.display());
        println!("scipy_oracle_script_sha256={}", sha256_file(&script)?);
        let mut oracles = Vec::with_capacity(fixtures.len());
        let mut engine_sha256 = None;
        for (index, fixture) in fixtures.iter().enumerate() {
            let (mut oracle, identity) = Scipy::start_splu(&script)?;
            println!("scipy_arm_{index}: {identity}");
            if !identity.starts_with("READY scipy=1.17.1 ")
                || !identity.contains("method=splu ")
                || !identity.contains("solver_mod=scipy.sparse.linalg._dsolve")
                || !identity.contains("actual_observed_worker_threads=1")
                || !identity.contains("fsci_loaded=False")
                || !identity.ends_with("genuine=True")
            {
                return Err(format!(
                    "live SciPy splu arm failed identity gate: {identity}"
                ));
            }
            let reported_engine = ready_value(&identity, "scipy_engine_sha256=")
                .ok_or_else(|| "SciPy splu identity omitted engine SHA-256".to_string())?;
            if !is_sha256(reported_engine) {
                return Err("SciPy splu identity reported an invalid engine SHA-256".to_string());
            }
            if engine_sha256
                .as_deref()
                .is_some_and(|expected| expected != reported_engine)
            {
                return Err("SciPy splu oracle processes reported different engines".to_string());
            }
            engine_sha256 = Some(reported_engine.to_string());
            oracle.initialize_splu(fixture, family.is_nonsymmetric())?;
            oracles.push(oracle);
        }
        println!(
            "scipy_engine_sha256={}",
            engine_sha256.expect("at least one SciPy splu engine identity")
        );

        let mut live = Vec::with_capacity(fixtures.len());
        let mut live_reported_residual = 0.0f64;
        let mut live_payload_bytes = 0usize;
        for oracle in &mut oracles {
            let (solution, residual, payload_bytes) = oracle.parity_splu()?;
            live_reported_residual = live_reported_residual.max(residual);
            live_payload_bytes = live_payload_bytes.saturating_add(payload_bytes);
            live.push(solution);
        }
        let live_recomputed_residual = fixtures
            .iter()
            .zip(&live)
            .map(|(fixture, solutions)| splu_max_relative_residual(fixture, solutions))
            .fold(0.0f64, f64::max);
        let candidate_live_l2 = relative_l2(&candidate, &live);
        let candidate_live_component_mismatches = component_mismatches(&candidate, &live);
        if live_reported_residual > RESIDUAL_LIMIT
            || live_recomputed_residual > RESIDUAL_LIMIT
            || candidate_live_l2 > L2_LIMIT
            || candidate_live_component_mismatches != 0
        {
            return Err(format!(
                "splu candidate/live conformance failed: \
                 reported_residual={live_reported_residual:.3e} \
                 recomputed_residual={live_recomputed_residual:.3e} \
                 relative_l2={candidate_live_l2:.3e} \
                 component_mismatches={candidate_live_component_mismatches}"
            ));
        }
        println!(
            "candidate_live_proof: genuine_scipy=1.17.1 input_sha_match=true \
             live_reported_max_relative_residual={live_reported_residual:.3e} \
             live_recomputed_max_relative_residual={live_recomputed_residual:.3e} \
             relative_l2={candidate_live_l2:.3e} \
             component_mismatches={candidate_live_component_mismatches} \
             component_tolerance=1e-10+1e-10*abs(live) \
             scipy_l_plus_u_array_payload_bytes={live_payload_bytes} memory_claim=false"
        );

        black_box(time_rust_splu_job(&fixtures, false, family)?);
        black_box(time_rust_splu_job(&fixtures, true, family)?);
        black_box(time_scipy_job(&mut oracles)?);
        let measurement = measure_splu(&fixtures, &mut oracles, rounds, family)?;
        if observed_os_threads()? != 1 || oracles.iter().any(|oracle| oracle.maximum_threads != 1) {
            return Err("observed worker count changed during splu measurement".to_string());
        }
        println!(
            "observed_workers: candidate=1 control=1 live_scipy=1 \
             matrix_rhs_sha256={shared_input_sha256}"
        );
        let _keep = print_measurement_named(
            &measurement,
            family.decision_label(),
            0.005,
            family.ab_control(),
        );
        Ok(())
    }

    #[cfg(test)]
    mod balanced_square_tests {
        use super::{Arm, BALANCED_SQUARE, summarize_balanced_square, time_balanced_square};

        #[test]
        fn balanced_square_times_mirrored_arm_positions() {
            let mut observed_slots = Vec::new();
            let timing = time_balanced_square(Arm::Candidate, Arm::Control, |arm| {
                let candidate = matches!(arm, Arm::Candidate);
                observed_slots.push(candidate);
                Ok(if candidate { 2.0 } else { 5.0 })
            })
            .expect("the deterministic timing closure succeeds");

            assert_eq!(observed_slots, BALANCED_SQUARE);
            assert_eq!(timing.first, 2.0);
            assert_eq!(timing.second, 5.0);
        }

        #[test]
        fn balanced_square_keeps_each_arm_in_both_halves() {
            let left_positions = BALANCED_SQUARE
                .iter()
                .enumerate()
                .filter_map(|(index, first)| (*first).then_some(index))
                .collect::<Vec<_>>();
            let right_positions = BALANCED_SQUARE
                .iter()
                .enumerate()
                .filter_map(|(index, first)| (!*first).then_some(index))
                .collect::<Vec<_>>();
            assert_eq!(left_positions, [0, 3, 4, 7]);
            assert_eq!(right_positions, [1, 2, 5, 6]);

            let balanced = summarize_balanced_square(vec![2.0; 4], vec![5.0; 4]);
            assert_eq!(balanced.first_null_left / balanced.first_null_right, 1.0);
            assert_eq!(balanced.second_null_left / balanced.second_null_right, 1.0);
        }

        #[test]
        fn balanced_square_null_exposes_single_arm_drift() {
            let drifted = summarize_balanced_square(vec![1.0, 1.0, 1.5, 1.5], vec![2.0; 4]);
            assert!(
                (drifted.first_null_left / drifted.first_null_right - 1.0).abs()
                    > super::NULL_MEDIAN_LIMIT
            );
            assert_eq!(drifted.second_null_left / drifted.second_null_right, 1.0);
        }

        /// The narrowed successor to `convection_live_refuses_deleted_control`.
        ///
        /// That test pinned a BLANKET refusal of the convection arm, added because its
        /// lazy-column A/B control had been deleted from production. The intent was right —
        /// do not claim an A/B decision on a control that no longer exists — but the refusal
        /// also blocked the plain vs-incumbent measurement, which never read that toggle.
        /// This keeps the intent and drops the over-reach: the arm runs, and it is pinned
        /// to claim no A/B.
        ///
        /// TWO ARMS, because a predicate that answered "no decision" for everything would
        /// pass a one-armed version of this test: the spectral families must still claim one.
        #[test]
        fn convection_reports_no_ab_decision_while_the_spectral_families_still_do() {
            use super::SpluFamily;

            // MUST-MISS: Convection has no A/B control, so no KEEP/REVERT at either truth
            // value. It briefly did -- wired to the merge-kernel toggle -- and that was
            // reverted when the toggle proved default-unreachable.
            assert!(!SpluFamily::Convection.ab_control());
            assert_eq!(super::decision_word(true, false), "NO_AB_DECISION");
            assert_eq!(super::decision_word(false, false), "NO_AB_DECISION");

            // MUST-HIT: the spectral families are unchanged and still decide.
            for family in [
                SpluFamily::Dirichlet,
                SpluFamily::Neumann,
                SpluFamily::PeriodicCuboid,
            ] {
                assert!(family.ab_control(), "{} must keep its A/B", family.name());
            }
            assert_eq!(super::decision_word(true, true), "KEEP");
            assert_eq!(super::decision_word(false, true), "REVERT");
        }

        /// The other half of the deleted test's intent: the convection arm must not advertise
        /// a spectral fast path it does not have, and its label must not name the deleted
        /// control. A stale label is what a later reader greps for and believes.
        #[test]
        fn convection_claims_no_spectral_hits_and_no_deleted_control_label() {
            use super::SpluFamily;

            SpluFamily::Convection.reset_hits();
            assert_eq!(SpluFamily::Convection.expected_hits(), (0, 0));
            assert_eq!(SpluFamily::Convection.hits(), (0, 0));
            // There is no toggle behind this family; flipping the switch must not invent one.
            SpluFamily::Convection.set_disabled(true);
            assert_eq!(SpluFamily::Convection.hits(), (0, 0));
            SpluFamily::Convection.set_disabled(false);

            let label = SpluFamily::Convection.decision_label();
            assert_eq!(label, "CONVECTION_SPLU_VS_INCUMBENT");
            assert!(
                !label.contains("LAZY_COLUMNS"),
                "the label must not name a control that was deleted: {label}"
            );
            // MUST-HIT arm for the same assertion: a family that DOES decide still says so.
            assert!(
                SpluFamily::Dirichlet
                    .decision_label()
                    .ends_with("_DECISION")
            );
        }

        /// The diagnostic added for `frankenscipy-run7d` must separate the two ways a null
        /// median can sit outside the 2% band, because they call for opposite responses:
        /// a CI excluding unity means the schedule really is biased and the ratio above it is
        /// contaminated; a CI including unity means the null is merely imprecise and the
        /// remedy is more replicates.
        ///
        /// BOTH ARMS, using the real numbers that motivated it. The pinned 2026-08-23 run of
        /// the convection cell had all four nulls spanning unity while three of them failed
        /// the 2% median gate — that is the MUST-HIT case, and reading those failures as an
        /// ordering effect (which is what happened) is the mistake this pins against. The
        /// MUST-MISS case is a genuinely biased null.
        #[test]
        fn null_ci_unity_predicate_separates_biased_from_imprecise() {
            // MUST-HIT: every null from that run, medians far off unity, CIs spanning it.
            for (median, low, high) in [
                (0.949420, 0.906662, 1.025266), // candidate_A/A   — failed the 2% gate
                (0.938835, 0.890318, 1.023263), // control_A/A     — failed the 2% gate
                (1.021569, 0.993653, 1.056437), // live_A/A        — failed the 2% gate
                (0.994113, 0.981449, 1.129099), // candidate_live_pair — passed it
            ] {
                assert!(
                    super::null_ci_spans_unity(low, high),
                    "median {median} with ci [{low},{high}] is consistent with no bias"
                );
            }
            // ... and three of those four really did fail the median gate, so the diagnostic
            // is reporting something the gate line did not already say.
            assert!(!(0.949420f64 - 1.0).abs().le(&0.02));
            assert!(!(0.938835f64 - 1.0).abs().le(&0.02));
            assert!(!(1.021569f64 - 1.0).abs().le(&0.02));

            // MUST-MISS: a null whose CI clears unity entirely is BIASED, and the predicate
            // has to say so. Without this arm a predicate returning `true` unconditionally
            // would pass — and would silently excuse every contaminated run.
            assert!(!super::null_ci_spans_unity(1.041, 1.078));
            assert!(!super::null_ci_spans_unity(0.902, 0.981));
            // Touching unity from either side still counts as spanning it.
            assert!(super::null_ci_spans_unity(1.0, 1.2));
            assert!(super::null_ci_spans_unity(0.8, 1.0));
        }

        #[test]
        fn source_freshness_detects_changed_bytes() {
            assert!(super::source_is_fresh(b"same", b"same"));
            assert!(!super::source_is_fresh(b"compiled", b"worker"));
        }
    }
}

fn main() {
    let raw_arguments = std::env::args().collect::<Vec<_>>();
    if raw_arguments.get(1).map(String::as_str) == Some("--source-freshness") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            if let Err(error) = cubic_live::run_source_freshness(&raw_arguments[2..]) {
                eprintln!("SOURCE_FRESHNESS_FATAL {error}");
                std::process::exit(1);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("--source-freshness requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if raw_arguments.get(1).map(String::as_str) == Some("--source-marker") {
        println!("perf_spsolve_source_marker={PERF_SPSOLVE_SOURCE_MARKER}");
        return;
    }
    if raw_arguments.get(1).map(String::as_str) == Some("--ordering-sweep") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            if let Err(error) = cubic_live::run_ordering_sweep(&raw_arguments[2..]) {
                eprintln!("ORDERING_SWEEP_FATAL {error}");
                std::process::exit(1);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("--ordering-sweep requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if raw_arguments.get(1).map(String::as_str) == Some("--convection-split") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            if let Err(error) = cubic_live::run_convection_split(&raw_arguments[2..]) {
                eprintln!("CONVECTION_SPLIT_FATAL {error}");
                std::process::exit(1);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("--convection-split requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if raw_arguments.get(1).map(String::as_str) == Some("--convection-splu-live") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            if let Err(error) = cubic_live::run_convection_splu(&raw_arguments[2..]) {
                eprintln!("CONVECTION_SPLU_LIVE_FATAL {error}");
                std::process::exit(1);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("--convection-splu-live requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if raw_arguments.get(1).map(String::as_str) == Some("--periodic-cuboid-spsolve-live") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            if let Err(error) = cubic_live::run_periodic_cuboid_spsolve(&raw_arguments[2..]) {
                eprintln!("PERIODIC_CUBOID_SPSOLVE_LIVE_FATAL {error}");
                std::process::exit(1);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("--periodic-cuboid-spsolve-live requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if raw_arguments.get(1).map(String::as_str) == Some("--periodic-cuboid-splu-live") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            if let Err(error) = cubic_live::run_periodic_cuboid_splu(&raw_arguments[2..]) {
                eprintln!("PERIODIC_CUBOID_SPLU_LIVE_FATAL {error}");
                std::process::exit(1);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("--periodic-cuboid-splu-live requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if raw_arguments.get(1).map(String::as_str) == Some("--neumann-cubic-splu-live") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            if let Err(error) = cubic_live::run_neumann_splu(&raw_arguments[2..]) {
                eprintln!("NEUMANN_CUBIC_SPLU_LIVE_FATAL {error}");
                std::process::exit(1);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("--neumann-cubic-splu-live requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if raw_arguments.get(1).map(String::as_str) == Some("--cubic-splu-live") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            if let Err(error) = cubic_live::run_splu(&raw_arguments[2..]) {
                eprintln!("CUBIC_SPLU_LIVE_FATAL {error}");
                std::process::exit(1);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("--cubic-splu-live requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if raw_arguments.get(1).map(String::as_str) == Some("--cubic-live") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            if let Err(error) = cubic_live::run(&raw_arguments[2..]) {
                eprintln!("CUBIC_LIVE_FATAL {error}");
                std::process::exit(1);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("--cubic-live requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }

    let mut arguments = raw_arguments.into_iter().skip(1);
    let mode = arguments.next();
    if mode.as_deref() == Some("--profile-rectangular-rust") {
        let repetitions = arguments
            .next()
            .map(|value| value.parse::<usize>().expect("positive repetition count"))
            .unwrap_or(50);
        assert!(repetitions > 0, "repetition count must be positive");
        profile_rectangular_rust(repetitions);
        return;
    }
    if mode.as_deref() == Some("--profile-cubic-rust") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let repetitions = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive repetition count"))
                .unwrap_or(10);
            let side = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive cubic side"))
                .unwrap_or(16);
            assert!(repetitions > 0, "repetition count must be positive");
            assert!(side > 1, "cubic side must exceed one");
            profile_cubic_rust(repetitions, side);
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("--profile-cubic-rust requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if mode.as_deref() == Some("--profile-cuboid-rust") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let repetitions = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive repetition count"))
                .unwrap_or(4);
            let x_extent = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive cuboid x extent"))
                .unwrap_or(12);
            let y_extent = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive cuboid y extent"))
                .unwrap_or(14);
            let z_extent = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive cuboid z extent"))
                .unwrap_or(16);
            assert!(repetitions > 0, "repetition count must be positive");
            assert!(
                x_extent > 1 && y_extent > 1 && z_extent > 1,
                "cuboid extents must exceed one"
            );
            profile_cuboid_rust(repetitions, x_extent, y_extent, z_extent);
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("--profile-cuboid-rust requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if mode.as_deref() == Some("--profile-ordering-sweep") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let fixture = arguments.next().expect("sweep fixture name");
            let size = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive fixture size"))
                .unwrap_or(64);
            let repetitions = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive repetition count"))
                .unwrap_or(1);
            assert!(size > 1 && repetitions > 0);
            profile_ordering_sweep(&fixture, size, repetitions);
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("--profile-ordering-sweep requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if mode.as_deref() == Some("--profile-cubic-splu-rust") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let repetitions = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive repetition count"))
                .unwrap_or(6);
            let side = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive cubic side"))
                .unwrap_or(16);
            let rhs_count = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive RHS count"))
                .unwrap_or(32);
            assert!(repetitions > 0, "repetition count must be positive");
            assert!(side > 1, "cubic side must exceed one");
            assert!(rhs_count > 0, "RHS count must be positive");
            profile_cubic_splu_rust(repetitions, side, rhs_count);
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("--profile-cubic-splu-rust requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if mode.as_deref() == Some("--profile-convection-splu-rust") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let repetitions = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive repetition count"))
                .unwrap_or(1);
            let side = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive grid side"))
                .unwrap_or(64);
            let rhs_count = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive RHS count"))
                .unwrap_or(16);
            let output_path = arguments.next();
            assert!(repetitions > 0, "repetition count must be positive");
            assert!(side > 1, "grid side must exceed one");
            assert!(rhs_count > 0, "RHS count must be positive");
            profile_convection_splu_rust(repetitions, side, rhs_count, output_path.as_deref());
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("--profile-convection-splu-rust requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if mode.as_deref() == Some("--profile-neumann-cubic-splu-rust") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let repetitions = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive repetition count"))
                .unwrap_or(3);
            let side = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive cubic side"))
                .unwrap_or(16);
            let rhs_count = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive RHS count"))
                .unwrap_or(32);
            assert!(repetitions > 0, "repetition count must be positive");
            assert!(side > 1, "cubic side must exceed one");
            assert!(rhs_count > 0, "RHS count must be positive");
            profile_neumann_cubic_splu_rust(repetitions, side, rhs_count);
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!(
                "--profile-neumann-cubic-splu-rust requires --features sparse-incumbent-bench"
            );
            std::process::exit(2);
        }
    }
    if mode.as_deref() == Some("--profile-neumann-cuboid-splu-rust") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let repetitions = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive repetition count"))
                .unwrap_or(3);
            let x_extent = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive cuboid x extent"))
                .unwrap_or(12);
            let y_extent = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive cuboid y extent"))
                .unwrap_or(14);
            let z_extent = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive cuboid z extent"))
                .unwrap_or(16);
            let rhs_count = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive RHS count"))
                .unwrap_or(32);
            let output_path = arguments.next();
            assert!(repetitions > 0, "repetition count must be positive");
            assert!(
                x_extent > 1 && y_extent > 1 && z_extent > 1,
                "cuboid extents must exceed one"
            );
            assert!(rhs_count > 0, "RHS count must be positive");
            profile_neumann_cuboid_splu_rust(
                repetitions,
                x_extent,
                y_extent,
                z_extent,
                rhs_count,
                output_path.as_deref(),
            );
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!(
                "--profile-neumann-cuboid-splu-rust requires --features sparse-incumbent-bench"
            );
            std::process::exit(2);
        }
    }
    if mode.as_deref() == Some("--profile-periodic-cuboid-spsolve-rust") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let repetitions = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive repetition count"))
                .unwrap_or(1);
            let x_extent = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive cuboid x extent"))
                .unwrap_or(13);
            let y_extent = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive cuboid y extent"))
                .unwrap_or(15);
            let z_extent = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive cuboid z extent"))
                .unwrap_or(17);
            let output_path = arguments.next();
            assert!(repetitions > 0, "repetition count must be positive");
            assert!(
                x_extent > 2 && y_extent > 2 && z_extent > 2,
                "periodic cuboid extents must exceed two"
            );
            profile_periodic_cuboid_spsolve_rust(
                repetitions,
                x_extent,
                y_extent,
                z_extent,
                output_path.as_deref(),
            );
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!(
                "--profile-periodic-cuboid-spsolve-rust requires --features sparse-incumbent-bench"
            );
            std::process::exit(2);
        }
    }
    if mode.as_deref() == Some("--profile-periodic-cuboid-splu-rust") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let repetitions = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive repetition count"))
                .unwrap_or(3);
            let x_extent = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive cuboid x extent"))
                .unwrap_or(13);
            let y_extent = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive cuboid y extent"))
                .unwrap_or(15);
            let z_extent = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive cuboid z extent"))
                .unwrap_or(17);
            let rhs_count = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive RHS count"))
                .unwrap_or(32);
            let output_path = arguments.next();
            assert!(repetitions > 0, "repetition count must be positive");
            assert!(
                x_extent > 2 && y_extent > 2 && z_extent > 2,
                "periodic cuboid extents must exceed two"
            );
            assert!(rhs_count > 0, "RHS count must be positive");
            profile_periodic_cuboid_splu_rust(
                repetitions,
                x_extent,
                y_extent,
                z_extent,
                rhs_count,
                output_path.as_deref(),
            );
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!(
                "--profile-periodic-cuboid-splu-rust requires --features sparse-incumbent-bench"
            );
            std::process::exit(2);
        }
    }
    if mode.as_deref() == Some("--profile-triangular-wavefront-rust") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let repetitions = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive repetition count"))
                .unwrap_or(8);
            let levels = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive level count"))
                .unwrap_or(64);
            let width = arguments
                .next()
                .map(|value| value.parse::<usize>().expect("positive level width"))
                .unwrap_or(16_384);
            assert!(repetitions > 0, "repetition count must be positive");
            assert!(levels > 1, "level count must exceed one");
            assert!(width > 1, "level width must exceed one");
            profile_triangular_wavefront_rust(repetitions, levels, width);
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!(
                "--profile-triangular-wavefront-rust requires --features sparse-incumbent-bench"
            );
            std::process::exit(2);
        }
    }

    // Wider-banded routing: matrices with >16 nnz/row but a narrow band now route to the
    // sparse LU (bandwidth gate) instead of densifying to an O(n³) dense LU.
    println!("--- wider-banded routing: dense(old) vs sparse(bandwidth gate) ---");
    for &(n, hb) in &[(1024usize, 16usize), (2048, 24), (3000, 30)] {
        let a = banded(n, hb);
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();
        let x_sparse = spsolve(&a, &b, SolveOptions::default())
            .expect("spsolve")
            .solution;
        let x_dense = dense_solve_baseline(&a, &b);
        let max_dx = x_sparse
            .iter()
            .zip(&x_dense)
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);
        let reps_s = (20_000_000 / (n + 1)).clamp(10, 3000);
        let t_sparse = time(reps_s, || {
            black_box(spsolve(black_box(&a), black_box(&b), SolveOptions::default()).unwrap());
        });
        let reps_d = if n >= 2048 { 2 } else { 4 };
        let t_dense = time(reps_d, || {
            black_box(dense_solve_baseline(black_box(&a), black_box(&b)));
        });
        println!(
            "banded n={n:>5} hb={hb:>3} ({} nnz/row): dense={t_dense:>10.4}ms  sparse={t_sparse:>9.5}ms  speedup={:>9.1}x  max|dx|={max_dx:.2e}",
            2 * hb + 1,
            t_dense / t_sparse,
        );
    }

    println!("===PARITY+AB===");
    for &n in &[512usize, 1024, 2048] {
        let a = pentadiagonal(n);
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();

        // correctness: sparse-routed result vs old dense result
        let x_sparse = spsolve(&a, &b, SolveOptions::default())
            .expect("spsolve")
            .solution;
        let x_dense = dense_solve_baseline(&a, &b);
        let max_dx = x_sparse
            .iter()
            .zip(x_dense.iter())
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);

        let reps_sparse = (50_000_000 / (n + 1)).clamp(20, 5000);
        let t_after = time(reps_sparse, || {
            black_box(spsolve(black_box(&a), black_box(&b), SolveOptions::default()).unwrap());
        });
        let reps_dense = if n >= 2048 { 1 } else { 3 };
        let t_before = time(reps_dense, || {
            black_box(dense_solve_baseline(black_box(&a), black_box(&b)));
        });

        println!(
            "spsolve n={n:>5}: dense={t_before:>10.4}ms  sparse={t_after:>9.5}ms  speedup={:>9.1}x  max|dx|={max_dx:.2e}",
            t_before / t_after
        );

        // splu factorization: same routing. Time factorize-only (the dominant cost).
        let a_csc: CscMatrix = a.to_csc().unwrap();
        let fac = splu(&a_csc, LuOptions::default()).expect("splu");
        let x_splu = splu_solve(&fac, &b).expect("splu_solve");
        let max_dx2 = x_splu
            .iter()
            .zip(x_dense.iter())
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);
        let t_splu = time(reps_sparse, || {
            black_box(splu(black_box(&a_csc), LuOptions::default()).unwrap());
        });
        let t_dense_fac = time(reps_dense, || {
            let n = a.shape().rows;
            let mut dense = vec![0.0f64; n * n];
            let indptr = a.indptr();
            let indices = a.indices();
            let data = a.data();
            for i in 0..n {
                for idx in indptr[i]..indptr[i + 1] {
                    dense[i * n + indices[idx]] = data[idx];
                }
            }
            black_box(DMatrix::from_row_slice(n, n, &dense).lu());
        });
        println!(
            "splu    n={n:>5}: dense={t_dense_fac:>10.4}ms  sparse={t_splu:>9.5}ms  speedup={:>9.1}x  max|dx|={max_dx2:.2e}",
            t_dense_fac / t_splu
        );
    }

    // ── NEW LEVER: fill-reducing ordering on a SCATTERED sparse matrix ──
    // natural-order sparse LU fills toward dense; RCM (default Colamd→RCM) recovers
    // the band. Both routes solve the SAME unique system (parity to rounding).
    println!("--- fill-reducing ordering (scattered pentadiagonal) ---");
    for &n in &[300usize, 600, 1000] {
        let a = scattered_pentadiagonal(n, 0x1234 ^ n as u64);
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();

        let x_nat = spsolve(&a, &b, opts_with(PermutationOrdering::Natural))
            .expect("spsolve natural")
            .solution;
        let x_rcm = spsolve(&a, &b, opts_with(PermutationOrdering::Colamd))
            .expect("spsolve rcm")
            .solution;
        let max_dx = x_nat
            .iter()
            .zip(x_rcm.iter())
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);

        let reps = (5_000_000 / (n + 1)).clamp(5, 2000);
        let t_nat = time(reps, || {
            black_box(
                spsolve(
                    black_box(&a),
                    black_box(&b),
                    opts_with(PermutationOrdering::Natural),
                )
                .unwrap(),
            );
        });
        let t_rcm = time(reps, || {
            black_box(
                spsolve(
                    black_box(&a),
                    black_box(&b),
                    opts_with(PermutationOrdering::Colamd),
                )
                .unwrap(),
            );
        });
        println!(
            "ordering n={n:>5}: natural={t_nat:>10.4}ms  rcm={t_rcm:>9.5}ms  speedup={:>9.1}x  max|dx|={max_dx:.2e}",
            t_nat / t_rcm
        );
    }

    // ── NEW LEVER: minimum-degree (MmdAtPlusA) vs RCM on a 2D Laplacian ──
    println!("--- minimum-degree vs RCM (2D 5-point Laplacian) ---");
    for &k in &[20usize, 32, 45, 64] {
        let a = laplacian_2d(k);
        let n = k * k;
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();
        let x_rcm = spsolve(&a, &b, opts_with(PermutationOrdering::Colamd))
            .expect("rcm")
            .solution;
        let x_mmd = spsolve(&a, &b, opts_with(PermutationOrdering::MmdAtPlusA))
            .expect("mmd")
            .solution;
        let max_dx = x_rcm
            .iter()
            .zip(&x_mmd)
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);
        let reps = (8_000_000 / (n + 1)).clamp(3, 2000);
        let t_rcm = time(reps, || {
            black_box(
                spsolve(
                    black_box(&a),
                    black_box(&b),
                    opts_with(PermutationOrdering::Colamd),
                )
                .unwrap(),
            );
        });
        let t_mmd = time(reps, || {
            black_box(
                spsolve(
                    black_box(&a),
                    black_box(&b),
                    opts_with(PermutationOrdering::MmdAtPlusA),
                )
                .unwrap(),
            );
        });
        println!(
            "lap2d k={k:>3} n={n:>5}: rcm={t_rcm:>10.4}ms  mmd={t_mmd:>9.5}ms  speedup={:>7.2}x  max|dx|={max_dx:.2e}",
            t_rcm / t_mmd
        );
    }

    // ── factor-once-solve-many: min-degree's smaller factor pays off per-solve ──
    println!("--- splu factor + 200 solves: RCM vs min-degree (2D Laplacian) ---");
    for &k in &[32usize, 45, 64] {
        let a = laplacian_2d(k);
        let a_csc: CscMatrix = a.to_csc().unwrap();
        let n = k * k;
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();
        let m = 200usize;

        let lu_rcm = splu(
            &a_csc,
            LuOptions {
                ordering: PermutationOrdering::Colamd,
                ..LuOptions::default()
            },
        )
        .expect("rcm");
        let lu_mmd = splu(
            &a_csc,
            LuOptions {
                ordering: PermutationOrdering::MmdAtPlusA,
                ..LuOptions::default()
            },
        )
        .expect("mmd");
        let xr = splu_solve(&lu_rcm, &b).unwrap();
        let xm = splu_solve(&lu_mmd, &b).unwrap();
        let max_dx = xr
            .iter()
            .zip(&xm)
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);

        let reps = 30;
        let t_rcm = time(reps, || {
            for _ in 0..m {
                black_box(splu_solve(black_box(&lu_rcm), black_box(&b)).unwrap());
            }
        });
        let t_mmd = time(reps, || {
            for _ in 0..m {
                black_box(splu_solve(black_box(&lu_mmd), black_box(&b)).unwrap());
            }
        });
        println!(
            "solve×{m} k={k:>3} n={n:>5}: rcm={t_rcm:>9.4}ms  mmd={t_mmd:>9.4}ms  per-solve speedup={:>6.2}x  max|dx|={max_dx:.2e}",
            t_rcm / t_mmd
        );
    }

    println!("--- minimum-degree ordering (arrowhead) ---");
    for &n in &[300usize, 600, 1000] {
        let a = arrowhead(n);
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();

        let x_rcm = spsolve(&a, &b, opts_with(PermutationOrdering::Colamd))
            .expect("spsolve rcm")
            .solution;
        let x_mmd = spsolve(&a, &b, opts_with(PermutationOrdering::MmdAtPlusA))
            .expect("spsolve mmd")
            .solution;
        let max_dx = x_rcm
            .iter()
            .zip(x_mmd.iter())
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);

        let reps = (5_000_000 / (n + 1)).clamp(5, 2000);
        let t_rcm = time(reps, || {
            black_box(
                spsolve(
                    black_box(&a),
                    black_box(&b),
                    opts_with(PermutationOrdering::Colamd),
                )
                .unwrap(),
            );
        });
        let t_mmd = time(reps, || {
            black_box(
                spsolve(
                    black_box(&a),
                    black_box(&b),
                    opts_with(PermutationOrdering::MmdAtPlusA),
                )
                .unwrap(),
            );
        });
        println!(
            "arrowhd n={n:>5}: rcm={t_rcm:>10.4}ms  mmd={t_mmd:>9.5}ms  speedup={:>9.1}x  max|dx|={max_dx:.2e}",
            t_rcm / t_mmd
        );
    }
}
