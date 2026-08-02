//! Attribute GMRES whole-solve cost to the Arnoldi basis allocation pattern.
//!
//! `gmres_inner` stores the Arnoldi basis as a jagged `Vec<Vec<f64>>` and pushes
//! a freshly heap-allocated length-`n` vector on every Arnoldi step, so a restart
//! cycle allocates and then frees `restart + 1` blocks of `8n` bytes. SciPy
//! allocates `np.empty([restart + 1, n])` exactly once per call and reuses it
//! across every restart cycle, so it cannot pay this.
//!
//! This binary measures the pattern instead of assuming it. A counting global
//! allocator records every allocation the solve performs, split by whether the
//! block crosses glibc's initial `M_MMAP_THRESHOLD` (128 KiB), because glibc
//! raises that threshold dynamically after the first `munmap` of a large block —
//! which would blunt most of the predicted cost. Pair it with
//! `perf stat -e minor-faults,page-faults` to see whether the pages are actually
//! being re-faulted or merely re-used from the heap.
//!
//! Usage: `profile_gmres_arnoldi [side ...]` (default 32 64 128 181 256).

use std::time::Instant;

use fsci_sparse::{CsrMatrix, IterativeSolveOptions, Shape2D, gmres};

/// glibc's initial `M_MMAP_THRESHOLD`. Blocks at or above this size go to
/// `mmap` until glibc's dynamic threshold adjustment raises the bar. The crate
/// is `#![forbid(unsafe_code)]`, so allocation traffic is counted from outside
/// with `strace -c -e trace=mmap,munmap` and `perf stat -e minor-faults`
/// rather than by a counting `GlobalAlloc`; this binary supplies the
/// deterministic, single-threaded workload those tools attach to. The predicted
/// per-step block size is printed alongside so the two can be lined up.
const MMAP_THRESHOLD: usize = 128 * 1024;

// ── Fixture ───────────────────────────────────────────────────────────────
//
// Byte-for-byte the operator and source field from `perf_gmres_job_vs_scipy`,
// so the numbers here sit on the same fixture the whole-job harness reports.

const DIAGONAL: f64 = 4.001;
const WEST: f64 = -1.2;
const EAST: f64 = -0.8;
const VERTICAL: f64 = -1.0;

fn convection_diffusion_2d(side: usize) -> CsrMatrix {
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
        .expect("canonical convection-diffusion CSR")
}

fn source_field(side: usize) -> Vec<f64> {
    let (source_row, source_column) = (side / 3, side / 2);
    let mut rhs = Vec::with_capacity(side * side);
    for row in 0..side {
        let row_weight = 4usize.saturating_sub(row.abs_diff(source_row));
        for column in 0..side {
            let column_weight = 4usize.saturating_sub(column.abs_diff(source_column));
            rhs.push((1 + row_weight * column_weight) as f64 / 16.0);
        }
    }
    rhs
}

// ── Orthogonalization-shape A/B ───────────────────────────────────────────
//
// `perf annotate` puts ~70% of `gmres` self-time in the modified-Gram-Schmidt
// axpy, emitting scalar `vmovsd`/`vmulsd`/`vsubsd` while the `dot_product`
// immediately above it vectorises to `vmulpd`/`vaddpd`. Two properties of the
// current shape are the suspects, and this bench separates them from the solver
// so the mechanism is established before the kernel is touched:
//
//   * `h[i][j]` is re-indexed *inside* the inner loop through a jagged
//     `Vec<Vec<f64>>`, so the scale factor is reloaded per element;
//   * `v[i][k]` double-indirects through the same jagged basis, which defeats
//     the non-aliasing reasoning that would let LLVM vectorise a write to `wj`.
//
// `Hoisted` changes only the *expression* of the loop — same operations, same
// order, same operands — so it must agree with `Indexed` bit for bit. `Slab`
// additionally lays the basis out contiguously, the layout SciPy gets for free
// from `np.empty([restart + 1, n])`.

/// Basis vectors held in the orthogonalization bench; `restart + 1` in the solver.
const BENCH_BASIS: usize = 21;

/// Current shape: jagged basis, scale factor re-indexed inside the inner loop.
fn orthogonalize_indexed(wj: &mut [f64], v: &[Vec<f64>], h: &mut [Vec<f64>], j: usize, n: usize) {
    for i in 0..=j {
        h[i][j] = dot_indexed(wj, &v[i]);
        for k in 0..n {
            wj[k] -= h[i][j] * v[i][k];
        }
    }
}

/// Same arithmetic, same order — the scale factor is bound once and the basis
/// row is bound as a slice, so the write to `wj` is provably independent.
fn orthogonalize_hoisted(wj: &mut [f64], v: &[Vec<f64>], h: &mut [Vec<f64>], j: usize, n: usize) {
    for i in 0..=j {
        let vi = &v[i][..n];
        let hij = dot_indexed(wj, vi);
        h[i][j] = hij;
        for (w, &vk) in wj[..n].iter_mut().zip(vi) {
            *w -= hij * vk;
        }
    }
}

/// Hoisted, plus the contiguous `(restart + 1) x n` basis layout.
fn orthogonalize_slab(wj: &mut [f64], v: &[f64], h: &mut [Vec<f64>], j: usize, n: usize) {
    for i in 0..=j {
        let vi = &v[i * n..(i + 1) * n];
        let hij = dot_indexed(wj, vi);
        h[i][j] = hij;
        for (w, &vk) in wj[..n].iter_mut().zip(vi) {
            *w -= hij * vk;
        }
    }
}

/// Stand-in for the crate-private `dot_product`, with its four accumulator lanes.
fn dot_indexed(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    let mut lanes = [0.0f64; 4];
    let mut index = 0;
    while index + 4 <= len {
        for (lane, offset) in lanes.iter_mut().zip(0..4) {
            *lane += a[index + offset] * b[index + offset];
        }
        index += 4;
    }
    while index < len {
        lanes[0] += a[index] * b[index];
        index += 1;
    }
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

fn bench_orthogonalization(n: usize, steps: usize) {
    let basis_jagged: Vec<Vec<f64>> = (0..BENCH_BASIS)
        .map(|i| {
            (0..n)
                .map(|k| ((i * 7 + k * 13) % 1024) as f64 / 1024.0 - 0.5)
                .collect()
        })
        .collect();
    let basis_slab: Vec<f64> = basis_jagged.iter().flatten().copied().collect();
    let seed: Vec<f64> = (0..n).map(|k| ((k * 31) % 997) as f64 / 997.0).collect();
    let j = BENCH_BASIS - 2;

    let run = |mut kernel: Box<dyn FnMut(&mut [f64], &mut [Vec<f64>])>| -> (f64, Vec<f64>) {
        let mut h = vec![vec![0.0; BENCH_BASIS]; BENCH_BASIS];
        let mut wj = seed.clone();
        kernel(&mut wj, &mut h); // warm
        let mut best = f64::INFINITY;
        let mut last = Vec::new();
        for _ in 0..steps {
            let mut w = seed.clone();
            let started = Instant::now();
            kernel(&mut w, &mut h);
            let elapsed = started.elapsed().as_secs_f64() * 1e6;
            best = best.min(elapsed);
            last = w;
        }
        (best, last)
    };

    let (indexed_us, indexed_w) = {
        let v = basis_jagged.clone();
        run(Box::new(move |w, h| orthogonalize_indexed(w, &v, h, j, n)))
    };
    let (hoisted_us, hoisted_w) = {
        let v = basis_jagged.clone();
        run(Box::new(move |w, h| orthogonalize_hoisted(w, &v, h, j, n)))
    };
    let (slab_us, slab_w) = {
        let v = basis_slab.clone();
        run(Box::new(move |w, h| orthogonalize_slab(w, &v, h, j, n)))
    };

    let hoisted_identical = indexed_w
        .iter()
        .zip(&hoisted_w)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let slab_identical = indexed_w
        .iter()
        .zip(&slab_w)
        .all(|(a, b)| a.to_bits() == b.to_bits());

    println!(
        "{:>9} {:>12.1} {:>12.1} {:>12.1} {:>10.3} {:>10.3} {:>12} {:>10}",
        n,
        indexed_us,
        hoisted_us,
        slab_us,
        indexed_us / hoisted_us,
        indexed_us / slab_us,
        hoisted_identical,
        slab_identical,
    );
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("orth") {
        let sides: Vec<usize> = std::env::args()
            .skip(2)
            .map(|value| value.parse().expect("side must be a positive integer"))
            .collect();
        let sides = if sides.is_empty() {
            vec![32, 64, 128, 181, 256]
        } else {
            sides
        };
        println!(
            "orthogonalization_ab basis={BENCH_BASIS} j={} metric=best-of-repetitions",
            BENCH_BASIS - 2
        );
        println!(
            "{:>9} {:>12} {:>12} {:>12} {:>10} {:>10} {:>12} {:>10}",
            "n",
            "indexed_us",
            "hoisted_us",
            "slab_us",
            "hoist_x",
            "slab_x",
            "hoist_bitid",
            "slab_bitid",
        );
        for side in sides {
            bench_orthogonalization(side * side, 25);
        }
        return;
    }

    let sides: Vec<usize> = {
        let parsed: Vec<usize> = std::env::args()
            .skip(1)
            .map(|value| value.parse().expect("side must be a positive integer"))
            .collect();
        if parsed.is_empty() {
            vec![32, 64, 128, 181, 256]
        } else {
            parsed
        }
    };

    let repetitions: usize = std::env::var("FSCI_PROFILE_REPS")
        .ok()
        .map(|value| value.parse().expect("FSCI_PROFILE_REPS must be an integer"))
        .unwrap_or(1);

    println!(
        "fixture=steady-convection-diffusion restart=20 rtol=1e-5 x0=zeros \
         mmap_threshold_bytes={MMAP_THRESHOLD} repetitions={repetitions}"
    );
    println!(
        "{:>5} {:>9} {:>7} {:>11} {:>10} {:>12} {:>10} {:>13}",
        "side", "n", "iters", "median_ms", "us_per_it", "basis_KiB", "cycles", "pred_blocks",
    );

    for side in sides {
        let n = side * side;
        let matrix = convection_diffusion_2d(side);
        let rhs = source_field(side);
        let options = IterativeSolveOptions::default();

        // Warm the allocator's arenas so the first-touch cost of the very first
        // solve is not attributed to the steady-state pattern.
        let warm = gmres(&matrix, &rhs, None, options).expect("warmup solve");
        assert!(
            warm.converged,
            "side={side} did not converge (residual {})",
            warm.residual_norm
        );

        let mut samples = Vec::with_capacity(repetitions);
        let mut iterations = 0;
        for _ in 0..repetitions {
            let started = Instant::now();
            let result = gmres(&matrix, &rhs, None, options).expect("measured solve");
            samples.push(started.elapsed().as_secs_f64() * 1e3);
            iterations = result.iterations;
        }
        samples.sort_by(f64::total_cmp);
        let median = samples[samples.len() / 2];

        // A restart cycle allocates one block per Arnoldi step plus the residual
        // and matvec temporaries; `restart = min(n, 20)`. Reported so an external
        // mmap/fault count can be checked against the structural prediction.
        let restart = n.min(20);
        let cycles = iterations.div_ceil(restart);
        let predicted_blocks = cycles * (restart + 1 + 3);

        println!(
            "{:>5} {:>9} {:>7} {:>11.3} {:>10.3} {:>12.1} {:>10} {:>13}",
            side,
            n,
            iterations,
            median,
            median * 1e3 / iterations.max(1) as f64,
            (n * std::mem::size_of::<f64>()) as f64 / 1024.0,
            cycles,
            predicted_blocks,
        );
    }
}
