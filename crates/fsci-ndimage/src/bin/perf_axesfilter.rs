//! Same-process A/B + byte-identity for the axes-subset rank_filter path: the new interior
//! flat-gather (library) vs an inline old-style per-pixel alloc-gather reference (parallel,
//! matching the old code). 3-D volume, filtering the last 2 axes. Run:
//! `cargo run --release -p fsci-ndimage --bin perf_axesfilter`.

use std::hint::black_box;
use std::time::Instant;

use fsci_ndimage::{BoundaryMode, NdArray, rank_filter_axes};

fn time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / reps as f64
}

fn digest(v: &[f64]) -> u64 {
    v.iter().fold(1469598103934665603u64, |h, &x| {
        (h ^ x.to_bits()).wrapping_mul(1099511628211)
    })
}

// Old-style axes-subset gather (alloc k_idx + in_idx per element per pixel), parallel across
// rows, Nearest, median over the last two axes of a 3-D [d0,d1,d2] volume.
/// The `vec!` allocations in this function are the POINT of it, not an oversight:
/// it reproduces the OLD path's per-pixel and per-tap allocation cost so the new
/// path can be timed against it. Taking clippy's advice and using arrays would
/// delete the very work being measured (frankenscipy-iit9c).
#[allow(clippy::useless_vec)]
fn old_axes_median(data: &[f64], shape: [usize; 3], size: usize) -> Vec<f64> {
    let [d0, d1, d2] = shape;
    let off = (size / 2) as i64;
    let ktot = size * size;
    let mut out = vec![0.0f64; d0 * d1 * d2];
    let clamp = |c: i64, n: usize| c.clamp(0, n as i64 - 1) as usize;
    std::thread::scope(|s| {
        let nth = std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(1)
            .min(d0.max(1));
        let chunk = d0.div_ceil(nth);
        for (t, oc) in out.chunks_mut(chunk * d1 * d2).enumerate() {
            let r0 = t * chunk;
            s.spawn(move || {
                for (lo0, plane) in oc.chunks_mut(d1 * d2).enumerate() {
                    let i0 = r0 + lo0;
                    for i1 in 0..d1 {
                        for i2 in 0..d2 {
                            // Faithful old-library shape: a fresh out_idx Vec per pixel and a
                            // k_idx Vec + in_idx Vec allocated per footprint element.
                            let out_idx = vec![i0, i1, i2];
                            let mut nb = Vec::with_capacity(ktot);
                            for k in 0..ktot {
                                let k_idx = vec![k / size, k % size];
                                let mut in_idx: Vec<i64> =
                                    out_idx.iter().map(|&c| c as i64).collect();
                                in_idx[1] += k_idx[0] as i64 - off;
                                in_idx[2] += k_idx[1] as i64 - off;
                                let j0 = clamp(in_idx[0], d0);
                                let j1 = clamp(in_idx[1], d1);
                                let j2 = clamp(in_idx[2], d2);
                                nb.push(data[(j0 * d1 + j1) * d2 + j2]);
                            }
                            let (_, m, _) =
                                nb.select_nth_unstable_by(ktot / 2, |a, b| a.total_cmp(b));
                            plane[i1 * d2 + i2] = *m;
                        }
                    }
                }
            });
        }
    });
    out
}

fn main() {
    let shape = [40usize, 200, 200];
    let total = shape[0] * shape[1] * shape[2];
    let data: Vec<f64> = (0..total)
        .map(|i| ((i * 2654435761usize) as f64 / u32::MAX as f64).fract() - 0.5)
        .collect();
    let arr = NdArray::new(data.clone(), shape.to_vec()).unwrap();

    for &size in &[3usize, 5, 7] {
        let rank = (size * size / 2) as isize;
        let new = rank_filter_axes(&arr, rank, size, &[1, 2], BoundaryMode::Nearest, 0.0).unwrap();
        let old = old_axes_median(&data, shape, size);
        let bit = digest(&new.data) == digest(&old);
        let reps = 8usize;
        let t_new = time(reps, || {
            black_box(
                rank_filter_axes(
                    black_box(&arr),
                    rank,
                    size,
                    &[1, 2],
                    BoundaryMode::Nearest,
                    0.0,
                )
                .unwrap(),
            );
        });
        let t_old = time(reps, || {
            black_box(old_axes_median(black_box(&data), shape, size));
        });
        println!(
            "axes-median {shape:?} axes=[1,2] size={size} (fp {}): old={t_old:>9.4}ms  new={t_new:>8.4}ms  speedup={:>6.2}x  bit_identical={bit}",
            size * size,
            t_old / t_new
        );
    }
}
