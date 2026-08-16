//! Same-process A/B for uniform_filter1d: the new O(n) running-sum (library) vs the old
//! O(n·size) per-window re-summation (replicated here, parallel across rows like the old
//! library path). Last axis, Nearest mode. Parity printed as max|Δ| (running sum is
//! tolerance-parity vs per-window). Run: `cargo run --release -p fsci-ndimage --bin perf_uniform`.

use std::hint::black_box;
use std::time::Instant;

use fsci_ndimage::{BoundaryMode, NdArray, uniform_filter1d, uniform_filter1d_perwindow_ref};

// Old algorithm: O(n·size) fresh window sum per output, parallel across rows (Nearest mode).
fn par_perwindow(data: &[f64], rows: usize, cols: usize, size: usize) -> Vec<f64> {
    let offset = (size / 2) as i64;
    let inv = 1.0 / size as f64;
    let mut out = vec![0.0; rows * cols];
    let nthreads = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(1)
        .min(rows.max(1));
    let chunk = rows.div_ceil(nthreads);
    std::thread::scope(|s| {
        for (ti, out_chunk) in out.chunks_mut(chunk * cols).enumerate() {
            let r0 = ti * chunk;
            s.spawn(move || {
                for (lr, row_out) in out_chunk.chunks_mut(cols).enumerate() {
                    let r = r0 + lr;
                    for (c, out_pixel) in row_out.iter_mut().enumerate().take(cols) {
                        let mut sum = 0.0;
                        for k in 0..size as i64 {
                            let j = (c as i64 + k - offset).clamp(0, cols as i64 - 1);
                            sum += data[r * cols + j as usize];
                        }
                        *out_pixel = sum * inv;
                    }
                }
            });
        }
    });
    out
}

fn time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / reps as f64
}

fn main() {
    let (rows, cols) = (2000usize, 2000usize);
    let data: Vec<f64> = (0..rows * cols)
        .map(|i| ((i * 2654435761usize) as f64 / u32::MAX as f64).fract())
        .collect();
    let arr = NdArray::new(data.clone(), vec![rows, cols]).unwrap();

    let _ = par_perwindow(&data, 1, 1, 1); // keep helper referenced
    for axis in [1usize, 0usize] {
        println!(
            "uniform_filter1d {rows}x{cols}, axis={axis}, Nearest (vs library old per-window path):"
        );
        for &size in &[3usize, 5, 9, 15, 31, 51] {
            let new = uniform_filter1d(&arr, size, axis, BoundaryMode::Nearest, 0.0).unwrap();
            let old =
                uniform_filter1d_perwindow_ref(&arr, size, axis, BoundaryMode::Nearest, 0.0, 0)
                    .unwrap();
            let max_dx = new
                .data
                .iter()
                .zip(&old.data)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let reps = 60usize;
            let t_new = time(reps, || {
                black_box(
                    uniform_filter1d(black_box(&arr), size, axis, BoundaryMode::Nearest, 0.0)
                        .unwrap(),
                );
            });
            let t_old = time(reps, || {
                black_box(
                    uniform_filter1d_perwindow_ref(
                        black_box(&arr),
                        size,
                        axis,
                        BoundaryMode::Nearest,
                        0.0,
                        0,
                    )
                    .unwrap(),
                );
            });
            println!(
                "  size={size:>3}: old={t_old:>8.4}ms  new(running-sum)={t_new:>8.4}ms  speedup={:>6.2}x  max|dx|={max_dx:.2e}",
                t_old / t_new
            );
        }
    }
}
