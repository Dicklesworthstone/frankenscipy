//! Same-process A/B bench + isomorphism harness for distance_transform_edt.
//!
//! `brute_edt` reproduces the original O(foreground · background) scan; the real
//! library `distance_transform_edt` now uses the separable Felzenszwalb–
//! Huttenlocher transform. We prove byte-identical f64 output (`.to_bits()`)
//! across randomized binary images and time both.
//! Run: `cargo run --release -p fsci-ndimage --bin perf_edt`.

use fsci_ndimage::{NdArray, distance_transform_edt};
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn bit(&mut self, fg_prob_pct: u64) -> f64 {
        // Foreground (nonzero) with probability fg_prob_pct%, else background (0).
        if (self.next_u64() >> 11) % 100 < fg_prob_pct {
            1.0
        } else {
            0.0
        }
    }
}

fn make_image(shape: &[usize], fg_pct: u64, seed: u64) -> NdArray {
    let total: usize = shape.iter().product();
    let mut r = Lcg(seed);
    let data: Vec<f64> = (0..total).map(|_| r.bit(fg_pct)).collect();
    NdArray::new(data, shape.to_vec()).unwrap()
}

/// Original brute-force EDT (distances only), verbatim arithmetic from the
/// pre-optimization source: for each foreground pixel, min over all background
/// pixels of sqrt(Σ_axis ((Δ·sampling)²)).
fn brute_edt(input: &NdArray, sampling: &[f64]) -> Vec<f64> {
    let shape = &input.shape;
    let ndim = shape.len();
    let strides = &input.strides;
    let unravel = |mut flat: usize| -> Vec<usize> {
        let mut idx = vec![0usize; ndim];
        for d in 0..ndim {
            idx[d] = flat / strides[d];
            flat %= strides[d];
        }
        idx
    };
    let backgrounds: Vec<Vec<usize>> = input
        .data
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v == 0.0)
        .map(|(flat, _)| unravel(flat))
        .collect();

    let mut out = vec![0.0f64; input.data.len()];
    for (flat, &value) in input.data.iter().enumerate() {
        if value == 0.0 {
            out[flat] = 0.0;
            continue;
        }
        let coords = unravel(flat);
        let mut min_d = f64::INFINITY;
        for bg in &backgrounds {
            let dist = coords
                .iter()
                .zip(bg)
                .zip(sampling)
                .map(|((&c, &b), &s)| {
                    let delta = (c as f64 - b as f64) * s;
                    delta * delta
                })
                .sum::<f64>()
                .sqrt();
            min_d = min_d.min(dist);
        }
        out[flat] = min_d;
    }
    out
}

fn main() {
    let shapes: &[&[usize]] = &[
        &[16],
        &[64],
        &[8, 8],
        &[16, 24],
        &[5, 7, 9],
        &[1, 13],
        &[13, 1],
        &[4, 4, 4],
    ];
    let samplings: &[Option<&[f64]>] = &[
        None,
        Some(&[2.0, 3.0]),
        Some(&[2.0]),
        Some(&[2.0, 1.0, 0.5]),
    ];

    let mut mismatches = 0usize;
    let mut compared = 0usize;
    let mut payload = String::new();
    let mut seed = 0x1234_5678_9abc_def0u64;
    for shape in shapes {
        for fg in [30u64, 50, 70, 90] {
            seed = seed
                .wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493);
            let img = make_image(shape, fg, seed);
            // Pick a sampling whose length matches ndim (or None / scalar).
            for &samp in samplings {
                let use_samp = match samp {
                    None => true,
                    Some(s) => s.len() == 1 || s.len() == shape.len(),
                };
                if !use_samp {
                    continue;
                }
                let samp_vec: Vec<f64> = match samp {
                    None => vec![1.0; shape.len()],
                    Some(s) if s.len() == 1 => vec![s[0]; shape.len()],
                    Some(s) => s.to_vec(),
                };
                let lib = distance_transform_edt(&img, samp).unwrap();
                let brute = brute_edt(&img, &samp_vec);
                for (a, b) in lib.data.iter().zip(brute.iter()) {
                    compared += 1;
                    if a.to_bits() != b.to_bits() {
                        mismatches += 1;
                    }
                }
                // Compact digest of the library output for golden hashing.
                let digest: u64 = lib
                    .data
                    .iter()
                    .enumerate()
                    .fold(1469598103934665603u64, |h, (i, &x)| {
                        (h ^ x.to_bits() ^ (i as u64)).wrapping_mul(1099511628211)
                    });
                payload.push_str(&format!(
                    "shape={shape:?} fg={fg} samp={samp:?} digest={digest:016x}\n"
                ));
            }
        }
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!("isomorphism: {mismatches} mismatches / {compared} cells (0 == byte-identical)");

    // ---- Timing: square 2D images where brute force is O(N^2) ----
    for &side in &[64usize, 128, 192] {
        let img = make_image(&[side, side], 50, 0xBEEF + side as u64);
        let samp_vec = vec![1.0, 1.0];

        let t0 = Instant::now();
        let mut acc = 0.0f64;
        for _ in 0..3 {
            acc += brute_edt(&img, &samp_vec)[0];
        }
        let brute_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            acc += distance_transform_edt(&img, None).unwrap().data[0];
        }
        let fh_t = t1.elapsed();

        let ratio = brute_t.as_secs_f64() / fh_t.as_secs_f64();
        println!(
            "N={:>6} ({side}x{side})  brute={:>10.3?}  fh={:>10.3?}  ratio={ratio:>7.1}x  (acc={acc:.1})",
            side * side,
            brute_t / 3,
            fh_t / 3
        );
    }
}
