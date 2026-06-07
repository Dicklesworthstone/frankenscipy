//! Profiling-only harness for FFT hot paths.
//!
//! NOT a product binary. It exists so RCH, hyperfine, and sha256 checks can
//! attach to a tight deterministic FFT scenario. Build under `release-perf`:
//!
//! ```bash
//! RUSTFLAGS="-C force-frame-pointers=yes" \
//!   cargo build -p fsci-fft --profile release-perf --bin perf_fft
//! ```
//!
//! Usage: `perf_fft <mode> <n> <repeats>`
//!   mode    = polymul | rfft | irfft | golden | rfft-golden | irfft-golden |
//!             fft2-golden | nd-golden | fft2t | fftnt
//!   n       = input length for timed modes
//!   repeats = timed iterations

use std::fmt::Write as _;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use fsci_fft::{FftOptions, fft, fft2, fftn, ifft, ifftn, irfft, polynomial_multiply_fft, rfft};

fn make_polynomial_input(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (3.0 * t).sin() - 0.5 * (5.0 * t).cos()
        })
        .collect()
}

fn make_real_input(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * std::f64::consts::PI * t).sin()
        })
        .collect()
}

fn make_complex_input(n: usize) -> Vec<(f64, f64)> {
    make_real_input(n)
        .into_iter()
        .map(|value| (value, 0.0))
        .collect()
}

fn polymul_golden_text() -> String {
    let opts = FftOptions::default();
    let mut output = String::new();
    for &n in &[16usize, 64, 256, 1024] {
        let lhs = make_polynomial_input(n);
        let rhs = make_polynomial_input(n);
        let product = polynomial_multiply_fft(&lhs, &rhs, &opts).expect("polynomial_multiply_fft");
        write!(&mut output, "mode=polymul n={n} len={} ", product.len())
            .expect("write polymul header");
        for value in &product {
            write!(&mut output, "{:016x} ", value.to_bits()).expect("write polymul bits");
        }
        output.push('\n');
    }
    output
}

fn rfft_golden_text() -> String {
    let opts = FftOptions::default();
    let mut output = String::new();
    for &n in &[16usize, 64, 256, 1024] {
        let input = make_real_input(n);
        let spectrum = rfft(&input, &opts).expect("rfft");
        write!(&mut output, "mode=rfft n={n} len={} ", spectrum.len()).expect("write rfft header");
        for &(real, imag) in &spectrum {
            write!(
                &mut output,
                "{:016x}:{:016x} ",
                real.to_bits(),
                imag.to_bits(),
            )
            .expect("write rfft bits");
        }
        output.push('\n');
    }
    output
}

fn irfft_golden_text() -> String {
    let opts = FftOptions::default();
    let mut output = String::new();
    for &n in &[16usize, 64, 256, 1024] {
        let input = make_real_input(n);
        let spectrum = rfft(&input, &opts).expect("rfft");
        let recovered = irfft(&spectrum, Some(n), &opts).expect("irfft");
        write!(
            &mut output,
            "mode=irfft n={n} spectrum_len={} len={} ",
            spectrum.len(),
            recovered.len()
        )
        .expect("write irfft header");
        for value in &recovered {
            write!(&mut output, "{:016x} ", value.to_bits()).expect("write irfft bits");
        }
        output.push('\n');
    }
    output
}

fn fft2_golden_text() -> String {
    let opts = FftOptions::default();
    let mut output = String::new();
    for &(rows, cols) in &[(8usize, 8usize), (16, 16), (32, 32)] {
        let input = make_complex_input(rows * cols);
        let spectrum = fft2(&input, (rows, cols), &opts).expect("fft2");
        write!(
            &mut output,
            "mode=fft2 shape={rows}x{cols} len={} ",
            spectrum.len()
        )
        .expect("write fft2 header");
        for &(real, imag) in &spectrum {
            write!(
                &mut output,
                "{:016x}:{:016x} ",
                real.to_bits(),
                imag.to_bits(),
            )
            .expect("write fft2 bits");
        }
        output.push('\n');
    }
    output
}

fn write_or_print_golden(output: String, path: Option<&str>) {
    if let Some(path) = path {
        let path = Path::new(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create golden artifact parent");
        }
        std::fs::write(path, output).expect("write golden artifact");
    } else {
        print!("{output}");
    }
}

fn nd_golden_text() -> String {
    let opts = FftOptions::default();
    let mut output = String::new();
    let shapes: &[&[usize]] = &[
        &[8, 8],
        &[6, 10],
        &[16, 16],
        &[4, 4, 4],
        &[3, 5, 7],
        &[8, 4, 2],
        // Larger ND regression fixtures covering 2D and 3D axis layouts.
        &[128, 128],
        &[32, 32, 32],
        &[64, 4, 64],
    ];
    for shape in shapes {
        let len: usize = shape.iter().product();
        let input = make_complex_input(len);
        let fwd = fftn(&input, shape, &opts).expect("fftn");
        let inv = ifftn(&fwd, shape, &opts).expect("ifftn");
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        write!(&mut output, "mode=fftn shape={} ", dims.join("x")).expect("write fftn header");
        for &(real, imag) in &fwd {
            write!(
                &mut output,
                "{:016x}:{:016x} ",
                real.to_bits(),
                imag.to_bits()
            )
            .expect("write fftn bits");
        }
        output.push('\n');
        write!(&mut output, "mode=ifftn shape={} ", dims.join("x")).expect("write ifftn header");
        for &(real, imag) in &inv {
            write!(
                &mut output,
                "{:016x}:{:016x} ",
                real.to_bits(),
                imag.to_bits()
            )
            .expect("write ifftn bits");
        }
        output.push('\n');
    }
    output
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("polymul");
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let repeats: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);

    if mode == "golden" {
        write_or_print_golden(polymul_golden_text(), args.get(2).map(String::as_str));
        return;
    }
    if mode == "rfft-golden" {
        write_or_print_golden(rfft_golden_text(), args.get(2).map(String::as_str));
        return;
    }
    if mode == "irfft-golden" {
        write_or_print_golden(irfft_golden_text(), args.get(2).map(String::as_str));
        return;
    }
    if mode == "fft2-golden" {
        write_or_print_golden(fft2_golden_text(), args.get(2).map(String::as_str));
        return;
    }
    if mode == "nd-golden" {
        write_or_print_golden(nd_golden_text(), args.get(2).map(String::as_str));
        return;
    }

    if mode == "fft2t" || mode == "fftnt" {
        // Timed nd transforms. `n` is the per-axis length. fft2t -> n x n,
        // fftnt -> n x n x n. Same-process median-free total over `repeats`.
        let opts = FftOptions::default();
        let shape: Vec<usize> = if mode == "fft2t" {
            vec![n, n]
        } else {
            vec![n, n, n]
        };
        let len: usize = shape.iter().product();
        let input = make_complex_input(len);
        let t0 = Instant::now();
        let mut checksum = 0.0_f64;
        for _ in 0..repeats {
            let out = fftn(black_box(&input), black_box(&shape), black_box(&opts)).expect("fftn");
            checksum += out.first().map(|&(re, _)| re).unwrap_or(0.0);
            black_box(&out);
        }
        let elapsed = t0.elapsed();
        let total_ms = elapsed.as_secs_f64() * 1e3;
        let per_call_ms = total_ms / repeats as f64;
        println!(
            "{{\"mode\":\"{mode}\",\"n\":{n},\"len\":{len},\"repeats\":{repeats},\"total_ms\":{total_ms:.3},\"per_call_ms\":{per_call_ms:.6},\"checksum\":{checksum:.12e}}}",
        );
        return;
    }

    if mode == "fftprobe" {
        // Decompose fft() cost at large n: full fft(), full ifft(), and the
        // isolated finiteness scan that the default (check_finite=false) path
        // currently runs and discards. Median-of-reps, one process.
        let probe_n = if n >= 2 { n } else { 262_144 };
        let input = make_complex_input(probe_n);
        let median = |f: &mut dyn FnMut() -> u64| -> u64 {
            let mut s: Vec<u64> = (0..repeats).map(|_| f()).collect();
            s.sort_unstable();
            s[s.len() / 2]
        };
        let opts = FftOptions::default();
        let mut fft_call = || {
            let t = Instant::now();
            let out = fft(black_box(&input), black_box(&opts)).expect("fft");
            let e = t.elapsed().as_nanos() as u64;
            black_box(out.len());
            e
        };
        let fft_ns = median(&mut fft_call);
        let spectrum = fft(&input, &opts).expect("fft");
        let mut ifft_call = || {
            let t = Instant::now();
            let out = ifft(black_box(&spectrum), black_box(&opts)).expect("ifft");
            let e = t.elapsed().as_nanos() as u64;
            black_box(out.len());
            e
        };
        let ifft_ns = median(&mut ifft_call);
        let mut scan_call = || {
            let t = Instant::now();
            let bad = black_box(&input)
                .iter()
                .any(|&(re, im)| !re.is_finite() || !im.is_finite());
            let e = t.elapsed().as_nanos() as u64;
            black_box(bad);
            e
        };
        let scan_ns = median(&mut scan_call);
        println!(
            "mode=fftprobe n={probe_n} reps={repeats} fft_ms={:.4} ifft_ms={:.4} dead_scan_ms={:.4} scan_pct_of_fft={:.2}",
            fft_ns as f64 / 1e6,
            ifft_ns as f64 / 1e6,
            scan_ns as f64 / 1e6,
            100.0 * scan_ns as f64 / fft_ns as f64,
        );
        return;
    }

    if mode != "polymul" && mode != "rfft" && mode != "irfft" {
        eprintln!("unknown mode: {mode}");
        std::process::exit(2);
    }

    let opts = FftOptions::default();

    let t0 = Instant::now();
    let mut checksum = 0.0_f64;
    if mode == "polymul" {
        let lhs = make_polynomial_input(n);
        let rhs = make_polynomial_input(n);
        for _ in 0..repeats {
            let product =
                polynomial_multiply_fft(black_box(&lhs), black_box(&rhs), black_box(&opts))
                    .expect("polynomial_multiply_fft");
            checksum += product.iter().sum::<f64>();
            black_box(&product);
        }
    } else if mode == "rfft" {
        let input = make_real_input(n);
        for _ in 0..repeats {
            let spectrum = rfft(black_box(&input), black_box(&opts)).expect("rfft");
            checksum += spectrum
                .iter()
                .map(|&(real, imag)| real + imag)
                .sum::<f64>();
            black_box(&spectrum);
        }
    } else {
        let input = make_real_input(n);
        let spectrum = rfft(&input, &opts).expect("rfft");
        for _ in 0..repeats {
            let recovered =
                irfft(black_box(&spectrum), black_box(Some(n)), black_box(&opts)).expect("irfft");
            checksum += recovered.iter().sum::<f64>();
            black_box(&recovered);
        }
    }
    let elapsed = t0.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1e3;
    let per_call_ms = total_ms / repeats as f64;
    println!(
        "{{\"mode\":\"{mode}\",\"n\":{n},\"repeats\":{repeats},\"total_ms\":{total_ms:.3},\"per_call_ms\":{per_call_ms:.6},\"checksum\":{checksum:.12e}}}",
    );
}
