//! Profiling-only harness for stats hot paths.
//!
//! Usage:
//!   `perf_stats golden [path]`
//!   `perf_stats psd <repeats>`
//!   `perf_stats qmc-golden [path]`
//!   `perf_stats halton4-golden [path]`
//!   `perf_stats sobol2-golden [path]`
//!   `perf_stats rand-index-golden [path]`
//!   `perf_stats acf-golden [path]`

use std::fmt::Write as _;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use fsci_stats::{
    HaltonSampler, SobolSampler, acf, centered_discrepancy, l2_star_discrepancy,
    mixture_discrepancy, pacf, psd_welch, rand_index, wraparound_discrepancy,
};

fn deterministic_data(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            (x * 0.017).sin() + (x * 0.031).cos() * 0.25 + (i % 17) as f64 * 0.001
        })
        .collect()
}

fn psd_case() -> Vec<f64> {
    let data = deterministic_data(4096);
    psd_welch(&data, 128, 64, 1.0)
}

fn golden_text() -> String {
    let values = psd_case();
    let mut output = String::new();
    writeln!(output, "case=psd_welch_4096_w128_o64 len={}", values.len()).expect("write header");
    output.push_str("psd=");
    for value in &values {
        write!(output, "{:016x},", value.to_bits()).expect("write psd bits");
    }
    output.push('\n');
    output
}

fn qmc_golden_text() -> String {
    let mut sampler = HaltonSampler::new(2).expect("valid Halton dimension");
    let sample = sampler.sample(512);
    let centered = centered_discrepancy(&sample, 2).expect("centered discrepancy");
    let mixture = mixture_discrepancy(&sample, 2).expect("mixture discrepancy");
    let l2_star = l2_star_discrepancy(&sample, 2).expect("l2 star discrepancy");
    let wraparound = wraparound_discrepancy(&sample, 2).expect("wraparound discrepancy");
    let mut output = String::new();
    writeln!(output, "case=qmc_discrepancy_512x2 len={}", sample.len()).expect("write qmc header");
    writeln!(output, "centered={:016x}", centered.to_bits()).expect("write centered bits");
    writeln!(output, "mixture={:016x}", mixture.to_bits()).expect("write mixture bits");
    writeln!(output, "l2_star={:016x}", l2_star.to_bits()).expect("write l2-star bits");
    writeln!(output, "wraparound={:016x}", wraparound.to_bits()).expect("write wraparound bits");
    output
}

fn halton4_golden_text() -> String {
    let mut sampler = HaltonSampler::new(4).expect("valid Halton dimension");
    let sample = sampler.sample(4096);
    let mut output = String::new();
    writeln!(output, "case=halton_4d_4096 len={}", sample.len()).expect("write halton header");
    output.push_str("sample=");
    for value in &sample {
        write!(output, "{:016x},", value.to_bits()).expect("write halton bits");
    }
    output.push('\n');
    output
}

fn sobol2_golden_text() -> String {
    let mut sampler = SobolSampler::new(2).expect("valid Sobol dimension");
    let sample = sampler.sample(4096);
    let mut output = String::new();
    writeln!(output, "case=sobol_2d_4096 len={}", sample.len()).expect("write sobol header");
    output.push_str("sample=");
    for value in &sample {
        write!(output, "{:016x},", value.to_bits()).expect("write sobol bits");
    }
    output.push('\n');
    output
}

fn rand_index_golden_text() -> String {
    let mut output = String::new();

    let dense_true: Vec<f64> = (0..8000).map(|i| (i % 10) as f64).collect();
    let dense_pred: Vec<f64> = (0..8000).map(|i| ((i * 7 + 3) % 11) as f64).collect();
    let dense = rand_index(&dense_true, &dense_pred);
    writeln!(
        output,
        "case=rand_index_dense_k10_8000 bits={:016x}",
        dense.to_bits()
    )
    .expect("write dense rand-index bits");

    let negative_true: Vec<f64> = (0..128).map(|i| (i as i64 % 7 - 3) as f64).collect();
    let negative_pred: Vec<f64> = (0..128).map(|i| (i as i64 % 5 - 2) as f64).collect();
    let negative = rand_index(&negative_true, &negative_pred);
    writeln!(
        output,
        "case=rand_index_negative_labels_128 bits={:016x}",
        negative.to_bits()
    )
    .expect("write negative rand-index bits");

    let sparse_true: Vec<f64> = (0..512).map(|i| (i * 1_000_003) as f64).collect();
    let sparse_pred: Vec<f64> = (0..512).map(|i| ((i * 17 + 5) % 257) as f64).collect();
    let sparse = rand_index(&sparse_true, &sparse_pred);
    writeln!(
        output,
        "case=rand_index_sparse_large_labels_512 bits={:016x}",
        sparse.to_bits()
    )
    .expect("write sparse rand-index bits");

    output
}

fn write_series_bits(output: &mut String, name: &str, values: &[f64]) {
    writeln!(output, "case={name} len={}", values.len()).expect("write series header");
    for value in values {
        write!(output, "{:016x},", value.to_bits()).expect("write series bits");
    }
    output.push('\n');
}

fn acf_golden_text() -> String {
    let mut output = String::new();

    let data = deterministic_data(4096);
    write_series_bits(&mut output, "acf_deterministic_4096_lag64", &acf(&data, 64));
    write_series_bits(
        &mut output,
        "pacf_deterministic_4096_lag64",
        &pacf(&data, 64),
    );

    let constant = [3.0; 8];
    write_series_bits(&mut output, "acf_constant_8_lag5", &acf(&constant, 5));
    write_series_bits(&mut output, "pacf_constant_8_lag5", &pacf(&constant, 5));

    let nonfinite_var = [1.0e308, -1.0e308, 1.0e308, -1.0e308];
    write_series_bits(
        &mut output,
        "acf_nonfinite_variance_4_lag3",
        &acf(&nonfinite_var, 3),
    );

    output
}

fn write_or_print(output: String, path: Option<&str>) {
    if let Some(path) = path {
        let path = Path::new(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create stats perf artifact parent");
        }
        std::fs::write(path, output).expect("write stats perf artifact");
    } else {
        print!("{output}");
    }
}

fn timed_psd(repeats: usize) {
    let data = deterministic_data(4096);
    let t0 = Instant::now();
    let mut checksum = 0.0_f64;
    let mut len = 0usize;
    for _ in 0..repeats {
        let result = psd_welch(&data, 128, 64, 1.0);
        checksum += result.iter().copied().sum::<f64>();
        len += result.len();
        black_box(&result);
    }
    let elapsed = t0.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1e3;
    let per_call_us = elapsed.as_secs_f64() * 1e6 / repeats as f64;
    println!(
        "{{\"mode\":\"psd\",\"repeats\":{repeats},\"total_ms\":{total_ms:.3},\"per_call_us\":{per_call_us:.6},\"len\":{len},\"checksum\":{checksum:.12e}}}",
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("golden");

    match mode {
        "golden" => write_or_print(golden_text(), args.get(2).map(String::as_str)),
        "qmc-golden" => write_or_print(qmc_golden_text(), args.get(2).map(String::as_str)),
        "halton4-golden" => write_or_print(halton4_golden_text(), args.get(2).map(String::as_str)),
        "sobol2-golden" => write_or_print(sobol2_golden_text(), args.get(2).map(String::as_str)),
        "rand-index-golden" => {
            write_or_print(rand_index_golden_text(), args.get(2).map(String::as_str));
        }
        "acf-golden" => {
            write_or_print(acf_golden_text(), args.get(2).map(String::as_str));
        }
        "psd" => {
            let repeats = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
            timed_psd(repeats);
        }
        _ => {
            eprintln!("unknown mode: {mode}");
            std::process::exit(2);
        }
    }
}
