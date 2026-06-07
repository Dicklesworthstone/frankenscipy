//! Profiling-only harness for sparse hot paths.
//!
//! NOT a product binary. It exists so RCH, hyperfine, and sha256 checks can
//! attach to deterministic sparse arithmetic scenarios.
//!
//! Usage:
//!   `perf_sparse add-csr <n> <density> <repeats>`
//!   `perf_sparse add-csr-golden [path]`
//!   `perf_sparse spilu <n> <half_bandwidth> <repeats>`
//!   `perf_sparse spilu-golden [path]`

use std::fmt::Write as _;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use fsci_sparse::{
    CooMatrix, CscMatrix, CsrMatrix, FormatConvertible, IluOptions, Shape2D, add_csr, diags,
    random, scale_csr, spilu,
};

const SEED: u64 = 0xBEEF_CAFE;

fn make_add_inputs(n: usize, density: f64) -> (CsrMatrix, CsrMatrix) {
    let shape = Shape2D::new(n, n);
    let lhs = random(shape, density, SEED)
        .expect("random lhs")
        .to_csr()
        .expect("lhs csr");
    let rhs = random(shape, density, SEED ^ 0x5EED_1234)
        .expect("random rhs")
        .to_csr()
        .expect("rhs csr");
    (lhs, rhs)
}

fn cancellation_inputs() -> (CsrMatrix, CsrMatrix) {
    let shape = Shape2D::new(3, 4);
    let lhs = CooMatrix::from_triplets(
        shape,
        vec![1.0, 2.0, -4.0, 5.0],
        vec![0, 1, 1, 2],
        vec![1, 0, 3, 2],
        false,
    )
    .expect("lhs coo")
    .to_csr()
    .expect("lhs csr");
    let rhs = CooMatrix::from_triplets(
        shape,
        vec![3.0, 4.0, -5.0, 6.0],
        vec![0, 1, 2, 2],
        vec![2, 3, 2, 3],
        false,
    )
    .expect("rhs coo")
    .to_csr()
    .expect("rhs csr");
    (lhs, rhs)
}

fn write_csr(output: &mut String, label: &str, matrix: &CsrMatrix) {
    let meta = matrix.canonical_meta();
    write!(
        output,
        "case={label} shape={}x{} nnz={} sorted={} deduplicated={} indptr=",
        matrix.shape().rows,
        matrix.shape().cols,
        matrix.nnz(),
        meta.sorted_indices,
        meta.deduplicated,
    )
    .expect("write header");
    for value in matrix.indptr() {
        write!(output, "{value},").expect("write indptr");
    }
    output.push_str(" indices=");
    for value in matrix.indices() {
        write!(output, "{value},").expect("write indices");
    }
    output.push_str(" data=");
    for value in matrix.data() {
        write!(output, "{:016x},", value.to_bits()).expect("write data");
    }
    output.push('\n');
}

fn add_csr_golden_text() -> String {
    let mut output = String::new();
    let cases = [(8usize, 0.25), (64, 0.05), (1024, 0.001)];
    for (n, density) in cases {
        let (lhs, rhs) = make_add_inputs(n, density);
        let sum = add_csr(&lhs, &rhs).expect("add csr");
        write_csr(&mut output, &format!("random-{n}-{density}"), &sum);
    }
    let (lhs, rhs) = cancellation_inputs();
    let sum = add_csr(&lhs, &rhs).expect("add csr cancellation");
    write_csr(&mut output, "cancellation", &sum);
    output
}

fn diags_golden_text() -> String {
    let mut output = String::new();

    let small = diags(
        &[
            vec![-1.0, -1.0, -1.0, -1.0, -1.0],
            vec![2.0; 6],
            vec![-1.0, -1.0, -1.0, -1.0, -1.0],
        ],
        &[-1, 0, 1],
        Some(Shape2D::new(6, 6)),
    )
    .expect("small tridiag");
    write_csr(&mut output, "diags-tridiag-6", &small);

    let rectangular = diags(
        &[vec![0.0, 3.0, -2.0], vec![4.0, 0.0]],
        &[1, -2],
        Some(Shape2D::new(4, 5)),
    )
    .expect("rectangular explicit-zero diags");
    write_csr(&mut output, "diags-rect-explicit-zero", &rectangular);

    let n = 10_000usize;
    let sub = vec![-1.0; n - 1];
    let main = vec![2.0; n];
    let sup = vec![-1.0; n - 1];
    let large =
        diags(&[sub, main, sup], &[-1, 0, 1], Some(Shape2D::new(n, n))).expect("large tridiag");
    write_csr(&mut output, "diags-tridiag-10000", &large);

    output
}

fn coo_csr_golden_text() -> String {
    let mut output = String::new();

    let duplicate = CooMatrix::from_triplets(
        Shape2D::new(4, 5),
        vec![7.0, 1.5, 0.0, -2.0, 3.25, -7.0, 2.0],
        vec![3, 0, 2, 0, 2, 3, 0],
        vec![1, 4, 2, 1, 2, 1, 1],
        false,
    )
    .expect("duplicate coo");
    write_csr(
        &mut output,
        "coo-csr-unsorted-duplicates",
        &duplicate.to_csr().expect("duplicate csr"),
    );

    let rectangular = CooMatrix::from_triplets(
        Shape2D::new(3, 6),
        vec![0.0, -4.0, 9.0, 1.25, -1.25, 5.5],
        vec![2, 1, 0, 1, 1, 2],
        vec![5, 2, 0, 4, 4, 1],
        false,
    )
    .expect("rectangular coo");
    write_csr(
        &mut output,
        "coo-csr-rect-explicit-zero",
        &rectangular.to_csr().expect("rectangular csr"),
    );

    let seeded = random(Shape2D::new(32, 32), 0.08, SEED)
        .expect("seeded coo")
        .to_csr()
        .expect("seeded csr");
    write_csr(&mut output, "coo-csr-seeded-32", &seeded);

    output
}

fn scale_csr_golden_text() -> String {
    let mut output = String::new();

    let canonical = CooMatrix::from_triplets(
        Shape2D::new(4, 5),
        vec![1.0, -2.0, -0.0, 3.5, 0.0, 9.25],
        vec![0, 1, 1, 2, 3, 3],
        vec![4, 0, 3, 2, 1, 4],
        false,
    )
    .expect("canonical scale coo")
    .to_csr()
    .expect("canonical scale csr");
    write_csr(
        &mut output,
        "scale-csr-canonical-neg",
        &scale_csr(&canonical, -2.5).expect("scale canonical"),
    );
    write_csr(
        &mut output,
        "scale-csr-canonical-zero-alpha",
        &scale_csr(&canonical, 0.0).expect("scale canonical zero"),
    );

    let unsorted = CsrMatrix::from_components(
        Shape2D::new(3, 4),
        vec![1.0, -0.0, 4.5, -3.0, 2.0],
        vec![3, 1, 1, 0, 0],
        vec![0, 2, 4, 5],
        false,
    )
    .expect("valid unsorted csr");
    write_csr(
        &mut output,
        "scale-csr-unsorted-preserve-meta",
        &scale_csr(&unsorted, -1.5).expect("scale unsorted"),
    );

    let duplicate = CsrMatrix::from_components(
        Shape2D::new(2, 3),
        vec![2.0, -2.0, 5.0],
        vec![1, 1, 2],
        vec![0, 2, 3],
        false,
    )
    .expect("valid duplicate csr");
    write_csr(
        &mut output,
        "scale-csr-duplicates-preserve-meta",
        &scale_csr(&duplicate, 3.0).expect("scale duplicate"),
    );

    output
}

fn make_spilu_banded_csc(n: usize, half_bandwidth: usize) -> CscMatrix {
    let entries_per_row = half_bandwidth.saturating_mul(2).saturating_add(1);
    let mut data = Vec::with_capacity(n.saturating_mul(entries_per_row));
    let mut rows = Vec::with_capacity(data.capacity());
    let mut cols = Vec::with_capacity(data.capacity());

    for row in 0..n {
        let start = row.saturating_sub(half_bandwidth);
        let end = row.saturating_add(half_bandwidth).min(n.saturating_sub(1));
        for col in start..=end {
            rows.push(row);
            cols.push(col);
            if row == col {
                data.push(entries_per_row as f64 + 2.0 + (row % 17) as f64 * 0.001);
            } else {
                data.push(-1.0 / (row.abs_diff(col) + 1) as f64);
            }
        }
    }

    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .expect("spilu banded coo")
        .to_csc()
        .expect("spilu banded csc")
}

fn spilu_rhs(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| ((i % 23) as f64 - 11.0) * 0.125 + 1.0)
        .collect()
}

fn spilu_golden_text() -> String {
    let mut output = String::new();
    for &(n, half_bandwidth) in &[(16usize, 3usize), (64, 5), (160, 7)] {
        let matrix = make_spilu_banded_csc(n, half_bandwidth);
        let ilu = spilu(&matrix, IluOptions::default()).expect("spilu golden");
        let solution = ilu.solve(&spilu_rhs(n)).expect("spilu solve");
        write!(
            output,
            "case=banded-{n}-{half_bandwidth} shape={}x{} solution=",
            ilu.shape.0, ilu.shape.1
        )
        .expect("write spilu golden header");
        for value in solution {
            write!(output, "{:016x},", value.to_bits()).expect("write spilu solve bits");
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
        std::fs::write(path, output.as_bytes()).expect("write golden artifact");
    }
    print!("{output}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("add-csr");
    if mode == "add-csr-golden" {
        write_or_print_golden(add_csr_golden_text(), args.get(2).map(String::as_str));
        return;
    }
    if mode == "diags-golden" {
        write_or_print_golden(diags_golden_text(), args.get(2).map(String::as_str));
        return;
    }
    if mode == "coo-csr-golden" {
        write_or_print_golden(coo_csr_golden_text(), args.get(2).map(String::as_str));
        return;
    }
    if mode == "scale-csr-golden" {
        write_or_print_golden(scale_csr_golden_text(), args.get(2).map(String::as_str));
        return;
    }
    if mode == "spilu-golden" {
        write_or_print_golden(spilu_golden_text(), args.get(2).map(String::as_str));
        return;
    }
    if mode == "spilu" {
        let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1_024);
        let half_bandwidth: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);
        let repeats: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(10);
        let matrix = make_spilu_banded_csc(n, half_bandwidth);
        let rhs = spilu_rhs(n);

        let t0 = Instant::now();
        let mut checksum = 0.0_f64;
        for _ in 0..repeats {
            let ilu = spilu(black_box(&matrix), IluOptions::default()).expect("spilu");
            checksum += ilu.shape.0 as f64 + ilu.shape.1 as f64;
            checksum += ilu.solve(black_box(&rhs)).expect("spilu solve")[n / 2];
            black_box(&ilu);
        }
        let elapsed = t0.elapsed();
        let total_ms = elapsed.as_secs_f64() * 1e3;
        let per_call_ms = total_ms / repeats as f64;
        println!(
            "{{\"mode\":\"{mode}\",\"n\":{n},\"half_bandwidth\":{half_bandwidth},\"repeats\":{repeats},\"total_ms\":{total_ms:.3},\"per_call_ms\":{per_call_ms:.6},\"checksum\":{checksum:.12e}}}",
        );
        return;
    }
    if mode != "add-csr" {
        eprintln!("unknown mode: {mode}");
        std::process::exit(2);
    }

    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let density: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.001);
    let repeats: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20);
    let (lhs, rhs) = make_add_inputs(n, density);

    let t0 = Instant::now();
    let mut checksum = 0.0_f64;
    for _ in 0..repeats {
        let sum = add_csr(black_box(&lhs), black_box(&rhs)).expect("add csr");
        checksum += sum.data().iter().sum::<f64>() + sum.nnz() as f64;
        black_box(&sum);
    }
    let elapsed = t0.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1e3;
    let per_call_ms = total_ms / repeats as f64;
    println!(
        "{{\"mode\":\"{mode}\",\"n\":{n},\"density\":{density},\"repeats\":{repeats},\"total_ms\":{total_ms:.3},\"per_call_ms\":{per_call_ms:.6},\"checksum\":{checksum:.12e}}}",
    );
}
