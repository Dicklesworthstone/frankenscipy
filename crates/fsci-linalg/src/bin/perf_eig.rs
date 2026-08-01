use fsci_linalg::*;
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

const EIGH_AB_N: usize = 512;

fn make_eigh_ab_matrix() -> Vec<Vec<f64>> {
    let mut seed = 0x2468_ace0_1357_9bdfu64;
    let mut random = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    let mut matrix = vec![vec![0.0; EIGH_AB_N]; EIGH_AB_N];
    for row in 0..EIGH_AB_N {
        for col in row..EIGH_AB_N {
            let value = random();
            matrix[row][col] = value;
            matrix[col][row] = value;
        }
    }
    matrix
}

fn eigh_residual_max(matrix: &[Vec<f64>], result: &EighResult) -> f64 {
    let mut maximum = 0.0_f64;
    for col in 0..EIGH_AB_N {
        for row in 0..EIGH_AB_N {
            let mut av = 0.0;
            for (entry, eigenvector_row) in matrix[row].iter().zip(&result.eigenvectors) {
                av += entry * eigenvector_row[col];
            }
            maximum =
                maximum.max((av - result.eigenvalues[col] * result.eigenvectors[row][col]).abs());
        }
    }
    maximum
}

fn eigh_orthogonality_max(result: &EighResult) -> f64 {
    let mut maximum = 0.0_f64;
    for left in 0..EIGH_AB_N {
        for right in left..EIGH_AB_N {
            let mut dot = 0.0;
            for row in &result.eigenvectors {
                dot += row[left] * row[right];
            }
            let expected = if left == right { 1.0 } else { 0.0 };
            maximum = maximum.max((dot - expected).abs());
        }
    }
    maximum
}

fn run_eigh_ab(arguments: &[String]) {
    let arm = arguments.get(2).map_or("candidate", String::as_str);
    let force_double_read = match arm {
        "candidate" => false,
        "control" => true,
        _ => {
            eprintln!("eigh-ab arm must be candidate or control");
            return;
        }
    };
    let repetitions = arguments
        .get(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    let matrix = make_eigh_ab_matrix();
    EIGH_DSYMV_FORCE_DOUBLE_READ.store(force_double_read, Ordering::Relaxed);
    black_box(eigh(&matrix, DecompOptions::default()).expect("eigh warmup"));
    let started_at = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..repetitions {
        let result = eigh(black_box(&matrix), DecompOptions::default()).expect("timed eigh");
        checksum += result.eigenvalues[0] + result.eigenvalues[EIGH_AB_N - 1];
        black_box(result);
    }
    let elapsed = started_at.elapsed().as_secs_f64();
    EIGH_DSYMV_FORCE_DOUBLE_READ.store(false, Ordering::Relaxed);
    println!(
        "EIGH_AB arm={arm} n={EIGH_AB_N} reps={repetitions} elapsed_s={elapsed:.17e} per_call_s={:.17e} checksum={checksum:.17e}",
        elapsed / repetitions as f64
    );
}

fn verify_eigh_ab() {
    let matrix = make_eigh_ab_matrix();
    EIGH_DSYMV_FORCE_DOUBLE_READ.store(false, Ordering::Relaxed);
    let candidate = eigh(&matrix, DecompOptions::default()).expect("candidate eigh");
    EIGH_DSYMV_FORCE_DOUBLE_READ.store(true, Ordering::Relaxed);
    let control = eigh(&matrix, DecompOptions::default()).expect("control eigh");
    EIGH_DSYMV_FORCE_DOUBLE_READ.store(false, Ordering::Relaxed);

    let mut eigenvalue_bit_mismatches = 0usize;
    let mut eigenvector_bit_mismatches = 0usize;
    let mut eigenvalue_max_abs = 0.0_f64;
    for (candidate_value, control_value) in candidate.eigenvalues.iter().zip(&control.eigenvalues) {
        eigenvalue_bit_mismatches +=
            usize::from(candidate_value.to_bits() != control_value.to_bits());
        eigenvalue_max_abs = eigenvalue_max_abs.max((candidate_value - control_value).abs());
    }
    for (candidate_row, control_row) in candidate.eigenvectors.iter().zip(&control.eigenvectors) {
        for (candidate_value, control_value) in candidate_row.iter().zip(control_row) {
            eigenvector_bit_mismatches +=
                usize::from(candidate_value.to_bits() != control_value.to_bits());
        }
    }

    let residual_max = eigh_residual_max(&matrix, &candidate);
    let orthogonality_max = eigh_orthogonality_max(&candidate);
    println!(
        "EIGH_AB_VERIFY n={EIGH_AB_N} eigenvalue_bit_mismatches={eigenvalue_bit_mismatches} eigenvector_bit_mismatches={eigenvector_bit_mismatches} eigenvalue_max_abs={eigenvalue_max_abs:.17e} residual_max={residual_max:.17e} orthogonality_max={orthogonality_max:.17e}"
    );
    print!("EIGH_AB_EIGENVALUES");
    for value in &candidate.eigenvalues {
        print!(" {value:.17e}");
    }
    println!();
}

fn main() {
    let arguments = std::env::args().collect::<Vec<_>>();
    match arguments.get(1).map(String::as_str) {
        Some("eigh-ab") => {
            run_eigh_ab(&arguments);
            return;
        }
        Some("eigh-ab-verify") => {
            verify_eigh_ab();
            return;
        }
        _ => {}
    }

    let mut seed = 42u64;
    let mut r = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    for &n in &[100usize, 150, 200, 300, 400] {
        let a: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| r()).collect()).collect();
        let _ = eig(&a, DecompOptions::default());
        let t = Instant::now();
        let reps = 2;
        for _ in 0..reps {
            let _ = eig(&a, DecompOptions::default());
        }
        let ms = t.elapsed().as_secs_f64() / reps as f64 * 1000.0;
        println!("eig n={n}: {ms:.1} ms");
    }
}
