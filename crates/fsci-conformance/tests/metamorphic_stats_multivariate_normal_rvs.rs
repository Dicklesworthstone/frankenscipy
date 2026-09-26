#![forbid(unsafe_code)]
//! Metamorphic invariants for fsci's
//! `multivariate_normal_rvs(mean, cov, n_samples, seed)`.
//!
//! Resolves [frankenscipy-sfg3m]. fsci uses a deterministic
//! LCG-based Box-Muller pipeline (mul=6364136223846793005,
//! seed-stepped per pair) — scipy.stats.multivariate_normal
//! uses PCG64. The two cannot agree at any seed regardless of
//! algorithm fidelity, so this harness verifies properties
//! that any sensible MV-Normal generator must satisfy:
//!
//!   1. Output shape: n_samples rows × d cols.
//!   2. Determinism: same seed → identical output.
//!   3. Sample-mean LLN: with n_samples = 50_000, ‖m̂ − μ‖∞
//!      converges to ≪ 0.05 for unit-scale covariances.
//!   4. Sample-cov LLN: with n_samples = 50_000, the entry-
//!      wise max abs error of Σ̂ vs Σ converges to ≪ 0.1 for
//!      unit-scale covariances.
//!   5. Dimension dispatch: 1-D and 3-D fixtures both produce
//!      the right column count.
//!
//! 4 fixtures × variable invariants ≈ 18 cases.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::multivariate_normal_rvs;
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const MEAN_TOL: f64 = 0.05; // LLN convergence at n=50_000 for unit-scale cov
const COV_TOL: f64 = 0.10; // LLN convergence — sample cov is noisier than sample mean

#[derive(Debug, Clone, Serialize)]
struct CaseLog {
    case_id: String,
    invariant: String,
    detail: String,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct MetamorphicLog {
    test_id: String,
    case_count: usize,
    pass_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseLog>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("fixtures/artifacts/{PACKET_ID}/metamorphic"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir())
        .expect("create multivariate_normal_rvs metamorphic output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &MetamorphicLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log)
        .expect("serialize multivariate_normal_rvs metamorphic log");
    fs::write(path, json).expect("write multivariate_normal_rvs metamorphic log");
}

fn sample_mean(samples: &[Vec<f64>], d: usize) -> Vec<f64> {
    let n = samples.len() as f64;
    let mut m = vec![0.0; d];
    for s in samples {
        for (i, x) in s.iter().enumerate() {
            m[i] += x;
        }
    }
    for v in &mut m {
        *v /= n;
    }
    m
}

fn sample_cov(samples: &[Vec<f64>], mean: &[f64], d: usize) -> Vec<Vec<f64>> {
    let n = samples.len() as f64;
    let mut cov = vec![vec![0.0; d]; d];
    for s in samples {
        for i in 0..d {
            let di = s[i] - mean[i];
            for j in 0..d {
                let dj = s[j] - mean[j];
                cov[i][j] += di * dj;
            }
        }
    }
    for row in cov.iter_mut() {
        for v in row.iter_mut() {
            *v /= n - 1.0;
        }
    }
    cov
}

fn max_abs_diff_vec(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, nan_max)
}

fn max_abs_diff_mat(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut m = 0.0_f64;
    for (ra, rb) in a.iter().zip(b.iter()) {
        for (va, vb) in ra.iter().zip(rb.iter()) {
            m = nan_max(m, (va - vb).abs());
        }
    }
    m
}

/// `f64::max` returns the other operand when one is NaN, so folding residuals with it reads a
/// NaN entry as agreement. This keeps the NaN, and `NaN <= tol` then fails the case.
fn nan_max(acc: f64, d: f64) -> f64 {
    if acc.is_nan() || d.is_nan() {
        f64::NAN
    } else {
        acc.max(d)
    }
}

#[test]
fn metamorphic_stats_multivariate_normal_rvs() {
    let start = Instant::now();
    let mut cases = Vec::new();

    let fixtures: Vec<(&str, Vec<f64>, Vec<Vec<f64>>, u64)> = vec![
        // 1-D standard normal
        ("d1_standard", vec![0.0], vec![vec![1.0]], 42),
        // 2-D, identity cov
        (
            "d2_identity",
            vec![1.0, -2.0],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            123,
        ),
        // 2-D, correlated
        (
            "d2_correlated",
            vec![5.0, -3.0],
            vec![vec![2.0, 0.5], vec![0.5, 1.0]],
            7,
        ),
        // 3-D, mild correlation
        (
            "d3_mild_corr",
            vec![0.0, 0.0, 0.0],
            vec![
                vec![1.0, 0.2, 0.1],
                vec![0.2, 1.0, 0.3],
                vec![0.1, 0.3, 1.0],
            ],
            999,
        ),
    ];

    for (name, mean, cov, seed) in &fixtures {
        let d = mean.len();
        let n_samples = 50_000;
        let samples = multivariate_normal_rvs(mean, cov, n_samples, *seed);

        // Invariant 1: shape n_samples × d.
        let shape_pass = samples.len() == n_samples && samples.iter().all(|row| row.len() == d);
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "shape_n_by_d".into(),
            detail: format!(
                "rows={}, expected_n={}, all_cols_eq_d={}",
                samples.len(),
                n_samples,
                samples.iter().all(|row| row.len() == d)
            ),
            pass: shape_pass,
        });

        // Invariant 2: determinism at fixed seed.
        let samples_again = multivariate_normal_rvs(mean, cov, n_samples, *seed);
        let determ_pass = samples == samples_again;
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "deterministic_under_fixed_seed".into(),
            detail: format!(
                "first_eq_second={determ_pass}; first[0][0]={}",
                samples
                    .first()
                    .and_then(|r| r.first())
                    .copied()
                    .unwrap_or(f64::NAN)
            ),
            pass: determ_pass,
        });

        // Invariant 3: sample-mean LLN convergence.
        let m_hat = sample_mean(&samples, d);
        let mean_err = max_abs_diff_vec(&m_hat, mean);
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "sample_mean_lln_converges".into(),
            detail: format!("max |m̂ - μ| = {mean_err}"),
            pass: mean_err <= MEAN_TOL,
        });

        // Invariant 4: sample-cov LLN convergence.
        let cov_hat = sample_cov(&samples, &m_hat, d);
        let cov_err = max_abs_diff_mat(&cov_hat, cov);
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "sample_cov_lln_converges".into(),
            detail: format!("max |Σ̂ - Σ| = {cov_err}"),
            pass: cov_err <= COV_TOL,
        });

        // Invariant 5: dimension matches.
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "first_row_dimension".into(),
            detail: format!(
                "first row has {} entries, expected {}",
                samples.first().map(|r| r.len()).unwrap_or(0),
                d
            ),
            pass: samples.first().map(|r| r.len()).unwrap_or(0) == d,
        });
    }

    let pass_count = cases.iter().filter(|c| c.pass).count();
    let all_pass = pass_count == cases.len();

    let log = MetamorphicLog {
        test_id: "metamorphic_stats_multivariate_normal_rvs".into(),
        case_count: cases.len(),
        pass_count,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: cases.clone(),
    };

    emit_log(&log);

    for c in &cases {
        if !c.pass {
            eprintln!(
                "multivariate_normal_rvs metamorphic fail: {} {} — {}",
                c.case_id, c.invariant, c.detail
            );
        }
    }

    assert!(
        all_pass,
        "multivariate_normal_rvs metamorphic failed: {} of {} cases pass",
        pass_count,
        cases.len()
    );
}
