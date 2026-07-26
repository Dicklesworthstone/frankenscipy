use criterion::{BenchmarkId, Criterion, criterion_group};
use fsci_stats::{
    BINNED_STATISTIC_DD_3D_PARALLEL_DISABLE, BIWEIGHT_MAD_HOIST_DISABLE,
    BRUNNERMUNZEL_MATRIX_PRESORT_DISABLE, HaltonSampler, MAD_FN_REUSE_DISABLE, MAD_REUSE_DISABLE,
    MAD_ZSCORE_HOIST_DISABLE, MOMENT_PAR_FORCE_SERIAL, PAR_SUM_FORCE_SERIAL,
    STATS_CROSS_PRESORT_DISABLE, SobolSampler, SomersDInput, acf, argsort, bayes_mvs,
    binned_statistic, binned_statistic_2d, binned_statistic_dd, biweight_midcorrelation,
    brier_score, brunnermunzel_matrix, centered_discrepancy, cohens_d, ecdf, energy_distance,
    excess_kurtosis, gstd, histogram, kendalltau, kruskal, ks_2samp, ks_2samp_cross,
    l2_star_discrepancy, mad, mad_zscore, mannkendall, mannwhitneyu, mannwhitneyu_cross,
    mean_absolute_error, mean_squared_error, median_abs_deviation, mixture_discrepancy, pacf,
    pooled_variance, psd_welch, rand_index, siegelslopes, somersd, theilslopes, ttest_1samp,
    ttest_ind, ttest_rel, wasserstein_distance, weighted_mean, wraparound_discrepancy,
};
use std::hint::black_box;
use std::io::{self, Write as _};
use std::process::ExitCode;
use std::sync::atomic::Ordering;
use std::time::Instant;

use sha2::{Digest, Sha256};

const FRONTIER_ROUNDS: usize = 41;
const FRONTIER_MIN_OF: usize = 3;
const FRONTIER_BOOTSTRAP_RESAMPLES: usize = 10_000;
const FRONTIER_MIN_SAMPLE_MS: f64 = 2.0;

fn deterministic_data(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            (x * 0.017).sin() + (x * 0.031).cos() * 0.25 + (i % 17) as f64 * 0.001
        })
        .collect()
}

fn weighted_mean_two_pass(values: &[f64], weights: &[f64]) -> f64 {
    let total_w: f64 = weights.iter().sum();
    values
        .iter()
        .zip(weights)
        .map(|(&value, &weight)| value * weight)
        .sum::<f64>()
        / total_w
}

fn bench_stats_cross_presort_ab(c: &mut Criterion) {
    // Two groups, m×k rectangular cross. Mild ties (integer-quantized).
    let (m, k, n) = (60usize, 60usize, 3000usize);
    let make_group = |salt: usize, count: usize| -> Vec<Vec<f64>> {
        (0..count)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let x = ((i * 131 + j * 977 + salt) % 4096) as f64 / 41.0;
                        (x * 8.0).round() / 8.0
                    })
                    .collect()
            })
            .collect()
    };
    let a = make_group(17, m);
    let b = make_group(4099, k);

    // Byte-identity gate for both two-output cross forms over all m·k pairs.
    for name in ["mannwhitneyu", "ks_2samp"] {
        STATS_CROSS_PRESORT_DISABLE.store(false, Ordering::Relaxed);
        let (fs, fp) = if name == "mannwhitneyu" {
            mannwhitneyu_cross(&a, &b).unwrap()
        } else {
            ks_2samp_cross(&a, &b).unwrap()
        };
        STATS_CROSS_PRESORT_DISABLE.store(true, Ordering::Relaxed);
        let (ss, sp) = if name == "mannwhitneyu" {
            mannwhitneyu_cross(&a, &b).unwrap()
        } else {
            ks_2samp_cross(&a, &b).unwrap()
        };
        STATS_CROSS_PRESORT_DISABLE.store(false, Ordering::Relaxed);
        for i in 0..m {
            for j in 0..k {
                assert_eq!(
                    fs[i][j].to_bits(),
                    ss[i][j].to_bits(),
                    "{name} stat ({i},{j})"
                );
                assert_eq!(
                    fp[i][j].to_bits(),
                    sp[i][j].to_bits(),
                    "{name} pval ({i},{j})"
                );
            }
        }
    }

    let mut group = c.benchmark_group("stats_cross_presort_ab");
    group.bench_function("mwu_original_per_pair_60x60_n3000", |bch| {
        STATS_CROSS_PRESORT_DISABLE.store(true, Ordering::Relaxed);
        bch.iter(|| black_box(mannwhitneyu_cross(black_box(&a), black_box(&b)).unwrap()));
        STATS_CROSS_PRESORT_DISABLE.store(false, Ordering::Relaxed);
    });
    group.bench_function("mwu_candidate_presort_60x60_n3000", |bch| {
        STATS_CROSS_PRESORT_DISABLE.store(false, Ordering::Relaxed);
        bch.iter(|| black_box(mannwhitneyu_cross(black_box(&a), black_box(&b)).unwrap()));
    });
    group.bench_function("ks_original_per_pair_60x60_n3000", |bch| {
        STATS_CROSS_PRESORT_DISABLE.store(true, Ordering::Relaxed);
        bch.iter(|| black_box(ks_2samp_cross(black_box(&a), black_box(&b)).unwrap()));
        STATS_CROSS_PRESORT_DISABLE.store(false, Ordering::Relaxed);
    });
    group.bench_function("ks_candidate_presort_60x60_n3000", |bch| {
        STATS_CROSS_PRESORT_DISABLE.store(false, Ordering::Relaxed);
        bch.iter(|| black_box(ks_2samp_cross(black_box(&a), black_box(&b)).unwrap()));
    });
    group.finish();
}

fn bench_brunnermunzel_matrix_presort_ab(c: &mut Criterion) {
    // m samples of length n, mild ties (integer-quantized) so the tie path exercises.
    let (m, n) = (120usize, 3000usize);
    let samples: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let x = ((i * 131 + j * 977 + 17) % 4096) as f64 / 41.0;
                    (x * 8.0).round() / 8.0
                })
                .collect()
        })
        .collect();

    // Byte-identity gate over all m² ordered pairs (stat AND pvalue), both matrices.
    BRUNNERMUNZEL_MATRIX_PRESORT_DISABLE.store(false, Ordering::Relaxed);
    let (fast_s, fast_p) = brunnermunzel_matrix(&samples).expect("presort matrix");
    BRUNNERMUNZEL_MATRIX_PRESORT_DISABLE.store(true, Ordering::Relaxed);
    let (slow_s, slow_p) = brunnermunzel_matrix(&samples).expect("per-pair matrix");
    BRUNNERMUNZEL_MATRIX_PRESORT_DISABLE.store(false, Ordering::Relaxed);
    for i in 0..m {
        for j in 0..m {
            assert_eq!(
                fast_s[i][j].to_bits(),
                slow_s[i][j].to_bits(),
                "stat ({i},{j})"
            );
            assert_eq!(
                fast_p[i][j].to_bits(),
                slow_p[i][j].to_bits(),
                "pval ({i},{j})"
            );
        }
    }

    let mut group = c.benchmark_group("brunnermunzel_matrix_presort_ab");
    group.bench_function("original_per_pair_m120_n3000", |b| {
        BRUNNERMUNZEL_MATRIX_PRESORT_DISABLE.store(true, Ordering::Relaxed);
        b.iter(|| black_box(brunnermunzel_matrix(black_box(&samples)).unwrap()));
        BRUNNERMUNZEL_MATRIX_PRESORT_DISABLE.store(false, Ordering::Relaxed);
    });
    group.bench_function("candidate_presort_m120_n3000", |b| {
        BRUNNERMUNZEL_MATRIX_PRESORT_DISABLE.store(false, Ordering::Relaxed);
        b.iter(|| black_box(brunnermunzel_matrix(black_box(&samples)).unwrap()));
    });
    group.finish();
}

fn bench_weighted_mean_fused_ab(c: &mut Criterion) {
    let len = 262_144usize;
    let values = deterministic_data(len);
    let weights: Vec<f64> = (0..len)
        .map(|i| 0.125 + ((i * 53 + 7) % 257) as f64 / 31.0)
        .collect();
    assert_eq!(
        weighted_mean(&values, &weights).to_bits(),
        weighted_mean_two_pass(&values, &weights).to_bits()
    );

    let mut group = c.benchmark_group("weighted_mean_fused_ab");
    group.bench_function("original_two_pass_n262144", |b| {
        b.iter(|| {
            black_box(weighted_mean_two_pass(
                black_box(&values),
                black_box(&weights),
            ))
        })
    });
    group.bench_function("candidate_fused_n262144", |b| {
        b.iter(|| black_box(weighted_mean(black_box(&values), black_box(&weights))))
    });
    group.finish();
}

fn mean_absolute_error_scalar_reference(y_true: &[f64], y_pred: &[f64]) -> f64 {
    y_true
        .iter()
        .zip(y_pred)
        .map(|(&truth, &prediction)| (truth - prediction).abs())
        .sum::<f64>()
        / y_true.len() as f64
}

fn bench_mean_absolute_error_simd_ab(c: &mut Criterion) {
    let len = 2_097_152usize;
    let y_true = deterministic_data(len);
    let y_pred: Vec<f64> = (0..len)
        .map(|idx| ((idx % 251) as f64 - 125.0) / 31.0)
        .collect();
    let mut group = c.benchmark_group("mean_absolute_error_simd_ab");
    group.sample_size(15);
    group.bench_function("current_simd_n2097152", |bencher| {
        bencher.iter(|| black_box(mean_absolute_error(black_box(&y_true), black_box(&y_pred))))
    });
    group.bench_function("orig_scalar_n2097152", |bencher| {
        bencher.iter(|| {
            black_box(mean_absolute_error_scalar_reference(
                black_box(&y_true),
                black_box(&y_pred),
            ))
        })
    });
    group.finish();
}

fn mean_squared_error_scalar_reference(y_true: &[f64], y_pred: &[f64]) -> f64 {
    y_true
        .iter()
        .zip(y_pred)
        .map(|(&truth, &prediction)| (truth - prediction).powi(2))
        .sum::<f64>()
        / y_true.len() as f64
}

fn bench_mean_squared_error_simd_ab(c: &mut Criterion) {
    let len = 262_144usize;
    let y_true = deterministic_data(len);
    let y_pred: Vec<f64> = (0..len)
        .map(|idx| ((idx % 251) as f64 - 125.0) / 31.0)
        .collect();
    let mut group = c.benchmark_group("mean_squared_error_simd_ab");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_millis(200));
    group.measurement_time(std::time::Duration::from_secs(1));
    group.bench_function("current_simd_n262144", |bencher| {
        bencher.iter(|| black_box(mean_squared_error(black_box(&y_true), black_box(&y_pred))))
    });
    group.bench_function("orig_scalar_n262144", |bencher| {
        bencher.iter(|| {
            black_box(mean_squared_error_scalar_reference(
                black_box(&y_true),
                black_box(&y_pred),
            ))
        })
    });
    group.finish();
}

fn brier_score_scalar_reference(y_true: &[f64], y_pred: &[f64]) -> f64 {
    y_true
        .iter()
        .zip(y_pred)
        .map(|(&truth, &prediction)| (prediction - truth).powi(2))
        .sum::<f64>()
        / y_true.len() as f64
}

fn bench_brier_score_simd_ab(c: &mut Criterion) {
    let len = 262_144usize;
    let y_true: Vec<f64> = (0..len).map(|idx| (idx % 2) as f64).collect();
    let y_pred: Vec<f64> = (0..len)
        .map(|idx| ((idx % 997) as f64 + 0.5) / 998.0)
        .collect();
    let mut group = c.benchmark_group("brier_score_simd_ab");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_millis(200));
    group.measurement_time(std::time::Duration::from_secs(1));
    group.bench_function("current_simd_n262144", |bencher| {
        bencher.iter(|| black_box(brier_score(black_box(&y_true), black_box(&y_pred))))
    });
    group.bench_function("orig_scalar_n262144", |bencher| {
        bencher.iter(|| {
            black_box(brier_score_scalar_reference(
                black_box(&y_true),
                black_box(&y_pred),
            ))
        })
    });
    group.finish();
}

fn qmc_halton_sample(n: usize, dimension: usize) -> Vec<f64> {
    let mut sampler = HaltonSampler::new(dimension).expect("valid Halton dimension");
    sampler.sample(n)
}

fn qmc_sobol_sample(n: usize, dimension: usize) -> Vec<f64> {
    let mut sampler = SobolSampler::new(dimension).expect("valid Sobol dimension");
    sampler.sample(n)
}

fn bench_qmc_discrepancy(c: &mut Criterion) {
    let sample = qmc_halton_sample(512, 2);
    let mut group = c.benchmark_group("qmc_discrepancy");
    group.bench_function("centered/512x2", |b| {
        b.iter(|| centered_discrepancy(&sample, 2).expect("centered discrepancy"))
    });
    group.bench_function("mixture/512x2", |b| {
        b.iter(|| mixture_discrepancy(&sample, 2).expect("mixture discrepancy"))
    });
    group.bench_function("l2_star/512x2", |b| {
        b.iter(|| l2_star_discrepancy(&sample, 2).expect("l2 star discrepancy"))
    });
    group.bench_function("wraparound/512x2", |b| {
        b.iter(|| wraparound_discrepancy(&sample, 2).expect("wraparound discrepancy"))
    });
    group.finish();
}

fn bench_qmc_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("qmc_sampling");
    for n in [1024usize, 4096] {
        group.bench_function(BenchmarkId::new("halton_4d", n), |b| {
            b.iter(|| qmc_halton_sample(n, 4))
        });
        group.bench_function(BenchmarkId::new("sobol_2d", n), |b| {
            b.iter(|| qmc_sobol_sample(n, 2))
        });
    }
    group.bench_function(BenchmarkId::new("sobol_8d", 65_536usize), |b| {
        b.iter(|| qmc_sobol_sample(65_536, 8))
    });
    group.finish();
}

fn bench_ordering_and_bins(c: &mut Criterion) {
    let data = deterministic_data(4096);
    let x_eval = deterministic_data(512);
    let data_big = deterministic_data(65536);
    let mut group = c.benchmark_group("ordering_and_bins");
    group.bench_function("argsort/4096", |b| b.iter(|| argsort(&data)));
    group.bench_function("argsort/65536", |b| b.iter(|| argsort(&data_big)));
    group.bench_function("histogram/4096x64", |b| b.iter(|| histogram(&data, 64)));
    group.bench_function("ecdf/4096x512", |b| b.iter(|| ecdf(&data, &x_eval)));
    group.finish();
}

fn bench_time_series(c: &mut Criterion) {
    let data = deterministic_data(4096);
    let mut group = c.benchmark_group("time_series");
    group.bench_function("acf/4096x64", |b| b.iter(|| acf(&data, 64)));
    group.bench_function("pacf/4096x64", |b| b.iter(|| pacf(&data, 64)));
    group.bench_function("psd_welch/4096_w128_o64", |b| {
        b.iter(|| psd_welch(&data, 128, 64, 1.0))
    });
    group.finish();
}

fn bench_rank_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank_correlation");
    for &n in &[2048usize, 4096] {
        // x roughly monotone, y a noisy/tied transform — exercises ties.
        let x = deterministic_data(n);
        let y: Vec<f64> = (0..n)
            .map(|i| ((i as f64 * 0.013).cos() + (i % 11) as f64 * 0.1).round())
            .collect();
        group.bench_function(BenchmarkId::new("kendalltau", n), |b| {
            b.iter(|| kendalltau(&x, &y))
        });
        group.bench_function(BenchmarkId::new("mannkendall", n), |b| {
            b.iter(|| mannkendall(&x))
        });
    }
    group.finish();
}

fn bench_somersd(c: &mut Criterion) {
    let mut group = c.benchmark_group("somersd");
    // Distinct ranks -> an n x n contingency table, the worst case for the
    // per-cell quadrant re-summation (O((R*C)^2)).
    for &n in &[64usize, 128] {
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % n) as f64).collect();
        group.bench_function(BenchmarkId::new("rankings", n), |b| {
            b.iter(|| somersd(SomersDInput::Rankings(&x, &y), None))
        });
    }
    group.finish();
}

fn bench_rand_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("rand_index");
    for &n in &[2000usize, 8000] {
        let lt: Vec<f64> = (0..n).map(|i| (i % 10) as f64).collect();
        let lp: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 11) as f64).collect();
        group.bench_function(BenchmarkId::new("k10", n), |b| {
            b.iter(|| rand_index(&lt, &lp))
        });
    }
    group.finish();
}

/// Batch density/mass evaluation vs mapping the scalar pdf/pmf (the "original").
/// Quantifies the constant-special-function-normalizer hoist: pdf_many/pmf_many
/// compute the expensive lgamma/ln_beta normalizer ONCE instead of per point.
fn bench_distribution_batch(c: &mut Criterion) {
    use fsci_stats::{
        BetaBinomial, BetaDist, BetaPrime, Binomial, Chi, ChiSquared, ContinuousDistribution,
        DiscreteDistribution, FDistribution, GammaDist, GenGamma, GenHyperbolic, GenInvGauss,
        GenNorm, Hypergeometric, InverseGamma, JfSkewT, Nakagami, NegBinomial, Poisson, StudentT,
        VonMises,
    };
    let n = 4096usize;
    let mut group = c.benchmark_group("distribution_batch");
    let positive_x: Vec<f64> = (0..n).map(|i| 0.001 + (i as f64 + 0.5) * 0.005).collect();
    let symmetric_x: Vec<f64> = (0..n)
        .map(|i| -8.0 + 16.0 * (i as f64 + 0.5) / n as f64)
        .collect();
    let angles: Vec<f64> = (0..n)
        .map(|i| -std::f64::consts::PI + std::f64::consts::TAU * (i as f64 + 0.5) / n as f64)
        .collect();

    // GammaDist: 1 lgamma/point hoisted.
    let g = GammaDist::new(2.7, 1.5);
    let gx: Vec<f64> = (0..n).map(|i| 0.01 + i as f64 * 0.01).collect();
    group.bench_function("gamma/pdf_many", |b| b.iter(|| g.pdf_many(&gx)));
    group.bench_function("gamma/map_pdf", |b| {
        b.iter(|| gx.iter().map(|&x| g.pdf(x)).collect::<Vec<_>>())
    });

    // BetaDist: 3 lgamma/point (ln_beta) hoisted.
    let bt = BetaDist::new(2.5, 3.5);
    let bx: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    group.bench_function("beta/pdf_many", |b| b.iter(|| bt.pdf_many(&bx)));
    group.bench_function("beta/map_pdf", |b| {
        b.iter(|| bx.iter().map(|&x| bt.pdf(x)).collect::<Vec<_>>())
    });

    // Beta-prime: three parameter-only lgamma evaluations in ln_beta hoisted.
    let beta_prime = BetaPrime::new(2.5, 3.75);
    group.bench_function("betaprime/logpdf_many", |b| {
        b.iter(|| beta_prime.logpdf_many(&positive_x))
    });
    group.bench_function("betaprime/map_logpdf", |b| {
        b.iter(|| {
            positive_x
                .iter()
                .map(|&x| beta_prime.logpdf(x))
                .collect::<Vec<_>>()
        })
    });

    // StudentT: gamma-ratio coefficient hoisted over a symmetric grid.
    let st = StudentT::new(7.5);
    group.bench_function("student_t/pdf_many", |b| {
        b.iter(|| st.pdf_many(&symmetric_x))
    });
    group.bench_function("student_t/map_pdf", |b| {
        b.iter(|| symmetric_x.iter().map(|&x| st.pdf(x)).collect::<Vec<_>>())
    });

    // Chi/ChiSquared: lgamma normalizers hoisted over positive support.
    let chi = Chi::new(5.5);
    group.bench_function("chi/pdf_many", |b| b.iter(|| chi.pdf_many(&positive_x)));
    group.bench_function("chi/map_pdf", |b| {
        b.iter(|| positive_x.iter().map(|&x| chi.pdf(x)).collect::<Vec<_>>())
    });
    let chi2 = ChiSquared::new(7.5);
    group.bench_function("chi2/pdf_many", |b| b.iter(|| chi2.pdf_many(&positive_x)));
    group.bench_function("chi2/map_pdf", |b| {
        b.iter(|| positive_x.iter().map(|&x| chi2.pdf(x)).collect::<Vec<_>>())
    });

    // FDistribution and generalized gamma: beta/gamma normalizers hoisted.
    let fdist = FDistribution::new(9.0, 17.0);
    group.bench_function("f/pdf_many", |b| b.iter(|| fdist.pdf_many(&positive_x)));
    group.bench_function("f/map_pdf", |b| {
        b.iter(|| positive_x.iter().map(|&x| fdist.pdf(x)).collect::<Vec<_>>())
    });
    let gengamma = GenGamma::new(2.5, 1.7);
    group.bench_function("gengamma/pdf_many", |b| {
        b.iter(|| gengamma.pdf_many(&positive_x))
    });
    group.bench_function("gengamma/map_pdf", |b| {
        b.iter(|| {
            positive_x
                .iter()
                .map(|&x| gengamma.pdf(x))
                .collect::<Vec<_>>()
        })
    });

    // Inverse-gamma, Nakagami, generalized normal, and Von Mises hoist
    // distribution-wide special-function constants out of the timed map.
    let invgamma = InverseGamma::new(3.5);
    group.bench_function("invgamma/pdf_many", |b| {
        b.iter(|| invgamma.pdf_many(&positive_x))
    });
    group.bench_function("invgamma/map_pdf", |b| {
        b.iter(|| {
            positive_x
                .iter()
                .map(|&x| invgamma.pdf(x))
                .collect::<Vec<_>>()
        })
    });
    let nakagami = Nakagami::new(2.25);
    group.bench_function("nakagami/pdf_many", |b| {
        b.iter(|| nakagami.pdf_many(&positive_x))
    });
    group.bench_function("nakagami/map_pdf", |b| {
        b.iter(|| {
            positive_x
                .iter()
                .map(|&x| nakagami.pdf(x))
                .collect::<Vec<_>>()
        })
    });
    let gennorm = GenNorm::new(1.5);
    group.bench_function("gennorm/pdf_many", |b| {
        b.iter(|| gennorm.pdf_many(&symmetric_x))
    });
    group.bench_function("gennorm/map_pdf", |b| {
        b.iter(|| {
            symmetric_x
                .iter()
                .map(|&x| gennorm.pdf(x))
                .collect::<Vec<_>>()
        })
    });
    let vonmises = VonMises::new(3.0, 0.25);
    group.bench_function("vonmises/pdf_many", |b| {
        b.iter(|| vonmises.pdf_many(&angles))
    });
    group.bench_function("vonmises/map_pdf", |b| {
        b.iter(|| angles.iter().map(|&x| vonmises.pdf(x)).collect::<Vec<_>>())
    });

    // Generalized inverse Gaussian: the scaled Bessel K normalizer depends only
    // on the distribution parameters and is shared across the batch.
    let geninvgauss = GenInvGauss::new(-0.75, 3.0);
    group.bench_function("geninvgauss/logpdf_many", |b| {
        b.iter(|| geninvgauss.logpdf_many(&positive_x))
    });
    group.bench_function("geninvgauss/map_logpdf", |b| {
        b.iter(|| {
            positive_x
                .iter()
                .map(|&x| geninvgauss.logpdf(x))
                .collect::<Vec<_>>()
        })
    });

    // Generalized hyperbolic: one of two Bessel-K calls is parameter-only.
    let genhyperbolic = GenHyperbolic::new(0.75, 2.5, -0.5);
    group.bench_function("genhyperbolic/logpdf_many", |b| {
        b.iter(|| genhyperbolic.logpdf_many(&symmetric_x))
    });
    group.bench_function("genhyperbolic/map_logpdf", |b| {
        b.iter(|| {
            symmetric_x
                .iter()
                .map(|&x| genhyperbolic.logpdf(x))
                .collect::<Vec<_>>()
        })
    });

    // Jones-Faddy skew-t: the beta normalizer depends only on the shape parameters.
    let jf_skew_t = JfSkewT::new(3.25, 1.75);
    group.bench_function("jf_skew_t/logpdf_many", |b| {
        b.iter(|| jf_skew_t.logpdf_many(&symmetric_x))
    });
    group.bench_function("jf_skew_t/map_logpdf", |b| {
        b.iter(|| {
            symmetric_x
                .iter()
                .map(|&x| jf_skew_t.logpdf(x))
                .collect::<Vec<_>>()
        })
    });

    // Binomial: 3 parameter-only terms hoisted over a full support sweep.
    let bin = Binomial::new(2000, 0.37);
    let bin_ks: Vec<u64> = (0..=2000).collect();
    group.bench_function("binomial/pmf_many", |b| b.iter(|| bin.pmf_many(&bin_ks)));
    group.bench_function("binomial/map_pmf", |b| {
        b.iter(|| bin_ks.iter().map(|&k| bin.pmf(k)).collect::<Vec<_>>())
    });

    // Negative binomial: parameter-only lnGamma/log terms hoisted across a long tail.
    let nbin = NegBinomial::new(20.0, 0.42);
    let tail_ks: Vec<u64> = (0..n as u64).collect();
    group.bench_function("negbinom/pmf_many", |b| b.iter(|| nbin.pmf_many(&tail_ks)));
    group.bench_function("negbinom/map_pmf", |b| {
        b.iter(|| tail_ks.iter().map(|&k| nbin.pmf(k)).collect::<Vec<_>>())
    });

    // Beta-binomial: 5 lgamma/point hoisted over a bounded support.
    let bb = BetaBinomial::new(2000, 2.5, 7.0);
    let bb_ks: Vec<u64> = (0..=2000).collect();
    group.bench_function("betabinom/pmf_many", |b| b.iter(|| bb.pmf_many(&bb_ks)));
    group.bench_function("betabinom/map_pmf", |b| {
        b.iter(|| bb_ks.iter().map(|&k| bb.pmf(k)).collect::<Vec<_>>())
    });

    // Hypergeometric: 5 lgamma/point hoisted (Fisher's-exact-test full support).
    let h = Hypergeometric::new(2000, 700, 1200);
    let ks: Vec<u64> = (0..=700).collect();
    group.bench_function("hypergeom/pmf_many", |b| b.iter(|| h.pmf_many(&ks)));
    group.bench_function("hypergeom/map_pmf", |b| {
        b.iter(|| ks.iter().map(|&k| h.pmf(k)).collect::<Vec<_>>())
    });

    // Poisson log-pmf: ln(mu) hoisted over a full count-data support sweep.
    let poisson = Poisson::new(37.0);
    let poisson_ks: Vec<u64> = (0..n as u64).collect();
    group.bench_function("poisson/logpmf_many", |b| {
        b.iter(|| poisson.logpmf_many(&poisson_ks))
    });
    group.bench_function("poisson/map_logpmf", |b| {
        b.iter(|| {
            poisson_ks
                .iter()
                .map(|&k| poisson.logpmf(k))
                .collect::<Vec<_>>()
        })
    });

    group.finish();
}

/// Gaussian KDE evaluation at many points — each point is an O(n_data) sum over the
/// dataset (heavy per-point work), parallelized in evaluate_many. Head-to-head vs
/// scipy.stats.gaussian_kde (docs/perf_oracle_kde.py).
fn bench_rank_tests(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank_tests");
    let n = 200_000usize;
    let x: Vec<f64> = (0..n)
        .map(|i| ((i * 2654435761usize) % 100003) as f64 * 1e-4)
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| ((i * 40503usize + 7) % 100003) as f64 * 1e-4 + 0.1)
        .collect();
    group.bench_function("ks_2samp_200k", |b| {
        b.iter(|| ks_2samp(black_box(&x), black_box(&y)))
    });
    group.bench_function("mannwhitneyu_200k", |b| {
        b.iter(|| mannwhitneyu(black_box(&x), black_box(&y)))
    });
    group.bench_function("kruskal_200k", |b| {
        b.iter(|| kruskal(&[black_box(&x), black_box(&y)]))
    });
    group.finish();
}

fn bench_mvt_pdf(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use fsci_stats::MultivariateT;
    let mut group = c.benchmark_group("multivariate_t_pdf");
    let m = 100_000usize;
    for &d in &[3usize, 10] {
        let loc: Vec<f64> = (0..d).map(|i| (i as f64) * 0.1).collect();
        let a: Vec<Vec<f64>> = (0..d)
            .map(|i| {
                (0..d)
                    .map(|j| ((i * 7 + j * 3) % 5) as f64 * 0.2 - 0.4)
                    .collect()
            })
            .collect();
        let mut shape = vec![vec![0.0; d]; d];
        for i in 0..d {
            for j in 0..d {
                let mut sv = 0.0;
                for k in 0..d {
                    sv += a[i][k] * a[j][k];
                }
                shape[i][j] = sv + if i == j { d as f64 } else { 0.0 };
            }
        }
        let mvt = MultivariateT::new(&loc, &shape, 5.0).expect("mvt");
        let q: Vec<Vec<f64>> = (0..m)
            .map(|t| {
                (0..d)
                    .map(|j| ((t * 13 + j * 17) % 1000) as f64 * 0.01 - 5.0)
                    .collect()
            })
            .collect();
        group.bench_function(BenchmarkId::new("pdf_many", d), |b| {
            b.iter(|| mvt.pdf_many(black_box(&q)).expect("pdf"))
        });
    }
    group.finish();
}

fn bench_mvn_pdf(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use fsci_stats::MultivariateNormal;
    let mut group = c.benchmark_group("multivariate_normal_pdf");
    let m = 100_000usize;
    for &d in &[3usize, 5, 8, 10] {
        let mean: Vec<f64> = (0..d).map(|i| (i as f64) * 0.1).collect();
        // SPD covariance: A A^T + d I
        let a: Vec<Vec<f64>> = (0..d)
            .map(|i| {
                (0..d)
                    .map(|j| ((i * 7 + j * 3) % 5) as f64 * 0.2 - 0.4)
                    .collect()
            })
            .collect();
        let mut cov = vec![vec![0.0; d]; d];
        for i in 0..d {
            for j in 0..d {
                let mut s = 0.0;
                for k in 0..d {
                    s += a[i][k] * a[j][k];
                }
                cov[i][j] = s + if i == j { d as f64 } else { 0.0 };
            }
        }
        let mvn = MultivariateNormal::new(&mean, &cov).expect("mvn");
        let q: Vec<Vec<f64>> = (0..m)
            .map(|t| {
                (0..d)
                    .map(|j| ((t * 13 + j * 17) % 1000) as f64 * 0.01 - 5.0)
                    .collect()
            })
            .collect();
        group.bench_function(BenchmarkId::new("pdf_many", d), |b| {
            b.iter(|| mvn.pdf_many(black_box(&q)).expect("pdf"))
        });
    }
    group.finish();
}

fn bench_kde_nd(c: &mut Criterion) {
    use fsci_stats::GaussianKdeNd;
    let mut group = c.benchmark_group("gaussian_kde_nd");
    let n = 2000usize;
    let m = 5000usize;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let t = i as f64;
            vec![
                (t * 0.017).sin(),
                (t * 0.0031).cos() * 2.0,
                (t * 0.011).sin() * 0.5,
            ]
        })
        .collect();
    let kde = GaussianKdeNd::new(&data).expect("kde");
    let q: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            let t = i as f64;
            vec![
                (t * 0.02).cos(),
                (t * 0.005).sin() * 2.0,
                (t * 0.013).cos() * 0.5,
            ]
        })
        .collect();
    group.bench_function("d3_eval5k", |b| b.iter(|| kde.evaluate_many(black_box(&q))));
    group.finish();
}

#[derive(Clone, Copy)]
enum KdeQueryTileArm {
    Baseline,
    Candidate,
}

impl KdeQueryTileArm {
    const fn tile_disabled(self) -> bool {
        matches!(self, Self::Baseline)
    }
}

struct KdeQueryTileFixture {
    kde: fsci_stats::GaussianKdeNd,
    queries: Vec<Vec<f64>>,
}

struct FrontierPairedStats {
    arm_a_median_ms: f64,
    arm_b_median_ms: f64,
    arm_a_mad_ms: f64,
    arm_b_mad_ms: f64,
    arm_a_cv: f64,
    arm_b_cv: f64,
    ratio_median: f64,
    ratio_mad: f64,
    ratio_cv: f64,
    ratio_ci_low: f64,
    ratio_ci_high: f64,
    ratios: Vec<f64>,
    checksum: u64,
}

struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }
}

fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "median requires at least one sample");
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let midpoint = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[midpoint - 1] + sorted[midpoint]) * 0.5
    } else {
        sorted[midpoint]
    }
}

fn median_absolute_deviation(values: &[f64]) -> f64 {
    let center = median(values);
    let deviations: Vec<f64> = values.iter().map(|value| (value - center).abs()).collect();
    median(&deviations)
}

fn coefficient_of_variation(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sum_squared_deviations = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>();
    let standard_deviation =
        (sum_squared_deviations / values.len().saturating_sub(1).max(1) as f64).sqrt();
    standard_deviation / mean
}

fn bootstrap_median_ci(values: &[f64], seed: u64) -> (f64, f64) {
    let mut generator = XorShift64(seed);
    let mut sample = Vec::with_capacity(values.len());
    let mut medians = Vec::with_capacity(FRONTIER_BOOTSTRAP_RESAMPLES);
    for _ in 0..FRONTIER_BOOTSTRAP_RESAMPLES {
        sample.clear();
        for _ in 0..values.len() {
            sample.push(values[generator.next() as usize % values.len()]);
        }
        medians.push(median(&sample));
    }
    medians.sort_by(f64::total_cmp);
    let low = FRONTIER_BOOTSTRAP_RESAMPLES * 25 / 1_000;
    let high = FRONTIER_BOOTSTRAP_RESAMPLES * 975 / 1_000;
    (medians[low], medians[high.min(medians.len() - 1)])
}

fn make_kde_query_tile_fixture() -> Result<KdeQueryTileFixture, String> {
    use fsci_stats::GaussianKdeNd;

    const DATA_POINTS: usize = 2_000;
    const QUERIES: usize = 1_024;
    let data: Vec<Vec<f64>> = (0..DATA_POINTS)
        .map(|i| {
            let t = i as f64;
            vec![
                (t * 0.017).sin(),
                (t * 0.0031).cos() * 2.0,
                (t * 0.011).sin() * 0.5,
            ]
        })
        .collect();
    let queries = (0..QUERIES)
        .map(|i| {
            let t = i as f64;
            vec![
                (t * 0.02).cos(),
                (t * 0.005).sin() * 2.0,
                (t * 0.013).cos() * 0.5,
            ]
        })
        .collect();
    let kde = GaussianKdeNd::new(&data)
        .ok_or_else(|| String::from("failed to construct N-D KDE fixture"))?;
    Ok(KdeQueryTileFixture { kde, queries })
}

fn kde_query_tile_checksum(values: &[f64], repetition: usize) -> u64 {
    let mut checksum = 0x6a09_e667_f3bc_c909_u64 ^ repetition as u64;
    for &value in values {
        checksum = checksum.rotate_left(9) ^ value.to_bits();
    }
    checksum
}

fn time_kde_query_tile_arm(
    fixture: &KdeQueryTileFixture,
    arm: KdeQueryTileArm,
    repetitions: usize,
) -> (f64, u64) {
    use fsci_stats::GAUSSIAN_KDE_ND_QUERY_TILE_DISABLE;

    GAUSSIAN_KDE_ND_QUERY_TILE_DISABLE.store(arm.tile_disabled(), Ordering::Relaxed);
    let started = Instant::now();
    let mut checksum = 0xbb67_ae85_84ca_a73b_u64;
    for repetition in 0..repetitions {
        let values = black_box(
            fixture
                .kde
                .evaluate_many(black_box(fixture.queries.as_slice())),
        );
        checksum = checksum
            .rotate_left(7)
            .wrapping_add(kde_query_tile_checksum(&values, repetition));
        black_box(checksum);
    }
    (started.elapsed().as_secs_f64() * 1_000.0, checksum)
}

fn min_kde_query_tile_sample(
    fixture: &KdeQueryTileFixture,
    arm: KdeQueryTileArm,
    repetitions: usize,
) -> (f64, u64) {
    let mut best_ms = f64::INFINITY;
    let mut best_checksum = 0;
    for _ in 0..FRONTIER_MIN_OF {
        let (elapsed_ms, checksum) = time_kde_query_tile_arm(fixture, arm, repetitions);
        if elapsed_ms < best_ms {
            best_ms = elapsed_ms;
            best_checksum = checksum;
        }
    }
    (best_ms, best_checksum)
}

fn calibrate_kde_query_tile_repetitions(fixture: &KdeQueryTileFixture) -> Result<usize, String> {
    let mut repetitions = 1usize;
    loop {
        let (elapsed_ms, checksum) =
            time_kde_query_tile_arm(fixture, KdeQueryTileArm::Candidate, repetitions);
        black_box(checksum);
        if elapsed_ms >= FRONTIER_MIN_SAMPLE_MS {
            return Ok(repetitions);
        }
        repetitions = repetitions
            .checked_mul(2)
            .ok_or_else(|| String::from("KDE frontier calibration repetition count overflowed"))?;
    }
}

fn paired_kde_query_tile(
    fixture: &KdeQueryTileFixture,
    arm_a: KdeQueryTileArm,
    arm_b: KdeQueryTileArm,
    repetitions: usize,
    bootstrap_seed: u64,
) -> FrontierPairedStats {
    let mut arm_a_ms = Vec::with_capacity(FRONTIER_ROUNDS);
    let mut arm_b_ms = Vec::with_capacity(FRONTIER_ROUNDS);
    let mut ratios = Vec::with_capacity(FRONTIER_ROUNDS);
    let mut combined_checksum = 0u64;

    for round in 0..FRONTIER_ROUNDS {
        let ((a_ms, a_checksum), (b_ms, b_checksum)) = if round.is_multiple_of(2) {
            (
                min_kde_query_tile_sample(fixture, arm_a, repetitions),
                min_kde_query_tile_sample(fixture, arm_b, repetitions),
            )
        } else {
            let b = min_kde_query_tile_sample(fixture, arm_b, repetitions);
            let a = min_kde_query_tile_sample(fixture, arm_a, repetitions);
            (a, b)
        };
        arm_a_ms.push(a_ms);
        arm_b_ms.push(b_ms);
        ratios.push(a_ms / b_ms);
        combined_checksum = combined_checksum
            .rotate_left(7)
            .wrapping_add(a_checksum.rotate_left(17))
            .wrapping_add(b_checksum.rotate_right(11))
            .wrapping_add(round as u64 + 1);
    }

    let (ratio_ci_low, ratio_ci_high) = bootstrap_median_ci(&ratios, bootstrap_seed);
    FrontierPairedStats {
        arm_a_median_ms: median(&arm_a_ms),
        arm_b_median_ms: median(&arm_b_ms),
        arm_a_mad_ms: median_absolute_deviation(&arm_a_ms),
        arm_b_mad_ms: median_absolute_deviation(&arm_b_ms),
        arm_a_cv: coefficient_of_variation(&arm_a_ms),
        arm_b_cv: coefficient_of_variation(&arm_b_ms),
        ratio_median: median(&ratios),
        ratio_mad: median_absolute_deviation(&ratios),
        ratio_cv: coefficient_of_variation(&ratios),
        ratio_ci_low,
        ratio_ci_high,
        ratios,
        checksum: combined_checksum,
    }
}

fn print_frontier_paired_stats(label: &str, stats: &FrontierPairedStats) {
    println!(
        "paired label={label} rounds={FRONTIER_ROUNDS} min_of={FRONTIER_MIN_OF} \
         arm_a_median_ms={:.9} arm_b_median_ms={:.9} arm_a_mad_ms={:.9} \
         arm_b_mad_ms={:.9} ratio_median={:.9} ratio_mad={:.9} \
         ratio_median_ci95=[{:.9},{:.9}] arm_a_cv_provenance={:.6} \
         arm_b_cv_provenance={:.6} ratio_cv_provenance={:.6} checksum={:016x}",
        stats.arm_a_median_ms,
        stats.arm_b_median_ms,
        stats.arm_a_mad_ms,
        stats.arm_b_mad_ms,
        stats.ratio_median,
        stats.ratio_mad,
        stats.ratio_ci_low,
        stats.ratio_ci_high,
        stats.arm_a_cv,
        stats.arm_b_cv,
        stats.ratio_cv,
        stats.checksum
    );
    print!("paired_ratios label={label}");
    for ratio in &stats.ratios {
        print!(" {ratio:.9}");
    }
    println!();
}

fn prove_kde_query_tile_contract(fixture: &KdeQueryTileFixture) -> Result<(), String> {
    use fsci_stats::GAUSSIAN_KDE_ND_QUERY_TILE_DISABLE;

    GAUSSIAN_KDE_ND_QUERY_TILE_DISABLE.store(true, Ordering::Relaxed);
    let baseline = fixture.kde.evaluate_many(&fixture.queries);
    GAUSSIAN_KDE_ND_QUERY_TILE_DISABLE.store(false, Ordering::Relaxed);
    let candidate = fixture.kde.evaluate_many(&fixture.queries);
    if candidate.len() != baseline.len() {
        return Err(format!(
            "KDE query tile changed output length: baseline={} candidate={}",
            baseline.len(),
            candidate.len()
        ));
    }
    let bit_mismatches = candidate
        .iter()
        .zip(&baseline)
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count();
    if bit_mismatches != 0 {
        return Err(format!(
            "KDE query tile changed {bit_mismatches} output bit patterns"
        ));
    }
    println!(
        "frontier_contract exact_bits=true outputs={} bit_mismatches={bit_mismatches} \
         baseline_checksum={:016x} candidate_checksum={:016x}",
        baseline.len(),
        kde_query_tile_checksum(&baseline, 0),
        kde_query_tile_checksum(&candidate, 0)
    );
    Ok(())
}

fn run_kde_query_tile_frontier() -> Result<bool, String> {
    use fsci_stats::GAUSSIAN_KDE_ND_QUERY_TILE_DISABLE;

    let fixture = make_kde_query_tile_fixture()?;
    prove_kde_query_tile_contract(&fixture)?;
    let repetitions = calibrate_kde_query_tile_repetitions(&fixture)?;
    for round in 0usize..4 {
        if round.is_multiple_of(2) {
            black_box(time_kde_query_tile_arm(
                &fixture,
                KdeQueryTileArm::Baseline,
                repetitions,
            ));
            black_box(time_kde_query_tile_arm(
                &fixture,
                KdeQueryTileArm::Candidate,
                repetitions,
            ));
        } else {
            black_box(time_kde_query_tile_arm(
                &fixture,
                KdeQueryTileArm::Candidate,
                repetitions,
            ));
            black_box(time_kde_query_tile_arm(
                &fixture,
                KdeQueryTileArm::Baseline,
                repetitions,
            ));
        }
    }

    println!(
        "frontier_fixture name=kde_nd_four_query_tile dimensions=3 data_points=2000 \
         queries=1024 repetitions={repetitions} min_sample_ms={FRONTIER_MIN_SAMPLE_MS}"
    );
    let null = paired_kde_query_tile(
        &fixture,
        KdeQueryTileArm::Baseline,
        KdeQueryTileArm::Baseline,
        repetitions,
        0x243f_6a88_85a3_08d3,
    );
    let candidate = paired_kde_query_tile(
        &fixture,
        KdeQueryTileArm::Baseline,
        KdeQueryTileArm::Candidate,
        repetitions,
        0x1319_8a2e_0370_7344,
    );
    GAUSSIAN_KDE_ND_QUERY_TILE_DISABLE.store(true, Ordering::Relaxed);
    print_frontier_paired_stats("aa_baseline_baseline", &null);
    print_frontier_paired_stats("ab_baseline_four_query_tile", &candidate);

    let null_half_width = (1.0 - null.ratio_ci_low)
        .abs()
        .max((null.ratio_ci_high - 1.0).abs());
    let decision_floor = (1.0 + 2.0 * null_half_width).max(1.01);
    let keep = candidate.ratio_ci_low > decision_floor;
    println!(
        "frontier_gate decision={} decision_basis=bootstrap_median_ci_vs_2x_aa \
         bootstrap_resamples={FRONTIER_BOOTSTRAP_RESAMPLES} aa_ci95=[{:.9},{:.9}] \
         aa_half_width={null_half_width:.9} decision_floor={decision_floor:.9} \
         candidate_ratio_median={:.9} candidate_ratio_median_ci95=[{:.9},{:.9}] \
         cv_used_for_decision=false",
        if keep { "KEEP" } else { "REJECT" },
        null.ratio_ci_low,
        null.ratio_ci_high,
        candidate.ratio_median,
        candidate.ratio_ci_low,
        candidate.ratio_ci_high
    );
    Ok(keep)
}

fn report_bench_elf_sha256() -> Result<(), String> {
    let identity = (|| {
        let executable = std::env::current_exe()?;
        let bytes = std::fs::read(executable)?;
        Ok::<_, io::Error>(format!("{:x}", Sha256::digest(bytes)))
    })();
    match &identity {
        Ok(hash) => println!("bench_elf_sha256={hash}"),
        Err(_) => println!("bench_elf_sha256=unavailable"),
    }
    io::stdout()
        .flush()
        .map_err(|error| format!("failed to flush benchmark identity: {error}"))?;
    identity
        .map(|_| ())
        .map_err(|error| format!("failed to hash benchmark executable: {error}"))
}

/// Same-binary A/B for the N-D KDE SIMD-exp path (batches the always-≤0 kernel exponent
/// through `kde_simd_exp_nonpos`, the 8-lane exp the 1-D KDE already uses). The scalar and
/// SIMD arms agree to ~1e-13 (polynomial exp + lane-group summation, well inside the KDE
/// tolerance); the A/B measures the exp-batching win at several dimensionalities.
fn bench_kde_nd_simd_ab(c: &mut Criterion) {
    use fsci_stats::{GAUSSIAN_KDE_ND_SIMD_EXP_DISABLE, GaussianKdeNd};
    use std::sync::atomic::Ordering;
    let mut group = c.benchmark_group("gaussian_kde_nd_simd_ab");
    let n = 2000usize;
    let m = 4000usize;
    for &d in &[2usize, 3, 6] {
        let data: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let t = i as f64;
                (0..d)
                    .map(|j| (t * (0.013 + 0.004 * j as f64) + j as f64).sin() * (1.0 + j as f64))
                    .collect()
            })
            .collect();
        let kde = GaussianKdeNd::new(&data).expect("kde nd");
        let q: Vec<Vec<f64>> = (0..m)
            .map(|i| {
                let t = i as f64;
                (0..d)
                    .map(|j| (t * (0.019 + 0.003 * j as f64)).cos() * (1.0 + j as f64))
                    .collect()
            })
            .collect();

        // Tolerance check: SIMD-exp arm vs scalar arm (NOT byte-identical — polynomial exp).
        GAUSSIAN_KDE_ND_SIMD_EXP_DISABLE.store(false, Ordering::Relaxed);
        let simd = kde.evaluate_many(&q);
        GAUSSIAN_KDE_ND_SIMD_EXP_DISABLE.store(true, Ordering::Relaxed);
        let scalar = kde.evaluate_many(&q);
        let maxrel = simd
            .iter()
            .zip(&scalar)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1e-300))
            .fold(0.0f64, f64::max);
        assert!(
            maxrel < 1e-11,
            "N-D KDE SIMD-exp vs scalar max reldiff {maxrel:e} exceeds 1e-11 (d={d})"
        );

        group.bench_function(BenchmarkId::new("current_simd", d), |b| {
            b.iter(|| {
                GAUSSIAN_KDE_ND_SIMD_EXP_DISABLE.store(false, Ordering::Relaxed);
                black_box(kde.evaluate_many(black_box(&q)))
            })
        });
        group.bench_function(BenchmarkId::new("orig_scalar", d), |b| {
            b.iter(|| {
                GAUSSIAN_KDE_ND_SIMD_EXP_DISABLE.store(true, Ordering::Relaxed);
                black_box(kde.evaluate_many(black_box(&q)))
            })
        });
    }
    GAUSSIAN_KDE_ND_SIMD_EXP_DISABLE.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_kde(c: &mut Criterion) {
    use fsci_stats::{
        GAUSSIAN_KDE_SIMD_EXP_DISABLE, GAUSSIAN_KDE_TAIL_WINDOW_DISABLE, GaussianKde,
    };
    use std::sync::atomic::Ordering;
    let mut group = c.benchmark_group("gaussian_kde");
    for &n in &[1000usize, 5000] {
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                (t * 0.017).sin() * 3.0 + (t * 0.0031).cos()
            })
            .collect();
        let kde = GaussianKde::new(&data);
        let pts: Vec<f64> = (0..n).map(|i| -5.0 + i as f64 * 10.0 / n as f64).collect();
        group.bench_function(BenchmarkId::new("evaluate_many", n), |b| {
            GAUSSIAN_KDE_TAIL_WINDOW_DISABLE.store(false, Ordering::Relaxed);
            GAUSSIAN_KDE_SIMD_EXP_DISABLE.store(false, Ordering::Relaxed);
            b.iter(|| kde.evaluate_many(black_box(&pts)))
        });
        group.bench_function(BenchmarkId::new("evaluate_many_tail_disabled", n), |b| {
            b.iter(|| {
                GAUSSIAN_KDE_TAIL_WINDOW_DISABLE.store(true, Ordering::Relaxed);
                GAUSSIAN_KDE_SIMD_EXP_DISABLE.store(false, Ordering::Relaxed);
                let values = kde.evaluate_many(black_box(&pts));
                GAUSSIAN_KDE_TAIL_WINDOW_DISABLE.store(false, Ordering::Relaxed);
                values
            })
        });
        group.bench_function(BenchmarkId::new("evaluate_many_legacy_original", n), |b| {
            b.iter(|| {
                GAUSSIAN_KDE_TAIL_WINDOW_DISABLE.store(true, Ordering::Relaxed);
                GAUSSIAN_KDE_SIMD_EXP_DISABLE.store(true, Ordering::Relaxed);
                let values = kde.evaluate_many(black_box(&pts));
                GAUSSIAN_KDE_TAIL_WINDOW_DISABLE.store(false, Ordering::Relaxed);
                GAUSSIAN_KDE_SIMD_EXP_DISABLE.store(false, Ordering::Relaxed);
                values
            })
        });
    }
    GAUSSIAN_KDE_TAIL_WINDOW_DISABLE.store(false, Ordering::Relaxed);
    GAUSSIAN_KDE_SIMD_EXP_DISABLE.store(false, Ordering::Relaxed);
    group.finish();
}

/// Multiscale graph correlation — mgc_map is O(n²) (prefix-sum) vs the naive O(n⁴),
/// and the `reps` permutation scoring is parallelized. Head-to-head vs
/// scipy.stats.multiscale_graphcorr (docs/perf_oracle_mgc.py).
fn bench_mgc(c: &mut Criterion) {
    use fsci_stats::multiscale_graphcorr;
    let n = 80usize;
    let x: Vec<Vec<f64>> = (0..n).map(|i| vec![(i as f64 * 0.1).sin()]).collect();
    let y: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![(i as f64 * 0.1).sin() + (i as f64 * 0.37).cos() * 0.3])
        .collect();
    let mut group = c.benchmark_group("mgc");
    group.bench_function("n80_reps100", |b| {
        b.iter(|| multiscale_graphcorr(&x, &y, 100, Some(0)))
    });
    group.finish();
}

/// Theil-Sen / Siegel robust regression slopes — head-to-head vs
/// scipy.stats.theilslopes / siegelslopes (both O(n^2) in scipy's C).
fn bench_robust_slopes(c: &mut Criterion) {
    let mut group = c.benchmark_group("robust_slopes");
    for &n in &[2000usize, 4000] {
        let x: Vec<f64> = (0..n)
            .map(|i| i as f64 * 0.001 + (i % 7) as f64 * 1e-4)
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 2.0 * x[i] + ((i * 2654435761usize) % 1000) as f64 * 1e-3)
            .collect();
        group.bench_function(BenchmarkId::new("theilslopes", n), |b| {
            b.iter(|| theilslopes(black_box(&x), black_box(&y), 0.95))
        });
        group.bench_function(BenchmarkId::new("siegelslopes", n), |b| {
            b.iter(|| siegelslopes(black_box(&x), black_box(&y)))
        });
    }
    group.finish();
}

fn bench_distribution_distances(c: &mut Criterion) {
    use criterion::BenchmarkId;
    let mut group = c.benchmark_group("distribution_distances");
    for &n in &[50_000usize, 200_000] {
        let u: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.0007).sin() * 2.0 + (i % 31) as f64 * 0.03)
            .collect();
        let v: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.0011).cos() * 2.0 + 0.3 + (i % 17) as f64 * 0.05)
            .collect();
        group.bench_function(BenchmarkId::new("wasserstein", n), |b| {
            b.iter(|| wasserstein_distance(black_box(&u), black_box(&v)))
        });
        group.bench_function(BenchmarkId::new("energy", n), |b| {
            b.iter(|| energy_distance(black_box(&u), black_box(&v)))
        });
    }
    group.finish();
}

fn bench_binned_statistic_1d(c: &mut Criterion) {
    use criterion::BenchmarkId;
    let mut group = c.benchmark_group("binned_statistic_1d");
    let n = 200_000usize;
    let xs: Vec<f64> = (0..n)
        .map(|i| ((i * 2654435761usize) % 100000) as f64 / 100000.0)
        .collect();
    let vs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.001).sin()).collect();
    for &bins in &[1000usize, 5000] {
        group.bench_function(BenchmarkId::new("mean", bins), |b| {
            b.iter(|| binned_statistic(black_box(&xs), black_box(&vs), bins, "mean"))
        });
    }
    group.finish();
}

fn bench_binned_statistic_dd(c: &mut Criterion) {
    use criterion::BenchmarkId;
    let mut group = c.benchmark_group("binned_statistic_dd");
    let n = 200_000usize;
    let sample: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            vec![
                ((i * 2654435761usize) % 100000) as f64 / 100000.0,
                ((i * 40503usize + 7) % 100000) as f64 / 100000.0,
                ((i * 92821usize + 3) % 100000) as f64 / 100000.0,
            ]
        })
        .collect();
    let vs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.001).sin()).collect();
    for &bins in &[20usize, 30] {
        for (label, disable_parallel) in [
            ("current_d3_mean", false),
            ("legacy_original_d3_mean", true),
        ] {
            group.bench_function(BenchmarkId::new(label, bins), |b| {
                b.iter(|| {
                    BINNED_STATISTIC_DD_3D_PARALLEL_DISABLE
                        .store(disable_parallel, std::sync::atomic::Ordering::Relaxed);
                    binned_statistic_dd(black_box(&sample), black_box(&vs), bins, "mean")
                })
            });
        }
        BINNED_STATISTIC_DD_3D_PARALLEL_DISABLE.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    group.finish();
}

fn bench_binned_statistic_2d(c: &mut Criterion) {
    use criterion::BenchmarkId;
    let mut group = c.benchmark_group("binned_statistic_2d");
    let n = 200_000usize;
    let xs: Vec<f64> = (0..n)
        .map(|i| ((i * 2654435761usize) % 100000) as f64 / 100000.0)
        .collect();
    let ys: Vec<f64> = (0..n)
        .map(|i| ((i * 40503usize + 7) % 100000) as f64 / 100000.0)
        .collect();
    let vs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.001).sin()).collect();
    for stat in &["mean", "sum", "count"] {
        group.bench_function(BenchmarkId::new(*stat, n), |b| {
            b.iter(|| binned_statistic_2d(black_box(&xs), black_box(&ys), black_box(&vs), 50, stat))
        });
    }
    group.finish();
}

fn bench_skellam_cdf(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use fsci_stats::{DiscreteDistribution, Skellam};
    let mut group = c.benchmark_group("skellam_cdf");
    // cdf/sf cost scales with μ: the prior Bessel-`ive` window sum spanned ~24σ+80
    // points, so large μ exercised hundreds of Bessel evals per call. Evaluate a
    // batch of k across the central body for each μ.
    for &(mu1, mu2) in &[(5.0_f64, 3.0_f64), (100.0, 100.0), (1000.0, 1000.0)] {
        let sk = Skellam::new(mu1, mu2);
        let mean = mu1 - mu2;
        let std = (mu1 + mu2).sqrt();
        let ks: Vec<u64> = (0..256)
            .map(|i| (mean + (i as f64 - 128.0) / 256.0 * 6.0 * std).max(0.0) as u64)
            .collect();
        group.bench_function(BenchmarkId::new("cdf", format!("mu{mu1}_{mu2}")), |b| {
            b.iter(|| ks.iter().map(|&k| sk.cdf(black_box(k))).sum::<f64>())
        });
        group.bench_function(BenchmarkId::new("sf", format!("mu{mu1}_{mu2}")), |b| {
            b.iter(|| ks.iter().map(|&k| sk.sf(black_box(k))).sum::<f64>())
        });
    }
    group.finish();
}

fn bench_betabinom_cdf(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use fsci_stats::{BetaBinomial, DiscreteDistribution};
    let mut group = c.benchmark_group("betabinom_cdf");
    // Default cdf sums pmf(0..=k) at ~6 ln_gamma/pmf — O(k) lgamma; the pmf-ratio
    // recurrence makes each term one multiply. Cost scales with n.
    for &n in &[50_u64, 200, 1000] {
        let d = BetaBinomial::new(n, 2.0, 3.0);
        let ks: Vec<u64> = (0..=n).collect();
        group.bench_function(BenchmarkId::new("cdf", n), |b| {
            b.iter(|| ks.iter().map(|&k| d.cdf(black_box(k))).sum::<f64>())
        });
    }
    group.finish();
}

fn bench_discrete_moments(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use fsci_stats::{BetaBinomial, DiscreteDistribution, NegHypergeometric};
    let mut group = c.benchmark_group("discrete_moments");
    // skewness/kurtosis summed central moments over the support at ~6 ln_gamma/pmf;
    // the pmf-ratio recurrence (mode-anchored) makes each term pure mults. Scales w/ n.
    for &n in &[200_u64, 1000] {
        let d = BetaBinomial::new(n, 2.0, 3.0);
        group.bench_function(BenchmarkId::new("betabinom_kurt", n), |b| {
            b.iter(|| black_box(d.kurtosis()))
        });
    }
    for &n in &[200_u64, 400] {
        let d = NegHypergeometric::new(2 * n + 100, n, n + 50);
        group.bench_function(BenchmarkId::new("neghypergeom_kurt", n), |b| {
            b.iter(|| black_box(d.kurtosis()))
        });
    }
    group.finish();
}

fn bench_discrete_entropy(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use fsci_stats::{
        BetaBinomial, BetaNegativeBinomial, Binomial, Boltzmann, DiscreteDistribution,
        NegHypergeometric, Poisson,
    };
    let mut group = c.benchmark_group("discrete_entropy");
    // entropy summed −Σ pmf·ln(pmf) over the support at ~6 ln_gamma/pmf; the
    // pmf-ratio recurrence (mode-anchored) makes each term one ln. Scales with n.
    for &n in &[200_u64, 1000] {
        let d = BetaBinomial::new(n, 2.0, 3.0);
        group.bench_function(BenchmarkId::new("betabinom", n), |b| {
            b.iter(|| black_box(d.entropy()))
        });
    }
    let bnb = BetaNegativeBinomial::new(10, 3.0, 3.0);
    group.bench_function(BenchmarkId::new("betanbinom", 10_u64), |b| {
        b.iter(|| black_box(bnb.entropy()))
    });
    for &n in &[1000_u64, 10000] {
        let d = Binomial::new(n, 0.5);
        group.bench_function(BenchmarkId::new("binomial", n), |b| {
            b.iter(|| black_box(d.entropy()))
        });
    }
    for &mu in &[100.0_f64, 900.0] {
        let d = Poisson::new(mu);
        group.bench_function(BenchmarkId::new("poisson", mu as u64), |b| {
            b.iter(|| black_box(d.entropy()))
        });
    }
    for &n in &[10_000_u32, 100_000] {
        let d = Boltzmann::new(0.2, n);
        group.bench_function(BenchmarkId::new("boltzmann", n), |b| {
            b.iter(|| black_box(d.entropy()))
        });
    }
    for &n in &[200_u64, 400] {
        let d = NegHypergeometric::new(2 * n + 100, n, n + 50);
        group.bench_function(BenchmarkId::new("neghypergeom", n), |b| {
            b.iter(|| black_box(d.entropy()))
        });
    }
    group.finish();
}

fn bench_neghypergeom_cdf(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use fsci_stats::{DiscreteDistribution, NegHypergeometric};
    let mut group = c.benchmark_group("neghypergeom_cdf");
    // cdf summed pmf(0..=k) at 3 ln_comb (~6 ln_gamma)/pmf; the pmf-ratio recurrence
    // (mode-anchored) makes each term one multiply. Cost scales with n (support [0,n]).
    for &n in &[50_u64, 200, 400] {
        let d = NegHypergeometric::new(2 * n + 100, n, n + 50);
        let ks: Vec<u64> = (0..=n).collect();
        group.bench_function(BenchmarkId::new("cdf", n), |b| {
            b.iter(|| ks.iter().map(|&k| d.cdf(black_box(k))).sum::<f64>())
        });
    }
    group.finish();
}

fn bench_betanbinom_cdf(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use fsci_stats::{BETANBINOM_CDF_MANY_DISABLE, BetaNegativeBinomial};
    use std::sync::atomic::Ordering;
    let mut group = c.benchmark_group("betanbinom_cdf");
    // Default cdf sums pmf(0..=k) at ~6 ln_gamma/pmf; the pmf-ratio recurrence
    // (mode=0, anchor at pmf(0)) makes each term one multiply. The batch path
    // builds the prefix once; the legacy row maps scalar cdf for every query.
    let d = BetaNegativeBinomial::new(10, 3.0, 3.0);
    for &kmax in &[100_u64, 500, 2000] {
        let ks: Vec<u64> = (0..=kmax).collect();
        group.bench_function(BenchmarkId::new("current_cdf_many", kmax), |b| {
            b.iter(|| {
                BETANBINOM_CDF_MANY_DISABLE.store(false, Ordering::Relaxed);
                d.cdf_many(black_box(&ks)).iter().sum::<f64>()
            })
        });
        group.bench_function(BenchmarkId::new("legacy_original_scalar_cdf", kmax), |b| {
            b.iter(|| {
                BETANBINOM_CDF_MANY_DISABLE.store(true, Ordering::Relaxed);
                let total = d.cdf_many(black_box(&ks)).iter().sum::<f64>();
                BETANBINOM_CDF_MANY_DISABLE.store(false, Ordering::Relaxed);
                total
            })
        });
    }
    BETANBINOM_CDF_MANY_DISABLE.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_zipfian_cdf(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use fsci_stats::{DiscreteDistribution, Zipfian};
    let mut group = c.benchmark_group("zipfian_cdf");
    // Old cdf summed j^{-a} over 1..=k (plus an O(n) z() sum) → O(n+k). The Hurwitz-zeta
    // generalized-harmonic closed form ζ(a)−ζ(a,m+1) makes it O(1) (matches scipy, which
    // is O(1)). Cost scales with n+k for the old path; flat for the closed form.
    for &n in &[1_000_u32, 50_000, 500_000] {
        let d = Zipfian::new(1.3, n);
        let ks: Vec<u64> = (1..=20).map(|i| (n as u64 * i) / 20).collect();
        group.bench_function(BenchmarkId::new("cdf", n), |b| {
            b.iter(|| ks.iter().map(|&k| d.cdf(black_box(k))).sum::<f64>())
        });
    }
    group.finish();
}

fn bench_mad_reuse_ab(c: &mut Criterion) {
    use std::sync::atomic::Ordering;
    let mut group = c.benchmark_group("median_abs_deviation_reuse_ab");
    group.sample_size(10);
    for &n in &[100_000usize, 1_000_000usize] {
        let data = deterministic_data(n);
        group.bench_with_input(BenchmarkId::new("current_reuse", n), &data, |b, d| {
            b.iter(|| {
                MAD_REUSE_DISABLE.store(false, Ordering::Relaxed);
                black_box(median_abs_deviation(black_box(d), black_box(1.4826)))
            });
        });
        group.bench_with_input(BenchmarkId::new("orig_three_alloc", n), &data, |b, d| {
            b.iter(|| {
                MAD_REUSE_DISABLE.store(true, Ordering::Relaxed);
                black_box(median_abs_deviation(black_box(d), black_box(1.4826)))
            });
        });
    }
    MAD_REUSE_DISABLE.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_biweight_midcorrelation_ab(c: &mut Criterion) {
    use std::sync::atomic::Ordering;
    let mut group = c.benchmark_group("biweight_midcorrelation_hoist_ab");
    group.sample_size(10);
    for &n in &[200_000usize, 2_000_000usize] {
        let x = deterministic_data(n);
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| v * 0.8 + (i as f64 * 0.013).sin())
            .collect();
        let data = (x, y);
        group.bench_with_input(BenchmarkId::new("current_hoist", n), &data, |b, (x, y)| {
            b.iter(|| {
                BIWEIGHT_MAD_HOIST_DISABLE.store(false, Ordering::Relaxed);
                black_box(biweight_midcorrelation(black_box(x), black_box(y), 9.0))
            });
        });
        group.bench_with_input(
            BenchmarkId::new("orig_double_median", n),
            &data,
            |b, (x, y)| {
                b.iter(|| {
                    BIWEIGHT_MAD_HOIST_DISABLE.store(true, Ordering::Relaxed);
                    black_box(biweight_midcorrelation(black_box(x), black_box(y), 9.0))
                });
            },
        );
    }
    BIWEIGHT_MAD_HOIST_DISABLE.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_gstd_par_reductions_ab(c: &mut Criterion) {
    use std::sync::atomic::Ordering;
    let mut group = c.benchmark_group("gstd_par_reductions_ab");
    group.sample_size(10);
    // The ln map stays parallel in both arms (GSTD_FORCE_SERIAL default false); toggle the reduction
    // gates so the A/B isolates parallelizing the mean_log + var_log reductions (the serial bottleneck).
    let data: Vec<f64> = deterministic_data(16_000_000)
        .iter()
        .map(|&v| v.abs() + 1.0) // gstd needs strictly-positive input
        .collect();
    group.bench_function("current_parallel", |bn| {
        bn.iter(|| {
            PAR_SUM_FORCE_SERIAL.store(false, Ordering::Relaxed);
            MOMENT_PAR_FORCE_SERIAL.store(false, Ordering::Relaxed);
            black_box(gstd(black_box(&data)))
        });
    });
    group.bench_function("orig_serial_reductions", |bn| {
        bn.iter(|| {
            PAR_SUM_FORCE_SERIAL.store(true, Ordering::Relaxed);
            MOMENT_PAR_FORCE_SERIAL.store(true, Ordering::Relaxed);
            black_box(gstd(black_box(&data)))
        });
    });
    PAR_SUM_FORCE_SERIAL.store(false, Ordering::Relaxed);
    MOMENT_PAR_FORCE_SERIAL.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_pooled_variance_par_reductions_ab(c: &mut Criterion) {
    use std::sync::atomic::Ordering;
    let mut group = c.benchmark_group("pooled_variance_par_reductions_ab");
    group.sample_size(10);
    // Two large groups (each above the 1<<22 gate) so the per-group mean+SS parallelize.
    let g1 = deterministic_data(8_000_000);
    let g2: Vec<f64> = g1.iter().map(|&v| v * 0.9 + 0.5).collect();
    let groups: Vec<&[f64]> = vec![g1.as_slice(), g2.as_slice()];
    group.bench_function("current_parallel", |bn| {
        bn.iter(|| {
            PAR_SUM_FORCE_SERIAL.store(false, Ordering::Relaxed);
            MOMENT_PAR_FORCE_SERIAL.store(false, Ordering::Relaxed);
            black_box(pooled_variance(black_box(&groups)))
        });
    });
    group.bench_function("orig_serial", |bn| {
        bn.iter(|| {
            PAR_SUM_FORCE_SERIAL.store(true, Ordering::Relaxed);
            MOMENT_PAR_FORCE_SERIAL.store(true, Ordering::Relaxed);
            black_box(pooled_variance(black_box(&groups)))
        });
    });
    PAR_SUM_FORCE_SERIAL.store(false, Ordering::Relaxed);
    MOMENT_PAR_FORCE_SERIAL.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_ttest_rel_par_reductions_ab(c: &mut Criterion) {
    use std::sync::atomic::Ordering;
    let mut group = c.benchmark_group("ttest_rel_par_reductions_ab");
    group.sample_size(10);
    // The diffs mean and Σ(d-d̄)² were both serial; toggle both reduction gates for the A/B.
    let a = deterministic_data(16_000_000);
    let b: Vec<f64> = a.iter().map(|&v| v * 0.9 + 0.5).collect();
    group.bench_function("current_parallel", |bn| {
        bn.iter(|| {
            PAR_SUM_FORCE_SERIAL.store(false, Ordering::Relaxed);
            MOMENT_PAR_FORCE_SERIAL.store(false, Ordering::Relaxed);
            black_box(ttest_rel(black_box(&a), black_box(&b), None).unwrap())
        });
    });
    group.bench_function("orig_serial", |bn| {
        bn.iter(|| {
            PAR_SUM_FORCE_SERIAL.store(true, Ordering::Relaxed);
            MOMENT_PAR_FORCE_SERIAL.store(true, Ordering::Relaxed);
            black_box(ttest_rel(black_box(&a), black_box(&b), None).unwrap())
        });
    });
    PAR_SUM_FORCE_SERIAL.store(false, Ordering::Relaxed);
    MOMENT_PAR_FORCE_SERIAL.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_cohens_d_par_reductions_ab(c: &mut Criterion) {
    use std::sync::atomic::Ordering;
    let mut group = c.benchmark_group("cohens_d_par_reductions_ab");
    group.sample_size(10);
    // Both means and both group SS were serial; toggle both reduction gates for the A/B.
    let a = deterministic_data(16_000_000);
    let b: Vec<f64> = a.iter().map(|&v| v * 0.9 + 0.5).collect();
    group.bench_function("current_parallel", |bn| {
        bn.iter(|| {
            PAR_SUM_FORCE_SERIAL.store(false, Ordering::Relaxed);
            MOMENT_PAR_FORCE_SERIAL.store(false, Ordering::Relaxed);
            black_box(cohens_d(black_box(&a), black_box(&b)))
        });
    });
    group.bench_function("orig_serial", |bn| {
        bn.iter(|| {
            PAR_SUM_FORCE_SERIAL.store(true, Ordering::Relaxed);
            MOMENT_PAR_FORCE_SERIAL.store(true, Ordering::Relaxed);
            black_box(cohens_d(black_box(&a), black_box(&b)))
        });
    });
    PAR_SUM_FORCE_SERIAL.store(false, Ordering::Relaxed);
    MOMENT_PAR_FORCE_SERIAL.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_ttest_ind_par_mean_ab(c: &mut Criterion) {
    use std::sync::atomic::Ordering;
    let mut group = c.benchmark_group("ttest_ind_par_mean_ab");
    group.sample_size(10);
    // Variances already parallel (sum_sq_dev); the two means were the serial straggler. Toggle
    // PAR_SUM_FORCE_SERIAL to swap the means between serial (orig) and parallel (current).
    let a = deterministic_data(16_000_000);
    let b: Vec<f64> = a.iter().map(|&v| v * 0.9 + 0.5).collect();
    group.bench_function("current_par_mean", |bn| {
        bn.iter(|| {
            PAR_SUM_FORCE_SERIAL.store(false, Ordering::Relaxed);
            black_box(ttest_ind(black_box(&a), black_box(&b)))
        });
    });
    group.bench_function("orig_serial_mean", |bn| {
        bn.iter(|| {
            PAR_SUM_FORCE_SERIAL.store(true, Ordering::Relaxed);
            black_box(ttest_ind(black_box(&a), black_box(&b)))
        });
    });
    PAR_SUM_FORCE_SERIAL.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_bayes_mvs_par_reductions_ab(c: &mut Criterion) {
    use std::sync::atomic::Ordering;
    let mut group = c.benchmark_group("bayes_mvs_par_reductions_ab");
    group.sample_size(10);
    // Mean and Σ(x-mean)² were both serial; toggle both reduction gates for the A/B.
    let data = deterministic_data(16_000_000);
    group.bench_function("current_parallel", |b| {
        b.iter(|| {
            PAR_SUM_FORCE_SERIAL.store(false, Ordering::Relaxed);
            MOMENT_PAR_FORCE_SERIAL.store(false, Ordering::Relaxed);
            black_box(bayes_mvs(black_box(&data), black_box(0.9)))
        });
    });
    group.bench_function("orig_serial", |b| {
        b.iter(|| {
            PAR_SUM_FORCE_SERIAL.store(true, Ordering::Relaxed);
            MOMENT_PAR_FORCE_SERIAL.store(true, Ordering::Relaxed);
            black_box(bayes_mvs(black_box(&data), black_box(0.9)))
        });
    });
    PAR_SUM_FORCE_SERIAL.store(false, Ordering::Relaxed);
    MOMENT_PAR_FORCE_SERIAL.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_ttest_1samp_par_reductions_ab(c: &mut Criterion) {
    use std::sync::atomic::Ordering;
    let mut group = c.benchmark_group("ttest_1samp_par_reductions_ab");
    group.sample_size(10);
    // Both the mean and the Σ(x-mean)² were serial; toggle both reduction gates for the A/B.
    let data = deterministic_data(16_000_000);
    group.bench_function("current_parallel", |b| {
        b.iter(|| {
            PAR_SUM_FORCE_SERIAL.store(false, Ordering::Relaxed);
            MOMENT_PAR_FORCE_SERIAL.store(false, Ordering::Relaxed);
            black_box(ttest_1samp(black_box(&data), black_box(0.1)))
        });
    });
    group.bench_function("orig_serial", |b| {
        b.iter(|| {
            PAR_SUM_FORCE_SERIAL.store(true, Ordering::Relaxed);
            MOMENT_PAR_FORCE_SERIAL.store(true, Ordering::Relaxed);
            black_box(ttest_1samp(black_box(&data), black_box(0.1)))
        });
    });
    PAR_SUM_FORCE_SERIAL.store(false, Ordering::Relaxed);
    MOMENT_PAR_FORCE_SERIAL.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_excess_kurtosis_par_mean_ab(c: &mut Criterion) {
    use std::sync::atomic::Ordering;
    let mut group = c.benchmark_group("excess_kurtosis_par_mean_ab");
    group.sample_size(10);
    // Above the 1<<22 gate the m2/m4 loop is already parallel; the mean was the serial straggler.
    // Toggling PAR_SUM_FORCE_SERIAL swaps the mean between serial (orig) and parallel (current).
    let data = deterministic_data(16_000_000);
    group.bench_function("current_par_mean", |b| {
        b.iter(|| {
            PAR_SUM_FORCE_SERIAL.store(false, Ordering::Relaxed);
            black_box(excess_kurtosis(black_box(&data)))
        });
    });
    group.bench_function("orig_serial_mean", |b| {
        b.iter(|| {
            PAR_SUM_FORCE_SERIAL.store(true, Ordering::Relaxed);
            black_box(excess_kurtosis(black_box(&data)))
        });
    });
    PAR_SUM_FORCE_SERIAL.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_mad_fn_reuse_ab(c: &mut Criterion) {
    use std::sync::atomic::Ordering;
    let mut group = c.benchmark_group("mad_fn_reuse_ab");
    group.sample_size(10);
    for &n in &[100_000usize, 1_000_000usize] {
        let data = deterministic_data(n);
        group.bench_with_input(BenchmarkId::new("current_reuse", n), &data, |b, d| {
            b.iter(|| {
                MAD_FN_REUSE_DISABLE.store(false, Ordering::Relaxed);
                black_box(mad(black_box(d), black_box(1.4826)))
            });
        });
        group.bench_with_input(BenchmarkId::new("orig_three_alloc", n), &data, |b, d| {
            b.iter(|| {
                MAD_FN_REUSE_DISABLE.store(true, Ordering::Relaxed);
                black_box(mad(black_box(d), black_box(1.4826)))
            });
        });
    }
    MAD_FN_REUSE_DISABLE.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_mad_zscore_hoist_ab(c: &mut Criterion) {
    use std::sync::atomic::Ordering;
    let mut group = c.benchmark_group("mad_zscore_hoist_ab");
    group.sample_size(10);
    for &n in &[100_000usize, 1_000_000usize] {
        let data = deterministic_data(n);
        group.bench_with_input(BenchmarkId::new("current_hoist", n), &data, |b, d| {
            b.iter(|| {
                MAD_ZSCORE_HOIST_DISABLE.store(false, Ordering::Relaxed);
                black_box(mad_zscore(black_box(d), true))
            });
        });
        group.bench_with_input(BenchmarkId::new("orig_double_median", n), &data, |b, d| {
            b.iter(|| {
                MAD_ZSCORE_HOIST_DISABLE.store(true, Ordering::Relaxed);
                black_box(mad_zscore(black_box(d), true))
            });
        });
    }
    MAD_ZSCORE_HOIST_DISABLE.store(false, Ordering::Relaxed);
    group.finish();
}

criterion_group!(
    benches,
    bench_stats_cross_presort_ab,
    bench_brunnermunzel_matrix_presort_ab,
    bench_weighted_mean_fused_ab,
    bench_mean_absolute_error_simd_ab,
    bench_mean_squared_error_simd_ab,
    bench_brier_score_simd_ab,
    bench_gstd_par_reductions_ab,
    bench_pooled_variance_par_reductions_ab,
    bench_ttest_rel_par_reductions_ab,
    bench_cohens_d_par_reductions_ab,
    bench_ttest_ind_par_mean_ab,
    bench_bayes_mvs_par_reductions_ab,
    bench_ttest_1samp_par_reductions_ab,
    bench_excess_kurtosis_par_mean_ab,
    bench_mad_fn_reuse_ab,
    bench_mad_zscore_hoist_ab,
    bench_biweight_midcorrelation_ab,
    bench_mad_reuse_ab,
    bench_zipfian_cdf,
    bench_discrete_moments,
    bench_discrete_entropy,
    bench_neghypergeom_cdf,
    bench_betanbinom_cdf,
    bench_betabinom_cdf,
    bench_skellam_cdf,
    bench_robust_slopes,
    bench_binned_statistic_2d,
    bench_distribution_distances,
    bench_binned_statistic_1d,
    bench_binned_statistic_dd,
    bench_mgc,
    bench_qmc_discrepancy,
    bench_qmc_sampling,
    bench_ordering_and_bins,
    bench_time_series,
    bench_rank_correlation,
    bench_somersd,
    bench_rand_index,
    bench_distribution_batch,
    bench_kde,
    bench_kde_nd,
    bench_kde_nd_simd_ab,
    bench_mvn_pdf,
    bench_mvt_pdf,
    bench_rank_tests
);

fn main() -> ExitCode {
    if let Err(error) = report_bench_elf_sha256() {
        eprintln!("{error}");
        return ExitCode::FAILURE;
    }

    if std::env::args().any(|argument| argument == "--frontier-kde-query-tile") {
        return match run_kde_query_tile_frontier() {
            Ok(_) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("{error}");
                ExitCode::FAILURE
            }
        };
    }

    benches();
    Criterion::default().configure_from_args().final_summary();
    ExitCode::SUCCESS
}
