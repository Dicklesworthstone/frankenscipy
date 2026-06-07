//! Metamorphic property tests for correctness verification.
//!
//! These tests encode mathematical invariants that must hold regardless of input,
//! allowing bug detection without exact oracles.

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    const PROPTEST_CASES: u32 = 256;
    const TOL: f64 = 1e-10;
    const LOOSE_TOL: f64 = 1e-6;

    // ═══════════════════════════════════════════════════════════════════════════
    // FFT METAMORPHIC RELATIONS
    // ═══════════════════════════════════════════════════════════════════════════

    mod fft_relations {
        use super::*;
        use fsci_fft::{FftOptions, fft, ifft, irfft, rfft};

        type Complex64 = (f64, f64);

        fn complex_norm(c: &Complex64) -> f64 {
            (c.0 * c.0 + c.1 * c.1).sqrt()
        }

        fn vec_max_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| complex_norm(&(x.0 - y.0, x.1 - y.1)))
                .fold(0.0, f64::max)
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

            /// MR-FFT-1: ifft(fft(x)) ≈ x (roundtrip identity)
            #[test]
            fn mr_fft_ifft_roundtrip(
                real_parts in proptest::collection::vec(-100.0f64..100.0, 2..=64),
                imag_parts in proptest::collection::vec(-100.0f64..100.0, 2..=64)
            ) {
                let n = real_parts.len().min(imag_parts.len());
                let input: Vec<Complex64> = real_parts.iter()
                    .zip(imag_parts.iter())
                    .take(n)
                    .map(|(&r, &i)| (r, i))
                    .collect();

                let opts = FftOptions::default();
                let forward = fft(&input, &opts).expect("fft failed");
                let roundtrip = ifft(&forward, &opts).expect("ifft failed");

                let max_diff = vec_max_diff(&input, &roundtrip);
                let max_input = input.iter().map(complex_norm).fold(0.0, f64::max);
                let rel_tol = if max_input > 1.0 { TOL * max_input } else { TOL };

                prop_assert!(
                    max_diff < rel_tol,
                    "FFT roundtrip failed: max_diff={}, rel_tol={}, n={}",
                    max_diff, rel_tol, n
                );
            }

            /// MR-FFT-2: rfft/irfft roundtrip for real signals
            #[test]
            fn mr_rfft_irfft_roundtrip(
                input in proptest::collection::vec(-100.0f64..100.0, 4..=64)
            ) {
                // Ensure even length for clean irfft
                let n = input.len() - (input.len() % 2);
                let input = &input[..n];

                let opts = FftOptions::default();
                let forward = rfft(input, &opts).expect("rfft failed");
                let roundtrip = irfft(&forward, Some(n), &opts).expect("irfft failed");

                let max_diff: f64 = input.iter()
                    .zip(roundtrip.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);

                let max_input = input.iter().map(|x| x.abs()).fold(0.0, f64::max);
                let rel_tol = if max_input > 1.0 { TOL * max_input } else { TOL };

                prop_assert!(
                    max_diff < rel_tol,
                    "RFFT roundtrip failed: max_diff={}, rel_tol={}, n={}",
                    max_diff, rel_tol, n
                );
            }

            /// MR-FFT-3: Parseval's theorem - energy preserved
            #[test]
            fn mr_fft_parseval(
                real_parts in proptest::collection::vec(-100.0f64..100.0, 2..=64),
                imag_parts in proptest::collection::vec(-100.0f64..100.0, 2..=64)
            ) {
                let n = real_parts.len().min(imag_parts.len());
                let input: Vec<Complex64> = real_parts.iter()
                    .zip(imag_parts.iter())
                    .take(n)
                    .map(|(&r, &i)| (r, i))
                    .collect();

                let opts = FftOptions::default();
                let spectrum = fft(&input, &opts).expect("fft failed");

                let time_energy: f64 = input.iter().map(|c| c.0*c.0 + c.1*c.1).sum();
                let freq_energy: f64 = spectrum.iter().map(|c| c.0*c.0 + c.1*c.1).sum::<f64>() / n as f64;

                let rel_diff = (time_energy - freq_energy).abs() / (time_energy.abs() + 1e-15);
                prop_assert!(
                    rel_diff < LOOSE_TOL,
                    "Parseval failed: time_energy={}, freq_energy={}, rel_diff={}",
                    time_energy, freq_energy, rel_diff
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // LINALG METAMORPHIC RELATIONS
    // ═══════════════════════════════════════════════════════════════════════════

    mod linalg_relations {
        use super::*;
        use fsci_linalg::{InvOptions, SolveOptions, inv, solve};

        fn make_diag_dominant(n: usize, seed: u64) -> Vec<Vec<f64>> {
            let mut a = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    let pseudo_rand = ((seed.wrapping_mul(i as u64 + 1).wrapping_add(j as u64))
                        % 1000) as f64
                        / 1000.0;
                    a[i][j] = if i == j {
                        (n as f64) * 2.0 + pseudo_rand
                    } else {
                        pseudo_rand - 0.5
                    };
                }
            }
            a
        }

        fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
            a.iter()
                .map(|row| row.iter().zip(x.iter()).map(|(aij, xj)| aij * xj).sum())
                .collect()
        }

        fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
            let n = a.len();
            let mut c = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    c[i][j] = (0..n).map(|k| a[i][k] * b[k][j]).sum();
                }
            }
            c
        }

        fn max_diff_vec(a: &[f64], b: &[f64]) -> f64 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max)
        }

        fn max_diff_mat(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
            a.iter()
                .zip(b.iter())
                .flat_map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(x, y)| (x - y).abs()))
                .fold(0.0, f64::max)
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

            /// MR-LINALG-1: solve(A,b) yields x such that A@x ≈ b
            #[test]
            fn mr_solve_residual(
                n in 2usize..=16,
                seed in 0u64..1000,
                b_vals in proptest::collection::vec(-10.0f64..10.0, 16..=16)
            ) {
                let a = make_diag_dominant(n, seed);
                let b: Vec<f64> = b_vals.into_iter().take(n).collect();

                let result = solve(&a, &b, SolveOptions::default()).expect("solve failed");
                let x = &result.x;
                let ax = matvec(&a, x);
                let max_diff = max_diff_vec(&ax, &b);

                let b_norm: f64 = b.iter().map(|v| v.abs()).fold(0.0, f64::max);
                let rel_tol = if b_norm > 1.0 { LOOSE_TOL * b_norm } else { LOOSE_TOL };

                prop_assert!(
                    max_diff < rel_tol,
                    "solve residual failed: ||Ax-b||_inf={}, rel_tol={}, n={}",
                    max_diff, rel_tol, n
                );
            }

            /// MR-LINALG-2: inv(A) @ A ≈ I
            #[test]
            fn mr_inv_identity(
                n in 2usize..=12,
                seed in 0u64..1000
            ) {
                let a = make_diag_dominant(n, seed);
                let inv_result = inv(&a, InvOptions::default()).expect("inv failed");
                let a_inv = &inv_result.inverse;
                let product = matmul(a_inv, &a);

                let identity: Vec<Vec<f64>> = (0..n)
                    .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
                    .collect();

                let max_diff = max_diff_mat(&product, &identity);
                prop_assert!(
                    max_diff < LOOSE_TOL,
                    "inv identity failed: ||A^-1 A - I||_inf={}, n={}",
                    max_diff, n
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // INTEGRATION METAMORPHIC RELATIONS (FTC)
    // ═══════════════════════════════════════════════════════════════════════════

    mod integrate_relations {
        use super::*;
        use fsci_integrate::{QuadOptions, quad};

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            /// MR-INTEGRATE-1: ∫[a,b] f'(x) dx = f(b) - f(a) (FTC)
            /// Using f(x) = x^n, f'(x) = n*x^(n-1)
            #[test]
            fn mr_ftc_polynomial(
                a in -5.0f64..0.0,
                b in 0.1f64..5.0,
                n in 2u32..=6
            ) {
                let derivative = move |x: f64| (n as f64) * x.powi(n as i32 - 1);
                let f_b = b.powi(n as i32);
                let f_a = a.powi(n as i32);
                let expected = f_b - f_a;

                let opts = QuadOptions::default();
                let result = quad(derivative, a, b, opts).expect("quad failed");

                let rel_diff = (result.integral - expected).abs() / (expected.abs() + 1e-10);
                prop_assert!(
                    rel_diff < LOOSE_TOL,
                    "FTC failed: integral={}, expected={}, n={}, [a,b]=[{},{}]",
                    result.integral, expected, n, a, b
                );
            }

            /// MR-INTEGRATE-2: ∫[a,b] sin(x) dx = -cos(b) + cos(a)
            #[test]
            fn mr_ftc_trig(
                a in -std::f64::consts::PI..0.0,
                b in 0.1f64..std::f64::consts::PI
            ) {
                let expected = -b.cos() + a.cos();

                let opts = QuadOptions::default();
                let result = quad(f64::sin, a, b, opts).expect("quad failed");

                let diff = (result.integral - expected).abs();
                prop_assert!(
                    diff < LOOSE_TOL,
                    "FTC trig failed: integral={}, expected={}, diff={}",
                    result.integral, expected, diff
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // INTERPOLATION METAMORPHIC RELATIONS
    // ═══════════════════════════════════════════════════════════════════════════

    mod interpolate_relations {
        use super::*;
        use fsci_interpolate::interp1d_linear;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

            /// MR-INTERP-1: Interpolation passes through sample points
            #[test]
            fn mr_interp_passthrough(
                y_vals in proptest::collection::vec(-100.0f64..100.0, 4..=20)
            ) {
                let n = y_vals.len();
                let x: Vec<f64> = (0..n).map(|i| i as f64).collect();

                let result = interp1d_linear(&x, &y_vals, &x).expect("interp1d failed");

                for (i, (&yi, &interp_val)) in y_vals.iter().zip(result.iter()).enumerate() {
                    let diff = (interp_val - yi).abs();
                    prop_assert!(
                        diff < TOL,
                        "Interp passthrough failed at point {}: interp={}, expected={}",
                        i, interp_val, yi
                    );
                }
            }

            /// MR-INTERP-2: Linear interpolation midpoint is average
            #[test]
            fn mr_interp_linear_midpoint(
                y0 in -100.0f64..100.0,
                y1 in -100.0f64..100.0
            ) {
                let x = vec![0.0, 1.0];
                let y = vec![y0, y1];
                let x_new = vec![0.5];

                let result = interp1d_linear(&x, &y, &x_new).expect("interp1d failed");
                let midpoint = result[0];
                let expected = (y0 + y1) / 2.0;
                let diff = (midpoint - expected).abs();

                prop_assert!(
                    diff < TOL,
                    "Linear midpoint failed: interp(0.5)={}, expected={}",
                    midpoint, expected
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // STATS METAMORPHIC RELATIONS
    // ═══════════════════════════════════════════════════════════════════════════

    mod stats_relations {
        use super::*;
        use fsci_stats::{kurtosis, nanmean, nanvar, skew};

        fn shuffle_with_seed(data: &[f64], seed: u64) -> Vec<f64> {
            let n = data.len();
            let mut result = data.to_vec();
            for i in 0..n {
                let j = ((seed.wrapping_mul(i as u64 + 1)) % n as u64) as usize;
                result.swap(i, j);
            }
            result
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

            /// MR-STATS-1: Mean is permutation-invariant
            #[test]
            fn mr_mean_permutation_invariant(
                data in proptest::collection::vec(-100.0f64..100.0, 3..=50),
                seed in 0u64..1000
            ) {
                let shuffled = shuffle_with_seed(&data, seed);

                let m1 = nanmean(&data);
                let m2 = nanmean(&shuffled);

                let diff = (m1 - m2).abs();
                prop_assert!(
                    diff < TOL,
                    "Mean permutation invariance failed: original={}, shuffled={}",
                    m1, m2
                );
            }

            /// MR-STATS-2: Variance is permutation-invariant
            #[test]
            fn mr_variance_permutation_invariant(
                data in proptest::collection::vec(-100.0f64..100.0, 3..=50),
                seed in 0u64..1000
            ) {
                let shuffled = shuffle_with_seed(&data, seed);

                let v1 = nanvar(&data);
                let v2 = nanvar(&shuffled);

                let rel_diff = (v1 - v2).abs() / (v1.abs() + 1e-10);
                prop_assert!(
                    rel_diff < TOL,
                    "Variance permutation invariance failed: original={}, shuffled={}",
                    v1, v2
                );
            }

            /// MR-STATS-3: Skewness is permutation-invariant
            #[test]
            fn mr_skewness_permutation_invariant(
                data in proptest::collection::vec(-100.0f64..100.0, 4..=50),
                seed in 0u64..1000
            ) {
                let shuffled = shuffle_with_seed(&data, seed);

                let s1 = skew(&data);
                let s2 = skew(&shuffled);

                let diff = (s1 - s2).abs();
                prop_assert!(
                    diff < TOL,
                    "Skewness permutation invariance failed: original={}, shuffled={}",
                    s1, s2
                );
            }

            /// MR-STATS-4: Kurtosis is permutation-invariant
            #[test]
            fn mr_kurtosis_permutation_invariant(
                data in proptest::collection::vec(-100.0f64..100.0, 5..=50),
                seed in 0u64..1000
            ) {
                let shuffled = shuffle_with_seed(&data, seed);

                let k1 = kurtosis(&data);
                let k2 = kurtosis(&shuffled);

                let diff = (k1 - k2).abs();
                prop_assert!(
                    diff < TOL,
                    "Kurtosis permutation invariance failed: original={}, shuffled={}",
                    k1, k2
                );
            }

            /// MR-STATS-5: Adding constant shifts mean by that constant
            #[test]
            fn mr_mean_shift(
                data in proptest::collection::vec(-100.0f64..100.0, 3..=50),
                shift in -50.0f64..50.0
            ) {
                let shifted: Vec<f64> = data.iter().map(|x| x + shift).collect();

                let m1 = nanmean(&data);
                let m2 = nanmean(&shifted);

                let diff = (m2 - m1 - shift).abs();
                prop_assert!(
                    diff < TOL,
                    "Mean shift failed: mean(x)={}, mean(x+c)={}, c={}",
                    m1, m2, shift
                );
            }

            /// MR-STATS-6: Scaling data scales variance by square
            #[test]
            fn mr_variance_scale(
                data in proptest::collection::vec(-10.0f64..10.0, 3..=50),
                scale in 0.5f64..3.0
            ) {
                let scaled: Vec<f64> = data.iter().map(|x| x * scale).collect();

                let v1 = nanvar(&data);
                let v2 = nanvar(&scaled);

                let expected = v1 * scale * scale;
                let rel_diff = (v2 - expected).abs() / (expected.abs() + 1e-10);
                prop_assert!(
                    rel_diff < LOOSE_TOL,
                    "Variance scale failed: var(x)={}, var(cx)={}, expected={}, c={}",
                    v1, v2, expected, scale
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SORTING/ORDER STATISTICS METAMORPHIC RELATIONS
    // ═══════════════════════════════════════════════════════════════════════════

    mod order_stats_relations {
        use super::*;
        use fsci_stats::{median, percentile};

        fn shuffle_with_seed(data: &[f64], seed: u64) -> Vec<f64> {
            let n = data.len();
            let mut result = data.to_vec();
            for i in 0..n {
                let j = ((seed.wrapping_mul(i as u64 + 1)) % n as u64) as usize;
                result.swap(i, j);
            }
            result
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

            /// MR-ORDER-1: Median is permutation-invariant
            #[test]
            fn mr_median_permutation_invariant(
                data in proptest::collection::vec(-100.0f64..100.0, 3..=50),
                seed in 0u64..1000
            ) {
                let shuffled = shuffle_with_seed(&data, seed);

                let med1 = median(&data);
                let med2 = median(&shuffled);

                let diff = (med1 - med2).abs();
                prop_assert!(
                    diff < TOL,
                    "Median permutation invariance failed: original={}, shuffled={}",
                    med1, med2
                );
            }

            /// MR-ORDER-2: Percentiles are permutation-invariant
            #[test]
            fn mr_percentile_permutation_invariant(
                data in proptest::collection::vec(-100.0f64..100.0, 5..=50),
                seed in 0u64..1000,
                q in 10.0f64..90.0
            ) {
                let shuffled = shuffle_with_seed(&data, seed);

                let p1 = percentile(&data, q);
                let p2 = percentile(&shuffled, q);

                let diff = (p1 - p2).abs();
                prop_assert!(
                    diff < TOL,
                    "Percentile({}) permutation invariance failed: original={}, shuffled={}",
                    q, p1, p2
                );
            }

            /// MR-ORDER-3: Percentile(50) ≈ median
            #[test]
            fn mr_percentile_50_equals_median(
                data in proptest::collection::vec(-100.0f64..100.0, 3..=50)
            ) {
                let med = median(&data);
                let p50 = percentile(&data, 50.0);

                let diff = (med - p50).abs();
                prop_assert!(
                    diff < TOL,
                    "Percentile(50) != median: median={}, percentile(50)={}",
                    med, p50
                );
            }

            /// MR-ORDER-4: Percentile monotonicity: q1 < q2 => percentile(q1) <= percentile(q2)
            #[test]
            fn mr_percentile_monotonic(
                data in proptest::collection::vec(-100.0f64..100.0, 5..=50),
                q1 in 10.0f64..50.0,
                q2 in 50.0f64..90.0
            ) {
                let p1 = percentile(&data, q1);
                let p2 = percentile(&data, q2);

                prop_assert!(
                    p1 <= p2 + TOL,
                    "Percentile monotonicity violated: percentile({})={} > percentile({})={}",
                    q1, p1, q2, p2
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SPECIAL FUNCTIONS METAMORPHIC RELATIONS
    // ═══════════════════════════════════════════════════════════════════════════

    mod special_relations {
        use super::*;
        use fsci_runtime::RuntimeMode;
        use fsci_special::{SpecialTensor, erf, erfc, gamma, gammaln};

        fn scalar(x: f64) -> SpecialTensor {
            SpecialTensor::RealScalar(x)
        }

        fn get_scalar(t: &SpecialTensor) -> f64 {
            match t {
                SpecialTensor::RealScalar(v) => *v,
                _ => panic!("expected scalar"),
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

            /// MR-SPECIAL-1: gamma(x+1) = x * gamma(x)
            #[test]
            fn mr_gamma_recurrence(x in 0.5f64..10.0) {
                let g_x = gamma(&scalar(x), RuntimeMode::Strict).expect("gamma(x) failed");
                let g_xp1 = gamma(&scalar(x + 1.0), RuntimeMode::Strict).expect("gamma(x+1) failed");

                let lhs = get_scalar(&g_xp1);
                let rhs = x * get_scalar(&g_x);

                let rel_diff = (lhs - rhs).abs() / (rhs.abs() + 1e-10);
                prop_assert!(
                    rel_diff < LOOSE_TOL,
                    "Gamma recurrence failed: gamma(x+1)={}, x*gamma(x)={}, x={}",
                    lhs, rhs, x
                );
            }

            /// MR-SPECIAL-2: gammaln(x) = ln(gamma(x)) for x > 0
            #[test]
            fn mr_gammaln_log_gamma(x in 0.5f64..20.0) {
                let g = gamma(&scalar(x), RuntimeMode::Strict).expect("gamma failed");
                let lg = gammaln(&scalar(x), RuntimeMode::Strict).expect("gammaln failed");

                let expected = get_scalar(&g).ln();
                let actual = get_scalar(&lg);

                let diff = (actual - expected).abs();
                prop_assert!(
                    diff < LOOSE_TOL,
                    "gammaln != ln(gamma): gammaln({})={}, ln(gamma({}))={}",
                    x, actual, x, expected
                );
            }

            /// MR-SPECIAL-3: erf(x) + erfc(x) = 1
            #[test]
            fn mr_erf_erfc_sum(x in -5.0f64..5.0) {
                let e = erf(&scalar(x), RuntimeMode::Strict).expect("erf failed");
                let ec = erfc(&scalar(x), RuntimeMode::Strict).expect("erfc failed");

                let sum = get_scalar(&e) + get_scalar(&ec);
                let diff = (sum - 1.0).abs();

                prop_assert!(
                    diff < TOL,
                    "erf(x) + erfc(x) != 1: sum={}, x={}",
                    sum, x
                );
            }

            /// MR-SPECIAL-4: erf(-x) = -erf(x) (odd function)
            #[test]
            fn mr_erf_odd(x in 0.1f64..5.0) {
                let e_pos = erf(&scalar(x), RuntimeMode::Strict).expect("erf(x) failed");
                let e_neg = erf(&scalar(-x), RuntimeMode::Strict).expect("erf(-x) failed");

                let lhs = get_scalar(&e_neg);
                let rhs = -get_scalar(&e_pos);

                let diff = (lhs - rhs).abs();
                prop_assert!(
                    diff < TOL,
                    "erf oddness failed: erf(-{})={}, -erf({})={}",
                    x, lhs, x, rhs
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // OPTIMIZE METAMORPHIC RELATIONS
    // ═══════════════════════════════════════════════════════════════════════════

    mod optimize_relations {
        use super::*;
        use fsci_opt::{
            MinimizeOptions, MinimizeScalarOptions, RootOptions, brentq, minimize, minimize_scalar,
        };

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

            /// MR-OPT-1: brentq finds a root, i.e. f(root) ≈ 0. The function
            /// x^3 + x - t is strictly increasing, so it has exactly one real
            /// root that the bracket [-100, 100] always straddles for |t|<=1e5.
            #[test]
            fn mr_brentq_root_is_zero(t in -1.0e5f64..1.0e5) {
                let f = |x: f64| x * x * x + x - t;
                let r = brentq(f, (-100.0, 100.0), RootOptions::default())
                    .expect("brentq should converge");
                prop_assert!(r.converged, "brentq did not converge for t={t}");
                prop_assert!(
                    f(r.root).abs() < 1e-6,
                    "brentq root not a zero: f({})={}, t={t}",
                    r.root, f(r.root)
                );
            }

            /// MR-OPT-2: minimizing the convex quadratic sum (x_i - c_i)^2
            /// recovers the center c (unique global minimum) with f(min) ≈ 0.
            #[test]
            fn mr_minimize_quadratic_recovers_center(
                c in proptest::collection::vec(-5.0f64..5.0, 2..=4)
            ) {
                let center = c.clone();
                let f = move |x: &[f64]| -> f64 {
                    x.iter().zip(&center).map(|(xi, ci)| (xi - ci) * (xi - ci)).sum()
                };
                let x0 = vec![0.0; c.len()];
                let res = minimize(f, &x0, MinimizeOptions::default())
                    .expect("minimize should succeed");
                let max_err = res.x.iter().zip(&c)
                    .map(|(xi, ci)| (xi - ci).abs())
                    .fold(0.0, f64::max);
                prop_assert!(
                    max_err < 1e-3,
                    "minimize did not recover center: x={:?}, c={:?}",
                    res.x, c
                );
                if let Some(fval) = res.fun {
                    prop_assert!(fval < 1e-6, "minimum value not ~0: {fval}");
                }
            }

            /// MR-OPT-3: minimize_scalar of (x - c)^2 recovers c.
            #[test]
            fn mr_minimize_scalar_recovers_center(c in -1.0e3f64..1.0e3) {
                let f = move |x: f64| (x - c) * (x - c);
                let res = minimize_scalar(f, MinimizeScalarOptions::default())
                    .expect("minimize_scalar should succeed");
                prop_assert!(
                    (res.x - c).abs() < 1e-4 * (c.abs() + 1.0),
                    "minimize_scalar did not recover c: x={}, c={c}",
                    res.x
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ODR METAMORPHIC RELATIONS
    // ═══════════════════════════════════════════════════════════════════════════

    mod odr_relations {
        use super::*;
        use fsci_odr::odr;

        proptest! {
            // ODR is an iterative solver; fewer cases keep wall-time reasonable.
            #![proptest_config(ProptestConfig::with_cases(64))]

            /// MR-ODR-1: noise-free linear data y = a*x + b has zero orthogonal
            /// distance to the line (a, b), so ODR must recover (a, b) exactly.
            #[test]
            fn mr_odr_recovers_exact_line(a in -5.0f64..5.0, b in -5.0f64..5.0) {
                let x: Vec<f64> = (0..12).map(|i| i as f64 * 0.5 - 2.0).collect();
                let y: Vec<f64> = x.iter().map(|xi| a * xi + b).collect();
                let fcn = |beta: &[f64], x: &[f64]| -> Vec<f64> {
                    x.iter().map(|xi| beta[0] * xi + beta[1]).collect()
                };
                let out = odr(fcn, vec![1.0, 0.0], y, x).expect("odr should run");
                prop_assert!(
                    (out.beta[0] - a).abs() < 1e-5 * (a.abs() + 1.0),
                    "ODR slope: got {}, want {a}", out.beta[0]
                );
                prop_assert!(
                    (out.beta[1] - b).abs() < 1e-5 * (b.abs() + 1.0),
                    "ODR intercept: got {}, want {b}", out.beta[1]
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // LINALG DECOMPOSITION METAMORPHIC RELATIONS (extends linalg_relations)
    // ═══════════════════════════════════════════════════════════════════════════

    mod linalg_decomp_relations {
        use super::*;
        use fsci_linalg::{DecompOptions, det, qr};
        use fsci_runtime::RuntimeMode;

        fn make_diag_dominant(n: usize, seed: u64) -> Vec<Vec<f64>> {
            let mut a = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    let r = ((seed.wrapping_mul(i as u64 + 1).wrapping_add(j as u64)) % 1000)
                        as f64
                        / 1000.0;
                    a[i][j] = if i == j { (n as f64) * 2.0 + r } else { r - 0.5 };
                }
            }
            a
        }

        fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
            let n = a.len();
            let mut c = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    c[i][j] = (0..n).map(|k| a[i][k] * b[k][j]).sum();
                }
            }
            c
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

            /// MR-LINALG-3: QR factorization reconstructs the matrix, Q@R ≈ A.
            #[test]
            fn mr_qr_reconstructs(n in 2usize..=10, seed in 0u64..1000) {
                let a = make_diag_dominant(n, seed);
                let res = qr(&a, DecompOptions::default()).expect("qr failed");
                let recon = matmul(&res.q, &res.r);
                let max_diff = a.iter().zip(&recon)
                    .flat_map(|(ra, rr)| ra.iter().zip(rr).map(|(x, y)| (x - y).abs()))
                    .fold(0.0, f64::max);
                prop_assert!(max_diff < LOOSE_TOL, "Q@R != A: max_diff={max_diff}, n={n}");
            }

            /// MR-LINALG-4: det is transpose-invariant, det(A^T) ≈ det(A).
            #[test]
            fn mr_det_transpose_invariant(n in 2usize..=8, seed in 0u64..1000) {
                let a = make_diag_dominant(n, seed);
                let at: Vec<Vec<f64>> = (0..n)
                    .map(|i| (0..n).map(|j| a[j][i]).collect())
                    .collect();
                let da = det(&a, RuntimeMode::Strict, true).expect("det(A)");
                let dat = det(&at, RuntimeMode::Strict, true).expect("det(A^T)");
                prop_assert!(
                    (da - dat).abs() < 1e-9 * (da.abs() + 1.0),
                    "det(A^T) != det(A): {da} vs {dat}, n={n}"
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SPECIAL BETA/GAMMA METAMORPHIC RELATIONS (extends special_relations)
    // ═══════════════════════════════════════════════════════════════════════════

    mod special_beta_relations {
        use super::*;
        use fsci_runtime::RuntimeMode;
        use fsci_special::{SpecialTensor, beta, gammaln};

        fn sc(x: f64) -> SpecialTensor {
            SpecialTensor::RealScalar(x)
        }
        fn val(t: &SpecialTensor) -> f64 {
            match t {
                SpecialTensor::RealScalar(v) => *v,
                _ => panic!("expected scalar"),
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

            /// MR-SPECIAL-4: B(a,b) = exp(lnΓ(a) + lnΓ(b) − lnΓ(a+b)).
            #[test]
            fn mr_beta_gamma_identity(a in 0.5f64..12.0, b in 0.5f64..12.0) {
                let beta_ab = val(&beta(&sc(a), &sc(b), RuntimeMode::Strict).expect("beta"));
                let lga = val(&gammaln(&sc(a), RuntimeMode::Strict).expect("lnG(a)"));
                let lgb = val(&gammaln(&sc(b), RuntimeMode::Strict).expect("lnG(b)"));
                let lgab = val(&gammaln(&sc(a + b), RuntimeMode::Strict).expect("lnG(a+b)"));
                let from_gamma = (lga + lgb - lgab).exp();
                prop_assert!(
                    (beta_ab - from_gamma).abs() < LOOSE_TOL * (from_gamma.abs() + 1.0),
                    "beta identity failed: B({a},{b})={beta_ab}, gamma-form={from_gamma}"
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // LINALG SVD/EIG METAMORPHIC RELATIONS (extends linalg_relations)
    // ═══════════════════════════════════════════════════════════════════════════

    mod linalg_svd_eig_relations {
        use super::*;
        use fsci_linalg::{DecompOptions, eig, svd};

        fn make_diag_dominant(n: usize, seed: u64) -> Vec<Vec<f64>> {
            let mut a = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    let r = ((seed.wrapping_mul(i as u64 + 1).wrapping_add(j as u64)) % 1000)
                        as f64
                        / 1000.0;
                    a[i][j] = if i == j { (n as f64) * 2.0 + r } else { r - 0.5 };
                }
            }
            a
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

            /// MR-LINALG-5: SVD reconstructs the matrix, U @ diag(s) @ V^T ≈ A.
            #[test]
            fn mr_svd_reconstructs(n in 2usize..=8, seed in 0u64..1000) {
                let a = make_diag_dominant(n, seed);
                let res = svd(&a, DecompOptions::default()).expect("svd failed");
                let mut max_diff = 0.0f64;
                for i in 0..n {
                    for j in 0..n {
                        let recon: f64 = (0..res.s.len())
                            .map(|k| res.u[i][k] * res.s[k] * res.vt[k][j])
                            .sum();
                        max_diff = max_diff.max((recon - a[i][j]).abs());
                    }
                }
                prop_assert!(
                    max_diff < LOOSE_TOL * (2.0 * n as f64 + 1.0),
                    "U diag(s) Vt != A: max_diff={max_diff}, n={n}"
                );
            }

            /// MR-LINALG-6: sum of eigenvalues equals the trace (layout-free;
            /// complex eigenvalues come in conjugate pairs so the real parts sum
            /// to the trace).
            #[test]
            fn mr_eig_trace_identity(n in 2usize..=8, seed in 0u64..1000) {
                let a = make_diag_dominant(n, seed);
                let trace: f64 = (0..n).map(|i| a[i][i]).sum();
                let res = eig(&a, DecompOptions::default()).expect("eig failed");
                let eig_sum: f64 = res.eigenvalues_re.iter().sum();
                prop_assert!(
                    (eig_sum - trace).abs() < LOOSE_TOL * (trace.abs() + 1.0),
                    "sum(eigenvalues) != trace: {eig_sum} vs {trace}, n={n}"
                );
            }
        }
    }
}
