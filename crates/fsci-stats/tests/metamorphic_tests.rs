//! Metamorphic tests for `fsci-stats`.
//!
//! Each test checks an oracle-free relation: distribution round-trips,
//! correlation symmetries, t-test sign invariance, etc.
//!
//! Run with: `cargo test -p fsci-stats --test metamorphic_tests`

use fsci_stats::{
    Bernoulli, BetaDist, Binomial, ChiSquared, ContinuousDistribution, DiscreteDistribution,
    Exponential, GammaDist, Geometric, Normal, Poisson, StudentT, Uniform, acf,
    bartlett as bartlett_var, cumfreq, diff, durbin_watson, ecdf, energy_distance, f_oneway, gmean,
    histogram, hmean, kruskal, ks_2samp, kurtosis, levene, linregress, mannkendall, mannwhitneyu,
    pacf, pearsonr, percentile, pmean, quantile, ranksums, relfreq, ridge_regression, runs_test,
    sigmaclip, skew, spearmanr, theil_sen, tmean, trim_mean, ttest_1samp, ttest_ind, ttest_rel,
    tukey_hsd, variation, wasserstein_distance, wilcoxon, zscore,
};

const ATOL: f64 = 1e-8;
const RTOL: f64 = 1e-7;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ATOL + RTOL * a.abs().max(b.abs()).max(1.0)
}

fn assert_close(a: f64, b: f64, msg: &str) {
    assert!(close(a, b), "{msg}: actual={a:.16e} expected={b:.16e}");
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — cdf-ppf round-trip: ppf(cdf(x)) ≈ x for x in the support.
// ─────────────────────────────────────────────────────────────────────

fn cdf_ppf_roundtrip<D: ContinuousDistribution>(d: &D, xs: &[f64], label: &str) {
    for &x in xs {
        let p = d.cdf(x);
        // Skip endpoints where ppf returns ±inf.
        if !(p > 1e-12 && p < 1.0 - 1e-12) {
            continue;
        }
        let x_back = d.ppf(p);
        // Use a slightly looser bound near the tails because ppf has
        // steep slope there.
        let tol = 1e-6 * (1.0 + x.abs());
        assert!(
            (x_back - x).abs() <= tol,
            "{label} ppf(cdf(x)) round-trip: x={x} cdf={p:.16e} ppf={x_back} diff={}",
            (x_back - x).abs()
        );
    }
}

#[test]
fn mr_normal_cdf_ppf_roundtrip() {
    let d = Normal {
        loc: 0.5,
        scale: 1.5,
    };
    let xs: Vec<f64> = (-30..=30).map(|i| 0.2 * i as f64).collect();
    cdf_ppf_roundtrip(&d, &xs, "Normal");
}

#[test]
fn mr_uniform_cdf_ppf_roundtrip() {
    let d = Uniform {
        loc: -2.0,
        scale: 5.0,
    };
    let xs: Vec<f64> = (1..=49).map(|i| -2.0 + 5.0 * i as f64 / 50.0).collect();
    cdf_ppf_roundtrip(&d, &xs, "Uniform");
}

#[test]
fn mr_exponential_cdf_ppf_roundtrip() {
    let d = Exponential::new(2.0);
    let xs: Vec<f64> = (1..=40).map(|i| 0.05 * i as f64).collect();
    cdf_ppf_roundtrip(&d, &xs, "Exponential");
}

#[test]
fn mr_chisquared_cdf_ppf_roundtrip() {
    let d = ChiSquared::new(5.0);
    let xs: Vec<f64> = (1..=40).map(|i| 0.5 * i as f64).collect();
    cdf_ppf_roundtrip(&d, &xs, "ChiSquared df=5");
}

#[test]
fn mr_studentt_cdf_ppf_roundtrip() {
    let d = StudentT::new(3.0);
    let xs: Vec<f64> = (-30..=30).map(|i| 0.25 * i as f64).collect();
    cdf_ppf_roundtrip(&d, &xs, "StudentT df=3");
}

#[test]
fn mr_gamma_cdf_ppf_roundtrip() {
    let d = GammaDist { a: 2.5, scale: 1.0 };
    let xs: Vec<f64> = (1..=40).map(|i| 0.2 * i as f64).collect();
    cdf_ppf_roundtrip(&d, &xs, "Gamma a=2.5");
}

#[test]
fn mr_beta_cdf_ppf_roundtrip() {
    let d = BetaDist { a: 2.0, b: 5.0 };
    let xs: Vec<f64> = (1..=49).map(|i| i as f64 / 50.0).collect();
    cdf_ppf_roundtrip(&d, &xs, "Beta a=2,b=5");
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — cdf monotonicity: x1 < x2 ⇒ cdf(x1) ≤ cdf(x2)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cdf_monotonic() {
    let d = Normal {
        loc: 0.0,
        scale: 1.0,
    };
    let xs: Vec<f64> = (-50..=50).map(|i| 0.1 * i as f64).collect();
    let cdfs: Vec<f64> = xs.iter().map(|&x| d.cdf(x)).collect();
    for i in 1..cdfs.len() {
        assert!(
            cdfs[i] >= cdfs[i - 1] - 1e-15,
            "Normal cdf not monotone at i={i}: cdfs[{}]={} cdfs[{}]={}",
            i - 1,
            cdfs[i - 1],
            i,
            cdfs[i]
        );
    }
    // Bounds.
    assert!(cdfs[0] >= 0.0 && cdfs[0] < 1e-3);
    assert!(cdfs[cdfs.len() - 1] > 1.0 - 1e-3 && cdfs[cdfs.len() - 1] <= 1.0);
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — cdf + sf = 1 (identity)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cdf_plus_sf_equals_one() {
    let d = Normal {
        loc: 0.0,
        scale: 1.0,
    };
    for x in (-50..=50).map(|i| 0.1 * i as f64) {
        let c = d.cdf(x);
        let s = d.sf(x);
        // Loosened tolerance for the deep tail: SciPy's default sf
        // implementation can drop precision when cdf saturates near 1.
        assert!(
            (c + s - 1.0).abs() <= 1e-9 * (1.0 + c.abs() + s.abs()),
            "MR3 cdf+sf != 1 at x={x}: cdf={c} sf={s} sum={}",
            c + s
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — Pearson correlation symmetry: pearsonr(x, y) == pearsonr(y, x)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_pearsonr_symmetric() {
    let x = vec![1.0, 2.0, 3.5, 4.2, 6.0, 7.5, 9.0, 10.0];
    let y = vec![2.1, 4.0, 5.8, 6.9, 9.5, 11.2, 13.0, 14.7];
    let rxy = pearsonr(&x, &y);
    let ryx = pearsonr(&y, &x);
    assert_close(rxy.statistic, ryx.statistic, "MR4 pearsonr symmetric stat");
    assert_close(rxy.pvalue, ryx.pvalue, "MR4 pearsonr symmetric pvalue");
}

#[test]
fn mr_spearmanr_symmetric() {
    let x = vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
    let rxy = spearmanr(&x, &y);
    let ryx = spearmanr(&y, &x);
    assert_close(rxy.statistic, ryx.statistic, "MR4 spearmanr symmetric stat");
    assert_close(rxy.pvalue, ryx.pvalue, "MR4 spearmanr symmetric pvalue");
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — Pearson on identical sequences = +1; on negated sequence = -1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_pearsonr_identical_and_anti() {
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let r_self = pearsonr(&x, &x);
    assert!(
        (r_self.statistic - 1.0).abs() < 1e-12,
        "MR5 pearsonr(x, x) != 1: got {}",
        r_self.statistic
    );
    let neg: Vec<f64> = x.iter().map(|v| -v).collect();
    let r_neg = pearsonr(&x, &neg);
    assert!(
        (r_neg.statistic + 1.0).abs() < 1e-12,
        "MR5 pearsonr(x, -x) != -1: got {}",
        r_neg.statistic
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — t-test sign symmetry: ttest_ind(b, a).statistic = -ttest_ind(a, b).statistic
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ttest_ind_sign_symmetry() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let ab = ttest_ind(&a, &b);
    let ba = ttest_ind(&b, &a);
    assert_close(ab.statistic, -ba.statistic, "MR6 ttest_ind sign symmetry");
    assert_close(ab.pvalue, ba.pvalue, "MR6 ttest_ind pvalue symmetry");
    assert_close(ab.df, ba.df, "MR6 ttest_ind df symmetry");
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — KS-2samp swap symmetry: ks_2samp(a, b) ≈ ks_2samp(b, a) (statistic
//        and pvalue identical because the test is symmetric).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ks_2samp_swap_symmetric() {
    let a = vec![1.0, 2.0, 2.5, 3.0, 4.0, 5.5, 7.0, 8.5, 10.0, 12.0];
    let b = vec![1.5, 3.5, 4.5, 5.0, 6.0, 7.5, 9.0, 10.5, 11.0, 13.5];
    let ab = ks_2samp(&a, &b);
    let ba = ks_2samp(&b, &a);
    assert_close(ab.statistic, ba.statistic, "MR7 ks_2samp swap stat");
    assert_close(ab.pvalue, ba.pvalue, "MR7 ks_2samp swap pvalue");
}

// ─────────────────────────────────────────────────────────────────────
// MR8 — ttest_1samp shift invariance: subtracting the popmean from
//        every observation moves the popmean to 0 and yields the same
//        t-statistic.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ttest_1samp_shift_invariance() {
    let data = vec![5.1, 5.4, 4.9, 5.5, 5.0, 5.6, 5.3, 5.2];
    let popmean = 5.0;
    let r1 = ttest_1samp(&data, popmean);
    let shifted: Vec<f64> = data.iter().map(|&v| v - popmean).collect();
    let r2 = ttest_1samp(&shifted, 0.0);
    assert_close(
        r1.statistic,
        r2.statistic,
        "MR8 ttest_1samp shift invariance stat",
    );
    assert_close(
        r1.pvalue,
        r2.pvalue,
        "MR8 ttest_1samp shift invariance pvalue",
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR9 — pdf is non-negative and cdf is in [0, 1] across the support.
// ─────────────────────────────────────────────────────────────────────

fn check_pdf_cdf<D: ContinuousDistribution>(d: &D, label: &str) {
    for x in (-50..=50).map(|i| 0.2 * i as f64) {
        let p = d.pdf(x);
        let c = d.cdf(x);
        assert!(p >= 0.0 || p.is_nan(), "MR9 {label} pdf({x}) < 0: {p}");
        assert!(
            (0.0..=1.0).contains(&c) || c.is_nan(),
            "MR9 {label} cdf({x}) outside [0,1]: {c}"
        );
    }
}

#[test]
fn mr_pdf_nonneg_cdf_in_unit_interval() {
    check_pdf_cdf(&Normal::default(), "Normal(0,1)");
    check_pdf_cdf(&Uniform::default(), "Uniform(0,1)");
    check_pdf_cdf(&Exponential::new(1.0), "Exponential(1)");
    check_pdf_cdf(&ChiSquared::new(3.0), "ChiSquared(3)");
    check_pdf_cdf(&StudentT::new(5.0), "StudentT(5)");
}

// ─────────────────────────────────────────────────────────────────────
// MR10 — Bernoulli pmf normalisation: pmf(0) + pmf(1) = 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_bernoulli_pmf_sums_to_one() {
    for &p in &[0.0_f64, 0.1, 0.25, 0.5, 0.75, 0.99, 1.0] {
        let d = Bernoulli::new(p);
        let sum = d.pmf(0) + d.pmf(1);
        assert_close(sum, 1.0, &format!("MR10 Bernoulli p={p}"));
        assert!(d.pmf(0) >= 0.0 && d.pmf(1) >= 0.0);
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR11 — Binomial pmf sums to ~1 over [0, n].
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_binomial_pmf_sums_to_one() {
    for &n in &[1_u64, 5, 10, 25, 50] {
        for &p in &[0.1_f64, 0.5, 0.9] {
            let d = Binomial::new(n, p);
            let sum: f64 = (0..=n).map(|k| d.pmf(k)).sum();
            assert_close(sum, 1.0, &format!("MR11 Binomial n={n}, p={p}"));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR12 — Poisson pmf sums to ~1 over a sufficiently wide range, and
// mean ≈ μ.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_poisson_pmf_sums_and_mean() {
    for &mu in &[0.5_f64, 1.0, 5.0, 12.0] {
        let d = Poisson::new(mu);
        // Sum over [0, μ + 30·√μ] should capture > 1 - 1e-10 of mass.
        let upper = (mu + 30.0 * mu.sqrt()).ceil() as u64 + 1;
        let sum: f64 = (0..=upper).map(|k| d.pmf(k)).sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "MR12 Poisson μ={mu} sum={sum}, expected 1"
        );
        // Mean test: Σ k · pmf(k) ≈ μ.
        let mean: f64 = (0..=upper).map(|k| (k as f64) * d.pmf(k)).sum();
        assert!(
            (mean - mu).abs() < 1e-7 * mu.max(1.0),
            "MR12 Poisson mean: got {mean}, expected {mu}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR13 — Geometric mean = 1/p (success probability p).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_geometric_mean_inverse_p() {
    for &p in &[0.1_f64, 0.25, 0.5, 0.75, 0.9] {
        let d = Geometric::new(p);
        let expected = 1.0 / p;
        let mean = d.mean();
        assert!(
            (mean - expected).abs() < 1e-9 * expected.abs().max(1.0),
            "MR13 Geometric mean: p={p} got {mean}, expected {expected}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR14 — Normal mean and variance match analytic values:
// Normal(loc=μ, scale=σ).mean() = μ, .var() = σ².
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_normal_mean_and_variance() {
    for &(loc, scale) in &[(0.0_f64, 1.0_f64), (1.5, 2.5), (-3.7, 0.7)] {
        let d = Normal { loc, scale };
        assert!(
            (d.mean() - loc).abs() < 1e-12,
            "MR14 Normal mean: got {}, expected {loc}",
            d.mean()
        );
        let expected_var = scale * scale;
        assert!(
            (d.var() - expected_var).abs() < 1e-12,
            "MR14 Normal var: got {}, expected {expected_var}",
            d.var()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR15 — Exponential mean = 1/λ.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_exponential_mean() {
    for &lambda in &[0.5_f64, 1.0, 2.0, 5.0] {
        let d = Exponential::new(lambda);
        let expected = 1.0 / lambda;
        let mean = d.mean();
        assert!(
            (mean - expected).abs() < 1e-12,
            "MR15 Exponential mean: λ={lambda} got {mean}, expected {expected}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR16 — Beta mean = a / (a + b).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_beta_mean() {
    for &(a, b) in &[(2.0_f64, 5.0_f64), (1.0, 1.0), (3.5, 2.5), (0.5, 0.5)] {
        let d = BetaDist { a, b };
        let expected = a / (a + b);
        let mean = d.mean();
        assert!(
            (mean - expected).abs() < 1e-12,
            "MR16 Beta mean: a={a}, b={b} got {mean}, expected {expected}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR17 — cdf(ppf(0.5)) = 0.5: the median is consistent with the cdf
// at the median value.
// ─────────────────────────────────────────────────────────────────────

fn check_median<D: ContinuousDistribution>(d: &D, label: &str) {
    let med = d.ppf(0.5);
    let v = d.cdf(med);
    assert!(
        (v - 0.5).abs() < 1e-8,
        "MR17 {label} cdf(ppf(0.5)) = {v}, expected 0.5"
    );
}

#[test]
fn mr_cdf_at_median_is_half() {
    check_median(
        &Normal {
            loc: 0.5,
            scale: 1.5,
        },
        "Normal",
    );
    check_median(
        &Uniform {
            loc: -2.0,
            scale: 5.0,
        },
        "Uniform",
    );
    check_median(&Exponential::new(2.0), "Exponential");
    check_median(&ChiSquared::new(5.0), "ChiSquared");
    check_median(&StudentT::new(3.0), "StudentT");
    check_median(&BetaDist { a: 2.0, b: 5.0 }, "Beta");
}

// ─────────────────────────────────────────────────────────────────────
// MR18 — sample skew of a symmetric distribution is ≈ 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_skew_of_symmetric_data_is_zero() {
    // Symmetric data around 0: [-3, -2, -1, 0, 1, 2, 3] mirrored.
    let data: Vec<f64> = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let s = skew(&data);
    assert!(s.abs() < 1e-12, "MR18 symmetric skew = {s}, expected 0");
}

// ─────────────────────────────────────────────────────────────────────
// MR19 — sample skew of a left-skewed distribution is negative,
// right-skewed is positive.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_skew_sign_for_skewed_data() {
    // Left-skewed: long lower tail, mass concentrated on the right.
    let left = vec![-10.0, -5.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0];
    let s_left = skew(&left);
    assert!(s_left < 0.0, "MR19 left-skew should be negative: {s_left}");
    // Right-skewed: long upper tail.
    let right: Vec<f64> = left.iter().map(|v| -v).collect();
    let s_right = skew(&right);
    assert!(
        s_right > 0.0,
        "MR19 right-skew should be positive: {s_right}"
    );
    // Mirror symmetry: skew(-x) = -skew(x).
    assert!(
        (s_left + s_right).abs() < 1e-9,
        "MR19 mirror symmetry: {s_left} + {s_right} != 0"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR20 — kurtosis of a constant array is well-defined: degenerate
// data (zero variance) returns NaN, matching scipy's policy on
// pathological inputs.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kurtosis_constant_is_nan_or_zero() {
    let data = vec![3.7_f64; 10];
    let k = kurtosis(&data);
    // Either NaN (well-defined "no shape") or 0 (constant ⇒ k=−3 or 0
    // with scipy's "fisher" convention) — both are acceptable; the
    // important thing is we don't return a finite garbage value.
    assert!(
        k.is_nan() || k.abs() < 1e-9,
        "MR20 kurtosis on constant: {k} — should be NaN or 0"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR21 — mannwhitneyu pvalue is invariant under sample swap (the test
// is two-sided by default).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_mannwhitneyu_swap_invariant_pvalue() {
    let a = vec![1.0_f64, 3.0, 5.0, 7.0, 9.0];
    let b = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
    let ab = mannwhitneyu(&a, &b);
    let ba = mannwhitneyu(&b, &a);
    assert_close(ab.pvalue, ba.pvalue, "MR21 mannwhitneyu pvalue swap");
}

// ─────────────────────────────────────────────────────────────────────
// MR22 — wilcoxon paired-sign-rank with x = y returns p-value of 1.0
// (no signed-rank evidence of difference).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_wilcoxon_identical_samples() {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = wilcoxon(&x, &x);
    // p-value should be 1.0 (or NaN for the all-zero case if implementation
    // chooses to flag degeneracy). Either is acceptable.
    assert!(
        result.pvalue.is_nan() || (result.pvalue - 1.0).abs() < 1e-9,
        "MR22 wilcoxon(x, x): pvalue = {}, expected 1 or NaN",
        result.pvalue
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR23 — ttest_rel statistic flips sign when (a, b) is replaced with
// (b, a): the t statistic of paired differences negates.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ttest_rel_sign_symmetry() {
    let a = vec![1.0_f64, 3.0, 5.0, 6.0, 8.0];
    let b = vec![2.0_f64, 4.0, 4.5, 7.0, 9.0];
    let ab = ttest_rel(&a, &b, None).unwrap();
    let ba = ttest_rel(&b, &a, None).unwrap();
    assert_close(ab.statistic, -ba.statistic, "MR23 ttest_rel sign");
    assert_close(ab.pvalue, ba.pvalue, "MR23 ttest_rel pvalue swap");
}

// ─────────────────────────────────────────────────────────────────────
// MR24 — Power-mean ordering on positive data:
//     hmean ≤ gmean ≤ amean (= pmean(p=1)) ≤ pmean(p=2).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_power_mean_ordering() {
    let xs = [
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0],
        vec![0.5_f64, 1.5, 4.0, 7.0, 10.0],
        vec![1.0_f64, 1.0, 1.0, 1.0],
        vec![0.1_f64, 0.5, 1.0, 5.0, 10.0],
    ];
    for x in &xs {
        let h = hmean(x);
        let g = gmean(x);
        let a = pmean(x, 1.0);
        let q = pmean(x, 2.0);
        assert!(h <= g + 1e-12, "MR24 hmean = {h} > gmean = {g} on {x:?}");
        assert!(g <= a + 1e-12, "MR24 gmean = {g} > amean = {a} on {x:?}");
        assert!(
            a <= q + 1e-12,
            "MR24 amean = {a} > pmean(p=2) = {q} on {x:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR25 — Quantile is monotonic in q: q1 ≤ q2 ⇒ quantile(x, q1) ≤
// quantile(x, q2).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_quantile_monotonic_in_q() {
    let x = vec![3.0_f64, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0, 5.5, 3.5];
    let qs = vec![0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
    let res = quantile(&x, &qs);
    for w in res.windows(2) {
        assert!(
            w[0] <= w[1] + 1e-12,
            "MR25 quantile not monotonic: {} > {}",
            w[0],
            w[1]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR26 — wasserstein_distance(u, u) = 0 (identity of indiscernibles).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_wasserstein_self_distance_zero() {
    for x in &[
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0],
        vec![0.5_f64; 8],
        vec![-3.0, -1.0, 0.5, 2.0, 4.0, 7.0],
    ] {
        let d = wasserstein_distance(x, x);
        assert!(d.abs() < 1e-12, "MR26 wasserstein(u, u) = {d}, expected 0");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR27 — energy_distance is symmetric: ed(u, v) = ed(v, u).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_energy_distance_symmetric() {
    let u = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let v = vec![2.0_f64, 3.0, 4.0, 5.0, 6.0, 7.0];
    let uv = energy_distance(&u, &v);
    let vu = energy_distance(&v, &u);
    assert!(
        (uv - vu).abs() < 1e-12,
        "MR27 ed(u, v) = {uv} vs ed(v, u) = {vu}"
    );
    assert!(uv >= -1e-12, "MR27 ed = {uv} < 0");
}

// ─────────────────────────────────────────────────────────────────────
// MR28 — One-way ANOVA F statistic is non-negative on any input.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_f_oneway_statistic_nonneg() {
    let g1 = [1.0_f64, 2.0, 3.0, 4.0];
    let g2 = [1.5_f64, 2.5, 3.5];
    let g3 = [1.2_f64, 2.1, 3.3, 4.4, 5.5];
    let groups: Vec<&[f64]> = vec![&g1, &g2, &g3];
    let res = f_oneway(&groups);
    assert!(
        res.statistic >= -1e-12,
        "MR28 f_oneway statistic = {} < 0",
        res.statistic
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR29 — gmean of a constant array equals the constant; hmean too.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gmean_hmean_pmean_constant() {
    for &c in &[1.0_f64, 2.5, 7.0, 100.0] {
        let x = vec![c; 12];
        assert!(
            (gmean(&x) - c).abs() < 1e-10,
            "MR29 gmean(const {c}) = {}",
            gmean(&x)
        );
        assert!(
            (hmean(&x) - c).abs() < 1e-10,
            "MR29 hmean(const {c}) = {}",
            hmean(&x)
        );
        for &p in &[-2.0_f64, 0.5, 1.0, 2.0, 3.0] {
            let pm = pmean(&x, p);
            assert!(
                (pm - c).abs() < 1e-10,
                "MR29 pmean(const {c}, p = {p}) = {pm}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR30 — diff(x) has length n - 1; diff of an arithmetic progression
// is a constant equal to the step.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_diff_length_and_arithmetic_progression() {
    let x: Vec<f64> = (0..16).map(|i| 2.5 * i as f64 + 1.0).collect();
    let d = diff(&x);
    assert_eq!(d.len(), x.len() - 1, "MR30 diff length");
    for (i, &v) in d.iter().enumerate() {
        assert!(
            (v - 2.5).abs() < 1e-12,
            "MR30 diff[{i}] = {v}, expected 2.5"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR31 — histogram counts sum to n (every input lands in some bin
// when the bin range covers [min, max]).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_histogram_counts_sum_to_n() {
    let data = vec![
        1.0_f64, 2.5, 3.0, 4.5, 5.0, 6.0, 7.5, 8.0, 9.5, 10.0, -1.0, -3.0,
    ];
    for &bins in &[3usize, 5, 8, 12] {
        let (counts, _edges) = histogram(&data, bins);
        let total: usize = counts.iter().sum();
        assert_eq!(total, data.len(), "MR31 histogram(bins={bins}) total");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR32 — ecdf is monotonic non-decreasing in x_eval and lies in [0, 1].
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ecdf_monotonic_in_unit_interval() {
    let data = vec![1.0_f64, 2.0, 3.5, 5.0, 6.5, 8.0, 9.0, 10.0];
    let mut x_eval: Vec<f64> = (0..21).map(|i| i as f64 * 0.5).collect();
    x_eval.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cdf = ecdf(&data, &x_eval);
    for w in cdf.windows(2) {
        assert!(
            w[0] <= w[1] + 1e-12,
            "MR32 ecdf not monotone: {} > {}",
            w[0],
            w[1]
        );
    }
    for &v in &cdf {
        assert!(
            (-1e-12..=1.0 + 1e-12).contains(&v),
            "MR32 ecdf value = {v} outside [0, 1]"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR33 — Theil-Sen recovers slope and intercept of a linear model
// y = a·x + b within tolerance.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_theil_sen_recovers_linear_params() {
    let a = 2.5_f64;
    let b = -1.0_f64;
    let x: Vec<f64> = (0..40).map(|i| 0.25 * i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| a * xi + b).collect();
    let (slope, intercept) = theil_sen(&x, &y);
    assert!(
        (slope - a).abs() < 1e-9,
        "MR33 theil_sen slope = {slope} vs {a}"
    );
    assert!(
        (intercept - b).abs() < 1e-9,
        "MR33 theil_sen intercept = {intercept} vs {b}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR34 — Ridge regression with α = 0 produces a coefficient vector
// that satisfies the linear system within the OLS tolerance for a
// well-conditioned design.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ridge_zero_alpha_recovers_known_signal() {
    // y = 0.5 + 1.5·x1 + 0.5·x2 (with intercept — ridge_regression
    // includes an intercept column at coeffs[0]).
    let x: Vec<Vec<f64>> = vec![
        vec![1.0, 2.0],
        vec![2.0, 1.0],
        vec![3.0, 4.0],
        vec![4.0, 5.0],
        vec![5.0, 1.0],
        vec![6.0, 3.0],
    ];
    let y: Vec<f64> = x.iter().map(|r| 0.5 + 1.5 * r[0] + 0.5 * r[1]).collect();
    let coeffs = ridge_regression(&x, &y, 0.0);
    // coeffs is [intercept, β1, β2]. Predictions follow that layout.
    let residual: f64 = x
        .iter()
        .zip(&y)
        .map(|(row, &yi)| {
            let pred = coeffs[0] + row[0] * coeffs[1] + row[1] * coeffs[2];
            (pred - yi).powi(2)
        })
        .sum::<f64>()
        .sqrt();
    assert!(
        residual < 1e-6,
        "MR34 ridge(α=0) residual = {residual}, expected ≈ 0"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR35 — pacf[0] is 1 by definition (lag-zero partial autocorrelation).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_pacf_at_lag_zero_is_one() {
    let x: Vec<f64> = (0..32)
        .map(|i| (i as f64 * 0.4).cos() + 0.5 * (i as f64 * 0.7).sin())
        .collect();
    let p = pacf(&x, 6);
    assert!(!p.is_empty(), "MR35 pacf empty");
    assert!(
        (p[0] - 1.0).abs() < 1e-9,
        "MR35 pacf[0] = {}, expected 1",
        p[0]
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR36 — linregress on y = a·x + b reproduces (a, b) exactly.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_linregress_recovers_exact_line() {
    let a = 2.0_f64;
    let b = -1.5_f64;
    let x: Vec<f64> = (0..30).map(|i| 0.2 * i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| a * xi + b).collect();
    let r = linregress(&x, &y);
    assert!(
        (r.slope - a).abs() < 1e-10,
        "MR36 linregress slope = {} vs {a}",
        r.slope
    );
    assert!(
        (r.intercept - b).abs() < 1e-10,
        "MR36 linregress intercept = {} vs {b}",
        r.intercept
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR37 — Kruskal-Wallis on identical groups returns near-zero statistic
// (all ranks equal in expectation; p-value ≈ 1).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kruskal_identical_groups_small_stat() {
    let g1 = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let g2 = g1;
    let g3 = g1;
    let groups: Vec<&[f64]> = vec![&g1, &g2, &g3];
    let r = kruskal(&groups);
    assert!(
        r.statistic.abs() < 1e-3,
        "MR37 kruskal identical statistic = {}",
        r.statistic
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR38 — ranksums of (x, y) and (y, x) yield negated statistics.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ranksums_swap_negates_statistic() {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![6.0_f64, 7.0, 8.0, 9.0, 10.0];
    let xy = ranksums(&x, &y);
    let yx = ranksums(&y, &x);
    assert!(
        (xy.statistic + yx.statistic).abs() < 1e-9,
        "MR38 ranksums(x,y) + ranksums(y,x) = {}",
        xy.statistic + yx.statistic
    );
    assert!(
        (xy.pvalue - yx.pvalue).abs() < 1e-9,
        "MR38 ranksums pvalue mismatch: {} vs {}",
        xy.pvalue,
        yx.pvalue
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR39 — Levene on identical groups yields near-zero statistic.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_levene_identical_groups_small_stat() {
    let g1 = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let g2 = g1;
    let groups: Vec<&[f64]> = vec![&g1, &g2];
    let r = levene(&groups);
    assert!(
        r.statistic.abs() < 1e-9,
        "MR39 levene identical = {}",
        r.statistic
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR40 — Bartlett on identical groups yields near-zero statistic.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_bartlett_identical_groups_small_stat() {
    let g1 = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let g2 = g1;
    let groups: Vec<&[f64]> = vec![&g1, &g2];
    let r = bartlett_var(&groups);
    assert!(
        r.statistic.abs() < 1e-9,
        "MR40 bartlett identical = {}",
        r.statistic
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR41 — Tukey HSD diagonal of pairwise statistic and p-value matrices
// indicates a group compared with itself: statistic = 0, p-value ≈ 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_tukey_hsd_self_comparison() {
    let g1 = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let g2 = [4.0_f64, 5.0, 6.0, 7.0, 8.0];
    let g3 = [10.0_f64, 11.0, 12.0, 13.0, 14.0];
    let groups: Vec<&[f64]> = vec![&g1, &g2, &g3];
    let r = tukey_hsd(&groups);
    for i in 0..groups.len() {
        let s = r.statistic[i][i];
        assert!(
            s.abs() < 1e-12,
            "MR41 tukey diagonal statistic[{i}, {i}] = {s}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR42 — Relative frequencies sum to 1.0 across all bins.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_relfreq_sums_to_one() {
    let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    for bins in [3usize, 5, 8, 10] {
        let (freqs, _edges) = relfreq(&data, bins);
        let sum: f64 = freqs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "MR42 relfreq(bins={bins}) Σ = {sum}, expected 1"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR43 — Cumulative frequencies are non-decreasing and end at n.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cumfreq_monotone_total_n() {
    let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    for bins in [3usize, 5, 8] {
        let (cum, _edges) = cumfreq(&data, bins);
        for w in cum.windows(2) {
            assert!(
                w[0] <= w[1] + 1e-12,
                "MR43 cumfreq not monotone: {} > {}",
                w[0],
                w[1]
            );
        }
        let last = *cum.last().unwrap();
        assert!(
            (last - data.len() as f64).abs() < 1e-9,
            "MR43 cumfreq last = {last}, expected {}",
            data.len()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR44 — Durbin-Watson statistic lies in [0, 4].
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_durbin_watson_in_zero_four() {
    let residuals: Vec<f64> = (0..30)
        .map(|i| (i as f64 * 0.4).sin() + 0.3 * ((i as f64) * 0.13).cos())
        .collect();
    let dw = durbin_watson(&residuals);
    assert!(
        (-1e-12..=4.0 + 1e-12).contains(&dw),
        "MR44 durbin_watson = {dw} outside [0, 4]"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR45 — acf[0] = 1 (autocorrelation at lag 0 is 1 by normalisation).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_acf_at_lag_zero_is_one() {
    let x: Vec<f64> = (0..40).map(|i| (i as f64 * 0.4).cos()).collect();
    let a = acf(&x, 6);
    assert!(!a.is_empty(), "MR45 acf empty");
    assert!(
        (a[0] - 1.0).abs() < 1e-9,
        "MR45 acf[0] = {}, expected 1",
        a[0]
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR46 — Mann-Kendall on a strictly monotone increasing sequence
// returns trend=+1 (positive trend).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_mannkendall_monotone_trend() {
    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let (_stat, _p, trend) = mannkendall(&x);
    assert_eq!(
        trend, 1,
        "MR46 mannkendall trend = {trend}, expected +1 for strictly increasing"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR47 — Runs test on a non-empty input returns at least 1 run.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_runs_test_at_least_one_run() {
    let x: Vec<f64> = (0..30).map(|i| (i as f64 * 0.4).sin()).collect();
    let (n_runs, _stat, _p) = runs_test(&x);
    assert!(
        n_runs >= 1,
        "MR47 runs_test n_runs = {n_runs}, expected ≥ 1"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR48 — zscore output has mean ≈ 0 and std ≈ 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_zscore_normalises_to_zero_mean_unit_std() {
    let x: Vec<f64> = (0..30).map(|i| (i as f64) * 1.5 + 7.0).collect();
    let z = zscore(&x);
    let n = z.len() as f64;
    let mean: f64 = z.iter().sum::<f64>() / n;
    let var: f64 = z.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    assert!(mean.abs() < 1e-9, "MR48 zscore mean = {mean}");
    assert!((std - 1.0).abs() < 1e-9, "MR48 zscore std = {std}");
}

// ─────────────────────────────────────────────────────────────────────
// MR49 — percentile is in [min(data), max(data)] for any q ∈ [0, 100].
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_percentile_in_data_range() {
    let data: Vec<f64> = vec![1.0, 5.0, 3.0, 8.0, 2.0, 7.0, 4.0, 6.0, 9.0, 10.0];
    let mn = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let mx = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    for q in [0.0_f64, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0] {
        let p = percentile(&data, q);
        assert!(
            p >= mn - 1e-9 && p <= mx + 1e-9,
            "MR49 percentile(q={q}) = {p} outside [{mn}, {mx}]"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR50 — trim_mean is in [min(data), max(data)].
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_trim_mean_in_data_range() {
    let data: Vec<f64> = vec![1.0, 5.0, 3.0, 8.0, 2.0, 7.0, 4.0, 6.0, 9.0, 100.0];
    let mn = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let mx = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    for &p in &[0.0_f64, 0.1, 0.25, 0.4] {
        let m = trim_mean(&data, p).expect("metamorphic proportions are all <= 0.5");
        assert!(
            m >= mn - 1e-9 && m <= mx + 1e-9,
            "MR50 trim_mean(p={p}) = {m} outside [{mn}, {mx}]"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR51 — tmean over the full range matches arithmetic mean.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_tmean_full_range_matches_mean() {
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    let tm = tmean(&data, (-1e9, 1e9), (true, true));
    assert!(
        (tm - mean).abs() < 1e-9,
        "MR51 tmean(full) = {tm} vs mean = {mean}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR52 — sigmaclip with very wide bounds preserves all input data.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sigmaclip_wide_bounds_preserves_data() {
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let r = sigmaclip(&data, 100.0, 100.0);
    assert_eq!(
        r.clipped.len(),
        data.len(),
        "MR52 sigmaclip wide bounds: clipped len = {} vs data = {}",
        r.clipped.len(),
        data.len()
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR53 — variation equals std/mean for positive data with non-zero mean.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_variation_equals_std_over_mean() {
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let n = x.len() as f64;
    let mean: f64 = x.iter().sum::<f64>() / n;
    let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    let v = variation(&x);
    let expected = std / mean;
    assert!(
        (v - expected).abs() < 1e-9,
        "MR53 variation = {v} vs std/mean = {expected}"
    );
}
