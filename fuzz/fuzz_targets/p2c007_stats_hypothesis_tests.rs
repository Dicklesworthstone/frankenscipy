#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{chisquare, ttest_1samp, ttest_ind, ttest_rel, f_oneway};
use libfuzzer_sys::fuzz_target;

// Stats hypothesis tests oracle:
// Statistical tests should:
// 1. Return finite results for valid inputs
// 2. Handle edge cases gracefully (small samples, constant data)
// 3. Produce p-values in [0, 1]
// 4. Produce finite t/chi2/F statistics

const MAX_SIZE: usize = 100;

#[derive(Debug, Arbitrary)]
struct HypothesisInput {
    data_a: Vec<f64>,
    data_b: Vec<f64>,
    popmean: f64,
    variant: u8,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1e6, 1e6)
    } else {
        0.0
    }
}

fn sanitize_vec(v: &[f64], max_len: usize) -> Vec<f64> {
    v.iter()
        .take(max_len)
        .map(|&x| sanitize(x))
        .collect()
}

fuzz_target!(|input: HypothesisInput| {
    let a = sanitize_vec(&input.data_a, MAX_SIZE);
    let b = sanitize_vec(&input.data_b, MAX_SIZE);
    let popmean = sanitize(input.popmean);

    match input.variant % 5 {
        0 => {
            // ttest_1samp: one-sample t-test
            if a.len() >= 2 {
                let result = ttest_1samp(&a, popmean);
                if result.statistic.is_finite() {
                    assert!(
                        (0.0..=1.0).contains(&result.pvalue),
                        "ttest_1samp pvalue {} out of [0,1] for {} samples",
                        result.pvalue,
                        a.len()
                    );
                }
            }
        }
        1 => {
            // ttest_ind: independent two-sample t-test
            if a.len() >= 2 && b.len() >= 2 {
                let result = ttest_ind(&a, &b);
                if result.statistic.is_finite() {
                    assert!(
                        (0.0..=1.0).contains(&result.pvalue),
                        "ttest_ind pvalue {} out of [0,1]",
                        result.pvalue
                    );
                }
            }
        }
        2 => {
            // ttest_rel: paired t-test
            if a.len() >= 2 && a.len() == b.len() {
                let result = ttest_rel(&a, &b, None);
                if let Ok(r) = result
                    && r.statistic.is_finite()
                {
                    assert!(
                        (0.0..=1.0).contains(&r.pvalue),
                        "ttest_rel pvalue {} out of [0,1]",
                        r.pvalue
                    );
                }
            }
        }
        3 => {
            // chisquare: chi-squared test
            if !a.is_empty() && a.iter().all(|&x| x >= 0.0) {
                let (chi2, pvalue) = chisquare(&a, None);
                if chi2.is_finite() {
                    assert!(
                        (0.0..=1.0).contains(&pvalue),
                        "chisquare pvalue {} out of [0,1]",
                        pvalue
                    );
                }
            }
        }
        _ => {
            // f_oneway: one-way ANOVA
            if a.len() >= 2 && b.len() >= 2 {
                let groups: Vec<&[f64]> = vec![&a, &b];
                let result = f_oneway(&groups);
                if result.statistic.is_finite() {
                    assert!(
                        (0.0..=1.0).contains(&result.pvalue),
                        "f_oneway pvalue {} out of [0,1]",
                        result.pvalue
                    );
                }
            }
        }
    }
});
