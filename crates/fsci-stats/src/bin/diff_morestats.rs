//! Less-common stats fns vs scipy.stats (gitignored). Capture `... 2>&1 | grep -E '^[a-z].*,'`.
use fsci_stats::*;
fn t(name: &str, s: f64, p: f64) {
    println!("{name},{s:.17e},{p:.17e}");
}
fn main() {
    // moderately-sized normal-ish sample for the normality tests (need n>=8 / 20).
    let a = [
        2.1, 3.4, 1.9, 5.2, 4.1, 3.3, 2.8, 4.9, 3.7, 2.2, 4.5, 3.1, 2.6, 5.0, 3.9, 2.4, 4.2, 3.6,
        2.9, 4.7, 3.2, 2.7,
    ];
    if let Ok(r) = skewtest(&a, None, None) {
        t("skewtest", r.statistic, r.pvalue);
    }
    if let Ok(r) = kurtosistest(&a, None, None) {
        t("kurtosistest", r.statistic, r.pvalue);
    }
    let r = normaltest(&a);
    t("normaltest", r.statistic, r.pvalue);
    let r = jarque_bera(&a);
    t("jarque_bera", r.statistic, r.pvalue);
    let r = shapiro(&a);
    t("shapiro", r.statistic, r.pvalue);

    // chisquare / power_divergence
    let obs = [16.0, 18.0, 16.0, 14.0, 12.0, 12.0];
    let (s, p) = chisquare(&obs, None);
    t("chisquare", s, p);
    for (lab, lam) in [
        ("pd_cressie", 2.0 / 3.0),
        ("pd_loglik", 0.0),
        ("pd_neyman", -2.0),
    ] {
        let (s, p) = power_divergence(&obs, None, lam);
        t(lab, s, p);
    }

    // combine_pvalues across methods
    let pv = [0.01, 0.2, 0.3, 0.05, 0.5];
    for m in [
        "fisher",
        "stouffer",
        "tippett",
        "pearson",
        "mudholkar_george",
    ] {
        if let Ok(r) = combine_pvalues(&pv, Some(m), None) {
            t(&format!("combine_{m}"), r.statistic, r.pvalue);
        }
    }

    // pointbiserialr
    let bin = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    let cont = [1.2, 2.3, 4.1, 3.8, 1.9, 4.5, 3.3, 2.1, 5.0, 1.5];
    let r = pointbiserialr(&bin, &cont);
    t("pointbiserialr", r.statistic, r.pvalue);

    // median_test
    let g1 = [
        10.0, 14.0, 14.0, 18.0, 20.0, 22.0, 24.0, 25.0, 31.0, 31.0, 35.0, 39.0, 43.0, 43.0, 48.0,
        49.0,
    ];
    let g2 = [
        28.0, 26.0, 25.0, 28.0, 27.0, 24.0, 22.0, 24.0, 21.0, 33.0, 27.0, 23.0, 24.0, 32.0, 31.0,
        32.0,
    ];
    let r = median_test(&[&g1, &g2]);
    t("median_test", r.statistic, r.pvalue);

    // binomtest (two-sided p-value)
    println!(
        "binomtest,{:.17e},0",
        binomtest(3, 10, 0.5).expect("valid binomtest args")
    );
    println!(
        "binomtest2,{:.17e},0",
        binomtest(8, 20, 0.3).expect("valid binomtest args")
    );
}
