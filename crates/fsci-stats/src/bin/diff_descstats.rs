use fsci_stats::*;
fn s(name: &str, ds: &str, v: f64) {
    println!("{name},{ds},{v:.17e}");
}
fn main() {
    let a: Vec<f64> = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0, 1.0, 3.0, 8.0, 6.0];
    let b: Vec<f64> = vec![
        1.3, -2.1, 0.5, 3.7, -1.2, 2.2, 0.8, -0.4, 1.9, 2.6, -3.1, 0.1,
    ];
    for (ds, d) in [("A", &a), ("B", &b)] {
        s("skew", ds, skew(d));
        s("kurtosis", ds, kurtosis(d));
        s("moment2", ds, moment(d, 2));
        s("moment3", ds, moment(d, 3));
        s("moment4", ds, moment(d, 4));
        s("variation", ds, variation(d));
        s("sem", ds, sem(d));
        s("iqr", ds, iqr(d));
        s("mad1", ds, median_abs_deviation(d, 1.0));
        s("trim_mean0.1", ds, trim_mean(d, 0.1).expect("0.1 is in range"));
        s("kstat1", ds, kstat(d, 1));
        s("kstat2", ds, kstat(d, 2));
        s("kstat3", ds, kstat(d, 3));
        s("kstat4", ds, kstat(d, 4));
        s("kstatvar1", ds, kstatvar(d, 1));
        s("kstatvar2", ds, kstatvar(d, 2));
        s("pmean2", ds, pmean(d, 2.0));
        s("pmean-1", ds, pmean(d, -1.0));
    }
    // positive-only array for gmean/hmean/gstd
    let pa: Vec<f64> = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0, 1.0, 3.0, 8.0, 6.0];
    s("gmean", "A", gmean(&pa));
    s("hmean", "A", hmean(&pa));
    s("gstd", "A", gstd(&pa));
    // correlations
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y: Vec<f64> = vec![2.1, 3.9, 6.2, 7.8, 11.0, 12.1, 13.5, 16.2];
    s("pearsonr", "XY", pearsonr(&x, &y).statistic);
    s("spearmanr", "XY", spearmanr(&x, &y).statistic);
    s("kendalltau", "XY", kendalltau(&x, &y).statistic);
}
