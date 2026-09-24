use fsci_stats::*;
fn em<D: DiscreteDistribution>(label: &str, d: &D, ks: &[i64]) {
    println!("{label},mean,0,{:.16e}", d.mean());
    println!("{label},var,0,{:.16e}", d.var());
    println!("{label},skew,0,{:.16e}", d.skewness());
    println!("{label},kurt,0,{:.16e}", d.kurtosis());
    println!("{label},entropy,0,{:.16e}", d.entropy());
    for &k in ks {
        println!("{label},cdf,{k},{:.16e}", d.cdf(k));
        println!("{label},sf,{k},{:.16e}", d.sf(k));
    }
}
fn main() {
    em("skellam|20_15", &Skellam::new(20.0, 15.0), &[5, 20, 35, 50]);
    em("skellam|50_50", &Skellam::new(50.0, 50.0), &[30, 50, 70]);
    em("poisson|100", &Poisson::new(100.0), &[80, 100, 120, 150]);
    em(
        "binom|1000_0.3",
        &Binomial::new(1000, 0.3),
        &[280, 300, 320, 360],
    );
    em("yulesimon|1.2", &YuleSimon::new(1.2), &[5, 50, 500]);
    em(
        "zipfian|2.0_1000",
        &Zipfian::new(2.0, 1000),
        &[1, 10, 100, 900],
    );
}
