use fsci_stats::{BetaBinomial, DiscreteDistribution, Hypergeometric};
fn main() {
    let h = Hypergeometric::new(20, 7, 12);
    for k in 0_i64..=7 {
        println!("hyperg_pmf_{k},{:.17e},0", h.pmf(k));
        println!("hyperg_cdf_{k},{:.17e},0", h.cdf(k));
        println!("hyperg_sf_{k},{:.17e},0", h.sf(k));
        println!("hyperg_logpmf_{k},{:.17e},0", h.logpmf(k));
    }
    println!("hyperg_mean,{:.17e},0", h.mean());
    println!("hyperg_var,{:.17e},0", h.var());
    let bb = BetaBinomial::new(10, 2.0, 3.0);
    for k in [0_i64, 2, 5, 8, 10] {
        println!("betabinom_pmf_{k},{:.17e},0", bb.pmf(k));
        println!("betabinom_cdf_{k},{:.17e},0", bb.cdf(k));
        println!("betabinom_sf_{k},{:.17e},0", bb.sf(k));
    }
    println!("betabinom_mean,{:.17e},0", bb.mean());
    println!("betabinom_var,{:.17e},0", bb.var());
}
