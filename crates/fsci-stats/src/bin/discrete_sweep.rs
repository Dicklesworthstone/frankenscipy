use fsci_stats::*;
fn emit<D: DiscreteDistribution>(label: &str, d: &D) {
    println!("{label},mean,{:.17e}", d.mean());
    println!("{label},var,{:.17e}", d.var());
    println!("{label},skew,{:.17e}", d.skewness());
    println!("{label},kurt,{:.17e}", d.kurtosis());
    println!("{label},entropy,{:.17e}", d.entropy());
    for k in [0_i64, 1, 2, 3, 5, 10] {
        println!("{label},pmf{k},{:.17e}", d.pmf(k));
        println!("{label},cdf{k},{:.17e}", d.cdf(k));
    }
}
fn main() {
    emit("poisson|3|0", &Poisson::new(3.0));
    emit("poisson|0.7|0", &Poisson::new(0.7));
    emit("binom|20|0.3", &Binomial::new(20, 0.3));
    emit("binom|10|0.6", &Binomial::new(10, 0.6));
    emit("betabinom|20|2|3", &BetaBinomial::new(20, 2.0, 3.0));
    emit("bernoulli|0.4|0", &Bernoulli::new(0.4));
    emit("geom|0.3|0", &Geometric::new(0.3));
    emit("geom|0.6|0", &Geometric::new(0.6));
    emit("boltzmann|0.5|10", &Boltzmann::new(0.5, 10));
    emit("planck|0.5|0", &Planck::new(0.5));
    emit("nbinom|5|0.4", &NegBinomial::new(5.0, 0.4));
    emit("nbinom|10|0.7", &NegBinomial::new(10.0, 0.7));
    emit("hypergeom|40|15|12", &Hypergeometric::new(40, 15, 12));
    emit("nhypergeom|20|7|3", &NegHypergeometric::new(20, 7, 3));
    emit("logser|0.6|0", &LogSeries::new(0.6));
    emit("yulesimon|3|0", &YuleSimon::new(3.0));
    emit("zipfian|2|20", &Zipfian::new(2.0, 20));
}
