//! discrete sf/logsf/logcdf right-tail probe vs scipy (gitignored).
use fsci_stats::*;
fn emit<D: DiscreteDistribution>(label: &str, d: &D, ks: &[i64]) {
    for &k in ks {
        println!("{label},sf,{k},{:.17e}", d.sf(k));
        println!("{label},logsf,{k},{:.17e}", d.logsf(k));
        println!("{label},logcdf,{k},{:.17e}", d.logcdf(k));
    }
}
fn main() {
    emit("geom|0.1", &Geometric::new(0.1), &[1, 5, 20, 100, 300, 700]);
    emit("geom|0.5", &Geometric::new(0.5), &[1, 5, 20, 50, 100]);
    emit("bernoulli|0.3", &Bernoulli::new(0.3), &[0, 1, 2]);
    emit("poisson|3", &Poisson::new(3.0), &[5, 20, 40, 80]);
    emit("yulesimon|2", &YuleSimon::new(2.0), &[1, 5, 50, 500]);
    emit("planck|0.5", &Planck::new(0.5), &[1, 10, 50, 150]);
    emit("boltzmann|0.5_10", &Boltzmann::new(0.5, 10), &[0, 5, 8, 9]);
}
