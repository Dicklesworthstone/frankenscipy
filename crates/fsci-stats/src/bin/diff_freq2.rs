//! Independent cross-check of cumfreq/relfreq vs scipy.stats incl edge cases.
use fsci_stats::{cumfreq, relfreq};
fn emit(tag: &str, data: &[f64], bins: usize) {
    let (cf, ce) = cumfreq(data, bins);
    let (rf, _re) = relfreq(data, bins);
    let cs: Vec<String> = cf.iter().map(|v| format!("{v:.10e}")).collect();
    let rs: Vec<String> = rf.iter().map(|v| format!("{v:.10e}")).collect();
    let es: Vec<String> = ce.iter().map(|v| format!("{v:.10e}")).collect();
    println!(
        "{tag}|cum:{}|rel:{}|edges:{}",
        cs.join(","),
        rs.join(","),
        es.join(",")
    );
}
fn main() {
    emit(
        "base",
        &[1.0, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0, 7.0, 8.0, 9.0],
        5,
    );
    emit("neg", &[-3.0, -1.0, 0.0, 0.5, 2.0, 4.0], 4);
    emit("twobin", &[1.0, 2.0, 3.0, 4.0, 10.0], 2);
    emit("allequal", &[5.0, 5.0, 5.0, 5.0], 3);
    emit(
        "manybins",
        &[0.1, 0.4, 0.9, 1.2, 2.5, 3.1, 3.9, 4.4, 5.0],
        8,
    );
}
