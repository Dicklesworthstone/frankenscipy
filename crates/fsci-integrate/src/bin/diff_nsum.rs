// Probe: fsci nsum vs scipy.integrate.nsum on standard convergent series.
use fsci_integrate::nsum;

/// One probe case: `(label, term, a, b, step)`. Named to keep
/// `clippy::type_complexity` quiet without reshaping the tuple
/// (frankenscipy-3qjah); same type, same order.
type ProbeCase = (&'static str, fn(f64) -> f64, f64, f64, f64);

fn main() {
    let inf = f64::INFINITY;
    let cases: Vec<ProbeCase> = vec![
        ("1/n^2 (1,inf)", |n| 1.0 / (n * n), 1.0, inf, 1.0),
        ("1/n^4 (1,inf)", |n| 1.0 / n.powi(4), 1.0, inf, 1.0),
        ("2^-n (1,inf)", |n| 2.0_f64.powf(-n), 1.0, inf, 1.0),
        ("1/n^3 (1,inf)", |n| 1.0 / n.powi(3), 1.0, inf, 1.0),
        ("n (1..10)", |n| n, 1.0, 10.0, 1.0),
        ("1/n^2 step2 (1,inf)", |n| 1.0 / (n * n), 1.0, inf, 2.0),
        ("exp(-n) (0,inf)", |n| (-n).exp(), 0.0, inf, 1.0),
    ];
    for (label, f, a, b, step) in cases {
        let r = nsum(f, a, b, step, 0.0, 1e-11);
        println!(
            "{label} -> {:.15e} err={:.2e} nfev={} conv={}",
            r.sum, r.error, r.nfev, r.converged
        );
    }
}
