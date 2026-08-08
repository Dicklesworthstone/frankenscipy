// Probe: fsci tanhsinh vs scipy.integrate.tanhsinh on endpoint-singular integrals.
use fsci_integrate::tanhsinh;

/// One probe case: `(label, integrand, a, b)`. Named to keep
/// `clippy::type_complexity` quiet without reshaping the tuple
/// (frankenscipy-3qjah); same type, same order.
type ProbeCase = (&'static str, fn(f64) -> f64, f64, f64);

fn main() {
    let cases: Vec<ProbeCase> = vec![
        ("1/sqrt(x) on (0,1) = 2", |x| 1.0 / x.sqrt(), 0.0, 1.0),
        ("ln(x) on (0,1) = -1", |x| x.ln(), 0.0, 1.0),
        (
            "1/sqrt(x(1-x)) on (0,1) = pi",
            |x| 1.0 / (x * (1.0 - x)).sqrt(),
            0.0,
            1.0,
        ),
        ("exp(x) on (0,1) = e-1", |x| x.exp(), 0.0, 1.0),
        (
            "cos(x) on (0,pi/2) = 1",
            |x| x.cos(),
            0.0,
            std::f64::consts::FRAC_PI_2,
        ),
        ("x^2 on (-1,2) = 3", |x| x * x, -1.0, 2.0),
        ("-ln(x)^3 on (0,1) = 6", |x| -(x.ln().powi(3)), 0.0, 1.0),
        ("exp(-x) on (0,inf) = 1", |x| (-x).exp(), 0.0, f64::INFINITY),
        (
            "1/x^2 on (1,inf) = 1",
            |x| 1.0 / (x * x),
            1.0,
            f64::INFINITY,
        ),
        (
            "exp(x) on (-inf,0) = 1",
            |x| x.exp(),
            f64::NEG_INFINITY,
            0.0,
        ),
        (
            "exp(-x^2) on (-inf,inf) = sqrt(pi)",
            |x| (-x * x).exp(),
            f64::NEG_INFINITY,
            f64::INFINITY,
        ),
        (
            "1/(1+x^2) on (-inf,inf) = pi",
            |x| 1.0 / (1.0 + x * x),
            f64::NEG_INFINITY,
            f64::INFINITY,
        ),
    ];
    for (label, f, a, b) in cases {
        let r = tanhsinh(f, a, b, 0.0, 1e-12, 16);
        println!(
            "{label} -> {:.15e} err={:.3e} neval={} conv={}",
            r.integral, r.error, r.neval, r.converged
        );
    }
}
