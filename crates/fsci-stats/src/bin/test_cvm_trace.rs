// Debug scratch bin: math-style variable names and trace-branch duplication.
#![allow(clippy::if_same_then_else, non_snake_case)]

fn main() {
    use fsci_runtime::RuntimeMode;
    use fsci_special::types::SpecialTensor;
    let kv = |order: f64, x: f64| -> f64 {
        let r = fsci_special::kv(
            &SpecialTensor::RealScalar(order),
            &SpecialTensor::RealScalar(x),
            RuntimeMode::Strict,
        )
        .unwrap();
        match r {
            SpecialTensor::RealScalar(v) => v,
            _ => f64::NAN,
        }
    };
    let pi: f64 = std::f64::consts::PI;
    let sqrt_pi = pi.sqrt();
    let x: f64 = 0.31666666666666665;
    let sx = 2.0 * x.sqrt();
    let y1 = x.powf(0.75);
    let y2 = x.powf(1.25);
    let ed2 = |y: f64| -> f64 {
        let z = y * y / 4.0;
        let b = kv(0.25, z) + kv(0.75, z);
        (-z).exp() * (y / 2.0).powf(1.5) * b / sqrt_pi
    };
    let ed3 = |y: f64| -> f64 {
        let z = y * y / 4.0;
        let kv_terms = 2.0 * kv(0.25, z) + 3.0 * kv(0.75, z) - kv(1.25, z);
        ((-z).exp() / sqrt_pi) * (y / 2.0).powf(2.5) * kv_terms
    };
    // ln_gamma helper
    let lng = |x: f64| -> f64 {
        // Stirling-ish — just use the libm-style
        // For our test, use specific: gamma(0.5) = sqrt(pi), gamma(1.5) = sqrt(pi)/2, gamma(1) = 1
        if x == 0.5 {
            sqrt_pi.ln()
        } else if x == 1.5 {
            (sqrt_pi / 2.0).ln()
        } else if x == 1.0 {
            0.0
        } else if x == 2.0 {
            0.0
        } else if x == 2.5 {
            (1.5 * 0.5 * sqrt_pi).ln()
        } else {
            panic!("not handled: {}", x)
        }
    };
    for k in 0..3 {
        let kf = k as f64;
        let m = 2.0 * kf + 1.0;
        let g1 = lng(kf + 0.5).exp();
        let g3 = lng(kf + 1.5).exp();
        let e1 = m * g1 * ed2((4.0 * kf + 3.0) / sx) / (9.0 * y1);
        let e2 = g1 * ed3((4.0 * kf + 1.0) / sx) / (72.0 * y2);
        let e3 = 2.0 * (m + 2.0) * g3 * ed3((4.0 * kf + 5.0) / sx) / (12.0 * y2);
        let e4 = 7.0 * m * g1 * ed2((4.0 * kf + 1.0) / sx) / (144.0 * y1);
        let e5 = 7.0 * m * g1 * ed2((4.0 * kf + 5.0) / sx) / (144.0 * y1);
        let A = e1 + e2 + e3 + e4 + e5;
        let z = -A / (pi * lng(kf + 1.0).exp());
        println!(
            "k={}: e1={:.6} e2={:.6} e3={:.6} e4={:.6} e5={:.6} A={:.6} z={:.6}",
            k, e1, e2, e3, e4, e5, A, z
        );
    }
}
