// Debug scratch bin: scipy reference-value literals carry extra precision.
#![allow(clippy::excessive_precision)]

fn main() {
    use fsci_runtime::RuntimeMode;
    use fsci_special::types::SpecialTensor;
    // (v, z, scipy.special.yv(v, z))
    let cases: [(f64, f64, f64); 12] = [
        (0.5, 1.0, -4.3109886801837610e-01),
        (0.5, 5.0, -1.0121770918510840e-01),
        (1.5, 1.0, -1.1024955751601793e+00),
        (1.5, 5.0, 3.2192444296114020e-01),
        (1.5, 30.0, 1.4318064368377220e-01),
        (2.5, 5.0, 2.9437237496179253e-01),
        (2.5, 10.0, -1.6417847961494103e-01),
        (1.3, 5.0, 2.6631384911040207e-01),
        (1.7, 5.0, 3.5626412768764792e-01),
        (2.3, 5.0, 3.3554008065355417e-01),
        (3.7, 5.0, -9.5770117401486277e-02),
        (5.5, 10.0, 2.3675446066584144e-01),
    ];
    let mut max_diff: f64 = 0.0;
    for (v, z, expected) in cases {
        let v_t = SpecialTensor::RealScalar(v);
        let z_t = SpecialTensor::RealScalar(z);
        let res = fsci_special::yv(&v_t, &z_t, RuntimeMode::Strict);
        let got = match res {
            Ok(SpecialTensor::RealScalar(v)) => v,
            Ok(_) => f64::NAN,
            Err(_) => f64::NAN,
        };
        let diff = (got - expected).abs();
        max_diff = max_diff.max(diff);
        println!(
            "yv({:>4}, {:>4}) = {:>22.15}  scipy: {:>22.15}  diff {:.2e}",
            v, z, got, expected, diff
        );
    }
    println!("max_diff = {:.2e}", max_diff);
}
